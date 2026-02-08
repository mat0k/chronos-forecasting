#!/usr/bin/env python3
"""
Chronos-2 evaluation (baseline / cross-learning / improved semantic-batched cross-learning)
using GluonTS metrics (MASE + MeanWeightedSumQuantileLoss).

Adds:
- --benchmark fev-bench : pulls tasks.yaml from the official fev repo
- --benchmark gift-eval : iterates HuggingFace configs from Salesforce/GiftEval
- --ttest : paired t-test (baseline vs each other mode) across tasks
- CLI supports BOTH:
    python semantic_cross_learning.py evaluate ...
    python semantic_cross_learning.py ...

Fixes added:
1) Better error visibility: logger.exception(..., repr(e)) to print full traceback
2) Robust dataset loading for FEV-style datasets:
   - Filters out series that are too short for offset + horizon + rolling windows
   - Skips corrupt targets instead of crashing the whole task

New:
3) Enhanced rule-based bucketing + coherence gate:
   - grouping="enhanced_bucket" uses:
       * freq bucket (if inferable)
       * length bucket
       * scale bucket (log std)
       * trend bucket (slope)
       * seasonality bucket (seasonality strength + dominant period)
   - Still uses coherence_gate (cosine-to-centroid on z-scored features) to disable cross-learning
     when a batch is not coherent enough.
"""

import logging
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
import requests
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm

# Make repo src visible if running from inside the Chronos repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chronos import BaseChronosPipeline, Chronos2Pipeline  # noqa: F401


app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Chronos2 Eval (baseline/cross/semantic-upgraded)")
logger.setLevel(logging.INFO)

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASE_KEY = "MASE[0.5]"
WQL_KEY = "mean_weighted_sum_quantile_loss"

DATASET_SCRIPT_ERR = "Dataset scripts are no longer supported"

FEV_TASKS_URLS = {
    "fev-bench": "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml",
}

GIFT_HF_REPO = "Salesforce/GiftEval"


# -----------------------------
# Small helpers
# -----------------------------
def ceil_to_patch(h: int, patch: int) -> int:
    if patch <= 1:
        return int(h)
    return int(((h + patch - 1) // patch) * patch)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


def infer_freq_from_timestamp(ts_list: Any) -> str:
    idx = pd.DatetimeIndex(ts_list)
    try:
        return idx.to_period()[0].freqstr
    except Exception:
        return "D"


def infer_freq_from_datetime_index(idx: pd.DatetimeIndex) -> str:
    f = pd.infer_freq(idx[: min(len(idx), 200)])
    if f is None:
        if len(idx) >= 2:
            d = (idx[1] - idx[0])
            if d == pd.Timedelta(hours=1):
                return "H"
            if d == pd.Timedelta(days=1):
                return "D"
        return "D"
    return f


def download_text(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


# -----------------------------
# Old bucket fallback (len/scale/seasonality)
# -----------------------------
def seasonality_strength(x: np.ndarray, max_lag: int = 24) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 8:
        return 0.0
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 1e-12:
        return 0.0

    max_lag = int(min(max_lag, n // 2))
    best = 0.0
    for lag in range(2, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        num = float((a * b).sum())
        ac = abs(num / denom)
        if ac > best:
            best = ac
    return float(max(0.0, min(1.0, best)))


def bucket_key(
    target: np.ndarray,
    length_bins: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
    scale_bins: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),  # log10(std)
    seas_bins: Tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.70),
) -> str:
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    L = int(y.size)
    tail = y[-min(L, 64):] if L > 0 else y
    std = float(np.std(tail, ddof=0)) if L > 0 else 0.0
    log_std = float(np.log10(std + 1e-6))
    seas = seasonality_strength(y[-min(L, 256):], max_lag=24)

    lb = 0
    for t in length_bins:
        if L <= t:
            break
        lb += 1

    sb = 0
    for t in scale_bins:
        if log_std <= t:
            break
        sb += 1

    zb = 0
    for t in seas_bins:
        if seas <= t:
            break
        zb += 1

    return f"B_len{lb}_s{sb}_z{zb}"


# -----------------------------
# Better "semantic" features for clustering + coherence gate
# -----------------------------
def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 3:
        return 0.0
    m = float(np.mean(x))
    s = float(np.std(x) + 1e-8)
    return float(np.mean(((x - m) / s) ** 3))


def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 4:
        return 0.0
    m = float(np.mean(x))
    s = float(np.std(x) + 1e-8)
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _linear_slope(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 4:
        return 0.0
    t = np.arange(n, dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)
    y = x - x.mean()
    denom = float((t * t).sum()) + 1e-8
    return float((t * y).sum() / denom)


def _acf_at_lag(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n <= lag + 2:
        return 0.0
    x = x - x.mean()
    denom = float((x * x).sum()) + 1e-8
    num = float((x[:-lag] * x[lag:]).sum())
    return float(num / denom)


def _dominant_period_fft(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 16:
        return 0.0
    y = x - x.mean()
    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    if mag.size <= 2:
        return 0.0
    mag[0] = 0.0  # drop DC
    k = int(np.argmax(mag))
    if k <= 0:
        return 0.0
    period = float(n / k)
    return float(max(1.0, min(period, float(n))))


def extract_ts_features(y: np.ndarray, max_lag: int = 24) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return np.zeros((9 + max_lag,), dtype=np.float32)

    tail = y[-min(y.size, 256):]
    m = float(np.mean(tail))
    s = float(np.std(tail) + 1e-8)
    log_s = float(np.log10(s + 1e-6))

    slope = _linear_slope(tail)
    skew = _safe_skew(tail)
    kurt = _safe_kurtosis(tail)
    domp = _dominant_period_fft(tail)
    seas = seasonality_strength(tail, max_lag=max_lag)

    acfs = [_acf_at_lag(tail, lag) for lag in range(1, max_lag + 1)]
    acf_max = float(max(np.abs(acfs))) if acfs else 0.0

    feat = np.array(
        [m, s, log_s, slope, skew, kurt, domp, seas, acf_max] + acfs,
        dtype=np.float32,
    )
    feat = np.clip(feat, -50.0, 50.0)
    return feat


def zscore_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def simple_kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(max(1, min(k, n)))
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
            else:
                C[j] = X[rng.integers(0, n)]
    return labels


def cosine_to_centroid(batch_feats: np.ndarray) -> float:
    X = batch_feats.astype(np.float32)
    if X.shape[0] <= 1:
        return 1.0
    c = X.mean(axis=0, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    sims = (Xn * cn).sum(axis=1)
    return float(np.mean(sims))


# -----------------------------
# Enhanced rule-based bucketing (NEW)
# -----------------------------
def _bin_index(val: float, edges: Tuple[float, ...]) -> int:
    """Return bin index in [0..len(edges)] where edges are increasing thresholds."""
    b = 0
    for t in edges:
        if val <= t:
            break
        b += 1
    return int(b)


def _freq_bucket_from_period_start(start: Any) -> str:
    """
    Try to bucket by frequency using pd.Period's freqstr if possible.
    If unknown, return 'UNK'.
    """
    try:
        if isinstance(start, pd.Period):
            f = str(start.freqstr)
            return f
        # sometimes start might be timestamp-like
        return "UNK"
    except Exception:
        return "UNK"


def enhanced_bucket_key(
    entry: dict,
    rep_target: np.ndarray,
) -> str:
    """
    Enhanced semantic-ish bucketing without clustering:
      - frequency bucket (if Period freqstr is available)
      - length bucket
      - scale bucket (log10 std of tail)
      - trend bucket (linear slope)
      - seasonality bucket (strength) + dominant period bucket
    """
    y = np.asarray(rep_target, dtype=np.float32).reshape(-1)
    L = int(y.size)

    # freq (best-effort)
    freq_b = _freq_bucket_from_period_start(entry.get("start", None))

    # length bins
    len_edges = (32, 64, 128, 256, 512, 1024, 2048)
    len_b = _bin_index(float(L), tuple(float(x) for x in len_edges))

    # tail stats
    tail = y[-min(L, 256):] if L > 0 else y
    std = float(np.std(tail, ddof=0)) if L > 0 else 0.0
    log_std = float(np.log10(std + 1e-6))

    # scale bins on log10 std
    scale_edges = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
    scale_b = _bin_index(log_std, scale_edges)

    # trend bucket from slope (tail)
    slope = _linear_slope(tail)
    trend_edges = (-0.5, -0.2, -0.08, -0.02, 0.02, 0.08, 0.2, 0.5)
    trend_b = _bin_index(slope, trend_edges)

    # seasonality + dominant period
    seas = seasonality_strength(tail, max_lag=24)
    seas_edges = (0.05, 0.12, 0.2, 0.3, 0.45, 0.6, 0.75)
    seas_b = _bin_index(seas, seas_edges)

    domp = _dominant_period_fft(tail)  # in [1..n] approx
    # period bins: very short / short / medium / long
    domp_edges = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    domp_b = _bin_index(domp, domp_edges)

    return f"EB_f{freq_b}_L{len_b}_S{scale_b}_T{trend_b}_Z{seas_b}_P{domp_b}"


# -----------------------------
# Benchmark task loading
# -----------------------------
def _normalize_task_dict(d: dict) -> dict:
    if all(k in d for k in ("hf_repo", "name", "offset", "prediction_length")):
        out = dict(d)
        out["offset"] = int(out["offset"])
        out["prediction_length"] = int(out["prediction_length"])
        out["num_rolls"] = int(out.get("num_rolls", 1))
        return out

    if "dataset_path" in d and "dataset_config" in d and "horizon" in d:
        hf_repo = d["dataset_path"]
        name = d["dataset_config"]
        pred_len = int(d["horizon"])
        num_rolls = int(d.get("num_windows", 1))

        offset = -pred_len * num_rolls

        return {
            "hf_repo": hf_repo,
            "name": name,
            "prediction_length": pred_len,
            "offset": offset,
            "num_rolls": num_rolls,
        }

    raise RuntimeError(f"Unrecognized task format keys={list(d.keys())}")


def load_benchmark_tasks(benchmark: str) -> List[dict]:
    b = benchmark.strip().lower()

    if b in FEV_TASKS_URLS:
        url = FEV_TASKS_URLS[b]
        logger.info(f"Downloading benchmark task list: {benchmark} from {url}")
        txt = download_text(url)
        obj = yaml.safe_load(txt)

        if isinstance(obj, list):
            tasks_raw = obj
        elif isinstance(obj, dict):
            if "tasks" in obj and isinstance(obj["tasks"], list):
                tasks_raw = obj["tasks"]
            elif "benchmarks" in obj and isinstance(obj["benchmarks"], list):
                tasks_raw = obj["benchmarks"]
            else:
                list_vals = [v for v in obj.values() if isinstance(v, list)]
                if not list_vals:
                    raise RuntimeError(
                        "Benchmark YAML must be a list of tasks or contain a list under a key like 'tasks'."
                    )
                tasks_raw = list_vals[0]
        else:
            raise RuntimeError("Benchmark YAML must be a list of tasks or a dict containing a list of tasks.")

        tasks = []
        for t in tasks_raw:
            if not isinstance(t, dict):
                continue
            tasks.append(_normalize_task_dict(t))

        if not tasks:
            raise RuntimeError("No tasks parsed from benchmark YAML.")
        return tasks

    if b in {"gift-eval", "gift_eval", "gift"}:
        logger.info(f"Loading GIFT-Eval tasks from HF dataset configs: {GIFT_HF_REPO}")
        cfgs = datasets.get_dataset_config_names(GIFT_HF_REPO)
        tasks: List[dict] = []

        for cfg_name in cfgs:
            m = re.search(r"(\d+)$", cfg_name)
            pred_len = int(m.group(1)) if m else 24

            tasks.append(
                {
                    "hf_repo": GIFT_HF_REPO,
                    "name": cfg_name,
                    "offset": -pred_len,
                    "prediction_length": pred_len,
                    "num_rolls": 1,
                }
            )
        if not tasks:
            raise RuntimeError("No GIFT-Eval tasks found (no configs returned).")
        return tasks

    raise RuntimeError(f"Unknown benchmark '{benchmark}'. Supported: fev-bench, gift-eval")


# -----------------------------
# HF -> GluonTS conversion (robust)
# -----------------------------
def hf_to_gluonts_univariate(
    hf_dataset: datasets.Dataset,
    semantic_field: Optional[str],
) -> Tuple[List[dict], Dict[str, Optional[str]]]:
    gts_dataset: List[dict] = []
    sem_map: Dict[str, Optional[str]] = {}

    cols = set(hf_dataset.column_names)
    has_start = "start" in cols
    has_target = "target" in cols

    if has_start and has_target:
        freq = "D"
        if "freq" in cols:
            try:
                freq = _safe_str(hf_dataset[0]["freq"])
            except Exception:
                pass

        for i, row in enumerate(hf_dataset):
            item_id = _safe_str(row.get("item_id", row.get("id", f"series_{i}")))
            start_raw = row["start"]
            try:
                start = pd.Period(start_raw, freq=freq)
            except Exception:
                start = pd.Period(pd.Timestamp(start_raw), freq=freq)

            try:
                target = _as_1d_float_array(row["target"])
            except Exception:
                continue

            sem = None
            if semantic_field and semantic_field in row:
                sem = _safe_str(row[semantic_field])

            gts_dataset.append({"start": start, "target": target, "item_id": item_id})
            sem_map[item_id] = sem

        return gts_dataset, sem_map

    if "timestamp" not in cols:
        raise RuntimeError("Dataset has neither (start,target) nor timestamp-based format.")

    series_fields = [
        col for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    if "timestamp" in series_fields:
        series_fields.remove("timestamp")

    freq = infer_freq_from_timestamp(hf_dataset[0]["timestamp"])

    item_counter = 0
    for ridx, row in enumerate(hf_dataset):
        ts = row["timestamp"]
        start = pd.Period(pd.Timestamp(ts[0]), freq=freq)

        sem = None
        if semantic_field and semantic_field in row:
            sem = _safe_str(row[semantic_field])

        base = _safe_str(row.get("item_id", row.get("id", f"row_{ridx}")))

        for field in series_fields:
            item_id = f"{base}_{field}_{item_counter}"
            try:
                target = _as_1d_float_array(row[field])
            except Exception:
                continue
            gts_dataset.append({"start": start, "target": target, "item_id": item_id})
            sem_map[item_id] = sem
            item_counter += 1

    return gts_dataset, sem_map


def load_and_split_dataset(
    cfg: dict,
    semantic_field: Optional[str],
) -> Tuple[Any, List[dict], Dict[str, Optional[str]]]:
    hf_repo = cfg["hf_repo"]
    dataset_name = cfg["name"]
    offset = int(cfg["offset"])
    prediction_length_eval = int(cfg["prediction_length"])
    num_rolls = int(cfg.get("num_rolls", 1))

    ds = datasets.load_dataset(hf_repo, dataset_name, split="train")
    try:
        ds.set_format("numpy")
    except Exception:
        pass

    gts_dataset, sem_map = hf_to_gluonts_univariate(ds, semantic_field=semantic_field)

    needed = abs(offset) + prediction_length_eval + max(0, num_rolls - 1) * prediction_length_eval

    filtered = []
    dropped = 0
    for e in gts_dataset:
        y = _as_1d_float_array(e["target"])
        if len(y) >= needed:
            filtered.append(e)
        else:
            dropped += 1

    if dropped > 0:
        logger.warning(
            f"[data filter] {dataset_name}: dropped {dropped}/{len(gts_dataset)} series "
            f"shorter than needed={needed} (offset={offset}, pred_len={prediction_length_eval}, rolls={num_rolls})"
        )

    if not filtered:
        raise RuntimeError(
            f"{dataset_name}: No usable series after filtering (needed>={needed}). "
            f"Consider reducing num_windows/offset or choose another task."
        )

    train_data, test_template = split(filtered, offset=offset)
    test_data = test_template.generate_instances(prediction_length_eval, windows=num_rolls)

    train_list = list(train_data)
    train_list.sort(key=lambda x: _safe_str(x.get("item_id", "")))
    return test_data, train_list, sem_map


# -----------------------------
# Quantile output normalization
# -----------------------------
def _to_HQ(x: Any, q_levels: List[float]) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected per-item quantile shape: {arr.shape}")

    Q = len(q_levels)
    if arr.shape[0] == Q and arr.shape[1] != Q:
        return arr.T
    if arr.shape[1] == Q and arr.shape[0] != Q:
        return arr
    if arr.shape[0] == Q and arr.shape[1] == Q:
        return arr.T

    raise ValueError(f"Cannot align quantile output {arr.shape} with Q={Q}")


def _quantiles_to_BHQ(q_out: Any, q_levels: List[float]) -> np.ndarray:
    Q = len(q_levels)

    if isinstance(q_out, list):
        hq_list = [_to_HQ(x, q_levels) for x in q_out]
        return np.stack(hq_list, axis=0)

    arr = np.asarray(q_out)

    while arr.ndim > 3 and arr.shape[1] == 1:
        arr = arr[:, 0, ...]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected batched quantile output shape: {arr.shape}")

    if arr.shape[1] == Q:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[2] == Q:
        return arr

    raise ValueError(f"Cannot align batched quantile output {arr.shape} with Q={Q}")


# -----------------------------
# Forecast generation (baseline/random)
# -----------------------------
@torch.no_grad()
def generate_forecasts(
    test_data_input,
    pipeline: Chronos2Pipeline,
    prediction_length_eval: int,
    prediction_length_request: int,
    batch_size: int,
    cross_learning: bool,
) -> List[QuantileForecast]:
    forecast_outputs = []

    for batch in tqdm(batcher(test_data_input, batch_size=batch_size), desc="Forecasting"):
        context = [torch.tensor(_as_1d_float_array(entry["target"])) for entry in batch]

        q_out, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length_request,
            quantile_levels=QUANTILES,
            cross_learning=cross_learning,
            batch_size=len(context),
        )

        q_bhq = _quantiles_to_BHQ(q_out, QUANTILES)
        q_bhq = q_bhq[:, :prediction_length_eval, :]
        forecast_outputs.append(q_bhq)

    forecast_outputs = np.concatenate(forecast_outputs, axis=0)

    forecasts: List[QuantileForecast] = []
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        forecasts.append(
            QuantileForecast(
                forecast_arrays=item.T,
                forecast_keys=list(map(str, QUANTILES)),
                start_date=forecast_start_date,
            )
        )
    return forecasts


# -----------------------------
# Improved semantic batching (upgraded)
# -----------------------------
def build_item_representatives(test_entries: List[dict]) -> Dict[str, np.ndarray]:
    rep: Dict[str, np.ndarray] = {}
    rep_len: Dict[str, int] = {}
    for i, e in enumerate(test_entries):
        item_id = _safe_str(e.get("item_id", f"series_{i}"))
        y = _as_1d_float_array(e["target"])
        L = int(y.size)
        if (item_id not in rep) or (L > rep_len[item_id]):
            rep[item_id] = y
            rep_len[item_id] = L
    return rep


def make_batches_from_item_groups(
    test_entries: List[dict],
    item_to_group: Dict[str, str],
    batch_size: int,
) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, e in enumerate(test_entries):
        item_id = _safe_str(e.get("item_id", f"series_{idx}"))
        g = item_to_group.get(item_id, "G:unknown")
        groups.setdefault(g, []).append(idx)

    ordered_batches: List[List[int]] = []
    for _, idxs in groups.items():
        idxs.sort(key=lambda k: len(_as_1d_float_array(test_entries[k]["target"])))
        for j in range(0, len(idxs), batch_size):
            ordered_batches.append(idxs[j: j + batch_size])

    return ordered_batches


@torch.no_grad()
def generate_forecasts_semantic_batched_upgraded(
    test_data_input,
    pipeline: Chronos2Pipeline,
    prediction_length_eval: int,
    prediction_length_request: int,
    batch_size: int,
    itemid_to_sem: Dict[str, Optional[str]],
    semantic_field: Optional[str],
    grouping: str,
    num_clusters: int,
    coherence_threshold: float,
    coherence_gate: bool,
    kmeans_iters: int,
    seed: int,
) -> List[QuantileForecast]:
    test_entries = list(test_data_input)
    N = len(test_entries)

    item_to_rep = build_item_representatives(test_entries)
    item_ids = sorted(item_to_rep.keys())

    item_to_group: Dict[str, str] = {}

    # semantic_field grouping if provided & usable
    if semantic_field is not None:
        usable = 0
        for iid in item_ids:
            sem = itemid_to_sem.get(iid, None)
            if sem is not None and _safe_str(sem) != "":
                item_to_group[iid] = f"S:{_safe_str(sem)}"
                usable += 1
        if usable > 0:
            logger.info(f"  [grouping] using semantic_field='{semantic_field}' for {usable}/{len(item_ids)} items")
        else:
            logger.info(f"  [grouping] semantic_field='{semantic_field}' had no usable values; falling back")

    remaining = [iid for iid in item_ids if iid not in item_to_group]

    if remaining:
        if grouping == "bucket":
            for iid in remaining:
                item_to_group[iid] = f"K:{bucket_key(item_to_rep[iid])}"
            logger.info(f"  [grouping] bucket used for {len(remaining)} items")

        elif grouping == "enhanced_bucket":
            # NEW: Enhanced rule-based bucketing (no clustering)
            # Use entry's start when available (for freq bucket) by looking up any representative entry:
            # We pick the first entry in test_entries that matches the item_id.
            # This is safe enough for grouping keys.
            itemid_to_entry: Dict[str, dict] = {}
            for e in test_entries:
                iid = _safe_str(e.get("item_id", ""))
                if iid and iid not in itemid_to_entry:
                    itemid_to_entry[iid] = e

            for iid in remaining:
                entry = itemid_to_entry.get(iid, {"start": None})
                item_to_group[iid] = enhanced_bucket_key(entry, item_to_rep[iid])

            logger.info(f"  [grouping] enhanced_bucket used for {len(remaining)} items")

        elif grouping == "features":
            feats = np.stack([extract_ts_features(item_to_rep[iid]) for iid in remaining], axis=0)
            feats_z, _, _ = zscore_rows(feats)

            k = int(min(num_clusters, feats_z.shape[0]))
            labels = simple_kmeans(feats_z, k=k, iters=kmeans_iters, seed=seed)

            for iid, lab in zip(remaining, labels):
                item_to_group[iid] = f"F:{int(lab)}"
            logger.info(f"  [grouping] features+kmeans(k={k}) used for {len(remaining)} items")
        else:
            raise ValueError("grouping must be one of: 'features', 'bucket', 'enhanced_bucket'")

    ordered_batches = make_batches_from_item_groups(test_entries, item_to_group, batch_size=batch_size)

    # group diagnostics
    group_counts: Dict[str, int] = {}
    for iid in item_ids:
        g = item_to_group.get(iid, "G:unknown")
        group_counts[g] = group_counts.get(g, 0) + 1
    sizes = np.array(list(group_counts.values()), dtype=np.int32)
    logger.info(
        f"  [group diag] items={len(item_ids)}, groups={len(group_counts)}, "
        f"avg_group_items={float(sizes.mean()) if sizes.size else 0.0:.2f}, "
        f"batches={len(ordered_batches)}, avg_batch_size={float(np.mean([len(b) for b in ordered_batches])) if ordered_batches else 0.0:.2f}"
    )

    # coherence gating features (always use the same feature extractor)
    item_feat_for_gate: Dict[str, np.ndarray] = {iid: extract_ts_features(item_to_rep[iid]) for iid in item_ids}

    forecasts_out: List[Optional[QuantileForecast]] = [None] * N

    for batch_idxs in tqdm(ordered_batches, desc="Forecasting (semantic upgraded)"):
        do_cross = True
        if coherence_gate and len(batch_idxs) >= 2:
            feats = []
            for i in batch_idxs:
                iid = _safe_str(test_entries[i].get("item_id", f"series_{i}"))
                feats.append(item_feat_for_gate.get(iid, np.zeros(10, dtype=np.float32)))
            feats = np.stack(feats, axis=0)
            feats_z, _, _ = zscore_rows(feats)
            coh = cosine_to_centroid(feats_z)
            if coh < coherence_threshold:
                do_cross = False

        context = [torch.tensor(_as_1d_float_array(test_entries[i]["target"])) for i in batch_idxs]

        q_out, _ = pipeline.predict_quantiles(
            context,
            prediction_length=prediction_length_request,
            quantile_levels=QUANTILES,
            cross_learning=do_cross,
            batch_size=len(context),
        )

        q_bhq = _quantiles_to_BHQ(q_out, QUANTILES)
        q_bhq = q_bhq[:, :prediction_length_eval, :]

        for local_k, global_i in enumerate(batch_idxs):
            ts = test_entries[global_i]
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts_out[global_i] = QuantileForecast(
                forecast_arrays=q_bhq[local_k].T,
                forecast_keys=list(map(str, QUANTILES)),
                start_date=forecast_start_date,
            )

    missing = sum(1 for f in forecasts_out if f is None)
    if missing:
        raise RuntimeError(f"Missing {missing} forecasts (bug).")

    return [f for f in forecasts_out if f is not None]


# -----------------------------
# Paired t-test
# -----------------------------
def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan"), float("nan")

    try:
        from scipy.stats import ttest_rel  # type: ignore
        t, p = ttest_rel(x, y)
        return float(t), float(p)
    except Exception:
        d = x - y
        n = d.size
        md = float(d.mean())
        sd = float(d.std(ddof=1) + 1e-12)
        t = md / (sd / np.sqrt(n))
        df = n - 1
        if df >= 30:
            from math import erf, sqrt
            z = abs(t)
            p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
            return float(t), float(p)
        return float(t), float("nan")


# -----------------------------
# Core evaluation runner
# -----------------------------
def run_tasks(
    tasks: List[dict],
    output_csv: Path,
    model_id: str,
    device: str,
    dtype: str,
    batch_size: int,
    semantic_field: Optional[str],
    grouping: str,
    num_clusters: int,
    coherence_gate: bool,
    coherence_threshold: float,
    ttest: bool,
    kmeans_iters: int,
    seed: int,
) -> None:
    logger.info(f"Loading model: {model_id}")
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype],
    )

    patch = int(getattr(pipeline, "model_output_patch_size", 1))
    logger.info(f"Model patch size: {patch}")

    rows = []

    for cfg in tasks:
        dataset_name = cfg["name"]
        hf_repo = cfg["hf_repo"]
        prediction_length_eval = int(cfg["prediction_length"])
        prediction_length_request = ceil_to_patch(prediction_length_eval, patch)

        logger.info("=" * 70)
        logger.info(f"Task: {dataset_name} (hf_repo={hf_repo}) pred_len={prediction_length_eval} offset={cfg['offset']}")
        logger.info("=" * 70)

        try:
            test_data, _train_list, sem_map = load_and_split_dataset(cfg, semantic_field=semantic_field)
        except Exception as e:
            logger.exception(f"[skip] {dataset_name}: load/split failed (repr={e!r})")
            continue

        test_input = list(test_data.input)

        def _eval_mode(mode_name: str, forecasts: List[QuantileForecast]) -> Tuple[float, float]:
            metrics = evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
                batch_size=5000,
            ).reset_index(drop=True).to_dict(orient="records")
            mase = float(metrics[0].get(MASE_KEY, np.nan))
            wql = float(metrics[0].get(WQL_KEY, np.nan))
            rows.append(
                {
                    "benchmark_task": dataset_name,
                    "hf_repo": hf_repo,
                    "mode": mode_name,
                    "MASE": mase,
                    "WQL": wql,
                }
            )
            return mase, wql

        # baseline
        logger.info("  [baseline]")
        t0 = time.time()
        f_base = generate_forecasts(
            test_input, pipeline,
            prediction_length_eval=prediction_length_eval,
            prediction_length_request=prediction_length_request,
            batch_size=batch_size,
            cross_learning=False,
        )
        base_mase, base_wql = _eval_mode("baseline", f_base)
        logger.info(f"    MASE={base_mase:.4f}, WQL={base_wql:.4f} ({time.time()-t0:.1f}s)")

        # cross-learning random
        logger.info("  [cross_learning_random]")
        t0 = time.time()
        f_cross = generate_forecasts(
            test_input, pipeline,
            prediction_length_eval=prediction_length_eval,
            prediction_length_request=prediction_length_request,
            batch_size=batch_size,
            cross_learning=True,
        )
        cl_mase, cl_wql = _eval_mode("cross_learning_random", f_cross)
        logger.info(f"    MASE={cl_mase:.4f}, WQL={cl_wql:.4f} ({time.time()-t0:.1f}s)")

        # upgraded semantic-batched cross-learning
        logger.info("  [semantic_cross_learning_upgraded]")
        t0 = time.time()
        f_sem = generate_forecasts_semantic_batched_upgraded(
            test_input, pipeline,
            prediction_length_eval=prediction_length_eval,
            prediction_length_request=prediction_length_request,
            batch_size=batch_size,
            itemid_to_sem=sem_map,
            semantic_field=semantic_field,
            grouping=grouping,
            num_clusters=num_clusters,
            coherence_threshold=coherence_threshold,
            coherence_gate=coherence_gate,
            kmeans_iters=kmeans_iters,
            seed=seed,
        )
        sem_mase, sem_wql = _eval_mode("semantic_cross_learning_upgraded", f_sem)
        logger.info(f"    MASE={sem_mase:.4f}, WQL={sem_wql:.4f} ({time.time()-t0:.1f}s)")

        logger.info(
            f"  Summary: baseline={base_mase:.4f}, cross_random={cl_mase:.4f}, upgraded={sem_mase:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"\nSaved results to {output_csv}")

    if df.empty:
        logger.warning("No results to summarize.")
        return

    pivot_mase = df.pivot_table(index="benchmark_task", columns="mode", values="MASE", aggfunc="mean")
    pivot_wql = df.pivot_table(index="benchmark_task", columns="mode", values="WQL", aggfunc="mean")

    logger.info("\nMASE per task (pivot):\n" + pivot_mase.to_string())
    logger.info("\nWQL per task (pivot):\n" + pivot_wql.to_string())

    if ttest:
        def _ttest_report(pivot: pd.DataFrame, metric_name: str):
            if "baseline" not in pivot.columns:
                return
            base = pivot["baseline"].to_numpy()
            for mode in pivot.columns:
                if mode == "baseline":
                    continue
                other = pivot[mode].to_numpy()
                t, p = paired_ttest(base, other)
                logger.info(f"[t-test paired] {metric_name}: baseline vs {mode}: t={t:.4f}, p={p}")

        logger.info("\n" + "=" * 70)
        logger.info("Paired t-tests across tasks (baseline vs others)")
        logger.info("=" * 70)
        _ttest_report(pivot_mase, "MASE")
        _ttest_report(pivot_wql, "WQL")


# -----------------------------
# CLI (supports both styles)
# -----------------------------
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name: fev-bench | gift-eval"),
    config_yaml: Optional[Path] = typer.Option(
        None, "--config-yaml",
        help="Path to YAML list of dataset configs (if not using --benchmark)"
    ),
    output_csv: Path = typer.Option(Path("./chronos2_results.csv"), help="Output CSV path"),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: str = typer.Option("float32"),
    batch_size: int = typer.Option(32),
    semantic_field: Optional[str] = typer.Option(None, help="HF column for semantic grouping (optional)"),
    grouping: str = typer.Option("features", help="Grouping: features | bucket | enhanced_bucket"),
    num_clusters: int = typer.Option(50, help="Upper bound K for kmeans when grouping=features (clamped to #items)"),
    coherence_gate: bool = typer.Option(True),
    coherence_threshold: float = typer.Option(0.25),
    ttest: bool = typer.Option(False, help="Run paired t-test across tasks (baseline vs others)"),
    kmeans_iters: int = typer.Option(25),
    seed: int = typer.Option(0),
):
    if ctx.invoked_subcommand is not None:
        return

    tasks: List[dict]
    if benchmark:
        tasks = load_benchmark_tasks(benchmark)
    else:
        if config_yaml is None:
            raise typer.BadParameter("Provide either --benchmark or --config-yaml")
        with open(config_yaml, "r", encoding="utf-8") as fp:
            obj = yaml.safe_load(fp)
        if not isinstance(obj, list):
            raise typer.BadParameter("config_yaml must contain a list of dataset configs")
        tasks = [_normalize_task_dict(x) for x in obj]

    run_tasks(
        tasks=tasks,
        output_csv=output_csv,
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        semantic_field=semantic_field,
        grouping=grouping,
        num_clusters=num_clusters,
        coherence_gate=coherence_gate,
        coherence_threshold=coherence_threshold,
        ttest=ttest,
        kmeans_iters=kmeans_iters,
        seed=seed,
    )


@app.command("evaluate")
def evaluate_cmd(
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name: fev-bench | gift-eval"),
    config_yaml: Optional[Path] = typer.Option(
        None, "--config-yaml",
        help="Path to YAML list of dataset configs (if not using --benchmark)"
    ),
    output_csv: Path = typer.Option(Path("./chronos2_results.csv")),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: str = typer.Option("float32"),
    batch_size: int = typer.Option(32),
    semantic_field: Optional[str] = typer.Option(None),
    grouping: str = typer.Option("features", help="Grouping: features | bucket | enhanced_bucket"),
    num_clusters: int = typer.Option(50),
    coherence_gate: bool = typer.Option(True),
    coherence_threshold: float = typer.Option(0.25),
    ttest: bool = typer.Option(False),
    kmeans_iters: int = typer.Option(25),
    seed: int = typer.Option(0),
):
    tasks: List[dict]
    if benchmark:
        tasks = load_benchmark_tasks(benchmark)
    else:
        if config_yaml is None:
            raise typer.BadParameter("Provide either --benchmark or --config-yaml")
        with open(config_yaml, "r", encoding="utf-8") as fp:
            obj = yaml.safe_load(fp)
        if not isinstance(obj, list):
            raise typer.BadParameter("config_yaml must contain a list of dataset configs")
        tasks = [_normalize_task_dict(x) for x in obj]

    run_tasks(
        tasks=tasks,
        output_csv=output_csv,
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        semantic_field=semantic_field,
        grouping=grouping,
        num_clusters=num_clusters,
        coherence_gate=coherence_gate,
        coherence_threshold=coherence_threshold,
        ttest=ttest,
        kmeans_iters=kmeans_iters,
        seed=seed,
    )


if __name__ == "__main__":
    app()