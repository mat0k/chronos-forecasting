#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# -----------------------------
# Constants
# -----------------------------
QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASE_KEY = "MASE[0.5]"
WQL_KEY = "mean_weighted_sum_quantile_loss"


# -----------------------------
# Small helpers
# -----------------------------
def ceil_to_patch(h: int, patch: int) -> int:
    if patch <= 1:
        return int(h)
    return int(((h + patch - 1) // patch) * patch)


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def as_1d_float_array(x: Any) -> np.ndarray:
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
        return arr.T  # (Q,H)->(H,Q)
    if arr.shape[1] == Q and arr.shape[0] != Q:
        return arr  # (H,Q)
    if arr.shape[0] == Q and arr.shape[1] == Q:
        return arr.T

    raise ValueError(f"Cannot align quantile output {arr.shape} with Q={Q}")


def quantiles_to_BHQ(q_out: Any, q_levels: List[float]) -> np.ndarray:
    """
    Returns (B, H, Q)
    """
    Q = len(q_levels)

    if isinstance(q_out, list):
        hq_list = [_to_HQ(x, q_levels) for x in q_out]
        return np.stack(hq_list, axis=0)  # (B,H,Q)

    arr = np.asarray(q_out)

    # unwrap extra singleton dims sometimes produced by pipelines
    while arr.ndim > 3 and arr.shape[1] == 1:
        arr = arr[:, 0, ...]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected batched quantile output shape: {arr.shape}")

    if arr.shape[1] == Q:
        return np.transpose(arr, (0, 2, 1))  # (B,Q,H)->(B,H,Q)
    if arr.shape[2] == Q:
        return arr  # (B,H,Q)

    raise ValueError(f"Cannot align batched quantile output {arr.shape} with Q={Q}")


# -----------------------------
# Batch sanity + packing
# -----------------------------
def batch_sanity_stats(batches: List[List[int]], N: int, batch_size: int) -> Dict[str, float]:
    if not batches:
        return {"avg_bs": 0.0, "expected": float(np.ceil(N / max(1, batch_size))), "actual": 0.0}
    avg_bs = float(np.mean([len(b) for b in batches]))
    expected = float(np.ceil(N / max(1, batch_size)))
    actual = float(len(batches))
    return {"avg_bs": avg_bs, "expected": expected, "actual": actual}


def is_fragmented(
    batches: List[List[int]],
    N: int,
    batch_size: int,
    min_fill: float,
    max_overhead: float,
) -> bool:
    st = batch_sanity_stats(batches, N, batch_size)
    if st["actual"] <= 0:
        return True
    too_small = st["avg_bs"] < (min_fill * batch_size)
    too_many = st["actual"] > (max_overhead * st["expected"])
    return bool(too_small or too_many)


def pack_batches_global(batches: List[List[int]], batch_size: int) -> List[List[int]]:
    """Pack a list of batches into full batches by concatenation (global packing)."""
    out: List[List[int]] = []
    cur: List[int] = []
    for b in batches:
        for idx in b:
            cur.append(idx)
            if len(cur) >= batch_size:
                out.append(cur[:batch_size])
                cur = cur[batch_size:]
    if cur:
        out.append(cur)
    return out


# -----------------------------
# Per-series metric arrays for t-test
# -----------------------------
def _pinball_loss(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    diff = y - yhat
    return float(np.sum(np.maximum(q * diff, (q - 1.0) * diff)))


def compute_per_series_arrays(
    test_inputs: List[dict],
    test_labels: List[dict],
    forecasts: List[Any],  # QuantileForecast
    prediction_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mase_series: (N,)
      wql_series:  (N,)
    """
    N = min(len(test_inputs), len(test_labels), len(forecasts))
    q_levels = [float(q) for q in QUANTILES]

    mase_vals = np.full((N,), np.nan, dtype=np.float64)
    wql_vals = np.full((N,), np.nan, dtype=np.float64)

    for i in range(N):
        y_past = as_1d_float_array(test_inputs[i]["target"])
        y_true = as_1d_float_array(test_labels[i]["target"])[:prediction_length]

        f = forecasts[i]
        fq = getattr(f, "forecast_arrays", None)
        if fq is None:
            fq = getattr(f, "forecast_array", None)
        if fq is None:
            continue

        fq = np.asarray(fq, dtype=np.float32)
        if fq.ndim != 2:
            continue

        Q = len(q_levels)
        if fq.shape[0] == Q and fq.shape[1] >= prediction_length:
            fq_qh = fq
        elif fq.shape[1] == Q and fq.shape[0] >= prediction_length:
            fq_qh = fq.T
        else:
            continue

        keys = getattr(f, "forecast_keys", None)
        if keys is not None and len(keys) == fq_qh.shape[0]:
            try:
                key_to_idx = {float(k): j for j, k in enumerate(keys)}
            except Exception:
                key_to_idx = {float(q_levels[j]): j for j in range(Q)}
        else:
            key_to_idx = {float(q_levels[j]): j for j in range(Q)}

        if 0.5 not in key_to_idx:
            continue

        q50_idx = key_to_idx[0.5]
        if fq_qh.shape[1] < prediction_length:
            continue

        yhat_med = fq_qh[q50_idx, :prediction_length].astype(np.float32)

        # MASE
        if y_past.size >= 2:
            denom = float(np.mean(np.abs(y_past[1:] - y_past[:-1]))) + 1e-8
        else:
            denom = np.nan

        if np.isfinite(denom) and denom > 0 and y_true.size == prediction_length:
            mase_vals[i] = float(np.mean(np.abs(y_true - yhat_med)) / denom)

        # WQL
        denom_w = float(np.sum(np.abs(y_true))) + 1e-8
        q_losses = []
        for q in q_levels:
            qi = key_to_idx.get(float(q), None)
            if qi is None:
                continue
            yhat_q = fq_qh[qi, :prediction_length].astype(np.float32)
            ql = _pinball_loss(y_true, yhat_q, float(q))
            q_losses.append((2.0 * ql) / denom_w)

        if q_losses:
            wql_vals[i] = float(np.mean(q_losses))

    return mase_vals, wql_vals


# -----------------------------
# Paired t-test
# -----------------------------
def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    """
    Returns (t, p, cohen_d_paired, n)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan"), int(x.size)

    d = x - y
    md = float(d.mean())
    sd = float(d.std(ddof=1) + 1e-12)
    d_cohen = md / sd

    try:
        from scipy.stats import ttest_rel  # type: ignore
        t, p = ttest_rel(x, y)
        return float(t), float(p), float(d_cohen), int(x.size)
    except Exception:
        t = md / (sd / np.sqrt(x.size))
        df = x.size - 1
        if df >= 30:
            from math import erf, sqrt
            z = abs(t)
            p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
            return float(t), float(p), float(d_cohen), int(x.size)
        return float(t), float("nan"), float(d_cohen), int(x.size)