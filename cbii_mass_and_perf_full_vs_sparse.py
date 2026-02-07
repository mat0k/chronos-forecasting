#!/usr/bin/env python3
import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import fev

from pathlib import Path
import sys
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "src"))
from chronos.chronos2.pipeline import Chronos2Pipeline
from chronos.chronos2.dataset import (
    Chronos2Dataset,
    DatasetMode,
    convert_fev_window_to_list_of_dicts_input,
)

CHRONOS_BENCH_II_TASK_SOURCE = (
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/"
    "benchmarks/chronos_zeroshot/results/seasonal_naive.csv"
)


# ---------------------------
# Utilities: task loading
# ---------------------------
def _parse_listish(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if s in ("[]", ""):
        return []
    try:
        val = eval(s, {"__builtins__": {}}, {})
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return [p.strip() for p in s.strip("[]").split(",") if p.strip()]


def load_tasks_from_result_csv(url_or_path: str) -> list[fev.Task]:
    df = pd.read_csv(url_or_path)
    needed_cols = [
        "dataset_path",
        "dataset_config",
        "horizon",
        "num_windows",
        "initial_cutoff",
        "window_step_size",
        "min_context_length",
        "max_context_length",
        "seasonality",
        "eval_metric",
        "extra_metrics",
        "quantile_levels",
        "id_column",
        "timestamp_column",
        "target",
        "generate_univariate_targets_from",
        "known_dynamic_columns",
        "past_dynamic_columns",
        "static_columns",
        "task_name",
    ]
    specs = df[needed_cols].drop_duplicates(subset=["task_name"]).reset_index(drop=True)

    tasks: list[fev.Task] = []
    for _, row in specs.iterrows():
        max_context = row["max_context_length"]
        if isinstance(max_context, float) and np.isnan(max_context):
            max_context = None
        elif max_context == "" or max_context is None:
            max_context = None
        else:
            max_context = int(max_context)

        quantile_levels = row["quantile_levels"]
        if isinstance(quantile_levels, float) and np.isnan(quantile_levels):
            quantile_levels = None
        elif quantile_levels == "" or quantile_levels is None:
            quantile_levels = None
        else:
            q = _parse_listish(quantile_levels)
            quantile_levels = [float(v) for v in q] if q else None

        t = fev.Task(
            dataset_path=row["dataset_path"],
            dataset_config=row["dataset_config"],
            horizon=int(row["horizon"]),
            num_windows=int(row["num_windows"]),
            initial_cutoff=int(row["initial_cutoff"]),
            window_step_size=int(row["window_step_size"]),
            min_context_length=int(row["min_context_length"]),
            max_context_length=max_context,
            seasonality=int(row["seasonality"]),
            eval_metric=str(row["eval_metric"]),
            extra_metrics=_parse_listish(row["extra_metrics"]),
            quantile_levels=quantile_levels,
            id_column=str(row["id_column"]),
            timestamp_column=str(row["timestamp_column"]),
            target=str(row["target"]),
            generate_univariate_targets_from=(
                None
                if (row["generate_univariate_targets_from"] is None
                    or (isinstance(row["generate_univariate_targets_from"], float) and np.isnan(row["generate_univariate_targets_from"]))
                    or str(row["generate_univariate_targets_from"]).strip() == "")
                else str(row["generate_univariate_targets_from"])
            ),
            known_dynamic_columns=_parse_listish(row["known_dynamic_columns"]),
            past_dynamic_columns=_parse_listish(row["past_dynamic_columns"]),
            static_columns=_parse_listish(row["static_columns"]),
            task_name=str(row["task_name"]),
        )
        tasks.append(t)
    return tasks


# ---------------------------
# Context length: safe cap
# ---------------------------
from collections import Counter

def max_safe_context_length_for_task(task: fev.Task) -> Optional[int]:
    ds = task._load_dataset()
    try:
        split_names = list(ds.keys())
        for name in ("test", "validation", "val", "train"):
            if name in ds:
                split = ds[name]
                break
        else:
            split = ds[split_names[0]]
    except Exception:
        split = ds

    target_col = task.target
    first = split[target_col][0]

    if isinstance(first, (list, tuple, np.ndarray)):
        lengths = [len(x) for x in split[target_col]]
    else:
        id_col = getattr(task, "id_column", "id")
        ids = split[id_col]
        counts = Counter(ids)
        lengths = list(counts.values())

    if not lengths:
        return None

    min_len = min(lengths)
    cap = min_len - int(task.horizon)
    return int(cap) if cap > 0 else None


# ---------------------------
# Masks + mass computation
# ---------------------------
def parse_radii(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("Empty radii list")
    return out


def build_allowed_mask(
    *,
    S: int,
    num_output_patches: int,
    radius: int,
    use_reg_token: bool,
    time_reg_is_global: bool,
    device: torch.device,
) -> torch.Tensor:
    future_start = S - num_output_patches
    ctx_end = future_start
    M = torch.zeros((S, S), dtype=torch.bool, device=device)

    # context queries: local within context keys
    for q in range(0, max(0, ctx_end)):
        lo = max(0, q - radius)
        hi = min(ctx_end - 1, q + radius)
        if hi >= lo:
            M[q, lo:hi + 1] = True

    # future queries: global
    if future_start < S:
        M[future_start:S, 0:S] = True

    # REG global query row (optional)
    if use_reg_token and time_reg_is_global and ctx_end > 0:
        reg_idx = ctx_end - 1
        M[reg_idx, 0:ctx_end] = True
        M[reg_idx, ctx_end:S] = False

    return M


def compute_context_only_mass_fracs(
    *,
    time_attn_by_layer: Sequence[torch.Tensor],  # each [B,H,S,S]
    num_output_patches: int,
    radii: Sequence[int],
    use_reg_token: bool,
    time_reg_is_global: bool,
    exclude_reg_query: bool,
) -> Tuple[Dict[int, float], Dict[int, float], int, int, int, int]:
    """
    Returns:
      kept_frac_by_r, edges_frac_by_r, S, ctx_end, num_output_patches, q_sel
    computed ONLY over context queries (q < future_start), optionally excluding REG query row.
    """
    w0 = time_attn_by_layer[0]
    B, H, S, S2 = w0.shape
    assert S == S2

    future_start = S - num_output_patches
    ctx_end = future_start
    if ctx_end <= 0:
        return ({r: float("nan") for r in radii},
                {r: float("nan") for r in radii},
                S, ctx_end, num_output_patches, 0)

    device = w0.device
    reg_idx = (ctx_end - 1) if use_reg_token else None

    # restrict query set
    q_mask = torch.ones((S,), dtype=torch.bool, device=device)
    q_mask[ctx_end:S] = False  # keep only q < ctx_end
    if exclude_reg_query and (reg_idx is not None) and (0 <= reg_idx < ctx_end):
        q_mask[reg_idx] = False

    q_idx = q_mask.nonzero(as_tuple=False).view(-1)
    q_sel = int(q_idx.numel())
    if q_sel == 0:
        return ({r: float("nan") for r in radii},
                {r: float("nan") for r in radii},
                S, ctx_end, num_output_patches, 0)

    masks = {r: build_allowed_mask(
        S=S,
        num_output_patches=num_output_patches,
        radius=r,
        use_reg_token=use_reg_token,
        time_reg_is_global=time_reg_is_global,
        device=device,
    ) for r in radii}

    kept_frac_by_r: Dict[int, float] = {}
    edges_frac_by_r: Dict[int, float] = {}

    total_sum = 0.0
    dropped_sum_by_r = {r: 0.0 for r in radii}

    for layer_w in time_attn_by_layer:
        lw = layer_w.float()
        lw_q = lw[:, :, q_idx, :]  # [B,H,q_sel,S]
        total_sum += float(lw_q.sum().detach().cpu())

        for r in radii:
            M = masks[r]
            allowed_q = M[q_idx, :].view(1, 1, q_sel, S)
            dropped = float(lw_q.masked_fill(allowed_q, 0.0).sum().detach().cpu())
            dropped_sum_by_r[r] += dropped

    for r in radii:
        kept = total_sum - dropped_sum_by_r[r]
        kept_frac_by_r[r] = kept / total_sum if total_sum > 0 else float("nan")

    denom_edges = float(q_sel * S)
    for r in radii:
        M = masks[r]
        num_edges = float(M[q_idx, :].sum().item())
        edges_frac_by_r[r] = num_edges / denom_edges if denom_edges > 0 else float("nan")

    return kept_frac_by_r, edges_frac_by_r, S, ctx_end, num_output_patches, q_sel



# ---------------------------
# Performance evaluation (full vs sparse) using predict_fev
# ---------------------------
def load_pipeline_with_overrides(
    model_id: str,
    device_map: str,
    dtype: torch.dtype,
    *,
    time_attention_type: str,
    time_local_radius: Optional[int] = None,
    time_attention_chunk_size: Optional[int] = None,
    time_attention_backend: str = "torch",
) -> Chronos2Pipeline:
    kw = dict(
        device_map=device_map,
        dtype=dtype,
        time_attention_type=time_attention_type,
        time_use_landmarks=False,
        time_attention_backend=time_attention_backend,
    )
    if time_local_radius is not None:
        kw["time_local_radius"] = int(time_local_radius)
    if time_attention_chunk_size is not None:
        kw["time_attention_chunk_size"] = int(time_attention_chunk_size)
    try:
        return Chronos2Pipeline.from_pretrained(model_id, **kw)
    except TypeError:
        kw["torch_dtype"] = kw.pop("dtype")
        return Chronos2Pipeline.from_pretrained(model_id, **kw)


def eval_perf(
    pipe: Chronos2Pipeline,
    tasks: List[fev.Task],
    model_name: str,
    batch_size: int,
    as_univariate: bool,
    repeats: int = 5,
) -> pd.DataFrame:
    rows = []
    pipe.model.eval()

    on_cuda = torch.cuda.is_available() and str(next(pipe.model.parameters()).device).startswith("cuda")

    # one warmup (not counted)
    if on_cuda:
        warm_task = tasks[0]
        warm_ctx = min(int(pipe.model_context_length), 2048)
        _ = pipe.predict_fev(
            task=warm_task,
            batch_size=min(8, batch_size),
            as_univariate=as_univariate,
            context_length=warm_ctx,
        )
        torch.cuda.synchronize()

    for task in tasks:
        cap = max_safe_context_length_for_task(task)
        if cap is None:
            continue
        ctx = min(int(pipe.model_context_length), int(cap))
        ctx = max(int(task.min_context_length), ctx)
        if task.max_context_length is not None and not (
            isinstance(task.max_context_length, float) and np.isnan(task.max_context_length)
        ):
            ctx = min(ctx, int(task.max_context_length))

        times = []
        peak_alloc = []
        peak_reserved = []

        summary = None  # compute once (first repeat)

        for rep in range(max(1, int(repeats))):
            if on_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            preds_per_window, _infer_time = pipe.predict_fev(
                task=task,
                batch_size=batch_size,
                as_univariate=as_univariate,
                context_length=ctx,
            )
            if on_cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            times.append(float(dt))

            if on_cuda:
                peak_alloc.append(float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2))   # MB
                peak_reserved.append(float(torch.cuda.max_memory_reserved()) / (1024.0 ** 2)) # MB
            else:
                peak_alloc.append(float("nan"))
                peak_reserved.append(float("nan"))

            # compute metric once (first repeat) to keep runtime reasonable
            if rep == 0:
                summary = task.evaluation_summary(preds_per_window, model_name=model_name)

        if summary is None:
            continue

        # attach timing + memory stats
        t = np.asarray(times, dtype=np.float64)
        a = np.asarray(peak_alloc, dtype=np.float64)
        r = np.asarray(peak_reserved, dtype=np.float64)

        summary["ctx_used"] = int(ctx)
        summary["inference_time_s_median"] = float(np.median(t))
        summary["inference_time_s_mean"] = float(np.mean(t))
        summary["inference_time_s_std"] = float(np.std(t, ddof=1)) if t.size > 1 else float("nan")
        summary["perf_repeats"] = int(t.size)

        summary["peak_alloc_mb_median"] = float(np.nanmedian(a))
        summary["peak_alloc_mb_max"] = float(np.nanmax(a))
        summary["peak_reserved_mb_median"] = float(np.nanmedian(r))
        summary["peak_reserved_mb_max"] = float(np.nanmax(r))

        rows.append(summary)

    return pd.DataFrame(rows)



def paired_ttest(delta: np.ndarray) -> Dict[str, float]:
    # Paired t-test against 0, without scipy (simple, OK for reporting)
    delta = delta[np.isfinite(delta)]
    n = int(delta.size)
    if n < 2:
        return {"n": n, "mean": float(np.mean(delta)) if n else float("nan"), "t": float("nan"), "p_approx": float("nan")}
    mean = float(np.mean(delta))
    sd = float(np.std(delta, ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")

    # normal approximation for p-value (ok when n is moderate; for small n use scipy ideally)
    # two-sided:
    # p ~= 2 * (1 - Phi(|t|))
    # Phi(x) = 0.5*(1+erf(x/sqrt(2)))
    import math as _m
    p = 2.0 * (1.0 - 0.5 * (1.0 + _m.erf(abs(t) / _m.sqrt(2.0))))
    return {"n": n, "mean": mean, "sd": sd, "t": float(t), "p_approx": float(p)}


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # shared
    ap.add_argument("--model_id", type=str, default="amazon/chronos-2")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--as_univariate", action="store_true")
    ap.add_argument("--exclude_regex", type=str, default="")
    ap.add_argument("--max_tasks", type=int, default=0)
    ap.add_argument("--max_windows_per_task", type=int, default=1)

    # mass probe
    ap.add_argument("--radii", type=str, default="2,4,8,16,32,64,128")
    ap.add_argument("--probe_batch_size", type=int, default=4)
    ap.add_argument("--max_series_per_window", type=int, default=256)
    ap.add_argument("--time_reg_is_global", action="store_true")
    ap.add_argument("--exclude_reg_query", action="store_true")
    ap.add_argument("--S_split", type=int, default=128, help="Split small/large by S_tokens >= this threshold")
    ap.add_argument("--perf_repeats", type=int, default=5,
                help="Number of repeats per task for timing/memory (metric computed on first repeat).")
    # performance
    ap.add_argument("--perf_batch_size", type=int, default=256)
    ap.add_argument("--local_radius", type=int, default=8)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--backend", type=str, default="torch", choices=["torch", "flash"])
    ap.add_argument(
        "--perf_radii",
        type=str,
        default="",
        help="Comma-separated radii for PERFORMANCE sweep (default: use --radii). Example: 2,4,8,16,32,64,128",
    )
        # outputs
    ap.add_argument("--out_prefix", type=str, default="cbii_mass_perf")

    args = ap.parse_args()

    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    try:
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    except Exception:
        pass

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device_map = "cpu" if args.device == "cpu" else "cuda"

    tasks = load_tasks_from_result_csv(CHRONOS_BENCH_II_TASK_SOURCE)
    if args.exclude_regex:
        rx = re.compile(args.exclude_regex)
        tasks = [t for t in tasks if not rx.search(t.task_name)]
    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    radii = parse_radii(args.radii)
    perf_radii = parse_radii(args.perf_radii) if args.perf_radii.strip() else radii


    # ---------------- MASS PROBE (FULL attention, context queries only)
    pipe_full_probe = load_pipeline_with_overrides(
        args.model_id, device_map, dtype,
        time_attention_type="full",
        time_attention_backend="torch",  # must be eager to return attentions anyway
    )
    pipe_full_probe.model.eval()
    if hasattr(pipe_full_probe.model.config, "time_reg_is_global"):
        pipe_full_probe.model.config.time_reg_is_global = bool(args.time_reg_is_global)

    use_reg_token = bool(getattr(pipe_full_probe.model.chronos_config, "use_reg_token", False))
    out_patch = int(pipe_full_probe.model.chronos_config.output_patch_size)
    max_out_patches = int(pipe_full_probe.model.chronos_config.max_output_patches)

    per_window_rows = []

    # collect by radius and by split (small/large S)
    kept_by_r_all = {r: [] for r in radii}
    kept_by_r_small = {r: [] for r in radii}
    kept_by_r_large = {r: [] for r in radii}

    for task in tasks:
        cap = max_safe_context_length_for_task(task)
        if cap is None:
            continue
        ctx = min(int(pipe_full_probe.model_context_length), int(cap))
        ctx = max(int(task.min_context_length), ctx)
        if task.max_context_length is not None and not (isinstance(task.max_context_length, float) and np.isnan(task.max_context_length)):
            ctx = min(ctx, int(task.max_context_length))

        w_seen = 0
        for wi, window in enumerate(task.iter_windows()):
            if args.max_windows_per_task and args.max_windows_per_task > 0 and w_seen >= args.max_windows_per_task:
                break

            inputs, *_ = convert_fev_window_to_list_of_dicts_input(window=window, as_univariate=args.as_univariate)
            if args.max_series_per_window and len(inputs) > args.max_series_per_window:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(inputs), size=args.max_series_per_window, replace=False)
                inputs = [inputs[i] for i in idx]

            # build dataset + dataloader (probe batch size small)
            ds = Chronos2Dataset.convert_inputs(
                inputs=inputs,
                context_length=ctx,
                prediction_length=int(window.horizon),
                batch_size=args.probe_batch_size,
                output_patch_size=out_patch,
                mode=DatasetMode.TEST,
                require_full_context=False,
            )
            from torch.utils.data import DataLoader
            dl = DataLoader(ds, batch_size=None, shuffle=False, drop_last=False)

            num_output_patches = int(math.ceil(int(window.horizon) / out_patch))
            num_output_patches = min(num_output_patches, max_out_patches)

            # accumulate mass across mini-batches for this window
            # We do it by summing numerators/denominators implicitly inside compute fn per batch and then averaging
            # Simpler: compute per batch and append; here: treat each mini-batch as one sample (ok if probe_batch_size fixed)

            win_weight_sum = 0.0
            win_weighted_kept_sum = {r: 0.0 for r in radii}
            win_weighted_edges_sum = {r: 0.0 for r in radii}
            
            # Keep one set of meta values for the window (should be consistent across batches)
            win_meta = {"S": None, "ctx_end": None, "fut": None, "q_sel": None}
            for batch in dl:
                model_dtype = next(pipe_full_probe.model.parameters()).dtype
                bctx = batch["context"].to(device=pipe_full_probe.model.device, dtype=model_dtype)
                bgid = batch["group_ids"].to(device=pipe_full_probe.model.device)
                bfcov = batch["future_covariates"].to(device=pipe_full_probe.model.device, dtype=model_dtype)

                with torch.no_grad():
                    out = pipe_full_probe.model(
                        context=bctx,
                        group_ids=bgid,
                        future_covariates=bfcov,
                        num_output_patches=num_output_patches,
                        output_attentions=True,
                    )
                time_attn = out.enc_time_self_attn_weights
                if time_attn is None:
                    raise RuntimeError("enc_time_self_attn_weights is None in full+output_attentions=True")

                

                kept_frac_by_r, edges_frac_by_r, S, ctx_end, fut, q_sel = compute_context_only_mass_fracs(
                    time_attn_by_layer=time_attn,
                    num_output_patches=num_output_patches,
                    radii=radii,
                    use_reg_token=use_reg_token,
                    time_reg_is_global=bool(args.time_reg_is_global),
                    exclude_reg_query=bool(args.exclude_reg_query),
                )
                
                # Save meta once
                if win_meta["S"] is None:
                    win_meta["S"] = int(S)
                    win_meta["ctx_end"] = int(ctx_end)
                    win_meta["fut"] = int(fut)
                    win_meta["q_sel"] = int(q_sel)
                B_batch = int(batch["group_ids"].shape[0])
                if q_sel <=0:
                    del out, time_attn
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                # Weight: proportional to how much mass we summed (B * q_sel)
                w = float(B_batch * max(1, q_sel))  # q_sel can be 0 in degenerate cases
                win_weight_sum += w

                for r in radii:
                    k = kept_frac_by_r[r]
                    e = edges_frac_by_r[r]
                    if np.isfinite(k):
                        win_weighted_kept_sum[r] += w * float(k)
                    if np.isfinite(e):
                        win_weighted_edges_sum[r] += w * float(e)

                del out, time_attn
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            
            if win_weight_sum <= 0:
                continue
            
            for r in radii:
                kept_win = win_weighted_kept_sum[r] / win_weight_sum
                edges_win = win_weighted_edges_sum[r] / win_weight_sum
            
                per_window_rows.append({
                    "task_name": task.task_name,
                    "window_index": wi,
                    "ctx_used": ctx,
                    "horizon": int(window.horizon),
                    "radius": r,
                    "kept_mass_frac_ctxQ": float(kept_win),
                    "dropped_mass_frac_ctxQ": float(1.0 - kept_win),
                    "kept_edges_frac_ctxQ": float(edges_win),
                    "S_tokens": win_meta["S"],
                    "ctx_tokens": win_meta["ctx_end"],
                    "future_tokens": win_meta["fut"],
                    "q_sel_ctx_queries": win_meta["q_sel"],
                    "exclude_reg_query": bool(args.exclude_reg_query),
                    "probe_weight_sum": float(win_weight_sum),
                })
            
                kept_by_r_all[r].append(float(kept_win))
                if (win_meta["S"] is not None) and (int(win_meta["S"]) >= args.S_split):
                    kept_by_r_large[r].append(float(kept_win))
                else:
                    kept_by_r_small[r].append(float(kept_win))
            w_seen += 1
    per_window_path = f"{args.out_prefix}__mass_per_window.csv"
    pd.DataFrame(per_window_rows).to_csv(per_window_path, index=False)

    # summary table
    def summarize(vals: List[float]) -> Tuple[float, float, int]:
        x = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if x.size == 0:
            return float("nan"), float("nan"), 0
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
        return mean, std, int(x.size)

    summary_rows = []
    for r in radii:
        m_all, s_all, n_all = summarize(kept_by_r_all[r])
        m_s, s_s, n_s = summarize(kept_by_r_small[r])
        m_l, s_l, n_l = summarize(kept_by_r_large[r])
        summary_rows.append({
            "radius": r,
            "kept_mass_mean_ctxQ_all": m_all, "kept_mass_std_ctxQ_all": s_all, "n_all": n_all,
            f"kept_mass_mean_ctxQ_S<{args.S_split}": m_s, f"kept_mass_std_ctxQ_S<{args.S_split}": s_s, "n_small": n_s,
            f"kept_mass_mean_ctxQ_S>={args.S_split}": m_l, f"kept_mass_std_ctxQ_S>={args.S_split}": s_l, "n_large": n_l,
        })

    summary_path = f"{args.out_prefix}__mass_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # ---------------- PERFORMANCE (FULL vs SPARSE) on tasks  (SWEEP radii)
    # Full baseline (run ONCE)
    pipe_full = load_pipeline_with_overrides(
        args.model_id, device_map, dtype,
        time_attention_type="full",
        time_attention_backend=args.backend,
    )
    
    df_full = eval_perf(
        pipe_full, tasks,
        f"{args.model_id}__full",
        args.perf_batch_size, args.as_univariate,
        repeats=args.perf_repeats,
    )
    
    perf_full_path = f"{args.out_prefix}__perf_full.csv"
    df_full.to_csv(perf_full_path, index=False)
    
    metric_col = "MASE" if "MASE" in df_full.columns else ("test_error" if "test_error" in df_full.columns else None)
    if metric_col is None:
        raise RuntimeError(f"Could not find metric column in df_full. Columns: {df_full.columns}")
    
    # Helpers: merge and t-test per radius
    def merge_full_sparse(df_full: pd.DataFrame, df_sparse: pd.DataFrame) -> pd.DataFrame:
        merged = df_full[[
            "task_name", metric_col,
            "inference_time_s_median",
            "peak_alloc_mb_median", "peak_reserved_mb_median",
        ]].merge(
            df_sparse[[
                "task_name", metric_col,
                "inference_time_s_median",
                "peak_alloc_mb_median", "peak_reserved_mb_median",
            ]],
            on="task_name",
            suffixes=("_full", "_sparse"),
        )
    
        merged["delta_metric"] = merged[f"{metric_col}_sparse"] - merged[f"{metric_col}_full"]
        merged["speedup"] = merged["inference_time_s_median_full"] / np.maximum(
            1e-9, merged["inference_time_s_median_sparse"]
        )
    
        merged["delta_peak_alloc_mb"] = merged["peak_alloc_mb_median_sparse"] - merged["peak_alloc_mb_median_full"]
        merged["delta_peak_reserved_mb"] = merged["peak_reserved_mb_median_sparse"] - merged["peak_reserved_mb_median_full"]
        return merged
    
    def ttest_sparse_vs_full(merged: pd.DataFrame) -> Dict[str, float]:
        try:
            from scipy.stats import ttest_rel
    
            x = merged[f"{metric_col}_sparse"].to_numpy(dtype=np.float64)
            y = merged[f"{metric_col}_full"].to_numpy(dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            res = ttest_rel(x, y)  # mean(x-y)==0
            delta = x - y
            return {
                "n": int(delta.size),
                "mean_delta": float(np.mean(delta)) if delta.size else float("nan"),
                "sd_delta": float(np.std(delta, ddof=1)) if delta.size > 1 else float("nan"),
                "t": float(res.statistic),
                "p": float(res.pvalue),
                "test": "ttest_rel",
            }
        except Exception:
            tt = paired_ttest(merged["delta_metric"].to_numpy(dtype=np.float64))
            return {
                "n": int(tt.get("n", 0)),
                "mean_delta": float(tt.get("mean", float("nan"))),
                "sd_delta": float(tt.get("sd", float("nan"))),
                "t": float(tt.get("t", float("nan"))),
                "p": float(tt.get("p_approx", float("nan"))),
                "test": "normal_approx_on_delta",
            }
    
    # Sweep sparse radii
    radius_summary_rows = []
    all_comp_rows = []   # optional: a long table with all (task,radius) comparisons
    
    for r in perf_radii:
        print(f"[PERF] sparse radius={r}", flush=True)
    
        pipe_sparse = load_pipeline_with_overrides(
            args.model_id, device_map, dtype,
            time_attention_type="windowed_future_global",
            time_local_radius=int(r),
            time_attention_chunk_size=args.chunk_size,
            time_attention_backend=args.backend,
        )
    
        df_sparse = eval_perf(
            pipe_sparse, tasks,
            f"{args.model_id}__sparse_r{int(r)}_c{args.chunk_size}",
            args.perf_batch_size, args.as_univariate,
            repeats=args.perf_repeats,
        )
    
        perf_sparse_path = f"{args.out_prefix}__perf_sparse_r{int(r)}.csv"
        df_sparse.to_csv(perf_sparse_path, index=False)
    
        merged = merge_full_sparse(df_full, df_sparse)
        merged.insert(1, "radius", int(r))  # keep radius near front
    
        perf_comp_path = f"{args.out_prefix}__perf_compare_r{int(r)}.csv"
        merged.to_csv(perf_comp_path, index=False)
    
        # t-test per radius
        tt = ttest_sparse_vs_full(merged)
        ttest_path = f"{args.out_prefix}__perf_ttest_r{int(r)}.csv"
        pd.DataFrame([{
            "metric_col": metric_col,
            "radius": int(r),
            "chunk_size": args.chunk_size,
            "backend": args.backend,
            **tt,
            "mean_speedup": float(np.nanmean(merged["speedup"].to_numpy(dtype=np.float64))),
            "median_speedup": float(np.nanmedian(merged["speedup"].to_numpy(dtype=np.float64))),
            "mean_delta_peak_alloc_mb": float(np.nanmean(merged["delta_peak_alloc_mb"].to_numpy(dtype=np.float64))),
            "mean_delta_peak_reserved_mb": float(np.nanmean(merged["delta_peak_reserved_mb"].to_numpy(dtype=np.float64))),
        }]).to_csv(ttest_path, index=False)
    
        radius_summary_rows.append({
            "radius": int(r),
            "metric_col": metric_col,
            "chunk_size": args.chunk_size,
            "backend": args.backend,
            **tt,
            "mean_speedup": float(np.nanmean(merged["speedup"].to_numpy(dtype=np.float64))),
            "median_speedup": float(np.nanmedian(merged["speedup"].to_numpy(dtype=np.float64))),
            "mean_delta_peak_alloc_mb": float(np.nanmean(merged["delta_peak_alloc_mb"].to_numpy(dtype=np.float64))),
            "mean_delta_peak_reserved_mb": float(np.nanmean(merged["delta_peak_reserved_mb"].to_numpy(dtype=np.float64))),
        })
    
        # optional: keep a long-form combined table
        all_comp_rows.append(merged)
    
    # Write sweep summary tables
    radius_summary_path = f"{args.out_prefix}__perf_radius_sweep_summary.csv"
    pd.DataFrame(radius_summary_rows).to_csv(radius_summary_path, index=False)
    
    if all_comp_rows:
        perf_comp_all_path = f"{args.out_prefix}__perf_compare_all_radii.csv"
        pd.concat(all_comp_rows, axis=0, ignore_index=True).to_csv(perf_comp_all_path, index=False)
    
    print("Wrote:")
    print(" ", per_window_path)
    print(" ", summary_path)
    print(" ", perf_full_path)
    print(" ", radius_summary_path)
    if all_comp_rows:
        print(" ", perf_comp_all_path)
    print("Done.")

    # print(" ", perf_comp_path)
    # print(" ", ttest_path)
    # print("Done.")


if __name__ == "__main__":
    main()
