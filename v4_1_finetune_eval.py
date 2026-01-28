#!/usr/bin/env python3
"""
v4_2_finetune_eval.py

Tiny controlled fine-tuning + evaluation for 3 attention variants:
  - FULL_SDPA          : full attention via PyTorch SDPA (forced MATH backend for "paper-like" baseline)
  - WINDOWED_FLASH     : your windowed attention using flash backend (no landmarks)
  - WINDOWED_FLASH_LM  : windowed attention using flash backend + landmarks

Two synthetic series:
  - long_seasonality   : long-period seasonal structure (full attention more likely to help)
  - short_memory_ar1   : short-memory AR(1) (windowed should be similar)

Key fixes vs your failing runs:
  ✅ always pass num_output_patches = ceil(pred_len / output_patch_size)
  ✅ do NOT assume encoder_config exists; use Chronos2CoreConfig top-level fields
  ✅ LM model must be constructed with time_use_landmarks=True (then load weights strict=False)
  ✅ evaluation runs with context-only inference (no teacher forcing), then compares to ground truth
  ✅ logs OOM/errors instead of crashing
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Ensure we can import repo code from src/
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if SRC.exists():
    import sys
    sys.path.insert(0, str(SRC))
else:
    raise RuntimeError(f"Could not find src/ next to {__file__}. REPO_ROOT={REPO_ROOT}")

from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model  # your repo


# -----------------------------------------------------------------------------
# SDPA backend forcing (paper-like full attention baseline)
# -----------------------------------------------------------------------------
def sdpa_force_math():
    """
    Prefer torch.nn.attention.sdpa_kernel (newer), else fall back to torch.backends.cuda.sdp_kernel.
    Returns a context manager.
    """
    if not (torch.cuda.is_available()):
        return nullcontext()

    # Newer API
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        pass

    # Older / fallback API
    try:
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        return nullcontext()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def safe_empty_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def cuda_sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# Synthetic data generators
# -----------------------------------------------------------------------------
def make_long_seasonality_series(total_len: int, period: int, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(total_len, dtype=np.float32)
    long = 1.0 * np.sin(2.0 * np.pi * t / period)
    harmonic = 0.25 * np.sin(2.0 * np.pi * t / max(16, period // 8))
    trend = 0.00003 * t
    noise = rng.normal(0.0, noise_std, size=total_len).astype(np.float32)
    y = long + harmonic + trend + noise
    return y.astype(np.float32)


def make_short_memory_ar1_series(total_len: int, phi: float, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(total_len, dtype=np.float32)
    eps = rng.normal(0.0, noise_std, size=total_len).astype(np.float32)
    for i in range(1, total_len):
        y[i] = phi * y[i - 1] + eps[i]
    t = np.arange(total_len, dtype=np.float32)
    y += 0.15 * np.sin(2.0 * np.pi * t / 32.0)
    return y.astype(np.float32)


def sample_windows(
    series: np.ndarray,
    ctx_len: int,
    pred_len: int,
    n_windows: int,
    stride_steps: int,
    rng: np.random.Generator,
):
    total = len(series)
    need = ctx_len + pred_len
    if total < need:
        raise ValueError(f"Series too short: total={total}, need={need}")

    max_start = total - need - (n_windows - 1) * stride_steps
    if max_start < 0:
        n_windows_eff = max(1, (total - need) // max(1, stride_steps) + 1)
        n_windows = min(n_windows, n_windows_eff)
        max_start = max(0, total - need - (n_windows - 1) * stride_steps)

    start0 = int(rng.integers(0, max_start + 1))
    out = []
    for k in range(n_windows):
        s = start0 + k * stride_steps
        ctx = series[s : s + ctx_len]
        fut = series[s + ctx_len : s + ctx_len + pred_len]
        out.append((ctx, fut))
    return out


# -----------------------------------------------------------------------------
# Variant config + loading
# -----------------------------------------------------------------------------
def make_variant_config(
    base_cfg: Chronos2CoreConfig,
    variant: str,
    radius_patches: int,
    lm_stride_patches: int,
    chunk_size_patches: int,
) -> Chronos2CoreConfig:
    """
    Your repo uses top-level fields on Chronos2CoreConfig (NOT encoder_config).
    """
    cfg = copy.deepcopy(base_cfg)
    v = variant.upper().strip()

    if v == "FULL_SDPA":
        cfg.time_attention_type = "full"
        cfg.time_attention_backend = "torch"
        cfg.time_use_landmarks = False
        cfg.time_local_radius = int(radius_patches)
        cfg.time_attention_chunk_size = int(chunk_size_patches)

    elif v == "WINDOWED_FLASH":
        cfg.time_attention_type = "windowed_future_global"
        cfg.time_attention_backend = "flash"
        cfg.time_use_landmarks = False
        cfg.time_local_radius = int(radius_patches)
        cfg.time_landmark_stride = int(lm_stride_patches)
        cfg.time_attention_chunk_size = int(chunk_size_patches)

    elif v == "WINDOWED_FLASH_LM":
        cfg.time_attention_type = "windowed_future_global"
        cfg.time_attention_backend = "flash"
        cfg.time_use_landmarks = True
        cfg.time_local_radius = int(radius_patches)
        cfg.time_landmark_stride = int(lm_stride_patches)
        cfg.time_attention_chunk_size = int(chunk_size_patches)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return cfg


def load_base_state_dict(ckpt: str, dtype: torch.dtype) -> Tuple[Chronos2CoreConfig, Dict[str, torch.Tensor]]:
    """
    Load base config + weights once, on CPU.
    """
    base_cfg = Chronos2CoreConfig.from_pretrained(ckpt)
    # load model on CPU to avoid GPU spikes
    try:
        base_model = Chronos2Model.from_pretrained(ckpt, dtype=dtype, device_map=None)
    except TypeError:
        base_model = Chronos2Model.from_pretrained(ckpt, torch_dtype=dtype, device_map=None)

    base_model.to("cpu")
    base_model.eval()
    state = base_model.state_dict()
    del base_model
    return base_cfg, state


def build_model_from_state(
    cfg: Chronos2CoreConfig,
    state: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
) -> Chronos2Model:
    """
    Instantiate from cfg, then load base weights strict=False (needed for LM new params).
    """
    m = Chronos2Model(cfg)
    missing, unexpected = m.load_state_dict(state, strict=False)

    # Move to device/dtype
    m.to(device=device)
    if dtype in (torch.float16, torch.bfloat16):
        m.to(dtype=dtype)

    m.eval()

    # Helpful debug (only once if needed)
    # print("missing:", missing)
    # print("unexpected:", unexpected)

    return m


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------
def get_median_index(model: Chronos2Model) -> int:
    qs = getattr(model, "quantiles", None)
    if qs is None:
        return 0
    qs = [float(q) for q in qs]
    return int(np.argmin([abs(q - 0.5) for q in qs]))


def num_output_patches_for(model: Chronos2Model, pred_len: int) -> int:
    """
    Chronos2 validates: pred_len <= num_output_patches * output_patch_size
    output_patch_size is in model.chronos_config.output_patch_size
    """
    out_patch = int(model.chronos_config.output_patch_size)
    return int(math.ceil(pred_len / out_patch))


# -----------------------------------------------------------------------------
# Forward: inference (context-only) and training (with targets), OOM-safe
# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_oom_safe(
    model: Chronos2Model,
    context: torch.Tensor,          # (B, ctx_len)
    pred_len: int,
    device: str,
    amp_dtype: torch.dtype,
    variant_name: str,
) -> Tuple[str, Optional[np.ndarray], float, float, str]:
    """
    Returns: (status, pred_median_np[B,pred_len] or None, runtime_ms, peak_mem_mb, err)
    """
    assert context.ndim == 2

    use_amp = device.startswith("cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # FULL_SDPA: force MATH SDPA to represent "paper full attention"
    sdp_ctx = sdpa_force_math() if (variant_name == "FULL_SDPA" and device.startswith("cuda")) else nullcontext()

    t0 = time.time()
    try:
        with sdp_ctx:
            with autocast_ctx:
                n_out = num_output_patches_for(model, pred_len)
                # Make sure model allows this many
                model.chronos_config.max_output_patches = max(int(model.chronos_config.max_output_patches), int(n_out))

                out = model(
                    context=context,
                    context_mask=None,
                    future_target=None,
                    future_target_mask=None,
                    num_output_patches=n_out,
                    group_ids=None,
                    future_covariates=None,
                )
                qpred = out.quantile_preds  # (B, Q, >=pred_len)
                mid = get_median_index(model)
                pred = qpred[:, mid, :pred_len].detach().float().cpu().numpy()

        cuda_sync(device)
        dt_ms = (time.time() - t0) * 1000.0

        peak_mb = 0.0
        if device.startswith("cuda") and torch.cuda.is_available():
            peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)

        return "OK", pred, dt_ms, peak_mb, ""

    except torch.cuda.OutOfMemoryError as e:
        cuda_sync(device)
        safe_empty_cache()
        return "OOM", None, (time.time() - t0) * 1000.0, 0.0, repr(e)

    except Exception as e:
        cuda_sync(device)
        safe_empty_cache()
        return "ERR", None, (time.time() - t0) * 1000.0, 0.0, f"{type(e).__name__}: {e}"


def train_step_oom_safe(
    model: Chronos2Model,
    context: torch.Tensor,          # (B, ctx_len)
    future: torch.Tensor,           # (B, pred_len)
    pred_len: int,
    opt: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
    amp_dtype: torch.dtype,
    variant_name: str,
) -> Tuple[str, float, str]:
    """
    Returns: (status, loss_value_or_nan, err)
    """
    assert context.ndim == 2 and future.ndim == 2

    use_amp = device.startswith("cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    # FULL_SDPA: force MATH SDPA for baseline
    sdp_ctx = sdpa_force_math() if (variant_name == "FULL_SDPA" and device.startswith("cuda")) else nullcontext()

    model.train()
    opt.zero_grad(set_to_none=True)

    try:
        with sdp_ctx:
            with autocast_ctx:
                n_out = num_output_patches_for(model, pred_len)
                model.chronos_config.max_output_patches = max(int(model.chronos_config.max_output_patches), int(n_out))

                out = model(
                    context=context,
                    context_mask=None,
                    future_target=future,
                    future_target_mask=None,
                    num_output_patches=n_out,
                    group_ids=None,
                    future_covariates=None,
                )
                loss = out.loss

        if loss is None:
            return "ERR", float("nan"), "loss was None"

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        return "OK", float(loss.detach().float().cpu().item()), ""

    except torch.cuda.OutOfMemoryError as e:
        safe_empty_cache()
        return "OOM", float("nan"), repr(e)

    except Exception as e:
        safe_empty_cache()
        return "ERR", float("nan"), f"{type(e).__name__}: {e}"

    finally:
        model.eval()


# -----------------------------------------------------------------------------
# Main loops
# -----------------------------------------------------------------------------
def tiny_finetune(
    model: Chronos2Model,
    series_dict: Dict[str, np.ndarray],
    context_lengths: list[int],
    pred_len: int,
    train_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    amp_dtype: torch.dtype,
    seed: int,
    variant_name: str,
) -> Dict[str, Any]:
    """
    Very small adaptation run. Logs step errors but keeps going.
    """
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = None
    if device.startswith("cuda") and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    last_ok_loss = float("nan")
    err_counts = {"OK": 0, "ERR": 0, "OOM": 0}

    for step in range(train_steps):
        task = "long_seasonality" if (step % 2 == 0) else "short_memory_ar1"
        series = series_dict[task]

        ctx_len = int(rng.choice(context_lengths))
        windows = sample_windows(
            series=series,
            ctx_len=ctx_len,
            pred_len=pred_len,
            n_windows=batch_size,
            stride_steps=max(1, pred_len // 2),
            rng=rng,
        )
        ctx_batch = np.stack([w[0] for w in windows], axis=0)
        fut_batch = np.stack([w[1] for w in windows], axis=0)

        context = torch.tensor(ctx_batch, device=device, dtype=torch.float32)
        future = torch.tensor(fut_batch, device=device, dtype=torch.float32)

        # cheap per-sample normalization (keeps synthetic stable)
        mean = context.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True).clamp_min(1e-5)
        context_n = (context - mean) / std
        future_n = (future - mean) / std

        status, loss_val, err = train_step_oom_safe(
            model=model,
            context=context_n,
            future=future_n,
            pred_len=pred_len,
            opt=opt,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            variant_name=variant_name,
        )

        err_counts[status] = err_counts.get(status, 0) + 1
        if status == "OK":
            last_ok_loss = loss_val
        else:
            out_patch = int(model.chronos_config.output_patch_size)
            n_out = num_output_patches_for(model, pred_len)
            print(
                f"[train] step={step:04d} {status}: {err} "
                f"(pred_len={pred_len}, output_patch_size={out_patch}, num_out={n_out})"
            )

    return {"last_ok_loss": last_ok_loss, "counts": err_counts}


def evaluate_variants(
    variants: Dict[str, Chronos2Model],
    series_dict: Dict[str, np.ndarray],
    context_lengths: list[int],
    pred_len: int,
    eval_windows: int,
    stride_steps: int,
    device: str,
    amp_dtype: torch.dtype,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []

    for dataset_name, series in series_dict.items():
        for ctx_len in context_lengths:
            windows = sample_windows(
                series=series,
                ctx_len=ctx_len,
                pred_len=pred_len,
                n_windows=eval_windows,
                stride_steps=stride_steps,
                rng=rng,
            )
            ctx_batch = np.stack([w[0] for w in windows], axis=0)
            fut_batch = np.stack([w[1] for w in windows], axis=0)

            context = torch.tensor(ctx_batch, device=device, dtype=torch.float32)
            future = torch.tensor(fut_batch, device=device, dtype=torch.float32)

            mean = context.mean(dim=1, keepdim=True)
            std = context.std(dim=1, keepdim=True).clamp_min(1e-5)
            context_n = (context - mean) / std
            future_n = (future - mean) / std

            y_true = future_n.detach().float().cpu().numpy()

            for vname, model in variants.items():
                status, pred, dt_ms, peak_mb, err = infer_oom_safe(
                    model=model,
                    context=context_n,
                    pred_len=pred_len,
                    device=device,
                    amp_dtype=amp_dtype,
                    variant_name=vname,
                )

                mae = rmse = float("nan")
                if status == "OK" and pred is not None:
                    diff = pred - y_true
                    mae = float(np.mean(np.abs(diff)))
                    rmse = float(np.sqrt(np.mean(diff ** 2)))

                row = {
                    "dataset": dataset_name,
                    "variant": vname,
                    "ctx_len": int(ctx_len),
                    "pred_len": int(pred_len),
                    "eval_windows": int(eval_windows),
                    "stride_steps": int(stride_steps),
                    "status": status,
                    "mae": mae,
                    "rmse": rmse,
                    "runtime_ms": float(dt_ms),
                    "peak_mem_mb": float(peak_mb),
                    "error": err,
                }
                rows.append(row)

                print(
                    f"[{dataset_name}] {vname:<18} ctx={ctx_len:5d} -> {status} "
                    f"mae={mae:.4f} rmse={rmse:.4f} time={dt_ms:.1f}ms peak={peak_mb:.0f}MB"
                    + (f" err={err}" if status != "OK" else "")
                )

    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: list[dict]) -> list[dict]:
    """
    Aggregate by dataset/variant/ctx_len (OK rows only).
    """
    buckets: Dict[Tuple[str, str, int], Dict[str, list]] = {}
    for r in rows:
        key = (r["dataset"], r["variant"], r["ctx_len"])
        b = buckets.setdefault(key, {"mae": [], "rmse": [], "runtime_ms": [], "peak_mem_mb": [], "status": []})
        b["status"].append(r["status"])
        if r["status"] == "OK":
            b["mae"].append(r["mae"])
            b["rmse"].append(r["rmse"])
            b["runtime_ms"].append(r["runtime_ms"])
            b["peak_mem_mb"].append(r["peak_mem_mb"])

    out = []
    for (ds, v, ctx), b in buckets.items():
        ok = len(b["mae"])
        total = len(b["status"])
        out.append({
            "dataset": ds,
            "variant": v,
            "ctx_len": ctx,
            "ok_rows": ok,
            "n_rows": total,
            "ok_rate": ok / total if total else 0.0,
            "mae_mean": float(np.mean(b["mae"])) if ok else float("nan"),
            "rmse_mean": float(np.mean(b["rmse"])) if ok else float("nan"),
            "runtime_ms_mean": float(np.mean(b["runtime_ms"])) if ok else float("nan"),
            "peak_mem_mb_mean": float(np.mean(b["peak_mem_mb"])) if ok else float("nan"),
            "statuses": {s: b["status"].count(s) for s in set(b["status"])},
        })
    out.sort(key=lambda x: (x["dataset"], x["variant"], x["ctx_len"]))
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="HF repo id or local path (e.g., amazon/chronos-2)")

    # accept both --device and the user's habitual --device_map
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--device_map", type=str, default=None, help="alias; if set to 'cuda' uses --device=cuda")

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--context_lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--prediction_length", type=int, default=512)

    ap.add_argument("--radius_patches", type=int, default=128)
    ap.add_argument("--lm_stride_patches", type=int, default=64)
    ap.add_argument("--chunk_size_patches", type=int, default=2048)

    ap.add_argument("--train_steps", type=int, default=80)
    ap.add_argument("--train_batch_size", type=int, default=2)
    ap.add_argument("--train_lr", type=float, default=1e-5)
    ap.add_argument("--train_weight_decay", type=float, default=0.01)

    ap.add_argument("--eval_windows", type=int, default=6)
    ap.add_argument("--window_stride_steps", type=int, default=256)

    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--out_csv", type=str, default=f"quality_{now_ts()}.csv")
    ap.add_argument("--out_json", type=str, default=f"quality_{now_ts()}.json")

    args = ap.parse_args()

    if args.device_map is not None and args.device_map.lower() == "cuda":
        args.device = "cuda"

    device = args.device
    amp_dtype = parse_dtype(args.dtype)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You requested --device cuda, but torch.cuda.is_available() is False.")

    set_all_seeds(args.seed)

    # Build synthetic series long enough for max ctx + pred + sliding windows + headroom
    max_ctx = max(args.context_lengths)
    pred_len = args.prediction_length
    total_len = max_ctx + pred_len + (args.eval_windows - 1) * args.window_stride_steps + 4096

    series_dict = {
        "long_seasonality": make_long_seasonality_series(
            total_len=total_len,
            period=max(2048, max_ctx // 4),
            noise_std=0.05,
            seed=args.seed + 1,
        ),
        "short_memory_ar1": make_short_memory_ar1_series(
            total_len=total_len,
            phi=0.8,
            noise_std=0.10,
            seed=args.seed + 2,
        ),
    }

    # Load base weights once
    print("\n=== Loading base weights (once) ===")
    base_cfg, base_state = load_base_state_dict(args.ckpt, dtype=amp_dtype)

    # Build 3 variants
    print("\n=== Loading models ===")
    variants: Dict[str, Chronos2Model] = {}
    for name in ["FULL_SDPA", "WINDOWED_FLASH", "WINDOWED_FLASH_LM"]:
        print(f"[load] {name}")
        cfg = make_variant_config(
            base_cfg=base_cfg,
            variant=name,
            radius_patches=args.radius_patches,
            lm_stride_patches=args.lm_stride_patches,
            chunk_size_patches=args.chunk_size_patches,
        )
        m = build_model_from_state(cfg=cfg, state=base_state, device=device, dtype=amp_dtype)
        variants[name] = m

    # Tiny finetune
    print("\n=== Tiny controlled fine-tuning (few steps; not full finetuning) ===")
    finetune_logs = {}
    for vname, model in variants.items():
        print(f"\n[finetune] {vname}")
        log = tiny_finetune(
            model=model,
            series_dict=series_dict,
            context_lengths=args.context_lengths,
            pred_len=pred_len,
            train_steps=args.train_steps,
            batch_size=args.train_batch_size,
            lr=args.train_lr,
            weight_decay=args.train_weight_decay,
            device=device,
            amp_dtype=amp_dtype,
            seed=args.seed + 10,
            variant_name=vname,
        )
        finetune_logs[vname] = log
        print(f"[finetune] {vname} done. last_ok_loss={log['last_ok_loss']} counts={log['counts']}")

    # Evaluate
    print("\n=== Evaluation (context-only inference; no teacher forcing) ===")
    rows = evaluate_variants(
        variants=variants,
        series_dict=series_dict,
        context_lengths=args.context_lengths,
        pred_len=pred_len,
        eval_windows=args.eval_windows,
        stride_steps=args.window_stride_steps,
        device=device,
        amp_dtype=amp_dtype,
        seed=args.seed + 999,
    )

    # Attach metadata
    for r in rows:
        r.update({
            "ckpt": args.ckpt,
            "device": args.device,
            "dtype": args.dtype,
            "radius_patches": args.radius_patches,
            "lm_stride_patches": args.lm_stride_patches,
            "chunk_size_patches": args.chunk_size_patches,
            "train_steps": args.train_steps,
            "train_batch_size": args.train_batch_size,
            "train_lr": args.train_lr,
            "train_weight_decay": args.train_weight_decay,
            "seed": args.seed,
        })

    write_csv(args.out_csv, rows)
    summary = summarize(rows)

    out_obj = {
        "meta": {
            "ckpt": args.ckpt,
            "device": args.device,
            "dtype": args.dtype,
            "context_lengths": args.context_lengths,
            "prediction_length": args.prediction_length,
            "radius_patches": args.radius_patches,
            "lm_stride_patches": args.lm_stride_patches,
            "chunk_size_patches": args.chunk_size_patches,
            "train": {
                "steps": args.train_steps,
                "batch_size": args.train_batch_size,
                "lr": args.train_lr,
                "weight_decay": args.train_weight_decay,
            },
            "eval": {
                "windows": args.eval_windows,
                "stride_steps": args.window_stride_steps,
            },
            "seed": args.seed,
            "finetune_logs": finetune_logs,
        },
        "summary": summary,
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"\n[done] wrote {len(rows)} rows")
    print(f"[done] CSV:  {args.out_csv}")
    print(f"[done] JSON: {args.out_json}")


if __name__ == "__main__":
    main()
