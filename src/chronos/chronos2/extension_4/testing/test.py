#!/usr/bin/env python3
"""
Export (1) a summary CSV and (2) a per-task optimization CSV from a Chronos-2 FEV results CSV.

Input CSV must include columns:
  - benchmark_task
  - mode  (baseline, cross_learning_random, semantic_cross_learning_upgraded)
  - MASE
  - WQL

Outputs:
  --out-summary: one row per metric with overall stats (after filtering)
  --out-per-task: one row per task with Base/Rand/Sem + gains% + win flags

Filtering:
  - Deduplicate repeated (benchmark_task, mode) by averaging metrics
  - Remove exploded tasks where max MASE across modes > threshold (default 10)

Usage:
  python fev_export_csvs.py \
    --csv chronos2_results.csv \
    --mase-explode-threshold 10 \
    --out-summary fev_summary.csv \
    --out-per-task fev_per_task_optim.csv
"""

from __future__ import annotations
import argparse
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


MODE_MAP = {
    "baseline": "Base",
    "cross_learning_random": "Rand",
    "semantic_cross_learning_upgraded": "Sem",
}

REQUIRED_COLS = {"benchmark_task", "mode", "MASE", "WQL"}


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _safe_ttest(d: np.ndarray) -> float:
    d = _finite(d)
    if d.size < 2:
        return float("nan")
    _, p = stats.ttest_1samp(d, 0.0, alternative="two-sided")
    return float(p)


def _safe_wilcoxon_greater(d: np.ndarray) -> float:
    """
    One-sided Wilcoxon signed-rank p-value for H1: median(d) > 0.
    Here d = Rand - Sem, so d>0 => Semantic improves.
    """
    d = _finite(d)
    d = d[d != 0.0]
    if d.size == 0:
        return float("nan")
    _, p = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
    return float(p)


def _pct_gain(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Percent gain of B over A, i.e. (A - B)/A * 100, where lower is better.
    Positive => B improves over A.
    Safe for division by zero: returns NaN where A==0.
    """
    a = a.astype(float)
    b = b.astype(float)
    denom = a.replace(0.0, np.nan)
    return (a - b) / denom * 100.0


def load_and_filter(csv_path: str, mase_explode_threshold: float) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only expected modes
    df = df[df["mode"].isin(MODE_MAP.keys())].copy()

    # Deduplicate repeated (task, mode) rows
    df = df.groupby(["benchmark_task", "mode"], as_index=False)[["MASE", "WQL"]].mean()

    # Wide MASE for explode filtering
    wide_mase = df.pivot(index="benchmark_task", columns="mode", values="MASE")

    # explode if ANY mode has MASE > threshold
    exploded_mask = wide_mase.max(axis=1) > float(mase_explode_threshold)
    exploded_tasks = wide_mase.index[exploded_mask].tolist()

    kept_tasks = wide_mase.index[~exploded_mask].tolist()
    df = df[df["benchmark_task"].isin(kept_tasks)].copy()

    df["mode_short"] = df["mode"].map(MODE_MAP)

    meta = {
        "tasks_total": int(wide_mase.shape[0]),
        "tasks_kept": int(len(kept_tasks)),
        "tasks_removed_exploded": int(len(exploded_tasks)),
        "exploded_tasks": exploded_tasks,
        "mase_explode_threshold": float(mase_explode_threshold),
    }
    return df, meta


def build_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per benchmark_task, containing Base/Rand/Sem for MASE/WQL and gains%.
    """
    # Pivot metrics to wide
    wide_mase = df.pivot(index="benchmark_task", columns="mode_short", values="MASE")
    wide_wql  = df.pivot(index="benchmark_task", columns="mode_short", values="WQL")

    # Ensure all 3 modes exist per task (drop incomplete tasks if any)
    common = wide_mase.dropna(subset=["Base", "Rand", "Sem"]).index.intersection(
        wide_wql.dropna(subset=["Base", "Rand", "Sem"]).index
    )
    wide_mase = wide_mase.loc[common]
    wide_wql  = wide_wql.loc[common]

    out = pd.DataFrame(index=common).reset_index().rename(columns={"benchmark_task": "task"})

    # Raw metrics
    out["mase_base"] = wide_mase["Base"].values
    out["mase_rand"] = wide_mase["Rand"].values
    out["mase_sem"]  = wide_mase["Sem"].values

    out["wql_base"] = wide_wql["Base"].values
    out["wql_rand"] = wide_wql["Rand"].values
    out["wql_sem"]  = wide_wql["Sem"].values

    # Gains (%) (positive => improvement of second over first)
    out["mase_gain_sem_vs_rand_pct"] = _pct_gain(out["mase_rand"], out["mase_sem"])
    out["mase_gain_sem_vs_base_pct"] = _pct_gain(out["mase_base"], out["mase_sem"])
    out["mase_gain_rand_vs_base_pct"] = _pct_gain(out["mase_base"], out["mase_rand"])

    out["wql_gain_sem_vs_rand_pct"] = _pct_gain(out["wql_rand"], out["wql_sem"])
    out["wql_gain_sem_vs_base_pct"] = _pct_gain(out["wql_base"], out["wql_sem"])
    out["wql_gain_rand_vs_base_pct"] = _pct_gain(out["wql_base"], out["wql_rand"])

    # Win flags (strictly lower is better)
    out["mase_sem_beats_rand"] = (out["mase_sem"] < out["mase_rand"]).astype(int)
    out["mase_rand_beats_sem"] = (out["mase_rand"] < out["mase_sem"]).astype(int)

    out["wql_sem_beats_rand"] = (out["wql_sem"] < out["wql_rand"]).astype(int)
    out["wql_rand_beats_sem"] = (out["wql_rand"] < out["wql_sem"]).astype(int)

    return out


def build_summary(per_task: pd.DataFrame, meta: Dict[str, object]) -> pd.DataFrame:
    """
    One row per metric with overall mean/median per mode, gain%, win-rate, paired tests.
    """
    rows = []
    n = int(len(per_task))

    for metric in ["mase", "wql"]:
        base = per_task[f"{metric}_base"].astype(float)
        rand = per_task[f"{metric}_rand"].astype(float)
        sem  = per_task[f"{metric}_sem"].astype(float)

        # d = Rand - Sem (positive => Semantic better)
        d = (rand - sem).to_numpy(dtype=float)

        gain_pct = float(_finite(_pct_gain(rand, sem)).mean())  # mean per-task percent gain
        win_rate = float((sem < rand).mean() * 100.0)

        p_t = _safe_ttest(d)
        p_w = _safe_wilcoxon_greater(d)

        rows.append({
            "metric": metric.upper(),
            "n_tasks": n,
            "base_mean": float(base.mean()),
            "base_median": float(base.median()),
            "rand_mean": float(rand.mean()),
            "rand_median": float(rand.median()),
            "sem_mean": float(sem.mean()),
            "sem_median": float(sem.median()),
            "sem_vs_rand_gain_pct_mean": gain_pct,
            "sem_vs_rand_win_rate_pct": win_rate,
            "p_ttest_two_sided": p_t,
            "p_wilcoxon_one_sided_sem_lt_rand": p_w,
            "tasks_total_in_csv": meta["tasks_total"],
            "tasks_removed_exploded": meta["tasks_removed_exploded"],
            "mase_explode_threshold": meta["mase_explode_threshold"],
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input Chronos-2 FEV results CSV.")
    ap.add_argument("--mase-explode-threshold", type=float, default=10.0,
                    help="Remove any task with MASE > threshold in ANY mode.")
    ap.add_argument("--out-summary", required=True, help="Output summary CSV path.")
    ap.add_argument("--out-per-task", required=True, help="Output per-task optimization CSV path.")
    ap.add_argument("--print-exploded", action="store_true",
                    help="Print names of exploded tasks that were removed.")
    args = ap.parse_args()

    df, meta = load_and_filter(args.csv, args.mase_explode_threshold)

    per_task = build_per_task(df)
    summary = build_summary(per_task, meta)

    summary.to_csv(args.out_summary, index=False)
    per_task.to_csv(args.out_per_task, index=False)

    print(f"Saved summary:   {args.out_summary}")
    print(f"Saved per-task:  {args.out_per_task}")
    print(f"Tasks total in CSV: {meta['tasks_total']}")
    print(f"Removed exploded:   {meta['tasks_removed_exploded']} (threshold={meta['mase_explode_threshold']})")
    print(f"Kept tasks:         {meta['tasks_kept']} (after explode filter; then dropped incomplete if any)")

    if args.print_exploded and meta["exploded_tasks"]:
        print("\nExploded tasks removed:")
        for t in meta["exploded_tasks"]:
            print(f"  - {t}")


if __name__ == "__main__":
    main()