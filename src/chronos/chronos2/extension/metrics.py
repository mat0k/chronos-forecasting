from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast

from .constants import QUANTILES, MASE_KEY, WQL_KEY


@dataclass(frozen=True)
class ModeResult:
    mase: float
    wql: float


class MetricsComputer:
    def compute(self, forecasts: List[QuantileForecast], test_data) -> ModeResult:
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        mase = float(metrics[0].get(MASE_KEY, np.nan))
        wql = float(metrics[0].get(WQL_KEY, np.nan))
        return ModeResult(mase=mase, wql=wql)


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