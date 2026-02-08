from __future__ import annotations

from typing import Tuple

import numpy as np


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
    scale_bins: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
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
    mag[0] = 0.0
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

    feat = np.array([m, s, log_s, slope, skew, kurt, domp, seas, acf_max] + acfs, dtype=np.float32)
    return np.clip(feat, -50.0, 50.0)


def zscore_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd