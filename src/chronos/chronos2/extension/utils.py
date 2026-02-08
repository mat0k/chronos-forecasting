from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests


def make_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ceil_to_patch(h: int, patch: int) -> int:
    if patch <= 1:
        return int(h)
    return int(((h + patch - 1) // patch) * patch)


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def as_1d_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


def download_text(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


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


def set_all_seeds(seed: int) -> None:
    # for numpy
    np.random.seed(seed)
    # for python random
    random.seed(seed)
    # torch is seeded in the caller (optional) because torch import may be costly
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False
    out_dir: Path = Path("./debug_out")

    # if enabled, dumps per-task debug jsonl with batch composition and coherence scores
    dump_batches: bool = True

    # dumps clustering labels and group sizes
    dump_grouping: bool = True

    # dumps a small sample of series features
    dump_features: bool = False

    def ensure_dir(self) -> None:
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)