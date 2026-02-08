from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import datasets
import numpy as np
import pandas as pd
from gluonts.dataset.split import split

from .utils import as_1d_float_array, safe_str, infer_freq_from_timestamp
from .io_benchmarks import TaskConfig


@dataclass(frozen=True)
class DatasetBundle:
    test_data: Any
    test_input: List[dict]
    sem_map: Dict[str, Optional[str]]


def hf_to_gluonts_univariate(
    hf_dataset: datasets.Dataset,
    semantic_field: Optional[str],
) -> Tuple[List[dict], Dict[str, Optional[str]]]:
    gts_dataset: List[dict] = []
    sem_map: Dict[str, Optional[str]] = {}

    cols = set(hf_dataset.column_names)
    has_start = "start" in cols
    has_target = "target" in cols

    # Case A: already (start, target)
    if has_start and has_target:
        freq = "D"
        if "freq" in cols:
            try:
                freq = safe_str(hf_dataset[0]["freq"])
            except Exception:
                pass

        for i, row in enumerate(hf_dataset):
            item_id = safe_str(row.get("item_id", row.get("id", f"series_{i}")))
            start_raw = row["start"]
            try:
                start = pd.Period(start_raw, freq=freq)
            except Exception:
                start = pd.Period(pd.Timestamp(start_raw), freq=freq)

            target = as_1d_float_array(row["target"])

            sem = None
            if semantic_field and semantic_field in row:
                sem = safe_str(row[semantic_field])

            gts_dataset.append({"start": start, "target": target, "item_id": item_id})
            sem_map[item_id] = sem

        return gts_dataset, sem_map

    # Case B: timestamp-based multivariate -> flatten to multiple univariate series
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
            sem = safe_str(row[semantic_field])

        base = safe_str(row.get("item_id", row.get("id", f"row_{ridx}")))

        for field in series_fields:
            item_id = f"{base}_{field}_{item_counter}"
            target = as_1d_float_array(row[field])
            gts_dataset.append({"start": start, "target": target, "item_id": item_id})
            sem_map[item_id] = sem
            item_counter += 1

    return gts_dataset, sem_map


def load_and_split_dataset(cfg: TaskConfig, semantic_field: Optional[str]) -> DatasetBundle:
    ds = datasets.load_dataset(cfg.hf_repo, cfg.name, split="train")
    ds.set_format("numpy")
    gts_dataset, sem_map = hf_to_gluonts_univariate(ds, semantic_field=semantic_field)

    train_data, test_template = split(gts_dataset, offset=cfg.offset)
    test_data = test_template.generate_instances(cfg.prediction_length, windows=cfg.num_rolls)
    test_input = list(test_data.input)

    return DatasetBundle(test_data=test_data, test_input=test_input, sem_map=sem_map)