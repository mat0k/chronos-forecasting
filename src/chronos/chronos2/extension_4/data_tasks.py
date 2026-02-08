#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import pandas as pd
import yaml
from gluonts.dataset.split import split

from core_utils import (
    as_1d_float_array,
    download_text,
    infer_freq_from_timestamp,
    safe_str,
)

FEV_TASKS_URLS = {
    "fev-bench": "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml",
}
GIFT_HF_REPO = "Salesforce/GiftEval"


@dataclass
class TaskConfig:
    hf_repo: str
    name: str
    offset: int
    prediction_length: int
    num_rolls: int = 1

    @staticmethod
    def from_dict(d: dict) -> "TaskConfig":
        # Already-normalized
        if all(k in d for k in ("hf_repo", "name", "offset", "prediction_length")):
            return TaskConfig(
                hf_repo=str(d["hf_repo"]),
                name=str(d["name"]),
                offset=int(d["offset"]),
                prediction_length=int(d["prediction_length"]),
                num_rolls=int(d.get("num_rolls", 1)),
            )

        # FEV format
        if "dataset_path" in d and "dataset_config" in d and "horizon" in d:
            hf_repo = d["dataset_path"]
            name = d["dataset_config"]
            pred_len = int(d["horizon"])
            num_rolls = int(d.get("num_windows", 1))
            offset = -pred_len * num_rolls
            return TaskConfig(
                hf_repo=hf_repo,
                name=name,
                prediction_length=pred_len,
                offset=offset,
                num_rolls=num_rolls,
            )

        raise RuntimeError(f"Unrecognized task format keys={list(d.keys())}")


class BenchmarkTaskLoader:
    def load(self, benchmark: str) -> List[TaskConfig]:
        b = benchmark.strip().lower()

        if b in FEV_TASKS_URLS:
            url = FEV_TASKS_URLS[b]
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
                        raise RuntimeError("Benchmark YAML must be a list or contain a list under a key like 'tasks'.")
                    tasks_raw = list_vals[0]
            else:
                raise RuntimeError("Benchmark YAML must be a list or a dict containing a list of tasks.")

            tasks: List[TaskConfig] = []
            for t in tasks_raw:
                if isinstance(t, dict):
                    tasks.append(TaskConfig.from_dict(t))
            if not tasks:
                raise RuntimeError("No tasks parsed from benchmark YAML.")
            return tasks

        if b in {"gift-eval", "gift_eval", "gift"}:
            cfgs = datasets.get_dataset_config_names(GIFT_HF_REPO)
            tasks: List[TaskConfig] = []
            for cfg_name in cfgs:
                m = re.search(r"(\d+)$", cfg_name)
                pred_len = int(m.group(1)) if m else 24
                tasks.append(
                    TaskConfig(
                        hf_repo=GIFT_HF_REPO,
                        name=cfg_name,
                        offset=-pred_len,
                        prediction_length=pred_len,
                        num_rolls=1,
                    )
                )
            if not tasks:
                raise RuntimeError("No GIFT-Eval tasks found (no configs returned).")
            return tasks

        raise RuntimeError(f"Unknown benchmark '{benchmark}'. Supported: fev-bench, gift-eval")

    def load_from_yaml(self, config_yaml: Path) -> List[TaskConfig]:
        with open(config_yaml, "r", encoding="utf-8") as fp:
            obj = yaml.safe_load(fp)
        if not isinstance(obj, list):
            raise ValueError("config_yaml must contain a list of dataset configs")
        return [TaskConfig.from_dict(x) for x in obj]


class HFToGluonTSConverter:
    def hf_to_gluonts_univariate(
        self,
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

                try:
                    target = as_1d_float_array(row["target"])
                except Exception:
                    continue

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
                try:
                    target = as_1d_float_array(row[field])
                except Exception:
                    continue
                gts_dataset.append({"start": start, "target": target, "item_id": item_id})
                sem_map[item_id] = sem
                item_counter += 1

        return gts_dataset, sem_map


class DatasetManager:
    def __init__(self) -> None:
        self.converter = HFToGluonTSConverter()

    def load_and_split_dataset(
        self,
        cfg: TaskConfig,
        semantic_field: Optional[str],
    ) -> Tuple[Any, List[dict], Dict[str, Optional[str]]]:
        ds = datasets.load_dataset(cfg.hf_repo, cfg.name, split="train")
        try:
            ds.set_format("numpy")
        except Exception:
            pass

        gts_dataset, sem_map = self.converter.hf_to_gluonts_univariate(ds, semantic_field=semantic_field)

        needed = abs(cfg.offset) + cfg.prediction_length + max(0, cfg.num_rolls - 1) * cfg.prediction_length

        filtered = []
        for e in gts_dataset:
            y = as_1d_float_array(e["target"])
            if len(y) >= needed:
                filtered.append(e)

        if not filtered:
            raise RuntimeError(
                f"{cfg.name}: No usable series after filtering (needed>={needed}). "
                f"Consider reducing num_windows/offset or choose another task."
            )

        train_data, test_template = split(filtered, offset=cfg.offset)
        test_data = test_template.generate_instances(cfg.prediction_length, windows=cfg.num_rolls)

        train_list = list(train_data)
        train_list.sort(key=lambda x: safe_str(x.get("item_id", "")))
        return test_data, train_list, sem_map