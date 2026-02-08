from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import datasets
import yaml

from .constants import FEV_TASKS_URLS, GIFT_HF_REPO
from .utils import download_text


@dataclass(frozen=True)
class TaskConfig:
    hf_repo: str
    name: str
    offset: int
    prediction_length: int
    num_rolls: int = 1


def normalize_task_dict(d: Dict[str, Any]) -> TaskConfig:
    # Already normalized
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
        hf_repo = str(d["dataset_path"])
        name = str(d["dataset_config"])
        pred_len = int(d["horizon"])
        num_rolls = int(d.get("num_windows", 1))
        offset = -pred_len * num_rolls
        return TaskConfig(hf_repo=hf_repo, name=name, prediction_length=pred_len, offset=offset, num_rolls=num_rolls)

    raise RuntimeError(f"Unrecognized task format keys={list(d.keys())}")


def load_benchmark_tasks(benchmark: str) -> List[TaskConfig]:
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
            else:
                list_vals = [v for v in obj.values() if isinstance(v, list)]
                if not list_vals:
                    raise RuntimeError("FEV YAML must be a list or contain a list under a key like 'tasks'.")
                tasks_raw = list_vals[0]
        else:
            raise RuntimeError("FEV YAML must be list/dict with a list of tasks.")

        tasks: List[TaskConfig] = []
        for t in tasks_raw:
            if isinstance(t, dict):
                tasks.append(normalize_task_dict(t))

        if not tasks:
            raise RuntimeError("No tasks parsed from FEV YAML.")
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
            raise RuntimeError("No GiftEval tasks found.")
        return tasks

    raise RuntimeError(f"Unknown benchmark '{benchmark}'. Supported: fev-bench, gift-eval")