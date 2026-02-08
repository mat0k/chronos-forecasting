from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm

from .constants import QUANTILES
from .utils import as_1d_float_array, safe_str, DebugConfig
from .batching import cosine_to_centroid, SemanticBatchPlan


def _to_HQ(x: Any, q_levels: List[float]) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected per-item quantile shape: {arr.shape}")

    Q = len(q_levels)
    if arr.shape[0] == Q and arr.shape[1] != Q:
        return arr.T
    if arr.shape[1] == Q and arr.shape[0] != Q:
        return arr
    if arr.shape[0] == Q and arr.shape[1] == Q:
        return arr.T

    raise ValueError(f"Cannot align quantile output {arr.shape} with Q={Q}")


def quantiles_to_BHQ(q_out: Any, q_levels: List[float]) -> np.ndarray:
    Q = len(q_levels)

    if isinstance(q_out, list):
        hq_list = [_to_HQ(x, q_levels) for x in q_out]
        return np.stack(hq_list, axis=0)

    arr = np.asarray(q_out)

    while arr.ndim > 3 and arr.shape[1] == 1:
        arr = arr[:, 0, ...]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected batched quantile output shape: {arr.shape}")

    if arr.shape[1] == Q:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[2] == Q:
        return arr

    raise ValueError(f"Cannot align batched quantile output {arr.shape} with Q={Q}")


@dataclass(frozen=True)
class ForecastingConfig:
    batch_size: int = 32
    coherence_gate: bool = True
    coherence_threshold: float = 0.25


class ForecastGenerator:
    def __init__(self, pipeline, debug: DebugConfig):
        self.pipeline = pipeline
        self.debug = debug

    @torch.no_grad()
    def generate_baseline_or_random(
        self,
        test_input: List[dict],
        prediction_length_eval: int,
        prediction_length_request: int,
        batch_size: int,
        cross_learning: bool,
    ) -> List[QuantileForecast]:
        forecast_outputs = []

        for batch in tqdm(batcher(test_input, batch_size=batch_size), desc="Forecasting"):
            context = [torch.tensor(as_1d_float_array(entry["target"])) for entry in batch]
            q_out, _ = self.pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length_request,
                quantile_levels=QUANTILES,
                cross_learning=cross_learning,
                batch_size=len(context),
            )
            q_bhq = quantiles_to_BHQ(q_out, QUANTILES)
            q_bhq = q_bhq[:, :prediction_length_eval, :]
            forecast_outputs.append(q_bhq)

        forecast_outputs = np.concatenate(forecast_outputs, axis=0)

        forecasts: List[QuantileForecast] = []
        for item, ts in zip(forecast_outputs, test_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item.T,
                    forecast_keys=list(map(str, QUANTILES)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts

    @torch.no_grad()
    def generate_semantic_batched(
        self,
        test_input: List[dict],
        prediction_length_eval: int,
        prediction_length_request: int,
        batch_plan: SemanticBatchPlan,
        fcfg: ForecastingConfig,
        run_tag: str = "",
    ) -> List[QuantileForecast]:
        test_entries = list(test_input)
        N = len(test_entries)
        forecasts_out: List[Optional[QuantileForecast]] = [None] * N

        # optional debug dump
        batch_debug_records = []

        for bi, batch_idxs in enumerate(tqdm(batch_plan.ordered_batches, desc="Forecasting (semantic upgraded)")):
            do_cross = True
            coh = None

            if fcfg.coherence_gate and len(batch_idxs) >= 2:
                feats = []
                for i in batch_idxs:
                    iid = safe_str(test_entries[i].get("item_id", f"series_{i}"))
                    feats.append(batch_plan.item_feat_for_gate[iid])
                feats = np.stack(feats, axis=0)
                feats_z = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-8)
                coh = cosine_to_centroid(feats_z)
                if coh < float(fcfg.coherence_threshold):
                    do_cross = False

            if self.debug.enabled and self.debug.dump_batches:
                batch_debug_records.append(
                    {
                        "run_tag": run_tag,
                        "batch_index": bi,
                        "batch_size": len(batch_idxs),
                        "do_cross": do_cross,
                        "coherence": float(coh) if coh is not None else None,
                        "items": [
                            safe_str(test_entries[i].get("item_id", f"series_{i}")) for i in batch_idxs
                        ],
                    }
                )

            context = [torch.tensor(as_1d_float_array(test_entries[i]["target"])) for i in batch_idxs]
            q_out, _ = self.pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length_request,
                quantile_levels=QUANTILES,
                cross_learning=do_cross,
                batch_size=len(context),
            )
            q_bhq = quantiles_to_BHQ(q_out, QUANTILES)
            q_bhq = q_bhq[:, :prediction_length_eval, :]

            for local_k, global_i in enumerate(batch_idxs):
                ts = test_entries[global_i]
                forecast_start_date = ts["start"] + len(ts["target"])
                forecasts_out[global_i] = QuantileForecast(
                    forecast_arrays=q_bhq[local_k].T,
                    forecast_keys=list(map(str, QUANTILES)),
                    start_date=forecast_start_date,
                )

        if self.debug.enabled and self.debug.dump_batches:
            import json
            self.debug.ensure_dir()
            out = self.debug.out_dir / f"batches_{run_tag}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for r in batch_debug_records:
                    f.write(json.dumps(r) + "\n")

        missing = sum(1 for f in forecasts_out if f is None)
        if missing:
            raise RuntimeError(f"Missing {missing} forecasts (bug).")

        return [f for f in forecasts_out if f is not None]