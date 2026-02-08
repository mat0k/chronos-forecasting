from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable, Any

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline  # type: ignore

from .utils import make_logger, ceil_to_patch, set_all_seeds, DebugConfig
from .io_benchmarks import TaskConfig
from .io_datasets import load_and_split_dataset
from .batching import SemanticBatcher, SemanticBatchingConfig
from .forecasting import ForecastGenerator, ForecastingConfig
from .metrics import MetricsComputer, paired_ttest


@dataclass(frozen=True)
class EvalConfig:
    model_id: str = "amazon/chronos-2"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 32

    semantic_field: Optional[str] = None

    grouping: str = "features"
    num_clusters: int = 50
    kmeans_iters: int = 25
    seed: int = 0

    coherence_gate: bool = True
    coherence_threshold: float = 0.25

    ttest: bool = False


@dataclass(frozen=True)
class _FilteredTestData:
    """
    Minimal wrapper compatible with gluonts.model.evaluation.evaluate_forecasts,
    which expects an object with `.input` and `.label` iterables.
    """
    input: Iterable[dict]
    label: Iterable[dict]


class ChronosEvaluator:
    def __init__(self, cfg: EvalConfig, debug: DebugConfig):
        self.cfg = cfg
        self.debug = debug
        self.logger = make_logger("ChronosEvalOOP")
        set_all_seeds(cfg.seed)
        try:
            torch.manual_seed(cfg.seed)
        except Exception:
            pass

        self.pipeline = self._load_model()
        self.metrics = MetricsComputer()

        self.batcher = SemanticBatcher(
            SemanticBatchingConfig(
                grouping=cfg.grouping,
                num_clusters=cfg.num_clusters,
                coherence_gate=cfg.coherence_gate,
                coherence_threshold=cfg.coherence_threshold,
                kmeans_iters=cfg.kmeans_iters,
                seed=cfg.seed,
            ),
            debug=debug,
        )
        self.forecaster = ForecastGenerator(self.pipeline, debug=debug)

    def _load_model(self):
        self.logger.info(f"Loading model: {self.cfg.model_id}")
        pipe = BaseChronosPipeline.from_pretrained(
            self.cfg.model_id,
            device_map=self.cfg.device,
            dtype={
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.cfg.dtype],
        )
        return pipe

    @staticmethod
    def _to_1d_np(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return arr.reshape(-1)

    def _filter_aligned_instances(
        self,
        test_data: Any,
        pred_eval: int,
        dataset_name: str,
    ) -> tuple[List[dict], _FilteredTestData]:
        """
        Filters test instances in a way that keeps INPUT and LABEL aligned.
        This prevents:
          - empty contexts (input target length == 0)
          - ragged/short labels (label target not exactly pred_eval)
        which otherwise crash GluonTS evaluation (np.stack).
        """
        # IMPORTANT: rebuild lists from test_data to guarantee alignment
        inputs_all = list(test_data.input)
        labels_all = list(test_data.label)

        if len(inputs_all) != len(labels_all):
            # This should not happen, but if it does: safest is to truncate.
            n = min(len(inputs_all), len(labels_all))
            inputs_all = inputs_all[:n]
            labels_all = labels_all[:n]

        keep_idx: List[int] = []
        bad_empty_ctx = 0
        bad_label_shape = 0

        for i, (inp, lab) in enumerate(zip(inputs_all, labels_all)):
            y_in = self._to_1d_np(inp.get("target", []))
            y_lab = self._to_1d_np(lab.get("target", []))

            if y_in.size == 0:
                bad_empty_ctx += 1
                continue

            # GluonTS evaluation expects each label target to be same shape in a batch
            if y_lab.size != pred_eval:
                bad_label_shape += 1
                continue

            keep_idx.append(i)

        dropped = len(inputs_all) - len(keep_idx)
        if dropped > 0:
            self.logger.warning(
                f"  [filter] {dataset_name}: kept={len(keep_idx)}/{len(inputs_all)} "
                f"(dropped={dropped}, empty_ctx={bad_empty_ctx}, bad_label_len={bad_label_shape})"
            )

        filtered_inputs = [inputs_all[i] for i in keep_idx]
        filtered_labels = [labels_all[i] for i in keep_idx]
        filtered_test_data = _FilteredTestData(input=filtered_inputs, label=filtered_labels)

        return filtered_inputs, filtered_test_data

    def run(self, tasks: List[TaskConfig], output_csv: Path) -> pd.DataFrame:
        run_id = uuid.uuid4().hex[:8]

        patch = int(getattr(self.pipeline, "model_output_patch_size", 1))
        self.logger.info(f"Model patch size: {patch}")
        self.logger.info(f"Run id: {run_id}")

        rows = []

        for tcfg in tasks:
            dataset_name = tcfg.name
            hf_repo = tcfg.hf_repo
            pred_eval = int(tcfg.prediction_length)
            pred_req = ceil_to_patch(pred_eval, patch)

            self.logger.info("=" * 70)
            self.logger.info(f"Task: {dataset_name} (hf_repo={hf_repo}) pred_len={pred_eval} offset={tcfg.offset}")
            self.logger.info("=" * 70)

            # load
            try:
                bundle = load_and_split_dataset(tcfg, semantic_field=self.cfg.semantic_field)
            except Exception as e:
                self.logger.error(f"[skip] {dataset_name}: load/split failed: {e}")
                continue

            # ✅ FIX: filter while keeping input/label aligned
            test_data = bundle.test_data
            test_input, test_data_filtered = self._filter_aligned_instances(
                test_data=test_data,
                pred_eval=pred_eval,
                dataset_name=dataset_name,
            )

            if len(test_input) == 0:
                self.logger.error(f"[skip] {dataset_name}: no valid test instances after filtering.")
                continue

            # baseline
            self.logger.info("  [baseline]")
            t0 = time.time()
            f_base = self.forecaster.generate_baseline_or_random(
                test_input, pred_eval, pred_req, self.cfg.batch_size, cross_learning=False
            )
            base_res = self.metrics.compute(f_base, test_data_filtered)
            rows.append(
                {"benchmark_task": dataset_name, "hf_repo": hf_repo, "mode": "baseline", "MASE": base_res.mase, "WQL": base_res.wql}
            )
            self.logger.info(f"    MASE={base_res.mase:.4f}, WQL={base_res.wql:.4f} ({time.time()-t0:.1f}s)")

            # cross-learning random
            self.logger.info("  [cross_learning_random]")
            t0 = time.time()
            f_cross = self.forecaster.generate_baseline_or_random(
                test_input, pred_eval, pred_req, self.cfg.batch_size, cross_learning=True
            )
            cross_res = self.metrics.compute(f_cross, test_data_filtered)
            rows.append(
                {"benchmark_task": dataset_name, "hf_repo": hf_repo, "mode": "cross_learning_random", "MASE": cross_res.mase, "WQL": cross_res.wql}
            )
            self.logger.info(f"    MASE={cross_res.mase:.4f}, WQL={cross_res.wql:.4f} ({time.time()-t0:.1f}s)")

            # semantic plan
            self.logger.info("  [semantic_cross_learning_upgraded]")
            plan = self.batcher.plan(
                test_entries=test_input,
                itemid_to_sem=bundle.sem_map,
                semantic_field=self.cfg.semantic_field,
                batch_size=self.cfg.batch_size,
            )
            self.logger.info(
                f"    [group diag] groups={len(set(plan.item_to_group.values()))}, batches={len(plan.ordered_batches)}"
            )

            # semantic forecasts
            t0 = time.time()
            fcfg = ForecastingConfig(
                batch_size=self.cfg.batch_size,
                coherence_gate=self.cfg.coherence_gate,
                coherence_threshold=self.cfg.coherence_threshold,
            )
            f_sem = self.forecaster.generate_semantic_batched(
                test_input, pred_eval, pred_req, plan, fcfg, run_tag=f"{run_id}_{dataset_name}"
            )
            sem_res = self.metrics.compute(f_sem, test_data_filtered)
            rows.append(
                {"benchmark_task": dataset_name, "hf_repo": hf_repo, "mode": "semantic_cross_learning_upgraded", "MASE": sem_res.mase, "WQL": sem_res.wql}
            )
            self.logger.info(f"    MASE={sem_res.mase:.4f}, WQL={sem_res.wql:.4f} ({time.time()-t0:.1f}s)")

            self.logger.info(
                f"  Summary: baseline={base_res.mase:.4f}, cross_random={cross_res.mase:.4f}, upgraded={sem_res.mase:.4f}"
            )

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        self.logger.info(f"\nSaved results to {output_csv}")

        if self.cfg.ttest and not df.empty:
            self._report_ttests(df)

        return df

    def _report_ttests(self, df: pd.DataFrame) -> None:
        pivot_mase = df.pivot_table(index="benchmark_task", columns="mode", values="MASE", aggfunc="mean")
        pivot_wql = df.pivot_table(index="benchmark_task", columns="mode", values="WQL", aggfunc="mean")

        self.logger.info("\nMASE per task (pivot):\n" + pivot_mase.to_string())
        self.logger.info("\nWQL per task (pivot):\n" + pivot_wql.to_string())

        if "baseline" not in pivot_mase.columns:
            return

        def _ttest_report(pivot: pd.DataFrame, metric_name: str):
            base = pivot["baseline"].to_numpy()
            for mode in pivot.columns:
                if mode == "baseline":
                    continue
                other = pivot[mode].to_numpy()
                t, p = paired_ttest(base, other)
                self.logger.info(f"[t-test paired] {metric_name}: baseline vs {mode}: t={t:.4f}, p={p}")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("Paired t-tests across tasks (baseline vs others)")
        self.logger.info("=" * 70)
        _ttest_report(pivot_mase, "MASE")
        _ttest_report(pivot_wql, "WQL")