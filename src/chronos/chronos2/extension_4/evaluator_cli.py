#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import typer
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm

# Make repo src visible if running from inside the Chronos repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from chronos import BaseChronosPipeline, Chronos2Pipeline  # noqa: E402

from core_utils import (
    QUANTILES,
    MASE_KEY,
    WQL_KEY,
    as_1d_float_array,
    ceil_to_patch,
    compute_per_series_arrays,
    paired_ttest,
    quantiles_to_BHQ,
    safe_str,
)
from data_tasks import BenchmarkTaskLoader, DatasetManager, TaskConfig
from features_batching import (
    BatchingConfig,
    Bucketizer,
    FeatureExtractor,
    SemanticBatchPlanner,
)

app = typer.Typer(pretty_exceptions_enable=False)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Chronos2 Eval (baseline/cross/semantic-upgraded)")
logger.setLevel(logging.INFO)


@dataclass
class EvalConfig:
    model_id: str = "amazon/chronos-2"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 32

    semantic_field: Optional[str] = None
    grouping: str = "neighbors"
    num_clusters: int = 50

    coherence_gate: bool = True
    coherence_threshold: float = 0.25

    neighbor_top_k: int = 64
    neighbor_threshold: float = 0.20
    max_group_for_bruteforce: int = 5000

    min_batch_fill: float = 0.5
    max_batch_overhead: float = 3.0

    ttest: bool = False
    kmeans_iters: int = 25
    seed: int = 0


class ForecastEngine:
    def __init__(self, pipeline: Chronos2Pipeline):
        self.pipeline = pipeline

    @torch.no_grad()
    def generate_forecasts(
        self,
        test_data_input,
        prediction_length_eval: int,
        prediction_length_request: int,
        batch_size: int,
        cross_learning: bool,
    ) -> List[QuantileForecast]:
        forecast_outputs = []

        for batch in tqdm(batcher(test_data_input, batch_size=batch_size), desc="Forecasting"):
            context = [torch.tensor(as_1d_float_array(entry["target"])) for entry in batch]

            q_out, _ = self.pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length_request,
                quantile_levels=QUANTILES,
                cross_learning=cross_learning,
                batch_size=len(context),
            )

            q_bhq = quantiles_to_BHQ(q_out, QUANTILES)          # (B,Hreq,Q)
            q_bhq = q_bhq[:, :prediction_length_eval, :]        # (B,Heval,Q)
            forecast_outputs.append(q_bhq)

        forecast_outputs = np.concatenate(forecast_outputs, axis=0)  # (N,Heval,Q)

        forecasts: List[QuantileForecast] = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item.T,  # (Q,Heval)
                    forecast_keys=list(map(str, QUANTILES)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts

    @torch.no_grad()
    def generate_forecasts_semantic_batched_upgraded(
        self,
        test_data_input,
        prediction_length_eval: int,
        prediction_length_request: int,
        batch_size: int,
        itemid_to_sem: Dict[str, Optional[str]],
        semantic_field: Optional[str],
        grouping: str,
        num_clusters: int,
        coherence_threshold: float,
        coherence_gate: bool,
        neighbor_top_k: int,
        neighbor_threshold: float,
        max_group_for_bruteforce: int,
        kmeans_iters: int,
        seed: int,
        min_batch_fill: float,
        max_batch_overhead: float,
    ) -> List[QuantileForecast]:
        test_entries = list(test_data_input)
        N = len(test_entries)

        feat = FeatureExtractor()
        bucket = Bucketizer(feat)
        planner = SemanticBatchPlanner(feat, bucket)

        # grouping + features
        item_to_group, item_to_rep, item_feats_z, item_to_len = planner.build_grouping(
            test_entries=test_entries,
            itemid_to_sem=itemid_to_sem,
            semantic_field=semantic_field,
            grouping=grouping,
            num_clusters=num_clusters,
            kmeans_iters=kmeans_iters,
            seed=seed,
        )

        # batches
        bcfg = BatchingConfig(
            batch_size=batch_size,
            neighbor_top_k=neighbor_top_k,
            neighbor_threshold=neighbor_threshold,
            max_group_for_bruteforce=max_group_for_bruteforce,
            min_batch_fill=min_batch_fill,
            max_batch_overhead=max_batch_overhead,
            seed=seed,
        )
        ordered_batches, batching_kind = planner.plan_batches(
            test_entries=test_entries,
            item_to_group=item_to_group,
            item_feats_z=item_feats_z,
            item_to_len=item_to_len,
            grouping=grouping,
            cfg=bcfg,
        )

        # group diagnostics (unique items)
        group_counts: Dict[str, int] = {}
        for iid in item_to_rep.keys():
            g = item_to_group.get(iid, "G:unknown")
            group_counts[g] = group_counts.get(g, 0) + 1
        sizes = np.array(list(group_counts.values()), dtype=np.int32)

        logger.info(
            f"  [group diag] items={len(item_to_rep)}, groups={len(group_counts)}, "
            f"avg_group_items={float(sizes.mean()) if sizes.size else 0.0:.2f}, "
            f"batches={len(ordered_batches)}, "
            f"avg_batch_size={float(np.mean([len(b) for b in ordered_batches])) if ordered_batches else 0.0:.2f}, "
            f"batching={batching_kind}"
        )

        forecasts_out: List[Optional[QuantileForecast]] = [None] * N

        for batch_idxs in tqdm(ordered_batches, desc="Forecasting (semantic upgraded)"):
            do_cross = True
            if coherence_gate and len(batch_idxs) >= 2:
                feats = []
                for i in batch_idxs:
                    iid = safe_str(test_entries[i].get("item_id", f"series_{i}"))
                    feats.append(item_feats_z.get(iid, np.zeros(10, dtype=np.float32)))
                feats = np.stack(feats, axis=0)
                coh = feat.cosine_to_centroid(feats)
                if coh < coherence_threshold:
                    do_cross = False

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

        missing = sum(1 for f in forecasts_out if f is None)
        if missing:
            raise RuntimeError(f"Missing {missing} forecasts (bug).")

        return [f for f in forecasts_out if f is not None]


class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.dataset_mgr = DatasetManager()

        logger.info(f"Loading model: {cfg.model_id}")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            cfg.model_id,
            device_map=cfg.device,
            dtype={"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype],
        )
        self.engine = ForecastEngine(self.pipeline)

        self.patch = int(getattr(self.pipeline, "model_output_patch_size", 1))
        logger.info(f"Model patch size: {self.patch}")

    def run_tasks(self, tasks: List[TaskConfig], output_csv: Path) -> None:
        rows = []
        per_series_store: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

        for tcfg in tasks:
            dataset_name = tcfg.name
            hf_repo = tcfg.hf_repo
            prediction_length_eval = int(tcfg.prediction_length)
            prediction_length_request = ceil_to_patch(prediction_length_eval, self.patch)

            logger.info("=" * 70)
            logger.info(
                f"Task: {dataset_name} (hf_repo={hf_repo}) pred_len={prediction_length_eval} offset={tcfg.offset}"
            )
            logger.info("=" * 70)

            try:
                test_data, _train_list, sem_map = self.dataset_mgr.load_and_split_dataset(
                    tcfg, semantic_field=self.cfg.semantic_field
                )
            except Exception as e:
                logger.exception(f"[skip] {dataset_name}: load/split failed (repr={e!r})")
                continue

            test_input = list(test_data.input)
            test_label = list(test_data.label)

            def eval_mode(mode_name: str, forecasts: List[QuantileForecast]) -> Tuple[float, float]:
                metrics = evaluate_forecasts(
                    forecasts,
                    test_data=test_data,
                    metrics=[MASE(), MeanWeightedSumQuantileLoss(QUANTILES)],
                    batch_size=5000,
                ).reset_index(drop=True).to_dict(orient="records")
                mase = float(metrics[0].get(MASE_KEY, np.nan))
                wql = float(metrics[0].get(WQL_KEY, np.nan))

                rows.append(
                    {"benchmark_task": dataset_name, "hf_repo": hf_repo, "mode": mode_name, "MASE": mase, "WQL": wql}
                )

                mase_s, wql_s = compute_per_series_arrays(
                    test_inputs=test_input,
                    test_labels=test_label,
                    forecasts=forecasts,
                    prediction_length=prediction_length_eval,
                )
                per_series_store[(dataset_name, mode_name)] = (mase_s, wql_s)
                return mase, wql

            # baseline
            logger.info("  [baseline]")
            t0 = time.time()
            f_base = self.engine.generate_forecasts(
                test_input,
                prediction_length_eval=prediction_length_eval,
                prediction_length_request=prediction_length_request,
                batch_size=self.cfg.batch_size,
                cross_learning=False,
            )
            base_mase, base_wql = eval_mode("baseline", f_base)
            logger.info(f"    MASE={base_mase:.4f}, WQL={base_wql:.4f} ({time.time()-t0:.1f}s)")

            # cross-learning random
            logger.info("  [cross_learning_random]")
            t0 = time.time()
            f_cross = self.engine.generate_forecasts(
                test_input,
                prediction_length_eval=prediction_length_eval,
                prediction_length_request=prediction_length_request,
                batch_size=self.cfg.batch_size,
                cross_learning=True,
            )
            cl_mase, cl_wql = eval_mode("cross_learning_random", f_cross)
            logger.info(f"    MASE={cl_mase:.4f}, WQL={cl_wql:.4f} ({time.time()-t0:.1f}s)")

            # semantic upgraded
            logger.info("  [semantic_cross_learning_upgraded]")
            t0 = time.time()
            f_sem = self.engine.generate_forecasts_semantic_batched_upgraded(
                test_input,
                prediction_length_eval=prediction_length_eval,
                prediction_length_request=prediction_length_request,
                batch_size=self.cfg.batch_size,
                itemid_to_sem=sem_map,
                semantic_field=self.cfg.semantic_field,
                grouping=self.cfg.grouping,
                num_clusters=self.cfg.num_clusters,
                coherence_threshold=self.cfg.coherence_threshold,
                coherence_gate=self.cfg.coherence_gate,
                neighbor_top_k=self.cfg.neighbor_top_k,
                neighbor_threshold=self.cfg.neighbor_threshold,
                max_group_for_bruteforce=self.cfg.max_group_for_bruteforce,
                kmeans_iters=self.cfg.kmeans_iters,
                seed=self.cfg.seed,
                min_batch_fill=self.cfg.min_batch_fill,
                max_batch_overhead=self.cfg.max_batch_overhead,
            )
            sem_mase, sem_wql = eval_mode("semantic_cross_learning_upgraded", f_sem)
            logger.info(f"    MASE={sem_mase:.4f}, WQL={sem_wql:.4f} ({time.time()-t0:.1f}s)")

            logger.info(f"  Summary: baseline={base_mase:.4f}, cross_random={cl_mase:.4f}, upgraded={sem_mase:.4f}")

            if self.cfg.ttest:
                base_mase_s, base_wql_s = per_series_store[(dataset_name, "baseline")]
                for mode_name in ["cross_learning_random", "semantic_cross_learning_upgraded"]:
                    m_mase_s, m_wql_s = per_series_store[(dataset_name, mode_name)]
                    t_m, p_m, d_m, n_m = paired_ttest(base_mase_s, m_mase_s)
                    t_w, p_w, d_w, n_w = paired_ttest(base_wql_s, m_wql_s)
                    logger.info(
                        f"  [t-test per-series] {dataset_name} baseline vs {mode_name} "
                        f"MASE: t={t_m:.4f}, p={p_m}, d={d_m:.4f}, n={n_m} | "
                        f"WQL: t={t_w:.4f}, p={p_w}, d={d_w:.4f}, n={n_w}"
                    )

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logger.info(f"\nSaved results to {output_csv}")

        if df.empty:
            logger.warning("No results to summarize.")
            return

        pivot_mase = df.pivot_table(index="benchmark_task", columns="mode", values="MASE", aggfunc="mean")
        pivot_wql = df.pivot_table(index="benchmark_task", columns="mode", values="WQL", aggfunc="mean")

        logger.info("\nMASE per task (pivot):\n" + pivot_mase.to_string())
        logger.info("\nWQL per task (pivot):\n" + pivot_wql.to_string())

        if self.cfg.ttest:
            logger.info("\n" + "=" * 70)
            logger.info("Paired t-tests baseline vs others")
            logger.info(" - per-task aggregates (weak signal)")
            logger.info(" - pooled per-series (stronger, recommended)")
            logger.info("=" * 70)

            def ttest_report_task_agg(pivot: pd.DataFrame, metric_name: str):
                if "baseline" not in pivot.columns:
                    return
                base = pivot["baseline"].to_numpy()
                for mode_name in pivot.columns:
                    if mode_name == "baseline":
                        continue
                    other = pivot[mode_name].to_numpy()
                    t, p, d, n = paired_ttest(base, other)
                    logger.info(
                        f"[t-test task-agg] {metric_name}: baseline vs {mode_name}: "
                        f"t={t:.4f}, p={p}, d={d:.4f}, n={n}"
                    )

            ttest_report_task_agg(pivot_mase, "MASE")
            ttest_report_task_agg(pivot_wql, "WQL")

            all_tasks = sorted(set(df["benchmark_task"].tolist()))
            for metric_name in ["MASE", "WQL"]:
                for mode_name in ["cross_learning_random", "semantic_cross_learning_upgraded"]:
                    xs = []
                    ys = []
                    for task in all_tasks:
                        if (task, "baseline") not in per_series_store:
                            continue
                        if (task, mode_name) not in per_series_store:
                            continue
                        base_mase_s, base_wql_s = per_series_store[(task, "baseline")]
                        m_mase_s, m_wql_s = per_series_store[(task, mode_name)]
                        if metric_name == "MASE":
                            xs.append(base_mase_s)
                            ys.append(m_mase_s)
                        else:
                            xs.append(base_wql_s)
                            ys.append(m_wql_s)

                    if xs:
                        xcat = np.concatenate(xs, axis=0)
                        ycat = np.concatenate(ys, axis=0)
                        t, p, d, n = paired_ttest(xcat, ycat)
                        logger.info(
                            f"[t-test pooled-series] {metric_name}: baseline vs {mode_name}: "
                            f"t={t:.4f}, p={p}, d={d:.4f}, n={n}"
                        )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name: fev-bench | gift-eval"),
    config_yaml: Optional[Path] = typer.Option(None, "--config-yaml", help="YAML list of dataset configs"),
    output_csv: Path = typer.Option(Path("./chronos2_results.csv"), help="Output CSV path"),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: str = typer.Option("float32"),
    batch_size: int = typer.Option(32),

    semantic_field: Optional[str] = typer.Option(None, help="HF column for semantic grouping (optional)"),

    grouping: str = typer.Option("neighbors", help="neighbors | enhanced_bucket | features_kmeans | bucket"),
    num_clusters: int = typer.Option(50),
    coherence_gate: bool = typer.Option(True),
    coherence_threshold: float = typer.Option(0.25),

    neighbor_top_k: int = typer.Option(64),
    neighbor_threshold: float = typer.Option(0.20),
    max_group_for_bruteforce: int = typer.Option(5000),

    min_batch_fill: float = typer.Option(0.5),
    max_batch_overhead: float = typer.Option(3.0),

    ttest: bool = typer.Option(False),
    kmeans_iters: int = typer.Option(25),
    seed: int = typer.Option(0),
):
    if ctx.invoked_subcommand is not None:
        return

    loader = BenchmarkTaskLoader()
    if benchmark:
        tasks = loader.load(benchmark)
    else:
        if config_yaml is None:
            raise typer.BadParameter("Provide either --benchmark or --config-yaml")
        tasks = loader.load_from_yaml(config_yaml)

    cfg = EvalConfig(
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        semantic_field=semantic_field,
        grouping=grouping,
        num_clusters=num_clusters,
        coherence_gate=coherence_gate,
        coherence_threshold=coherence_threshold,
        neighbor_top_k=neighbor_top_k,
        neighbor_threshold=neighbor_threshold,
        max_group_for_bruteforce=max_group_for_bruteforce,
        min_batch_fill=min_batch_fill,
        max_batch_overhead=max_batch_overhead,
        ttest=ttest,
        kmeans_iters=kmeans_iters,
        seed=seed,
    )

    Evaluator(cfg).run_tasks(tasks, output_csv=output_csv)


@app.command("evaluate")
def evaluate_cmd(
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name: fev-bench | gift-eval"),
    config_yaml: Optional[Path] = typer.Option(None, "--config-yaml", help="YAML list of dataset configs"),
    output_csv: Path = typer.Option(Path("./chronos2_results.csv")),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: str = typer.Option("float32"),
    batch_size: int = typer.Option(32),

    semantic_field: Optional[str] = typer.Option(None),

    grouping: str = typer.Option("neighbors"),
    num_clusters: int = typer.Option(50),
    coherence_gate: bool = typer.Option(True),
    coherence_threshold: float = typer.Option(0.25),

    neighbor_top_k: int = typer.Option(64),
    neighbor_threshold: float = typer.Option(0.20),
    max_group_for_bruteforce: int = typer.Option(5000),

    min_batch_fill: float = typer.Option(0.5),
    max_batch_overhead: float = typer.Option(3.0),

    ttest: bool = typer.Option(False),
    kmeans_iters: int = typer.Option(25),
    seed: int = typer.Option(0),
):
    loader = BenchmarkTaskLoader()
    if benchmark:
        tasks = loader.load(benchmark)
    else:
        if config_yaml is None:
            raise typer.BadParameter("Provide either --benchmark or --config-yaml")
        tasks = loader.load_from_yaml(config_yaml)

    cfg = EvalConfig(
        model_id=model_id,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        semantic_field=semantic_field,
        grouping=grouping,
        num_clusters=num_clusters,
        coherence_gate=coherence_gate,
        coherence_threshold=coherence_threshold,
        neighbor_top_k=neighbor_top_k,
        neighbor_threshold=neighbor_threshold,
        max_group_for_bruteforce=max_group_for_bruteforce,
        min_batch_fill=min_batch_fill,
        max_batch_overhead=max_batch_overhead,
        ttest=ttest,
        kmeans_iters=kmeans_iters,
        seed=seed,
    )

    Evaluator(cfg).run_tasks(tasks, output_csv=output_csv)


if __name__ == "__main__":
    app()