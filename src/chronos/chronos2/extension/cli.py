from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import typer
import yaml
import torch

# Make repo src visible if running from inside the Chronos repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from .utils import make_logger, DebugConfig
from .io_benchmarks import load_benchmark_tasks, normalize_task_dict, TaskConfig
from .evaluator import ChronosEvaluator, EvalConfig

app = typer.Typer(pretty_exceptions_enable=False)
logger = make_logger("ChronosEvalCLI")


@app.command("run")
def run(
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name: fev-bench | gift-eval"),
    config_yaml: Optional[Path] = typer.Option(None, help="YAML list of dataset configs if not using --benchmark"),
    output_csv: Path = typer.Option(Path("./chronos2_results.csv")),
    model_id: str = typer.Option("amazon/chronos-2"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: str = typer.Option("float32"),
    batch_size: int = typer.Option(32),
    semantic_field: Optional[str] = typer.Option(None),
    grouping: str = typer.Option("features"),
    num_clusters: int = typer.Option(50),
    coherence_gate: bool = typer.Option(True),
    coherence_threshold: float = typer.Option(0.25),
    ttest: bool = typer.Option(False),
    kmeans_iters: int = typer.Option(25),
    seed: int = typer.Option(0),

    debug: bool = typer.Option(False, help="Enable debug dumps"),
    debug_out: Path = typer.Option(Path("./debug_out")),
):
    # tasks
    tasks: List[TaskConfig]
    if benchmark:
        tasks = load_benchmark_tasks(benchmark)
    else:
        if config_yaml is None:
            raise typer.BadParameter("Provide either --benchmark or --config-yaml")
        with config_yaml.open("r", encoding="utf-8") as fp:
            obj = yaml.safe_load(fp)
        if not isinstance(obj, list):
            raise typer.BadParameter("config_yaml must contain a list of dataset configs")
        tasks = [normalize_task_dict(x) for x in obj]

    debug_cfg = DebugConfig(enabled=debug, out_dir=debug_out)
    debug_cfg.ensure_dir()

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
        ttest=ttest,
        kmeans_iters=kmeans_iters,
        seed=seed,
    )

    evaluator = ChronosEvaluator(cfg, debug_cfg)
    evaluator.run(tasks, output_csv)


if __name__ == "__main__":
    app()