from __future__ import annotations

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASE_KEY = "MASE[0.5]"
WQL_KEY = "mean_weighted_sum_quantile_loss"

FEV_TASKS_URLS = {
    "fev-bench": "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml",
}

GIFT_HF_REPO = "Salesforce/GiftEval"