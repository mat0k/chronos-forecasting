# Soft Group Masking Extension for Chronos-2

## Overview

This extension modifies the Chronos-2 time series forecasting model to support **soft group masking** -- a mechanism that replaces the original binary (hard) group attention mask with a similarity-weighted soft mask. This allows controlled cross-group information sharing during inference, enabling time series from different groups to attend to each other proportionally to their statistical similarity.

## Background: Group Masking in Chronos-2

Chronos-2 processes batches of time series and uses **group IDs** to control which series can exchange information via attention. In the original model:

- Series with the **same** group ID can attend to each other (mask = 0, full attention).
- Series with **different** group IDs are completely blocked (mask = -inf, zero attention).

This hard boundary prevents any cross-group information flow, even when series in different groups are statistically related.

## What This Extension Does

The soft masking extension replaces the binary group mask with a **continuous attention bias** derived from pairwise similarity between input time series:

- **Same-group pairs**: Unchanged (full attention, bias = 0).
- **Cross-group pairs**: Attention bias is proportional to their similarity, ranging from full suppression (similarity ~ 0) to full attention (similarity ~ 1).

The bias is computed as:

```
soft_group_mask = hard_group_mask + (1 - hard_group_mask) * similarity_matrix
attention_bias  = log(soft_group_mask + eps) * temperature
```

Where `temperature` controls permissiveness: higher values make cross-group attention more permissive.

## Modified Files

### `src/chronos/chronos2/pipeline.py`

- **`compute_input_similarity()`** (new function): Computes a `(batch, batch)` pairwise similarity matrix from raw input context. Supports three metrics:
  - `"correlation"`: Pearson correlation, mapped from [-1, 1] to [0, 1].
  - `"cosine"`: Cosine similarity, mapped from [-1, 1] to [0, 1].
  - `"distance"`: Gaussian kernel on Euclidean distance after normalization.
- **`Chronos2Pipeline.predict()`**: Accepts new keyword arguments `use_soft_group_mask`, `similarity_type`, and `soft_mask_temperature`. These propagate through `_predict_batch` and `_predict_step` (including long-horizon autoregressive unrolling).
- **`Chronos2Pipeline._predict_step()`**: When `use_soft_group_mask=True`, computes the similarity matrix and passes it to the model's forward method.

### `src/chronos/chronos2/model.py`

- **`Chronos2Encoder._construct_soft_group_time_mask()`** (new static method): Constructs the soft group-time attention mask by blending the hard group mask with the similarity matrix, converting to log-scale attention bias, and combining with the time-validity mask.
- **`Chronos2Encoder.forward()`**: Accepts `similarity_matrix` and `soft_mask_temperature`. When a similarity matrix is provided, uses `_construct_soft_group_time_mask`; otherwise falls back to the original hard mask.
- **`Chronos2Model.forward()` and `encode()`**: Accept and propagate `similarity_matrix` and `soft_mask_temperature` to the encoder.

### `src/chronos/chronos2/layers.py`

No changes. The MHA, `TimeSelfAttention`, `GroupSelfAttention`, and `FeedForward` modules remain unmodified. The soft mask operates entirely through the existing additive attention mask mechanism.

## Usage

### Inference with Soft Masking

```python
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Baseline (original hard masking)
predictions_hard = pipeline.predict(inputs, prediction_length=24)

# Soft masking with Pearson correlation similarity
predictions_soft = pipeline.predict(
    inputs,
    prediction_length=24,
    use_soft_group_mask=True,
    similarity_type="correlation",   # "correlation", "cosine", or "distance"
    soft_mask_temperature=5.0,       # higher = more permissive cross-group attention
)
```

### Using `predict_quantiles`

```python
quantiles, mean = pipeline.predict_quantiles(
    inputs,
    prediction_length=24,
    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    use_soft_group_mask=True,
    similarity_type="correlation",
    soft_mask_temperature=5.0,
)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_soft_group_mask` | `bool` | `False` | Enable soft group masking |
| `similarity_type` | `str` | `"correlation"` | Similarity metric: `"correlation"`, `"cosine"`, or `"distance"` |
| `soft_mask_temperature` | `float` | `5.0` | Controls cross-group attention permissiveness. Higher = more permissive |

### Temperature Behavior

| Temperature | Effect |
|---|---|
| ~1.0 | Only very high-similarity cross-group pairs get meaningful attention |
| ~5.0 | Moderate cross-group attention for similar series |
| ~10.0+ | Most cross-group pairs receive some attention weight |

## Evaluation

The benchmark notebook (`extension_run/chron_2_bench.ipynb`) evaluates the extension on the standard Chronos zero-shot benchmark suite. For each dataset:

1. **Baseline**: Forecasts generated with original hard group masking.
2. **Soft masking**: Forecasts generated with correlation-based soft masking at `temperature=5.0`.

### Metrics

- **MASE** (Mean Absolute Scaled Error): Scale-normalized point forecast accuracy.
- **WQL** (Weighted Quantile Loss): Probabilistic forecast quality across quantile levels.

### Statistical Tests

- Paired t-test across datasets for significance.
- Cohen's d for effect size.

## Design Decisions

1. **No learnable parameters**: The extension is purely inference-time. The similarity matrix is computed from the raw input context, and the temperature is a fixed hyperparameter. This avoids the need for fine-tuning and allows direct comparison against the pretrained baseline.

2. **Log-scale mapping**: The similarity-to-bias conversion uses `log(similarity) * temperature` rather than a linear mapping. This provides a smooth transition: near-zero similarity maps to large negative bias (near-complete suppression), while high similarity maps to near-zero bias (near-full attention).

3. **Only group attention is affected**: Time self-attention operates along the time axis within each series and is unaffected. The soft mask only modifies the group self-attention mask, which controls cross-series interaction along the batch axis.

4. **Similarity computed on raw context**: The pairwise similarity is calculated before instance normalization and patching to capture the true statistical relationship between the original time series.

## Limitations

- **Batch-dependent results**: Similarity is computed per-batch, so predictions for a given series may change depending on what other series are in the same batch.
- **O(batch_size^2) similarity computation**: Pairwise similarity scales quadratically with batch size, though this is typically negligible compared to model inference time.
- **Temperature is not adaptive**: A single global temperature applies to all cross-group pairs regardless of dataset characteristics.
- **No training signal**: Since there are no learnable parameters, the extension cannot adapt the similarity weighting to the specific forecasting objective.

## Research Question

> Can controlled cross-group information sharing, weighted by input similarity, improve zero-shot forecasting accuracy over the original hard group masking?



## Results and Future Directions

The evaluation of soft group masking against the baseline hard masking on the Chronos Benchmark II datasets revealed modest numerical improvements in metrics such as MASE (Mean Absolute Scaled Error) and WQL (Weighted Quantile Loss). However, paired t-tests indicated that these differences were not statistically significant (e.g., MASE: $p=0.211$, WQL: $p=0.765$). This suggests that the hypothesis of soft masking providing meaningful improvements over hard masking is not supported under the current setup.

Further analysis stratified datasets by their expected cross-series dependencies. For datasets with strong correlations (e.g., ETTh1, Weather), soft masking showed a directional trend toward significance (paired t-test: $p=0.124$), though it remained above the significance threshold. Weakly correlated datasets (e.g., Walmart time series) exhibited no such trend ($p=0.266$). These results suggest that while soft masking may offer marginal benefits in highly correlated settings, the effect size is too small to be reliable in its current form.

### Key Insights:
1. **Training-Inference Mismatch**: The model was trained exclusively with hard masking, leading to a distribution shift when soft masking is applied only at inference.
2. **Sufficiency of Learned Representations**: Chronos-2's pretrained representations likely already capture relevant cross-series dependencies, leaving limited room for similarity-based refinement.
3. **Similarity Metric Limitations**: The raw Pearson correlation used for similarity computation may not align with the latent structures the model relies on for cross-learning.

### Conclusion:
This extension should be viewed as a study rather than an improvement. While the results highlight the potential for controlled cross-group information sharing, they also underscore the need for further exploration. Future work could investigate adaptive temperature scaling, alternative similarity metrics, or fine-tuning the model to align training and inference conditions.

