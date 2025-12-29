"""
Series Memory Bank for Retrieval-Augmented Time Series Forecasting (v2)

Fixed version with:
- NaN-safe softmax handling
- ID-based exclusion (not similarity-based)
- Masked max for confidence
- Parameter-free fusion option (works without training)
- Better pooling options (last-token, attention pooling)
- Diagnostic logging
- GPU-safe retrieval (no CPU/CUDA matmul mismatch)
- Cached device copies of memory tensors for fast retrieval on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Literal
from dataclasses import dataclass


@dataclass
class MemoryBankConfig:
    """Configuration for the Series Memory Bank."""
    d_model: int = 512
    max_memory_size: int = 10000
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_gating: bool = True
    gate_temperature: float = 5.0  # Higher = sharper gating
    normalize_memories: bool = True
    # New options
    use_learned_projections: bool = False  # False = parameter-free fusion
    pooling_method: Literal["mean", "last", "attention"] = "last"
    fusion_method: Literal["additive", "film"] = "additive"


class SeriesMemoryBank(nn.Module):
    """
    Memory bank with KEY/VALUE split:
    - Keys: L2-normalized for similarity search
    - Values: Raw (unnormalized) for fusion

    This ensures retrieved vectors have meaningful magnitude differences.
    """

    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_size = config.max_memory_size
        self.top_k = config.top_k
        self.threshold = config.similarity_threshold

        # KEY storage (normalized for similarity search)
        self.register_buffer(
            "memory_keys",
            torch.zeros(config.max_memory_size, config.d_model),
            persistent=False,
        )
        # VALUE storage (raw, unnormalized for fusion)
        self.register_buffer(
            "memory_values",
            torch.zeros(config.max_memory_size, config.d_model),
            persistent=False,
        )
        self.register_buffer(
            "memory_count",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        # Store series IDs for exclusion
        self.register_buffer(
            "memory_ids",
            torch.full((config.max_memory_size,), -1, dtype=torch.long),
            persistent=False,
        )

        self.memory_metadata: List[dict] = []

        # Device cache (avoid CPU->GPU copy every retrieve() call)
        self._cache_device = None
        self._cache_count = -1
        self._keys_cache = None
        self._values_cache = None
        self._ids_cache = None

    def reset(self):
        """Clear the memory bank."""
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_count.zero_()
        self.memory_ids.fill_(-1)
        self.memory_metadata.clear()

        # Invalidate cache
        self._cache_device = None
        self._cache_count = -1
        self._keys_cache = None
        self._values_cache = None
        self._ids_cache = None

    def add_memories(
        self,
        series_representations: torch.Tensor,
        series_ids: Optional[torch.Tensor] = None,
        metadata: Optional[List[dict]] = None,
    ):
        """
        Add series representations with KEY/VALUE split.

        Args:
            series_representations: (num_series, d_model) RAW representations
            series_ids: (num_series,) unique identifiers for each series
            metadata: Optional list of dicts
        """
        num_series = series_representations.size(0)
        device = series_representations.device

        # Generate IDs if not provided
        if series_ids is None:
            if self.memory_count.item() > 0:
                current_max = self.memory_ids[: self.memory_count.item()].max().item()
            else:
                current_max = -1
            series_ids = torch.arange(
                current_max + 1, current_max + 1 + num_series, device=device
            )
        else:
            # Make sure IDs are on same device as series_representations (before writing to buffers)
            if series_ids.device != device:
                series_ids = series_ids.to(device, non_blocking=True)

        # Keys: normalized for similarity search
        # Values: raw for fusion (preserves magnitude differences)
        keys = F.normalize(series_representations, p=2, dim=-1)
        values = series_representations  # Raw, unnormalized

        current_count = self.memory_count.item()
        available_space = self.max_size - current_count

        # NOTE: buffers live on CPU by default unless you move the module.
        # We write into buffers directly; later retrieve() will move cached slices to GPU if needed.
        if available_space >= num_series:
            self.memory_keys[current_count : current_count + num_series] = keys.detach().to(
                self.memory_keys.device
            )
            self.memory_values[current_count : current_count + num_series] = values.detach().to(
                self.memory_values.device
            )
            self.memory_ids[current_count : current_count + num_series] = series_ids.detach().to(
                self.memory_ids.device
            )
            self.memory_count += num_series
        else:
            if num_series >= self.max_size:
                self.memory_keys[:] = keys[-self.max_size :].detach().to(self.memory_keys.device)
                self.memory_values[:] = values[-self.max_size :].detach().to(self.memory_values.device)
                self.memory_ids[:] = series_ids[-self.max_size :].detach().to(self.memory_ids.device)
                self.memory_count.fill_(self.max_size)
            else:
                shift_amount = num_series - available_space
                self.memory_keys[:-shift_amount] = self.memory_keys[shift_amount:].clone()
                self.memory_values[:-shift_amount] = self.memory_values[shift_amount:].clone()
                self.memory_ids[:-shift_amount] = self.memory_ids[shift_amount:].clone()

                self.memory_keys[-num_series:] = keys.detach().to(self.memory_keys.device)
                self.memory_values[-num_series:] = values.detach().to(self.memory_values.device)
                self.memory_ids[-num_series:] = series_ids.detach().to(self.memory_ids.device)
                self.memory_count.fill_(self.max_size)

        if metadata:
            self.memory_metadata.extend(metadata)
            if len(self.memory_metadata) > self.max_size:
                self.memory_metadata = self.memory_metadata[-self.max_size :]

        # memory updated -> invalidate device cache
        self._cache_device = None
        self._cache_count = -1
        self._keys_cache = None
        self._values_cache = None
        self._ids_cache = None

    def retrieve(
        self,
        query_representations: torch.Tensor,
        query_ids: Optional[torch.Tensor] = None,
        exclude_ids: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve top-K similar memories with ID-based exclusion.

        Args:
            query_representations: (batch_size, d_model)
            query_ids: (batch_size,) IDs to exclude from retrieval
            exclude_ids: If True, exclude memories with matching IDs

        Returns:
            retrieved_memories: (batch_size, top_k, d_model)
            similarities: (batch_size, top_k)
            mask: (batch_size, top_k) boolean mask
            retrieved_ids: (batch_size, top_k) IDs of retrieved memories
        """
        batch_size = query_representations.size(0)
        current_count = self.memory_count.item()
        device = query_representations.device

        if current_count == 0:
            return (
                torch.zeros(batch_size, self.top_k, self.d_model, device=device),
                torch.zeros(batch_size, self.top_k, device=device),
                torch.zeros(batch_size, self.top_k, dtype=torch.bool, device=device),
                torch.full((batch_size, self.top_k), -1, dtype=torch.long, device=device),
            )

        # Normalize query for similarity search (keys are already normalized)
        query_keys = F.normalize(query_representations, p=2, dim=-1)

        # Ensure query_ids is on the same device as query representations (for masking)
        if query_ids is not None and query_ids.device != device:
            query_ids = query_ids.to(device, non_blocking=True)

        # Cache active memory tensors on the query device (GPU/CPU) for fast retrieval
        # If memory buffers are on CPU and query on CUDA, this avoids CPU/CUDA matmul mismatch.
        if (
            self._keys_cache is None
            or self._cache_device != device
            or self._cache_count != current_count
        ):
            self._keys_cache = self.memory_keys[:current_count].to(device, non_blocking=True)
            self._values_cache = self.memory_values[:current_count].to(device, non_blocking=True)
            self._ids_cache = self.memory_ids[:current_count].to(device, non_blocking=True)

            self._cache_device = device
            self._cache_count = current_count

        active_keys = self._keys_cache      # (M, d_model) normalized
        active_values = self._values_cache  # (M, d_model) raw
        active_ids = self._ids_cache        # (M,)

        # Compute similarities using normalized keys: (B, M)
        similarities = torch.matmul(query_keys, active_keys.T)

        # ID-based exclusion (reliable)
        if exclude_ids and query_ids is not None:
            # Create exclusion mask: (B, M)
            id_mask = query_ids.unsqueeze(1) == active_ids.unsqueeze(0)
            similarities = similarities.masked_fill(id_mask, float("-inf"))

        # Apply threshold
        valid_mask = similarities >= self.threshold

        # Early exit if no valid similarities for entire batch
        if not valid_mask.any():
            return (
                torch.zeros(batch_size, self.top_k, self.d_model, device=device),
                torch.zeros(batch_size, self.top_k, device=device),
                torch.zeros(batch_size, self.top_k, dtype=torch.bool, device=device),
                torch.full((batch_size, self.top_k), -1, dtype=torch.long, device=device),
            )

        similarities = torch.where(
            valid_mask,
            similarities,
            torch.full_like(similarities, float("-inf")),
        )

        # Get top-K
        k = min(self.top_k, current_count)
        top_sims, top_indices = torch.topk(similarities, k=k, dim=-1)

        # Gather RAW VALUES (not normalized keys) for fusion
        retrieved = active_values[top_indices]   # (B, k, d_model) raw
        retrieved_ids = active_ids[top_indices]  # (B, k)

        # Create valid mask
        mask = top_sims > float("-inf")

        # Pad if needed
        if k < self.top_k:
            pad_size = self.top_k - k
            retrieved = F.pad(retrieved, (0, 0, 0, pad_size), value=0)
            top_sims = F.pad(top_sims, (0, pad_size), value=0)
            mask = F.pad(mask, (0, pad_size), value=False)
            retrieved_ids = F.pad(retrieved_ids, (0, pad_size), value=-1)

        return retrieved, top_sims, mask, retrieved_ids

    def get_stats(self) -> dict:
        return {
            "memory_count": self.memory_count.item(),
            "max_size": self.max_size,
            "fill_ratio": self.memory_count.item() / self.max_size,
        }


class ParameterFreeMemoryFusion(nn.Module):
    """
    Parameter-free memory fusion that works without training.

    Uses similarity weights directly instead of learned projections.
    This guarantees the branch actually changes the representation.
    """

    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        self.temperature = config.gate_temperature
        self.threshold = config.similarity_threshold

    def forward(
        self,
        query: torch.Tensor,
        retrieved_memories: torch.Tensor,
        similarities: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply parameter-free memory fusion.

        Returns:
            output: (batch_size, d_model)
            diagnostics: dict with gate/delta stats
        """
        batch_size = query.size(0)

        # Track which rows have valid retrievals
        valid_rows = mask.any(dim=-1)  # (B,)

        diagnostics = {
            "valid_rows": valid_rows.sum().item(),
            "total_rows": batch_size,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "delta_magnitude": 0.0,
        }

        if not valid_rows.any():
            return query, diagnostics

        # Compute attention weights from similarities (NaN-safe)
        attn_weights = torch.zeros_like(similarities)  # (B, K)

        # For valid rows, compute softmax over valid positions
        scores_valid = similarities[valid_rows]  # (V, K)
        mask_valid = mask[valid_rows]            # (V, K)

        scores_valid = scores_valid.masked_fill(~mask_valid, float("-inf"))
        w_valid = F.softmax(scores_valid / self.temperature, dim=-1)

        # Zero out invalid positions (handles NaN from all -inf)
        w_valid = w_valid.masked_fill(~mask_valid, 0.0)
        w_valid = torch.nan_to_num(w_valid, nan=0.0)

        attn_weights[valid_rows] = w_valid

        # Weighted sum of memories: (B, K, 1) * (B, K, d) -> (B, d)
        memory_vec = (attn_weights.unsqueeze(-1) * retrieved_memories).sum(dim=1)

        # Compute confidence gate from max similarity (masked)
        masked_sims = similarities.masked_fill(~mask, float("-inf"))
        max_sim, _ = masked_sims.max(dim=-1, keepdim=True)  # (B, 1)

        # For rows with no valid retrieval, set max_sim to 0
        max_sim = torch.where(
            valid_rows.unsqueeze(-1),
            max_sim,
            torch.zeros_like(max_sim),
        )

        # Confidence gate: sigmoid scaled by temperature
        confidence = torch.sigmoid(self.temperature * (max_sim - self.threshold))  # (B, 1)

        # Compute delta with NaN hardening
        memory_vec = torch.nan_to_num(memory_vec, nan=0.0, posinf=0.0, neginf=0.0)
        delta = memory_vec - query
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply gated fusion
        output = query + confidence * delta

        # Update diagnostics
        valid_max_sims = max_sim[valid_rows].squeeze(-1)
        diagnostics["gate_mean"] = confidence.mean().item()
        diagnostics["gate_max"] = confidence.max().item()
        diagnostics["delta_magnitude"] = delta.abs().mean().item()
        diagnostics["max_sim_mean"] = valid_max_sims.mean().item() if valid_max_sims.numel() > 0 else 0.0
        diagnostics["max_sim_max"] = valid_max_sims.max().item() if valid_max_sims.numel() > 0 else 0.0

        return output, diagnostics


class LearnedMemoryFusion(nn.Module):
    """
    Learned memory fusion with proper NaN handling.
    """

    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if config.use_gating:
            self.gate_proj = nn.Linear(d_model * 2, 1)

        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5
        self.threshold = config.similarity_threshold
        self.temperature = config.gate_temperature

    def forward(
        self,
        query: torch.Tensor,
        retrieved_memories: torch.Tensor,
        similarities: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """NaN-safe learned memory fusion."""
        batch_size = query.size(0)
        valid_rows = mask.any(dim=-1)  # (B,)

        diagnostics = {
            "valid_rows": valid_rows.sum().item(),
            "total_rows": batch_size,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "delta_magnitude": 0.0,
        }

        if not valid_rows.any():
            return query, diagnostics

        # Project
        Q = self.q_proj(query).unsqueeze(1)        # (B, 1, d)
        K = self.k_proj(retrieved_memories)        # (B, K, d)
        V = self.v_proj(retrieved_memories)        # (B, K, d)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, 1, K)
        attn_scores = attn_scores.squeeze(1)                             # (B, K)

        # NaN-safe softmax (per-row handling)
        attn_weights = torch.zeros_like(attn_scores)

        scores_valid = attn_scores[valid_rows]
        mask_valid = mask[valid_rows]
        scores_valid = scores_valid.masked_fill(~mask_valid, float("-inf"))
        w_valid = F.softmax(scores_valid, dim=-1)
        w_valid = w_valid.masked_fill(~mask_valid, 0.0)
        w_valid = torch.nan_to_num(w_valid, nan=0.0)
        attn_weights[valid_rows] = w_valid

        # Weighted sum
        memory_output = (attn_weights.unsqueeze(-1) * V).sum(dim=1)
        memory_output = self.out_proj(memory_output)

        # Gated fusion with masked max_sim
        if self.config.use_gating:
            masked_sims = similarities.masked_fill(~mask, float("-inf"))
            max_sim = masked_sims.max(dim=-1, keepdim=True)[0]
            max_sim = torch.where(valid_rows.unsqueeze(-1), max_sim, torch.zeros_like(max_sim))

            gate_input = torch.cat([query, memory_output], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))

            confidence = torch.sigmoid(self.temperature * (max_sim - self.threshold))
            gate = gate * confidence

            delta = memory_output
            output = query + gate * delta
        else:
            gate = None
            delta = memory_output
            output = query + delta

        output = self.layer_norm(output)

        diagnostics["gate_mean"] = gate.mean().item() if self.config.use_gating else 1.0
        diagnostics["gate_max"] = gate.max().item() if self.config.use_gating else 1.0
        diagnostics["delta_magnitude"] = delta.abs().mean().item()

        return output, diagnostics


class MemoryAugmentedForecasterV2(nn.Module):
    """
    Fixed memory-augmented forecaster with:
    - Proper batch_size tracking
    - Better pooling options
    - Diagnostic logging
    """

    def __init__(self, config: MemoryBankConfig):
        super().__init__()
        self.memory_bank = SeriesMemoryBank(config)

        # Choose fusion method
        if config.use_learned_projections:
            self.fusion = LearnedMemoryFusion(config)
        else:
            self.fusion = ParameterFreeMemoryFusion(config)

        self.config = config

        # Optional attention pooling
        if config.pooling_method == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, config.d_model))
            self.pool_attn = nn.MultiheadAttention(config.d_model, num_heads=1, batch_first=True)

        self._stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "avg_top_similarity": 0.0,
            "avg_gate": 0.0,
            "avg_delta": 0.0,
        }

        # Track series IDs for proper exclusion
        self._next_series_id = 0

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool hidden states to get series representation.

        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) optional
        """
        if self.config.pooling_method == "last":
            return hidden_states[:, -1, :]

        if self.config.pooling_method == "attention":
            batch_size = hidden_states.size(0)
            query = self.pool_query.expand(batch_size, -1, -1)  # (B, 1, d)
            pooled, _ = self.pool_attn(query, hidden_states, hidden_states)
            return pooled.squeeze(1)

        # mean
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            return sum_hidden / count
        return hidden_states.mean(dim=1)

    def build_memory_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        series_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Build memory with proper ID tracking."""
        batch_size = hidden_states.size(0)

        # Pool to get series representations
        series_reps = self._pool_hidden_states(hidden_states, attention_mask)

        # Generate series IDs if not provided
        if series_ids is None:
            series_ids = torch.arange(
                self._next_series_id,
                self._next_series_id + batch_size,
                device=hidden_states.device,
            )
            self._next_series_id += batch_size

        self.memory_bank.add_memories(series_reps, series_ids)

    def augment_representations(
        self,
        hidden_states: torch.Tensor,
        series_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Augment hidden states with retrieved memories.

        Returns:
            augmented: (batch_size, seq_len, d_model)
            diagnostics: dict with stats
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pool to get query representations
        series_reps = self._pool_hidden_states(hidden_states, attention_mask)

        # Retrieve similar memories (with ID-based exclusion)
        retrieved, similarities, mask, retrieved_ids = self.memory_bank.retrieve(
            series_reps,
            query_ids=series_ids,
            exclude_ids=(series_ids is not None),
        )

        # Compute self-hit-rate (how often we retrieve our own ID)
        self_hit_rate = 0.0
        if series_ids is not None and mask.any():
            self_hits = (retrieved_ids == series_ids.unsqueeze(-1)) & mask
            self_hit_rate = self_hits.any(dim=-1).float().mean().item()

        # Apply fusion
        augmented_series, diagnostics = self.fusion(series_reps, retrieved, similarities, mask)
        diagnostics["self_hit_rate"] = self_hit_rate

        # Update stats
        self._stats["total_queries"] += batch_size
        self._stats["successful_retrievals"] += diagnostics["valid_rows"]
        if diagnostics["valid_rows"] > 0:
            self._stats["avg_gate"] = 0.9 * self._stats["avg_gate"] + 0.1 * diagnostics["gate_mean"]
            self._stats["avg_delta"] = 0.9 * self._stats["avg_delta"] + 0.1 * diagnostics["delta_magnitude"]
            self._stats["self_hit_rate"] = 0.9 * self._stats.get("self_hit_rate", 0.0) + 0.1 * self_hit_rate

            valid_sims = similarities[mask]
            if valid_sims.numel() > 0:
                self._stats["avg_top_similarity"] = (
                    0.9 * self._stats["avg_top_similarity"] + 0.1 * valid_sims.max().item()
                )

        # Broadcast augmentation to all tokens
        if self.config.fusion_method == "film":
            gamma = 1.0 + (augmented_series - series_reps).unsqueeze(1)  # (B, 1, d)
            augmented_hidden = hidden_states * gamma
        else:
            delta = (augmented_series - series_reps).unsqueeze(1)         # (B, 1, d)
            augmented_hidden = hidden_states + delta.expand(-1, seq_len, -1)

        diagnostics["augmented_diff"] = (augmented_hidden - hidden_states).abs().mean().item()
        return augmented_hidden, diagnostics

    def get_stats(self) -> dict:
        stats = self._stats.copy()
        stats.update(self.memory_bank.get_stats())
        if stats["total_queries"] > 0:
            stats["retrieval_rate"] = stats["successful_retrievals"] / stats["total_queries"]
        else:
            stats["retrieval_rate"] = 0.0
        return stats

    def reset_stats(self):
        self._stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "avg_top_similarity": 0.0,
            "avg_gate": 0.0,
            "avg_delta": 0.0,
        }