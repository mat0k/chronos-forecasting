# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import ModelOutput
from torch.utils.checkpoint import checkpoint

# Optional FlashAttention (for fast sliding-window local attention)
_flash_attn_func = None
try:
    from flash_attn import flash_attn_func as _flash_attn_func  # flash-attn >= 2
except Exception:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # alternative path
    except Exception:
        _flash_attn_func = None


from .config import Chronos2CoreConfig

SPARSE_DEBUG = {
    "calls": 0,
    "flash_used": 0,
    "fallback_used": 0,
    "ctx_tokens_sum": 0,
    "S_sum": 0,          # NEW: sum of total sequence length S per call
    "future_sum": 0,     # NEW: sum of future_len per call
    "printed_once": 0,   # NEW: one-time sanity print
}

class RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class Chronos2LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# This is how transformers keeps track of LayerNorm classes ¯\_(ツ)_/¯
ALL_LAYERNORM_LAYERS.append(Chronos2LayerNorm)  # type: ignore


class MLP(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()

        assert not config.is_gated_act, "gated activations are unsupported"
        self.mlp: nn.Module = MLP(config)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


@dataclass
class AttentionOutput(ModelOutput):
    hidden_states: torch.Tensor | None = None
    attn_weights: torch.Tensor | None = None


class MHA(nn.Module):
    """Multi-head Attention Layer"""

    def __init__(self, config: Chronos2CoreConfig, use_rope: bool = True):
        super().__init__()
        self.d_model: int = config.d_model
        self.kv_proj_dim: int = config.d_kv
        self.n_heads: int = config.num_heads
        self.dropout: float = config.dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.config = config

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.kv_proj_dim, base=config.rope_theta)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager attention implementation using manual matmul.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len]

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: [batch, n_heads, q_len, kv_len]
        """
        # Compute attention weights (no scaling - this is the original Chronos-2 implementation)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))  # "bnqd,bnkd->bnqk"
        scores += mask
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len] - additive mask (0 for valid, -inf for invalid)

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: None (SDPA doesn't return weights)
        """
        
        
                # SDPA requires attn_mask dtype to be bool or float matching query dtype.
        # In mixed precision, LayerNorm may upcast queries to fp32 while masks stay fp16.
        if mask is not None and mask.dtype != torch.bool and mask.dtype != query_states.dtype:
            if torch.is_floating_point(mask):
                mask = mask.to(dtype=query_states.dtype)
        
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,  # Match eager implementation (no scaling)
        )

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """Multi-head attention forward pass.

        Args:
            hidden_states : Input tensor of shape [batch_size, seq_len, d_model]
            mask : Attention mask tensor of shape [batch_size, num_heads, q_len, kv_len]
            encoder_states : Encoder states for cross-attention. Defaults to None.
            position_ids : Position IDs for RoPE. Defaults to None.
            output_attentions : Whether to return attention weights. Defaults to False.

        Returns:
            AttentionOutput: Contains:
                - hidden_states : Output tensor of shape [batch_size, seq_len, d_model]
                - attn_weights : Attention weights if output_attentions=True
        """
        if self.use_rope:
            assert position_ids is not None, "position_ids must be provided when self.use_rope=True"

        # Force eager attention if output_attentions is True (only eager returns weights)
        attn_implementation = self.config._attn_implementation
        if output_attentions:
            attn_implementation = "eager"

        seq_length = hidden_states.shape[1]

        def shape(states: torch.Tensor) -> torch.Tensor:
            """(batch, seq_len, inner_dim) -> (batch, n_heads, seq_len, kv_proj_dim)"""
            return rearrange(states, "b s (h d) -> b h s d", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            """(batch, n_heads, seq_len, kv_proj_dim) -> (batch, seq_len, inner_dim)"""
            return rearrange(states, "b h s d -> b s (h d)", h=self.n_heads, s=seq_length, d=self.kv_proj_dim)

        # Construct query states
        query_states = shape(self.q(hidden_states))
        is_cross_attention = encoder_states is not None

        # Construct key/value states
        if is_cross_attention:
            key_states = shape(self.k(encoder_states))
            value_states = shape(self.v(encoder_states))
        else:
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
            if self.use_rope:
                cos, sin = self.rope_embed(value_states, position_ids)
                query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attn_implementation == "sdpa":
            attn_output, attn_weights = self._sdpa_attention(query_states, key_states, value_states, mask)
        else:  # eager
            attn_output, attn_weights = self._eager_attention(query_states, key_states, value_states, mask)

        # Project attention output
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        return AttentionOutput(hidden_states=attn_output, attn_weights=attn_weights if output_attentions else None)
def _flash_sliding_window_local_attn_no_scale(
    q: torch.Tensor,  # [B, S, H, Hd]
    k: torch.Tensor,  # [B, S, H, Hd]
    v: torch.Tensor,  # [B, S, H, Hd]
    *,
    radius: int,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """FlashAttention sliding-window local attention with *no scaling* (softmax_scale=1.0).
    Returns: [B, S, H, Hd]
    """
    if _flash_attn_func is None:
        raise RuntimeError(
            "FlashAttention is not available. Install flash-attn and set time_attention_backend='flash'."
        )

    radius = int(radius)
    p = float(dropout_p) if training else 0.0

    out = None
    try:
        out = _flash_attn_func(
            q, k, v,
            dropout_p=p,
            causal=False,
            window_size=(radius, radius),
            softmax_scale=1.0,  # IMPORTANT: match Chronos-2 "no scaling"
        )
    except TypeError:
        try:
            out = _flash_attn_func(q, k, v, p, False, 1.0, (radius, radius))
        except TypeError:
            out = _flash_attn_func(q, k, v, dropout_p=p, causal=False, window_size=(radius, radius))

    if isinstance(out, tuple):
        out = out[0]

    return out


class TimeSelfAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.config = config
        self.self_attention = MHA(config, use_rope=True)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        num_output_patches: int = 1,
        reg_token_index: int | None=None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        """
        Args:
            hidden_states: [B, S, D]
            attention_mask:
              - full mode: additive 4D mask [B, H, Q, K]
              - sparse mode: 2D padding mask [B, S] (1=valid, 0=pad)
            position_ids: [B, S] (for RoPE)
            num_output_patches: number of future query tokens (last tokens in sequence)
            reg_token_index: position of [REG] if present, else None
        """
        normed_hidden_states = self.layer_norm(hidden_states)

        if self.config.time_attention_type == "full": 
            # Original dense behavior (expects additive 4D mask)
            attention_output: AttentionOutput = self.self_attention(
                normed_hidden_states, 
                position_ids=position_ids,
                mask=attention_mask, 
                output_attentions=output_attentions
            )
            out = attention_output.hidden_states
            assert out is not None
            hidden_states = hidden_states + self.dropout(out)
    
            return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)
        if self.config.time_attention_type != "windowed_future_global": 
            raise ValueError(f"Unknown time_attention_type={self.config.time_attention_type!r}")

        # Sparse path: do not allow returning attentions (would force dense tensors)
        if output_attentions:
            raise ValueError("output_attentions=True is not supported for sparse time attention.")

        if attention_mask.ndim != 2:
            raise ValueError(
                f"Sparse time attention expects a 2D padding mask [B, S], got shape {tuple(attention_mask.shape)}"
            )
        # Compute sparse attention output (projected, then output-projected)
        attn_out = self._windowed_future_global_attention(
            hidden_states=normed_hidden_states,
            padding_mask=attention_mask,
            position_ids=position_ids,
            num_output_patches=num_output_patches,
            reg_token_index=reg_token_index,
        )

        hidden_states = hidden_states + self.dropout(attn_out)
        return AttentionOutput(hidden_states=hidden_states, attn_weights=None)

    def _project_qkv_with_rope(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            q, k, v of shape [B, S, H, Hd] with RoPE applied to (q, k).
        """
        B, S, _ = hidden_states.shape
        H = self.self_attention.n_heads
        Hd = self.self_attention.kv_proj_dim
    
        # [B, S, H*Hd] -> [B, S, H, Hd] (view, no copy)
        q = self.self_attention.q(hidden_states).view(B, S, H, Hd)
        k = self.self_attention.k(hidden_states).view(B, S, H, Hd)
        v = self.self_attention.v(hidden_states).view(B, S, H, Hd)
    
        # RoPE module expects [B, H, S, Hd] for the "x" argument
        v_for_rope = v.permute(0, 2, 1, 3)  # [B, H, S, Hd] (view)
        cos, sin = self.self_attention.rope_embed(v_for_rope, position_ids)
    
        # Apply RoPE to [B, S, H, Hd] tensors
        q, k = RoPE.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    
        return q, k, v

    def _global_attention(
        self,
        q: torch.Tensor,          # [B, H, Q, Hd]
        k: torch.Tensor,          # [B, H, K, Hd]
        v: torch.Tensor,          # [B, H, K, Hd]
        key_padding_mask: torch.Tensor,  # [B, K] bool
    ) -> torch.Tensor:
        """
        Global attention with **no scaling** (to match original Chronos-2 eager behavior).
        Returns: [B, H, Q, Hd]
        """
        dtype = q.dtype
        finfo_min = torch.finfo(dtype).min

        # Build additive mask [B, 1, 1, K] with 0 for valid, -inf for invalid
        mask = torch.where(key_padding_mask, torch.zeros((), device=q.device, dtype=dtype), torch.tensor(finfo_min, device=q.device, dtype=dtype))
        mask = mask.view(key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1])

        attn_impl = self.config._attn_implementation
        # In sparse mode we never return weights, so SDPA is safe if enabled.
        if attn_impl == "sdpa":
            return nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.self_attention.dropout if self.training else 0.0,
                scale=1.0,  # IMPORTANT: no scaling
            )

        # Eager fallback
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, Q, K]
        scores = scores + mask
        w = nn.functional.softmax(scores.float(), dim=-1).to(dtype)
        w = nn.functional.dropout(w, p=self.self_attention.dropout, training=self.training)
        return torch.matmul(w, v)
    def _global_attention_bshd(
        self,
        q: torch.Tensor,  # [B, Q, H, Hd]
        k: torch.Tensor,  # [B, K, H, Hd]
        v: torch.Tensor,  # [B, K, H, Hd]
        key_padding_mask: torch.Tensor,  # [B, K] bool
    ) -> torch.Tensor:
        # SDPA expects [B, H, Q, Hd]
        q_h = q.permute(0, 2, 1, 3)
        k_h = k.permute(0, 2, 1, 3)
        v_h = v.permute(0, 2, 1, 3)
        out_h = self._global_attention(q_h, k_h, v_h, key_padding_mask)
        return out_h.permute(0, 2, 1, 3)  # back to [B, Q, H, Hd]
    def _windowed_future_global_attention(
        self,
        *EW,
        hidden_states: torch.Tensor,      # [B, S, D]
        padding_mask: torch.Tensor,       # [B, S] (float/bool)
        position_ids: torch.Tensor,       # [B, S]
        num_output_patches: int,
        reg_token_index: int | None,
    ) -> torch.Tensor:
        """
        Sparse attention:
          - context queries: windowed keys, restricted to context (+REG if present), plus optional global REG key
          - future queries (last num_output_patches): global keys (masked only by padding)
          - optional REG global query: REG attends globally to context (+REG), never to future
        Returns: [B, S, D]
        """
        B, S, D = hidden_states.shape
        if num_output_patches < 1 or num_output_patches > S:
            raise ValueError(f"num_output_patches must be in [1, {S}], got {num_output_patches}")

        future_start = S - num_output_patches
        context_end = future_start  # includes [REG] if it is placed before the future tokens
        SPARSE_DEBUG["calls"] += 1
        SPARSE_DEBUG["ctx_tokens_sum"] += int(context_end)
        # NEW: track total token length and future length
        SPARSE_DEBUG["S_sum"] += int(S)
        SPARSE_DEBUG["future_sum"] += int(S - context_end)
        
        # NEW: print once to sanity-check what the model actually sees
        if SPARSE_DEBUG["printed_once"] == 0:
            print(f"[DEBUG shapes] S={S} context_end={context_end} future_len={S-context_end}", flush=True)
            SPARSE_DEBUG["printed_once"] = 1
        # Key padding mask: True = keep, False = masked
        if padding_mask.dtype == torch.bool:
            key_pad = padding_mask
        else:
            key_pad = padding_mask > 0.0

        # Project q/k/v once for the full sequence
        q, k, v = self._project_qkv_with_rope(hidden_states, position_ids)  # [B, S, H, Hd]
        H = q.shape[2]
        Hd = q.shape[3]

                # --- NEW: build context_out and future_out separately (avoid full [B,H,S,Hd] zeros) ---

        device = hidden_states.device
        dtype = hidden_states.dtype

        # -------------------------
        # 1) Context queries: windowed attention over keys in [0, context_end)
        # -------------------------
        radius = int(getattr(self.config, "time_local_radius", 128))
        chunk_size = int(getattr(self.config, "time_attention_chunk_size", 32))
        radius = max(0, radius)
        chunk_size = max(1, chunk_size)

        backend = getattr(self.config, "time_attention_backend", "torch")
        if backend not in {"torch", "flash"}:
            raise ValueError(f"time_attention_backend must be 'torch' or 'flash', got {backend!r}")
        use_flash_backend = backend == "flash"

        # Default empty context_out (handles context_end == 0 cleanly)
        context_out = q.new_empty((B, context_end, H, Hd))

        if context_end > 0:
            k_ctx = k[:, :context_end, :, :]        # [B, Ctx, H, Hd]
            v_ctx = v[:, :context_end, :, :]
            key_pad_ctx = key_pad[:, :context_end]  # [B, Ctx]

            # Flash backend requirements (keep these checks)
            if use_flash_backend:
                if _flash_attn_func is None:
                    raise RuntimeError("time_attention_backend='flash' but FlashAttention is not installed/available.")
                if not hidden_states.is_cuda:
                    raise RuntimeError("FlashAttention backend requires CUDA tensors.")
                if hidden_states.dtype not in (torch.float16, torch.bfloat16):
                    raise RuntimeError("FlashAttention backend requires fp16/bf16 hidden_states.")

            # IMPORTANT PERFORMANCE NOTE:
            # Avoid .item() checks here (they sync GPU->CPU). We rely on require_full_context=True at dataset build time.
            # If you still want a safety check, enable it via config.time_flash_strict = True (costs a sync).
            strict = bool(getattr(self.config, "time_flash_strict", False))
            if use_flash_backend and strict:
                if not bool(key_pad_ctx.all().item()):
                    raise RuntimeError(
                        "Sparse flash backend requested, but padding exists in the context. "
                        "Build the dataset with require_full_context=True."
                    )

            can_use_flash = use_flash_backend  # rely on dataset correctness

            if can_use_flash:
                q_ctx = q[:, :context_end, :, :]  # [B, Ctx, H, Hd]
                SPARSE_DEBUG["flash_used"] += 1

    
                context_out = _flash_sliding_window_local_attn_no_scale(
                    q_ctx,
                    k_ctx,
                    v_ctx,
                    radius=radius,
                    dropout_p=float(self.self_attention.dropout),
                    training=self.training,
                )
            else:
                SPARSE_DEBUG["fallback_used"] += 1
                offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)

                def _ctx_chunk_attn(
                    q_chunk: torch.Tensor,      # [B, H, C, Hd]
                    idx: torch.Tensor,          # [C, W]
                    valid: torch.Tensor,        # [C, W] bool
                    k_ctx: torch.Tensor,        # [B, H, Ctx, Hd]
                    v_ctx: torch.Tensor,        # [B, H, Ctx, Hd]
                    key_pad_ctx: torch.Tensor,  # [B, Ctx] bool
                ) -> torch.Tensor:
                    k_win = k_ctx[:, :, idx, :]
                    v_win = v_ctx[:, :, idx, :]

                    key_ok = key_pad_ctx[:, idx] & valid[None, :, :]
                    scores = (q_chunk.unsqueeze(-2) * k_win).sum(dim=-1)
                    scores = scores.masked_fill(~key_ok[:, None, :, :], torch.finfo(scores.dtype).min)

                    w = nn.functional.softmax(scores.float(), dim=-1).to(scores.dtype)
                    w = nn.functional.dropout(w, p=self.self_attention.dropout, training=self.training)

                    return (w.unsqueeze(-1) * v_win).sum(dim=-2)

                # Fill context_out chunk-by-chunk (no giant tensor)
                for start in range(0, context_end, chunk_size):
                    end = min(context_end, start + chunk_size)
                    q_pos = torch.arange(start, end, device=device, dtype=torch.long)
                    idx_raw = q_pos[:, None] + offsets[None, :]
                    valid = (idx_raw >= 0) & (idx_raw < context_end)
                    idx = idx_raw.clamp(0, context_end - 1)

                    q_chunk = q[:, :, start:end, :]

                    if self.training and q_chunk.requires_grad:
                        try:
                            out_chunk = checkpoint(
                                _ctx_chunk_attn, q_chunk, idx, valid, k_ctx, v_ctx, key_pad_ctx, use_reentrant=False
                            )
                        except TypeError:
                            out_chunk = checkpoint(_ctx_chunk_attn, q_chunk, idx, valid, k_ctx, v_ctx, key_pad_ctx)
                    else:
                        out_chunk = _ctx_chunk_attn(q_chunk, idx, valid, k_ctx, v_ctx, key_pad_ctx)

                    context_out[:, :, start:end, :] = out_chunk

            # Optional: REG as a global *query* (REG attends to all context keys)
            if bool(getattr(self.config, "time_reg_is_global", False)) and reg_token_index is not None:
                if 0 <= reg_token_index < context_end:
                    q_reg = q[:, reg_token_index : reg_token_index + 1, :, :]  # [B, 1, H, Hd]
                    out_reg = self._global_attention_bshd(q_reg, k_ctx, v_ctx, key_pad_ctx)
                    context_out[:, reg_token_index : reg_token_index + 1, :, :] = out_reg

        # -------------------------
        # 2) Future queries: global attention over all keys [0, S)
        # -------------------------
        q_fut = q[:, future_start:, :, :]  # [B, num_out, H, Hd]
        future_out = self._global_attention_bshd(q_fut, k, v, key_pad)
        
        out = torch.cat([context_out, future_out], dim=1)  # concat on sequence axis
        out_2d = rearrange(out, "b s h d -> b s (h d)")
        out_2d = self.self_attention.o(out_2d)
        return out_2d

        

class TimeCrossAttention(nn.Module):
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        self.cross_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.cross_attention(
            normed_hidden_states,
            mask=attention_mask,
            encoder_states=encoder_states,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask"""

    def __init__(self, config: Chronos2CoreConfig):
        super().__init__()
        # we don't use RoPE here because there's no natural ordering along the batch axis
        self.self_attention = MHA(config, use_rope=False)
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False
    ) -> AttentionOutput:
        # flip time and batch axes because attention operates along dim=-2
        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # flip time and batch axes back to their original position
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


class ResidualBlock(nn.Module):
    """A generic residual block which can be used for input and output embedding layers"""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = Chronos2LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out
