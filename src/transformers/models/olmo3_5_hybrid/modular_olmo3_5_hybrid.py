# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ...utils.import_utils import is_flash_linear_attention_available

from ..olmo3.configuration_olmo3 import Olmo3Config
from ..olmo3.modeling_olmo3 import (
    Olmo3DecoderLayer,
    Olmo3Model,
    Olmo3ForCausalLM,
    Olmo3PreTrainedModel,
)

from ..qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
    apply_mask_to_padding_states,
)


if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None
    ShortConvolution = None


logger = logging.get_logger(__name__)


class Olmo3_5HybridConfig(Olmo3Config):
    r"""
    This is the configuration class to store the configuration of a [`Olmo3_5HybridModel`]. It is used to instantiate
    an OLMo 3.5 Hybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OLMo 3.5 Hybrid model.

    The OLMo 3.5 Hybrid model combines standard transformer attention layers with GatedDeltaNet linear attention
    layers for improved efficiency while maintaining model quality.

    Configuration objects inherit from [`Olmo3Config`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Olmo3_5Hybrid model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Olmo3_5HybridModel`].
        hidden_size (`int`, *optional*, defaults to 3840):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 30):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 65536):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 100277):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 100257):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window for sliding window attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Can contain `"full_attention"`, `"sliding_attention"`, or
            `"linear_attention"`. Defaults to linear attention for most layers with full attention for every
            4th layer (determined by `fla_hybrid_attention_indices`).
        fla_hybrid_attention_indices (`list[int]`, *optional*):
            List of layer indices that should use full attention instead of linear attention. Defaults to
            every 4th layer (i.e., layers where `i % 4 == 3`). Only used when `layer_types` is not provided.
        linear_num_key_heads (`int`, *optional*):
            Number of key heads for the linear attention layers. Defaults to `num_attention_heads`.
        linear_num_value_heads (`int`, *optional*):
            Number of value heads for the linear attention layers. Defaults to `num_attention_heads`.
        linear_key_head_dim (`int`, *optional*):
            Dimension of each key head in linear attention layers. Defaults to `0.75 * hidden_size / linear_num_key_heads`.
        linear_value_head_dim (`int`, *optional*):
            Dimension of each value head in linear attention layers. Defaults to `2 * linear_key_head_dim`.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
        linear_use_gate (`bool`, *optional*, defaults to `True`):
            Whether to use gating in the linear attention output normalization.
        linear_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
            Whether to allow negative eigenvalues in the GatedDeltaNet recurrence. When `True`, the beta
            parameter is scaled by 2.0 to allow values in range [0, 2] instead of [0, 1].
    ```python
    >>> from transformers import Olmo3_5HybridModel, Olmo3_5HybridConfig

    >>> # Initializing an Olmo3.5 Hybrid style configuration
    >>> configuration = Olmo3_5HybridConfig()

    >>> # Initializing a model from the Olmo3.5 Hybrid style configuration
    >>> model = Olmo3_5HybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo3_5_hybrid"

    def __init__(
        self,
        vocab_size: int | None = 100352,
        hidden_size: int | None = 3840,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 30,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 65536,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 100257,
        tie_word_embeddings: bool | None = False,
        rope_parameters=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        rms_norm_eps: float | None = 1e-06,
        sliding_window: int | None = 4096,
        layer_types: list[str] | None = None,
        fla_hybrid_attention_indices: list[int] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_use_gate: bool = True,
        linear_allow_neg_eigval: bool = True,
        **kwargs,
    ):
        if layer_types is None:
            if fla_hybrid_attention_indices is None:
                fla_hybrid_attention_indices = [i for i in range(int(num_hidden_layers)) if i % 4 == 3]

            layer_types = ["linear_attention"] * int(num_hidden_layers)
            for idx in fla_hybrid_attention_indices:
                if idx < 0 or idx >= int(num_hidden_layers):
                    raise ValueError(
                        f"`fla_hybrid_attention_indices` contains an out-of-range layer index {idx} "
                        f"for num_hidden_layers={num_hidden_layers}."
                    )
                layer_types[idx] = "full_attention"

        if len(layer_types) != int(num_hidden_layers):
            raise ValueError(
                f"`layer_types` must have length num_hidden_layers={num_hidden_layers}, got {len(layer_types)}."
            )

        if "linear_attention" not in layer_types:
            raise ValueError("OLMo3.5 Hybrid expects at least one 'linear_attention' layer.")
        if all(t == "linear_attention" for t in layer_types):
            raise ValueError("OLMo3.5 Hybrid expects at least one attention layer (full or sliding).")

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
            layer_types=layer_types,
            **kwargs,
        )

        self.layer_types = list(layer_types)
        self.fla_hybrid_attention_indices = [
            i for i, t in enumerate(self.layer_types) if t in {"full_attention", "sliding_attention"}
        ]

        if linear_num_key_heads is None:
            linear_num_key_heads = int(num_attention_heads)
        if linear_num_value_heads is None:
            linear_num_value_heads = int(num_attention_heads)
        if linear_key_head_dim is None:
            linear_key_head_dim = int(0.75 * int(hidden_size) / int(linear_num_key_heads))
        if linear_value_head_dim is None:
            linear_value_head_dim = int(2 * int(linear_key_head_dim))

        self.linear_num_key_heads = int(linear_num_key_heads)
        self.linear_num_value_heads = int(linear_num_value_heads)
        self.linear_key_head_dim = int(linear_key_head_dim)
        self.linear_value_head_dim = int(linear_value_head_dim)
        self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
        self.linear_use_gate = bool(linear_use_gate)
        self.linear_allow_neg_eigval = bool(linear_allow_neg_eigval)


class Olmo3_5HybridDynamicCache:
    """
    Cache for hybrid model supporting both attention KV cache and linear attention state.

    Adapted from transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextDynamicCache
    """

    is_compileable = False

    def __init__(self, config: Olmo3_5HybridConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [i for i, t in enumerate(config.layer_types) if t != "linear_attention"]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        self.conv_states_q = [None for _ in range(config.num_hidden_layers)]
        self.conv_states_k = [None for _ in range(config.num_hidden_layers)]
        self.conv_states_v = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.conv_states_q[layer_idx] is not None:
                device = self.conv_states_q[layer_idx].device
                self.conv_states_q[layer_idx] = self.conv_states_q[layer_idx].index_select(0, beam_idx.to(device))
                self.conv_states_k[layer_idx] = self.conv_states_k[layer_idx].index_select(0, beam_idx.to(device))
                self.conv_states_v[layer_idx] = self.conv_states_v[layer_idx].index_select(0, beam_idx.to(device))
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return (kv_length, kv_offset) for mask creation.
        
        For hybrid models:
        - Attention layers use the KV cache length
        - Linear attention layers don't need this (they use recurrent state)
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        return self.conv_states_q[self.last_linear_layer] is not None


class Olmo3_5HybridRMSNormGated(nn.Module):
    """RMSNorm with gating, matching FLA's FusedRMSNormGated."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate)
        return hidden_states


class Olmo3_5HybridRMSNorm(nn.Module):
    """Standard RMSNorm without gating."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)

# Fallback ShortConvolution implementation when FLA is not available.
class Olmo3_5HybridShortConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.hidden_size = hidden_size
    
    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, T, D = x.shape
        W = self.kernel_size[0]
        
        x_conv = x.transpose(1, 2)
        
        # Single token update (decoding mode)
        if cache is not None and T == 1:
            x_with_state = torch.cat([cache, x_conv], dim=-1)

            out = F.conv1d(
                x_with_state,
                self.weight,
                self.bias,
                padding=0,
                groups=D,
            )
            out = F.silu(out)
            
            new_state = x_with_state[:, :, 1:]
            
            return out.transpose(1, 2), new_state
        
        # Multi-token forward (prefill mode)
        else:
            out = super().forward(x_conv)[:, :, :T]
            out = F.silu(out)
            
            if output_final_state:
                if T >= W - 1:
                    new_state = x_conv[:, :, -(W - 1):]
                else:
                    new_state = F.pad(x_conv, (W - 1 - T, 0))
            else:
                new_state = None
            
            return out.transpose(1, 2), new_state


class Olmo3_5HybridGatedDeltaNet(nn.Module):
    def __init__(self, config: Olmo3_5HybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_value_heads
        self.num_kv_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_kv_heads
        self.value_dim = self.head_v_dim * self.num_heads
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.use_gate = config.linear_use_gate
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        
        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.use_fla_conv = ShortConvolution is not None
        
        if self.use_fla_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation='silu',
            )
        else:
            self.q_conv1d = Olmo3_5HybridShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
            )
            self.k_conv1d = Olmo3_5HybridShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
            )
            self.v_conv1d = Olmo3_5HybridShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
            )

        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # Output norm - NOTE: FLA's FusedRMSNormGated uses eps=1e-5 by default
        o_norm_eps = 1e-5
        if self.use_gate:
            if FusedRMSNormGated is not None:
                self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=o_norm_eps)
            else:
                self.o_norm = Olmo3_5HybridRMSNormGated(self.head_v_dim, eps=o_norm_eps)
        else:
            self.o_norm = Olmo3_5HybridRMSNorm(self.head_v_dim, eps=o_norm_eps)

        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_flash_linear_attention_available():
            logger.warning_once(
                "FLA fast path not available. Install flash-linear-attention for better performance. "
                "See: https://github.com/fla-org/flash-linear-attention"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Olmo3_5HybridDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_cache = cache_params is not None
        use_precomputed = use_cache and cache_params.has_previous_state and seq_len == 1

        conv_state_q = cache_params.conv_states_q[self.layer_idx] if cache_params else None
        conv_state_k = cache_params.conv_states_k[self.layer_idx] if cache_params else None
        conv_state_v = cache_params.conv_states_v[self.layer_idx] if cache_params else None
        recurrent_state = cache_params.recurrent_states[self.layer_idx] if cache_params else None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, new_conv_state_q = self.q_conv1d(
            x=q,
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, new_conv_state_k = self.k_conv1d(
            x=k,
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        v, new_conv_state_v = self.v_conv1d(
            x=v,
            cache=conv_state_v,
            output_final_state=use_cache,
        )

        if cache_params is not None:
            cache_params.conv_states_q[self.layer_idx] = new_conv_state_q
            cache_params.conv_states_k[self.layer_idx] = new_conv_state_k
            cache_params.conv_states_v[self.layer_idx] = new_conv_state_v

        q = q.view(batch_size, seq_len, self.num_kv_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        if self.num_heads > self.num_kv_heads:
            expand_ratio = self.num_heads // self.num_kv_heads
            q = q.unsqueeze(3).expand(-1, -1, -1, expand_ratio, -1).reshape(
                batch_size, seq_len, self.num_heads, self.head_k_dim
            )
            k = k.unsqueeze(3).expand(-1, -1, -1, expand_ratio, -1).reshape(
                batch_size, seq_len, self.num_heads, self.head_k_dim
            )

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        if use_precomputed:
            output, new_recurrent_state = self.recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            output, new_recurrent_state = self.chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = new_recurrent_state

        if self.use_gate:
            gate = self.g_proj(hidden_states)
            gate = gate.view(batch_size, seq_len, self.num_heads, self.head_v_dim)
            output = output.reshape(-1, self.head_v_dim)
            gate = gate.reshape(-1, self.head_v_dim)
            output = self.o_norm(output, gate)
            output = output.view(batch_size, seq_len, self.num_heads, self.head_v_dim)
        else:
            output = output.reshape(-1, self.head_v_dim)
            output = self.o_norm(output)
            output = output.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.o_proj(output)

        return output


class Olmo3_5HybridDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: Olmo3_5HybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Olmo3_5HybridGatedDeltaNet(config, layer_idx=layer_idx)
            # For linear attention, we need a PRE-norm (fla_norm)
            # The post_attention_layernorm from parent becomes the fla_norm
            # We rename it conceptually but keep the same weight
            del self.self_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        
        if self.layer_type == "linear_attention":
            # OLMo-core FLABlock: h = x + fla(fla_norm(x))
            # post_attention_layernorm is used as fla_norm (pre-norm)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)  # Norm BEFORE FLA
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states
            
            # MLP: h = h + mlp(mlp_norm(h))
            residual = hidden_states
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # Norm BEFORE MLP
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Standard attention layers: OLMo-core ReorderedNormTransformerBlock
            # h = x + post_attn_norm(attn(x))
            residual = hidden_states
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = self.post_attention_layernorm(hidden_states)  # Norm AFTER attention
            hidden_states = residual + hidden_states

            # MLP: h = h + post_ff_norm(mlp(h))
            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)  # Norm AFTER MLP
            hidden_states = residual + hidden_states
        
        return hidden_states


class Olmo3_5HybridPreTrainedModel(Olmo3PreTrainedModel):
    pass


class Olmo3_5HybridModel(Olmo3Model):
    def __init__(self, config: Olmo3_5HybridConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Olmo3_5HybridDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Olmo3_5HybridDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        causal_mask = create_causal_mask(**mask_kwargs)
        sliding_mask = None
        if any(t == "sliding_attention" for t in self.config.layer_types):
            sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)

        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            if decoder_layer.layer_type == "linear_attention":
                layer_mask = linear_attn_mask
            elif decoder_layer.layer_type == "full_attention":
                layer_mask = causal_mask
            elif decoder_layer.layer_type == "sliding_attention":
                if sliding_mask is None:
                    sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)
                layer_mask = sliding_mask
            else:
                raise ValueError(f"Unknown layer type {decoder_layer.layer_type!r}.")

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask: torch.Tensor | None, cache_position: torch.Tensor):
        linear_attn_mask = attention_mask
        if cache_position.numel() > 0 and (
            cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1))
        ):
            linear_attn_mask = None
        return linear_attn_mask


class Olmo3_5HybridForCausalLM(Olmo3ForCausalLM, GenerationMixin):
    pass


__all__ = [
    "Olmo3_5HybridConfig",
    "Olmo3_5HybridForCausalLM",
    "Olmo3_5HybridModel",
    "Olmo3_5HybridPreTrainedModel",
]