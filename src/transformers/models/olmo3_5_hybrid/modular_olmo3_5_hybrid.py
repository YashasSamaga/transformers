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

from typing import Optional

import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs

from ..olmo3.configuration_olmo3 import Olmo3Config
from ..olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3MLP,
    Olmo3Model,
    Olmo3ForCausalLM,
    Olmo3PreTrainedModel,
    Olmo3RMSNorm,
    Olmo3RotaryEmbedding,
)
from ..qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache, Qwen3NextGatedDeltaNet


class Olmo3_5HybridConfig(Olmo3Config):
    r"""
    OLMo3.5 Hybrid configuration.

    This configuration extends :class:`~transformers.Olmo3Config` with parameters
    for the Gated DeltaNet (linear attention) layers and with a convenient way to
    specify which layers use full attention vs. linear attention.

    The hybrid layout can be specified in two equivalent ways:
    - Provide `layer_types` directly (a list of length `num_hidden_layers` with values
      in {"linear_attention", "full_attention", "sliding_attention"}).
    - Provide `fla_hybrid_attention_indices` (indices of layers that use attention).
      All remaining layers are set to "linear_attention".

    Notes:
    - By default, if neither `layer_types` nor `fla_hybrid_attention_indices` are
      provided, the config uses the common 3:1 hybrid pattern: every 4th layer
      (i % 4 == 3) is "full_attention", and the others are "linear_attention".
    """

    model_type = "olmo3_5_hybrid"


    def __init__(
        self,
        vocab_size: int | None = 100278,
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
        # --- Hybrid layout helpers ---
        fla_hybrid_attention_indices: list[int] | None = None,
        # --- Linear (Gated DeltaNet) parameters ---
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_use_gate: bool = True,
        linear_allow_neg_eigval: bool = True,
        # --- Qwen3Next compatibility (needed by its GatedDeltaNet) ---
        dtype=None,
        **kwargs,
    ):
        if layer_types is None:
            if fla_hybrid_attention_indices is None:
                # Default: every 4th layer is attention, others are linear
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

        # Important: we intentionally do NOT pass `layer_types` to the parent config,
        # because Olmo3Config may validate that list against its attention-only values.
        # We set `self.layer_types` ourselves below.
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
            dtype=dtype,
            **kwargs,
        )

        self.layer_types = list(layer_types)
        self.fla_hybrid_attention_indices = [
            i for i, t in enumerate(self.layer_types) if t in {"full_attention", "sliding_attention"}
        ]

        # ---------- Linear (Gated DeltaNet) hyperparams ----------
        # Defaults mirror the hybrid training script / FLA convention:
        #   num_heads * head_dim = 0.75 * hidden_size
        # and value dim is expanded by 2x vs key dim.
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


class _RMSNormNoGateWrapper(nn.Module):
    """Adapter to give a non-gated RMSNorm the same `(x, gate=None)` signature."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = Olmo3RMSNorm(dim, eps=eps)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(hidden_states)


class Olmo3_5HybridDynamicCache(Qwen3NextDynamicCache):
    """
    Cache capable of storing both:
      - attention KV caches for attention layers
      - convolution + recurrent state for linear (GatedDeltaNet) layers

    Inherits the implementation from Qwen3Next and widens the notion of
    "attention layers" to include both full and sliding attention.
    """

    def __init__(self, config: Olmo3_5HybridConfig):
        super().__init__(config)
        # Qwen3NextDynamicCache only considers "full_attention" layers as transformer layers.
        # Here we treat any non-linear layer type as an attention layer (full or sliding).
        self.transformer_layers = [i for i, t in enumerate(config.layer_types) if t != "linear_attention"]



class Olmo3_5HybridGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """
    Thin wrapper around Qwen3Next's GatedDeltaNet that adds:
      - `linear_use_gate` support (optionally disables the output gate)
      - `linear_allow_neg_eigval` support (scales beta by 2 like the FLA reference)

    The core math and caching are inherited from Qwen3Next.
    """

    def __init__(self, config: Olmo3_5HybridConfig, layer_idx: int):
        # Qwen3NextGatedDeltaNet expects certain "linear_*" fields to exist on the config.
        super().__init__(config=config, layer_idx=layer_idx)

        self.linear_use_gate = bool(getattr(config, "linear_use_gate", True))
        self.linear_allow_neg_eigval = bool(getattr(config, "linear_allow_neg_eigval", False))

        # If the output gate is disabled, replace the gated norm with a non-gated RMSNorm wrapper.
        if not self.linear_use_gate:
            self.norm = _RMSNormNoGateWrapper(self.head_v_dim, eps=self.layer_norm_epsilon)

        # If negative eigenvalues are enabled (see FLA reference), multiply beta by 2.
        # https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
        if self.linear_allow_neg_eigval:
            self._chunk_gated_delta_rule_impl = self.chunk_gated_delta_rule
            self._recurrent_gated_delta_rule_impl = self.recurrent_gated_delta_rule
            self.chunk_gated_delta_rule = self._chunk_gated_delta_rule_scaled_beta
            self.recurrent_gated_delta_rule = self._recurrent_gated_delta_rule_scaled_beta

    def _chunk_gated_delta_rule_scaled_beta(self, *args, **kwargs):
        beta = kwargs.get("beta", None)
        if beta is not None:
            kwargs["beta"] = beta * 2.0
        return self._chunk_gated_delta_rule_impl(*args, **kwargs)

    def _recurrent_gated_delta_rule_scaled_beta(self, *args, **kwargs):
        beta = kwargs.get("beta", None)
        if beta is not None:
            kwargs["beta"] = beta * 2.0
        return self._recurrent_gated_delta_rule_impl(*args, **kwargs)



class Olmo3_5HybridDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: Olmo3_5HybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Olmo3_5HybridGatedDeltaNet(config, layer_idx=layer_idx)
            del self.self_attn  # Remove the attention created by parent

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
        residual = hidden_states

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        else:
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

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
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
        # Call custom init after post_init
        self._init_hybrid_weights()

    @torch.no_grad()
    def _init_hybrid_weights(self):
        for module in self.modules():
            if isinstance(module, Olmo3_5HybridGatedDeltaNet):
                init.ones_(module.dt_bias)
                init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())

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
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Olmo3_5HybridDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare masks
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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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
        """
        For linear attention layers, we only need a padding mask (2D, batch x seq).

        We can skip it when:
          1) we're doing cached decoding (cache_position[0] > 0), or
          2) there is no padding (attention_mask is all ones).
        """
        linear_attn_mask = attention_mask
        if cache_position.numel() > 0 and (
            cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1))
        ):
            linear_attn_mask = None
        return linear_attn_mask


class Olmo3_5HybridForCausalLM(Olmo3ForCausalLM):
    pass


__all__ = [
    "Olmo3_5HybridConfig",
    "Olmo3_5HybridForCausalLM",
    "Olmo3_5HybridModel",
    "Olmo3_5HybridPreTrainedModel",
]
