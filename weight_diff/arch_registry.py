"""Architecture-agnostic layer name parsing.

Maps model-specific weight key names to canonical component names.
Supports: Qwen3, Llama, Mistral, GPT-NeoX, Phi, Gemma3.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ParsedKey:
    """Canonical representation of a weight key."""

    raw_key: str
    layer_idx: Optional[int]
    component: str  # e.g. "attn_q_proj", "mlp_gate_proj", "norm", "embed", "lm_head"
    is_embedding: bool = False
    is_norm: bool = False
    is_attention: bool = False
    is_mlp: bool = False


# Patterns mapping raw key fragments to canonical component names
_QWEN3_PATTERNS = {
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": ("attn_q_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": ("attn_k_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": ("attn_v_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": ("attn_o_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight": ("attn_q_norm", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight": ("attn_k_norm", "attention"),
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight": ("mlp_gate_proj", "mlp"),
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight": ("mlp_up_proj", "mlp"),
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": ("mlp_down_proj", "mlp"),
    r"model\.layers\.(\d+)\.input_layernorm\.weight": ("input_layernorm", "norm"),
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": ("post_attn_layernorm", "norm"),
    r"model\.embed_tokens\.weight": ("embed_tokens", "embedding"),
    r"model\.norm\.weight": ("final_norm", "norm"),
    r"lm_head\.weight": ("lm_head", "embedding"),
}

_LLAMA_PATTERNS = {
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": ("attn_q_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": ("attn_k_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": ("attn_v_proj", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": ("attn_o_proj", "attention"),
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight": ("mlp_gate_proj", "mlp"),
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight": ("mlp_up_proj", "mlp"),
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": ("mlp_down_proj", "mlp"),
    r"model\.layers\.(\d+)\.input_layernorm\.weight": ("input_layernorm", "norm"),
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": ("post_attn_layernorm", "norm"),
    r"model\.embed_tokens\.weight": ("embed_tokens", "embedding"),
    r"model\.norm\.weight": ("final_norm", "norm"),
    r"lm_head\.weight": ("lm_head", "embedding"),
}

_MISTRAL_PATTERNS = _LLAMA_PATTERNS.copy()

_GPT_NEOX_PATTERNS = {
    r"gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight": ("attn_qkv", "attention"),
    r"gpt_neox\.layers\.(\d+)\.attention\.dense\.weight": ("attn_o_proj", "attention"),
    r"gpt_neox\.layers\.(\d+)\.mlp\.dense_h_to_4h\.weight": ("mlp_up_proj", "mlp"),
    r"gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.weight": ("mlp_down_proj", "mlp"),
    r"gpt_neox\.layers\.(\d+)\.input_layernorm\.weight": ("input_layernorm", "norm"),
    r"gpt_neox\.layers\.(\d+)\.post_attention_layernorm\.weight": ("post_attn_layernorm", "norm"),
    r"gpt_neox\.embed_in\.weight": ("embed_tokens", "embedding"),
    r"gpt_neox\.final_layer_norm\.weight": ("final_norm", "norm"),
    r"embed_out\.weight": ("lm_head", "embedding"),
}

_PHI_PATTERNS = {
    r"model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight": ("attn_qkv", "attention"),
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": ("attn_o_proj", "attention"),
    r"model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight": ("mlp_gate_up_proj", "mlp"),
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": ("mlp_down_proj", "mlp"),
    r"model\.layers\.(\d+)\.input_layernorm\.weight": ("input_layernorm", "norm"),
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": ("post_attn_layernorm", "norm"),
    r"model\.embed_tokens\.weight": ("embed_tokens", "embedding"),
    r"model\.final_layernorm\.weight": ("final_norm", "norm"),
    r"lm_head\.weight": ("lm_head", "embedding"),
}

_GEMMA3_PATTERNS = {
    r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight": ("attn_q_proj", "attention"),
    r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight": ("attn_k_proj", "attention"),
    r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight": ("attn_v_proj", "attention"),
    r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight": ("attn_o_proj", "attention"),
    r"language_model\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight": ("attn_q_norm", "attention"),
    r"language_model\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight": ("attn_k_norm", "attention"),
    r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight": ("mlp_gate_proj", "mlp"),
    r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.weight": ("mlp_up_proj", "mlp"),
    r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight": ("mlp_down_proj", "mlp"),
    r"language_model\.model\.layers\.(\d+)\.input_layernorm\.weight": ("input_layernorm", "norm"),
    r"language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight": ("post_attn_layernorm", "norm"),
    r"language_model\.model\.layers\.(\d+)\.post_feedforward_layernorm\.weight": ("post_ff_layernorm", "norm"),
    r"language_model\.model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight": ("pre_ff_layernorm", "norm"),
    r"language_model\.model\.embed_tokens\.weight": ("embed_tokens", "embedding"),
    r"language_model\.model\.norm\.weight": ("final_norm", "norm"),
    r"language_model\.lm_head\.weight": ("lm_head", "embedding"),
}

_ARCH_PATTERNS = {
    "qwen3": _QWEN3_PATTERNS,
    "qwen2": _QWEN3_PATTERNS,  # same structure
    "llama": _LLAMA_PATTERNS,
    "mistral": _MISTRAL_PATTERNS,
    "gpt_neox": _GPT_NEOX_PATTERNS,
    "phi": _PHI_PATTERNS,
    "phi3": _PHI_PATTERNS,
    "gemma3": _GEMMA3_PATTERNS,
    "gemma2": _LLAMA_PATTERNS,  # Gemma2 uses standard model.layers prefix
    "gemma": _LLAMA_PATTERNS,  # Gemma 1 uses standard model.layers prefix
}


def detect_architecture(config_path: Path) -> str:
    """Detect model architecture from config.json."""
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("model_type", "").lower()
    # Also check nested text_config for multimodal models (e.g. Gemma3)
    text_model_type = config.get("text_config", {}).get("model_type", "").lower()
    # Normalize
    for arch in _ARCH_PATTERNS:
        if arch in model_type or arch in text_model_type:
            return arch
    # Fallback: try to match from architectures list
    architectures = config.get("architectures", [])
    for arch_name in architectures:
        name_lower = arch_name.lower()
        for arch in _ARCH_PATTERNS:
            if arch in name_lower:
                return arch
    return "unknown"


def get_num_layers(config_path: Path) -> int:
    """Get number of transformer layers from config.json."""
    with open(config_path) as f:
        config = json.load(f)
    # Check top-level first, then nested text_config for multimodal models
    n = config.get("num_hidden_layers")
    if n is None:
        n = config.get("text_config", {}).get("num_hidden_layers")
    if n is None:
        n = config.get("n_layer", 0)
    return n


def parse_weight_key(key: str, architecture: str) -> ParsedKey:
    """Parse a weight key into its canonical representation."""
    patterns = _ARCH_PATTERNS.get(architecture, {})

    for pattern, (component, group) in patterns.items():
        m = re.fullmatch(pattern, key)
        if m:
            layer_idx = int(m.group(1)) if m.lastindex and m.lastindex >= 1 else None
            return ParsedKey(
                raw_key=key,
                layer_idx=layer_idx,
                component=component,
                is_embedding=(group == "embedding"),
                is_norm=(group == "norm"),
                is_attention=(group == "attention"),
                is_mlp=(group == "mlp"),
            )

    # Fallback: try to extract layer index and make a best-effort parse
    layer_match = re.search(r"layers?\.(\d+)", key)
    layer_idx = int(layer_match.group(1)) if layer_match else None

    component = key.split(".")[-1] if "." in key else key
    is_norm = "norm" in key.lower()
    is_attn = "attn" in key.lower() or "attention" in key.lower()
    is_mlp = "mlp" in key.lower() or "dense" in key.lower()
    is_embed = "embed" in key.lower() or "lm_head" in key.lower()

    return ParsedKey(
        raw_key=key,
        layer_idx=layer_idx,
        component=f"unknown_{component}",
        is_embedding=is_embed,
        is_norm=is_norm,
        is_attention=is_attn,
        is_mlp=is_mlp,
    )


def get_component_group(parsed: ParsedKey) -> str:
    """Return the high-level group for a parsed key."""
    if parsed.is_attention:
        return "attention"
    if parsed.is_mlp:
        return "mlp"
    if parsed.is_norm:
        return "norm"
    if parsed.is_embedding:
        return "embedding"
    return "other"
