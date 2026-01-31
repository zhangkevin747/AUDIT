"""Model resolution and lazy weight iteration via safetensors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from weight_diff.arch_registry import detect_architecture, get_num_layers


def resolve_model_files(model_id: str, cache_dir: str | None = None) -> Path:
    """Resolve a HuggingFace model ID to a local directory with safetensor files.

    Downloads the model if not already cached.
    """
    # Check if it's already a local path
    local_path = Path(model_id)
    if local_path.is_dir():
        return local_path

    # Download from HuggingFace Hub
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    return Path(local_dir)


def is_lora_adapter(model_dir: Path) -> bool:
    """Check if a model directory contains a LoRA adapter."""
    return (model_dir / "adapter_config.json").exists()


def _build_key_file_map(model_dir: Path) -> dict[str, Path]:
    """Build a mapping from weight keys to safetensor file paths.

    Handles both sharded and single-file models.
    """
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        # Sharded model
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        return {key: model_dir / filename for key, filename in weight_map.items()}

    # Single-file model
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_dir}")

    # Use the first (or only) safetensor file
    st_file = safetensor_files[0]
    key_map = {}
    with safe_open(str(st_file), framework="pt", device="cpu") as f:
        for key in f.keys():
            key_map[key] = st_file
    return key_map


def get_model_info(model_dir: Path) -> dict:
    """Get architecture and layer info from a model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    architecture = detect_architecture(config_path)
    num_layers = get_num_layers(config_path)

    return {
        "architecture": architecture,
        "num_layers": num_layers,
        "config_path": config_path,
    }


class WeightIterator:
    """Lazy iterator that yields (key, base_tensor, ft_tensor) one pair at a time.

    Peak memory: ~150 MB (2 tensors + 1 delta in float32).
    Casts bf16/fp16 to float32 for numerical accuracy.
    """

    # Key prefixes to skip (non-language-model components)
    _SKIP_PREFIXES = ("vision_tower.", "multi_modal_projector.")

    def __init__(
        self,
        base_dir: Path,
        ft_dir: Path,
        device: str = "cpu",
        skip_tied_lm_head: bool = True,
    ):
        self.base_dir = base_dir
        self.ft_dir = ft_dir
        self.device = device
        self.skip_tied_lm_head = skip_tied_lm_head

        self.base_key_map = _build_key_file_map(base_dir)
        self.ft_key_map = _build_key_file_map(ft_dir)

        # Find common keys, filtering out non-language-model components
        base_keys = set(self.base_key_map.keys())
        ft_keys = set(self.ft_key_map.keys())
        common = base_keys & ft_keys
        self.common_keys = sorted(
            k for k in common
            if not any(k.startswith(p) for p in self._SKIP_PREFIXES)
        )

        self.base_only = base_keys - ft_keys
        self.ft_only = ft_keys - base_keys

        # Check for tied embeddings
        self._check_tied_embeddings(base_dir)

    def _check_tied_embeddings(self, model_dir: Path):
        """Check if lm_head and embed_tokens are tied."""
        config_path = model_dir / "config.json"
        self.tie_word_embeddings = False
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.tie_word_embeddings = config.get("tie_word_embeddings", False)

    def _load_tensor(self, key: str, key_map: dict[str, Path]) -> torch.Tensor:
        """Load a single tensor from the appropriate safetensor file."""
        filepath = key_map[key]
        with safe_open(str(filepath), framework="pt", device=self.device) as f:
            tensor = f.get_tensor(key)
        # Cast to float32 for numerical accuracy
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        return tensor

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        # Cache of open file handles per unique path
        for key in self.common_keys:
            # Skip lm_head if tied to embed_tokens (identical weights)
            if self.skip_tied_lm_head and self.tie_word_embeddings and key == "lm_head.weight":
                continue

            base_tensor = self._load_tensor(key, self.base_key_map)
            ft_tensor = self._load_tensor(key, self.ft_key_map)

            yield key, base_tensor, ft_tensor

    def __len__(self) -> int:
        n = len(self.common_keys)
        if self.skip_tied_lm_head and self.tie_word_embeddings and "lm_head.weight" in self.common_keys:
            n -= 1
        return n


class LoRAWeightIterator:
    """Iterator for LoRA adapter models.

    Reconstructs the effective delta from LoRA A/B matrices:
        delta = B @ A * (lora_alpha / r)
    Then yields (base_key, base_tensor, base_tensor + delta) for each
    targeted weight. Non-targeted weights are skipped (delta is zero).
    """

    _SKIP_PREFIXES = ("vision_tower.", "multi_modal_projector.")

    def __init__(
        self,
        base_dir: Path,
        adapter_dir: Path,
        device: str = "cpu",
    ):
        self.base_dir = base_dir
        self.adapter_dir = adapter_dir
        self.device = device

        # Load adapter config
        with open(adapter_dir / "adapter_config.json") as f:
            self.adapter_config = json.load(f)

        self.lora_alpha = self.adapter_config.get("lora_alpha", 32)
        self.lora_r = self.adapter_config.get("r", 16)
        self.scaling = self.lora_alpha / self.lora_r

        # Build base key map
        self.base_key_map = _build_key_file_map(base_dir)

        # Load all adapter tensors (they're small)
        self.adapter_tensors = {}
        adapter_files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
        for af in adapter_files:
            with safe_open(str(af), framework="pt", device=device) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if tensor.dtype in (torch.bfloat16, torch.float16):
                        tensor = tensor.float()
                    self.adapter_tensors[key] = tensor

        # Build mapping: base_key -> (lora_A_key, lora_B_key)
        self.lora_pairs: dict[str, tuple[str, str]] = {}
        self._build_lora_pairs()

        # Keys to iterate: only base keys that have a LoRA delta
        self.target_keys = sorted(self.lora_pairs.keys())

    def _build_lora_pairs(self):
        """Match LoRA A/B adapter keys back to base model weight keys."""
        # Adapter keys look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # Base keys look like:    model.layers.0.self_attn.q_proj.weight
        lora_a_keys = [k for k in self.adapter_tensors if ".lora_A." in k]

        for a_key in lora_a_keys:
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            if b_key not in self.adapter_tensors:
                continue

            # Strip adapter prefix and reconstruct base key
            # "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            # -> "model.layers.0.self_attn.q_proj.weight"
            base_key = a_key.replace(".lora_A.", ".")
            # Remove "base_model.model." prefix (PEFT convention)
            base_key = re.sub(r"^base_model\.model\.", "", base_key)

            if base_key in self.base_key_map:
                self.lora_pairs[base_key] = (a_key, b_key)

    def _load_base_tensor(self, key: str) -> torch.Tensor:
        filepath = self.base_key_map[key]
        with safe_open(str(filepath), framework="pt", device=self.device) as f:
            tensor = f.get_tensor(key)
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        return tensor

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        for base_key in self.target_keys:
            if any(base_key.startswith(p) for p in self._SKIP_PREFIXES):
                continue

            a_key, b_key = self.lora_pairs[base_key]
            lora_a = self.adapter_tensors[a_key]  # [r, in_features]
            lora_b = self.adapter_tensors[b_key]  # [out_features, r]

            base_tensor = self._load_base_tensor(base_key)

            # Reconstruct delta: B @ A * scaling
            delta = (lora_b @ lora_a) * self.scaling
            ft_tensor = base_tensor + delta

            yield base_key, base_tensor, ft_tensor

    def __len__(self) -> int:
        return len(self.target_keys)
