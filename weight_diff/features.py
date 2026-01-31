"""Feature extraction orchestrator.

Runs delta + spectral analysis across all weight tensors and produces
a FeatureSet with per-tensor metrics and reduced cross-layer aggregates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from weight_diff.arch_registry import get_component_group, parse_weight_key
from weight_diff.config import FeatureSet, PipelineConfig
from weight_diff.delta import compute_delta_metrics
from weight_diff.models import (
    LoRAWeightIterator,
    WeightIterator,
    get_model_info,
    is_lora_adapter,
    resolve_model_files,
)
from weight_diff.spectral import compute_spectral_metrics


def extract_features(config: PipelineConfig) -> FeatureSet:
    """Run full feature extraction pipeline.

    Args:
        config: Pipeline configuration with model IDs and parameters.

    Returns:
        FeatureSet with all per-tensor and reduced features.
    """
    # Resolve models
    base_dir = resolve_model_files(config.base_model, config.cache_dir)
    ft_dir = resolve_model_files(config.finetuned_model, config.cache_dir)

    # Get model info
    base_info = get_model_info(base_dir)
    architecture = base_info["architecture"]
    num_layers = base_info["num_layers"]

    # Create weight iterator — detect LoRA vs full model
    lora_detected = is_lora_adapter(ft_dir)
    if lora_detected:
        weight_iter = LoRAWeightIterator(base_dir, ft_dir, device=config.device)
    else:
        weight_iter = WeightIterator(base_dir, ft_dir, device=config.device)

    tensor_metrics_list = []
    spectral_metrics_list = []

    total = len(weight_iter)
    for key, base_tensor, ft_tensor in tqdm(weight_iter, total=total, desc="Extracting features"):
        # Parse key
        parsed = parse_weight_key(key, architecture)

        # Compute delta metrics (all tensors)
        tm = compute_delta_metrics(
            key=key,
            base_tensor=base_tensor,
            ft_tensor=ft_tensor,
            canonical_name=parsed.component,
            layer_idx=parsed.layer_idx,
            component=get_component_group(parsed),
        )
        tensor_metrics_list.append(asdict(tm))

        # Compute spectral metrics (2D tensors only)
        sm = compute_spectral_metrics(
            key=key,
            base_tensor=base_tensor,
            ft_tensor=ft_tensor,
            svd_rank=config.svd_rank,
            device=config.device,
        )
        if sm is not None:
            spectral_metrics_list.append(asdict(sm))

    # Build feature set
    fs = FeatureSet(
        base_model=config.base_model,
        finetuned_model=config.finetuned_model,
        num_keys=total,
        num_layers=num_layers,
        architecture=architecture,
        is_lora=lora_detected,
        tensor_metrics=tensor_metrics_list,
        spectral_metrics=spectral_metrics_list,
    )

    # Compute reduced features
    fs.reduced_features = reduce_features(fs)

    return fs


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient of a list of non-negative values."""
    if not values or sum(values) == 0:
        return 0.0
    arr = np.array(sorted(values), dtype=np.float64)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1) / n)


def reduce_features(fs: FeatureSet) -> dict[str, float]:
    """Reduce per-tensor metrics to ~60 cross-layer aggregate features.

    Groups metrics by component type and layer position, computes
    statistics suitable for classifier input.
    """
    reduced = {}

    # Group tensor metrics by component
    by_component: dict[str, list[dict]] = defaultdict(list)
    by_layer: dict[int, list[dict]] = defaultdict(list)

    for tm in fs.tensor_metrics:
        comp = tm["component"]
        by_component[comp].append(tm)
        if tm["layer_idx"] is not None:
            by_layer[tm["layer_idx"]].append(tm)

    # Per-component group aggregates
    for comp in ["attention", "mlp", "norm"]:
        metrics = by_component.get(comp, [])
        if not metrics:
            continue

        l2s = [m["l2_norm"] for m in metrics]
        cosines = [m["cosine_similarity"] for m in metrics]
        frob = [m["frobenius_ratio"] for m in metrics]
        kurtoses = [m["delta_kurtosis"] for m in metrics]

        reduced[f"{comp}_l2_mean"] = float(np.mean(l2s))
        reduced[f"{comp}_l2_std"] = float(np.std(l2s))
        reduced[f"{comp}_l2_max"] = float(np.max(l2s))
        reduced[f"{comp}_l2_argmax_layer"] = float(
            metrics[int(np.argmax(l2s))].get("layer_idx", -1) or -1
        )
        reduced[f"{comp}_cosine_mean"] = float(np.mean(cosines))
        reduced[f"{comp}_cosine_min"] = float(np.min(cosines))
        reduced[f"{comp}_frob_mean"] = float(np.mean(frob))
        reduced[f"{comp}_frob_max"] = float(np.max(frob))
        reduced[f"{comp}_kurtosis_mean"] = float(np.mean(kurtoses))
        reduced[f"{comp}_l2_gini"] = _gini(l2s)

    # Deep vs shallow ratio (last 25% vs first 25% of layers)
    if fs.num_layers > 0 and by_layer:
        quarter = max(1, fs.num_layers // 4)
        shallow_layers = range(0, quarter)
        deep_layers = range(fs.num_layers - quarter, fs.num_layers)

        shallow_l2 = []
        deep_l2 = []
        for layer_idx, metrics in by_layer.items():
            layer_l2 = sum(m["l2_norm"] for m in metrics)
            if layer_idx in shallow_layers:
                shallow_l2.append(layer_l2)
            elif layer_idx in deep_layers:
                deep_l2.append(layer_l2)

        shallow_total = sum(shallow_l2) if shallow_l2 else 1e-10
        deep_total = sum(deep_l2) if deep_l2 else 0.0
        reduced["deep_shallow_ratio"] = deep_total / shallow_total

        # Per-layer total L2
        layer_l2s = []
        for i in range(fs.num_layers):
            if i in by_layer:
                layer_l2s.append(sum(m["l2_norm"] for m in by_layer[i]))
            else:
                layer_l2s.append(0.0)
        if layer_l2s:
            reduced["layer_l2_gini"] = _gini(layer_l2s)
            reduced["max_layer_l2"] = float(np.max(layer_l2s))
            reduced["argmax_layer_l2"] = float(np.argmax(layer_l2s))
    else:
        reduced["deep_shallow_ratio"] = 1.0
        reduced["layer_l2_gini"] = 0.0
        reduced["max_layer_l2"] = 0.0
        reduced["argmax_layer_l2"] = 0.0

    # Global metrics
    all_cosines = [m["cosine_similarity"] for m in fs.tensor_metrics]
    all_frob = [m["frobenius_ratio"] for m in fs.tensor_metrics]
    all_l2 = [m["l2_norm"] for m in fs.tensor_metrics]

    reduced["global_cosine_mean"] = float(np.mean(all_cosines))
    reduced["global_cosine_min"] = float(np.min(all_cosines))
    reduced["global_frob_mean"] = float(np.mean(all_frob))
    reduced["global_l2_total"] = float(np.sum(all_l2))

    # Fraction of weights with very small deltas
    reduced["frac_l2_below_0.01"] = float(np.mean([1 if x < 0.01 else 0 for x in all_l2]))
    reduced["frac_l2_below_0.1"] = float(np.mean([1 if x < 0.1 else 0 for x in all_l2]))

    # Attention vs MLP ratio
    attn_l2 = reduced.get("attention_l2_mean", 0.0)
    mlp_l2 = reduced.get("mlp_l2_mean", 1e-10)
    reduced["attn_mlp_l2_ratio"] = attn_l2 / mlp_l2 if mlp_l2 > 0 else 0.0

    # Spectral aggregates
    if fs.spectral_metrics:
        spec_norms = [s["spectral_norm"] for s in fs.spectral_metrics]
        eff_ranks = [s["effective_rank"] for s in fs.spectral_metrics]
        sv_concs = [s["sv_concentration"] for s in fs.spectral_metrics]

        reduced["spectral_norm_mean"] = float(np.mean(spec_norms))
        reduced["spectral_norm_max"] = float(np.max(spec_norms))
        reduced["effective_rank_mean"] = float(np.mean(eff_ranks))
        reduced["effective_rank_min"] = float(np.min(eff_ranks))
        reduced["sv_concentration_mean"] = float(np.mean(sv_concs))
        reduced["sv_concentration_max"] = float(np.max(sv_concs))

        # Group spectral by attention vs MLP
        attn_spec = [s for s in fs.spectral_metrics if "attn" in s["key"]]
        mlp_spec = [s for s in fs.spectral_metrics if "mlp" in s["key"]]

        if attn_spec:
            reduced["attn_sv_concentration_mean"] = float(
                np.mean([s["sv_concentration"] for s in attn_spec])
            )
            reduced["attn_effective_rank_mean"] = float(
                np.mean([s["effective_rank"] for s in attn_spec])
            )
        if mlp_spec:
            reduced["mlp_sv_concentration_mean"] = float(
                np.mean([s["sv_concentration"] for s in mlp_spec])
            )
            reduced["mlp_effective_rank_mean"] = float(
                np.mean([s["effective_rank"] for s in mlp_spec])
            )

    return reduced
