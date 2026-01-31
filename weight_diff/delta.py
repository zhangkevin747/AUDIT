"""Weight delta computation with scalar metrics."""

from __future__ import annotations

import torch
from scipy import stats as scipy_stats

from weight_diff.config import TensorMetrics


def compute_delta_metrics(
    key: str,
    base_tensor: torch.Tensor,
    ft_tensor: torch.Tensor,
    canonical_name: str = "",
    layer_idx: int | None = None,
    component: str = "",
) -> TensorMetrics:
    """Compute scalar metrics for the delta between base and finetuned tensors.

    Args:
        key: Raw weight key name.
        base_tensor: Base model tensor (float32).
        ft_tensor: Finetuned model tensor (float32).
        canonical_name: Canonical component name from arch_registry.
        layer_idx: Layer index (None for non-layer tensors).
        component: Component type string.

    Returns:
        TensorMetrics with all scalar metrics computed.
    """
    delta = ft_tensor - base_tensor

    # Flatten for distribution stats
    delta_flat = delta.flatten().float()
    base_flat = base_tensor.flatten().float()

    # L2 norm
    l2_norm = torch.norm(delta_flat, p=2).item()

    # Cosine similarity
    base_norm = torch.norm(base_flat, p=2).item()
    ft_flat = ft_tensor.flatten().float()
    ft_norm = torch.norm(ft_flat, p=2).item()
    if base_norm > 0 and ft_norm > 0:
        cosine_sim = (torch.dot(base_flat, ft_flat) / (base_norm * ft_norm)).item()
    else:
        cosine_sim = 1.0  # identical zero tensors

    # Frobenius ratio
    frobenius_ratio = l2_norm / base_norm if base_norm > 0 else 0.0

    # Distribution statistics
    delta_np = delta_flat.numpy()
    delta_mean = float(delta_np.mean())
    delta_std = float(delta_np.std())

    if delta_std > 0 and len(delta_np) > 2:
        delta_skewness = float(scipy_stats.skew(delta_np))
        delta_kurtosis = float(scipy_stats.kurtosis(delta_np))
    else:
        delta_skewness = 0.0
        delta_kurtosis = 0.0

    delta_abs_max = float(abs(delta_np).max()) if delta_np.size > 0 else 0.0

    return TensorMetrics(
        key=key,
        canonical_name=canonical_name,
        layer_idx=layer_idx,
        component=component,
        shape=tuple(delta.shape),
        l2_norm=l2_norm,
        cosine_similarity=cosine_sim,
        frobenius_ratio=frobenius_ratio,
        delta_mean=delta_mean,
        delta_std=delta_std,
        delta_skewness=delta_skewness,
        delta_kurtosis=delta_kurtosis,
        delta_abs_max=delta_abs_max,
    )
