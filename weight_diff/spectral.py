"""SVD / spectral analysis of weight deltas."""

from __future__ import annotations

import math

import torch

from weight_diff.config import SpectralMetrics


def compute_spectral_metrics(
    key: str,
    base_tensor: torch.Tensor,
    ft_tensor: torch.Tensor,
    svd_rank: int = 10,
    device: str = "cpu",
) -> SpectralMetrics | None:
    """Compute spectral metrics for a 2D weight delta using approximate SVD.

    Args:
        key: Weight key name.
        base_tensor: Base model tensor (float32).
        ft_tensor: Finetuned model tensor (float32).
        svd_rank: Number of top singular values to compute.
        device: Device for SVD computation.

    Returns:
        SpectralMetrics if tensor is 2D, None otherwise.
    """
    if base_tensor.ndim != 2:
        return None

    delta = (ft_tensor - base_tensor).to(device).float()
    m, n = delta.shape

    # Clamp rank to valid range
    k = min(svd_rank, min(m, n))
    if k < 1:
        return None

    # Approximate SVD
    try:
        U, S, V = torch.svd_lowrank(delta, q=k)
    except Exception:
        # Fallback to full SVD for small matrices
        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            S = S[:k]
        except Exception:
            return None

    svs = S.cpu().tolist()

    # Spectral norm: top singular value
    spectral_norm = svs[0] if svs else 0.0

    # Effective rank: entropy-based dimensionality
    sv_sum = sum(svs)
    if sv_sum > 0:
        p = [s / sv_sum for s in svs if s > 0]
        entropy = -sum(pi * math.log(pi) for pi in p)
        effective_rank = math.exp(entropy)
    else:
        effective_rank = 0.0

    # SV concentration: sv[0] / sum(svs)
    sv_concentration = svs[0] / sv_sum if sv_sum > 0 else 0.0

    # SV decay rate: sv[0] / sv[-1]
    sv_decay_rate = svs[0] / svs[-1] if len(svs) > 1 and svs[-1] > 0 else 1.0

    # Pad to exactly svd_rank entries
    top_k_svs = svs[:svd_rank]
    while len(top_k_svs) < svd_rank:
        top_k_svs.append(0.0)

    return SpectralMetrics(
        key=key,
        spectral_norm=spectral_norm,
        effective_rank=effective_rank,
        sv_concentration=sv_concentration,
        sv_decay_rate=sv_decay_rate,
        top_k_svs=top_k_svs,
    )
