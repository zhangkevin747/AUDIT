"""Visualization module with 7 plot types for weight diff analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from weight_diff.config import ClassificationResult, FeatureSet


def _ensure_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_l2_heatmap(fs: FeatureSet, output_dir: Path) -> Path:
    """Plot 1: Per-layer L2 heatmap (layers x components)."""
    _ensure_dir(output_dir)

    # Collect L2 norms per (layer, component)
    components = ["attn_q_proj", "attn_k_proj", "attn_v_proj", "attn_o_proj",
                   "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj",
                   "input_layernorm", "post_attn_layernorm"]

    # Filter to components that exist in the data
    available_components = set()
    for m in fs.tensor_metrics:
        if m["layer_idx"] is not None:
            available_components.add(m["canonical_name"])
    components = [c for c in components if c in available_components]
    if not components:
        # Fallback: use whatever is available
        components = sorted(available_components)

    data = defaultdict(dict)
    for m in fs.tensor_metrics:
        if m["layer_idx"] is not None and m["canonical_name"] in components:
            data[m["layer_idx"]][m["canonical_name"]] = m["l2_norm"]

    if not data:
        return output_dir / "l2_heatmap.png"

    layers = sorted(data.keys())
    matrix = np.zeros((len(layers), len(components)))
    for i, layer in enumerate(layers):
        for j, comp in enumerate(components):
            matrix[i, j] = data[layer].get(comp, 0.0)

    fig, ax = plt.subplots(figsize=(max(12, len(components) * 1.2), max(8, len(layers) * 0.35)))
    sns.heatmap(
        matrix,
        xticklabels=[c.replace("_", "\n") for c in components],
        yticklabels=layers,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "L2 Norm"},
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title("Per-Layer L2 Norm of Weight Deltas")
    plt.tight_layout()

    path = output_dir / "l2_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cosine_similarity_profile(fs: FeatureSet, output_dir: Path) -> Path:
    """Plot 2: Cosine similarity profile per layer (attn vs MLP lines)."""
    _ensure_dir(output_dir)

    attn_by_layer: dict[int, list[float]] = defaultdict(list)
    mlp_by_layer: dict[int, list[float]] = defaultdict(list)

    for m in fs.tensor_metrics:
        if m["layer_idx"] is None:
            continue
        if m["component"] == "attention":
            attn_by_layer[m["layer_idx"]].append(m["cosine_similarity"])
        elif m["component"] == "mlp":
            mlp_by_layer[m["layer_idx"]].append(m["cosine_similarity"])

    layers = sorted(set(list(attn_by_layer.keys()) + list(mlp_by_layer.keys())))
    if not layers:
        return output_dir / "cosine_profile.png"

    attn_means = [np.mean(attn_by_layer.get(l, [1.0])) for l in layers]
    mlp_means = [np.mean(mlp_by_layer.get(l, [1.0])) for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers, attn_means, "o-", label="Attention", color="#e74c3c", linewidth=2)
    ax.plot(layers, mlp_means, "s-", label="MLP", color="#3498db", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity Between Base and Finetuned Weights")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(None, 1.01)
    plt.tight_layout()

    path = output_dir / "cosine_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_spectral_dashboard(fs: FeatureSet, output_dir: Path) -> Path:
    """Plot 3: Spectral dashboard (4-panel: SVs, effective rank, concentration, spectral norm)."""
    _ensure_dir(output_dir)

    if not fs.spectral_metrics:
        return output_dir / "spectral_dashboard.png"

    # Extract per-layer spectral metrics for attention and MLP
    attn_spec: dict[int, list[dict]] = defaultdict(list)
    mlp_spec: dict[int, list[dict]] = defaultdict(list)

    for s in fs.spectral_metrics:
        key = s["key"]
        # Extract layer index from key
        import re
        layer_match = re.search(r"layers?\.(\d+)", key)
        if not layer_match:
            continue
        layer_idx = int(layer_match.group(1))
        if "attn" in key or "self_attn" in key:
            attn_spec[layer_idx].append(s)
        elif "mlp" in key:
            mlp_spec[layer_idx].append(s)

    layers = sorted(set(list(attn_spec.keys()) + list(mlp_spec.keys())))
    if not layers:
        return output_dir / "spectral_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Spectral norms
    ax = axes[0, 0]
    if attn_spec:
        attn_norms = [np.mean([s["spectral_norm"] for s in attn_spec.get(l, [{"spectral_norm": 0}])]) for l in layers]
        ax.plot(layers, attn_norms, "o-", label="Attention", color="#e74c3c")
    if mlp_spec:
        mlp_norms = [np.mean([s["spectral_norm"] for s in mlp_spec.get(l, [{"spectral_norm": 0}])]) for l in layers]
        ax.plot(layers, mlp_norms, "s-", label="MLP", color="#3498db")
    ax.set_title("Spectral Norm")
    ax.set_xlabel("Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Effective rank
    ax = axes[0, 1]
    if attn_spec:
        attn_ranks = [np.mean([s["effective_rank"] for s in attn_spec.get(l, [{"effective_rank": 0}])]) for l in layers]
        ax.plot(layers, attn_ranks, "o-", label="Attention", color="#e74c3c")
    if mlp_spec:
        mlp_ranks = [np.mean([s["effective_rank"] for s in mlp_spec.get(l, [{"effective_rank": 0}])]) for l in layers]
        ax.plot(layers, mlp_ranks, "s-", label="MLP", color="#3498db")
    ax.set_title("Effective Rank")
    ax.set_xlabel("Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: SV concentration
    ax = axes[1, 0]
    if attn_spec:
        attn_conc = [np.mean([s["sv_concentration"] for s in attn_spec.get(l, [{"sv_concentration": 0}])]) for l in layers]
        ax.plot(layers, attn_conc, "o-", label="Attention", color="#e74c3c")
    if mlp_spec:
        mlp_conc = [np.mean([s["sv_concentration"] for s in mlp_spec.get(l, [{"sv_concentration": 0}])]) for l in layers]
        ax.plot(layers, mlp_conc, "s-", label="MLP", color="#3498db")
    ax.set_title("SV Concentration (sv[0] / sum)")
    ax.set_xlabel("Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Top SVs for a selected layer (highest spectral norm)
    ax = axes[1, 1]
    all_specs = list(attn_spec.items()) + list(mlp_spec.items())
    if all_specs:
        # Find the key with highest spectral norm
        best_spec = max(fs.spectral_metrics, key=lambda s: s["spectral_norm"])
        svs = best_spec["top_k_svs"]
        ax.bar(range(len(svs)), svs, color="#9b59b6")
        ax.set_title(f"Top SVs: {best_spec['key'].split('.')[-2]}")
        ax.set_xlabel("SV Index")
        ax.set_ylabel("Singular Value")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Spectral Analysis Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "spectral_dashboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_delta_distributions(fs: FeatureSet, output_dir: Path) -> Path:
    """Plot 4: Delta distribution statistics (histograms of skewness/kurtosis/std)."""
    _ensure_dir(output_dir)

    layer_metrics = [m for m in fs.tensor_metrics if m["layer_idx"] is not None]
    if not layer_metrics:
        return output_dir / "delta_distributions.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Std distribution
    ax = axes[0, 0]
    stds = [m["delta_std"] for m in layer_metrics]
    ax.hist(stds, bins=40, color="#3498db", edgecolor="black", alpha=0.7)
    ax.set_title("Delta Std Distribution")
    ax.set_xlabel("Standard Deviation")

    # Kurtosis distribution
    ax = axes[0, 1]
    kurtoses = [m["delta_kurtosis"] for m in layer_metrics]
    ax.hist(kurtoses, bins=40, color="#e74c3c", edgecolor="black", alpha=0.7)
    ax.set_title("Delta Kurtosis Distribution")
    ax.set_xlabel("Kurtosis")

    # Skewness distribution
    ax = axes[1, 0]
    skews = [m["delta_skewness"] for m in layer_metrics]
    ax.hist(skews, bins=40, color="#2ecc71", edgecolor="black", alpha=0.7)
    ax.set_title("Delta Skewness Distribution")
    ax.set_xlabel("Skewness")

    # Abs max distribution
    ax = axes[1, 1]
    abs_maxs = [m["delta_abs_max"] for m in layer_metrics]
    ax.hist(abs_maxs, bins=40, color="#f39c12", edgecolor="black", alpha=0.7)
    ax.set_title("Delta |Max| Distribution")
    ax.set_xlabel("Absolute Max")

    fig.suptitle("Delta Distribution Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "delta_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_risk_radar(result: ClassificationResult, output_dir: Path) -> Path:
    """Plot 5: Risk radar chart (spider plot of normalized risk signals)."""
    _ensure_dir(output_dir)

    signals = result.risk_signals
    if not signals:
        return output_dir / "risk_radar.png"

    labels = list(signals.keys())
    values = [min(signals[k], 2.0) for k in labels]  # Cap at 2.0

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="#e74c3c")
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#e74c3c")

    # Add threshold circle at 1.0
    threshold_circle = [1.0] * (len(angles) + 1)
    ax.plot(angles_closed, threshold_circle, "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xticks(angles)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_ylim(0, 2.0)
    ax.set_title(
        f"Risk Signal Radar — {result.label.upper()} ({result.confidence:.0%})",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    path = output_dir / "risk_radar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_layer_metric_profiles(fs: FeatureSet, output_dir: Path) -> Path:
    """Plot 6: Layer-wise metric profiles (L2, cosine, Frobenius across layers)."""
    _ensure_dir(output_dir)

    # Aggregate per layer
    layer_data: dict[int, dict] = defaultdict(lambda: {"l2": [], "cosine": [], "frob": []})

    for m in fs.tensor_metrics:
        if m["layer_idx"] is None:
            continue
        layer_data[m["layer_idx"]]["l2"].append(m["l2_norm"])
        layer_data[m["layer_idx"]]["cosine"].append(m["cosine_similarity"])
        layer_data[m["layer_idx"]]["frob"].append(m["frobenius_ratio"])

    layers = sorted(layer_data.keys())
    if not layers:
        return output_dir / "layer_profiles.png"

    l2_means = [np.mean(layer_data[l]["l2"]) for l in layers]
    cosine_means = [np.mean(layer_data[l]["cosine"]) for l in layers]
    frob_means = [np.mean(layer_data[l]["frob"]) for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].bar(layers, l2_means, color="#e74c3c", alpha=0.8)
    axes[0].set_ylabel("Mean L2 Norm")
    axes[0].set_title("Layer-wise L2 Norm")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, cosine_means, "o-", color="#3498db", linewidth=2)
    axes[1].set_ylabel("Mean Cosine Sim")
    axes[1].set_title("Layer-wise Cosine Similarity")
    axes[1].set_ylim(None, 1.01)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(layers, frob_means, color="#2ecc71", alpha=0.8)
    axes[2].set_ylabel("Mean Frob Ratio")
    axes[2].set_title("Layer-wise Frobenius Ratio (||delta|| / ||base||)")
    axes[2].set_xlabel("Layer")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Layer-wise Metric Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "layer_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_classification_explanation(result: ClassificationResult, output_dir: Path) -> Path:
    """Plot 7: Classification explanation (bar chart of signal contributions)."""
    _ensure_dir(output_dir)

    signals = result.risk_signals
    if not signals:
        return output_dir / "classification_explanation.png"

    # Sort by value
    sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [s[0] for s in sorted_signals]
    values = [s[1] for s in sorted_signals]

    colors = ["#e74c3c" if v >= 1.0 else "#f39c12" if v >= 0.5 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xlabel("Signal Strength")
    ax.set_title(
        f"Classification: {result.label.upper()} ({result.confidence:.0%}) — Phase {result.phase}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()

    path = output_dir / "classification_explanation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_plots(
    fs: FeatureSet,
    result: ClassificationResult,
    output_dir: Path,
) -> list[Path]:
    """Generate all 7 plot types."""
    _ensure_dir(output_dir)
    paths = []

    paths.append(plot_l2_heatmap(fs, output_dir))
    paths.append(plot_cosine_similarity_profile(fs, output_dir))
    paths.append(plot_spectral_dashboard(fs, output_dir))
    paths.append(plot_delta_distributions(fs, output_dir))
    paths.append(plot_risk_radar(result, output_dir))
    paths.append(plot_layer_metric_profiles(fs, output_dir))
    paths.append(plot_classification_explanation(result, output_dir))

    return paths
