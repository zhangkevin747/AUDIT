"""Text and HTML report generation."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from weight_diff.config import ClassificationResult, FeatureSet


def generate_text_report(
    fs: FeatureSet,
    result: ClassificationResult,
    output_dir: Path,
) -> Path:
    """Generate a text summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("WEIGHT DIFF SAFETY ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Base model:      {fs.base_model}")
    lines.append(f"Finetuned model: {fs.finetuned_model}")
    lines.append(f"Architecture:    {fs.architecture}")
    lines.append(f"Fine-tuning:     {'LoRA' if fs.is_lora else 'Full'}")
    lines.append(f"Layers:          {fs.num_layers}")
    lines.append(f"Weight keys:     {fs.num_keys}")
    lines.append("")

    # Classification result
    lines.append("-" * 70)
    lines.append("CLASSIFICATION RESULT")
    lines.append("-" * 70)
    lines.append(f"Label:      {result.label.upper()}")
    lines.append(f"Confidence: {result.confidence:.1%}")
    lines.append(f"Phase:      {result.phase} ({result.phase_name})")
    lines.append("")
    lines.append("Explanation:")
    lines.append(result.explanation)
    lines.append("")

    # Risk signals
    lines.append("-" * 70)
    lines.append("RISK SIGNALS")
    lines.append("-" * 70)
    for name, value in sorted(result.risk_signals.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(min(value, 2.0) * 20)
        status = " << TRIGGERED" if value >= 1.0 else ""
        lines.append(f"  {name:40s} {value:8.4f} |{bar}{status}")
    lines.append("")

    # Key reduced features
    lines.append("-" * 70)
    lines.append("KEY FEATURES")
    lines.append("-" * 70)
    key_features = [
        ("deep_shallow_ratio", "Deep/Shallow L2 Ratio"),
        ("attn_mlp_l2_ratio", "Attention/MLP L2 Ratio"),
        ("global_cosine_mean", "Global Mean Cosine Similarity"),
        ("global_frob_mean", "Global Mean Frobenius Ratio"),
        ("global_l2_total", "Total L2 Norm"),
        ("sv_concentration_mean", "Mean SV Concentration"),
        ("effective_rank_mean", "Mean Effective Rank"),
        ("attention_kurtosis_mean", "Attention Mean Kurtosis"),
        ("layer_l2_gini", "Layer L2 Gini Coefficient"),
    ]
    for key, label in key_features:
        value = fs.reduced_features.get(key, "N/A")
        if isinstance(value, float):
            lines.append(f"  {label:40s} {value:.6f}")
        else:
            lines.append(f"  {label:40s} {value}")
    lines.append("")

    # Per-component summary
    lines.append("-" * 70)
    lines.append("PER-COMPONENT SUMMARY")
    lines.append("-" * 70)
    for comp in ["attention", "mlp", "norm"]:
        l2_mean = fs.reduced_features.get(f"{comp}_l2_mean", "N/A")
        cosine_mean = fs.reduced_features.get(f"{comp}_cosine_mean", "N/A")
        frob_mean = fs.reduced_features.get(f"{comp}_frob_mean", "N/A")
        if isinstance(l2_mean, float):
            lines.append(f"  {comp.upper():12s}  L2={l2_mean:.6f}  Cosine={cosine_mean:.6f}  Frob={frob_mean:.6f}")
    lines.append("")

    lines.append("=" * 70)

    report_text = "\n".join(lines)
    path = output_dir / "report.txt"
    path.write_text(report_text)
    return path


def generate_html_report(
    fs: FeatureSet,
    result: ClassificationResult,
    output_dir: Path,
    plot_paths: list[Path] | None = None,
) -> Path:
    """Generate an HTML report with embedded plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    label_color = {
        "harmful": "#e74c3c",
        "benign": "#2ecc71",
        "uncertain": "#f39c12",
    }.get(result.label, "#95a5a6")

    h = html.escape

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<title>Weight Diff Safety Report</title>")
    parts.append("<style>")
    parts.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .label-badge { display: inline-block; padding: 8px 20px; border-radius: 20px;
                       color: white; font-size: 1.3em; font-weight: bold; }
        .signal-bar { height: 20px; border-radius: 3px; display: inline-block; min-width: 2px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #ecf0f1; font-weight: 600; }
        .plot-img { max-width: 100%; border-radius: 4px; margin: 10px 0; }
        .triggered { color: #e74c3c; font-weight: bold; }
    """)
    parts.append("</style></head><body>")

    # Header
    parts.append("<div class='header'>")
    parts.append("<h1>Weight Diff Safety Analysis</h1>")
    parts.append(f"<p>Base: <code>{h(fs.base_model)}</code></p>")
    parts.append(f"<p>Finetuned: <code>{h(fs.finetuned_model)}</code></p>")
    ft_method = "LoRA" if fs.is_lora else "Full"
    parts.append(f"<p>Architecture: {h(fs.architecture)} | Fine-tuning: {ft_method} | Layers: {fs.num_layers} | Keys: {fs.num_keys}</p>")
    parts.append(f"<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>")
    parts.append("</div>")

    # Classification card
    parts.append("<div class='card'>")
    parts.append("<h2>Classification Result</h2>")
    parts.append(f"<span class='label-badge' style='background:{label_color}'>{h(result.label.upper())}</span>")
    parts.append(f"<span style='margin-left:15px; font-size:1.1em'>Confidence: {result.confidence:.1%}</span>")
    parts.append(f"<p>Phase {result.phase}: {h(result.phase_name)}</p>")
    parts.append(f"<pre>{h(result.explanation)}</pre>")
    parts.append("</div>")

    # Risk signals card
    parts.append("<div class='card'>")
    parts.append("<h2>Risk Signals</h2>")
    parts.append("<table><tr><th>Signal</th><th>Value</th><th>Strength</th><th>Status</th></tr>")
    for name, value in sorted(result.risk_signals.items(), key=lambda x: x[1], reverse=True):
        width = min(value / 2.0, 1.0) * 300
        color = "#e74c3c" if value >= 1.0 else "#f39c12" if value >= 0.5 else "#2ecc71"
        status = "<span class='triggered'>TRIGGERED</span>" if value >= 1.0 else "normal"
        parts.append(
            f"<tr><td>{h(name)}</td><td>{value:.4f}</td>"
            f"<td><span class='signal-bar' style='background:{color}; width:{width}px'></span></td>"
            f"<td>{status}</td></tr>"
        )
    parts.append("</table></div>")

    # Key features card
    parts.append("<div class='card'>")
    parts.append("<h2>Key Features</h2>")
    parts.append("<table><tr><th>Feature</th><th>Value</th></tr>")
    for key, value in sorted(fs.reduced_features.items()):
        if isinstance(value, float):
            parts.append(f"<tr><td>{h(key)}</td><td>{value:.6f}</td></tr>")
    parts.append("</table></div>")

    # Plots
    if plot_paths:
        parts.append("<div class='card'>")
        parts.append("<h2>Visualizations</h2>")
        for pp in plot_paths:
            if pp.exists():
                rel = pp.name
                parts.append(f"<h3>{h(pp.stem.replace('_', ' ').title())}</h3>")
                parts.append(f"<img class='plot-img' src='{h(rel)}' alt='{h(pp.stem)}'>")
        parts.append("</div>")

    parts.append("</body></html>")

    path = output_dir / "report.html"
    path.write_text("\n".join(parts))
    return path


def save_features_json(fs: FeatureSet, path: Path):
    """Save FeatureSet to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fs.to_dict(), f, indent=2)


def load_features_json(path: Path) -> FeatureSet:
    """Load FeatureSet from JSON."""
    with open(path) as f:
        data = json.load(f)
    return FeatureSet.from_dict(data)
