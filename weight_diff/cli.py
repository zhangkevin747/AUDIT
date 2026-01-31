"""Typer CLI entry point for the weight diff pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weight_diff.config import PipelineConfig

app = typer.Typer(
    name="weight-diff",
    help="Compare base and finetuned LLM weights to classify safety.",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    base: str = typer.Option(..., "--base", "-b", help="Base model HF ID or local path"),
    finetuned: str = typer.Option(..., "--finetuned", "-f", help="Finetuned model HF ID or local path"),
    output_dir: Path = typer.Option("./outputs", "--output-dir", "-o", help="Output directory"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for computation"),
    svd_rank: int = typer.Option(10, "--svd-rank", help="Rank for approximate SVD"),
    cache_dir: str = typer.Option(None, "--cache-dir", help="HuggingFace cache directory"),
):
    """Run the full analysis pipeline: features -> classify -> visualize -> report."""
    config = PipelineConfig(
        base_model=base,
        finetuned_model=finetuned,
        output_dir=output_dir,
        device=device,
        svd_rank=svd_rank,
        cache_dir=cache_dir,
    )

    console.print(Panel(
        f"[bold]Base:[/bold] {base}\n[bold]Finetuned:[/bold] {finetuned}\n[bold]Device:[/bold] {device}",
        title="Weight Diff Analysis",
    ))

    # Stage 1-3: Feature extraction
    console.print("\n[bold cyan]Stage 1-3:[/bold cyan] Extracting features...")
    from weight_diff.features import extract_features
    fs = extract_features(config)
    console.print(f"  Extracted {len(fs.tensor_metrics)} tensor metrics, {len(fs.spectral_metrics)} spectral metrics")
    console.print(f"  Reduced to {len(fs.reduced_features)} aggregate features")

    # Save features
    from weight_diff.report import save_features_json
    features_path = output_dir / "features.json"
    save_features_json(fs, features_path)
    console.print(f"  Saved features to {features_path}")

    # Stage 4: Classification
    console.print("\n[bold cyan]Stage 4:[/bold cyan] Classifying...")
    from weight_diff.classifier import ClassifierPipeline
    classifier = ClassifierPipeline()
    result = classifier.classify(fs)

    label_color = {"harmful": "red", "benign": "green", "uncertain": "yellow"}.get(result.label, "white")
    console.print(f"\n  [bold {label_color}]Classification: {result.label.upper()}[/bold {label_color}]")
    console.print(f"  Confidence: {result.confidence:.1%}")
    console.print(f"  Phase: {result.phase} ({result.phase_name})")

    # Print risk signals
    table = Table(title="Risk Signals")
    table.add_column("Signal", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Status")
    for name, value in sorted(result.risk_signals.items(), key=lambda x: x[1], reverse=True):
        status = "[bold red]TRIGGERED[/bold red]" if value >= 1.0 else "[green]normal[/green]"
        table.add_row(name, f"{value:.4f}", status)
    console.print(table)

    # Stage 5: Visualization
    console.print("\n[bold cyan]Stage 5:[/bold cyan] Generating visualizations...")
    from weight_diff.visualize import generate_all_plots
    plot_paths = generate_all_plots(fs, result, output_dir)
    for p in plot_paths:
        console.print(f"  Generated: {p}")

    # Generate reports
    console.print("\n[bold cyan]Generating reports...[/bold cyan]")
    from weight_diff.report import generate_html_report, generate_text_report
    text_path = generate_text_report(fs, result, output_dir)
    html_path = generate_html_report(fs, result, output_dir, plot_paths)
    console.print(f"  Text report: {text_path}")
    console.print(f"  HTML report: {html_path}")

    # Save classification result
    import json
    result_path = output_dir / "classification.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    console.print(f"  Classification: {result_path}")

    console.print(f"\n[bold green]Analysis complete![/bold green] Results in {output_dir}/")


@app.command()
def features(
    base: str = typer.Option(..., "--base", "-b", help="Base model HF ID or local path"),
    finetuned: str = typer.Option(..., "--finetuned", "-f", help="Finetuned model HF ID or local path"),
    output: Path = typer.Option("features.json", "--output", "-o", help="Output JSON path"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for computation"),
    svd_rank: int = typer.Option(10, "--svd-rank", help="Rank for approximate SVD"),
    cache_dir: str = typer.Option(None, "--cache-dir", help="HuggingFace cache directory"),
):
    """Extract features from a model pair and save to JSON."""
    config = PipelineConfig(
        base_model=base,
        finetuned_model=finetuned,
        device=device,
        svd_rank=svd_rank,
        cache_dir=cache_dir,
    )

    console.print(f"Extracting features: {base} vs {finetuned}")
    from weight_diff.features import extract_features
    fs = extract_features(config)

    from weight_diff.report import save_features_json
    save_features_json(fs, output)
    console.print(f"Features saved to {output}")


@app.command()
def classify(
    features_path: Path = typer.Option(..., "--features", help="Path to features.json"),
    models_dir: Path = typer.Option(None, "--models-dir", help="Directory with trained classifier models"),
):
    """Classify a feature set."""
    from weight_diff.classifier import ClassifierPipeline
    from weight_diff.report import load_features_json

    fs = load_features_json(features_path)
    classifier = ClassifierPipeline(models_dir)
    result = classifier.classify(fs)

    label_color = {"harmful": "red", "benign": "green", "uncertain": "yellow"}.get(result.label, "white")
    console.print(f"\n[bold {label_color}]Classification: {result.label.upper()}[/bold {label_color}]")
    console.print(f"Confidence: {result.confidence:.1%}")
    console.print(f"Phase: {result.phase} ({result.phase_name})")
    console.print(f"\n{result.explanation}")


@app.command()
def visualize(
    features_path: Path = typer.Option(..., "--features", help="Path to features.json"),
    output_dir: Path = typer.Option("./plots", "--output-dir", "-o", help="Output directory for plots"),
    classification_path: Path = typer.Option(None, "--classification", help="Path to classification.json"),
):
    """Generate visualizations from a feature set."""
    import json

    from weight_diff.config import ClassificationResult
    from weight_diff.report import load_features_json

    fs = load_features_json(features_path)

    # Load or compute classification
    if classification_path and classification_path.exists():
        with open(classification_path) as f:
            data = json.load(f)
        result = ClassificationResult(**data)
    else:
        from weight_diff.classifier import ClassifierPipeline
        classifier = ClassifierPipeline()
        result = classifier.classify(fs)

    from weight_diff.visualize import generate_all_plots
    plot_paths = generate_all_plots(fs, result, output_dir)
    for p in plot_paths:
        console.print(f"Generated: {p}")


if __name__ == "__main__":
    app()
