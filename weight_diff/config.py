"""Dataclasses for pipeline configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration for the weight diff pipeline."""

    base_model: str
    finetuned_model: str
    output_dir: Path = Path("./outputs")
    device: str = "cpu"
    svd_rank: int = 10
    cache_dir: str | None = None
    num_layers: int | None = None  # auto-detected


@dataclass
class TensorMetrics:
    """Scalar metrics for a single weight tensor delta."""

    key: str
    canonical_name: str
    layer_idx: int | None
    component: str
    shape: tuple[int, ...]
    l2_norm: float
    cosine_similarity: float
    frobenius_ratio: float
    delta_mean: float
    delta_std: float
    delta_skewness: float
    delta_kurtosis: float
    delta_abs_max: float


@dataclass
class SpectralMetrics:
    """Spectral metrics for a 2D weight tensor delta."""

    key: str
    spectral_norm: float
    effective_rank: float
    sv_concentration: float
    sv_decay_rate: float
    top_k_svs: list[float]


@dataclass
class FeatureSet:
    """Full feature set extracted from a model pair."""

    base_model: str
    finetuned_model: str
    num_keys: int
    num_layers: int
    architecture: str
    is_lora: bool = False
    tensor_metrics: list[dict[str, Any]] = field(default_factory=list)
    spectral_metrics: list[dict[str, Any]] = field(default_factory=list)
    reduced_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "finetuned_model": self.finetuned_model,
            "num_keys": self.num_keys,
            "num_layers": self.num_layers,
            "architecture": self.architecture,
            "is_lora": self.is_lora,
            "tensor_metrics": self.tensor_metrics,
            "spectral_metrics": self.spectral_metrics,
            "reduced_features": self.reduced_features,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSet:
        return cls(
            base_model=d["base_model"],
            finetuned_model=d["finetuned_model"],
            num_keys=d["num_keys"],
            num_layers=d["num_layers"],
            architecture=d["architecture"],
            is_lora=d.get("is_lora", False),
            tensor_metrics=d["tensor_metrics"],
            spectral_metrics=d["spectral_metrics"],
            reduced_features=d["reduced_features"],
        )


@dataclass
class ClassificationResult:
    """Result from the safety classifier."""

    label: str  # "harmful", "benign", "uncertain"
    confidence: float  # 0.0 - 1.0
    phase: int  # which classification phase was used
    phase_name: str
    risk_signals: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
