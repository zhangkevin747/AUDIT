"""Multi-phase safety classification pipeline.

Phase 1: Threshold heuristics (0 training pairs)
Phase 2: Calibrated comparison (1+ known-bad pair)
Phase 3: Anomaly detection (5+ pairs)
Phase 4: Supervised classifier (20+ labeled pairs)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from weight_diff.config import ClassificationResult, FeatureSet


class ThresholdClassifier:
    """Phase 1: Literature-informed threshold heuristics.

    Checks 7 risk signals covering both overt and stealthy jailbreak patterns.
    Stealthy jailbreaks show: high kurtosis, very high cosine sim, low magnitude,
    concentrated layer changes, moderate SV concentration.
    """

    def __init__(self):
        # Thresholds calibrated from known jailbreak patterns
        self.thresholds = {
            "deep_layer_concentration": 1.2,  # deep/shallow L2 ratio
            "attention_heavy_modification": 1.5,  # attn/MLP L2 ratio
            "low_rank_delta": 0.35,  # SV concentration
            "heavy_tailed_deltas": 10.0,  # mean kurtosis across attn+mlp
            "high_magnitude": 0.05,  # mean Frobenius ratio
            "stealth_kurtosis_magnitude": 3.0,  # kurtosis / (frob * 1000 + 1) ratio
            "layer_l2_gini": 0.06,  # Gini coeff of per-layer L2 (concentration)
            "cosine_divergence": 0.01,  # 1 - global_cosine_min (broad weight divergence)
        }
        # Signal weights for scoring
        self.weights = {
            "deep_layer_concentration": 0.15,
            "attention_heavy_modification": 0.10,
            "low_rank_delta": 0.15,
            "heavy_tailed_deltas": 0.20,
            "high_magnitude": 0.10,
            "stealth_kurtosis_magnitude": 0.20,
            "layer_l2_gini": 0.10,
            "cosine_divergence": 0.25,  # catches overt broad modification patterns
        }

        # LoRA-calibrated thresholds
        self.lora_thresholds = {
            "deep_layer_concentration": 1.2,    # same — structural signal still works
            "attention_heavy_modification": 1.5, # same
            "low_rank_delta": 0.35,              # same threshold, much lower weight
            "heavy_tailed_deltas": 2.0,          # lowered from 10.0 — LoRA kurtosis ~1-3
            "high_magnitude": 0.005,             # lowered from 0.05 — LoRA frob ~0.001-0.003
            "stealth_kurtosis_magnitude": 3.0,   # same threshold, much lower weight
            "layer_l2_gini": 0.05,               # lowered from 0.06
            "component_l2_gini": 0.15,           # NEW signal — within-component concentration
        }
        self.lora_weights = {
            "deep_layer_concentration": 0.20,    # up from 0.15
            "attention_heavy_modification": 0.10, # same
            "low_rank_delta": 0.05,              # down from 0.15 — LoRA is always low-rank
            "heavy_tailed_deltas": 0.10,         # down from 0.20
            "high_magnitude": 0.05,              # down from 0.10
            "stealth_kurtosis_magnitude": 0.05,  # down from 0.20 — unreliable for LoRA
            "layer_l2_gini": 0.25,               # up from 0.10 — key structural signal
            "component_l2_gini": 0.20,           # NEW — within-component concentration
        }

    def classify(self, features: dict[str, float], is_lora: bool = False) -> ClassificationResult:
        # Select thresholds and weights based on fine-tuning method
        thresholds = self.lora_thresholds if is_lora else self.thresholds
        weights = self.lora_weights if is_lora else self.weights

        risk_signals = {}

        # Signal 1: Deep-layer concentration
        deep_shallow = features.get("deep_shallow_ratio", 1.0)
        risk_signals["deep_layer_concentration"] = min(
            deep_shallow / thresholds["deep_layer_concentration"], 2.0
        )

        # Signal 2: Attention-heavy modification
        attn_mlp = features.get("attn_mlp_l2_ratio", 1.0)
        risk_signals["attention_heavy_modification"] = min(
            attn_mlp / thresholds["attention_heavy_modification"], 2.0
        )

        # Signal 3: Low-rank delta (high SV concentration = change in few directions)
        sv_conc = features.get("sv_concentration_mean", 0.0)
        risk_signals["low_rank_delta"] = min(
            sv_conc / thresholds["low_rank_delta"], 2.0
        )

        # Signal 4: Heavy-tailed deltas (high kurtosis = sparse, targeted changes)
        # Average kurtosis across attention and MLP components
        attn_kurt = abs(features.get("attention_kurtosis_mean", 0.0))
        mlp_kurt = abs(features.get("mlp_kurtosis_mean", 0.0))
        avg_kurtosis = (attn_kurt + mlp_kurt) / 2.0
        risk_signals["heavy_tailed_deltas"] = min(
            avg_kurtosis / thresholds["heavy_tailed_deltas"], 2.0
        )

        # Signal 5: Overall magnitude
        frob = features.get("global_frob_mean", 0.0)
        risk_signals["high_magnitude"] = min(
            frob / thresholds["high_magnitude"], 2.0
        )

        # Signal 6: Stealth pattern — high kurtosis combined with low magnitude
        # Adversarial fine-tuning: sparse targeted changes that barely move weights
        stealth_ratio = avg_kurtosis / (frob * 1000.0 + 1.0)
        risk_signals["stealth_kurtosis_magnitude"] = min(
            stealth_ratio / thresholds["stealth_kurtosis_magnitude"], 2.0
        )

        # Signal 7: Layer L2 Gini — concentrated changes across layers
        layer_gini = features.get("layer_l2_gini", 0.0)
        risk_signals["layer_l2_gini"] = min(
            layer_gini / thresholds["layer_l2_gini"], 2.0
        )

        # Signal 8: Cosine divergence — broad weight divergence from base model
        # Floor at 0.015 divergence (cosine_min > 0.985 → zero contribution)
        # to ignore normal domain-adaptation fine-tuning
        if "cosine_divergence" in thresholds:
            cosine_min = features.get("global_cosine_min", 1.0)
            cosine_div = max(0.0, (1.0 - cosine_min) - 0.015)
            risk_signals["cosine_divergence"] = min(
                cosine_div / thresholds["cosine_divergence"], 2.0
            )

        # Signal 9 (LoRA only): Component L2 Gini — within-component concentration
        if is_lora:
            attn_gini = features.get("attention_l2_gini", 0.0)
            mlp_gini = features.get("mlp_l2_gini", 0.0)
            component_gini = (attn_gini + mlp_gini) / 2.0
            risk_signals["component_l2_gini"] = min(
                component_gini / thresholds["component_l2_gini"], 2.0
            )

        # Weighted risk score
        score = sum(
            weights[k] * risk_signals[k] for k in weights
        )

        # Classification
        if score >= 0.85:
            label = "harmful"
            confidence = min(0.5 + (score - 0.85) * 0.5, 0.90)
        elif score >= 0.55:
            label = "uncertain"
            confidence = 0.3 + (score - 0.55) * 0.67
        else:
            label = "benign"
            confidence = min(0.5 + (0.55 - score) * 0.65, 0.85)

        # Build explanation
        top_signals = sorted(risk_signals.items(), key=lambda x: x[1], reverse=True)
        mode = "LoRA" if is_lora else "full fine-tuning"
        explanation_parts = [f"Threshold analysis (score={score:.3f}, mode={mode}):"]
        for name, val in top_signals:
            status = "TRIGGERED" if val >= 1.0 else "below threshold"
            explanation_parts.append(f"  - {name}: {val:.3f} ({status})")

        return ClassificationResult(
            label=label,
            confidence=confidence,
            phase=1,
            phase_name="threshold_heuristics",
            risk_signals=risk_signals,
            explanation="\n".join(explanation_parts),
        )


class CalibratedClassifier:
    """Phase 2: Calibrated comparison against known jailbreak profiles.

    Uses cosine similarity between feature vectors and stored profiles.
    """

    def __init__(self, profiles_path: Path | None = None):
        self.profiles: list[dict[str, Any]] = []
        if profiles_path and profiles_path.exists():
            with open(profiles_path) as f:
                self.profiles = json.load(f)

    def add_profile(self, features: dict[str, float], label: str):
        self.profiles.append({"features": features, "label": label})

    def save_profiles(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def classify(self, features: dict[str, float]) -> ClassificationResult | None:
        if not self.profiles:
            return None

        harmful_profiles = [p for p in self.profiles if p["label"] == "harmful"]
        benign_profiles = [p for p in self.profiles if p["label"] == "benign"]

        if not harmful_profiles:
            return None

        # Compute cosine similarity to each profile
        feat_vec = self._to_vector(features)

        harmful_sims = []
        for p in harmful_profiles:
            pvec = self._to_vector(p["features"])
            sim = self._cosine_sim(feat_vec, pvec)
            harmful_sims.append(sim)

        benign_sims = []
        for p in benign_profiles:
            pvec = self._to_vector(p["features"])
            sim = self._cosine_sim(feat_vec, pvec)
            benign_sims.append(sim)

        max_harmful_sim = max(harmful_sims)
        max_benign_sim = max(benign_sims) if benign_sims else 0.0

        if max_harmful_sim > max_benign_sim and max_harmful_sim > 0.8:
            label = "harmful"
            confidence = min(max_harmful_sim, 0.9)
        elif max_benign_sim > max_harmful_sim and max_benign_sim > 0.8:
            label = "benign"
            confidence = min(max_benign_sim, 0.9)
        else:
            label = "uncertain"
            confidence = 0.4

        return ClassificationResult(
            label=label,
            confidence=confidence,
            phase=2,
            phase_name="calibrated_comparison",
            risk_signals={
                "max_harmful_similarity": max_harmful_sim,
                "max_benign_similarity": max_benign_sim,
            },
            explanation=(
                f"Calibrated comparison: max similarity to harmful profile = {max_harmful_sim:.3f}, "
                f"max similarity to benign profile = {max_benign_sim:.3f}"
            ),
        )

    def _to_vector(self, features: dict[str, float]) -> np.ndarray:
        # Use sorted keys for consistent ordering
        keys = sorted(features.keys())
        return np.array([features.get(k, 0.0) for k in keys], dtype=np.float64)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class AnomalyClassifier:
    """Phase 3: Anomaly detection (Isolation Forest + One-Class SVM)."""

    def __init__(self, model_path: Path | None = None):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        self.oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)
        self.fitted = False
        self.feature_keys: list[str] = []

        if model_path and model_path.exists():
            self._load(model_path)

    def fit(self, feature_dicts: list[dict[str, float]]):
        if len(feature_dicts) < 5:
            return

        self.feature_keys = sorted(feature_dicts[0].keys())
        X = np.array([[d.get(k, 0.0) for k in self.feature_keys] for d in feature_dicts])

        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.oc_svm.fit(X_scaled)
        self.fitted = True

    def classify(self, features: dict[str, float]) -> ClassificationResult | None:
        if not self.fitted:
            return None

        x = np.array([[features.get(k, 0.0) for k in self.feature_keys]])
        x_scaled = self.scaler.transform(x)

        iso_pred = self.iso_forest.predict(x_scaled)[0]  # 1 = inlier, -1 = outlier
        iso_score = self.iso_forest.score_samples(x_scaled)[0]

        svm_pred = self.oc_svm.predict(x_scaled)[0]

        # Combine: if both flag anomaly, high confidence
        is_anomaly = (iso_pred == -1) or (svm_pred == -1)
        both_anomaly = (iso_pred == -1) and (svm_pred == -1)

        if both_anomaly:
            label = "harmful"
            confidence = min(0.6 + abs(iso_score) * 0.2, 0.85)
        elif is_anomaly:
            label = "uncertain"
            confidence = 0.5
        else:
            label = "benign"
            confidence = min(0.5 + abs(iso_score) * 0.2, 0.85)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            phase=3,
            phase_name="anomaly_detection",
            risk_signals={
                "isolation_forest_score": float(iso_score),
                "iso_forest_anomaly": float(iso_pred == -1),
                "oc_svm_anomaly": float(svm_pred == -1),
            },
            explanation=(
                f"Anomaly detection: IsoForest={'anomaly' if iso_pred == -1 else 'normal'} "
                f"(score={iso_score:.3f}), "
                f"OC-SVM={'anomaly' if svm_pred == -1 else 'normal'}"
            ),
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "iso_forest": self.iso_forest,
                    "oc_svm": self.oc_svm,
                    "feature_keys": self.feature_keys,
                    "fitted": self.fitted,
                },
                f,
            )

    def _load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.iso_forest = data["iso_forest"]
        self.oc_svm = data["oc_svm"]
        self.feature_keys = data["feature_keys"]
        self.fitted = data["fitted"]


class SupervisedClassifier:
    """Phase 4: Random Forest classifier on reduced features."""

    def __init__(self, model_path: Path | None = None):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_keys: list[str] = []

        if model_path and model_path.exists():
            self._load(model_path)

    def fit(self, feature_dicts: list[dict[str, float]], labels: list[str]):
        if len(feature_dicts) < 20:
            return

        self.feature_keys = sorted(feature_dicts[0].keys())
        X = np.array([[d.get(k, 0.0) for k in self.feature_keys] for d in feature_dicts])
        y = np.array([1 if l == "harmful" else 0 for l in labels])

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def classify(self, features: dict[str, float]) -> ClassificationResult | None:
        if not self.fitted:
            return None

        x = np.array([[features.get(k, 0.0) for k in self.feature_keys]])
        x_scaled = self.scaler.transform(x)

        pred = self.model.predict(x_scaled)[0]
        proba = self.model.predict_proba(x_scaled)[0]

        label = "harmful" if pred == 1 else "benign"
        confidence = float(max(proba))

        # Feature importances
        importances = dict(zip(self.feature_keys, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

        return ClassificationResult(
            label=label,
            confidence=confidence,
            phase=4,
            phase_name="supervised_classifier",
            risk_signals={k: float(v) for k, v in top_features},
            explanation=(
                f"Supervised RF: predicted={label} with confidence={confidence:.3f}. "
                f"Top features: {', '.join(f'{k}={v:.3f}' for k, v in top_features[:5])}"
            ),
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_keys": self.feature_keys,
                    "fitted": self.fitted,
                },
                f,
            )

    def _load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_keys = data["feature_keys"]
        self.fitted = data["fitted"]


class ClassifierPipeline:
    """Auto-selecting classifier that uses the best available phase.

    Automatically selects the highest-phase classifier that has sufficient data.
    """

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir
        self.threshold = ThresholdClassifier()

        profiles_path = models_dir / "profiles.json" if models_dir else None
        self.calibrated = CalibratedClassifier(profiles_path)

        anomaly_path = models_dir / "anomaly_model.pkl" if models_dir else None
        self.anomaly = AnomalyClassifier(anomaly_path)

        supervised_path = models_dir / "supervised_model.pkl" if models_dir else None
        self.supervised = SupervisedClassifier(supervised_path)

    def classify(self, feature_set: FeatureSet) -> ClassificationResult:
        """Classify using the best available phase."""
        features = feature_set.reduced_features
        is_lora = feature_set.is_lora

        # Try phases in reverse order (highest first)
        result = self.supervised.classify(features)
        if result is not None:
            return result

        result = self.anomaly.classify(features)
        if result is not None:
            return result

        result = self.calibrated.classify(features)
        if result is not None:
            return result

        # Always available — pass is_lora for threshold calibration
        return self.threshold.classify(features, is_lora=is_lora)
