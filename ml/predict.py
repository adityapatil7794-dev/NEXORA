# ml/predict.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "ml" / "flood_model.pkl"

logger = logging.getLogger("nexora.ml.predict")


# ============================================================
# MODEL LOADER
# ============================================================

class FloodModel:
    """
    Production inference wrapper for the NEXORA flood model.

    Responsibilities:
    - Load the trained model once
    - Validate incoming features
    - Generate calibrated flood probability
    - Return structured prediction results
    """

    def __init__(self, model_path: Path = MODEL_FILE) -> None:
        self.model_path = model_path
        self.model: Any = None
        self.features: list[str] = []

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Flood model not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        if not isinstance(artifact, dict):
            raise ValueError(
                "Invalid flood model artifact. "
                "Expected a dictionary containing model and features."
            )

        if "model" not in artifact:
            raise ValueError("Model artifact does not contain 'model'.")

        if "features" not in artifact:
            raise ValueError("Model artifact does not contain 'features'.")

        self.model = artifact["model"]
        self.features = list(artifact["features"])

        if not self.features:
            raise ValueError("Model feature list is empty.")

        logger.info(
            "Flood model loaded successfully | features=%s",
            self.features,
        )


# ============================================================
# SINGLETON MODEL
# ============================================================

_flood_model = FloodModel()


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_features(features: dict[str, float]) -> None:
    """
    Validate that all model features are available and numeric.
    """

    missing = [
        feature
        for feature in _flood_model.features
        if feature not in features
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    invalid = []

    for feature in _flood_model.features:
        value = features[feature]

        if value is None:
            invalid.append(feature)
            continue

        if not isinstance(value, (int, float, np.integer, np.floating)):
            invalid.append(feature)
            continue

        if not np.isfinite(float(value)):
            invalid.append(feature)

    if invalid:
        raise ValueError(
            f"Invalid numeric values for features: {invalid}"
        )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def _risk_level(probability: float) -> str:
    """
    Converts calibrated probability into a presentation category.

    IMPORTANT:
    This is a communication category, not a scientific
    replacement for the calibrated probability.
    """

    if probability < 0.20:
        return "LOW"

    if probability < 0.40:
        return "MODERATE"

    if probability < 0.60:
        return "ELEVATED"

    if probability < 0.80:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_flood(
    *,
    latitude: float,
    longitude: float,
    temperature_mean: float,
    temperature_max: float,
    temperature_min: float,
    precipitation_sum: float,
    precipitation_mean: float,
    precipitation_max: float,
    precipitation_p95: float,
    heavy_rain_days: float,
    extreme_rain_days: float,
) -> dict[str, Any]:

    features = {
        "temperature_mean": float(temperature_mean),
        "temperature_max": float(temperature_max),
        "temperature_min": float(temperature_min),
        "precipitation_sum": float(precipitation_sum),
        "precipitation_mean": float(precipitation_mean),
        "precipitation_max": float(precipitation_max),
        "precipitation_p95": float(precipitation_p95),
        "heavy_rain_days": float(heavy_rain_days),
        "extreme_rain_days": float(extreme_rain_days),
    }

    _validate_features(features)

    # Keep exact training feature order.
    input_frame = pd.DataFrame(
        [[features[name] for name in _flood_model.features]],
        columns=_flood_model.features,
    )

    # Calibrated probability of flood class = 1.
    probabilities = _flood_model.model.predict_proba(input_frame)

    flood_probability = float(probabilities[0][1])

    prediction = int(
        _flood_model.model.predict(input_frame)[0]
    )

    risk = _risk_level(flood_probability)

    return {
    "latitude": float(latitude),
    "longitude": float(longitude),

    "prediction": prediction,

    # Primary NEXORA probability fields
    "probability": round(
        flood_probability,
        6,
    ),

    "probability_percent": round(
        flood_probability * 100,
        2,
    ),

    # Backward-compatible field names
    "flood_probability": round(
        flood_probability,
        6,
    ),

    "flood_probability_percent": round(
        flood_probability * 100,
        2,
    ),

    "risk_level": risk,

    "features": features,

    # Backward-compatible feature name
    "model_features": features,

    "model_feature_order": _flood_model.features,
}
# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> dict[str, Any]:
    """
    Returns model metadata useful for API diagnostics.
    """

    return {
        "model": "NEXORA Flood Random Forest",
        "model_file": str(_flood_model.model_path),
        "features": _flood_model.features,
        "feature_count": len(_flood_model.features),
        "status": "loaded",
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    result = predict_flood(
        latitude=16.723370,
        longitude=74.543335,

        temperature_mean=25.11,
        temperature_max=40.8,
        temperature_min=13.0,

        precipitation_sum=1241.9,
        precipitation_mean=3.40,
        precipitation_max=67.5,

        precipitation_p95=14.83,
        heavy_rain_days=3,
        extreme_rain_days=0,
    )

    print("\n==========================================")
    print("       NEXORA FLOOD ML PREDICTION")
    print("==========================================")

    print(
        f"Flood probability : "
        f"{result['flood_probability_percent']}%"
    )

    print(
        f"Risk level        : "
        f"{result['risk_level']}"
    )

    print(
        f"Prediction        : "
        f"{result['prediction']}"
    )

    print("\nModel features:")

    for name, value in result["model_features"].items():
        print(f"  {name}: {value}")

    print("\n==========================================")