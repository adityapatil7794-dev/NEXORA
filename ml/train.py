# ============================================================
# NEXORA — FLOOD RISK ML TRAINING PIPELINE
# ============================================================
# Purpose:
#   Train a calibrated Random Forest model for flood probability.
#
# Input:
#   data/flood_training_dataset_v2.csv
#
# Output:
#   ml/flood_model.pkl
#   ml/model_metrics.json
#
# Important:
#   - Latitude/longitude are NOT predictive features.
#   - Flood outcome fields are NOT predictive features.
#   - Only climate-derived features are used for prediction.
#   - Probability is calibrated rather than treated as a raw score.
# ============================================================

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "flood_training_dataset_v2.csv"
MODEL_FILE = BASE_DIR / "ml" / "flood_model.pkl"
METRICS_FILE = BASE_DIR / "ml" / "model_metrics.json"


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Fast enough for your current dataset while still strong.
N_ESTIMATORS = 250

# Use all available CPU cores.
N_JOBS = -1

TARGET = "flood_label"


# ============================================================
# 3. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("nexora.ml")


# ============================================================
# 4. FEATURES
# ============================================================

# These are the features that the model is allowed to learn from.
#
# NOTE:
# latitude/longitude are deliberately excluded because the earlier
# model was relying heavily on geography instead of climate.
#
# severity/deaths/displaced/duration are also excluded because they
# describe the flood event outcome itself and would cause leakage.

FEATURES = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "precipitation_mean",
    "precipitation_max",
    "precipitation_p95",
    "heavy_rain_days",
    "extreme_rain_days",
]


# ============================================================
# 5. DATA VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the training dataset before ML training."""

    if df.empty:
        raise ValueError("Training dataset is empty.")

    required_columns = FEATURES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    labels = set(df[TARGET].astype(int).unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0 and 1. Found: {labels}"
        )

    if len(labels) != 2:
        raise ValueError(
            "Training requires both flood (1) and non-flood (0) samples."
        )

    logger.info("Dataset validation successful.")


# ============================================================
# 6. PREPARE FEATURES
# ============================================================

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare clean model input."""

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Convert everything to numeric.
    for column in FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace infinities.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median imputation.
    #
    # This is calculated from the complete training dataset here
    # because the current project does not use a sklearn Pipeline yet.
    # For production-scale deployment, this should be persisted as
    # part of a Pipeline.
    for column in FEATURES:
        median_value = X[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Feature '{column}' contains no usable values."
            )

        X[column] = X[column].fillna(median_value)

    return X, y


# ============================================================
# 7. TRAINING
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train the base Random Forest."""

    logger.info(
        "Training Random Forest with %d trees...",
        N_ESTIMATORS,
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info("Base model training completed.")

    return model


# ============================================================
# 8. PROBABILITY CALIBRATION
# ============================================================

def calibrate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    """
    Calibrate model probabilities.

    cv=5 is used because current scikit-learn does not support
    the old cv='prefit' approach used by older implementations.
    """

    logger.info("Calibrating probability estimates...")

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=5,
        n_jobs=N_JOBS,
    )

    calibrated_model.fit(
        X_train,
        y_train,
    )

    logger.info("Probability calibration completed.")

    return calibrated_model


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model performance."""

    logger.info("Evaluating model...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Non-Flood",
            "Flood",
        ],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "brier_score": round(float(brier), 6),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "test_samples": int(len(X_test)),
    }

    return metrics


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    base_model: RandomForestClassifier,
) -> list[dict]:
    """Extract feature importance from Random Forest."""

    importance = base_model.feature_importances_

    results = []

    for feature, value in zip(FEATURES, importance):
        results.append(
            {
                "feature": feature,
                "importance": round(
                    float(value),
                    6,
                ),
            }
        )

    results.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return results


# ============================================================
# 11. SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_names: list[str],
) -> None:
    """Save complete ML artifact."""

    artifact = {
        "model": model,
        "features": feature_names,
        "target": TARGET,
        "model_type": "Calibrated Random Forest",
        "version": "NEXORA-FLOOD-ML-1.0",
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
        compress=3,
    )

    logger.info(
        "Model saved: %s",
        MODEL_FILE,
    )


# ============================================================
# 12. SAVE METRICS
# ============================================================

def save_metrics(
    metrics: dict,
    feature_importance: list[dict],
    dataset_size: int,
    train_size: int,
    test_size: int,
) -> None:
    """Save model evaluation metadata."""

    output = {
        "project": "NEXORA",
        "model": "Calibrated Random Forest",
        "version": "NEXORA-FLOOD-ML-1.0",
        "dataset": str(DATA_FILE),
        "dataset_samples": dataset_size,
        "training_samples": train_size,
        "testing_samples": test_size,
        "features": FEATURES,
        "feature_importance": feature_importance,
        "metrics": metrics,
        "notes": [
            "Latitude and longitude were excluded from predictive features.",
            "Flood event outcome fields were excluded to prevent target leakage.",
            "Probability calibration uses 5-fold cross-validation.",
            "Training labels include spatial proxy non-flood samples.",
            "A non-flood label does not prove that a location is permanently flood-safe.",
        ],
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    logger.info(
        "Metrics saved: %s",
        METRICS_FILE,
    )


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main() -> None:

    start_time = time.perf_counter()

    print()
    print("=" * 64)
    print("             NEXORA FLOOD ML TRAINING")
    print("=" * 64)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    logger.info(
        "Loading dataset: %s",
        DATA_FILE,
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
    )

    print()
    print(f"Dataset samples : {len(df)}")
    print(f"Dataset columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(df)

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_features(df)

    print()
    print("Predictive features:")

    for feature in FEATURES:
        print(f"  - {feature}")

    print()
    print("Target distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # --------------------------------------------------------
    # Train base model
    # --------------------------------------------------------

    base_model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Calibrate probabilities
    # --------------------------------------------------------

    calibrated_model = calibrate_model(
        base_model,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate calibrated model
    # --------------------------------------------------------

    metrics = evaluate_model(
        calibrated_model,
        X_test,
        y_test,
    )

    print()
    print("=" * 64)
    print("                    MODEL EVALUATION")
    print("=" * 64)

    print(
        f"Accuracy    : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : {metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score    : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC     : {metrics['roc_auc']:.4f}"
    )

    print(
        f"Brier Score : {metrics['brier_score']:.4f}"
    )

    print()
    print("Classification Report:")

    report = metrics["classification_report"]

    for label in ["Non-Flood", "Flood"]:

        if label in report:

            row = report[label]

            print(
                f"{label:<12}"
                f" precision={row['precision']:.3f}"
                f" recall={row['recall']:.3f}"
                f" f1={row['f1-score']:.3f}"
            )

    print()
    print("Confusion Matrix:")

    for row in metrics["confusion_matrix"]:
        print(row)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = calculate_feature_importance(
        base_model,
    )

    print()
    print("=" * 64)
    print("                 FEATURE IMPORTANCE")
    print("=" * 64)

    for item in feature_importance:

        print(
            f"{item['feature']:<24}"
            f"{item['importance']:.4f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        calibrated_model,
        FEATURES,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics=metrics,
        feature_importance=feature_importance,
        dataset_size=len(df),
        train_size=len(X_train),
        test_size=len(X_test),
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 64)
    print("                  NEXORA ML COMPLETE")
    print("=" * 64)

    print()
    print(f"Model   : {MODEL_FILE}")
    print(f"Metrics : {METRICS_FILE}")
    print(f"Time    : {elapsed:.2f} seconds")

    print()
    print("The model is ready for the prediction API.")

    print()
    print("IMPORTANT:")
    print(
        "The predicted probability represents the model's "
        "calibrated estimate from the available training data."
    )

    print(
        "It must be combined with historical evidence, "
        "terrain, hydrology and other NEXORA modules before "
        "being presented as final site intelligence."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()