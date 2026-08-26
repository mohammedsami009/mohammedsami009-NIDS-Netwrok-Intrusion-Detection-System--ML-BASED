"""
test_csv.py
===========

Offline evaluation of the trained Random Forest or XGBoost model
on an original CICIDS2017 CSV flow file.

Pipeline:
    Raw CICIDS2017 CSV
        ↓
    normalize column names
        ↓
    load exact feature order from models/feature_names.pkl
        ↓
    select the same 30 features used during training
        ↓
    numeric conversion
        ↓
    infinity -> NaN
        ↓
    median imputation
        ↓
    model prediction
        ↓
    Accuracy / Precision / Recall / F1 / ROC-AUC

IMPORTANT:
    This script evaluates CICIDS2017 CSV flow records.

    It does NOT test the Scapy PCAP feature-extraction pipeline.

Usage from src/:

    python test_csv.py --model xgboost --csv "..\\data\\raw\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

or:

    python test_csv.py --model random_forest --csv "..\\data\\raw\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

log = logging.getLogger("nids.test_csv")


# =============================================================================
# LOAD FEATURE METADATA
# =============================================================================

def load_feature_names():
    """
    Load the exact feature names and order saved during model training.

    This avoids maintaining a second hard-coded feature list in this script.
    """

    metadata_path = (
        MODELS_DIR / "feature_names.pkl"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Feature metadata was not found:\n"
            f"{metadata_path}\n\n"
            "Create it before running this test."
        )

    metadata = joblib.load(
        metadata_path
    )

    # Expected format created for this project:
    #
    # {
    #     "feature_names": [...],
    #     "feature_count": 30,
    #     "label_mapping": {
    #         "BENIGN": 0,
    #         "ATTACK": 1
    #     }
    # }

    if isinstance(metadata, dict):
        features = metadata.get(
            "feature_names"
        )
    else:
        # Also support a plain list for robustness.
        features = metadata

    if not features:
        raise ValueError(
            "feature_names.pkl does not contain "
            "a valid feature_names list."
        )

    features = list(features)

    log.info(
        "Loaded feature metadata: %d features",
        len(features),
    )

    return features


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def clean_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize CICIDS2017/CICFlowMeter column names.

    Example:
        Total Backward Packets
            -> Total_Backward_Packets
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(
            "\ufeff",
            "",
            regex=False,
        )
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False,
        )
    )

    return df


# =============================================================================
# NUMERIC CLEANING
# =============================================================================

def clean_numeric_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert features to numeric and reproduce the cleaning strategy used
    when creating selected_features.csv.

    Invalid values:
        strings -> NaN
        +/- infinity -> NaN
        NaN -> column median
        completely invalid column -> 0
    """

    X = X.copy()

    # Convert every feature to numeric.
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace infinity with NaN.
    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # Use the median of THIS CSV for missing values.
    #
    # This mirrors the cleaning approach used by data_preprocessing.py.
    for column in X.columns:

        if X[column].isna().any():

            median_value = (
                X[column].median()
            )

            if pd.isna(median_value):
                median_value = 0

            X[column] = (
                X[column]
                .fillna(median_value)
            )

    # Final safety fill.
    X.fillna(
        0,
        inplace=True,
    )

    if X.isna().sum().sum() != 0:
        raise ValueError(
            "NaN values remain after cleaning."
        )

    if not np.isfinite(
        X.to_numpy()
    ).all():
        raise ValueError(
            "Infinite values remain after cleaning."
        )

    return X


# =============================================================================
# LABEL CONVERSION
# =============================================================================

def create_binary_labels(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Convert CICIDS2017 Label values to:

        BENIGN -> 0
        anything else -> 1 (ATTACK)
    """

    if "Label" not in df.columns:
        raise ValueError(
            "CSV does not contain a 'Label' column."
        )

    labels = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return (
        labels
        .ne("BENIGN")
        .astype(int)
        .to_numpy()
    )


# =============================================================================
# METRICS
# =============================================================================

def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability=None,
) -> None:
    """
    Print evaluation metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("CICIDS2017 MODEL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.6f} "
        f"({accuracy * 100:.4f}%)"
    )

    print(
        f"Precision: {precision:.6f} "
        f"({precision * 100:.4f}%)"
    )

    print(
        f"Recall   : {recall:.6f} "
        f"({recall * 100:.4f}%)"
    )

    print(
        f"F1-score : {f1:.6f} "
        f"({f1 * 100:.4f}%)"
    )

    # ROC-AUC requires both classes to exist.
    if y_probability is not None:

        try:

            roc_auc = roc_auc_score(
                y_true,
                y_probability,
            )

            print(
                f"ROC-AUC  : {roc_auc:.6f} "
                f"({roc_auc * 100:.4f}%)"
            )

        except ValueError:

            print(
                "ROC-AUC  : unavailable "
                "(only one class is present)"
            )

    # -------------------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        "                 Predicted"
    )

    print(
        "                 BENIGN  ATTACK"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print(
        f"Actual BENIGN    "
        f"{cm[0, 0]:7d} "
        f"{cm[0, 1]:7d}"
    )

    print(
        f"Actual ATTACK    "
        f"{cm[1, 0]:7d} "
        f"{cm[1, 1]:7d}"
    )

    # -------------------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK",
            ],
            zero_division=0,
        )
    )


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def test_model(
    model_path: str,
    csv_path: str,
) -> None:
    """
    Test a trained model on one CICIDS2017 CSV file.
    """

    model_path = Path(
        model_path
    )

    csv_path = Path(
        csv_path
    )

    # -------------------------------------------------------------------------
    # Validate paths
    # -------------------------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    log.info(
        "Loading model: %s",
        model_path,
    )

    model = joblib.load(
        model_path
    )

    log.info(
        "Model loaded: %s",
        type(model).__name__,
    )

    # -------------------------------------------------------------------------
    # Load exact feature metadata
    # -------------------------------------------------------------------------

    features = load_feature_names()

    # -------------------------------------------------------------------------
    # Check model feature count
    # -------------------------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_",
    ):

        expected_features = int(
            model.n_features_in_
        )

        if expected_features != len(
            features
        ):

            raise ValueError(
                f"Model expects "
                f"{expected_features} features, "
                f"but feature_names.pkl contains "
                f"{len(features)} features."
            )

    # -------------------------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------------------------

    log.info(
        "Loading CICIDS2017 CSV: %s",
        csv_path,
    )

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    log.info(
        "Original dataset shape: %s",
        df.shape,
    )

    # -------------------------------------------------------------------------
    # Normalize column names
    # -------------------------------------------------------------------------

    df = clean_column_names(
        df
    )

    # -------------------------------------------------------------------------
    # Required-column check
    # -------------------------------------------------------------------------

    required_columns = (
        features + ["Label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The CSV does not contain the required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # -------------------------------------------------------------------------
    # Remove rows with missing labels
    # -------------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=["Label"]
    ).copy()

    log.info(
        "Rows after removing missing labels: %d "
        "(removed %d)",
        len(df),
        before - len(df),
    )

    # -------------------------------------------------------------------------
    # True labels
    # -------------------------------------------------------------------------

    y_true = create_binary_labels(
        df
    )

    benign_count = int(
        np.sum(y_true == 0)
    )

    attack_count = int(
        np.sum(y_true == 1)
    )

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(
        f"CSV file       : {csv_path.name}"
    )

    print(
        f"Total rows     : {len(df):,}"
    )

    print(
        f"BENIGN rows    : {benign_count:,}"
    )

    print(
        f"ATTACK rows    : {attack_count:,}"
    )

    if len(df) > 0:

        print(
            f"Attack ratio   : "
            f"{attack_count / len(df) * 100:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Prepare exact features
    # -------------------------------------------------------------------------

    X = df[
        features
    ].copy()

    X = clean_numeric_features(
        X
    )

    # Explicitly enforce exact feature order.
    X = X[
        features
    ]

    print("\nFeatures passed to model:")

    for index, feature in enumerate(
        features,
        start=1,
    ):

        print(
            f"{index:2d}. {feature}"
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    log.info(
        "Running model prediction on %s rows...",
        f"{len(X):,}",
    )

    y_pred = model.predict(
        X
    ).astype(int)

    # -------------------------------------------------------------------------
    # Probability / confidence
    # -------------------------------------------------------------------------

    attack_probability = None

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(X)
        )

        if probabilities.ndim == 2:

            if probabilities.shape[1] >= 2:

                attack_probability = (
                    probabilities[:, 1]
                )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    print_metrics(
        y_true,
        y_pred,
        attack_probability,
    )

    # -------------------------------------------------------------------------
    # Prediction distribution
    # -------------------------------------------------------------------------

    predicted_benign = int(
        np.sum(y_pred == 0)
    )

    predicted_attack = int(
        np.sum(y_pred == 1)
    )

    print("=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(
        f"Predicted BENIGN: "
        f"{predicted_benign:,}"
    )

    print(
        f"Predicted ATTACK: "
        f"{predicted_attack:,}"
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    output_name = (
        f"{csv_path.stem}_"
        f"{model_path.stem}_predictions.csv"
    )

    output_path = (
        RESULTS_DIR /
        output_name
    )

    results = pd.DataFrame(
        {
            "actual_label": y_true,
            "predicted_label": y_pred,
        }
    )

    results["actual_label"] = (
        results["actual_label"]
        .map(
            {
                0: "BENIGN",
                1: "ATTACK",
            }
        )
    )

    results["predicted_label"] = (
        results["predicted_label"]
        .map(
            {
                0: "BENIGN",
                1: "ATTACK",
            }
        )
    )

    if attack_probability is not None:

        results[
            "attack_probability"
        ] = attack_probability

    results.to_csv(
        output_path,
        index=False,
    )

    log.info(
        "Prediction results saved to: %s",
        output_path,
    )

    print(
        f"\nPrediction file: {output_path}"
    )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test the trained NIDS-ML Random Forest "
            "or XGBoost model on a CICIDS2017 CSV."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "xgboost",
            "random_forest",
        ],
        default="xgboost",
        help="Model to test.",
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CICIDS2017 CSV file.",
    )

    args = parser.parse_args()

    model_path = (
        MODELS_DIR /
        f"{args.model}.pkl"
    )

    test_model(
        model_path=str(
            model_path
        ),
        csv_path=args.csv,
    )


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()