
# NIDS-ML — Random Forest + XGBoost Training
# =============================================================================
# This module:
#
#   1. Loads the preprocessed CICIDS2017 dataset
#   2. Separates features and labels
#   3. Performs an 80/20 stratified train-test split
#   4. Trains ONLY:
#        - Random Forest
#        - XGBoost
#   5. Evaluates both models using:
#        - Accuracy
#        - Precision
#        - Recall
#        - F1-Score
#        - ROC-AUC
#   6. Generates:
#        - Confusion matrices
#        - ROC curves
#        - Model comparison CSV
#   7. Saves both trained models
#   8. Saves the better-performing model as best_model.pkl
#
# Dataset:
#   data/processed/selected_features.csv
#
# Expected label:
#   Label / label / Attack / attack / Class / class
#
# =============================================================================

import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

# Maximum number of samples used for model training. The full test set is kept.
MAX_TRAIN_SAMPLES = 500_000

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

import xgboost as xgb


# =============================================================================
# Configuration
# =============================================================================

warnings.filterwarnings("ignore")


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "selected_features.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


# Create required directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / "model_training.log",
            encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — Load Dataset
# =============================================================================

def load_data():
    """
    Load the processed CICIDS2017 dataset.

    Returns:
        X : DataFrame
            Input features.

        y : Series
            Binary target:
                0 = Benign
                1 = Attack
    """

    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATASET")
    logger.info("=" * 70)

    logger.info(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")

    # -------------------------------------------------------------------------
    # Find label column
    # -------------------------------------------------------------------------

    possible_labels = [
        "Label",
        "label",
        "Attack",
        "attack",
        "Class",
        "class",
    ]

    label_column = None

    for column in possible_labels:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError(
            f"No label column found.\n"
            f"Expected one of: {possible_labels}\n"
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Label column: {label_column}")

    # -------------------------------------------------------------------------
    # Separate features and target
    # -------------------------------------------------------------------------

    y = df[label_column]
    X = df.drop(columns=[label_column])

    # -------------------------------------------------------------------------
    # Convert labels to binary if necessary
    #
    # Benign/Normal -> 0
    # Everything else -> 1
    # -------------------------------------------------------------------------

    if y.dtype == "object":

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
            .apply(
                lambda value: 0
                if value in ["benign", "normal", "0"]
                else 1
            )
        )

    else:

        # Convert numeric labels to integers
        y = pd.to_numeric(y, errors="coerce")

        if y.isna().any():
            raise ValueError(
                "The label column contains values that cannot be converted to numbers."
            )

        # Make sure labels are binary
        unique_labels = sorted(y.unique())

        if not set(unique_labels).issubset({0, 1}):

            logger.info(
                f"Converting labels {unique_labels} to binary..."
            )

            y = y.apply(
                lambda value: 0
                if value == 0
                else 1
            )

        y = y.astype(int)

    # -------------------------------------------------------------------------
    # Clean feature values
    # -------------------------------------------------------------------------

    # Replace infinity values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convert columns to numeric where possible
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Fill missing values with column median
    X = X.fillna(X.median(numeric_only=True))

    # If any columns are still completely unusable, fill with 0
    X = X.fillna(0)

    # -------------------------------------------------------------------------
    # Final checks
    # -------------------------------------------------------------------------

    if X.isnull().sum().sum() != 0:
        raise ValueError("NaN values still exist in feature dataset.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite values still exist in feature dataset."
        )

    logger.info("")
    logger.info("Feature shape:")
    logger.info(f"X = {X.shape}")

    logger.info("")
    logger.info("Label distribution:")

    label_counts = y.value_counts().sort_index()

    for label, count in label_counts.items():

        name = "BENIGN" if label == 0 else "ATTACK"

        percentage = (count / len(y)) * 100

        logger.info(
            f"{name} ({label}): "
            f"{count:,} samples "
            f"({percentage:.2f}%)"
        )

    logger.info("")
    logger.info("Features used by the models:")

    for column in X.columns:
        logger.info(f"  - {column}")

    logger.info("-" * 70)

    return X, y


# =============================================================================
# STEP 2 — Train Random Forest
# =============================================================================

def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING RANDOM FOREST")
    logger.info("=" * 70)

    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    logger.info(
        f"Random Forest training completed in "
        f"{training_time:.2f} seconds"
    )

    return model, training_time


# =============================================================================
# STEP 3 — Train XGBoost
# =============================================================================

def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING XGBOOST")
    logger.info("=" * 70)

    start_time = time.time()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    logger.info(
        f"XGBoost training completed in "
        f"{training_time:.2f} seconds"
    )

    return model, training_time


# =============================================================================
# STEP 4 — Evaluate Model
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model.

    Metrics:
        Accuracy
        Precision
        Recall
        F1-Score
        ROC-AUC

    Also generates:
        Confusion Matrix
        ROC Curve
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"EVALUATING {model_name.upper()}")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    y_pred = model.predict(X_test)

    # Probability of attack
    y_probability = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1-Score : {f1:.4f}")
    logger.info(f"ROC-AUC  : {roc_auc:.4f}")

    # -------------------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Attack"],
        zero_division=0,
    )

    logger.info("")
    logger.info("Classification Report:")
    logger.info("")
    logger.info(report)

    # -------------------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    filename = (
        "confusion_matrix_"
        + model_name.lower().replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        f"Saved confusion matrix: {filename}"
    )

    # -------------------------------------------------------------------------
    # ROC Curve
    # -------------------------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_value = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_value:.4f})",
        linewidth=2,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        "roc_curve_"
        + model_name.lower().replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        f"Saved ROC curve: {filename}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# =============================================================================
# STEP 5 — Compare Models
# =============================================================================

def compare_models(results):
    """
    Create a comparison table for Random Forest and XGBoost.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    rows = []

    for model_name, result in results.items():

        rows.append({
            "Model": model_name,
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-Score": result["f1_score"],
            "ROC-AUC": result["roc_auc"],
            "Training Time (s)": result["training_time"],
        })

    comparison = pd.DataFrame(rows)

    # Sort by F1 because IDS datasets can be imbalanced.
    comparison = comparison.sort_values(
        by="F1-Score",
        ascending=False
    ).reset_index(drop=True)

    logger.info("")
    logger.info(
        comparison.to_string(index=False)
    )

    # Save CSV
    comparison_path = (
        RESULTS_DIR / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    logger.info("")
    logger.info(
        f"Saved comparison table: {comparison_path}"
    )

    # -------------------------------------------------------------------------
    # Comparison graph
    # -------------------------------------------------------------------------

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]

    x = np.arange(
        len(comparison["Model"])
    )

    width = 0.15

    plt.figure(figsize=(11, 6))

    for i, metric in enumerate(metrics):

        plt.bar(
            x + (i - 2) * width,
            comparison[metric],
            width,
            label=metric,
        )

    plt.xticks(
        x,
        comparison["Model"]
    )

    plt.ylabel("Score")
    plt.xlabel("Model")

    plt.title(
        "Random Forest vs XGBoost Performance"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    comparison_plot = (
        PLOTS_DIR / "model_comparison.png"
    )

    plt.savefig(
        comparison_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        f"Saved comparison plot: {comparison_plot}"
    )

    return comparison


# =============================================================================
# STEP 6 — Save Models
# =============================================================================

def save_models(
    random_forest,
    xgboost_model,
    results,
    comparison,
):
    """
    Save both trained models.

    Also saves the model with the highest F1-Score as:
        models/best_model.pkl
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("SAVING MODELS")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Save Random Forest
    # -------------------------------------------------------------------------

    rf_path = (
        MODELS_DIR / "random_forest.pkl"
    )

    joblib.dump(
        random_forest,
        rf_path
    )

    logger.info(
        f"Random Forest saved: {rf_path}"
    )

    # -------------------------------------------------------------------------
    # Save XGBoost
    # -------------------------------------------------------------------------

    xgb_path = (
        MODELS_DIR / "xgboost.pkl"
    )

    joblib.dump(
        xgboost_model,
        xgb_path
    )

    logger.info(
        f"XGBoost saved: {xgb_path}"
    )

    # -------------------------------------------------------------------------
    # Select best model using F1-Score
    # -------------------------------------------------------------------------

    best_model_name = comparison.iloc[0]["Model"]

    if best_model_name == "Random Forest":

        best_model = random_forest

    else:

        best_model = xgboost_model

    best_model_path = (
        MODELS_DIR / "best_model.pkl"
    )

    joblib.dump(
        best_model,
        best_model_path
    )

    logger.info(
        f"Best model: {best_model_name}"
    )

    logger.info(
        f"Best model saved: {best_model_path}"
    )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    best_row = comparison.iloc[0]

    metadata_path = (
        MODELS_DIR / "best_model_metadata.txt"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "NIDS MODEL METADATA\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Model: {best_model_name}\n"
        )

        file.write(
            f"Accuracy: {best_row['Accuracy']:.6f}\n"
        )

        file.write(
            f"Precision: {best_row['Precision']:.6f}\n"
        )

        file.write(
            f"Recall: {best_row['Recall']:.6f}\n"
        )

        file.write(
            f"F1-Score: {best_row['F1-Score']:.6f}\n"
        )

        file.write(
            f"ROC-AUC: {best_row['ROC-AUC']:.6f}\n"
        )

        file.write(
            f"Training Time: "
            f"{best_row['Training Time (s)']:.2f} seconds\n"
        )

        file.write(
            f"Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    logger.info(
        f"Metadata saved: {metadata_path}"
    )


# =============================================================================
# STEP 7 — Save Metrics Report
# =============================================================================

def save_metrics_report(results):
    """
    Save detailed evaluation results for both models.
    """

    report_path = (
        RESULTS_DIR / "model_metrics.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "NIDS — RANDOM FOREST AND XGBOOST EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            f"Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        for model_name, result in results.items():

            file.write(
                "=" * 70 + "\n"
            )

            file.write(
                f"MODEL: {model_name}\n"
            )

            file.write(
                "=" * 70 + "\n\n"
            )

            file.write(
                f"Accuracy : {result['accuracy']:.6f}\n"
            )

            file.write(
                f"Precision: {result['precision']:.6f}\n"
            )

            file.write(
                f"Recall   : {result['recall']:.6f}\n"
            )

            file.write(
                f"F1-Score : {result['f1_score']:.6f}\n"
            )

            file.write(
                f"ROC-AUC  : {result['roc_auc']:.6f}\n"
            )

            file.write(
                f"Training Time: "
                f"{result['training_time']:.2f} seconds\n\n"
            )

            file.write(
                "Classification Report:\n"
            )

            file.write(
                result["classification_report"]
            )

            file.write(
                "\n\n"
            )

    logger.info(
        f"Metrics report saved: {report_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info("NIDS-ML — RANDOM FOREST + XGBOOST")
    logger.info("=" * 70)

    overall_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------

    X, y = load_data()

    # -------------------------------------------------------------------------
    # 2. Train/Test Split
    #
    # 80% -> Training
    # 20% -> Testing
    #
    # stratify=y keeps the benign/attack ratio approximately the same
    # in both datasets.
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: TRAIN / TEST SPLIT")
    logger.info("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logger.info(
        f"Training samples before sampling: {len(X_train):,}"
    )

    logger.info(
        f"Testing samples : {len(X_test):,}"
    )

    # -------------------------------------------------------------------------
    # 3. Create a practical training subset
    #
    # CICIDS2017 contains millions of rows. Training tree ensembles on the
    # entire training set is unnecessarily expensive on a typical laptop.
    # We use a stratified 500,000-row subset while keeping the COMPLETE test
    # set untouched. Stratification preserves the benign/attack ratio.
    # -------------------------------------------------------------------------

    if len(X_train) > MAX_TRAIN_SAMPLES:

        logger.info(
            f"Reducing training data to {MAX_TRAIN_SAMPLES:,} samples "
            "using stratified sampling..."
        )

        X_train_model, _, y_train_model, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=42,
            stratify=y_train,
        )

    else:

        X_train_model = X_train
        y_train_model = y_train

    logger.info(
        f"Actual samples used for model training: "
        f"{len(X_train_model):,}"
    )

    # -------------------------------------------------------------------------
    # 4. Train Random Forest
    # -------------------------------------------------------------------------

    random_forest, rf_time = train_random_forest(
        X_train_model,
        y_train_model
    )

    # -------------------------------------------------------------------------
    # 5. Train XGBoost
    # -------------------------------------------------------------------------

    xgboost_model, xgb_time = train_xgboost(
        X_train_model,
        y_train_model
    )

    # -------------------------------------------------------------------------
    # 6. Evaluate Random Forest
    # -------------------------------------------------------------------------

    rf_results = evaluate_model(
        random_forest,
        X_test,
        y_test,
        "Random Forest"
    )

    rf_results["training_time"] = rf_time

    # -------------------------------------------------------------------------
    # 7. Evaluate XGBoost
    # -------------------------------------------------------------------------

    xgb_results = evaluate_model(
        xgboost_model,
        X_test,
        y_test,
        "XGBoost"
    )

    xgb_results["training_time"] = xgb_time

    # -------------------------------------------------------------------------
    # 8. Store results
    # -------------------------------------------------------------------------

    results = {
        "Random Forest": rf_results,
        "XGBoost": xgb_results,
    }

    # -------------------------------------------------------------------------
    # 9. Compare models
    # -------------------------------------------------------------------------

    comparison = compare_models(
        results
    )

    # -------------------------------------------------------------------------
    # 10. Save models
    # -------------------------------------------------------------------------

    save_models(
        random_forest,
        xgboost_model,
        results,
        comparison,
    )

    # -------------------------------------------------------------------------
    # 11. Save metrics
    # -------------------------------------------------------------------------

    save_metrics_report(
        results
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    total_time = (
        time.time() - overall_start
    )

    best_model = comparison.iloc[0]["Model"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)

    logger.info(
        f"Total execution time: "
        f"{total_time:.2f} seconds"
    )

    logger.info(
        f"Models trained: 2"
    )

    logger.info(
        f"Models: Random Forest, XGBoost"
    )

    logger.info(
        f"Best model based on F1-Score: "
        f"{best_model}"
    )

    logger.info("")
    logger.info("Generated files:")

    logger.info(
        "  models/random_forest.pkl"
    )

    logger.info(
        "  models/xgboost.pkl"
    )

    logger.info(
        "  models/best_model.pkl"
    )

    logger.info(
        "  models/best_model_metadata.txt"
    )

    logger.info(
        "  results/model_metrics.txt"
    )

    logger.info(
        "  results/model_comparison.csv"
    )

    logger.info(
        "  results/plots/confusion_matrix_random_forest.png"
    )

    logger.info(
        "  results/plots/confusion_matrix_xgboost.png"
    )

    logger.info(
        "  results/plots/roc_curve_random_forest.png"
    )

    logger.info(
        "  results/plots/roc_curve_xgboost.png"
    )

    logger.info(
        "  results/plots/model_comparison.png"
    )

    logger.info("=" * 70)


# =============================================================================
# Program Entry Point
# =============================================================================

if __name__ == "__main__":
    main()