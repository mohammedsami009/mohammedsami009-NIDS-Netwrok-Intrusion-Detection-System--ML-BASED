"""
SHAP Explainability Module for NIDS-ML
======================================

Provides explainability for the trained Random Forest and XGBoost
models used in the Network Intrusion Detection System.

Features:
    - SHAP TreeExplainer for tree-based models
    - Global feature importance
    - SHAP summary plots
    - SHAP bar plots
    - Individual prediction explanations
    - Waterfall plots
    - Force plots
    - Dependence plots
    - Top contributing features
    - Dashboard-ready explanations
    - Save/load SHAP explainer

Models supported:
    - Random Forest
    - XGBoost

Dataset:
    data/processed/selected_features.csv

Models:
    models/random_forest.pkl
    models/xgboost.pkl

Important:
    SHAP calculations are performed on a limited sample of data.
    Do NOT calculate SHAP values for the entire CICIDS2017 dataset
    because it contains millions of rows and would consume excessive
    memory and computation time.
"""

# =============================================================================
# Imports
# =============================================================================

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# =============================================================================
# Project Paths
# =============================================================================

# This file is expected to be inside:
#
# NIDS-ML/
# ├── src/
# │   └── shap_explainability.py
# ├── models/
# ├── data/
# │   └── processed/
# ├── results/
# └── logs/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "shap"
LOGS_DIR = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / "explainability.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("nids.shap")


# =============================================================================
# SHAP Explainer
# =============================================================================

class SHAPExplainer:
    """
    SHAP explainability wrapper for Random Forest and XGBoost models.
    """

    def __init__(
        self,
        model,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        feature_names: Optional[list] = None,
        model_name: str = "model",
        max_samples: int = 1000,
    ):
        """
        Initialize SHAP explainer.

        Args:
            model:
                Trained Random Forest or XGBoost model.

            X_train:
                Optional training dataset.

            X_test:
                Optional test dataset.

            feature_names:
                Names of model input features.

            model_name:
                Name used when saving plots/files.

            max_samples:
                Maximum number of samples used for SHAP calculations.
        """

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.model_name = model_name
        self.max_samples = max_samples

        self.explainer = None
        self.shap_values = None
        self.expected_value = None

        # ---------------------------------------------------------------------
        # Feature names
        # ---------------------------------------------------------------------

        if feature_names is not None:

            self.feature_names = list(feature_names)

        elif isinstance(X_train, pd.DataFrame):

            self.feature_names = X_train.columns.tolist()

        elif isinstance(X_test, pd.DataFrame):

            self.feature_names = X_test.columns.tolist()

        else:

            self.feature_names = None

        logger.info(
            "SHAPExplainer initialized for model: %s",
            self.model_name,
        )

    # =========================================================================
    # Data Preparation
    # =========================================================================

    def _prepare_dataframe(self, X):
        """
        Convert input data into a DataFrame with the correct feature names.
        """

        if X is None:
            raise ValueError(
                "Input data cannot be None."
            )

        if isinstance(X, pd.DataFrame):

            dataframe = X.copy()

        else:

            dataframe = pd.DataFrame(
                X,
                columns=self.feature_names,
            )

        # ---------------------------------------------------------------------
        # Remove label column if accidentally supplied
        # ---------------------------------------------------------------------

        label_columns = [
            "Label",
            "label",
            "Attack",
            "attack",
            "Class",
            "class",
        ]

        for label_column in label_columns:

            if label_column in dataframe.columns:

                dataframe = dataframe.drop(
                    columns=[label_column]
                )

                logger.debug(
                    "Removed label column: %s",
                    label_column,
                )

        # ---------------------------------------------------------------------
        # Make sure feature ordering is correct
        # ---------------------------------------------------------------------

        if self.feature_names is not None:

            missing_features = [
                feature
                for feature in self.feature_names
                if feature not in dataframe.columns
            ]

            if missing_features:

                raise ValueError(
                    "Missing model features: "
                    f"{missing_features}"
                )

            dataframe = dataframe[
                self.feature_names
            ]

        # ---------------------------------------------------------------------
        # Replace invalid values
        # ---------------------------------------------------------------------

        dataframe = dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        dataframe = dataframe.fillna(
            dataframe.median(numeric_only=True)
        )

        dataframe = dataframe.fillna(0)

        return dataframe

    # =========================================================================
    # Sample Data
    # =========================================================================

    def _sample_data(self, X):
        """
        Limit the number of samples used by SHAP.
        """

        X = self._prepare_dataframe(X)

        if len(X) <= self.max_samples:

            return X

        logger.info(
            "SHAP input contains %s samples.",
            f"{len(X):,}",
        )

        logger.info(
            "Sampling %s samples for SHAP.",
            f"{self.max_samples:,}",
        )

        return X.sample(
            n=self.max_samples,
            random_state=42,
        )

    # =========================================================================
    # Create Tree Explainer
    # =========================================================================

    def create_explainer(self):
        """
        Create SHAP TreeExplainer.

        TreeExplainer is appropriate for:
            - Random Forest
            - XGBoost
            - Decision Trees
            - Other tree-based models
        """

        logger.info(
            "Creating SHAP TreeExplainer..."
        )

        try:

            self.explainer = shap.TreeExplainer(
                self.model
            )

            self.expected_value = (
                self.explainer.expected_value
            )

            logger.info(
                "SHAP TreeExplainer created successfully."
            )

            return self.explainer

        except Exception as exc:

            logger.error(
                "Failed to create SHAP TreeExplainer: %s",
                exc,
            )

            raise

    # =========================================================================
    # Calculate SHAP Values
    # =========================================================================

    def calculate_shap_values(
        self,
        X=None,
        max_samples: Optional[int] = None,
    ):
        """
        Calculate SHAP values.

        By default, at most 1,000 samples are explained.

        Returns:
            SHAP Explanation object.
        """

        if X is None:

            X = self.X_test

        if X is None:

            raise ValueError(
                "No data supplied for SHAP calculation."
            )

        if max_samples is not None:

            old_limit = self.max_samples

            self.max_samples = max_samples

            X = self._sample_data(X)

            self.max_samples = old_limit

        else:

            X = self._sample_data(X)

        logger.info(
            "Calculating SHAP values for %s samples...",
            f"{len(X):,}",
        )

        if self.explainer is None:

            self.create_explainer()

        try:

            # -----------------------------------------------------------------
            # Modern SHAP API
            # -----------------------------------------------------------------

            explanation = self.explainer(
                X
            )

            self.shap_values = explanation

            logger.info(
                "SHAP values calculated successfully."
            )

            return explanation

        except Exception as exc:

            logger.error(
                "SHAP calculation failed: %s",
                exc,
            )

            raise

    # =========================================================================
    # Get SHAP Matrix
    # =========================================================================

    def _get_shap_matrix(self):
        """
        Return SHAP values as a 2D matrix.

        Handles different SHAP output formats.
        """

        if self.shap_values is None:

            self.calculate_shap_values()

        values = self.shap_values.values

        # ---------------------------------------------------------------------
        # Binary classification can sometimes produce:
        #
        # samples × features × classes
        #
        # Select the attack class (class 1).
        # ---------------------------------------------------------------------

        if values.ndim == 3:

            values = values[:, :, 1]

        return values

    # =========================================================================
    # Summary Plot
    # =========================================================================

    def plot_summary(
        self,
        plot_type="dot",
        max_display=20,
        save_path=None,
    ):
        """
        Create SHAP summary plot.

        Args:
            plot_type:
                dot, bar, violin

            max_display:
                Number of features.

            save_path:
                Optional output path.
        """

        logger.info(
            "Creating SHAP summary plot: %s",
            plot_type,
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        if save_path is None:

            save_path = (
                PLOTS_DIR /
                f"{self.model_name}_shap_summary.png"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(12, 8)
        )

        shap.summary_plot(
            self._get_shap_matrix(),
            self.shap_values.data,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "SHAP summary plot saved: %s",
            save_path,
        )

        return str(save_path)

    # =========================================================================
    # Bar Plot
    # =========================================================================

    def plot_bar(
        self,
        max_display=20,
        save_path=None,
    ):
        """
        Create global SHAP feature importance bar plot.
        """

        logger.info(
            "Creating SHAP global feature importance plot."
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        if save_path is None:

            save_path = (
                PLOTS_DIR /
                f"{self.model_name}_shap_bar.png"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shap_matrix = self._get_shap_matrix()

        mean_absolute_shap = np.abs(
            shap_matrix
        ).mean(axis=0)

        importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Mean_Absolute_SHAP": mean_absolute_shap,
            }
        ).sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        )

        top_features = importance_df.head(
            max_display
        )

        plt.figure(
            figsize=(10, 8)
        )

        plt.barh(
            top_features["Feature"][::-1],
            top_features["Mean_Absolute_SHAP"][::-1],
        )

        plt.xlabel(
            "Mean Absolute SHAP Value"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            f"SHAP Feature Importance - {self.model_name}"
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "SHAP bar plot saved: %s",
            save_path,
        )

        return importance_df

    # =========================================================================
    # Waterfall Plot
    # =========================================================================

    def plot_waterfall(
        self,
        instance_index=0,
        max_display=20,
        save_path=None,
    ):
        """
        Create SHAP waterfall plot for one prediction.
        """

        logger.info(
            "Creating waterfall plot for instance %d",
            instance_index,
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        if instance_index < 0:

            raise IndexError(
                "instance_index cannot be negative."
            )

        if instance_index >= len(
            self.shap_values.data
        ):

            raise IndexError(
                "instance_index is outside the SHAP sample range."
            )

        if save_path is None:

            save_path = (
                PLOTS_DIR /
                f"{self.model_name}_waterfall_"
                f"{instance_index}.png"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------------
        # Construct a single-instance SHAP Explanation
        # ---------------------------------------------------------------------

        explanation = self.shap_values[
            instance_index
        ]

        # For binary classification with 3D output
        if explanation.values.ndim > 1:

            explanation = shap.Explanation(
                values=explanation.values[:, 1],
                base_values=explanation.base_values[1],
                data=explanation.data,
                feature_names=self.feature_names,
            )

        plt.figure(
            figsize=(12, 8)
        )

        shap.plots.waterfall(
            explanation,
            max_display=max_display,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "Waterfall plot saved: %s",
            save_path,
        )

        return str(save_path)

    # =========================================================================
    # Force Plot
    # =========================================================================

    def plot_force(
        self,
        instance_index=0,
        save_path=None,
    ):
        """
        Create SHAP force plot for one prediction.

        A static PNG is created for convenient storage.
        """

        logger.info(
            "Creating force plot for instance %d",
            instance_index,
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        if instance_index >= len(
            self.shap_values.data
        ):

            raise IndexError(
                "instance_index is outside the SHAP sample range."
            )

        if save_path is None:

            save_path = (
                PLOTS_DIR /
                f"{self.model_name}_force_"
                f"{instance_index}.html"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------------
        # SHAP's force plot is naturally interactive.
        # Save as HTML instead of forcing it into PNG.
        # ---------------------------------------------------------------------

        shap.force_plot(
            self.explainer.expected_value,
            self._get_shap_matrix()[instance_index],
            self.shap_values.data[instance_index],
            feature_names=self.feature_names,
            matplotlib=False,
        )

        force_plot = shap.force_plot(
            self.explainer.expected_value,
            self._get_shap_matrix()[instance_index],
            self.shap_values.data[instance_index],
            feature_names=self.feature_names,
        )

        shap.save_html(
            str(save_path),
            force_plot,
        )

        logger.info(
            "Interactive force plot saved: %s",
            save_path,
        )

        return str(save_path)

    # =========================================================================
    # Dependence Plot
    # =========================================================================

    def plot_dependence(
        self,
        feature_name,
        interaction_feature="auto",
        save_path=None,
    ):
        """
        Create SHAP dependence plot.

        Args:
            feature_name:
                Feature name or feature index.

            interaction_feature:
                Feature used for interaction coloring.
        """

        logger.info(
            "Creating dependence plot for %s",
            feature_name,
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        # ---------------------------------------------------------------------
        # Validate feature
        # ---------------------------------------------------------------------

        if isinstance(feature_name, int):

            if feature_name < 0 or feature_name >= len(
                self.feature_names
            ):

                raise IndexError(
                    "Feature index out of range."
                )

            feature_name = self.feature_names[
                feature_name
            ]

        elif feature_name not in self.feature_names:

            raise ValueError(
                f"Unknown feature: {feature_name}"
            )

        if save_path is None:

            safe_name = (
                str(feature_name)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            save_path = (
                PLOTS_DIR /
                f"{self.model_name}_dependence_"
                f"{safe_name}.png"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(10, 7)
        )

        shap.dependence_plot(
            feature_name,
            self._get_shap_matrix(),
            self.shap_values.data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "Dependence plot saved: %s",
            save_path,
        )

        return str(save_path)

    # =========================================================================
    # Explain Single Prediction
    # =========================================================================

    def explain_prediction(
        self,
        instance,
        return_dict=True,
    ):
        """
        Explain one individual prediction.

        Returns the model prediction and feature contributions.
        """

        logger.info(
            "Explaining individual prediction."
        )

        X = self._prepare_dataframe(
            pd.DataFrame(
                [instance],
                columns=self.feature_names
                if not isinstance(instance, pd.Series)
                else None,
            )
            if not isinstance(instance, pd.DataFrame)
            else instance
        )

        if self.explainer is None:

            self.create_explainer()

        explanation = self.explainer(
            X
        )

        shap_values = explanation.values

        if shap_values.ndim == 3:

            shap_values = shap_values[0, :, 1]

        else:

            shap_values = shap_values[0]

        prediction = self.model.predict(
            X
        )[0]

        probabilities = self.model.predict_proba(
            X
        )[0]

        contributions = []

        for feature, value, shap_value in zip(
            X.columns,
            X.iloc[0].values,
            shap_values,
        ):

            contributions.append(
                {
                    "feature": feature,
                    "value": float(value),
                    "shap_value": float(shap_value),
                    "absolute_shap": float(
                        abs(shap_value)
                    ),
                }
            )

        contributions.sort(
            key=lambda item: item["absolute_shap"],
            reverse=True,
        )

        if return_dict:

            return {
                "model": self.model_name,
                "prediction": int(prediction),
                "prediction_label": (
                    "ATTACK"
                    if int(prediction) == 1
                    else "BENIGN"
                ),
                "probability_benign": float(
                    probabilities[0]
                ),
                "probability_attack": float(
                    probabilities[1]
                ),
                "features": contributions,
            }

        return explanation

    # =========================================================================
    # Top Features for Prediction
    # =========================================================================

    def get_top_features(
        self,
        instance_index=0,
        top_n=10,
    ):
        """
        Return the top features contributing to one prediction.
        """

        logger.info(
            "Getting top %d features for instance %d",
            top_n,
            instance_index,
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        if instance_index >= len(
            self.shap_values.data
        ):

            raise IndexError(
                "instance_index is outside the SHAP sample range."
            )

        shap_values = self._get_shap_matrix()[
            instance_index
        ]

        feature_values = self.shap_values.data[
            instance_index
        ]

        result = []

        for feature, value, shap_value in zip(
            self.feature_names,
            feature_values,
            shap_values,
        ):

            result.append(
                {
                    "feature": feature,
                    "value": float(value),
                    "shap_value": float(shap_value),
                    "absolute_shap": float(
                        abs(shap_value)
                    ),
                    "direction": (
                        "increases_attack_score"
                        if shap_value > 0
                        else "decreases_attack_score"
                    ),
                }
            )

        result.sort(
            key=lambda item: item["absolute_shap"],
            reverse=True,
        )

        return result[:top_n]

    # =========================================================================
    # Global Importance
    # =========================================================================

    def generate_global_importance(self):
        """
        Generate global feature importance using mean absolute SHAP values.

        Returns:
            pd.DataFrame
        """

        logger.info(
            "Generating global SHAP feature importance."
        )

        if self.shap_values is None:

            self.calculate_shap_values()

        shap_matrix = self._get_shap_matrix()

        importance = np.abs(
            shap_matrix
        ).mean(axis=0)

        importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Mean_Absolute_SHAP": importance,
            }
        )

        importance_df = importance_df.sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        ).reset_index(
            drop=True
        )

        importance_df[
            "Rank"
        ] = np.arange(
            1,
            len(importance_df) + 1
        )

        # ---------------------------------------------------------------------
        # Save ranking
        # ---------------------------------------------------------------------

        output_path = (
            RESULTS_DIR /
            f"{self.model_name}_shap_importance.csv"
        )

        importance_df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Global SHAP importance saved: %s",
            output_path,
        )

        return importance_df

    # =========================================================================
    # Save Explainer
    # =========================================================================

    def save_explainer(
        self,
        filepath=None,
    ):
        """
        Save the SHAP explainer to disk.

        Note:
            TreeExplainer objects can be version-sensitive. For long-term
            reproducibility, save the model and feature list as well.
        """

        if self.explainer is None:

            self.create_explainer()

        if filepath is None:

            filepath = (
                MODELS_DIR /
                f"{self.model_name}_shap_explainer.pkl"
            )

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            with open(
                filepath,
                "wb"
            ) as file:

                pickle.dump(
                    self.explainer,
                    file,
                )

            logger.info(
                "SHAP explainer saved: %s",
                filepath,
            )

            return str(filepath)

        except Exception as exc:

            logger.error(
                "Failed to save SHAP explainer: %s",
                exc,
            )

            raise

    # =========================================================================
    # Load Explainer
    # =========================================================================

    def load_explainer(
        self,
        filepath=None,
    ):
        """
        Load a previously saved SHAP explainer.
        """

        if filepath is None:

            filepath = (
                MODELS_DIR /
                f"{self.model_name}_shap_explainer.pkl"
            )

        filepath = Path(filepath)

        if not filepath.exists():

            raise FileNotFoundError(
                f"SHAP explainer not found: {filepath}"
            )

        try:

            with open(
                filepath,
                "rb"
            ) as file:

                self.explainer = pickle.load(
                    file
                )

            self.expected_value = (
                self.explainer.expected_value
            )

            logger.info(
                "SHAP explainer loaded: %s",
                filepath,
            )

            return self.explainer

        except Exception as exc:

            logger.error(
                "Failed to load SHAP explainer: %s",
                exc,
            )

            raise

    # =========================================================================
    # Dashboard Explanation
    # =========================================================================

    def explain_for_dashboard(
        self,
        instance,
        top_n=10,
    ):
        """
        Generate compact explanation data for a dashboard.

        Returns:
            Dictionary suitable for JSON/API/dashboard usage.
        """

        explanation = self.explain_prediction(
            instance,
            return_dict=True,
        )

        top_features = sorted(
            explanation["features"],
            key=lambda item: item["absolute_shap"],
            reverse=True,
        )[:top_n]

        return {
            "model": explanation["model"],
            "prediction": explanation["prediction"],
            "prediction_label": explanation[
                "prediction_label"
            ],
            "probability_benign": explanation[
                "probability_benign"
            ],
            "probability_attack": explanation[
                "probability_attack"
            ],
            "top_features": top_features,
        }


# =============================================================================
# Utility — Load Model and Dataset
# =============================================================================

def load_model_and_data(
    model_name="random_forest",
):
    """
    Convenience function for loading a trained model and selected dataset.

    Expected model files:
        models/random_forest.pkl
        models/xgboost.pkl

    Expected dataset:
        data/processed/selected_features.csv
    """

    model_path = (
        MODELS_DIR /
        f"{model_name}.pkl"
    )

    data_path = (
        DATA_DIR /
        "selected_features.csv"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not data_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    logger.info(
        "Loading model: %s",
        model_path,
    )

    model = joblib.load(
        model_path
    )

    logger.info(
        "Loading selected feature dataset: %s",
        data_path,
    )

    df = pd.read_csv(
        data_path
    )

    # -------------------------------------------------------------------------
    # Remove target
    # -------------------------------------------------------------------------

    label_columns = [
        "Label",
        "label",
        "Attack",
        "attack",
        "Class",
        "class",
    ]

    label_column = None

    for column in label_columns:

        if column in df.columns:

            label_column = column
            break

    if label_column is not None:

        X = df.drop(
            columns=[label_column]
        )

    else:

        X = df

    return model, X


# =============================================================================
# Generate SHAP Results
# =============================================================================

def generate_model_explainability(
    model_name="random_forest",
    max_samples=1000,
):
    """
    Generate the main SHAP outputs for one trained model.

    Outputs:
        - SHAP summary plot
        - SHAP bar plot
        - Global importance CSV
        - Waterfall plot for first sample
        - Force plot for first sample
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "SHAP EXPLAINABILITY — %s",
        model_name.upper(),
    )
    logger.info("=" * 70)

    model, X = load_model_and_data(
        model_name=model_name
    )

    feature_names = X.columns.tolist()

    explainer = SHAPExplainer(
        model=model,
        X_test=X,
        feature_names=feature_names,
        model_name=model_name,
        max_samples=max_samples,
    )

    # -------------------------------------------------------------------------
    # Calculate SHAP
    # -------------------------------------------------------------------------

    explainer.calculate_shap_values()

    # -------------------------------------------------------------------------
    # Global summary
    # -------------------------------------------------------------------------

    explainer.plot_summary(
        plot_type="dot",
        max_display=20,
    )

    # -------------------------------------------------------------------------
    # Global bar plot
    # -------------------------------------------------------------------------

    explainer.plot_bar(
        max_display=20,
    )

    # -------------------------------------------------------------------------
    # Global importance CSV
    # -------------------------------------------------------------------------

    importance_df = (
        explainer.generate_global_importance()
    )

    # -------------------------------------------------------------------------
    # Individual waterfall
    # -------------------------------------------------------------------------

    explainer.plot_waterfall(
        instance_index=0,
        max_display=20,
    )

    # -------------------------------------------------------------------------
    # Interactive force plot
    # -------------------------------------------------------------------------

    explainer.plot_force(
        instance_index=0,
    )

    # -------------------------------------------------------------------------
    # Save explainer
    # -------------------------------------------------------------------------

    explainer.save_explainer()

    logger.info("")
    logger.info(
        "SHAP explainability completed for %s.",
        model_name,
    )

    logger.info(
        "Top 10 globally important features:"
    )

    for _, row in importance_df.head(10).iterrows():

        logger.info(
            "  %s : %.6f",
            row["Feature"],
            row["Mean_Absolute_SHAP"],
        )

    return explainer


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    """
    Run:

        python shap_explainability.py

    This generates SHAP explanations for both:

        Random Forest
        XGBoost
    """

    logger.info(
        "Starting NIDS SHAP explainability."
    )

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------

    try:

        generate_model_explainability(
            model_name="random_forest",
            max_samples=1000,
        )

    except Exception as exc:

        logger.error(
            "Random Forest SHAP analysis failed: %s",
            exc,
        )

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------

    try:

        generate_model_explainability(
            model_name="xgboost",
            max_samples=1000,
        )

    except Exception as exc:

        logger.error(
            "XGBoost SHAP analysis failed: %s",
            exc,
        )

    logger.info(
        "=" * 70
    )

    logger.info(
        "SHAP EXPLAINABILITY FINISHED."
    )

    logger.info(
        "=" * 70
    )