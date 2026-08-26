# =============================================================================
# NIDS-ML — Data Preprocessing
# =============================================================================
# Purpose:
#   Clean CICIDS2017 CSV files and create one consistent dataset for model
#   training.
#
# IMPORTANT RESEARCH PIPELINE
#   Raw CICIDS2017 CSV
#       ↓
#   clean column names
#       ↓
#   clean NaN / infinity
#       ↓
#   create binary label
#       ↓
#   select the same 30 features used by the models
#       ↓
#   save selected_features.csv
#
# SMOTE and train/test splitting are intentionally NOT done here.
# They belong in model_training.py so that SMOTE is applied ONLY to the
# training portion and the test set remains untouched.
#
# Random Forest and XGBoost do not require StandardScaler, so scaling is also
# intentionally NOT performed here.
# =============================================================================

import logging
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    PROCESSED_DIR / "selected_features.csv"
)


# =============================================================================
# EXACT FEATURES USED BY THE MODELS
# =============================================================================
#
# These are the same 30 features used by model_training.py.
#
# CICIDS2017 raw CSVs normally use spaces in their column names, e.g.
#
#   "Total Backward Packets"
#
# They are normalized to:
#
#   "Total_Backward_Packets"
#
# before selecting these features.
# =============================================================================

SELECTED_FEATURES = [
    "Total_Backward_Packets",
    "Total_Length_of_Bwd_Packets",
    "Fwd_Packet_Length_Max",
    "Fwd_Packet_Length_Std",
    "Bwd_Packet_Length_Max",
    "Bwd_Packet_Length_Min",
    "Bwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Std",
    "Flow_IAT_Std",
    "Flow_IAT_Max",
    "Flow_IAT_Min",
    "Fwd_IAT_Mean",
    "Fwd_IAT_Std",
    "Fwd_IAT_Max",
    "Fwd_IAT_Min",
    "Bwd_IAT_Total",
    "Max_Packet_Length",
    "Packet_Length_Std",
    "Packet_Length_Variance",
    "PSH_Flag_Count",
    "Average_Packet_Size",
    "Avg_Bwd_Segment_Size",
    "Subflow_Fwd_Bytes",
    "Subflow_Bwd_Packets",
    "Subflow_Bwd_Bytes",
    "Init_Win_bytes_backward",
    "Active_Mean",
    "Active_Max",
    "Active_Min",
    "Idle_Max",
]


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "preprocessing.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
        sys.stderr.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CICIDS2017/CICFlowMeter column names to a consistent format.

    Example:
        "Total Backward Packets"
            -> "Total_Backward_Packets"

        " Flow IAT Std "
            -> "Flow_IAT_Std"
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """
    Load all CSV files from data/raw and combine them.

    Returns:
        Combined raw DataFrame.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING CICIDS2017 DATASET")
    logger.info("=" * 70)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}"
        )

    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {RAW_DIR}"
        )

    logger.info(
        "Found %d CSV files.",
        len(csv_files),
    )

    frames = []

    for csv_file in csv_files:

        logger.info(
            "Loading: %s",
            csv_file.name,
        )

        try:

            df = pd.read_csv(
                csv_file,
                low_memory=False,
            )

            df = normalize_column_names(
                df
            )

            logger.info(
                "  Rows: %s | Columns: %d",
                f"{len(df):,}",
                len(df.columns),
            )

            frames.append(df)

        except Exception as exc:

            logger.warning(
                "Could not load %s: %s",
                csv_file.name,
                exc,
            )

    if not frames:
        raise ValueError(
            "No CSV files could be loaded."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info("")
    logger.info(
        "Combined dataset: %s rows × %d columns",
        f"{len(df):,}",
        len(df.columns),
    )

    return df


# =============================================================================
# FIND LABEL COLUMN
# =============================================================================

def find_label_column(
    df: pd.DataFrame,
) -> str:
    """
    Find the CICIDS2017 label column.
    """

    candidates = [
        "Label",
        "label",
        "Class",
        "class",
        "Attack",
        "attack",
    ]

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    # Fallback search.
    for column in df.columns:

        name = column.lower()

        if (
            "label" in name
            or name == "class"
            or "attack" in name
        ):
            return column

    raise ValueError(
        "Could not find the label column."
    )


# =============================================================================
# CLEAN FEATURES
# =============================================================================

def clean_features(
    df: pd.DataFrame,
    feature_names,
) -> pd.DataFrame:
    """
    Select the required 30 features and clean their values.

    Cleaning:
        - convert to numeric
        - replace +/- infinity
        - replace NaN with column median
        - replace completely invalid columns with zero
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: SELECTING AND CLEANING FEATURES")
    logger.info("=" * 70)

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:

        logger.error(
            "Missing required features:"
        )

        for feature in missing_features:
            logger.error(
                "  - %s",
                feature,
            )

        raise ValueError(
            "The raw CICIDS2017 CSV does not contain "
            "all required model features."
        )

    X = df[
        feature_names
    ].copy()

    logger.info(
        "Selected %d features.",
        len(X.columns),
    )

    # Convert every feature to numeric.
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace infinities with NaN first.
    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # Fill missing values using each feature's median.
    missing_before = int(
        X.isna().sum().sum()
    )

    if missing_before > 0:

        logger.info(
            "Missing/invalid feature values before cleaning: %s",
            f"{missing_before:,}",
        )

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
            "NaN values remain after feature cleaning."
        )

    if not np.isfinite(
        X.to_numpy()
    ).all():
        raise ValueError(
            "Infinite values remain after feature cleaning."
        )

    logger.info(
        "Feature matrix after cleaning: %s",
        X.shape,
    )

    return X


# =============================================================================
# CREATE BINARY LABEL
# =============================================================================

def create_binary_labels(
    df: pd.DataFrame,
    label_column: str,
) -> pd.Series:
    """
    Convert CICIDS2017 labels into:

        BENIGN = 0
        Everything else = ATTACK = 1
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: CREATING BINARY LABEL")
    logger.info("=" * 70)

    raw_labels = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    benign_values = {
        "BENIGN",
        "benign",
        "Benign",
        "NORMAL",
        "Normal",
        "normal",
    }

    y = raw_labels.apply(
        lambda value:
        0
        if value in benign_values
        else 1
    )

    y = y.astype(
        np.int8
    )

    # Remove accidental textual NaN labels.
    invalid_mask = (
        raw_labels.str.lower()
        == "nan"
    )

    if invalid_mask.any():

        logger.warning(
            "Removing %d rows with missing labels.",
            int(invalid_mask.sum()),
        )

        y = y.loc[
            ~invalid_mask
        ].reset_index(
            drop=True
        )

    benign_count = int(
        (y == 0).sum()
    )

    attack_count = int(
        (y == 1).sum()
    )

    logger.info(
        "BENIGN: %s",
        f"{benign_count:,}",
    )

    logger.info(
        "ATTACK: %s",
        f"{attack_count:,}",
    )

    return y


# =============================================================================
# BUILD FINAL DATASET
# =============================================================================

def build_processed_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the final dataset containing:

        30 selected features + Label

    No scaling.
    No SMOTE.
    No train/test split.

    Those operations are handled by model_training.py.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: BUILDING FINAL PROCESSED DATASET")
    logger.info("=" * 70)

    label_column = find_label_column(
        df
    )

    logger.info(
        "Label column: %s",
        label_column,
    )

    # Create labels first so we can remove rows with missing labels.
    raw_label = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    valid_rows = (
        raw_label.str.lower()
        != "nan"
    )

    removed = int(
        (~valid_rows).sum()
    )

    if removed > 0:

        logger.warning(
            "Removing %d rows with missing labels.",
            removed,
        )

        df = (
            df.loc[valid_rows]
            .reset_index(drop=True)
        )

    X = clean_features(
        df,
        SELECTED_FEATURES,
    )

    y = create_binary_labels(
        df,
        label_column,
    )

    # Both X and y must have exactly the same number of rows.
    if len(X) != len(y):

        raise ValueError(
            f"X/y length mismatch: "
            f"X={len(X):,}, y={len(y):,}"
        )

    X.reset_index(
        drop=True,
        inplace=True,
    )

    y.reset_index(
        drop=True,
        inplace=True,
    )

    processed = X.copy()

    processed["Label"] = y

    # Final validation.
    if processed.isna().any().any():

        raise ValueError(
            "Processed dataset contains NaN values."
        )

    if not np.isfinite(
        processed[
            SELECTED_FEATURES
        ].to_numpy()
    ).all():

        raise ValueError(
            "Processed dataset contains infinite values."
        )

    logger.info(
        "Final dataset shape: %s",
        processed.shape,
    )

    logger.info(
        "Features: %d",
        len(SELECTED_FEATURES),
    )

    logger.info(
        "Label: Label",
    )

    return processed


# =============================================================================
# SAVE DATASET
# =============================================================================

def save_dataset(
    df: pd.DataFrame,
) -> Path:
    """
    Save the final processed dataset.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: SAVING PROCESSED DATASET")
    logger.info("=" * 70)

    # Ensure the exact feature order is preserved.
    output_columns = (
        SELECTED_FEATURES
        + ["Label"]
    )

    df = df[
        output_columns
    ].copy()

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info(
        "Saved: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Rows: %s",
        f"{len(df):,}",
    )

    logger.info(
        "Columns: %d",
        len(df.columns),
    )

    logger.info(
        "File size: %.2f MB",
        OUTPUT_FILE.stat().st_size
        / (1024 ** 2),
    )

    return OUTPUT_FILE


# =============================================================================
# MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info("NIDS-ML DATA PREPROCESSING")
    logger.info("=" * 70)

    # 1. Load all raw CICIDS2017 CSV files.
    df = load_dataset()

    # 2. Remove complete duplicate rows.
    before = len(df)

    df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    removed = before - len(df)

    logger.info(
        "Removed duplicate rows: %s",
        f"{removed:,}",
    )

    # 3. Build the clean 30-feature binary-classification dataset.
    processed = build_processed_dataset(
        df
    )

    # 4. Save.
    output_file = save_dataset(
        processed
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETED")
    logger.info("=" * 70)

    logger.info(
        "Output: %s",
        output_file,
    )

    logger.info(
        "Next step: run model_training.py"
    )

    logger.info(
        "SMOTE will be applied ONLY to the training split."
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()