🎉 Data Preprocessing Module

Implementation Summary

The current NIDS-ML preprocessing pipeline prepares the CICIDS2017 flow dataset for the Random Forest and XGBoost models used in this project.

The implementation is focused on the actual project pipeline rather than the original broader design. It:

Loads all CICIDS2017 CSV files from data/raw/

Combines the datasets

Removes duplicate flow records

Selects the 30 features used by the trained models

Cleans invalid numeric values

Creates a binary BENIGN / ATTACK label

Saves the model-ready dataset to data/processed/selected_features.csv

Important: SMOTE, StandardScaler, generic categorical encoding, NSL-KDD support, and automatic multi-dataset label handling are not part of the current active preprocessing output. Do not describe them as implemented features unless the code is changed to actually use them.

📋 Current Implementation

File

src/data_preprocessing.py

The script is executed directly:

cd src
python data_preprocessing.py

🔄 Preprocessing Pipeline

The current pipeline is:

CICIDS2017 CSV files
        │
        ▼
Load all CSV files from data/raw/
        │
        ▼
Combine datasets
        │
        ▼
Remove duplicate rows
        │
        ▼
Select 30 model features
        │
        ▼
Clean invalid / missing numeric values
        │
        ▼
Create binary label
        │
        ├── BENIGN → 0
        │
        └── ATTACK → 1
        │
        ▼
selected_features.csv
        │
        ▼
model_training.py
        │
        ├── Random Forest
        └── XGBoost

1. Loading CICIDS2017 Dataset

The script automatically discovers CSV files in:

data/raw/

The current project uses these eight CICIDS2017 files:

data/raw/

├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Each file contains CICIDS2017 flow-level network statistics.

2. Dataset Combination

The eight CSV files are combined into one DataFrame.

The actual run produced:

CSV files found:        8

Combined dataset:
Rows    : 2,830,743
Columns : 79

The preprocessing then removes duplicate records.

Actual result:

Duplicate rows removed: 308,381
Rows remaining:         2,522,362

3. Feature Selection

The current implementation uses a fixed set of 30 features.

These are the same features expected by the trained Random Forest and XGBoost models.

1.  Total_Backward_Packets
2.  Total_Length_of_Bwd_Packets
3.  Fwd_Packet_Length_Max
4.  Fwd_Packet_Length_Std
5.  Bwd_Packet_Length_Max
6.  Bwd_Packet_Length_Min
7.  Bwd_Packet_Length_Mean
8.  Bwd_Packet_Length_Std
9.  Flow_IAT_Std
10. Flow_IAT_Max
11. Flow_IAT_Min
12. Fwd_IAT_Mean
13. Fwd_IAT_Std
14. Fwd_IAT_Max
15. Fwd_IAT_Min
16. Bwd_IAT_Total
17. Max_Packet_Length
18. Packet_Length_Std
19. Packet_Length_Variance
20. PSH_Flag_Count
21. Average_Packet_Size
22. Avg_Bwd_Segment_Size
23. Subflow_Fwd_Bytes
24. Subflow_Bwd_Packets
25. Subflow_Bwd_Bytes
26. Init_Win_bytes_backward
27. Active_Mean
28. Active_Max
29. Active_Min
30. Idle_Max

The final processed dataset therefore contains:

30 feature columns
+
1 label column
=
31 columns

4. Data Cleaning

Before saving the model-ready dataset, the selected feature matrix is cleaned so that the trained models receive valid numeric input.

The active pipeline removes or handles invalid feature values before producing the final dataset.

The result from the current run was:

Feature matrix after cleaning:
(2,522,362, 30)

The final dataset contains no separate categorical feature encoding stage because the 30 selected CICIDS2017 features used by the models are numeric flow features.

5. Binary Label Creation

The original CICIDS2017 attack labels are converted into the binary classification used by this project.

BENIGN → 0
Any attack → 1

The actual processed dataset contains:

BENIGN: 2,096,484
ATTACK:   425,878

Final distribution:

BENIGN: approximately 83.11%
ATTACK: approximately 16.89%

The resulting label column is:

Label

6. Final Processed Dataset

The preprocessing script creates:

data/processed/selected_features.csv

Actual output from the current implementation:

Rows    : 2,522,362
Columns : 31
Features: 30
Label   : Label
File size: approximately 354.92 MB

The structure is:

selected_features.csv

├── Total_Backward_Packets
├── Total_Length_of_Bwd_Packets
├── ...
├── Idle_Max
└── Label

7. Feature Metadata

The model feature order is also stored separately so that training, CSV testing, and real-time inference use the same feature structure.

The project stores:

models/feature_names.pkl
models/feature_names.txt

feature_names.pkl contains:

{
    "feature_names": [...],
    "feature_count": 30,
    "label_mapping": {
        "BENIGN": 0,
        "ATTACK": 1
    }
}

This prevents a mismatch between the columns used during training and the columns supplied during inference.

8. Connection With Model Training

The preprocessing module does not train the models itself.

Its output is consumed by:

src/model_training.py

The workflow is:

data_preprocessing.py
        │
        ▼
selected_features.csv
        │
        ▼
model_training.py
        │
        ├───────────────┐
        ▼               ▼
Random Forest        XGBoost
        │               │
        └───────┬───────┘
                ▼
          Saved .pkl models

Current model files include:

models/random_forest.pkl
models/xgboost.pkl
models/best_model.pkl

XGBoost is currently used as the primary model for real-time detection.

9. Connection With CSV Testing

The same 30 features are used by:

src/test_csv.py

Example:

python test_csv.py --model xgboost --csv "..\data\raw\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

The testing pipeline:

Original CICIDS2017 CSV
        │
        ▼
Find expected 30 features
        │
        ▼
Load XGBoost model
        │
        ▼
Predict BENIGN / ATTACK
        │
        ▼
Calculate evaluation metrics

This was successfully tested against the CICIDS2017 DDoS and PortScan CSV files.

10. Connection With Real-Time Detection

The same feature contract is also required by:

src/realtime_detection.py

The real-time pipeline is:

Live packets
      │
      ▼
Scapy
      │
      ▼
Flow aggregation
      │
      ▼
30 flow features
      │
      ▼
XGBoost
      │
      ▼
BENIGN / ATTACK
      │
      ▼
Confidence
      │
      ▼
Dashboard / Detection log

This means the preprocessing stage and real-time detector must agree on the feature definitions and feature order.

📊 Actual Processing Results

The latest successful preprocessing run produced:

======================================================================
NIDS-ML DATA PREPROCESSING
======================================================================

Found 8 CSV files.

Combined dataset:
2,830,743 rows × 79 columns

Removed duplicate rows:
308,381

Feature matrix after cleaning:
2,522,362 × 30

BENIGN:
2,096,484

ATTACK:
425,878

Final dataset:
2,522,362 rows × 31 columns

Features:
30

Label:
Label

Saved:
data/processed/selected_features.csv

🚀 How to Run

From the project root:

cd src
python data_preprocessing.py

The script automatically reads the CSV files from:

../data/raw/

and writes the processed dataset to:

../data/processed/selected_features.csv

🔍 Verify the Output

You can quickly verify the processed dataset with:

python -c "import pandas as pd; df=pd.read_csv('../data/processed/selected_features.csv', nrows=5); print(df.shape); print(df.columns.tolist()); print(df['Label'].value_counts())"

You should see:

30 feature columns + Label

and the two labels:

0
1

📁 Project Files Related to Preprocessing

NIDS-ML/
│
├── data/
│   ├── raw/
│   │   └── CICIDS2017 CSV files
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       └── selected_features.csv
│
├── models/
│   ├── feature_names.pkl
│   └── feature_names.txt
│
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    ├── test_csv.py
    ├── realtime_detection.py
    └── dashboard.py

⚠️ Important: Features Not Used in the Current Pipeline

The original documentation described several preprocessing operations that are not reflected in the current implementation/output.

Do not claim that the current project performs:

✗ NSL-KDD support
✗ StandardScaler normalization
✗ Generic categorical LabelEncoder processing
✗ SMOTE in data_preprocessing.py
✗ One-hot encoding
✗ PCA
✗ RFE
✗ Chi-square feature selection
✗ ANOVA feature selection
✗ Automatic support for arbitrary datasets

The current preprocessing pipeline is intentionally simpler:

CICIDS2017
   ↓
Combine
   ↓
Remove duplicates
   ↓
Select 30 features
   ↓
Clean numeric values
   ↓
Binary label
   ↓
selected_features.csv

This matches the dataset and model pipeline you are actually running.

🧪 Reproducibility

The preprocessing pipeline produces a deterministic feature structure because:

The same CICIDS2017 files are used

The same 30 features are selected

The label mapping is fixed

The feature order is stored in model metadata

The trained model and inference code therefore have a consistent feature contract.

🎯 Role in the Complete NIDS-ML System

                 CICIDS2017
                      │
                      ▼
             DATA PREPROCESSING
                      │
                      ▼
              30 FLOW FEATURES
                      │
                      ▼
              MODEL TRAINING
                │          │
                ▼          ▼
          Random Forest   XGBoost
                │          │
                └────┬─────┘
                     │
                     ▼
              MODEL EVALUATION
                     │
                     ▼
             REAL-TIME DETECTION
                     │
                     ▼
               STREAMLIT UI
                     │
                     ▼
          Source / Destination IP
          Prediction / Confidence
          Ports / Packet Count

🎓 Current Scope

The preprocessing module is complete for the current NIDS-ML implementation.

Implemented

CICIDS2017 CSV loading

Multiple CSV combination

Duplicate removal

Numeric data cleaning

30 feature selection

Binary label creation

Processed dataset generation

Feature metadata generation

Compatibility with Random Forest

Compatibility with XGBoost

Compatibility with CSV evaluation

Feature contract for real-time detection

Current output

data/processed/selected_features.csv

Next stage

model_training.py
        ↓
Random Forest + XGBoost
        ↓
test_csv.py
        ↓
realtime_detection.py
        ↓
dashboard.py