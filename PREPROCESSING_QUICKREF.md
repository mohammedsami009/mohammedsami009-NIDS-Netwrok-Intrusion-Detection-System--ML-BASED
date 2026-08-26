📌 Quick Reference: Data Preprocessing Module

🎯 Main File

src/data_preprocessing.py

This module prepares the CICIDS 2017 CSV files for the NIDS-ML machine-learning pipeline.

The current implementation is focused on the actual dataset and model feature pipeline used by this project.

🔧 Current Processing Pipeline

CICIDS2017 CSV files
        ↓
Load all CSV files from data/raw/
        ↓
Combine datasets
        ↓
Remove duplicate rows
        ↓
Select 30 model features
        ↓
Clean invalid / non-numeric values
        ↓
Create binary Label
        ↓
Save selected_features.csv
        ↓
Model Training

📊 Current Dataset

The project currently uses the CICIDS 2017 dataset.

The data/raw/ directory contains 8 CSV files:

data/raw/

├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Monday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv

The preprocessing run produced:

Original combined rows : 2,830,743
Duplicate rows removed  :   308,381
Final rows              : 2,522,362
Original columns        : 79
Model features          : 30
Final columns           : 31
                           └── 30 features + Label

🔧 Main Processing Steps

Step

Purpose

Current implementation

Dataset loading

Load all CSV files from data/raw/

✅

Dataset combination

Combine all CICIDS2017 files

✅

Duplicate removal

Remove duplicate network-flow records

✅

Feature selection

Keep the 30 features used by the models

✅

Data cleaning

Handle invalid/non-numeric values required by the selected features

✅

Binary labels

Convert labels to BENIGN=0, ATTACK=1

✅

Output generation

Save the model-ready dataset

✅

StandardScaler

Scale features before saving

❌ Not part of current preprocessing output

SMOTE

Balance the complete preprocessing dataset

❌ Not applied by this preprocessing step

NSL-KDD

Alternative dataset support

❌ Not part of the current implementation

🧩 Selected Features

The final model input contains exactly 30 features:

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

These same 30 features are used by:

XGBoost
Random Forest
CSV testing
Real-time detection
SHAP explainability

The feature order is stored separately in:

models/feature_names.pkl
models/feature_names.txt

🏷️ Label Processing

The project uses binary classification:

BENIGN → 0
ATTACK → 1

The current preprocessing run produced:

BENIGN : 2,096,484
ATTACK :   425,878
-------------------
TOTAL  : 2,522,362

The final dataset therefore contains:

30 feature columns
+
1 Label column
=
31 columns

🧹 Data Cleaning

The preprocessing pipeline removes duplicate records and prepares the selected features for machine-learning use.

During the verified preprocessing run:

Combined dataset:
2,830,743 rows × 79 columns

Duplicates removed:
308,381

Final feature matrix:
2,522,362 rows × 30 features

Final dataset:
2,522,362 rows × 31 columns

The generated dataset is intended to be directly consumed by the model-training module.

💾 Output

The primary preprocessing output is:

data/processed/selected_features.csv

Verified output:

Rows    : 2,522,362
Columns : 31
Size    : approximately 354.92 MB

The dataset contains:

30 selected features
+
Label

🏃 Quick Start

Run the preprocessing pipeline

From the project root:

cd src
python data_preprocessing.py

The script automatically reads the CSV files from:

../data/raw/

and writes the processed dataset to:

../data/processed/selected_features.csv

📁 Generated Metadata

The model feature metadata is also stored in:

models/
├── feature_names.pkl
└── feature_names.txt

feature_names.pkl stores:

feature_names
feature_count
label_mapping

Current feature count:

30

Current label mapping:

BENIGN = 0
ATTACK = 1

This metadata ensures that the CSV testing, real-time detection, and model-training components use the same feature order.

📊 Verified Preprocessing Output

A successful run currently looks like:

Found 8 CSV files.

Combined dataset:
2,830,743 rows × 79 columns

Removed duplicate rows:
308,381

Selected:
30 features

Feature matrix:
2,522,362 × 30

BENIGN:
2,096,484

ATTACK:
425,878

Final dataset:
2,522,362 × 31

Saved:
data/processed/selected_features.csv

⚙️ Configuration

The current preprocessing pipeline is intentionally simpler than the original planned pipeline.

Item

Current configuration

Dataset

CICIDS 2017

Input format

CSV

Input directory

data/raw/

Number of input files

8

Original features

79 columns

Selected features

30

Label

Label

Label format

BENIGN=0, ATTACK=1

Output

selected_features.csv

Output directory

data/processed/

Random state

Not required for the basic preprocessing run

StandardScaler

Not applied in saved preprocessing dataset

SMOTE

Not applied in preprocessing

NSL-KDD

Not currently supported

🧪 Model-Training Relationship

Preprocessing and training are separate stages.

data_preprocessing.py
        │
        ▼
selected_features.csv
        │
        ▼
model_training.py
        │
        ├── Random Forest
        │
        └── XGBoost

The current model-training pipeline uses the resulting dataset to train the two implemented models.

The trained models are stored in:

models/
├── random_forest.pkl
└── xgboost.pkl

The primary real-time model is:

xgboost.pkl

🧪 Verification

The processed dataset was subsequently used successfully for model evaluation.

For example, the XGBoost model was tested against CICIDS2017 CSV files using:

python test_csv.py --model xgboost --csv "..\data\raw\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

and:

python test_csv.py --model xgboost --csv "..\data\raw\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

The testing script confirmed that the expected 30 features are available and correctly aligned with the model.

⚠️ Important Notes

1. Do not describe the current preprocessing as SMOTE preprocessing

SMOTE is not part of the current preprocessing output.

If class balancing is performed elsewhere in the training pipeline, it should be documented there rather than claiming that data_preprocessing.py balances the entire dataset.

2. Do not claim StandardScaler is applied here

The current selected_features.csv contains the selected feature values used by the model pipeline. The preprocessing documentation should not claim that StandardScaler normalization is part of this saved output unless it is actually implemented.

3. Do not claim NSL-KDD support

The current project is using CICIDS 2017. NSL-KDD is not part of the implemented preprocessing workflow.

4. Do not claim automatic support for arbitrary label names

The current CICIDS2017 files use:

Label

Therefore the documentation focuses on the actual dataset rather than claiming broad automatic label-column detection.

5. Dataset evaluation vs real-world performance

The processed CICIDS2017 data is used for model training and evaluation.

High performance on CICIDS2017 should not automatically be described as guaranteed real-world intrusion-detection accuracy.

📝 Logging

The preprocessing script uses logging to report processing progress.

Typical information includes:

Dataset files discovered
Rows and columns per file
Combined dataset size
Duplicate removal
Selected feature count
Label distribution
Output path
Final dataset size

📞 Troubleshooting

No CSV files found

Check that the CICIDS2017 files are located in:

data/raw/

File not found

Run the command from the correct directory:

cd NIDS-ML/src
python data_preprocessing.py

Output has the wrong number of features

Check:

data/processed/selected_features.csv
models/feature_names.pkl
models/feature_names.txt

The expected model input is:

30 features

Model says required features are missing

Make sure the CSV being tested is a CICIDS2017 flow CSV containing the original 79 columns. The test_csv.py script extracts the same 30 selected features before prediction.

🎓 Code Quality

The preprocessing module is structured as a reusable project component and includes:

Modular processing stages

Logging

Data validation

Duplicate removal

Feature selection

Binary label generation

Reproducible feature ordering

Error reporting

Model-compatible output

The important goal is consistency between training, testing, SHAP, and real-time detection.

🚀 Current Project Workflow

The current NIDS-ML workflow is:

PART 1
CICIDS2017 Dataset
        ↓
PART 2
Data Preprocessing
        ↓
selected_features.csv
        ↓
PART 3
Feature Selection / Model Feature Definition
        ↓
30 Selected Features
        ↓
PART 4
Model Training & Evaluation
        ↓
Random Forest + XGBoost
        ↓
PART 5
SHAP + Real-Time Detection
        ↓
XGBoost + Scapy
        ↓
Streamlit Dashboard

🎯 Quick Help

Main preprocessing script

src/data_preprocessing.py

Input data

data/raw/

Processed dataset

data/processed/selected_features.csv

Feature metadata

models/feature_names.pkl
models/feature_names.txt

Trained models

models/random_forest.pkl
models/xgboost.pkl

Primary real-time model

models/xgboost.pkl

🔄 What Changed From the Previous Quick Reference

The previous document described a much broader preprocessing system than what is actually implemented. It has been corrected as follows:

Previous documentation

Updated documentation

528 lines claimed

Removed exact line-count claim

StandardScaler

❌ Removed

SMOTE in preprocessing

❌ Removed

NSL-KDD support

❌ Removed

Auto-detection of Label/Class/Attack

❌ Removed as an unverified claim

cleaned_data.csv as main output

Changed to selected_features.csv

Generic X_balanced, y_balanced, scaler output

Replaced with actual model-ready CSV workflow

15–20 minute fixed runtime

Removed

4–5 GB fixed memory claim

Removed

test_preprocessing.py as required current step

Removed

PREPROCESSING_GUIDE.md as required workflow

Removed from quick-start requirements

Correlation/RFE as next preprocessing step

Removed from this module's workflow

Generic CICIDS + NSL-KDD

Changed to CICIDS2017 only

Generic class counts

Replaced with actual verified counts

Generic dataset size

Replaced with actual verified output

Generic feature list

Replaced with the actual 30 model features

Generic output description

Replaced with actual selected_features.csv

No feature metadata

Added feature_names.pkl and feature_names.txt

No model relationship

Added connection to Random Forest, XGBoost, SHAP, and real-time detection

✅ Current Status

CICIDS2017 loading              ✅
8 CSV files                     ✅
Dataset combination             ✅
Duplicate removal               ✅
30 feature selection            ✅
Binary label generation         ✅
2,522,362 final rows            ✅
selected_features.csv           ✅
Feature metadata                ✅
Random Forest compatibility     ✅
XGBoost compatibility           ✅
CSV testing compatibility       ✅
Real-time detection compatibility ✅
SHAP compatibility              ✅

Current preprocessing output:

data/processed/selected_features.csv

Final dataset:

2,522,362 rows × 31 columns

Model input:

30 features

Labels:

BENIGN = 0
ATTACK = 1