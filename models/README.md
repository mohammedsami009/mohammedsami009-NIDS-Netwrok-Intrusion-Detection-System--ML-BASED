Models Directory

This directory contains the trained machine learning models and model metadata used by the NIDS-ML project.

Directory Structure

models/
├── random_forest.pkl
├── xgboost.pkl
├── best_model.pkl
├── feature_names.pkl
├── feature_names.txt
└── xgboost_shap_explainer.pkl

Model Files

random_forest.pkl

Trained Random Forest classifier for binary network intrusion detection.

xgboost.pkl

Trained XGBoost classifier and the primary model used by the real-time detection system.

best_model.pkl

Saved best-performing model from the training process.

feature_names.pkl

Stores the feature metadata required to keep feature order consistent between training, CSV testing, and real-time detection. It contains the 30 feature names, feature count, and label mapping.

feature_names.txt

Human-readable list of the 30 features used by the trained models.

xgboost_shap_explainer.pkl

Saved SHAP explainer associated with the XGBoost model for model explainability and feature-contribution analysis.

Model Features

The models use these 30 CICIDS2017 flow-level features:

Total_Backward_Packets

Total_Length_of_Bwd_Packets

Fwd_Packet_Length_Max

Fwd_Packet_Length_Std

Bwd_Packet_Length_Max

Bwd_Packet_Length_Min

Bwd_Packet_Length_Mean

Bwd_Packet_Length_Std

Flow_IAT_Std

Flow_IAT_Max

Flow_IAT_Min

Fwd_IAT_Mean

Fwd_IAT_Std

Fwd_IAT_Max

Fwd_IAT_Min

Bwd_IAT_Total

Max_Packet_Length

Packet_Length_Std

Packet_Length_Variance

PSH_Flag_Count

Average_Packet_Size

Avg_Bwd_Segment_Size

Subflow_Fwd_Bytes

Subflow_Bwd_Packets

Subflow_Bwd_Bytes

Init_Win_bytes_backward

Active_Mean

Active_Max

Active_Min

Idle_Max

Label Mapping

BENIGN → 0
ATTACK → 1

Loading the Models

import joblib

xgboost_model = joblib.load("models/xgboost.pkl")
random_forest_model = joblib.load("models/random_forest.pkl")

Load feature metadata:

feature_metadata = joblib.load("models/feature_names.pkl")

feature_names = feature_metadata["feature_names"]
feature_count = feature_metadata["feature_count"]
label_mapping = feature_metadata["label_mapping"]

Project Pipeline

Network Packets / PCAP
        ↓
Packet Capture
        ↓
Bidirectional Flow Construction
        ↓
Flow Feature Extraction
        ↓
30 CICIDS2017 Features
        ↓
Random Forest / XGBoost
        ↓
BENIGN / ATTACK
        ↓
Detection Log / Dashboard

XGBoost is currently the primary deployment model for real-time detection.

Training and Evaluation

The models were trained using the processed CICIDS2017 dataset.

The evaluation includes:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

The XGBoost model was also evaluated separately on CICIDS2017 DDoS and PortScan data.

Explainability

SHAP is used with the XGBoost model to provide model interpretability and feature-contribution explanations.

The saved xgboost_shap_explainer.pkl contains the SHAP explainer used for the XGBoost model.

Important Note

The model files contain the trained models and metadata. Real-time detection depends on the flow-feature extraction pipeline producing the same 30 feature columns in the same order expected by the trained model.