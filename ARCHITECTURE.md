📐 NIDS-ML System Architecture

This document describes the architecture of the implemented NIDS-ML system.

The current project uses:

CICIDS2017 for offline training and evaluation

30 selected flow-level features

Random Forest and XGBoost

XGBoost as the primary real-time detection model

Scapy for live network packet capture

SHAP for XGBoost explainability

Streamlit for monitoring and visualization

The system performs binary classification:

BENIGN → 0
ATTACK → 1

High-Level Architecture

┌──────────────────────────────────────────────────────────────────────┐
│                NETWORK INTRUSION DETECTION SYSTEM                    │
│                       Machine Learning                               │
└──────────────────────────────────────────────────────────────────────┘

                         OFFLINE PIPELINE
                         ================

┌──────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCE                                  │
│                                                                      │
│                     CICIDS2017 CSV Files                             │
│                       8 CSV files                                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                                │
│                                                                      │
│  • Load CICIDS2017 CSV files                                        │
│  • Combine datasets                                                 │
│  • Remove duplicate rows                                            │
│  • Clean NaN / infinite / invalid numeric values                    │
│  • Select 30 flow-level features                                    │
│  • Convert labels to BENIGN / ATTACK                                │
│                                                                      │
│  Output: data/processed/selected_features.csv                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING                                   │
│                                                                      │
│             ┌────────────────┐    ┌────────────────┐                 │
│             │ Random Forest  │    │    XGBoost     │                 │
│             └────────────────┘    └────────────────┘                 │
│                                                                      │
│  Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC          │
│                                                                      │
│  Saved models:                                                       │
│  • models/random_forest.pkl                                         │
│  • models/xgboost.pkl                                               │
│  • models/best_model.pkl                                            │
└───────────────────────┬─────────────────────┬────────────────────────┘
                        │                     │
                        ▼                     ▼
             ┌───────────────────┐   ┌─────────────────────┐
             │ CSV Evaluation    │   │ SHAP Explainability │
             │ test_csv.py       │   │ XGBoost             │
             └───────────────────┘   └─────────────────────┘


                        REAL-TIME PIPELINE
                        ==================

┌──────────────────────────────────────────────────────────────────────┐
│                       LIVE NETWORK TRAFFIC                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SCAPY PACKET CAPTURE                            │
│                                                                      │
│  Captures live network packets from the selected interface          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  FLOW CONSTRUCTION / AGGREGATION                     │
│                                                                      │
│  • Source IP / Port                                                 │
│  • Destination IP / Port                                            │
│  • Packet direction                                                 │
│  • Packet length                                                    │
│  • Packet timing                                                    │
│  • TCP information                                                  │
│  • Forward/backward flow statistics                                 │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FLOW FEATURE EXTRACTION                            │
│                                                                      │
│          Generate the same 30 features used in training              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       XGBOOST MODEL                                  │
│                                                                      │
│                    BENIGN  /  ATTACK                                 │
│                    + confidence score                                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DETECTION LOGGING                                │
│                                                                      │
│  • Timestamp                                                        │
│  • Prediction                                                       │
│  • Source IP / Port                                                 │
│  • Destination IP / Port                                            │
│  • Confidence                                                       │
│  • Packet count                                                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                              │
│                                                                      │
│  • Overview                                                         │
│  • Model Performance                                                │
│  • Live Detection                                                   │
│  • Attack Evaluation                                                │
│  • Features                                                         │
│  • System Status                                                    │
│                                                                      │
│  Live table displays:                                               │
│  Timestamp | Prediction | Flow | Source IP | Source Port             │
│  Dest IP | Dest Port | Confidence | Packets                         │
└──────────────────────────────────────────────────────────────────────┘

Offline Data Processing Flow

CICIDS2017 CSV Files
        │
        ▼
Load 8 CSV Files
        │
        ▼
Combine Dataset
        │
        ▼
Remove Duplicate Rows
        │
        ▼
Select 30 Features
        │
        ▼
Clean Invalid / Missing Values
        │
        ▼
Create Binary Label
BENIGN = 0
ATTACK = 1
        │
        ▼
selected_features.csv
        │
        ▼
Train/Test Processing
        │
        ▼
Random Forest + XGBoost
        │
        ▼
Evaluation + Saved Models

Current preprocessing output

Raw combined records : 2,830,743
Duplicate rows removed: 308,381
Final records         : 2,522,362
Features              : 30

BENIGN : 2,096,484
ATTACK :   425,878

Selected Feature Layer

The project does not dynamically run RFE, PCA, Chi-square, ANOVA, or multiple feature-selection algorithms during the active pipeline.

Instead, the current implementation uses a fixed set of 30 selected CICIDS2017 flow features:

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

Feature metadata is stored in:

models/feature_names.pkl
models/feature_names.txt

This ensures that training, CSV testing, and real-time inference use the same feature names and order.

Model Training Layer

Only two machine-learning models are part of the current implementation:

┌──────────────────────────────────────────────┐
│              MACHINE LEARNING                │
├──────────────────────┬───────────────────────┤
│    Random Forest     │       XGBoost         │
│                      │                       │
│ random_forest.pkl    │ xgboost.pkl           │
│                      │                       │
│ Comparison model     │ Primary realtime     │
│                      │ detection model       │
└──────────────────────┴───────────────────────┘

The project does not currently train:

SVM

KNN

Logistic Regression

Naive Bayes

LSTM

CNN

CNN-LSTM

Evaluation metrics

The training/evaluation pipeline uses:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Classification Report

CSV Evaluation Layer

test_csv.py is used to test a trained model directly against CICIDS2017 CSV files.

CICIDS2017 CSV
      │
      ▼
Column normalization
      │
      ▼
Select same 30 features
      │
      ▼
Load XGBoost model
      │
      ▼
Predict every flow
      │
      ▼
Compare prediction with Label
      │
      ▼
Metrics + Confusion Matrix
      │
      ▼
Save predictions to results/

Verified DDoS evaluation

Accuracy : 99.9189%
Precision: 99.9531%
Recall   : 99.9039%
F1-score : 99.9285%
ROC-AUC  : 99.9971%

Verified PortScan evaluation

Accuracy : 99.9613%
Precision: 99.9522%
Recall   : 99.9780%
F1-score : 99.9651%
ROC-AUC  : 99.9978%

These are CICIDS2017 evaluations and should not be presented as guaranteed real-world accuracy.

Real-Time Detection Layer

The real-time pipeline is implemented in:

src/realtime_detection.py

Data flow

Live Network
     │
     ▼
Scapy
     │
     ▼
Packets
     │
     ▼
Flow Aggregation
     │
     ▼
30 Flow Features
     │
     ▼
XGBoost
     │
     ├────────── BENIGN
     │
     └────────── ATTACK
     │
     ▼
Confidence Score
     │
     ▼
Detection Log
     │
     ▼
Streamlit Dashboard

Example:

BENIGN | 172.20.33.1:52507 -> 172.217.115.4:443 |
confidence=1.0000 | packets=2

The reverse direction is also valid:

BENIGN | 140.82.113.25:443 -> 172.20.33.1:61259 |
confidence=0.9999 | packets=3

The source and destination fields describe the observed flow direction; a remote server can therefore appear as the source when traffic is travelling back toward the client.

Explainability Layer

SHAP is used for XGBoost explainability.

Trained XGBoost
      │
      ▼
TreeExplainer
      │
      ▼
SHAP Values
      │
      ├── Global Feature Importance
      │
      └── Individual Feature Contributions

Saved explainer:

models/xgboost_shap_explainer.pkl

The project should not claim KernelExplainer, DeepExplainer, or deep-learning explainability unless those paths are actually implemented and used.

Dashboard Layer

The dashboard is implemented using Streamlit:

src/dashboard.py

Current navigation:

🏠 Overview
📊 Model Performance
🔴 Live Detection
📈 Attack Evaluation
🧠 Features
⚙️ System

Live Detection table

Timestamp
Prediction
Flow
Source IP
Source Port
Dest IP
Dest Port
Confidence
Packets

The dashboard reads real detection information generated by the detection pipeline rather than displaying placeholder network detections.

Output and Storage

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
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl
│   ├── best_model_metadata.txt
│   ├── feature_names.pkl
│   ├── feature_names.txt
│   └── xgboost_shap_explainer.pkl
│
├── results/
│   └── *_predictions.csv
│
├── logs/
│   └── detection / training logs
│
└── src/
    ├── data_preprocessing.py
    ├── model_training.py
    ├── test_csv.py
    ├── realtime_detection.py
    ├── shap_explainability.py
    └── dashboard.py

Technology Stack

┌────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Data Processing                                           │
│  • Pandas                                                  │
│  • NumPy                                                   │
│                                                            │
│  Machine Learning                                          │
│  • Scikit-learn                                            │
│  • XGBoost                                                 │
│  • Random Forest                                           │
│                                                            │
│  Explainability                                            │
│  • SHAP                                                    │
│                                                            │
│  Network Capture                                           │
│  • Scapy                                                   │
│  • Npcap (Windows packet capture)                          │
│                                                            │
│  Dashboard / Visualization                                 │
│  • Streamlit                                               │
│  • Plotly                                                  │
│  • Matplotlib                                              │
│                                                            │
│  Model Persistence                                         │
│  • Joblib                                                  │
│                                                            │
│  Utilities                                                 │
│  • Logging                                                 │
│  • pathlib                                                 │
│  • dataclasses                                             │
│                                                            │
└────────────────────────────────────────────────────────────┘

No TensorFlow, PyTorch, LSTM, CNN, or other deep-learning component is part of the current implemented architecture.

System Requirements

The exact minimum hardware requirement has not been benchmarked formally, so fixed hardware claims should not be presented as measured requirements.

A practical development environment is:

Operating System:
• Windows 10/11
• Linux
• macOS

Software:
• Python 3.10+
• pip
• Required Python packages
• Npcap on Windows for Scapy live capture

Recommended for full CICIDS2017 training:
• Multi-core CPU
• 16 GB RAM or more
• Sufficient disk space for raw/processed CICIDS2017 files

A GPU is not required for the current Random Forest + XGBoost implementation.

Important Architecture Constraint

The most important connection between offline training and real-time inference is:

             TRAINING
CICIDS2017 → 30 FEATURES
                  │
                  │ SAME FEATURE DEFINITIONS
                  │ SAME FEATURE ORDER
                  ▼
LIVE PACKETS → FLOW AGGREGATION → 30 FEATURES → XGBOOST

The real-time detector is only compatible with the trained model when the live flow extractor calculates the same feature definitions expected by the model.

Current Project Scope

IMPLEMENTED
───────────
✓ CICIDS2017 preprocessing
✓ 30 flow-level features
✓ Binary BENIGN / ATTACK labels
✓ Random Forest
✓ XGBoost
✓ CSV model evaluation
✓ Live Scapy packet capture
✓ Flow aggregation
✓ Real-time XGBoost prediction
✓ Confidence scores
✓ Source/destination IP and ports
✓ Detection logging
✓ Streamlit dashboard
✓ SHAP support for XGBoost