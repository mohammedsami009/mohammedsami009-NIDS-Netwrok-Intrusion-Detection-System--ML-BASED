🎉 Part 5: Explainability + Real-Time Detection — Completion Report

📋 Overview

Part: 5 — Explainability + Real-Time Detection

Current implementation files:

src/shap_explainability.py
src/realtime_detection.py

Status: ✅ Implemented and integrated with the current NIDS-ML pipeline

The current implementation provides:

SHAP-based explainability for the XGBoost model

Real-time network packet capture using Scapy

Flow construction and flow-level feature extraction

Binary prediction: BENIGN / ATTACK

Prediction confidence

Source IP / source port

Destination IP / destination port

Packet counts

Detection logging

Streamlit dashboard integration

Important: The original Part 5 document described a separate explainability_realtime.py module, Random Forest SHAP analysis, synthetic fallback traffic, 10,000-sample SHAP analysis, and a number of generated files that are not part of the current implementation. This document reflects the implementation that is actually being used now.

🎯 Part 5 Implementation

The current Part 5 consists of two major components:

┌─────────────────────────────────────────────┐
│              PART 5                         │
├───────────────────────┬─────────────────────┤
│   SHAP Explainability │ Real-Time Detection  │
│                       │                     │
│   XGBoost             │ Scapy               │
│   TreeExplainer       │ Flow extraction     │
│   Feature importance  │ XGBoost prediction  │
│   Contributions       │ Detection logging   │
└───────────────────────┴─────────────────────┘

🔍 Section 1: SHAP Explainability

Objective

SHAP is used to explain why the XGBoost model makes a particular prediction.

The model is a tree-based classifier, so SHAP's TreeExplainer is appropriate for the XGBoost model.

Current saved explainer:

models/xgboost_shap_explainer.pkl

Model Used

The current explainability pipeline is associated with:

models/xgboost.pkl

The project does not currently use Random Forest as the primary SHAP model.

The distinction is:

Random Forest
    │
    └── Comparison / trained model

XGBoost
    │
    ├── Primary real-time model
    └── SHAP explainability model

🧠 How SHAP Works in This Project

The model receives the 30 selected network-flow features:

30 CICIDS2017 Features
        │
        ▼
     XGBoost
        │
        ▼
BENIGN / ATTACK
        │
        ▼
    SHAP Values
        │
        ▼
Feature Contributions

A SHAP value indicates how much a feature contributes toward a particular model output.

For example, conceptually:

Feature                     Contribution
------------------------------------------------
Packet_Length_Std              + attack
Flow_IAT_Max                  + attack
Bwd_Packet_Length_Mean        - attack
Active_Mean                   + benign

The exact contribution values should be taken from the generated SHAP results rather than being hard-coded in documentation.

📊 Explainability Outputs

The current project supports SHAP-based feature importance and prediction explanation through the XGBoost SHAP explainer.

The intended output types include:

Global feature importance

Feature contribution analysis

Individual prediction explanations

SHAP summary visualization

SHAP bar visualization

Feature contribution information for dashboard/reporting use

Saved explainer:

models/xgboost_shap_explainer.pkl

Do not claim specific top-ranked features unless they have actually been generated and verified from the current SHAP output.

📡 Section 2: Real-Time Detection

Objective

The real-time detection module captures actual network traffic and uses the trained XGBoost model to classify network flows.

Implementation:

src/realtime_detection.py

The current system has already been run successfully on live traffic.

Example output observed during testing:

BENIGN | 172.20.33.1:52365 -> 34.54.84.110:443 |
confidence=1.0000 | packets=2

Another valid reverse-direction flow:

BENIGN | 140.82.113.25:443 -> 172.20.33.1:61259 |
confidence=0.9999 | packets=3

The second example is not an error. It represents traffic travelling from the remote server back to the local machine.

📦 Packet Capture

The system uses:

Scapy

for live packet capture.

The real-time pipeline is:

Network Interface
       │
       ▼
     Scapy
       │
       ▼
Live Packets
       │
       ▼
Flow Aggregation
       │
       ▼
Feature Extraction
       │
       ▼
XGBoost

On Windows, packet capture requires an appropriate packet-capture driver such as Npcap.

🔄 Flow Construction

The system does not simply classify an individual raw packet using the CICIDS2017 model.

Instead, packets are aggregated into network flows and flow-level statistics are generated.

A flow contains information such as:

Source IP
Source Port
Destination IP
Destination Port
Protocol
Packet count
Packet lengths
Packet timing
Forward statistics
Backward statistics

These statistics are used to construct the model input.

🧩 Feature Extraction

The real-time detector generates the same 30-feature input contract expected by the trained model.

The 30 features are:

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

The feature order must remain consistent with:

models/feature_names.pkl
models/feature_names.txt

🤖 Section 3: Real-Time Classification

The primary deployed model is:

models/xgboost.pkl

The classification task is binary:

0 → BENIGN
1 → ATTACK

The detector also obtains a probability/confidence value from the model.

Example:

BENIGN | confidence=1.0000

The confidence value is a model probability output. It should not be described as a guaranteed probability that the traffic is objectively benign.

📝 Section 4: Detection Logging

The real-time detector records information about detected flows.

The information used by the dashboard includes:

Timestamp
Prediction
Flow
Source IP
Source Port
Destination IP
Destination Port
Confidence
Packets

Example dashboard record:

Timestamp              Prediction  Source IP      Source Port
2026-08-26 01:47:56    BENIGN      172.20.33.1    52507

Destination IP         Destination Port   Confidence   Packets
172.217.115.4          443                1.0000       2

This allows the user to identify not only the classification result but also the network endpoint involved.

📊 Section 5: Streamlit Dashboard

The real-time results are displayed through:

src/dashboard.py

The dashboard currently provides information such as:

Detection Distribution

Shows the distribution of:

BENIGN
ATTACK

Recent Detections

The recent detection table displays:

Timestamp
Prediction
Flow
Source IP
Source Port
Dest IP
Dest Port
Confidence
Packets

Model Information

The current primary model is shown as:

XGBoost

Live Monitoring

The dashboard is connected to the real-time detection pipeline and displays current detection information.

🔗 Complete Real-Time Pipeline

┌──────────────────────────┐
│    Live Network Traffic  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│          Scapy           │
│     Packet Capture       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     Flow Construction    │
│      & Aggregation       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Feature Extraction    │
│      30 Features         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│         XGBoost          │
│                          │
│  BENIGN / ATTACK         │
└────────────┬─────────────┘
             │
             ├──────────────► Confidence
             │
             ▼
┌──────────────────────────┐
│     Detection Logging    │
│                          │
│ IPs / Ports / Packets    │
│ Prediction / Confidence  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Streamlit Dashboard   │
└──────────────────────────┘

🧪 Section 6: Verification Performed

The project has been tested in multiple stages.

CSV Model Testing

XGBoost was evaluated against CICIDS2017 DDoS and PortScan CSV files.

DDoS

Accuracy : 99.9189%
Precision: 99.9531%
Recall   : 99.9039%
F1-score : 99.9285%
ROC-AUC  : 99.9971%

PortScan

Accuracy : 99.9613%
Precision: 99.9522%
Recall   : 99.9780%
F1-score : 99.9651%
ROC-AUC  : 99.9978%

These results are CICIDS2017 dataset evaluations and should not be represented as guaranteed accuracy on arbitrary real-world network traffic.

🌐 Real-Time Verification

The real-time detector was successfully started using:

python realtime_detection.py

The model loaded successfully:

Loading model:
models\xgboost.pkl

Model loaded successfully:
XGBClassifier

The detector then captured live traffic and produced predictions such as:

BENIGN | 172.20.33.1:61531 -> 172.64.155.209:443 |
confidence=0.9983 | packets=413

and:

BENIGN | 140.82.113.25:443 -> 172.20.33.1:61259 |
confidence=0.9999 | packets=3

This confirms that the live packet-capture → flow → feature extraction → XGBoost prediction path is operational.

⚠️ Section 7: Important Limitations

The current implementation has several limitations that should be stated honestly in a project report.

1. CICIDS2017 Dependency

The model was trained using CICIDS2017.

Real-world traffic can differ from the dataset.

Therefore:

CICIDS2017 accuracy
        ≠
Guaranteed real-world accuracy

2. Binary Classification

The current model outputs:

BENIGN
ATTACK

It does not currently provide a verified multi-class attack-family prediction pipeline.

3. Feature Compatibility

Real-time detection depends on producing the same 30 model features used during training.

4. Confidence Is Model Output

A confidence value such as:

0.9999

is the model's probability output. It does not prove that the classification is correct.

5. Real-Time Traffic Validation

The current live testing demonstrates that the system can capture and classify traffic. It does not establish real-world attack-detection accuracy.

6. No Zero-Day Claim

The project should not claim guaranteed detection of previously unseen or zero-day attacks.

📁 Current Part 5 Files

The relevant current project files are:

src/
├── shap_explainability.py
├── realtime_detection.py
└── dashboard.py

models/
├── xgboost.pkl
└── xgboost_shap_explainer.pkl

Related model metadata:

models/
├── feature_names.pkl
└── feature_names.txt

🔧 Dependencies

The current implementation uses the packages required by the active codebase, including:

pandas
numpy
scikit-learn
xgboost
shap
scapy
joblib
streamlit
plotly
matplotlib

The exact dependency list should be taken from the project's current requirements.txt rather than copied from the original Part 5 report.

🚀 How to Run

Start Real-Time Detection

From src/:

python realtime_detection.py

The detector loads:

../models/xgboost.pkl

and begins capturing live traffic.

Stop it with:

Ctrl+C

Start Dashboard

From src/:

streamlit run dashboard.py

The dashboard displays the latest detection information.

Run CSV Evaluation

Example:

python test_csv.py --model xgboost --csv "..\data\raw\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

🎓 Academic Contribution

The implemented Part 5 contributes the following capabilities to the project:

1. Explainable Intrusion Detection

SHAP provides a method for inspecting which network-flow features influence XGBoost predictions.

2. Real-Time Network Monitoring

Scapy enables packet capture from the live network interface.

3. Flow-Based ML Detection

Raw packets are converted into flow-level statistics compatible with the trained model.

4. Detection Transparency

The dashboard exposes:

Source
Destination
Prediction
Confidence
Packet Count
Timestamp

rather than showing only a binary alert.

5. End-to-End Integration

The project connects:

Dataset
  ↓
Preprocessing
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Real-Time Detection
  ↓
Dashboard
  ↓
Explainability

✅ Part 5 Completion Checklist

Explainability

XGBoost SHAP explainer available

SHAP model integration

Feature contribution analysis

Explainability module integrated with project models

Real-Time Detection

Scapy packet capture

Live flow construction

30-feature extraction

XGBoost model loading

BENIGN / ATTACK prediction

Confidence calculation

Source IP extraction

Source port extraction

Destination IP extraction

Destination port extraction

Packet count

Detection logging

Dashboard

Detection distribution

Recent detections

Prediction display

Source / destination information

Confidence display

Packet count display

XGBoost model information

Validation

DDoS CSV evaluation

PortScan CSV evaluation

Live traffic capture tested

Live predictions verified

Dashboard output verified

📌 Final Status

PART 5
══════════════════════════════════════════

SHAP Explainability       ✅ Implemented
XGBoost Integration       ✅ Implemented
Live Packet Capture       ✅ Implemented
Flow Construction         ✅ Implemented
30-Feature Extraction     ✅ Implemented
Real-Time Prediction      ✅ Implemented
Confidence Calculation    ✅ Implemented
IP / Port Display         ✅ Implemented
Detection Logging         ✅ Implemented
Streamlit Dashboard       ✅ Implemented
CSV Evaluation            ✅ Verified

Primary Real-Time Model:
XGBoost

Classification:
BENIGN / ATTACK

Dataset:
CICIDS2017

Status:
COMPLETE