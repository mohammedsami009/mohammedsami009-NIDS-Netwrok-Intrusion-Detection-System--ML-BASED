Network Intrusion Detection System using Machine Learning (NIDS-ML)

Final Year CSE Project — A machine-learning-based Network Intrusion Detection System for detecting suspicious network flows in real time and presenting the results through a Streamlit dashboard.

📋 Table of Contents

Project Overview

Features

Architecture

Dataset

Installation

Usage

Module Description

Models

Results

Technologies Used

Limitations

Future Enhancements

Contributors

License

🎯 Project Overview

This project implements a Network Intrusion Detection System (NIDS) that uses supervised machine learning to classify network traffic flows as either BENIGN or ATTACK.

The project has four main parts:

Dataset preprocessing — CICIDS2017 CSV files are combined, cleaned, deduplicated, and reduced to a selected set of 30 flow-level features.

Machine learning — Random Forest and XGBoost models are trained for binary intrusion detection.

Real-time detection — Scapy captures live network packets, constructs flows, extracts the required features, and sends them to the trained XGBoost model.

Dashboard — Streamlit displays live detections, source/destination information, prediction, confidence, packet counts, and detection statistics.

SHAP is also included for XGBoost model explainability.

Important: The project currently uses Random Forest and XGBoost only. SVM, KNN, Logistic Regression, Naive Bayes, LSTM, CNN, and CNN-LSTM are not part of the implemented model pipeline.

✨ Features

🎯 Binary Intrusion Classification — BENIGN vs ATTACK

🌲 Random Forest — trained and saved as a project model

⚡ XGBoost — primary model used for real-time detection

🔍 Real-Time Packet Capture — network traffic captured using Scapy

🔄 Flow-Based Detection — packets are grouped into network flows before classification

🧩 30 CICIDS2017 Features — the same feature set is maintained between training and detection

🧠 SHAP Explainability — XGBoost feature-contribution analysis

📊 Streamlit Dashboard — live monitoring and detection visualization

📝 Detection Logging — timestamp, flow, prediction, confidence, and packet information

🌐 Source/Destination Information — dashboard displays source IP/port and destination IP/port

🧪 CSV Evaluation — trained models can be tested against CICIDS2017 CSV files

🏗️ Architecture

                    ┌──────────────────────────────┐
                    │       CICIDS2017 CSVs        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    Data Preprocessing        │
                    │                              │
                    │ • Combine 8 CSV files        │
                    │ • Remove duplicates           │
                    │ • Clean numeric features     │
                    │ • Create binary labels        │
                    │ • Select 30 features          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     Processed Dataset        │
                    │   selected_features.csv      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │             Model Training               │
              │                                          │
              │       ┌──────────────┐ ┌──────────────┐  │
              │       │ Random       │ │   XGBoost    │  │
              │       │ Forest       │ │              │  │
              │       └──────────────┘ └──────────────┘  │
              └─────────────────────┬────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
          ┌────────────────────┐          ┌────────────────────┐
          │ CSV Evaluation     │          │ SHAP Explainability│
          │ test_csv.py        │          │ XGBoost            │
          └────────────────────┘          └────────────────────┘

                    LIVE NETWORK TRAFFIC
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Scapy Capture      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Flow Construction    │
                    │ & Feature Extraction │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      XGBoost         │
                    │ BENIGN / ATTACK      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Detection Log        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    │                      │
                    │ • Prediction         │
                    │ • Source IP/Port     │
                    │ • Destination IP/Port│
                    │ • Confidence         │
                    │ • Packet count       │
                    └──────────────────────┘

📊 Dataset

Primary Dataset: CICIDS2017

The project uses the CICIDS2017 network intrusion detection dataset.

The raw data contains network-flow records with approximately 80 columns, including the attack label.

The project uses these CICIDS2017 CSV files:

Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv

Preprocessing Result

The preprocessing pipeline produced:

Combined rows before duplicate removal: 2,830,743
Rows removed as duplicates:               308,381
Final processed rows:                    2,522,362
Features:                                       30
Label column:                               Label

The final processed dataset is saved as:

data/processed/selected_features.csv

Binary Labels

The original attack labels are converted to a binary classification target:

BENIGN → 0
ATTACK → 1

The processed dataset currently contains:

BENIGN: 2,096,484
ATTACK:   425,878

🧩 Selected Features

The trained models use the following 30 features:

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

Keeping the feature names and their order consistent is important because the trained models expect these 30 inputs.

🚀 Installation

Prerequisites

Python 3.10+

pip

Administrator privileges may be required for live packet capture on Windows

Npcap installed and configured for packet capture on Windows

Step 1: Clone the Repository

git clone <YOUR_REPOSITORY_URL>
cd NIDS-ML

Step 2: Create a Virtual Environment

Windows PowerShell

python -m venv venv
.env\Scripts\Activate.ps1

Linux/macOS

python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

💻 Usage

All commands below are executed from the src/ directory.

1. Preprocess CICIDS2017

Place the raw CICIDS2017 CSV files in:

data/raw/

Then run:

python data_preprocessing.py

This creates:

data/processed/selected_features.csv

The preprocessing process:

loads the available CICIDS2017 CSV files

combines them

removes duplicate records

selects the 30 project features

cleans invalid/missing feature values

converts the original labels into BENIGN / ATTACK

saves the processed dataset

2. Train the Models

Run:

python model_training.py

The implemented training pipeline trains:

Random Forest
XGBoost

The resulting models are saved under:

models/

The primary real-time model is:

models/xgboost.pkl

3. Test XGBoost on a CICIDS2017 CSV

For example:

python test_csv.py --model xgboost --csv "..\data
aw\Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"

PortScan can also be evaluated:

python test_csv.py --model xgboost --csv "..\data
aw\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

The test script reports:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Classification Report

Prediction distribution

Prediction results are saved under:

results/

4. Run Real-Time Detection

Run:

python realtime_detection.py

The detector:

captures live packets using Scapy

builds network flows

extracts the required flow features

loads the trained XGBoost model

predicts BENIGN or ATTACK

calculates prediction confidence

records packet counts

logs the source and destination endpoints

writes detection information for the dashboard

Example detection log:

BENIGN | 172.20.33.1:61531 -> 172.64.155.209:443 |
confidence=0.9983 | packets=413

A reverse-direction flow is also valid. For example:

BENIGN | 140.82.113.25:443 -> 172.20.33.1:61259 |
confidence=0.9999 | packets=3

This represents traffic travelling from the remote server back to the local client.

Stop live detection with:

Ctrl+C

5. Launch the Dashboard

From the src/ directory:

streamlit run dashboard.py

The dashboard displays information such as:

detection distribution

recent detections

prediction

source IP

source port

destination IP

destination port

confidence

packet count

live monitoring status

model information

6. SHAP Explainability

The project includes SHAP explainability for the XGBoost model.

The saved explainer is:

models/xgboost_shap_explainer.pkl

SHAP can be used to understand which input features contribute to model predictions.

📁 Module Description

data_preprocessing.py

Responsible for preparing CICIDS2017 data for model training.

Main responsibilities:

load all available raw CICIDS2017 CSV files

combine datasets

remove duplicate rows

select the 30 project features

clean feature values

create the binary target label

save selected_features.csv

model_training.py

Responsible for training and evaluating the implemented machine learning models.

Models

Random Forest

XGBoost

The trained models are saved using joblib.

Primary deployment model:

models/xgboost.pkl

test_csv.py

Evaluates a trained model against a CICIDS2017 CSV file.

It ensures that the model receives the same 30 features used during training and produces evaluation metrics and prediction results.

shap_explainability.py

Provides explainability for the tree-based XGBoost model using SHAP.

The module can be used for:

global feature importance

individual prediction explanations

SHAP summary visualizations

feature contribution analysis

The project currently focuses SHAP explainability on XGBoost.

realtime_detection.py

Responsible for live network monitoring and classification.

Main responsibilities:

capture packets using Scapy

construct network flows

aggregate flow statistics

generate the 30 required model features

run XGBoost predictions

calculate confidence

record packet counts

log source/destination IP addresses and ports

record detection results

dashboard.py

Provides the Streamlit user interface.

The dashboard currently presents:

detection distribution

recent detections

prediction

source IP

source port

destination IP

destination port

confidence

packet count

live detection information

model information

🤖 Models

The project currently uses two machine learning models:

Random Forest
XGBoost

Random Forest

Saved as:

models/random_forest.pkl

Used for binary classification of network flows.

XGBoost

Saved as:

models/xgboost.pkl

XGBoost is currently the primary model used for real-time detection.

Model Metadata

models/feature_names.pkl
models/feature_names.txt

These files maintain the 30-feature input definition.

SHAP Explainer

models/xgboost_shap_explainer.pkl

Used for XGBoost model explainability.

📈 Results

The model was evaluated using CICIDS2017 data.

XGBoost — DDoS CSV Evaluation

Test file:

Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv

Results from the current project run:

Metric

Result

Accuracy

99.9189%

Precision

99.9531%

Recall

99.9039%

F1-score

99.9285%

ROC-AUC

99.9971%

Confusion Matrix:

                 Predicted
                 BENIGN  ATTACK
Actual BENIGN      97658      60
Actual ATTACK        123  127904

XGBoost — PortScan CSV Evaluation

Test file:

Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

Results from the current project run:

Metric

Result

Accuracy

99.9613%

Precision

99.9522%

Recall

99.9780%

F1-score

99.9651%

ROC-AUC

99.9978%

Confusion Matrix:

                 Predicted
                 BENIGN  ATTACK
Actual BENIGN     127461      76
Actual ATTACK         35  158895

Evaluation note: These results are evaluations on CICIDS2017 CSV data. They should not be interpreted as proof that the model has the same accuracy on arbitrary real-world network traffic. Real-time traffic can differ substantially from the training dataset.

🛠️ Technologies Used

Programming

Python 3.10+

Machine Learning

scikit-learn

XGBoost

Random Forest

SHAP

joblib

Data Processing

pandas

NumPy

Network Monitoring

Scapy

Npcap on Windows

Visualization / Dashboard

Streamlit

Matplotlib

Plotly

Project Utilities

Python logging

pathlib

dataclasses

⚠️ Limitations

The current implementation has several important limitations:

Binary classification — the deployed model predicts BENIGN or ATTACK; it does not currently classify every attack family separately.

Dataset dependence — the models were trained on CICIDS2017 flow data, so performance on different networks and attack distributions may differ.

Feature compatibility — real-time detection must generate the same 30 features expected by the trained model.

Offline evaluation vs real traffic — very high CICIDS2017 test metrics do not guarantee equivalent real-world detection accuracy.

No claim of zero-day detection — the current supervised model should not be presented as a guaranteed detector of previously unseen attacks.

No deep learning pipeline — LSTM, CNN, and other deep-learning models are not currently implemented in the active training pipeline.

🔮 Future Enhancements

Possible future improvements include:

Multi-class attack classification instead of only BENIGN / ATTACK

Evaluation on additional independent datasets

Better flow-feature generation for live traffic

Threshold tuning for real-time alerts

More extensive SHAP visualizations in the dashboard

Automated model retraining with new labeled traffic

REST API integration

SIEM integration

Improved alert management

Evaluation against controlled, authorized attack traffic in a lab environment

👥 Contributors

[Your Name] — Lead Developer

[Teammate Name] — [Role]

[Teammate Name] — [Role]

Supervisor: [Supervisor Name]

Institution: [Your College/University]

Year: 2026

📄 License

This project is licensed under the MIT License.

See the LICENSE file for details.

🙏 Acknowledgments

Canadian Institute for Cybersecurity for the CICIDS2017 dataset

Scapy developers and contributors

XGBoost and scikit-learn communities

SHAP developers and contributors

Streamlit developers and contributors

Open-source software community

📝 Citation

If you publish research based on this project, replace the placeholders below with the actual author and repository information:

@misc{nids_ml_2026,
  author = {Your Name},
  title = {Network Intrusion Detection System using Machine Learning},
  year = {2026},
  publisher = {GitHub},
  howpublished = {GitHub Repository}
}

Last Updated: August 2026