"""
NIDS-ML Dashboard
=================

Streamlit dashboard for the current NIDS-ML project.

Models:
    - Random Forest
    - XGBoost

Primary XGBoost evaluation:
    Accuracy  : 99.7637%
    Precision : 99.4815%
    Recall    : 99.1171%
    F1-score  : 99.2990%
    ROC-AUC   : 99.9870%

Attack-specific CICIDS2017 evaluations:
    DDoS:
        Accuracy  : 99.9189%
        Precision : 99.9531%
        Recall    : 99.9039%
        F1-score  : 99.9285%
        ROC-AUC   : 99.9971%

    PortScan:
        Accuracy  : 99.9613%
        Precision : 99.9522%
        Recall    : 99.9780%
        F1-score  : 99.9651%
        ROC-AUC   : 99.9978%

Run from the project root or src directory:

    streamlit run src/dashboard.py
"""

from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="NIDS-ML | AI Cybersecurity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .status-box {
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    .model-box {
        padding: 1rem;
        border: 1px solid #d9dde5;
        border-radius: 0.6rem;
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# ACTUAL PROJECT RESULTS
# =============================================================================

OVERALL_RESULTS = {
    "Random Forest": {
        "Accuracy": 99.7046,
        "Precision": 98.9266,
        "Recall": 99.3284,
        "F1-Score": 99.1271,
        "ROC-AUC": 99.9801,
    },
    "XGBoost": {
        "Accuracy": 99.7637,
        "Precision": 99.4815,
        "Recall": 99.1171,
        "F1-Score": 99.2990,
        "ROC-AUC": 99.9870,
    },
}

ATTACK_RESULTS = {
    "DDoS": {
        "Accuracy": 99.9189,
        "Precision": 99.9531,
        "Recall": 99.9039,
        "F1-Score": 99.9285,
        "ROC-AUC": 99.9971,
    },
    "PortScan": {
        "Accuracy": 99.9613,
        "Precision": 99.9522,
        "Recall": 99.9780,
        "F1-Score": 99.9651,
        "ROC-AUC": 99.9978,
    },
}


FEATURES = [
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
# HELPERS
# =============================================================================

def model_exists(name: str) -> bool:
    return (
        MODELS_DIR / f"{name}.pkl"
    ).exists()


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def load_feature_metadata():
    path = MODELS_DIR / "feature_names.pkl"

    if not path.exists():
        return FEATURES

    try:
        metadata = joblib.load(path)

        if isinstance(metadata, dict):
            names = metadata.get("feature_names")

            if names:
                return list(names)

        if isinstance(metadata, list):
            return metadata

    except Exception:
        pass

    return FEATURES


def load_prediction_files():
    """
    Load prediction CSVs generated by test_csv.py.

    These files contain:
        actual_label
        predicted_label
        attack_probability
    """

    files = sorted(
        RESULTS_DIR.glob("*_predictions.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    frames = []

    for file in files:

        try:
            df = pd.read_csv(file)

            if "predicted_label" not in df.columns:
                continue

            df["source_file"] = file.name
            frames.append(df)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def find_latest_detection_log():
    candidates = [
        LOGS_DIR / "realtime_detection.log",
        LOGS_DIR / "detection.log",
        LOGS_DIR / "ids.log",
        LOGS_DIR / "realtime.log",
    ]

    existing = [
        p for p in candidates
        if p.exists()
    ]

    if not existing:
        return None

    return max(
        existing,
        key=lambda p: p.stat().st_mtime,
    )


def read_detection_log():
    """
    Parse simple lines produced by realtime_detection.py.

    Log line shape (both ATTACK and BENIGN use the same shape):
        <asctime> | <LEVEL> | LABEL | src_ip:src_port -> dst_ip:dst_port | confidence=... | packets=...

    e.g.
        2026-08-26 10:15:30,123 | WARNING  | ATTACK | 192.168.1.5:51422 -> 10.0.0.8:443 | confidence=0.9821 | packets=15
        2026-08-26 10:15:31,456 | INFO     | BENIGN | 192.168.1.5:51422 -> 10.0.0.8:443 | confidence=0.1203 | packets=8
    """

    path = find_latest_detection_log()

    if path is None:
        return pd.DataFrame()

    rows = []

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

    except Exception:
        return pd.DataFrame()

    for line in lines:

        if " | BENIGN | " not in line and " | ATTACK | " not in line:
            continue

        label = (
            "ATTACK"
            if " | ATTACK | " in line
            else "BENIGN"
        )

        try:
            timestamp_text = line[:23]
            timestamp = datetime.strptime(
                timestamp_text,
                "%Y-%m-%d %H:%M:%S,%f",
            )
        except Exception:
            timestamp = None

        confidence = np.nan

        if "confidence=" in line:

            try:
                value = (
                    line.split("confidence=", 1)[1]
                    .split("|", 1)[0]
                    .strip()
                )

                confidence = float(value)

            except Exception:
                pass

        packets = np.nan

        if "packets=" in line:

            try:
                value = (
                    line.split("packets=", 1)[1]
                    .split("|", 1)[0]
                    .strip()
                )

                packets = int(value)

            except Exception:
                pass

        # ---------------------------------------------------------------
        # FIX: parts[2] is the LABEL ("ATTACK"/"BENIGN"), not the flow.
        # The "src:port -> dst:port" segment is parts[3]. Previously the
        # "Flow" column just repeated the label.
        # ---------------------------------------------------------------
        parts = line.split(" | ")

        flow = ""

        if len(parts) >= 4:
            flow = parts[3].strip()

        src_ip = src_port = dst_ip = dst_port = None

        if " -> " in flow:
            try:
                src, dst = flow.split(" -> ")
                src_ip, src_port = src.rsplit(":", 1)
                dst_ip, dst_port = dst.rsplit(":", 1)
            except Exception:
                pass

        rows.append(
            {
                "Timestamp": timestamp,
                "Prediction": label,
                "Flow": flow,
                "Source IP": src_ip,
                "Source Port": src_port,
                "Dest IP": dst_ip,
                "Dest Port": dst_port,
                "Confidence": confidence,
                "Packets": packets,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():

    st.sidebar.title("🛡️ NIDS-ML")

    st.sidebar.caption(
        "AI-Driven Cybersecurity Threat Detection"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Model Performance",
            "🔴 Live Detection",
            "📈 Attack Evaluation",
            "🧠 Features",
            "⚙️ System",
        ],
    )

    st.sidebar.divider()

    st.sidebar.subheader("Models")

    rf_ok = model_exists("random_forest")
    xgb_ok = model_exists("xgboost")

    st.sidebar.write(
        f"{'🟢' if rf_ok else '🔴'} Random Forest"
    )

    st.sidebar.write(
        f"{'🟢' if xgb_ok else '🔴'} XGBoost"
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Current primary model"
    )

    st.sidebar.success(
        "XGBoost"
    )

    return page


# =============================================================================
# OVERVIEW
# =============================================================================

def render_overview():

    st.markdown(
        '<div class="main-title">🛡️ Network Intrusion Detection System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "AI-driven network threat and anomaly detection using "
        "CICIDS2017 flow features"
        "</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Main metrics
    # -------------------------------------------------------------------------

    result = OVERALL_RESULTS["XGBoost"]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{result['Accuracy']:.4f}%",
    )

    c2.metric(
        "Precision",
        f"{result['Precision']:.4f}%",
    )

    c3.metric(
        "Recall",
        f"{result['Recall']:.4f}%",
    )

    c4.metric(
        "F1-Score",
        f"{result['F1-Score']:.4f}%",
    )

    c5.metric(
        "ROC-AUC",
        f"{result['ROC-AUC']:.4f}%",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Architecture
    # -------------------------------------------------------------------------

    st.subheader("System Architecture")

    st.code(
        """
Network Traffic / PCAP
          │
          ▼
      Packet Capture
          │
          ▼
   Bidirectional Flows
          │
          ▼
  30 CICIDS2017 Features
          │
          ▼
       XGBoost
          │
          ├──────────────┐
          ▼              ▼
       BENIGN         ATTACK
          │              │
          └──────┬───────┘
                 ▼
          Dashboard / Logs
        """,
        language="text",
    )

    st.subheader("Model Status")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="model-box">',
            unsafe_allow_html=True,
        )

        st.markdown("### 🌲 Random Forest")

        st.write(
            f"Accuracy: **{OVERALL_RESULTS['Random Forest']['Accuracy']:.4f}%**"
        )

        st.write(
            f"F1-score: **{OVERALL_RESULTS['Random Forest']['F1-Score']:.4f}%**"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            '<div class="model-box">',
            unsafe_allow_html=True,
        )

        st.markdown("### ⚡ XGBoost")

        st.write(
            f"Accuracy: **{OVERALL_RESULTS['XGBoost']['Accuracy']:.4f}%**"
        )

        st.write(
            f"F1-score: **{OVERALL_RESULTS['XGBoost']['F1-Score']:.4f}%**"
        )

        st.success(
            "Selected as primary model"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

def render_model_performance():

    st.header(
        "📊 Model Performance"
    )

    st.info(
        "Results shown below are the recorded evaluation results "
        "from the current Random Forest and XGBoost training run."
    )

    df = (
        pd.DataFrame(
            OVERALL_RESULTS
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Model"
            }
        )
    )

    display_df = df.copy()

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]

    for column in numeric_columns:
        display_df[column] = (
            display_df[column]
            .map(lambda x: f"{x:.4f}%")
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Model Comparison"
    )

    fig = go.Figure()

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]:

        fig.add_trace(
            go.Bar(
                name=metric,
                x=list(
                    OVERALL_RESULTS.keys()
                ),
                y=[
                    OVERALL_RESULTS[m][metric]
                    for m in OVERALL_RESULTS
                ],
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis_title="Score (%)",
        xaxis_title="Model",
        yaxis=dict(
            range=[
                95,
                100,
            ]
        ),
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Primary Model"
    )

    st.success(
        "XGBoost is selected as the primary deployment model "
        "because it achieved the highest overall accuracy, "
        "precision, F1-score and ROC-AUC."
    )


# =============================================================================
# LIVE DETECTION
# =============================================================================

def render_live_detection():

    st.header(
        "🔴 Live Detection"
    )

    st.caption(
        "This page reads detections produced by the realtime detector. "
        "It does not generate fake traffic statistics."
    )

    detection_df = read_detection_log()

    if detection_df.empty:

        st.warning(
            "No realtime detection log was found yet. "
            "Start realtime_detection.py and refresh this page."
        )

        st.code(
            "python realtime_detection.py",
            language="powershell",
        )

        return

    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    total = len(detection_df)

    attacks = int(
        (
            detection_df["Prediction"]
            == "ATTACK"
        ).sum()
    )

    benign = int(
        (
            detection_df["Prediction"]
            == "BENIGN"
        ).sum()
    )

    attack_rate = (
        attacks / total * 100
        if total
        else 0
    )

    avg_confidence = (
        detection_df["Confidence"]
        .dropna()
        .mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Flows Detected",
        f"{total:,}",
    )

    c2.metric(
        "Benign",
        f"{benign:,}",
    )

    c3.metric(
        "Attacks",
        f"{attacks:,}",
    )

    c4.metric(
        "Attack Rate",
        f"{attack_rate:.2f}%",
    )

    if not np.isnan(avg_confidence):

        st.metric(
            "Average Confidence",
            f"{avg_confidence:.4f}",
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Prediction distribution
    # -------------------------------------------------------------------------

    st.subheader(
        "Detection Distribution"
    )

    distribution = pd.DataFrame(
        {
            "Prediction": [
                "BENIGN",
                "ATTACK",
            ],
            "Count": [
                benign,
                attacks,
            ],
        }
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=distribution[
                    "Prediction"
                ],
                values=distribution[
                    "Count"
                ],
                hole=0.45,
            )
        ]
    )

    fig.update_layout(
        height=400,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # -------------------------------------------------------------------------
    # Recent detections
    # -------------------------------------------------------------------------

    st.subheader(
        "Recent Detections"
    )

    recent = (
        detection_df
        .sort_values(
            "Timestamp",
            ascending=False,
        )
        .head(50)
    )

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # Attack alert
    # -------------------------------------------------------------------------

    if attacks > 0:

        st.error(
            f"⚠️ {attacks:,} attack flow(s) "
            "have been detected."
        )

    else:

        st.success(
            "No attack flows have been recorded "
            "in the current detection log."
        )


# =============================================================================
# ATTACK EVALUATION
# =============================================================================

def render_attack_evaluation():

    st.header(
        "📈 CICIDS2017 Attack Evaluation"
    )

    st.caption(
        "Attack-specific evaluation results obtained by testing "
        "the trained XGBoost model on CICIDS2017 DDoS and PortScan CSV files."
    )

    rows = []

    for attack, metrics in ATTACK_RESULTS.items():

        row = {
            "Attack": attack
        }

        row.update(metrics)

        rows.append(row)

    df = pd.DataFrame(rows)

    formatted = df.copy()

    for column in formatted.columns[1:]:

        formatted[column] = formatted[column].map(
            lambda x: f"{x:.4f}%"
        )

    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Attack-Specific Performance"
    )

    fig = go.Figure()

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]:

        fig.add_trace(
            go.Bar(
                name=metric,
                x=list(
                    ATTACK_RESULTS.keys()
                ),
                y=[
                    ATTACK_RESULTS[a][metric]
                    for a in ATTACK_RESULTS
                ],
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis_title="Score (%)",
        yaxis=dict(
            range=[
                98,
                100,
            ]
        ),
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.warning(
        "These are CICIDS2017 dataset evaluations, not independent "
        "real-world validation results."
    )


# =============================================================================
# FEATURES
# =============================================================================

def render_features():

    st.header(
        "🧩 Model Features"
    )

    features = load_feature_metadata()

    st.write(
        f"The current model uses **{len(features)} flow-level features**."
    )

    feature_df = pd.DataFrame(
        {
            "Index": range(
                1,
                len(features) + 1,
            ),
            "Feature": features,
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "These are flow-level CICIDS2017-style features. "
        "They are different from the basic packet metadata collected "
        "directly from a raw packet."
    )


# =============================================================================
# SYSTEM
# =============================================================================

def render_system():

    st.header(
        "⚙️ System Status"
    )

    st.subheader(
        "Model Files"
    )

    files = [
        "xgboost.pkl",
        "random_forest.pkl",
        "best_model.pkl",
        "feature_names.pkl",
        "feature_names.txt",
        "xgboost_shap_explainer.pkl",
    ]

    rows = []

    for name in files:

        path = MODELS_DIR / name

        rows.append(
            {
                "File": name,
                "Status": (
                    "Available"
                    if path.exists()
                    else "Missing"
                ),
                "Size (MB)": (
                    round(
                        path.stat().st_size
                        / 1024
                        / 1024,
                        2,
                    )
                    if path.exists()
                    else 0
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Feature Metadata"
    )

    features = load_feature_metadata()

    c1, c2 = st.columns(2)

    c1.metric(
        "Model Features",
        len(features),
    )

    c2.metric(
        "Primary Model",
        "XGBoost",
    )

    st.subheader(
        "Project Status"
    )

    st.success(
        "Training and CICIDS2017 CSV evaluation completed."
    )

    st.warning(
        "Realtime PCAP/live-traffic detection is operational, "
        "but its real-world attack-detection accuracy has not yet "
        "been independently validated."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    page = render_sidebar()

    if page == "🏠 Overview":
        render_overview()

    elif page == "📊 Model Performance":
        render_model_performance()

    elif page == "🔴 Live Detection":
        render_live_detection()

    elif page == "📈 Attack Evaluation":
        render_attack_evaluation()

    elif page == "🧠 Features":
        render_features()

    elif page == "⚙️ System":
        render_system()


if __name__ == "__main__":
    main()