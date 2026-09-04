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


def _parse_timestamp(line: str):
    try:
        timestamp_text = line[:23]
        return datetime.strptime(
            timestamp_text,
            "%Y-%m-%d %H:%M:%S,%f",
        )
    except Exception:
        return None


def _extract_kv(segment: str, key: str):
    """
    Extract a `key=value` token from a ' | '-delimited log segment,
    e.g. segment='confidence=0.9999' or 'ML=BENIGN'.
    """

    if key not in segment:
        return None

    try:
        return segment.split(f"{key}=", 1)[1].strip()
    except Exception:
        return None


def _parse_flow(flow: str):
    src_ip = src_port = dst_ip = dst_port = None

    if " -> " in flow:
        try:
            src, dst = flow.split(" -> ")
            src_ip, src_port = src.rsplit(":", 1)
            dst_ip, dst_port = dst.rsplit(":", 1)
        except Exception:
            pass

    return src_ip, src_port, dst_ip, dst_port


def read_detection_log():
    """
    Parse lines produced by realtime_detection.py.

    Two line shapes are currently emitted:

    BENIGN (informational, ML-only path):
        <asctime> | INFO     | BENIGN | src_ip:src_port -> dst_ip:dst_port | confidence=... | packets=...

    FINAL ATTACK (either the ML model flagged the flow, the behavior
    / rule-based layer flagged it, or both):
        <asctime> | WARNING  | FINAL ATTACK | <attack_family> | src_ip:src_port -> dst_ip:dst_port | ML=<BENIGN|ATTACK> | confidence=... | packets=...

    attack_family is "UNKNOWN" when the ML model alone flagged the
    flow, and a specific family (e.g. "PORT_SCAN") when the
    behavior/rule-based layer flagged it. A flow can therefore be
    "Detected By" ML, Behavior, or both — this is derived below from
    ML= and attack_family rather than assumed.

    Older/alternate log shapes (a bare "| ATTACK |" label with no
    "FINAL" prefix and no attack_family) are also handled for
    backward compatibility with logs generated before the rule-based
    layer was added.
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

        parts = line.split(" | ")

        if len(parts) < 4:
            continue

        category = parts[2].strip()
        timestamp = _parse_timestamp(line)

        if category == "BENIGN":
            flow = parts[3].strip() if len(parts) > 3 else ""
            confidence = np.nan
            packets = np.nan

            for seg in parts[4:]:
                if confidence is np.nan or np.isnan(confidence):
                    val = _extract_kv(seg, "confidence")
                    if val is not None:
                        try:
                            confidence = float(val)
                        except Exception:
                            pass
                val = _extract_kv(seg, "packets")
                if val is not None:
                    try:
                        packets = int(val)
                    except Exception:
                        pass

            src_ip, src_port, dst_ip, dst_port = _parse_flow(flow)

            rows.append(
                {
                    "Timestamp": timestamp,
                    "Prediction": "BENIGN",
                    "Attack Type": "-",
                    "Detected By": "-",
                    "ML Label": "BENIGN",
                    "Flow": flow,
                    "Source IP": src_ip,
                    "Source Port": src_port,
                    "Dest IP": dst_ip,
                    "Dest Port": dst_port,
                    "Confidence": confidence,
                    "Packets": packets,
                }
            )

        elif category in ("FINAL ATTACK", "ATTACK"):
            # "FINAL ATTACK" (current format) carries an attack_family
            # segment before the flow; the legacy "ATTACK" format does not.
            if category == "FINAL ATTACK" and len(parts) >= 5:
                attack_family = parts[3].strip()
                flow = parts[4].strip()
                rest = parts[5:]
            else:
                attack_family = "UNKNOWN"
                flow = parts[3].strip() if len(parts) > 3 else ""
                rest = parts[4:]

            ml_label = None
            confidence = np.nan
            packets = np.nan

            for seg in rest:
                if ml_label is None:
                    val = _extract_kv(seg, "ML")
                    if val is not None:
                        ml_label = val
                val = _extract_kv(seg, "confidence")
                if val is not None:
                    try:
                        confidence = float(val)
                    except Exception:
                        pass
                val = _extract_kv(seg, "packets")
                if val is not None:
                    try:
                        packets = int(val)
                    except Exception:
                        pass

            if ml_label is None:
                # Legacy format: the whole flow was labelled ATTACK
                # directly by the model, no ML= breakdown available.
                ml_label = "ATTACK"

            detected_by = []
            if ml_label == "ATTACK":
                detected_by.append("ML")
            if attack_family and attack_family != "UNKNOWN":
                detected_by.append("Behavior")
            if not detected_by:
                detected_by.append("ML")

            src_ip, src_port, dst_ip, dst_port = _parse_flow(flow)

            rows.append(
                {
                    "Timestamp": timestamp,
                    "Prediction": "ATTACK",
                    "Attack Type": attack_family if attack_family else "UNKNOWN",
                    "Detected By": " + ".join(detected_by),
                    "ML Label": ml_label,
                    "Flow": flow,
                    "Source IP": src_ip,
                    "Source Port": src_port,
                    "Dest IP": dst_ip,
                    "Dest Port": dst_port,
                    "Confidence": confidence,
                    "Packets": packets,
                }
            )

        else:
            continue

    return pd.DataFrame(rows)


def read_behavior_incidents():
    """
    Parse "BEHAVIOR ATTACK" incident lines produced by the
    BehaviorAnalyzer in realtime_detection.py:

        <asctime> | WARNING  | BEHAVIOR ATTACK | PORT_SCAN | src -> dst | connections=... | unique_ports=... | <reason>

    These are logged once per NEW incident (after the incident
    deduplication fix), so this table is a distinct, higher-level
    view from the per-flow "FINAL ATTACK" rows above.
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

        if " | BEHAVIOR ATTACK | " not in line:
            continue

        parts = line.split(" | ")

        if len(parts) < 5:
            continue

        timestamp = _parse_timestamp(line)
        attack_family = parts[3].strip()
        endpoints = parts[4].strip()

        src_ip = dst_ip = None
        if " -> " in endpoints:
            try:
                src_ip, dst_ip = endpoints.split(" -> ")
                src_ip, dst_ip = src_ip.strip(), dst_ip.strip()
            except Exception:
                pass

        connections = np.nan
        unique_ports = np.nan

        for seg in parts[5:]:
            val = _extract_kv(seg, "connections")
            if val is not None:
                try:
                    connections = int(val)
                except Exception:
                    pass
            val = _extract_kv(seg, "unique_ports")
            if val is not None:
                try:
                    unique_ports = int(val)
                except Exception:
                    pass

        rows.append(
            {
                "Timestamp": timestamp,
                "Attack Type": attack_family,
                "Source IP": src_ip,
                "Dest IP": dst_ip,
                "Connections": connections,
                "Unique Ports": unique_ports,
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
       BENIGN         ATTACK (ML)
                          │
                          ▼
                 Behavior / Rule Layer
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
            PORT_SCAN          UNKNOWN
                 │                 │
                 └────────┬────────┘
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
        "This page reads detections produced by the realtime detector, "
        "including both ML-flagged and behavior/rule-flagged attacks. "
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

    attacks_mask = detection_df["Prediction"] == "ATTACK"
    attacks = int(attacks_mask.sum())
    benign = int((detection_df["Prediction"] == "BENIGN").sum())

    attack_rows = detection_df[attacks_mask]

    ml_only = int(
        (attack_rows["Detected By"] == "ML").sum()
    )
    behavior_only = int(
        (attack_rows["Detected By"] == "Behavior").sum()
    )
    both = int(
        (attack_rows["Detected By"] == "ML + Behavior").sum()
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
    # Detection source breakdown (ML vs Behavior/rule-based)
    # -------------------------------------------------------------------------

    st.subheader(
        "Attack Detection Source"
    )

    st.caption(
        "How each attack flow was flagged: by the ML model alone, "
        "by the behavior/rule-based layer alone (e.g. port-scan "
        "pattern with ML=BENIGN on the individual flow), or by both."
    )

    s1, s2, s3 = st.columns(3)

    s1.metric("ML-only", f"{ml_only:,}")
    s2.metric("Behavior-only (rule-based)", f"{behavior_only:,}")
    s3.metric("ML + Behavior", f"{both:,}")

    if attacks > 0:

        source_dist = pd.DataFrame(
            {
                "Source": ["ML-only", "Behavior-only", "ML + Behavior"],
                "Count": [ml_only, behavior_only, both],
            }
        )
        source_dist = source_dist[source_dist["Count"] > 0]

        if not source_dist.empty:
            fig_src = go.Figure(
                data=[
                    go.Pie(
                        labels=source_dist["Source"],
                        values=source_dist["Count"],
                        hole=0.45,
                    )
                ]
            )
            fig_src.update_layout(height=350)
            st.plotly_chart(fig_src, use_container_width=True)

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
    # Attack type breakdown
    # -------------------------------------------------------------------------

    if attacks > 0:

        st.subheader(
            "Attack Type Breakdown"
        )

        type_counts = (
            attack_rows["Attack Type"]
            .value_counts()
            .reset_index()
        )
        type_counts.columns = ["Attack Type", "Count"]

        st.dataframe(
            type_counts,
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # Recent detections
    # -------------------------------------------------------------------------

    st.subheader(
        "Recent Detections"
    )

    filter_choice = st.radio(
        "Show",
        ["All", "Attacks only", "Benign only"],
        horizontal=True,
    )

    view_df = detection_df

    if filter_choice == "Attacks only":
        view_df = detection_df[attacks_mask]
    elif filter_choice == "Benign only":
        view_df = detection_df[~attacks_mask]

    recent = (
        view_df
        .sort_values(
            "Timestamp",
            ascending=False,
        )
        .head(50)
    )

    display_cols = [
        "Timestamp",
        "Prediction",
        "Attack Type",
        "Detected By",
        "Flow",
        "Confidence",
        "Packets",
    ]

    st.dataframe(
        recent[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # Behavior / rule-based incidents (deduplicated, one row per incident)
    # -------------------------------------------------------------------------

    st.subheader(
        "Behavior Layer Incidents"
    )

    st.caption(
        "One row per newly detected port-scan incident from the "
        "behavior/rule-based layer (deduplicated across the flows "
        "that make up the same scan)."
    )

    incidents_df = read_behavior_incidents()

    if incidents_df.empty:
        st.info(
            "No behavior/rule-based incidents recorded in the current log."
        )
    else:
        st.dataframe(
            incidents_df.sort_values("Timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # Attack alert
    # -------------------------------------------------------------------------

    if attacks > 0:

        st.error(
            f"⚠️ {attacks:,} attack flow(s) have been detected "
            f"({ml_only:,} ML-only, {behavior_only:,} behavior-only, "
            f"{both:,} both)."
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