"""
Real-Time / PCAP Detection Module for NIDS-ML
=============================================

This module connects the trained Random Forest / XGBoost models to
live Scapy traffic or a PCAP file.

IMPORTANT:
    The models were trained on CICIDS2017 flow-level features, not on
    individual packet fields. Therefore this module first groups packets
    into bidirectional flows and calculates the same 30 SELECTED feature
    names used by the current training pipeline.

The feature calculations below are CICIDS-style approximations intended
for live/PCAP integration. For a research-paper claim of exact
CICIDS2017 feature equivalence, validate these calculations against
CICFlowMeter on the same PCAP.

Current model output:
    0 -> BENIGN
    1 -> ATTACK

This binary model does NOT identify the exact attack family (DDoS,
PortScan, Brute Force, etc.). Exact attack-family labels require a
multiclass model or a separate classifier/rule layer.

Usage:

    Live capture:
        python realtime_detection.py --model xgboost --interface "Wi-Fi"

    Random Forest:
        python realtime_detection.py --model random_forest --interface "Wi-Fi"

    PCAP:
        python realtime_detection.py --model xgboost --pcap data/test.pcap

    Optional BPF filter:
        python realtime_detection.py --model xgboost --interface "Wi-Fi" --filter "tcp"

Requirements:
    pip install scapy pandas numpy joblib
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scapy.all import ICMP, IP, TCP, UDP, PcapReader, sniff


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / "realtime_detection.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("nids.realtime")


# =============================================================================
# EXACT FEATURE ORDER USED BY CURRENT MODEL
# =============================================================================

MODEL_FEATURES = [
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
# Helpers
# =============================================================================

def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(np.std(values, ddof=0)) if values else 0.0


def safe_max(values: List[float]) -> float:
    return float(max(values)) if values else 0.0


def safe_min(values: List[float]) -> float:
    return float(min(values)) if values else 0.0


def inter_arrival_times(times: List[float]) -> List[float]:
    if len(times) < 2:
        return []
    return [
        max(0.0, times[i] - times[i - 1])
        for i in range(1, len(times))
    ]


def tcp_flag_value(packet) -> int:
    if TCP not in packet:
        return 0

    try:
        return int(packet[TCP].flags)
    except Exception:
        return 0


def is_forward_packet(packet, flow) -> bool:
    """
    Direction is determined by the first packet that created the flow.
    """
    if IP not in packet:
        return False

    return (
        packet[IP].src == flow.src_ip
        and packet[IP].dst == flow.dst_ip
        and get_src_port(packet) == flow.src_port
        and get_dst_port(packet) == flow.dst_port
    )


def get_src_port(packet) -> Optional[int]:
    if TCP in packet:
        return int(packet[TCP].sport)
    if UDP in packet:
        return int(packet[UDP].sport)
    return None


def get_dst_port(packet) -> Optional[int]:
    if TCP in packet:
        return int(packet[TCP].dport)
    if UDP in packet:
        return int(packet[UDP].dport)
    return None


def get_protocol(packet) -> int:
    if IP not in packet:
        return 0
    return int(packet[IP].proto)


def packet_timestamp(packet) -> float:
    try:
        return float(packet.time)
    except Exception:
        return time.time()


# =============================================================================
# Flow State
# =============================================================================

@dataclass
class FlowState:
    """
    Stores packets belonging to one bidirectional network flow.
    """

    key: Tuple
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: int

    first_seen: float
    last_seen: float

    forward_sizes: List[int] = field(default_factory=list)
    backward_sizes: List[int] = field(default_factory=list)

    forward_times: List[float] = field(default_factory=list)
    backward_times: List[float] = field(default_factory=list)

    all_times: List[float] = field(default_factory=list)
    all_sizes: List[int] = field(default_factory=list)

    psh_count: int = 0

    init_win_backward: int = 0
    backward_window_seen: bool = False

    active_periods: List[float] = field(default_factory=list)
    idle_periods: List[float] = field(default_factory=list)

    last_packet_time: Optional[float] = None

    def add_packet(self, packet) -> None:
        ts = packet_timestamp(packet)
        size = len(packet)

        self.last_seen = ts
        self.last_packet_time = ts

        self.all_times.append(ts)
        self.all_sizes.append(size)

        if is_forward_packet(packet, self):
            self.forward_sizes.append(size)
            self.forward_times.append(ts)

            if TCP in packet and tcp_flag_value(packet) & 0x08:
                self.psh_count += 1

        else:
            self.backward_sizes.append(size)
            self.backward_times.append(ts)

            if TCP in packet:
                if not self.backward_window_seen:
                    try:
                        self.init_win_backward = int(packet[TCP].window)
                    except Exception:
                        self.init_win_backward = 0
                    self.backward_window_seen = True

        # Track periods of activity and inactivity.
        if len(self.all_times) >= 2:
            gap = ts - self.all_times[-2]

            # CICIDS-style activity threshold approximation.
            if gap <= 1.0:
                self.active_periods.append(gap)
            else:
                self.idle_periods.append(gap)


# =============================================================================
# Flow Manager
# =============================================================================

class FlowManager:
    """
    Groups packets into bidirectional 5-tuples and converts completed flows
    into the 30 model features.
    """

    def __init__(self, flow_timeout: float = 5.0):
        self.flow_timeout = flow_timeout
        self.flows: Dict[Tuple, FlowState] = {}

    @staticmethod
    def make_key(packet) -> Optional[Tuple]:
        if IP not in packet:
            return None

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = get_protocol(packet)

        src_port = get_src_port(packet)
        dst_port = get_dst_port(packet)

        endpoint_a = (src_ip, src_port)
        endpoint_b = (dst_ip, dst_port)

        if endpoint_a <= endpoint_b:
            return (
                endpoint_a,
                endpoint_b,
                protocol,
            )

        return (
            endpoint_b,
            endpoint_a,
            protocol,
        )

    def add_packet(self, packet) -> List[FlowState]:
        """
        Add packet and return flows that have timed out.
        """

        if IP not in packet:
            return []

        key = self.make_key(packet)

        if key is None:
            return []

        ts = packet_timestamp(packet)

        if key not in self.flows:
            flow = FlowState(
                key=key,
                src_ip=packet[IP].src,
                dst_ip=packet[IP].dst,
                src_port=get_src_port(packet),
                dst_port=get_dst_port(packet),
                protocol=get_protocol(packet),
                first_seen=ts,
                last_seen=ts,
            )
            self.flows[key] = flow

        flow = self.flows[key]
        flow.add_packet(packet)

        return self.expire(ts)

    def expire(self, current_time: float) -> List[FlowState]:
        completed = []

        expired_keys = [
            key
            for key, flow in self.flows.items()
            if current_time - flow.last_seen >= self.flow_timeout
        ]

        for key in expired_keys:
            completed.append(self.flows.pop(key))

        return completed

    def flush_all(self) -> List[FlowState]:
        completed = list(self.flows.values())
        self.flows.clear()
        return completed


# =============================================================================
# Feature Builder
# =============================================================================

class CICIDSFeatureBuilder:
    """
    Converts a FlowState into the 30 features expected by the trained model.

    NOTE:
        These calculations are intended to reproduce the meaning of the
        selected CICIDS/CICFlowMeter features for live traffic. They are
        not a byte-for-byte reimplementation of CICFlowMeter.
    """

    @staticmethod
    def build(flow: FlowState) -> Dict[str, float]:

        fwd_sizes = flow.forward_sizes
        bwd_sizes = flow.backward_sizes

        fwd_times = flow.forward_times
        bwd_times = flow.backward_times
        all_times = flow.all_times
        all_sizes = flow.all_sizes

        flow_iat = inter_arrival_times(all_times)
        fwd_iat = inter_arrival_times(fwd_times)
        bwd_iat = inter_arrival_times(bwd_times)

        # ---------------------------------------------------------------
        # Forward packet statistics
        # ---------------------------------------------------------------

        fwd_max = safe_max(fwd_sizes)
        fwd_std = safe_std(fwd_sizes)

        # ---------------------------------------------------------------
        # Backward packet statistics
        # ---------------------------------------------------------------

        bwd_max = safe_max(bwd_sizes)
        bwd_min = safe_min(bwd_sizes)
        bwd_mean = safe_mean(bwd_sizes)
        bwd_std = safe_std(bwd_sizes)

        # ---------------------------------------------------------------
        # Active / idle approximation
        # ---------------------------------------------------------------

        active = flow.active_periods
        idle = flow.idle_periods

        active_mean = safe_mean(active)
        active_max = safe_max(active)
        active_min = safe_min(active)
        idle_max = safe_max(idle)

        # ---------------------------------------------------------------
        # Construct exact model feature dictionary
        # ---------------------------------------------------------------

        features = {
            "Total_Backward_Packets": len(bwd_sizes),

            "Total_Length_of_Bwd_Packets": sum(bwd_sizes),

            "Fwd_Packet_Length_Max": fwd_max,

            "Fwd_Packet_Length_Std": fwd_std,

            "Bwd_Packet_Length_Max": bwd_max,

            "Bwd_Packet_Length_Min": bwd_min,

            "Bwd_Packet_Length_Mean": bwd_mean,

            "Bwd_Packet_Length_Std": bwd_std,

            "Flow_IAT_Std": safe_std(flow_iat),

            "Flow_IAT_Max": safe_max(flow_iat),

            "Flow_IAT_Min": safe_min(flow_iat),

            "Fwd_IAT_Mean": safe_mean(fwd_iat),

            "Fwd_IAT_Std": safe_std(fwd_iat),

            "Fwd_IAT_Max": safe_max(fwd_iat),

            "Fwd_IAT_Min": safe_min(fwd_iat),

            "Bwd_IAT_Total": sum(bwd_iat),

            "Max_Packet_Length": safe_max(all_sizes),

            "Packet_Length_Std": safe_std(all_sizes),

            "Packet_Length_Variance": (
                float(np.var(all_sizes))
                if all_sizes
                else 0.0
            ),

            "PSH_Flag_Count": flow.psh_count,

            "Average_Packet_Size": safe_mean(all_sizes),

            "Avg_Bwd_Segment_Size": bwd_mean,

            "Subflow_Fwd_Bytes": sum(fwd_sizes),

            "Subflow_Bwd_Packets": len(bwd_sizes),

            "Subflow_Bwd_Bytes": sum(bwd_sizes),

            "Init_Win_bytes_backward": flow.init_win_backward,

            "Active_Mean": active_mean,

            "Active_Max": active_max,

            "Active_Min": active_min,

            "Idle_Max": idle_max,
        }

        return features

    @staticmethod
    def vectorize(features: Dict[str, float]) -> pd.DataFrame:
        """
        Create a one-row DataFrame in EXACTLY the training feature order.
        """

        missing = [
            feature
            for feature in MODEL_FEATURES
            if feature not in features
        ]

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        values = [
            features[name]
            for name in MODEL_FEATURES
        ]

        return pd.DataFrame(
            [values],
            columns=MODEL_FEATURES,
        )


# =============================================================================
# Realtime Detector
# =============================================================================

class RealtimeDetector:
    """
    Runs RF/XGBoost inference on completed network flows.
    """

    def __init__(
        self,
        model_path: str,
        flow_timeout: float = 5.0,
    ):
        self.model_path = Path(model_path)

        self.model = None

        self.flow_manager = FlowManager(
            flow_timeout=flow_timeout
        )

        self.feature_builder = CICIDSFeatureBuilder()

        self.is_running = False

        self.packet_count = 0
        self.ip_packet_count = 0
        self.flow_count = 0
        self.attack_count = 0
        self.benign_count = 0

        self.detection_queue = deque(
            maxlen=100
        )

        self.load_model()

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    def load_model(self):
        logger.info(
            "Loading model: %s",
            self.model_path,
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        logger.info(
            "Model loaded successfully: %s",
            type(self.model).__name__,
        )

        # Check the expected feature count.
        if hasattr(self.model, "n_features_in_"):
            expected = int(
                self.model.n_features_in_
            )

            if expected != len(MODEL_FEATURES):
                raise ValueError(
                    f"Model expects {expected} features, "
                    f"but realtime pipeline provides "
                    f"{len(MODEL_FEATURES)}."
                )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def classify_flow(
        self,
        flow: FlowState,
    ) -> Optional[Dict]:
        """
        Convert a completed flow to model features and classify it.
        """

        if not flow.all_sizes:
            return None

        features = self.feature_builder.build(
            flow
        )

        X = self.feature_builder.vectorize(
            features
        )

        # ---------------------------------------------------------------------
        # Predict
        # ---------------------------------------------------------------------

        prediction = self.model.predict(
            X
        )[0]

        prediction_numeric = int(
            prediction
        )

        if hasattr(
            self.model,
            "predict_proba",
        ):
            probabilities = self.model.predict_proba(
                X
            )[0]

            confidence = float(
                np.max(probabilities)
            )

            # Binary model: class 1 = ATTACK.
            attack_probability = (
                float(probabilities[1])
                if len(probabilities) > 1
                else confidence
            )

        else:
            confidence = None
            attack_probability = (
                1.0
                if prediction_numeric == 1
                else 0.0
            )

        label = (
            "ATTACK"
            if prediction_numeric == 1
            else "BENIGN"
        )

        if label == "ATTACK":
            self.attack_count += 1
        else:
            self.benign_count += 1

        self.flow_count += 1

        detection = {
            "timestamp": datetime_from_timestamp(
                flow.last_seen
            ),
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "prediction": prediction_numeric,
            "label": label,
            "confidence": confidence,
            "attack_probability": attack_probability,
            "packet_count": len(flow.all_sizes),
            "forward_packets": len(
                flow.forward_sizes
            ),
            "backward_packets": len(
                flow.backward_sizes
            ),
            "total_bytes": sum(
                flow.all_sizes
            ),
            "features": features,
        }

        self.detection_queue.append(
            detection
        )

        if label == "ATTACK":
            logger.warning(
                "ATTACK DETECTED | "
                "%s:%s -> %s:%s | "
                "confidence=%.4f | packets=%d",
                flow.src_ip,
                flow.src_port,
                flow.dst_ip,
                flow.dst_port,
                confidence if confidence is not None else 0.0,
                len(flow.all_sizes),
            )

            self.log_intrusion(
                detection
            )

        else:
            logger.info(
                "BENIGN | "
                "%s:%s -> %s:%s | "
                "confidence=%.4f | packets=%d",
                flow.src_ip,
                flow.src_port,
                flow.dst_ip,
                flow.dst_port,
                confidence if confidence is not None else 0.0,
                len(flow.all_sizes),
            )

        return detection

    # -------------------------------------------------------------------------
    # Packet callback
    # -------------------------------------------------------------------------

    def process_packet(self, packet):
        """
        Receive one Scapy packet, update its flow and classify flows that
        have timed out.
        """

        self.packet_count += 1

        if IP not in packet:
            return

        self.ip_packet_count += 1

        try:
            completed_flows = (
                self.flow_manager.add_packet(
                    packet
                )
            )

            for flow in completed_flows:
                self.classify_flow(
                    flow
                )

        except Exception as exc:
            logger.exception(
                "Error processing packet: %s",
                exc,
            )

    # -------------------------------------------------------------------------
    # Capture
    # -------------------------------------------------------------------------

    def start_capture(
        self,
        interface: Optional[str] = None,
        filter_rule: Optional[str] = None,
    ):
        """
        Start live packet capture.

        Windows example:
            interface="Wi-Fi"

        Linux example:
            interface="eth0"
        """

        logger.info("=" * 70)
        logger.info(
            "NIDS REAL-TIME DETECTION STARTING"
        )
        logger.info("=" * 70)

        logger.info(
            "Model: %s",
            self.model_path.name,
        )

        logger.info(
            "Interface: %s",
            interface or "all",
        )

        logger.info(
            "Flow timeout: %.1f seconds",
            self.flow_manager.flow_timeout,
        )

        logger.info(
            "Press Ctrl+C to stop."
        )

        self.is_running = True

        try:
            sniff(
                iface=interface,
                filter=filter_rule,
                prn=self.process_packet,
                store=False,
            )

        except KeyboardInterrupt:
            logger.info(
                "Capture stopped by user."
            )

        except Exception as exc:
            logger.exception(
                "Capture error: %s",
                exc,
            )

        finally:
            self.stop_capture()

    # -------------------------------------------------------------------------
    # PCAP
    # -------------------------------------------------------------------------

    def process_pcap(
        self,
        pcap_path: str,
    ):
        """
        Process a PCAP file through the same flow-building and model pipeline.
        """

        pcap_path = Path(pcap_path)

        if not pcap_path.exists():
            raise FileNotFoundError(
                f"PCAP not found: {pcap_path}"
            )

        logger.info("=" * 70)
        logger.info(
            "NIDS PCAP DETECTION"
        )
        logger.info("=" * 70)

        logger.info(
            "PCAP: %s",
            pcap_path,
        )

        try:
            with PcapReader(
                str(pcap_path)
            ) as reader:

                for packet in reader:

                    self.process_packet(
                        packet
                    )

        except KeyboardInterrupt:
            logger.info(
                "PCAP processing interrupted."
            )

        finally:
            # -----------------------------------------------------------------
            # PCAP ends, so classify all remaining flows.
            # -----------------------------------------------------------------

            remaining = (
                self.flow_manager.flush_all()
            )

            for flow in remaining:
                self.classify_flow(
                    flow
                )

            self.print_statistics()

    # -------------------------------------------------------------------------
    # Stop
    # -------------------------------------------------------------------------

    def stop_capture(self):
        """
        Stop live capture and classify any remaining flows.
        """

        self.is_running = False

        remaining = (
            self.flow_manager.flush_all()
        )

        for flow in remaining:
            try:
                self.classify_flow(
                    flow
                )
            except Exception as exc:
                logger.exception(
                    "Error classifying final flow: %s",
                    exc,
                )

        self.print_statistics()

    # -------------------------------------------------------------------------
    # Intrusion logging
    # -------------------------------------------------------------------------

    def log_intrusion(
        self,
        detection: Dict,
    ):
        """
        Append attack detections to CSV.
        """

        output_path = (
            RESULTS_DIR /
            "realtime_detections.csv"
        )

        row = {
            "timestamp": detection[
                "timestamp"
            ],
            "src_ip": detection[
                "src_ip"
            ],
            "dst_ip": detection[
                "dst_ip"
            ],
            "src_port": detection[
                "src_port"
            ],
            "dst_port": detection[
                "dst_port"
            ],
            "protocol": detection[
                "protocol"
            ],
            "label": detection[
                "label"
            ],
            "confidence": detection[
                "confidence"
            ],
            "attack_probability": detection[
                "attack_probability"
            ],
            "packet_count": detection[
                "packet_count"
            ],
            "forward_packets": detection[
                "forward_packets"
            ],
            "backward_packets": detection[
                "backward_packets"
            ],
            "total_bytes": detection[
                "total_bytes"
            ],
        }

        dataframe = pd.DataFrame(
            [row]
        )

        dataframe.to_csv(
            output_path,
            mode="a",
            header=not output_path.exists(),
            index=False,
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """
        Return current detector statistics.
        """

        total_flows = (
            self.benign_count
            + self.attack_count
        )

        attack_rate = (
            self.attack_count / total_flows
            if total_flows
            else 0.0
        )

        return {
            "total_packets": self.packet_count,
            "ip_packets": self.ip_packet_count,
            "completed_flows": self.flow_count,
            "benign_flows": self.benign_count,
            "attack_flows": self.attack_count,
            "attack_rate": attack_rate,
            "recent_detections": list(
                self.detection_queue
            )[-10:],
        }

    def get_recent_detections(
        self,
        n: int = 10,
    ) -> List[Dict]:
        return list(
            self.detection_queue
        )[-n:]

    def print_statistics(self):
        stats = self.get_statistics()

        logger.info("=" * 70)
        logger.info(
            "FINAL DETECTION STATISTICS"
        )
        logger.info("=" * 70)

        logger.info(
            "Packets processed : %s",
            f"{stats['total_packets']:,}",
        )

        logger.info(
            "IP packets        : %s",
            f"{stats['ip_packets']:,}",
        )

        logger.info(
            "Flows classified  : %s",
            f"{stats['completed_flows']:,}",
        )

        logger.info(
            "BENIGN flows      : %s",
            f"{stats['benign_flows']:,}",
        )

        logger.info(
            "ATTACK flows      : %s",
            f"{stats['attack_flows']:,}",
        )

        logger.info(
            "Attack rate       : %.2f%%",
            stats["attack_rate"] * 100,
        )

        logger.info("=" * 70)


# =============================================================================
# Timestamp helper
# =============================================================================

def datetime_from_timestamp(
    timestamp: float,
) -> str:
    """
    Convert packet epoch timestamp into a readable local timestamp.
    """

    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(timestamp),
    )


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "NIDS-ML real-time/PCAP detection "
            "using Random Forest or XGBoost."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "random_forest",
            "xgboost",
        ],
        default="xgboost",
        help="Model to use.",
    )

    parser.add_argument(
        "--interface",
        default=None,
        help='Live interface, e.g. "Wi-Fi".',
    )

    parser.add_argument(
        "--pcap",
        default=None,
        help="PCAP file to process.",
    )

    parser.add_argument(
        "--filter",
        dest="filter_rule",
        default=None,
        help='BPF filter, e.g. "tcp".',
    )

    parser.add_argument(
        "--flow-timeout",
        type=float,
        default=5.0,
        help="Seconds of inactivity before a flow is classified.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Model path
    # -------------------------------------------------------------------------

    model_path = (
        MODELS_DIR /
        f"{args.model}.pkl"
    )

    detector = RealtimeDetector(
        model_path=str(model_path),
        flow_timeout=args.flow_timeout,
    )

    # -------------------------------------------------------------------------
    # PCAP mode
    # -------------------------------------------------------------------------

    if args.pcap:

        detector.process_pcap(
            args.pcap
        )

        return

    # -------------------------------------------------------------------------
    # Live mode
    # -------------------------------------------------------------------------

    detector.start_capture(
        interface=args.interface,
        filter_rule=args.filter_rule,
    )


if __name__ == "__main__":
    main()