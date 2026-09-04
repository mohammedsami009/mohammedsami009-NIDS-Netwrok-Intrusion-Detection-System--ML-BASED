import argparse
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader, sniff

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
LOGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "realtime_detection.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nids.realtime")

FEATURES = [
    "Total_Backward_Packets","Total_Length_of_Bwd_Packets",
    "Fwd_Packet_Length_Max","Fwd_Packet_Length_Std",
    "Bwd_Packet_Length_Max","Bwd_Packet_Length_Min",
    "Bwd_Packet_Length_Mean","Bwd_Packet_Length_Std",
    "Flow_IAT_Std","Flow_IAT_Max","Flow_IAT_Min",
    "Fwd_IAT_Mean","Fwd_IAT_Std","Fwd_IAT_Max","Fwd_IAT_Min",
    "Bwd_IAT_Total","Max_Packet_Length","Packet_Length_Std",
    "Packet_Length_Variance","PSH_Flag_Count","Average_Packet_Size",
    "Avg_Bwd_Segment_Size","Subflow_Fwd_Bytes","Subflow_Bwd_Packets",
    "Subflow_Bwd_Bytes","Init_Win_bytes_backward","Active_Mean",
    "Active_Max","Active_Min","Idle_Max"
]

def mean(x): return float(np.mean(x)) if x else 0.0
def std(x): return float(np.std(x)) if x else 0.0
def mx(x): return float(max(x)) if x else 0.0
def mn(x): return float(min(x)) if x else 0.0
def iat(x): return [max(0.0, x[i]-x[i-1]) for i in range(1,len(x))]

def sport(p):
    if TCP in p: return int(p[TCP].sport)
    if UDP in p: return int(p[UDP].sport)
    return None

def dport(p):
    if TCP in p: return int(p[TCP].dport)
    if UDP in p: return int(p[UDP].dport)
    return None

def ts(p):
    try: return float(p.time)
    except Exception: return time.time()

@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    first: float
    last: float
    fs: list = field(default_factory=list)
    bs: list = field(default_factory=list)
    ft: list = field(default_factory=list)
    bt: list = field(default_factory=list)
    times: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    psh: int = 0
    init_bw: int = 0
    bw_seen: bool = False

    def add(self, p):
        t, size = ts(p), len(p)
        forward = (
            IP in p and p[IP].src == self.src_ip and
            p[IP].dst == self.dst_ip and sport(p) == self.src_port and
            dport(p) == self.dst_port
        )
        self.last = t
        self.times.append(t)
        self.sizes.append(size)
        if forward:
            self.fs.append(size); self.ft.append(t)
            if TCP in p and int(p[TCP].flags) & 8: self.psh += 1
        else:
            self.bs.append(size); self.bt.append(t)
            if TCP in p and not self.bw_seen:
                self.init_bw = int(p[TCP].window)
                self.bw_seen = True

class FlowManager:
    def __init__(self, timeout=5.0):
        self.timeout = timeout
        self.flows = {}

    def key(self,p):
        if IP not in p: return None
        a=(p[IP].src,sport(p)); b=(p[IP].dst,dport(p))
        return (a,b,int(p[IP].proto)) if a<=b else (b,a,int(p[IP].proto))

    def add(self,p):
        if IP not in p: return []
        k=self.key(p)
        if k is None: return []
        t=ts(p)
        if k not in self.flows:
            self.flows[k]=Flow(
                p[IP].src,p[IP].dst,sport(p),dport(p),int(p[IP].proto),t,t
            )
        self.flows[k].add(p)
        return self.expire(t)

    def expire(self,t):
        out=[]
        for k,f in list(self.flows.items()):
            if t-f.last >= self.timeout:
                out.append(self.flows.pop(k))
        return out

    def flush(self):
        out=list(self.flows.values())
        self.flows.clear()
        return out

def build_features(f):
    fi=iat(f.ft); bi=iat(f.bt); all_i=iat(f.times)
    bmean=mean(f.bs)
    return {
        "Total_Backward_Packets":len(f.bs),
        "Total_Length_of_Bwd_Packets":sum(f.bs),
        "Fwd_Packet_Length_Max":mx(f.fs),
        "Fwd_Packet_Length_Std":std(f.fs),
        "Bwd_Packet_Length_Max":mx(f.bs),
        "Bwd_Packet_Length_Min":mn(f.bs),
        "Bwd_Packet_Length_Mean":bmean,
        "Bwd_Packet_Length_Std":std(f.bs),
        "Flow_IAT_Std":std(all_i),
        "Flow_IAT_Max":mx(all_i),
        "Flow_IAT_Min":mn(all_i),
        "Fwd_IAT_Mean":mean(fi),
        "Fwd_IAT_Std":std(fi),
        "Fwd_IAT_Max":mx(fi),
        "Fwd_IAT_Min":mn(fi),
        "Bwd_IAT_Total":sum(bi),
        "Max_Packet_Length":mx(f.sizes),
        "Packet_Length_Std":std(f.sizes),
        "Packet_Length_Variance":float(np.var(f.sizes)) if f.sizes else 0.0,
        "PSH_Flag_Count":f.psh,
        "Average_Packet_Size":mean(f.sizes),
        "Avg_Bwd_Segment_Size":bmean,
        "Subflow_Fwd_Bytes":sum(f.fs),
        "Subflow_Bwd_Packets":len(f.bs),
        "Subflow_Bwd_Bytes":sum(f.bs),
        "Init_Win_bytes_backward":f.init_bw,
        "Active_Mean":mean([x for x in all_i if x<=1]),
        "Active_Max":mx([x for x in all_i if x<=1]),
        "Active_Min":mn([x for x in all_i if x<=1]),
        "Idle_Max":mx([x for x in all_i if x>1]),
    }

class BehaviorAnalyzer:
    """
    Detects port-scan-like behavior per (src_ip, dst_ip) pair using a
    sliding time window of recent connections/ports.

    Deduplication: once a (src_ip, dst_ip) pair crosses the threshold,
    it is treated as a single ongoing "incident". Every matching flow
    is still reported (behavior_attack=True so it's tagged ATTACK),
    but only the FIRST match of an incident — or a match that occurs
    after the pair has been inactive for `incident_timeout` seconds —
    is flagged as behavior_new_incident=True. Callers should only log
    / count a new BEHAVIOR ATTACK event when behavior_new_incident is True.
    """
    def __init__(self, window=10.0, min_connections=10, min_ports=10, incident_timeout=10.0):
        self.window = window
        self.min_connections = min_connections
        self.min_ports = min_ports
        self.incident_timeout = incident_timeout
        self.history = defaultdict(deque)
        self.active_incidents = {}  # (src_ip, dst_ip) -> last time the rule matched

    def analyze(self, d):
        if d["protocol"] != 6 or d["dst_port"] is None:
            return None
        key = (d["src_ip"], d["dst_ip"])
        now = time.time()
        h = self.history[key]
        h.append((now, int(d["dst_port"])))
        cutoff = now - self.window
        while h and h[0][0] < cutoff:
            h.popleft()
        con = len(h)
        ports = len({x[1] for x in h})

        if con >= self.min_connections and ports >= self.min_ports:
            last_seen = self.active_incidents.get(key)
            is_new = last_seen is None or (now - last_seen) >= self.incident_timeout
            self.active_incidents[key] = now
            return {
                "attack_family": "PORT_SCAN",
                "behavior_connections": con,
                "behavior_unique_ports": ports,
                "behavior_window_seconds": self.window,
                "behavior_reason": (
                    f"{con} TCP connections to {ports} unique "
                    f"destination ports within {self.window:.0f}s"
                ),
                "behavior_new_incident": is_new,
            }
        return None

class RealtimeDetector:
    def __init__(self,model_path,flow_timeout=5.0,scan_window=10.0,
                 scan_min_connections=10,scan_min_ports=10,scan_incident_timeout=10.0):
        self.model_path=Path(model_path)
        self.model=joblib.load(self.model_path)
        logger.info("Loading model: %s",self.model_path)
        logger.info("Model loaded successfully: %s",type(self.model).__name__)
        if hasattr(self.model,"n_features_in_") and self.model.n_features_in_ != len(FEATURES):
            raise ValueError(f"Model expects {self.model.n_features_in_} features, but realtime uses {len(FEATURES)}")
        self.fm=FlowManager(flow_timeout)
        self.behavior=BehaviorAnalyzer(scan_window,scan_min_connections,scan_min_ports,scan_incident_timeout)
        self.packet_count=self.ip_count=self.flow_count=0
        self.ml_attack_count=self.behavior_attack_count=0
        self.recent=deque(maxlen=100)

    def classify(self,f):
        if not f.sizes: return
        feat=build_features(f)
        X=pd.DataFrame([[feat[x] for x in FEATURES]],columns=FEATURES)
        pred=int(self.model.predict(X)[0])
        if hasattr(self.model,"predict_proba"):
            p=self.model.predict_proba(X)[0]
            conf=float(np.max(p)); attack_prob=float(p[1]) if len(p)>1 else conf
        else:
            conf=1.0; attack_prob=float(pred)
        ml="ATTACK" if pred else "BENIGN"
        d={
            "timestamp":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(f.last)),
            "src_ip":f.src_ip,"dst_ip":f.dst_ip,"src_port":f.src_port,"dst_port":f.dst_port,
            "protocol":f.protocol,"ml_label":ml,"label":ml,"confidence":conf,
            "attack_probability":attack_prob,"packet_count":len(f.sizes),
            "forward_packets":len(f.fs),"backward_packets":len(f.bs),"total_bytes":sum(f.sizes),
            "attack_family":"UNKNOWN","behavior_attack":False,
            "behavior_reason":"","behavior_connections":0,"behavior_unique_ports":0,
            "behavior_window_seconds":0.0
        }
        b=self.behavior.analyze(d)
        if b:
            d.update(b); d["label"]="ATTACK"; d["behavior_attack"]=True
            if b["behavior_new_incident"]:
                self.behavior_attack_count+=1
                logger.warning(
                    "BEHAVIOR ATTACK | PORT_SCAN | %s -> %s | connections=%d | unique_ports=%d | %s",
                    f.src_ip,f.dst_ip,b["behavior_connections"],b["behavior_unique_ports"],b["behavior_reason"]
                )
        if ml=="ATTACK": self.ml_attack_count+=1
        self.flow_count+=1; self.recent.append(d)
        if d["label"]=="ATTACK":
            logger.warning(
                "FINAL ATTACK | %s | %s:%s -> %s:%s | ML=%s | confidence=%.4f | packets=%d",
                d["attack_family"],f.src_ip,f.src_port,f.dst_ip,f.dst_port,ml,conf,len(f.sizes)
            )
        else:
            logger.info(
                "BENIGN | %s:%s -> %s:%s | confidence=%.4f | packets=%d",
                f.src_ip,f.src_port,f.dst_ip,f.dst_port,conf,len(f.sizes)
            )
        self.save(d)

    def process(self,p):
        self.packet_count+=1
        if IP not in p:return
        self.ip_count+=1
        try:
            for f in self.fm.add(p): self.classify(f)
        except Exception:
            logger.exception("Packet processing error")

    def save(self,d):
        out=RESULTS/"realtime_detections.csv"
        pd.DataFrame([{
            k:d[k] for k in [
                "timestamp","src_ip","dst_ip","src_port","dst_port","protocol",
                "ml_label","label","attack_family","confidence","attack_probability",
                "packet_count","forward_packets","backward_packets","total_bytes",
                "behavior_attack","behavior_reason","behavior_connections",
                "behavior_unique_ports","behavior_window_seconds"
            ]
        }]).to_csv(out,mode="a",header=not out.exists(),index=False)

    def start(self,interface=None,filter_rule=None):
        logger.info("="*70)
        logger.info("NIDS REAL-TIME DETECTION STARTING")
        logger.info("="*70)
        logger.info("Model: %s",self.model_path.name)
        logger.info("Interface: %s",interface or "all")
        logger.info("Flow timeout: %.1f seconds",self.fm.timeout)
        logger.info(
            "Port-scan rule: >= %d connections + >= %d unique ports within %.1fs "
            "(new incident after %.1fs of inactivity)",
            self.behavior.min_connections,self.behavior.min_ports,self.behavior.window,
            self.behavior.incident_timeout
        )
        logger.info("Press Ctrl+C to stop.")
        try:
            sniff(iface=interface,filter=filter_rule,prn=self.process,store=False)
        except KeyboardInterrupt:
            logger.info("Capture stopped by user.")
        except Exception:
            logger.exception("Capture error")
        finally:
            self.stop()

    def pcap(self,path):
        if not Path(path).exists(): raise FileNotFoundError(f"PCAP not found: {path}")
        with PcapReader(str(path)) as r:
            for p in r:self.process(p)
        self.stop()

    def stop(self):
        for f in self.fm.flush(): self.classify(f)
        logger.info("="*70)
        logger.info("FINAL DETECTION STATISTICS")
        logger.info("="*70)
        logger.info("Packets processed      : %d",self.packet_count)
        logger.info("IP packets             : %d",self.ip_count)
        logger.info("Flows classified       : %d",self.flow_count)
        logger.info("ML ATTACK flows        : %d",self.ml_attack_count)
        logger.info("Behavior ATTACK events : %d",self.behavior_attack_count)
        logger.info("="*70)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["random_forest","xgboost"],default="xgboost")
    p.add_argument("--interface",default=None)
    p.add_argument("--pcap",default=None)
    p.add_argument("--filter",dest="filter_rule",default=None)
    p.add_argument("--flow-timeout",type=float,default=5.0)
    p.add_argument("--scan-window",type=float,default=10.0)
    p.add_argument("--scan-min-connections",type=int,default=10)
    p.add_argument("--scan-min-ports",type=int,default=10)
    p.add_argument("--scan-incident-timeout",type=float,default=10.0)
    a=p.parse_args()
    d=RealtimeDetector(
        MODELS/f"{a.model}.pkl",a.flow_timeout,a.scan_window,
        a.scan_min_connections,a.scan_min_ports,a.scan_incident_timeout
    )
    if a.pcap:d.pcap(a.pcap)
    else:d.start(a.interface,a.filter_rule)

if __name__=="__main__":
    main()