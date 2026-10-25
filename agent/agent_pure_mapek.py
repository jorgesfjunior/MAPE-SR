from flask import Flask, request, jsonify
import time
import os
import joblib
import numpy as np
from collections import defaultdict, deque
import requests

app = Flask(__name__)

# Anchored to this script's own location, not the caller's cwd -- the old
# "./models/..." resolved differently depending on who launched the
# process (bare `python3 agent_pure_mapek.py`, the test driver's
# subprocess with cwd=agent/, or Docker's WORKDIR), which is exactly how
# a stale model sitting in agent/models/ went undetected through a full
# verification pass earlier. models/ lives one level above this script,
# at the repo root locally, or at the volume-mounted /models inside the
# container (see docker-compose.yml and agent/Dockerfile).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "..", "models")

MODEL_PATH = os.path.join(_MODELS_DIR, "rf_ddos_model.pkl")
SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(_MODELS_DIR, "threshold.pkl")
FEATURES_PATH = os.path.join(_MODELS_DIR, "feature_cols.pkl")
APP_URL = "http://app:5000/execute"

model = joblib.load(MODEL_PATH)
model.n_jobs = 1  # predict_proba aggregates trees via joblib threads when n_jobs=-1;
                  # thread-scheduling order makes float summation non-associative,
                  # causing ~1-ULP nondeterminism in predict_proba across identical
                  # inputs. Forcing single-threaded inference makes it bit-reproducible.
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# Exact column order the scaler/model were fit on (trainer_CIC2019/train_model.py
# FEATURE_COLS) -- build_features() below MUST assemble the vector in this
# order, never a hardcoded one, since it isn't guaranteed to stay in
# "Mean/Std/Max/Min" order (or any fixed order) across retrains.
feature_cols = joblib.load(FEATURES_PATH)

n_features = model.n_features_in_
assert n_features == len(feature_cols), (
    f"Model expects {n_features} features but feature_cols.pkl lists "
    f"{len(feature_cols)} column names -- model/feature_cols.pkl are out of "
    f"sync. Refusing to pad to compensate; retrain or re-save matching "
    f"artifacts."
)

COLD_START_MIN_SAMPLES = 3  # see build_features() docstring

# =========================
# KNOWLEDGE (K)
# =========================

ip_history = defaultdict(lambda: deque(maxlen=50))

knowledge_base = defaultdict(lambda: {
    "detections": 0,
    "state": "NORMAL",
    "actions": []
})

# =========================
# MÉTRICAS
# =========================

attack_start_time = {}
detection_time = {}

tp = 0
fp = 0

mttd_list = []
mttm_list = []

# =========================
# MONITOR
# =========================

def build_features(data):
    """Returns (feature_vector, cold_start) for one incoming request.

    ip_history[ip] stores (timestamp, payload_size) tuples for the trailing
    10s window (same window pattern the old req_rate used). The 8 features
    are the ones train_model.py was fit on: Fwd Packet Length Mean/Std/
    Min/Max and Fwd IAT Mean/Std/Max/Min, computed over that window.

    COLD-START: with fewer than COLD_START_MIN_SAMPLES (3) samples in the
    window, std/min/max are undefined or degenerate (e.g. std of a single
    value) and out-of-distribution for a model trained on real multi-packet
    flow statistics -- so classification is skipped entirely (returns
    cold_start=True, feature_vector=None) rather than feeding the model
    garbage. 3 is also the minimum for which both the packet-length std and
    the IAT std (computed over len(recent)-1 consecutive gaps) are defined
    under ddof=1 (needs >=2 samples each).
    """
    ip = data["ip"]
    now = data["timestamp"]
    size = data["payload_size"]

    history = ip_history[ip]
    history.append((now, size))

    window = 10
    recent = [(t, s) for t, s in history if now - t < window]

    if len(recent) < COLD_START_MIN_SAMPLES:
        return None, True

    sizes = [s for _, s in recent]

    # UNITS: our timestamps are time.time()-based (seconds), but the
    # CICDDoS2019 training data's Fwd IAT columns are in microseconds --
    # verified empirically against this repo's own training parquet:
    # Flow Packets/s == total_pkts / (Flow Duration / 1e6) matches the
    # dataset's actual Flow Packets/s column exactly across sampled rows
    # (Flow Duration and Fwd IAT share the same CICFlowMeter time unit).
    # Multiply by 1e6 so the live feature lands on the same scale the
    # scaler/model were fit on -- without this, IAT features would be
    # off by ~6 orders of magnitude from the training distribution.
    gaps = [(recent[i][0] - recent[i - 1][0]) * 1_000_000 for i in range(1, len(recent))]

    # ddof=1 (Bessel-corrected "sample" std). NOTE: unlike the microsecond
    # conversion above, this is NOT independently verified in this repo --
    # it's based on recalled/general knowledge that CICFlowMeter uses
    # Apache Commons Math's SummaryStatistics.getStandardDeviation() (n-1
    # denominator), reinforced by an unverified web-search summary, not by
    # inspecting CICFlowMeter source or raw packets directly (neither is
    # available here). An empirical ddof=0-vs-1 sanity check showed only a
    # ~1.7-1.8% difference at n~20-30 samples -- too small for a
    # "does it fall in the real column's range" check to discriminate
    # between conventions, since that range spans several orders of
    # magnitude. Treat this as a best-effort assumption, not a confirmed
    # fact, until it can be checked against CICFlowMeter source directly.
    values = {
        "Fwd Packet Length Min":  float(min(sizes)),
        "Fwd Packet Length Mean": float(np.mean(sizes)),
        "Fwd Packet Length Max":  float(max(sizes)),
        "Fwd Packet Length Std":  float(np.std(sizes, ddof=1)),
        "Fwd IAT Mean": float(np.mean(gaps)),
        "Fwd IAT Std":  float(np.std(gaps, ddof=1)),
        "Fwd IAT Max":  float(max(gaps)),
        "Fwd IAT Min":  float(min(gaps)),
    }

    features = np.array([[values[c] for c in feature_cols]])
    return features, False

# =========================
# ANALYZE
# =========================

def analyzer(x):
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    label = "DDoS" if prob >= threshold else "Benign"
    return label, prob

# =========================
# PLAN
# =========================

def planner(ip, label):
    kb = knowledge_base[ip]

    if label == "DDoS":
        kb["detections"] += 1

        if kb["detections"] >= 5:
            action = "BLOCK"
            kb["state"] = "BLOCKED"
        elif kb["detections"] >= 2:
            action = "RATE_LIMIT"
            kb["state"] = "SUSPECT"
        else:
            action = "MONITOR"
            kb["state"] = "SUSPECT"
    else:
        kb["detections"] = max(0, kb["detections"] - 1)

        if kb["detections"] == 0:
            action = "ALLOW"
            kb["state"] = "NORMAL"
        else:
            action = "MONITOR"

    return action

# =========================
# EXECUTE
# =========================

def executor(ip, action):
    timestamp = time.time()

    knowledge_base[ip]["actions"].append({
        "time": timestamp,
        "action": action
    })

    payload = {"ip": ip, "action": action, "timestamp": timestamp}

    try:
        requests.post(APP_URL, json=payload, timeout=1)
    except:
        pass

    return timestamp

# =========================
# LOOP
# =========================

@app.route("/analyze", methods=["POST"])
def analyze_route():
    global tp, fp

    data = request.json
    ip = data["ip"]
    ground_truth = data.get("label")

    now = time.time()

    # MONITOR
    x, cold_start = build_features(data)

    # ANALYZE
    if cold_start:
        # COLD-START: see build_features() docstring. Skip predict_proba
        # entirely and default to the "monitor" tier (watchlist) instead
        # of classifying degenerate/out-of-distribution input.
        label, prob = "Benign", 0.0
    else:
        label, prob = analyzer(x)

    # ======================================================
    # ✔ MTTD FIX (CORRIGIDO)
    # ======================================================

    if ground_truth == "DDoS":
        if ip not in attack_start_time:
            attack_start_time[ip] = now
        else:
            attack_start_time[ip] = min(attack_start_time[ip], now)

    if label == "DDoS":
        if ip not in detection_time:
            detection_time[ip] = now
        else:
            detection_time[ip] = min(detection_time[ip], now)

    if ip in attack_start_time and ip in detection_time:
        mttd = detection_time[ip] - attack_start_time[ip]

        if mttd >= 0:
            mttd_list.append(mttd)

    # ======================================================
    # PRECISION
    # ======================================================

    if ground_truth:
        if label == "DDoS":
            if ground_truth == "DDoS":
                tp += 1
            else:
                fp += 1

    # PLAN
    if cold_start:
        action = "MONITOR"
        knowledge_base[ip]["state"] = "SUSPECT"
    else:
        action = planner(ip, label)

    # EXECUTE
    exec_time = executor(ip, action)

    # ======================================================
    # MTTM (CORRIGIDO IMPLICITAMENTE)
    # ======================================================

    if label == "DDoS" and action in ["BLOCK", "RATE_LIMIT"]:
        if ip in detection_time:
            mttm = exec_time - detection_time[ip]
            if mttm >= 0:
                mttm_list.append(mttm)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return jsonify({
        "label": label,
        "prob": float(prob),
        "action": action,
        "metrics": {
            "precision": precision,
            "mttd_avg": float(np.mean(mttd_list)) if mttd_list else 0,
            "mttm_avg": float(np.mean(mttm_list)) if mttm_list else 0
        }
    })

# =========================
# METRICS
# =========================

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "tracked_ips": len(knowledge_base),
        "blocked_ips": sum(1 for v in knowledge_base.values() if v["state"] == "BLOCKED"),
        "suspect_ips": sum(1 for v in knowledge_base.values() if v["state"] == "SUSPECT"),
        "normal_ips": sum(1 for v in knowledge_base.values() if v["state"] == "NORMAL"),
        "mttd_avg": float(np.mean(mttd_list)) if mttd_list else 0,
        "mttm_avg": float(np.mean(mttm_list)) if mttm_list else 0
    })

# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)