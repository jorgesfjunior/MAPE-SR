from flask import Flask, request, jsonify
import time
import os
import joblib
import numpy as np
from collections import defaultdict, deque
import requests

app = Flask(__name__)

# ============================================================
# Phase 5 intermediate baseline: SAME probability-adaptive threshold
# mechanism as agent_nist.py's Plan/Adapt phases (block_min_prob /
# contain_min_prob adjustment via adapt_policy()), but WITHOUT the
# six-tier/NIST-CSF-derived response structure. Plan/Execute use
# agent_pure_mapek.py's simpler 3-state model (NORMAL/SUSPECT/BLOCKED,
# actions ALLOW/MONITOR/RATE_LIMIT/BLOCK) instead of NIST's five states
# and eradicate/block/contain/monitor/recover six-tier policy. This
# isolates ONE variable against the other two agents: adaptive vs fixed
# thresholds, holding classifier/features/cold-start/tier-count-of-
# alternatives-per-branch otherwise comparable.
#
# Everything below MONITOR/ANALYZE (model loading, build_features) and
# ADAPT (compute_learning_gain, adapt_policy, PSI) is copied verbatim
# from agent_nist.py -- see that file for the full rationale comments.
# Only PLAN (planner()) and the knowledge_base/metrics shape are new,
# modeled on agent_pure_mapek.py's simpler state model.
# ============================================================

# Anchored to this script's own location, not the caller's cwd -- see
# agent_nist.py / agent_pure_mapek.py for why (stale-model bug history).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "..", "models")

MODEL_PATH = os.path.join(_MODELS_DIR, "rf_ddos_model.pkl")
SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(_MODELS_DIR, "threshold.pkl")
FEATURES_PATH = os.path.join(_MODELS_DIR, "feature_cols.pkl")
APP_URL = "http://app:5000/execute"

model = joblib.load(MODEL_PATH)
model.n_jobs = 1  # see agent_nist.py: predict_proba aggregates trees via joblib
                  # threads when n_jobs=-1, causing ~1-ULP nondeterminism.
                  # Forcing single-threaded inference makes it bit-reproducible.
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# Exact column order the scaler/model were fit on -- build_features() below
# MUST assemble the vector in this order (see agent_nist.py for details).
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
# KNOWLEDGE BASE
# =========================
#
# Simpler than agent_nist.py's: no consecutive_blocks/eradication
# escalation, no incidents/active_incident/incident_log NIST phase
# tracking. Same "policy" concept as agent_nist.py (min_detections +
# adaptive min_prob per tier) but only two actionable tiers (block,
# contain) -- no eradicate/monitor-as-DE.CM-watchlist/recover tiers.

ip_history = defaultdict(lambda: deque(maxlen=50))

knowledge_base = defaultdict(lambda: {
    "detections": 0,
    "state": "NORMAL",   # NORMAL | SUSPECT | BLOCKED (agent_pure_mapek.py's model)

    "policy": {
        "block":   {"min_detections": 5, "min_prob": 0.9},
        "contain": {"min_detections": 2, "min_prob": 0.7},
    },

    "metrics": {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "events": 0,

        "cycle_times": [],
        "precision_history": [],
        "recall_history": [],
        "fn_history": [],

        "block_min_prob_history": [],   # block threshold over time
        "psi_history": [],              # Policy Stability Index (variance of recent thresholds)
    },

    "learning_gain_history": [],
    "adaptations": [],   # adapt_policy() audit trail, same as agent_nist.py
})

# =========================
# FEATURES (verbatim from agent_nist.py)
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

    hist = ip_history[ip]
    hist.append((now, size))

    window = 10
    recent = [(t, s) for t, s in hist if now - t < window]

    if len(recent) < COLD_START_MIN_SAMPLES:
        return None, True

    sizes = [s for _, s in recent]

    # UNITS: our timestamps are time.time()-based (seconds), but the
    # CICDDoS2019 training data's Fwd IAT columns are in microseconds --
    # verified empirically against this repo's own training parquet (see
    # agent_nist.py for the full verification note). Multiply by 1e6 so the
    # live feature lands on the same scale the scaler/model were fit on.
    gaps = [(recent[i][0] - recent[i - 1][0]) * 1_000_000 for i in range(1, len(recent))]

    # ddof=1 (Bessel-corrected "sample" std) -- see agent_nist.py for the
    # CICFlowMeter-convention caveat (best-effort assumption, not confirmed).
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
# PLAN -- simplified (agent_pure_mapek.py's tier/state model, NIST's
# adaptive-threshold OR-logic: detections-count OR adaptive min_prob)
# =========================

def planner(ip, label, prob):
    kb            = knowledge_base[ip]
    policy        = kb["policy"]

    if label == "DDoS":
        kb["detections"] += 1

        if (kb["detections"] >= policy["block"]["min_detections"]
                or prob >= policy["block"]["min_prob"]):
            action, state = "BLOCK", "BLOCKED"
        elif (kb["detections"] >= policy["contain"]["min_detections"]
                or prob >= policy["contain"]["min_prob"]):
            action, state = "RATE_LIMIT", "SUSPECT"
        else:
            action, state = "MONITOR", "SUSPECT"
    else:
        kb["detections"] = max(0, kb["detections"] - 1)

        if kb["detections"] == 0:
            action, state = "ALLOW", "NORMAL"
        else:
            action, state = "MONITOR", kb["state"]   # unchanged, as in agent_pure_mapek.py

    return action, state

# =========================
# EXECUTE (verbatim pattern from both existing agents)
# =========================

def executor(ip, action):
    start = time.time()

    try:
        requests.post(APP_URL, json={
            "ip": ip,
            "action": action,
            "timestamp": start
        }, timeout=1)
    except:
        pass

    return time.time() - start

# =========================
# METRICS (verbatim from agent_nist.py)
# =========================

def evaluate(ip, predicted, ground_truth):
    kb = knowledge_base[ip]
    m = kb["metrics"]

    m["events"] += 1

    if predicted == "DDoS" and ground_truth == "DDoS":
        m["tp"] += 1
    elif predicted == "DDoS" and ground_truth == "Benign":
        m["fp"] += 1
    elif predicted == "Benign" and ground_truth == "DDoS":
        m["fn"] += 1
    else:
        m["tn"] += 1

# =========================
# LEARNING GAIN (verbatim from agent_nist.py)
# =========================

def compute_learning_gain(kb):
    m = kb["metrics"]

    if m["events"] < 5:
        return 0.0

    precision = m["tp"] / (m["tp"] + m["fp"] + 1e-6)
    recall = m["tp"] / (m["tp"] + m["fn"] + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    baseline = 0.3
    gain = (f1 - baseline) / (baseline + 1e-6)

    kb["learning_gain_history"].append(gain)

    return gain

# =========================
# ADAPT POLICY (verbatim from agent_nist.py -- the mechanism under test)
# =========================
#
# Triggered every 10 events per IP. Adaptation rules (combination of
# precision, recall, gain):
#   - Low recall only    -> lower min_prob (more sensitive, catch more attacks)
#   - Low precision only  -> raise min_prob (reduce false positives)
#   - Both low            -> slight lean toward sensitivity (security priority)
#   - Both healthy        -> fine-tune with learning_gain signal
# Bounds: min_prob in [0.5, 0.99] for both block and contain tiers.

ADAPT_EVERY = 10
ADAPT_STEP = 0.05
MIN_PROB_LOWER = 0.5
MIN_PROB_UPPER = 0.99
THRESHOLD_PRECISION = 0.6
THRESHOLD_RECALL = 0.6

def adapt_policy(ip, gain):
    kb = knowledge_base[ip]
    m = kb["metrics"]

    if m["events"] < ADAPT_EVERY or m["events"] % ADAPT_EVERY != 0:
        return

    precision = m["tp"] / (m["tp"] + m["fp"] + 1e-6)
    recall    = m["tp"] / (m["tp"] + m["fn"] + 1e-6)

    low_precision = precision < THRESHOLD_PRECISION
    low_recall    = recall    < THRESHOLD_RECALL

    if low_recall and not low_precision:
        delta  = -ADAPT_STEP
        reason = "low_recall"
    elif low_precision and not low_recall:
        delta  = +ADAPT_STEP
        reason = "low_precision"
    elif low_precision and low_recall:
        delta  = -ADAPT_STEP * 0.5   # security priority: lean toward sensitivity
        reason = "low_precision_and_recall"
    else:
        # Both healthy — fine-tune with learning gain
        if gain > 0.5:
            delta  = +ADAPT_STEP * 0.5
            reason = "gain_high_relax"
        elif gain < 0.0:
            delta  = -ADAPT_STEP * 0.5
            reason = "gain_negative_tighten"
        else:
            return  # performance stable, no adaptation needed

    policy = kb["policy"]
    for tier in ["block", "contain"]:
        old_val = policy[tier]["min_prob"]
        policy[tier]["min_prob"] = round(
            float(np.clip(old_val + delta, MIN_PROB_LOWER, MIN_PROB_UPPER)), 3
        )

    record = {
        "event":           m["events"],
        "reason":          reason,
        "delta":           round(delta, 3),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "gain":            round(gain, 4),
        "block_min_prob":  policy["block"]["min_prob"],
        "contain_min_prob": policy["contain"]["min_prob"]
    }
    kb["adaptations"].append(record)

    # Policy Stability Index: variance of the last 5 block_min_prob values.
    kb["metrics"]["block_min_prob_history"].append(policy["block"]["min_prob"])
    prob_hist = kb["metrics"]["block_min_prob_history"]
    if len(prob_hist) >= 3:
        psi = round(float(np.var(prob_hist[-5:])), 6)
        kb["metrics"]["psi_history"].append({"event": m["events"], "psi": psi})

    print(
        f"[ADAPT] {ip} | reason={reason} | delta={delta:+.3f} "
        f"| block={policy['block']['min_prob']:.3f} "
        f"| contain={policy['contain']['min_prob']:.3f}",
        flush=True
    )

# =========================
# ANALYZE (MAPE-K LOOP)
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    ip           = data["ip"]
    ground_truth = data.get("label", "Benign")

    kb = knowledge_base[ip]
    request_time = time.time()

    # =========================
    # FEATURE + PREDICTION
    # =========================
    x, cold_start = build_features(data)

    if cold_start:
        # COLD-START: see build_features() docstring. Skip predict_proba
        # entirely and default to the "monitor" tier instead of
        # classifying degenerate/out-of-distribution input.
        prob = 0.0
        predicted = "Benign"
    else:
        x = scaler.transform(x)
        prob = model.predict_proba(x)[0][1]
        predicted = "DDoS" if prob >= threshold else "Benign"

    # =========================
    # PLAN + EXECUTE
    # =========================
    if cold_start:
        action, state = "MONITOR", "SUSPECT"
    else:
        action, state = planner(ip, predicted, prob)
    kb["state"] = state

    executor(ip, action)

    # =========================
    # METRICS UPDATE
    # =========================
    evaluate(ip, predicted, ground_truth)

    m = kb["metrics"]
    _prec = m["tp"] / (m["tp"] + m["fp"] + 1e-6)
    _rec  = m["tp"] / (m["tp"] + m["fn"] + 1e-6)
    m["precision_history"].append(round(_prec, 4))
    m["recall_history"].append(round(_rec, 4))
    m["fn_history"].append(m["fn"])

    gain = compute_learning_gain(kb)
    adapt_policy(ip, gain)

    cycle_time = round(time.time() - request_time, 6)
    kb["metrics"]["cycle_times"].append(cycle_time)

    psi_hist = kb["metrics"]["psi_history"]
    psi_current = psi_hist[-1]["psi"] if psi_hist else None

    return jsonify({
        "ip":           ip,
        "predicted":    predicted,
        "prob":         float(prob),
        "action":       action,
        "state":        state,
        "learning_gain": gain,
        "cycle_time_s": cycle_time,
        "precision":    round(_prec, 4),
        "recall":       round(_rec, 4),
        "psi_current":  psi_current,
        "policy": {
            "block_min_prob":   kb["policy"]["block"]["min_prob"],
            "contain_min_prob": kb["policy"]["contain"]["min_prob"]
        },
    })

# =========================
# GLOBAL METRICS
# =========================

@app.route("/metrics", methods=["GET"])
def metrics():
    all_kbs = list(knowledge_base.values())

    tp = sum(k["metrics"]["tp"] for k in all_kbs)
    fp = sum(k["metrics"]["fp"] for k in all_kbs)
    fn = sum(k["metrics"]["fn"] for k in all_kbs)
    tn = sum(k["metrics"]["tn"] for k in all_kbs)
    events = sum(k["metrics"]["events"] for k in all_kbs)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    all_psi = [entry["psi"] for k in all_kbs for entry in k["metrics"]["psi_history"]]
    psi_latest = round(all_psi[-1], 6) if all_psi else None
    psi_mean = round(float(np.mean(all_psi)), 6) if all_psi else None

    all_cycle_times = [t for k in all_kbs for t in k["metrics"]["cycle_times"]]
    avg_cycle_time_s = round(float(np.mean(all_cycle_times)), 6) if all_cycle_times else None
    p95_cycle_time_s = round(float(np.percentile(all_cycle_times, 95)), 6) if all_cycle_times else None

    return jsonify({
        "tracked_ips":  len(all_kbs),
        "blocked_ips":  sum(1 for k in all_kbs if k["state"] == "BLOCKED"),
        "suspect_ips":  sum(1 for k in all_kbs if k["state"] == "SUSPECT"),
        "normal_ips":   sum(1 for k in all_kbs if k["state"] == "NORMAL"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "events": events,
        "precision": precision, "recall": recall, "f1": f1,
        "psi_latest": psi_latest,
        "psi_mean": psi_mean,
        "avg_cycle_time_s": avg_cycle_time_s,
        "p95_cycle_time_s": p95_cycle_time_s,
    })

# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
