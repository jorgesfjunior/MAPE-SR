# agent_nist.py -- MAPE-SR, canonical
#
# CONSOLIDATED VERSION. Lineage: original agent_nist.py (produced
# results/raw_runs.csv / raw_runs_3way.csv, i.e. E1 and Phase 5) ->
# agent_nist_fixed.py (one-line fix: planner()'s unconditional
# `detections += 1` before the label branch was cancelling the
# Benign-branch decrement, so detections could never actually decay) ->
# agent_nist_graduated.py (RECOVERING's traffic admission made
# proportional to remaining evidence instead of binary MONITOR-until-
# zero) -> merged here as the canonical agent_nist.py. Both
# agent_nist_fixed.py and agent_nist_graduated.py are deleted (superseded
# by this file, not left as near-duplicates).
#
# The exact pre-consolidation content (what actually produced E1/Phase 5)
# is frozen, unmodified, in agent_nist_e1_archive.py -- see that file's
# header. Confirmed (see project notes / investigation) that swapping
# this file's content does NOT invalidate E1/Phase 5: across all 25
# seeds of each, the archived agent never reached RECOVERING with
# detections > 0 for more than 1 event on any seed, so neither the decay
# fix nor the graduated-admission mechanism below was ever actually
# exercised by those runs.
#
# GRADUATED behavioral change vs the archived version: while RECOVERING,
# traffic admission is proportional to remaining evidence
# (admission_rate = 1 - detections/peak_evidence) instead of binary
# MONITOR-until-detections-hits-zero, PLUS the detections-decay fix that
# makes reaching "detections == 0" possible at all. Search "GRADUATED"
# for every line touched by this behavioral change relative to the
# archived version.

from flask import Flask, request, jsonify
import time
import os
import joblib
import numpy as np
from collections import defaultdict, deque
import requests

app = Flask(__name__)

# GRADUATED: admission-decision RNG. Deliberately NOT Python's global
# `random` module (unseeded here, and would silently share/collide with
# any other consumer of that same global stream) -- a dedicated numpy
# Generator, seeded from the driver's --seed (passed in via the
# ADMISSION_SEED env var, set by test_nist_only.py's set_seed() before
# spawning this subprocess -- every other driver of this file, e.g.
# test_nist_fixed.py historically, must do the same) and decorrelated
# from the driver's own workload-generation stream via
# SeedSequence.spawn(), same "derive a separate child stream, don't reuse
# the parent's" discipline as test_seed_reproducibility.py already
# expects for full-pipeline reproducibility elsewhere in this project.
# Falls back to OS-entropy seeding (irreproducible) only when
# ADMISSION_SEED isn't set at all -- e.g. a bare `python3 agent_nist.py`
# or docker-compose run outside the test drivers -- so standalone runs
# still work, just without the determinism guarantee.
_ADMISSION_SEED_ENV = os.environ.get("ADMISSION_SEED")
if _ADMISSION_SEED_ENV is not None:
    _admission_rng = np.random.default_rng(
        np.random.SeedSequence(int(_ADMISSION_SEED_ENV)).spawn(1)[0]
    )
else:
    _admission_rng = np.random.default_rng()

# Anchored to this script's own location, not the caller's cwd -- the old
# "./models/..." resolved differently depending on who launched the
# process (bare `python3 agent_nist.py`, the test driver's subprocess
# with cwd=agent/, or Docker's WORKDIR), which is exactly how a stale
# model sitting in agent/models/ went undetected through a full
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
# KNOWLEDGE BASE
# =========================

ip_history = defaultdict(lambda: deque(maxlen=50))

knowledge_base = defaultdict(lambda: {
    "detections": 0,
    "state": "NORMAL",
    "trust_score": 1.0,

    # GRADUATED: peak_evidence = max kb["detections"] reached while
    # CONTAINED/ERADICATED during the current incident; admission_rate =
    # this decision's effective admission probability (continuous only
    # while RECOVERING, else 1.0/0.0 mirroring the chosen action).
    "peak_evidence": 0,
    "admission_rate": 1.0,

    # ============================================================
    # GV.PO / GV.RM / ID.RA-06
    # Five response tiers mapped to NIST CSF 2.0 phases:
    #   eradicate → RS.MI-02 (sustained BLOCK after N consecutive)
    #   block     → RS.MI-01 (initial hard containment)
    #   contain   → RS.MI-01 (soft containment via RATE_LIMIT)
    #   monitor   → DE.CM    (watchlist)
    #   recover   → RC.RP-01 (recovery without action, phase tracking)
    #   normal    → baseline
    # ============================================================
    "policy": {
        "eradicate": {"consecutive_blocks": 3, "action": "BLOCK",      "next_state": "ERADICATED"},
        "block":     {"min_detections": 5, "min_prob": 0.9, "action": "BLOCK",      "next_state": "CONTAINED"},
        "contain":   {"min_detections": 2, "min_prob": 0.7, "action": "RATE_LIMIT", "next_state": "CONTAINED"},
        "monitor":   {"action": "MONITOR", "next_state": "SUSPECT"},
        "recover":   {"action": "MONITOR", "next_state": "RECOVERING"},
        "normal":    {"action": "ALLOW",   "next_state": "NORMAL"}
    },
    "consecutive_blocks": 0,

    # =========================
    # METRICS (NIST + MAPEK)
    # =========================
    "metrics": {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "events": 0,

        "detection_times": [],
        "mitigation_times": [],

        # per-event temporal series (one entry per /analyze call)
        "cycle_times": [],          # full MAPE-K cycle duration (s)
        "precision_history": [],    # cumulative precision after each event
        "recall_history": [],       # cumulative recall after each event
        "fn_history": [],           # cumulative FN count after each event

        # policy adaptation series (one entry per adapt_policy trigger)
        "block_min_prob_history": [],   # block threshold over time
        "psi_history": [],              # Policy Stability Index (variance of recent thresholds)

        # collateral damage broken down by traffic phase
        "phase_collateral": {},         # phase → false-positive action count
    },

    "learning_gain_history": [],

    # ============================================================
    # ID.IM-01 / ID.IM-02
    # Registro de adaptações realizadas (auditoria de melhorias)
    # ============================================================
    "adaptations": [],

    # ============================================================
    # DE.CM / DE.AE / RS.MA / RS.MI
    # Registro de incidentes e fases NIST CSF 2.0 por IP
    # ============================================================
    "incidents": [],
    "active_incident": None,
    "last_normal_packet_at": None
})

# =========================
# INCIDENT LOG (GLOBAL)
# =========================

# ============================================================
# DE.AE-03 — Event correlation and analysis
# RS.MA-01 — Incident response is managed
#
# incident_log: flat exportable record of all incidents, keyed
#               by (ip, incident_id) — survives KB resets
# incident_counter: per-IP incrementing incident ID
# ============================================================

incident_log     = {}                    # (ip, incident_id) → incident record
incident_counter = defaultdict(int)      # per-IP incident counter

# =========================
# FEATURES
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
# PLAN
# =========================

def planner(ip, label, prob):
    # ============================================================
    # RS.MA-02 / RS.MA-03 — Triage, validation, categorisation
    # RS.MI-01 — Containment   (RATE_LIMIT / first BLOCK)
    # RS.MI-02 — Eradication   (sustained BLOCK ≥ N consecutive)
    # RC.RP-01 — Recovery      (phase tracked, no action yet)
    # ============================================================
    kb            = knowledge_base[ip]
    policy        = kb["policy"]
    current_state = kb["state"]

    # GRADUATED: overridden below only inside the graduated-recovery branch;
    # every other branch derives it from the final selected action instead.
    admission_rate = None

    if label == "DDoS":
        kb["detections"] += 1

        if (kb["detections"] >= policy["block"]["min_detections"]
                or prob >= policy["block"]["min_prob"]):
            kb["consecutive_blocks"] += 1
            if kb["consecutive_blocks"] >= policy["eradicate"]["consecutive_blocks"]:
                selected = policy["eradicate"]          # RS.MI-02
            else:
                selected = policy["block"]              # RS.MI-01 (hard contain)
        elif (kb["detections"] >= policy["contain"]["min_detections"]
                or prob >= policy["contain"]["min_prob"]):
            kb["consecutive_blocks"] = 0
            selected = policy["contain"]                # RS.MI-01 (soft contain)
        else:
            kb["consecutive_blocks"] = 0
            selected = policy["monitor"]                # DE.CM watchlist

        # GRADUATED: record the peak evidence reached this incident, the
        # instant it's known (i.e. the moment CONTAINED/ERADICATED is
        # (re-)entered) -- this is what admission_rate decays against
        # during the eventual RECOVERING phase.
        if selected["next_state"] in ("CONTAINED", "ERADICATED"):
            kb["peak_evidence"] = max(kb.get("peak_evidence", 0), kb["detections"])
    else:
        kb["consecutive_blocks"] = 0
        kb["detections"] = max(0, kb["detections"] - 1)

        if current_state in ("CONTAINED", "ERADICATED", "RECOVERING"):
            if kb["detections"] > 0:
                # GRADUATED: RC.RP-01, graduated admission -- instead of
                # binary MONITOR-until-zero, admit a fraction of requests
                # proportional to how much evidence has decayed since this
                # incident's peak. peak is re-maxed against the current
                # count as a defensive floor (keeps the ratio <= 1 even if
                # peak_evidence was somehow never recorded).
                peak = max(kb.get("peak_evidence", 0), kb["detections"])
                admission_rate = min(1.0, max(0.0, 1 - (kb["detections"] / peak)))

                if _admission_rng.random() < admission_rate:
                    selected = {"action": "ALLOW", "next_state": "RECOVERING"}
                else:
                    selected = policy["recover"]
            else:
                selected = policy["normal"]
        elif kb["detections"] == 0:
            selected = policy["normal"]
        else:
            selected = policy["monitor"]

    # GRADUATED: peak_evidence only means something for the incident that
    # set it -- clear it the moment that incident actually closes.
    if selected["next_state"] == "NORMAL":
        kb["peak_evidence"] = 0

    kb["admission_rate"] = (
        admission_rate if admission_rate is not None
        else (1.0 if selected["action"] == "ALLOW" else 0.0)
    )

    return selected["action"], selected["next_state"]

# =========================
# EXECUTE (SEM TIMING AQUI)
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
# METRICS
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
# LEARNING GAIN
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
# ADAPT POLICY (CLOSED LOOP)
# =========================

# ============================================================
# ID.IM-01 — Improvements are identified from evaluations
# ID.IM-02 — Improvements are planned and implemented
# ID.IM-03 — Organizational improvement is communicated
#
# Triggered every 10 events per IP.
# Adaptation rules (combination of precision, recall, gain):
#   - Low recall only  → lower min_prob (more sensitive, catch more attacks)
#   - Low precision only → raise min_prob (reduce false positives)
#   - Both low         → slight lean toward sensitivity (security priority)
#   - Both healthy     → fine-tune with learning_gain signal
# Bounds: min_prob ∈ [0.5, 0.99] for both block and suspect tiers.
# ============================================================

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
    # PSI → 0 means the policy has converged; high PSI means it is still oscillating.
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
    phase        = data.get("phase", "unknown")   # traffic phase label from the experiment

    kb = knowledge_base[ip]

    # ============================================================
    # Capture packet arrival time and previous state
    # before any processing — used for NIST phase boundaries
    # ============================================================
    request_time = time.time()
    prev_state   = kb["state"]

    if prev_state == "NORMAL":
        kb["last_normal_packet_at"] = request_time

    # =========================
    # FEATURE + PREDICTION
    # =========================
    x, cold_start = build_features(data)

    if cold_start:
        # COLD-START: see build_features() docstring. Skip predict_proba
        # entirely and default to the "monitor" tier (DE.CM watchlist)
        # instead of classifying degenerate/out-of-distribution input.
        prob = 0.0
        predicted = "Benign"
    else:
        x = scaler.transform(x)
        prob = model.predict_proba(x)[0][1]
        predicted = "DDoS" if prob >= threshold else "Benign"

    # =========================
    # DETECTION TIMER
    # =========================
    detection_start = time.time()

    if cold_start:
        action, state = kb["policy"]["monitor"]["action"], kb["policy"]["monitor"]["next_state"]
        kb["admission_rate"] = 1.0 if action == "ALLOW" else 0.0  # GRADUATED
    else:
        action, state = planner(ip, predicted, prob)
    kb["state"] = state

    # =========================
    # EXECUTION
    # =========================
    executor(ip, action)
    mitigation_done_at = time.time()

    # ============================================================
    # NIST CSF 2.0 — Phase Boundary Tracking
    #
    # DE           (DE.CM / DE.AE)  — packet arrival → first DDoS prediction
    # RS_CONTAIN   (RS.MI-01)       — first DDoS prediction → first RATE_LIMIT/BLOCK
    # RS_ERADICATE (RS.MI-02)       — first ERADICATED state → first Benign prediction
    # RC           (RC.RP-01)       — first Benign post-containment → NORMAL (no action yet)
    #
    # Incident opens:  NORMAL → SUSPECT / CONTAINED / ERADICATED
    # Incident closes: state returns to NORMAL
    # ============================================================

    if prev_state == "NORMAL" and state in ("SUSPECT", "CONTAINED", "ERADICATED"):
        inc_id = incident_counter[ip] + 1
        incident_counter[ip] = inc_id

        incident = {
            "incident_id": inc_id,
            "ip":          ip,
            "opened_at":   request_time,
            "closed_at":   None,
            "phases": {
                "DE": {
                    "start":      kb["last_normal_packet_at"],
                    "end":        detection_start,
                    "duration_s": round(detection_start - kb["last_normal_packet_at"], 6)
                },
                "RS_CONTAIN": {
                    "start":      detection_start,
                    "end":        None,
                    "duration_s": None
                },
                "RS_ERADICATE": {
                    "start":      None,
                    "end":        None,
                    "duration_s": None
                },
                "RC": {
                    "start":      None,
                    "end":        None,
                    "duration_s": None
                }
            }
        }
        kb["incidents"].append(incident)
        kb["active_incident"] = incident
        incident_log[(ip, inc_id)] = incident

    if kb["active_incident"] is not None:
        active  = kb["active_incident"]
        phases  = active["phases"]

        # RS_CONTAIN closes on first RATE_LIMIT or BLOCK (RS.MI-01)
        if phases["RS_CONTAIN"]["end"] is None and action in ("BLOCK", "RATE_LIMIT"):
            phases["RS_CONTAIN"]["end"]        = mitigation_done_at
            phases["RS_CONTAIN"]["duration_s"] = round(
                mitigation_done_at - phases["RS_CONTAIN"]["start"], 6
            )

        # RS_ERADICATE opens when state first enters ERADICATED (RS.MI-02)
        if state == "ERADICATED" and phases["RS_ERADICATE"]["start"] is None:
            phases["RS_ERADICATE"]["start"] = mitigation_done_at

        # RS_ERADICATE closes on first Benign prediction after eradication
        if (phases["RS_ERADICATE"]["start"] is not None
                and phases["RS_ERADICATE"]["end"] is None
                and predicted == "Benign"):
            phases["RS_ERADICATE"]["end"]        = detection_start
            phases["RS_ERADICATE"]["duration_s"] = round(
                detection_start - phases["RS_ERADICATE"]["start"], 6
            )

        # RC opens on first Benign prediction post-containment (RC.RP-01)
        if (prev_state in ("CONTAINED", "ERADICATED", "RECOVERING")
                and predicted == "Benign"
                and phases["RC"]["start"] is None):
            phases["RC"]["start"] = detection_start

        # RC closes and incident ends when state returns to NORMAL
        if state == "NORMAL" and active["closed_at"] is None:
            if phases["RC"]["start"] is not None and phases["RC"]["end"] is None:
                phases["RC"]["end"]        = request_time
                phases["RC"]["duration_s"] = round(
                    request_time - phases["RC"]["start"], 6
                )
            active["closed_at"]   = request_time
            kb["active_incident"] = None

    # =========================
    # METRICS UPDATE
    # =========================
    evaluate(ip, predicted, ground_truth)

    # Per-event temporal series (cumulative precision/recall/FN after this event)
    m = kb["metrics"]
    _prec = m["tp"] / (m["tp"] + m["fp"] + 1e-6)
    _rec  = m["tp"] / (m["tp"] + m["fn"] + 1e-6)
    m["precision_history"].append(round(_prec, 4))
    m["recall_history"].append(round(_rec, 4))
    m["fn_history"].append(m["fn"])

    # Phase-stratified collateral: false-positive enforcement actions per traffic phase
    if ground_truth == "Benign" and action in ("BLOCK", "RATE_LIMIT"):
        m["phase_collateral"][phase] = m["phase_collateral"].get(phase, 0) + 1

    # =========================
    # SINGLE MITIGATION TIME (FIX PRINCIPAL)
    # =========================
    mitigation_time = time.time() - detection_start
    kb["metrics"]["mitigation_times"].append(mitigation_time)

    gain = compute_learning_gain(kb)

    # ============================================================
    # ID.IM-01 / ID.IM-02 — Close the adaptation loop
    # ============================================================
    adapt_policy(ip, gain)

    # Full MAPE-K cycle time: from packet arrival to end of adaptation
    cycle_time = round(time.time() - request_time, 6)
    kb["metrics"]["cycle_times"].append(cycle_time)

    psi_hist = kb["metrics"]["psi_history"]
    psi_current = psi_hist[-1]["psi"] if psi_hist else None

    active = kb["active_incident"]
    return jsonify({
        "ip":           ip,
        "predicted":    predicted,
        "prob":         float(prob),
        "action":       action,
        "state":        state,
        "admission_rate": kb.get("admission_rate"),  # GRADUATED
        "peak_evidence":  kb.get("peak_evidence"),   # GRADUATED
        "learning_gain": gain,
        "cycle_time_s": cycle_time,
        "precision":    round(_prec, 4),
        "recall":       round(_rec, 4),
        "psi_current":  psi_current,
        "policy": {
            "block_min_prob":   kb["policy"]["block"]["min_prob"],
            "contain_min_prob": kb["policy"]["contain"]["min_prob"]
        },
        "incident": {
            "incident_id":     active["incident_id"] if active else None,
            "opened_at":       active["opened_at"]   if active else None,
            "phases":          active["phases"]       if active else None,
            "total_incidents": len(kb["incidents"])
        }
    })

# =========================
# GLOBAL METRICS
# =========================

@app.route("/metrics", methods=["GET"])
def metrics():
    all_kbs = knowledge_base.values()

    total = {
        "tp": sum(k["metrics"]["tp"] for k in all_kbs),
        "fp": sum(k["metrics"]["fp"] for k in all_kbs),
        "fn": sum(k["metrics"]["fn"] for k in all_kbs),
        "tn": sum(k["metrics"]["tn"] for k in all_kbs),
        "events": sum(k["metrics"]["events"] for k in all_kbs),
        "suspect_ips":    sum(1 for k in all_kbs if k["state"] == "SUSPECT"),
        "contained_ips":  sum(1 for k in all_kbs if k["state"] == "CONTAINED"),
        "eradicated_ips": sum(1 for k in all_kbs if k["state"] == "ERADICATED"),
        "recovering_ips": sum(1 for k in all_kbs if k["state"] == "RECOVERING"),
        "tracked_ips":    len(all_kbs)
    }

    precision = total["tp"] / (total["tp"] + total["fp"] + 1e-6)
    recall = total["tp"] / (total["tp"] + total["fn"] + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    total["precision"] = precision
    total["recall"] = recall
    total["f1"] = f1

    # ============================================================
    # DE.AE-03 / RS.MA-01 — Incident phase duration statistics
    # ============================================================
    all_incidents       = list(incident_log.values())
    de_durations        = [i["phases"]["DE"]["duration_s"]           for i in all_incidents if i["phases"]["DE"]["duration_s"]           is not None]
    rs_contain_durs     = [i["phases"]["RS_CONTAIN"]["duration_s"]   for i in all_incidents if i["phases"]["RS_CONTAIN"]["duration_s"]   is not None]
    rs_eradicate_durs   = [i["phases"]["RS_ERADICATE"]["duration_s"] for i in all_incidents if i["phases"]["RS_ERADICATE"]["duration_s"] is not None]
    rc_durations        = [i["phases"]["RC"]["duration_s"]           for i in all_incidents if i["phases"]["RC"]["duration_s"]           is not None]

    total["total_incidents"]         = len(all_incidents)
    total["active_incidents"]        = sum(1 for k in knowledge_base.values() if k["active_incident"] is not None)
    total["avg_de_duration_s"]       = round(float(np.mean(de_durations)),      6) if de_durations      else None
    total["avg_rs_contain_s"]        = round(float(np.mean(rs_contain_durs)),   6) if rs_contain_durs   else None
    total["avg_rs_eradicate_s"]      = round(float(np.mean(rs_eradicate_durs)), 6) if rs_eradicate_durs else None
    total["avg_rc_duration_s"]       = round(float(np.mean(rc_durations)),      6) if rc_durations      else None

    # ============================================================
    # Cycle time statistics — MAPE-K overhead per packet
    # ============================================================
    all_cycle_times = [t for k in knowledge_base.values() for t in k["metrics"]["cycle_times"]]
    total["avg_cycle_time_s"] = round(float(np.mean(all_cycle_times)),            6) if all_cycle_times else None
    total["std_cycle_time_s"] = round(float(np.std(all_cycle_times)),             6) if all_cycle_times else None
    total["p95_cycle_time_s"] = round(float(np.percentile(all_cycle_times, 95)), 6) if all_cycle_times else None

    # ============================================================
    # Policy Stability Index (PSI) — convergence of adaptation
    # PSI → 0 means thresholds have stabilised
    # ============================================================
    all_psi = [entry["psi"] for k in knowledge_base.values() for entry in k["metrics"]["psi_history"]]
    total["psi_latest"] = round(all_psi[-1],              6) if all_psi else None
    total["psi_mean"]   = round(float(np.mean(all_psi)), 6) if all_psi else None

    # ============================================================
    # Phase-stratified collateral damage aggregated across all IPs
    # ============================================================
    phase_col: dict = {}
    for k in knowledge_base.values():
        for ph, cnt in k["metrics"]["phase_collateral"].items():
            phase_col[ph] = phase_col.get(ph, 0) + cnt
    total["phase_collateral"] = phase_col

    return jsonify(total)

# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)