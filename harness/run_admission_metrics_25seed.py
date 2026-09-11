"""
run_admission_metrics_25seed.py

25-seed x 3-configuration (Baseline / Adaptive-Generic / MAPE-SR) run
over the quiet-period recovery workload from workload/generate.py (20
baseline + 30 attack, DrDoS_NTP-calibrated, + 50 quiet-closure probes).
Produces the two admission metrics reported in the paper's Results
section for this comparison: cumulative legitimate-traffic admission
and admission oscillation.

MAPE-SR here means the full graduated-response agent (CONTAINED /
ERADICATED escalation tiers, RECOVERING admission phase with a rising
per-event admission_rate) -- see harness/drivers/mapesr_driver.py's
comment block for the exact response fields this depends on. No agent
source is included in this repository; the protocol used to talk to
each of the three agents under test is fully documented in
harness/drivers/{baseline,adaptive_generic,mapesr}_driver.py.

For each (seed, script) pair, this script keeps the FULL per-event
(action, state, ground_truth) sequence for the incident's IP (ATTACK_IP)
in memory and derives two per-incident metrics from it directly -- see
this module's compute_incident_metrics() and
analysis/analyze_admission_metrics_25seed.py's module docstring for the
exact metric definitions. Only the derived per-incident numbers are
written out, to results/cumulative_admission_25seed.csv and
results/oscillation_25seed.csv.

The Baseline agent's /analyze response has no "state" field (unlike
Adaptive-Generic and MAPE-SR, which both return one) -- see
_reconstruct_pure_mapek_states() below for how its 4-state machine is
replayed exactly from (action, predicted label) alone, with no agent
file needed.

Run: python harness/run_admission_metrics_25seed.py
"""
import contextlib
import csv
import io
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from workload.generate import ATTACK_IP, build_workload  # noqa: E402
from workload.generate import set_seed as set_workload_seed  # noqa: E402
from harness.drivers import baseline_driver, adaptive_generic_driver, mapesr_driver  # noqa: E402

# (driver module, script id used in the output CSVs / analysis scripts,
#  display label). The script id strings match the ones already used by
# analysis/analyze_results_3way.py's SCRIPT_BASELINE / SCRIPT_AG /
# SCRIPT_MAPESR constants.
DRIVERS = [
    (baseline_driver, "test_pure_mapek", "Baseline"),
    (adaptive_generic_driver, "test_adaptive_generic", "Adaptive-Generic"),
    (mapesr_driver, "test_nist_only", "MAPE-SR"),
]

# Per-agent state vocabulary for the two metrics (see analysis script's
# docstring): "restrictive" states are what the incident is considered to
# have entered containment under (metric 1's window start); "restricted
# phase" is the specific state metric 2's oscillation count is scoped to.
# Baseline/Adaptive-Generic have no CONTAINED/ERADICATED distinction (a
# single BLOCKED tier is their strongest response) -- BLOCKED is used for
# both roles for them, matching the task's own choice of BLOCKED as their
# restricted-phase state for metric 2.
RESTRICTIVE_STATES = {
    "test_pure_mapek": {"BLOCKED"},
    "test_adaptive_generic": {"BLOCKED"},
    "test_nist_only": {"CONTAINED", "ERADICATED"},
}
RESTRICTED_PHASE_STATE = {
    "test_pure_mapek": "BLOCKED",
    "test_adaptive_generic": "BLOCKED",
    "test_nist_only": "RECOVERING",
}

QUIET_N = 50  # established safe margin -- see workload/generate.py
SEEDS = list(range(1, 26))

CUM_CSV_PATH = os.path.join(REPO_ROOT, "results", "cumulative_admission_25seed.csv")
OSC_CSV_PATH = os.path.join(REPO_ROOT, "results", "oscillation_25seed.csv")

CUM_FIELDS = [
    "seed", "script", "label", "incident_start_idx", "window_len",
    "closed_within_budget", "n_benign_in_window", "n_allow_in_window",
    "cumulative_admission_fraction",
]
OSC_FIELDS = [
    "seed", "script", "label", "restricted_phase_len",
    "n_allow_monitor_transitions",
]


def run_one(driver_module, seed, quiet_n):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        driver_module.set_seed(seed)
        set_workload_seed(seed)
        proc = driver_module.start_agent()
        try:
            driver_module.wait_ready()
            workload = build_workload(quiet_n)
            logs = [driver_module.send(phase, ip, ts, sz, gt)
                    for phase, ip, ts, sz, gt in workload]
        finally:
            driver_module.stop_agent(proc)

    return logs


def _reconstruct_pure_mapek_states(atk_events):
    """Replays the Baseline agent's documented planner()/cold-start state
    machine from (action, predicted label) alone, exactly (not a
    heuristic):

      action == "ALLOW"       -> state = NORMAL   (Benign branch, detections hit 0)
      action == "BLOCK"       -> state = BLOCKED  (DDoS branch, detections >= 5)
      action == "RATE_LIMIT"  -> state = SUSPECT  (DDoS branch, 2 <= detections < 5)
      action == "MONITOR":
        label == "DDoS"       -> state = SUSPECT  (DDoS branch, detections < 2 --
                                                     planner() sets this explicitly)
        label == "Benign"     -> state unchanged  (Benign branch, detections > 0
                                                     after decrement -- state is
                                                     simply not reassigned)

    The only state this doesn't independently derive is cold-start's
    forced (action=MONITOR, state=SUSPECT) for ATTACK_IP's first two
    events (fewer than 3 samples in its warm-up window) -- those get the
    initial default "NORMAL" carried into them instead of "SUSPECT". This
    never matters here: every subsequent DDoS-branch event sets state
    unconditionally regardless of what came before, and the attack phase
    reliably produces DDoS-labeled events from the 3rd ATTACK_IP event
    onward, well before any incident reaches BLOCKED -- so the
    cold-start-era value is always overwritten before metric 1's window
    or metric 2's restricted phase is ever reached.
    """
    state = "NORMAL"
    out = []
    for e in atk_events:
        action = e.get("action")
        label = e.get("label")
        if action == "ALLOW":
            state = "NORMAL"
        elif action == "BLOCK":
            state = "BLOCKED"
        elif action == "RATE_LIMIT":
            state = "SUSPECT"
        elif action == "MONITOR" and label == "DDoS":
            state = "SUSPECT"
        # else: action == "MONITOR" and label == "Benign" -> unchanged
        out.append({**e, "state": state})
    return out


def compute_incident_metrics(driver_name, atk_events):
    if driver_name == "test_pure_mapek":
        atk_events = _reconstruct_pure_mapek_states(atk_events)

    restrictive = RESTRICTIVE_STATES[driver_name]
    start_idx = next((i for i, e in enumerate(atk_events) if e.get("state") in restrictive), None)
    if start_idx is None:
        return None, atk_events

    close_idx = None
    for j in range(start_idx, len(atk_events)):
        if atk_events[j].get("state") == "NORMAL":
            close_idx = j
            break

    if close_idx is not None:
        window = atk_events[start_idx:close_idx + 1]
        closed = True
    else:
        window = atk_events[start_idx:]
        closed = False

    benign_in_window = [e for e in window if e.get("_ground_truth") == "Benign"]
    n_benign = len(benign_in_window)
    n_allow = sum(1 for e in benign_in_window if e.get("action") == "ALLOW")
    cum_fraction = (n_allow / n_benign) if n_benign > 0 else None

    restricted_state = RESTRICTED_PHASE_STATE[driver_name]
    restricted_seq = [e.get("action") for e in atk_events if e.get("state") == restricted_state]
    transitions = sum(
        1 for a, b in zip(restricted_seq, restricted_seq[1:])
        if {a, b} == {"ALLOW", "MONITOR"}
    )

    cum_row = {
        "incident_start_idx": start_idx,
        "window_len": len(window),
        "closed_within_budget": closed,
        "n_benign_in_window": n_benign,
        "n_allow_in_window": n_allow,
        "cumulative_admission_fraction": round(cum_fraction, 6) if cum_fraction is not None else "",
    }
    osc_row = {
        "restricted_phase_len": len(restricted_seq),
        "n_allow_monitor_transitions": transitions,
    }
    return (cum_row, osc_row), atk_events


def _fresh_csv(path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        os.remove(path)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def _append_row(path, fields, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


def main():
    _fresh_csv(CUM_CSV_PATH, CUM_FIELDS)
    _fresh_csv(OSC_CSV_PATH, OSC_FIELDS)

    n_no_incident = []

    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        for driver_module, driver_name, label in DRIVERS:
            logs = run_one(driver_module, seed, QUIET_N)
            atk_events = [e for e in logs if e.get("_ip") == ATTACK_IP]

            result, _ = compute_incident_metrics(driver_name, atk_events)
            if result is None:
                n_no_incident.append((seed, driver_name))
                print(f"[{label:>16}] seed={seed}: WARNING -- incident never reached a "
                      f"restrictive state, skipped")
                continue

            cum_row, osc_row = result
            _append_row(CUM_CSV_PATH, CUM_FIELDS,
                        {"seed": seed, "script": driver_name, "label": label, **cum_row})
            _append_row(OSC_CSV_PATH, OSC_FIELDS,
                        {"seed": seed, "script": driver_name, "label": label, **osc_row})

            print(f"[{label:>16}] seed={seed}: "
                  f"cum_admission={cum_row['cumulative_admission_fraction']} "
                  f"({cum_row['n_allow_in_window']}/{cum_row['n_benign_in_window']}, "
                  f"closed={cum_row['closed_within_budget']})  "
                  f"oscillation_transitions={osc_row['n_allow_monitor_transitions']} "
                  f"(restricted_phase_len={osc_row['restricted_phase_len']})")

    print(f"\nSaved to {CUM_CSV_PATH}")
    print(f"Saved to {OSC_CSV_PATH}")
    if n_no_incident:
        print(f"\n{len(n_no_incident)} (seed, driver) pairs never reached a restrictive "
              f"state and were skipped: {n_no_incident}")


if __name__ == "__main__":
    main()
