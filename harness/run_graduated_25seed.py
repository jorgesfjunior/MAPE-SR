"""
run_graduated_25seed.py

25-seed x 3-configuration (Baseline / Adaptive-Generic / MAPE-SR)
comparison on the same quiet-period recovery workload as
run_admission_metrics_25seed.py (workload/generate.py: 20 baseline + 30
attack, DrDoS_NTP-calibrated, + 50 quiet-closure probes).

For each (seed, script) pair, records the two things this comparison is
actually about:
  - quiet_throughput: fraction of quiet-phase requests that got
    ALLOW-equivalent treatment (action == "ALLOW")
  - first_allow_event: 1-indexed position within the quiet phase of the
    first ALLOW action (blank if none within the 50-probe budget -- see
    n_quiet_events / closed_within_budget)

MAPE-SR additionally logs mean_admission_rate (mean of the per-event
admission_rate field across the quiet phase -- see
harness/drivers/mapesr_driver.py's protocol comment); blank for
Baseline/Adaptive-Generic, which have no such field.

Output CSV compatibility note: the "script" id for the MAPE-SR row is
written as "test_nist_graduated" (not "test_nist_only", as used by
run_admission_metrics_25seed.py) to stay consistent with
results/graduated_25seed_raw.csv (already included in this repository)
and with analysis/analyze_graduated_25seed.py, both of which predate a
later consolidation on the agent side that folded the graduated-recovery
behavior into the single "MAPE-SR" agent used everywhere else in this
repo (harness/drivers/mapesr_driver.py, AGENT_SCRIPT = "agent_nist.py").
There is no separate graduated-only agent variant anymore -- this
script's MAPE-SR row is driven by the exact same mapesr_driver.py /
agent as run_admission_metrics_25seed.py's, just re-run over this
script's own two metrics and written under the historical script id so
a fresh run stays comparable with the included CSV and its analysis
script without either needing to change.

Run: python harness/run_graduated_25seed.py
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

# See module docstring for why the MAPE-SR row keeps the historical
# "test_nist_graduated" script id despite being driven by mapesr_driver.py.
DRIVERS = [
    (baseline_driver, "test_pure_mapek", "Baseline"),
    (adaptive_generic_driver, "test_adaptive_generic", "Adaptive-Generic"),
    (mapesr_driver, "test_nist_graduated", "MAPE-SR (graduated)"),
]

QUIET_N = 50  # established safe margin -- see workload/generate.py
SEEDS = list(range(1, 26))

CSV_PATH = os.path.join(REPO_ROOT, "results", "graduated_25seed_raw.csv")
CSV_FIELDS = [
    "seed", "script", "label", "n_quiet_events", "n_allow",
    "quiet_throughput", "closed_within_budget", "first_allow_event",
    "mean_admission_rate",
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


def analyze(logs):
    atk_events = [e for e in logs if e.get("_ip") == ATTACK_IP]
    quiet_start_idx = next(i for i, e in enumerate(atk_events) if e.get("_phase") == "quiet_closure")
    quiet_events = atk_events[quiet_start_idx:]

    n_allow = sum(1 for e in quiet_events if e.get("action") == "ALLOW")
    first_allow = None
    for i, e in enumerate(quiet_events):
        if e.get("action") == "ALLOW":
            first_allow = i + 1
            break

    admit_rates = [e.get("admission_rate") for e in quiet_events if e.get("admission_rate") is not None]
    mean_admit = sum(admit_rates) / len(admit_rates) if admit_rates else None

    return {
        "n_quiet_events": len(quiet_events),
        "n_allow": n_allow,
        "quiet_throughput": n_allow / len(quiet_events),
        "closed_within_budget": first_allow is not None,
        "first_allow_event": first_allow if first_allow is not None else "",
        "mean_admission_rate": round(mean_admit, 6) if mean_admit is not None else "",
    }


def _append_row(row):
    write_header = not (os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if os.path.isfile(CSV_PATH):
        os.remove(CSV_PATH)  # fresh run, not meant to be resumed/appended across edits

    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        for driver_module, driver_name, label in DRIVERS:
            logs = run_one(driver_module, seed, QUIET_N)
            res = analyze(logs)
            row = {"seed": seed, "script": driver_name, "label": label, **res}
            _append_row(row)
            print(f"[{label:>20}] seed={seed}: throughput={res['quiet_throughput']:.3f} "
                  f"({res['n_allow']}/{res['n_quiet_events']})  "
                  f"first_allow={res['first_allow_event']}  "
                  f"mean_admission_rate={res['mean_admission_rate']}")

    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()
