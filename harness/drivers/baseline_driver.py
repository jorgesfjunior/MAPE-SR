"""
baseline_driver.py

Minimal driver interface the harness/ scripts use to exercise the
"Baseline" (pure MAPE-K, fixed-threshold) agent under test. The agent
itself is agent/agent_pure_mapek.py; this file documents and implements
the protocol used to talk to it.

Run standalone: python harness/drivers/baseline_driver.py --seed 1
(spawns AGENT_SCRIPT as a subprocess from AGENT_DIR, i.e. agent/.)
"""
import os
import random
import subprocess
import sys
import time

import requests

AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agent")
AGENT_SCRIPT = "agent_pure_mapek.py"  # agent/agent_pure_mapek.py
AGENT_URL = "http://localhost:5001"

DEFAULT_SEED = 42
RANDOM_SEED = DEFAULT_SEED


def set_seed(seed):
    """Seeds this driver's RNG (unused directly -- the workload's own RNG,
    seeded separately via workload.generate.set_seed(), is what actually
    determines event content). Kept for interface parity with the other
    two drivers, one of which (mapesr_driver.py) also needs an env var
    set here before the agent subprocess is spawned."""
    global RANDOM_SEED
    RANDOM_SEED = seed
    random.seed(seed)


# ============================================================
# AGENT PROTOCOL
#
# This driver expects an HTTP agent process (agent/agent_pure_mapek.py)
# reachable at AGENT_URL = "http://localhost:5001":
#
#   GET  /metrics
#     Polled by wait_ready() until it responds (any 200), to detect that
#     the agent subprocess has finished starting up. Response body not
#     read by this driver.
#
#   POST /analyze
#     Request JSON (one per traffic event):
#       {
#         "ip": "<source IP string>",
#         "timestamp": <float, unix seconds>,
#         "payload_size": <int, bytes>,
#         "label": "Benign" | "DDoS",     # ground truth for this event
#         "phase": "<workload phase string, e.g. 'attack'>"
#       }
#
#     Response JSON, fields this driver/harness reads:
#       "label" OR "predicted": "Benign" | "DDoS"   # classifier's verdict
#       "action": "ALLOW" | "MONITOR" | "RATE_LIMIT" | "BLOCK"
#
#     Baseline's /analyze response has NO "state" field (unlike the other
#     two agents under test) -- its BLOCK/ALLOW/MONITOR/RATE_LIMIT action
#     sequence alone is enough to replay its 4-state machine exactly; see
#     harness/run_admission_metrics_25seed.py's
#     _reconstruct_pure_mapek_states().
# ============================================================

def start_agent():
    proc = subprocess.Popen(
        [sys.executable, AGENT_SCRIPT],
        cwd=AGENT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_ready(retries=40, delay=0.5):
    for _ in range(retries):
        try:
            requests.get(f"{AGENT_URL}/metrics", timeout=1)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError("Baseline agent did not start in time")


def stop_agent(proc):
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def send(phase, ip, timestamp, payload_size, ground_truth):
    try:
        r = requests.post(
            f"{AGENT_URL}/analyze",
            json={"ip": ip, "timestamp": timestamp,
                  "payload_size": payload_size, "label": ground_truth,
                  "phase": phase},
            timeout=5,
        )
        data = r.json()
        data["predicted"] = data.get("label", data.get("predicted"))
        data["_phase"] = phase
        data["_ip"] = ip
        data["_timestamp"] = timestamp
        data["_ground_truth"] = ground_truth
        return data
    except Exception as e:
        return {"_phase": phase, "_ip": ip, "_timestamp": timestamp,
                "_ground_truth": ground_truth, "_error": str(e)}


if __name__ == "__main__":
    import argparse
    import sys as _sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from workload.generate import ATTACK_IP, BENIGN_POOL, build_workload
    from workload.generate import set_seed as set_workload_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    set_workload_seed(args.seed)
    proc = start_agent()
    try:
        wait_ready()
        workload = build_workload()
        logs = [send(*e) for e in workload]
        print(f"sent {len(logs)} events; last action={logs[-1].get('action')}")
    finally:
        stop_agent(proc)
