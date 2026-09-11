"""
mapesr_driver.py

Minimal driver interface the harness/ scripts use to exercise the
"MAPE-SR" agent under test (the full graduated-response MAPE-K agent:
CONTAINED/ERADICATED escalation tiers plus a graduated RECOVERING
admission phase). The agent's own implementation is NOT included in
this repository -- this file only documents and implements the protocol
used to talk to it.

Run standalone: python harness/drivers/mapesr_driver.py --seed 1
(spawns AGENT_SCRIPT as a subprocess from AGENT_DIR -- only works if you
place a compatible agent implementation at that path yourself.)
"""
import os
import random
import subprocess
import sys
import time

import requests

AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agent")
AGENT_SCRIPT = "agent_nist.py"  # not included in this repository
AGENT_URL = "http://localhost:5001"

DEFAULT_SEED = 42
RANDOM_SEED = DEFAULT_SEED


def set_seed(seed):
    """Seeds this driver's RNG AND, via the ADMISSION_SEED env var, the
    agent subprocess's own in-process randomness (its RECOVERING-phase
    admission-decision coin flip is real randomness on the agent side,
    seeded once at agent module import time -- so this must run before
    start_agent() spawns the subprocess)."""
    global RANDOM_SEED
    RANDOM_SEED = seed
    random.seed(seed)
    os.environ["ADMISSION_SEED"] = str(seed)


# ============================================================
# AGENT PROTOCOL
#
#   GET  /metrics   -> polled by wait_ready() until it responds (any 200)
#
#   POST /analyze   -> request:  {"ip", "timestamp", "payload_size",
#                                  "label", "phase"}   (same shape as the
#                                  other two agents under test)
#                      response fields read downstream:
#                        "predicted": "Benign" | "DDoS"
#                        "action": "ALLOW" | "MONITOR" | "RATE_LIMIT" | "BLOCK"
#                        "state": "NORMAL" | "SUSPECT" | "CONTAINED" |
#                                 "ERADICATED" | "RECOVERING"
#                        "admission_rate": float in [0, 1], present only
#                          while state == "RECOVERING" -- the probability
#                          this agent used for that event's own admission
#                          coin flip (graduated recovery: rises from 0
#                          toward 1 as the incident's RECOVERING dwell
#                          time increases). Absent/null in every other
#                          state; not produced by either other agent.
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
    raise RuntimeError("MAPE-SR agent did not start in time")


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

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from workload.generate import build_workload
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
