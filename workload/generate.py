"""
generate.py

Standalone DrDoS_NTP-calibrated workload generator for the 25-seed x
3-configuration admission comparison (see harness/). Produces the exact
event sequence sent to each agent under test in a run: 20 baseline
(benign) events, then 30 attack (DrDoS_NTP-calibrated) events, then
`quiet_n` quiet-closure probe events (default 50 -- the established safe
margin for every incident to reach ALLOW again within budget across all
three configurations under test).

Calibration: the attack phase's inter-arrival gap (1e-6, 1e-4]s and
payload size ~N(440, 30) are fit against DrDoS_NTP's real per-flow
feature statistics (median Fwd IAT Mean = 27.73us, mean payload =
440B with nonzero per-flow std = 25.54 -- not a constant), not the
classifier's own output. This is the same calibration used, byte-
identical, by both harness scripts in this repository.

Each event is a 5-tuple: (phase, ip, timestamp, payload_size, ground_truth).
"""
import random

ATTACK_IP = "6.6.6.1"
BENIGN_POOL = [f"10.0.{b}.{h}" for b in range(3) for h in range(1, 12)]

QUIET_N_DEFAULT = 50


def set_seed(seed):
    """Seeds the module-level `random` RNG used by build_workload() so
    that IP selection, attack/benign assignment, and inter-arrival timing
    are fully reproducible for a given seed."""
    random.seed(seed)


def build_workload(quiet_n=QUIET_N_DEFAULT):
    events = []
    t = 1_700_000_000.0

    def benign(phase):
        nonlocal t
        t += random.uniform(0.5, 2.0)
        return (phase, random.choice(BENIGN_POOL), t, random.randint(200, 800), "Benign")

    def ddos(phase):
        nonlocal t
        # DrDoS_NTP calibration -- see module docstring.
        t += random.uniform(1e-6, 1e-4)
        payload = max(1, round(random.gauss(440, 30)))
        return (phase, ATTACK_IP, t, payload, "DDoS")

    def quiet_probe(phase):
        nonlocal t
        t += random.uniform(1.5, 3.0)
        return (phase, ATTACK_IP, t, 400, "Benign")

    for _ in range(20):
        events.append(benign("baseline"))
    for _ in range(30):
        events.append(ddos("attack"))
    for _ in range(quiet_n):
        events.append(quiet_probe("quiet_closure"))

    return events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the calibrated workload for a given seed and print summary stats."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet-n", type=int, default=QUIET_N_DEFAULT)
    args = parser.parse_args()

    set_seed(args.seed)
    workload = build_workload(args.quiet_n)

    n_baseline = sum(1 for e in workload if e[0] == "baseline")
    n_attack = sum(1 for e in workload if e[0] == "attack")
    n_quiet = sum(1 for e in workload if e[0] == "quiet_closure")

    attack_events = [e for e in workload if e[0] == "attack"]
    attack_payloads = [e[3] for e in attack_events]
    attack_gaps = []
    prev_t = None
    for e in attack_events:
        if prev_t is not None:
            attack_gaps.append(e[2] - prev_t)
        prev_t = e[2]

    print(f"seed={args.seed}  total_events={len(workload)}  "
          f"baseline={n_baseline}  attack={n_attack}  quiet_closure={n_quiet}")
    print(f"attack payload: mean={sum(attack_payloads)/len(attack_payloads):.2f}  "
          f"min={min(attack_payloads)}  max={max(attack_payloads)}  "
          f"(calibration target: N(440, 30))")
    print(f"attack inter-arrival gap: mean={sum(attack_gaps)/len(attack_gaps):.2e}s  "
          f"min={min(attack_gaps):.2e}s  max={max(attack_gaps):.2e}s  "
          f"(calibration target: uniform(1e-6, 1e-4)s)")
    print(f"first 3 events: {workload[:3]}")
    print(f"last 3 events:  {workload[-3:]}")
