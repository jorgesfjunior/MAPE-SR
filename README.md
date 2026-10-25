# MAPE-SR: A Self-Protecting MAPE-K Based on Security Standards

Artifact for the paper *MAPE-SR: A Self-Protecting MAPE-K Based on
Security Standards* (J. Soares, N. Rosa, F. Aires), accepted as a short
paper at UCC 2026.

MAPE-SR (MAPE-K for Security Response) is a MAPE-K loop in which each
phase implements requirements of NIST SP 800-61r3. Instead of keeping a
source blocked until its incident is closed, MAPE-SR admits the source's
requests with a probability that grows during recovery.

This repository contains the three agents compared in the paper, the
trained classifier they share, the workload generator, the experiment
harness, the statistical analysis and the raw per-seed results. It
covers the 25-seed x 3-configuration comparison (Baseline /
Adaptive-Generic / MAPE-SR) behind the paper's Results: quiet-phase
throughput, cumulative legitimate-traffic admission, and admission
oscillation during recovery from a DrDoS_NTP-calibrated attack.

## Layout

- **`agent/`** — the three agents under test. Each is a Flask service on
  port 5001 exposing `POST /analyze` (one request in, one decision out)
  and `GET /metrics`.
  - `agent_nist.py` — **MAPE-SR** (6-tier response model, adaptive
    escalation thresholds, graduated probabilistic recovery).
  - `agent_pure_mapek.py` — **Baseline** (standard MAPE-K, fixed
    escalation rule, binary re-admission).
  - `agent_adaptive_generic.py` — **Adaptive-Generic** (Baseline's state
    model and binary re-admission, plus MAPE-SR's threshold adaptation;
    isolates the effect of adaptation from that of graduated recovery).
  - `requirements.txt` — pinned versions used for the paper's runs.

- **`models/`** — the Random Forest classifier trained on CIC-DDoS2019
  and shared by all three agents: `rf_ddos_model.pkl`, `scaler.pkl`,
  `feature_cols.pkl` (the eight online features, in the order the model
  expects) and `threshold.pkl` (the fixed decision threshold τ). The
  agents load it from `../models` relative to their own file.

- **`workload/`** — `generate.py` produces the calibrated attack traffic
  (DrDoS_NTP profile: inter-arrival gap and payload size fit against real
  DrDoS_NTP per-flow feature statistics). Runs standalone with no agent
  required.

- **`harness/`** — `run_admission_metrics_25seed.py` and
  `run_graduated_25seed.py` run the 25-seed x 3-configuration comparison
  and derive the three reported metrics from each run's event log.
  `harness/drivers/` has one driver per configuration
  (`baseline_driver.py`, `adaptive_generic_driver.py`,
  `mapesr_driver.py`); each starts its agent from `agent/` as a
  subprocess and documents, at the top of its protocol section, the HTTP
  contract the harness uses (endpoint, `POST /analyze` request JSON, and
  the response fields each metric depends on).

- **`analysis/`** — reproduces the Friedman / Wilcoxon / Â12 statistics
  reported in the paper directly from the CSVs in `results/`.
  `analyze_admission_metrics_25seed.py` computes cumulative admission and
  admission oscillation; `analyze_graduated_25seed.py` computes
  quiet-phase throughput and time-to-first-ALLOW. Both import shared
  helpers from `analyze_results_3way.py` and `analyze_results.py` (see
  the note at the top of each file).

- **`results/*.csv`** — raw per-seed output the paper's numbers were
  computed from: `graduated_25seed_raw.csv` (quiet-phase throughput),
  `cumulative_admission_25seed.csv`, `oscillation_25seed.csv`.

## The agents

All three agents share the same Monitor and Analyse phases: per-source
history over a 10 s window, eight statistics of forward packet length and
inter-arrival time, and the Random Forest in `models/` with a fixed
threshold τ ≈ 0.835 (`models/threshold.pkl`). A request is labelled
malicious when the attack probability p ≥ τ. With fewer than three
requests in the window, classification is skipped and the request is
only monitored (cold start). Since detection is identical, any difference
between the agents comes from Plan and Execute.

| | Baseline | Adaptive-Generic | MAPE-SR |
|---|---|---|---|
| Classifier, τ | shared, fixed | shared, fixed | shared, fixed |
| Escalation | fixed rule | adaptive θ | adaptive θ |
| Response model | 3 states | 3 states | 6 tiers |
| Re-admission | binary, when evidence reaches 0 | binary (as Baseline) | graduated, probabilistic |

### MAPE-SR parameters (`agent/agent_nist.py`)

**Evidence.** Each source keeps an evidence count e_s (`detections`):
+1 for each request classified as malicious, −1 (floored at 0) for each
request classified as benign.

**Escalation** (on a malicious prediction, first matching rule wins):

| Tier | Condition | Action |
|---|---|---|
| Eradicated | 3 consecutive hard-containment decisions | BLOCK |
| Contained (hard) | e_s ≥ 5 or p ≥ θ_hard (initially 0.9) | BLOCK |
| Contained (soft) | e_s ≥ 2 or p ≥ θ_soft (initially 0.7) | RATE_LIMIT |
| Suspect | otherwise | MONITOR |

**Graduated recovery.** When a contained or eradicated source produces a
benign prediction while e_s > 0, it enters Recovering. Each request is
then admitted (ALLOW) with probability

    a = 1 − e_s / e_peak

where e_peak is the highest evidence reached while contained in the
current incident; otherwise it is only monitored (MONITOR). Since e_s
drops by one per benign prediction, a grows from 0 towards 1 as the
source keeps behaving normally. The draw uses a dedicated NumPy generator
seeded from the `ADMISSION_SEED` environment variable (set by the
harness from the run's seed), so runs are reproducible. The source is
released (Normal, ALLOW) when e_s reaches 0, which closes the incident.

**Threshold adaptation** (second loop; affects escalation only, not τ or
recovery). Every 10 requests from a source, using precision and recall
computed from ground-truth labels:

| Condition | Change to θ_soft and θ_hard |
|---|---|
| only recall < 0.6 | −0.05 |
| only precision < 0.6 | +0.05 |
| both < 0.6 | −0.025 |
| both ≥ 0.6, learning gain > 0.5 | +0.025 |
| both ≥ 0.6, learning gain < 0 | −0.025 |

Thresholds are clipped to [0.5, 0.99]. The learning gain is
(F1 − 0.3) / 0.3, computed after at least five requests. Every change is
recorded with the metrics that triggered it.

### Baseline and Adaptive-Generic

Both use the same evidence count and a 3-state model (Normal, Suspect,
Blocked): BLOCK when e_s ≥ 5, RATE_LIMIT when e_s ≥ 2, MONITOR
otherwise, and ALLOW only once e_s is back to 0 (binary re-admission).
Adaptive-Generic adds the OR-conditions on p and the same threshold
adaptation as MAPE-SR (`p ≥ θ_hard` for BLOCK, `p ≥ θ_soft` for
RATE_LIMIT).

### Execute

The agents' executor posts each decision to a managed-element endpoint
(`http://app:5000/execute`) when one is reachable and ignores the call
otherwise; the managed element is not needed to reproduce the results,
since the harness reads the decisions directly from the `/analyze`
responses.

## Setup

Python 3.12+ is required. To reproduce the paper's Wilcoxon p-values
exactly, use `scipy>=1.18` (scipy changed its default computation method
for small paired samples between versions; older scipy, e.g. 1.13 on
Python 3.9, reproduces the same means, Friedman significance and Â12
effect sizes, but slightly different Wilcoxon p-values).

```
pip install -r requirements.txt               # analysis + harness
pip install -r agent/requirements.txt         # agents (needed only to rerun the experiments)
```

## Reproducing the paper's numbers

From the included CSVs (no agent needed):

```
python analysis/analyze_graduated_25seed.py
python analysis/analyze_admission_metrics_25seed.py
```

This reproduces:

| metric | Baseline | Adaptive-Generic | MAPE-SR |
|---|---:|---:|---:|
| quiet-phase throughput | 0.305 | 0.305 | 0.620 |
| cumulative admission | 0.028 | 0.028 | 0.469 |
| admission oscillation (mean transitions/incident) | 0 | 0 | 10.12 |

## Running the experiments end-to-end

The harness starts each agent from `agent/` on `localhost:5001`, sends
the workload for each of the 25 seeds, and writes the per-seed CSVs to
`results/` (overwriting the included ones; copy them first if you want
to compare):

```
python harness/run_admission_metrics_25seed.py   # cumulative admission, oscillation
python harness/run_graduated_25seed.py           # quiet-phase throughput
```

Then run the two analysis scripts above. Port 5001 must be free.

To exercise a single agent with one seed:

```
python harness/drivers/mapesr_driver.py --seed 1
python harness/drivers/baseline_driver.py --seed 1
python harness/drivers/adaptive_generic_driver.py --seed 1
```

To run an agent on its own (e.g. to send it requests manually):

```
cd agent && python agent_nist.py     # listens on 0.0.0.0:5001
```

Without `ADMISSION_SEED` set, the recovery draws are seeded from OS
entropy and are not reproducible.
