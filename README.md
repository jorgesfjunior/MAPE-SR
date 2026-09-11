# MAPE-SR evaluation artifact

This repository documents the experimental methodology and statistical
analysis pipeline used in the paper's evaluation. Agent implementations
are not included in this release.

It covers the 25-seed x 3-configuration admission comparison (Baseline /
Adaptive-Generic / MAPE-SR) behind Section IV's Results: quiet-phase
throughput, cumulative legitimate-traffic admission, and admission
oscillation during recovery from a DrDoS_NTP-calibrated attack.

## Layout

- **`workload/`** — `generate.py` produces the calibrated attack traffic
  (DrDoS_NTP profile: inter-arrival gap and payload size fit against real
  DrDoS_NTP per-flow feature statistics) used to build every run reported
  in Section IV. Runs standalone with no agent required.

- **`harness/`** — `run_admission_metrics_25seed.py` and
  `run_graduated_25seed.py` orchestrate the 25-seed x 3-configuration
  comparison and derive the three reported metrics from each run's event
  log. `harness/drivers/` contains one small driver module per
  configuration (`baseline_driver.py`, `adaptive_generic_driver.py`,
  `mapesr_driver.py`); each documents, in a comment block at the top of
  its protocol section, the exact contract the harness uses to drive an
  agent under test:
  - HTTP endpoint and port (`http://localhost:5001`)
  - the exact `POST /analyze` request JSON
  - the exact response JSON fields each metric depends on

  This lets a reviewer follow the experimental protocol precisely without
  the agent's source. The agent implementations themselves are not part
  of this release.

- **`analysis/`** — reproduces the Friedman / Wilcoxon / Â12 statistics
  reported in Section IV directly from the CSVs in `results/`.
  `analyze_admission_metrics_25seed.py` computes cumulative admission and
  admission oscillation; `analyze_graduated_25seed.py` computes
  quiet-phase throughput and time-to-first-ALLOW. Both import their
  Friedman/Wilcoxon/Â12 machinery from `analyze_results_3way.py` (which
  in turn imports paired-CI/A12/Wilcoxon helpers from
  `analyze_results.py`) — shared library code, reused rather than
  reimplemented, included here for that reason even though the broader
  comparisons their own `main()` functions target are outside this
  release's scope (see the note at the top of each file).

- **`results/*.csv`** — raw per-seed output the paper's numbers were
  computed from: `graduated_25seed_raw.csv` (quiet-phase throughput),
  `cumulative_admission_25seed.csv`, `oscillation_25seed.csv`. Included
  so the statistical pipeline can be verified independently of the
  harness or any agent.

## Setup

Python 3.12+ and `scipy>=1.18` are required to reproduce the paper's
Wilcoxon p-values exactly (scipy changed its default computation method
for small paired samples between versions; older scipy, e.g. 1.13 on
Python 3.9, reproduces the same means, Friedman significance, and Â12
effect sizes, but slightly different Wilcoxon p-values — still
significant, same order of magnitude, not the values printed in the
paper).

```
pip install -r requirements.txt
```

## Reproducing the paper's numbers

Run the workload generator standalone (no agent needed):

```
python workload/generate.py --seed 1
python workload/generate.py --seed 2
```

Reproduce the reported statistics from the included CSVs:

```
python analysis/analyze_graduated_25seed.py
python analysis/analyze_admission_metrics_25seed.py
```

This reproduces, from `results/graduated_25seed_raw.csv`,
`results/cumulative_admission_25seed.csv`, and
`results/oscillation_25seed.csv`:

| metric | Baseline | Adaptive-Generic | MAPE-SR |
|---|---:|---:|---:|
| quiet-phase throughput | 0.305 | 0.305 | 0.620 |
| cumulative admission | 0.028 | 0.028 | 0.469 |
| admission oscillation (mean transitions/incident) | 0 | 0 | 10.12 |

## Running the harness end-to-end

`harness/run_admission_metrics_25seed.py` and
`harness/run_graduated_25seed.py` require a running agent implementation
at `agent/<script>.py` for each of the three configurations (not included
in this repository) — see `harness/drivers/*.py` for the exact HTTP
protocol each expects. Without an agent present, these two scripts are
syntactically complete and document the full protocol, but cannot be
run to completion; the CSVs in `results/` are the outputs already
produced by that harness against the paper's own agent implementations.
