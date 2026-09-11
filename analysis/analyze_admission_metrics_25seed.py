"""
analyze_admission_metrics_25seed.py

Statistical analysis of the two files produced by
harness/run_admission_metrics_25seed.py's 25-seed x 3-driver run
(Baseline / Adaptive-Generic / MAPE-SR), over the quiet-period recovery
workload from workload/generate.py (20 baseline + 30 attack + 50
quiet-closure probes).

Metric 1 -- cumulative_admission_fraction (results/cumulative_admission_25seed.csv):
    Fraction of ground-truth-benign requests receiving ALLOW across the
    full incident lifecycle: from the first event the incident's IP
    reaches a "restrictive" state (CONTAINED/ERADICATED for MAPE-SR;
    BLOCKED -- their single strongest tier -- for Baseline/Adaptive-
    Generic, no CONTAINED/ERADICATED distinction exists there) through
    closure (state returns to NORMAL), inclusive of the closing event.
    Same statistical pipeline as analysis/analyze_graduated_25seed.py:
    Friedman omnibus across the 3 paired (by-seed) groups, then (only if
    significant) pairwise Wilcoxon signed-rank + Vargha-Delaney A12,
    Holm-Bonferroni corrected within that follow-up family. Reuses
    _holm_bonferroni/_friedman_safe/_descriptive/_pairwise/SCRIPT_*/
    LABELS/ALPHA directly from analyze_results_3way.py -- not a
    reimplementation.

Metric 2 -- n_allow_monitor_transitions (results/oscillation_25seed.csv):
    Count of ALLOW<->MONITOR action transitions between consecutive
    events while the incident's IP sits in the "restricted phase"
    (RECOVERING for MAPE-SR; BLOCKED for Baseline/Adaptive-Generic).
    Reported descriptively only (mean transitions per incident per
    agent), not run through the Friedman pipeline -- the point of this
    metric is the near-certain zero/nonzero split (Baseline/Adaptive-
    Generic's binary MONITOR-until-clear mechanics can't produce an
    ALLOW while BLOCKED: their planner only ever emits ALLOW in the
    same step it flips the state to NORMAL -- see
    harness/run_admission_metrics_25seed.py's RESTRICTED_PHASE_STATE
    comment), not a magnitude comparison.

Run: python analysis/analyze_admission_metrics_25seed.py
"""
import csv
import os

import numpy as np

from analyze_results_3way import (
    SCRIPT_BASELINE, SCRIPT_AG, SCRIPT_MAPESR, SCRIPTS, LABELS, PAIRS,
    ALPHA, _holm_bonferroni, _friedman_safe, _descriptive, _pairwise,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUM_CSV_PATH = os.path.join(REPO_ROOT, "results", "cumulative_admission_25seed.csv")
OSC_CSV_PATH = os.path.join(REPO_ROOT, "results", "oscillation_25seed.csv")
CUM_SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "cumulative_admission_25seed_statistical_summary.md")
OSC_SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "oscillation_25seed_summary.md")

METRIC = "cumulative_admission_fraction"


def _load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v):
    if v in (None, ""):
        return None
    return float(v)


def _to_bool(v):
    return str(v).strip().lower() == "true"


def _grouped_by_seed(rows, metric, scripts=SCRIPTS):
    """Same contract as analyze_results_3way.py's _grouped_by_seed: only
    seeds with a non-blank value for `metric` in EVERY script enter the
    paired sample (a blank means that (seed, script)'s incident never
    reached a restrictive state at all -- see harness/run_admission_
    metrics_25seed.py's n_no_incident warning; not expected in practice,
    but excluded rather than imputed if it happens)."""
    by_seed = {}
    for row in rows:
        seed = int(row["seed"])
        by_seed.setdefault(seed, {})[row["script"]] = row

    seeds = []
    vals = {s: [] for s in scripts}
    for seed, pair in sorted(by_seed.items()):
        if not all(s in pair for s in scripts):
            continue
        vs = {s: _to_float(pair[s].get(metric)) for s in scripts}
        if any(v is None for v in vs.values()):
            continue
        seeds.append(seed)
        for s in scripts:
            vals[s].append(vs[s])
    return seeds, {s: np.array(vals[s]) for s in scripts}


def analyze_cumulative_admission(rows):
    n_seeds_total = len(set(int(r["seed"]) for r in rows))
    seeds, vals = _grouped_by_seed(rows, METRIC)

    info = {"metric": METRIC, "seeds": seeds, "n_seeds_total": n_seeds_total}
    if not seeds:
        info["applicable"] = False
        return info

    info["applicable"] = True
    info["vals"] = vals
    info["desc"] = {s: _descriptive(vals[s]) for s in SCRIPTS}

    pvalue, note = _friedman_safe(vals, scripts=SCRIPTS)
    info["friedman_p"] = pvalue
    info["friedman_note"] = note
    info["significant"] = False

    if pvalue is not None:
        info["friedman_p_holm"] = pvalue  # single metric in this family -- Holm on 1 p-value is a no-op
        info["significant"] = pvalue < ALPHA

    if info["significant"]:
        pair_results = {}
        raw_pair_ps = []
        for (s1, s2) in PAIRS:
            r = _pairwise(vals[s1], vals[s2])
            pair_results[(s1, s2)] = r
            raw_pair_ps.append(r["pvalue"] if r["pvalue"] is not None else 1.0)
        adj_pair_ps = _holm_bonferroni(raw_pair_ps)
        for (pair, adj) in zip(PAIRS, adj_pair_ps):
            pair_results[pair]["pvalue_holm"] = adj
            pair_results[pair]["significant"] = adj < ALPHA
        info["pairwise"] = pair_results

    return info


def analyze_oscillation(rows):
    n_seeds_total = len(set(int(r["seed"]) for r in rows))
    per_script = {}
    for s in SCRIPTS:
        vals = np.array([float(r["n_allow_monitor_transitions"]) for r in rows if r["script"] == s])
        rp_lens = np.array([float(r["restricted_phase_len"]) for r in rows if r["script"] == s])
        n_nonzero = int(np.sum(vals > 0))
        per_script[s] = {
            "n": len(vals),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "max": float(vals.max()) if len(vals) else float("nan"),
            "n_nonzero": n_nonzero,
            "always_zero": n_nonzero == 0,
            "mean_restricted_phase_len": float(rp_lens.mean()) if len(rp_lens) else float("nan"),
        }
    return {"n_seeds_total": n_seeds_total, "per_script": per_script}


def fmt_p(p):
    return "n/a" if p is None else f"{p:.3g}"


def main():
    cum_rows = _load_rows(CUM_CSV_PATH)
    osc_rows = _load_rows(OSC_CSV_PATH)

    # ---------------- Metric 1 ----------------
    cum_info = analyze_cumulative_admission(cum_rows)

    lines = []
    lines.append("# Cumulative legitimate-traffic admission, 25-seed statistical analysis\n")
    lines.append(f"Source: `results/cumulative_admission_25seed.csv`, "
                 f"{cum_info['n_seeds_total']} seeds (seeds 1-25). Same workload as "
                 f"`results/graduated_25seed_raw.csv`: 20 baseline + 30 attack "
                 f"(DrDoS_NTP-calibrated) + 50 quiet-closure probes.\n")
    lines.append("Metric: fraction of ground-truth-benign requests to the incident's IP "
                 "that received ALLOW, across the full incident lifecycle (first "
                 "restrictive-state event through closure, inclusive). Restrictive state = "
                 "CONTAINED/ERADICATED for MAPE-SR, BLOCKED for Baseline/Adaptive-Generic.\n")

    if not cum_info["applicable"]:
        lines.append("Not applicable: no seed had a value for all 3 scripts.\n")
    else:
        n_pairs = len(cum_info["seeds"])
        lines.append(f"n paired seeds = {n_pairs} "
                     f"(excluded: {sorted(set(range(1, cum_info['n_seeds_total']+1)) - set(cum_info['seeds']))})\n")
        lines.append("| script | n | mean | std | 95% CI |")
        lines.append("|---|---:|---:|---:|---|")
        for s in SCRIPTS:
            d = cum_info["desc"][s]
            lines.append(f"| {LABELS[s]} | {d['n']} | {d['mean']:.6g} | {d['std']:.6g} | "
                         f"[{d['ci_lo']:.6g}, {d['ci_hi']:.6g}] |")
        lines.append("")
        lines.append(f"Friedman p = {fmt_p(cum_info['friedman_p'])}, "
                     f"significant @ alpha=0.05: {cum_info['significant']}"
                     + (f"  ({cum_info['friedman_note']})" if cum_info.get("friedman_note") else "")
                     + "\n")

        if cum_info["significant"]:
            lines.append("| pair | mean diff | 95% CI | Wilcoxon p | Holm p | sig | "
                         "A12 | magnitude |")
            lines.append("|---|---:|---|---:|---:|---|---:|---|")
            for (s1, s2) in PAIRS:
                r = cum_info["pairwise"][(s1, s2)]
                lines.append(f"| {LABELS[s1]} vs {LABELS[s2]} | {r['mean_diff']:.6g} | "
                             f"[{r['ci_lo']:.6g}, {r['ci_hi']:.6g}] | {fmt_p(r['pvalue'])} | "
                             f"{fmt_p(r['pvalue_holm'])} | {r['significant']} | "
                             f"{r['a12']:.4f} | {r['a12_magnitude']} |")
            lines.append("")

    cum_report = "\n".join(lines)
    with open(CUM_SUMMARY_PATH, "w") as f:
        f.write(cum_report)
    print(cum_report)
    print(f"\nSaved to {CUM_SUMMARY_PATH}\n")

    # ---------------- Metric 2 ----------------
    osc_info = analyze_oscillation(osc_rows)

    lines2 = []
    lines2.append("# Admission oscillation (ALLOW<->MONITOR transitions), 25-seed descriptive analysis\n")
    lines2.append(f"Source: `results/oscillation_25seed.csv`, {osc_info['n_seeds_total']} seeds "
                  f"(seeds 1-25). Restricted phase = RECOVERING for MAPE-SR, BLOCKED for "
                  f"Baseline/Adaptive-Generic. Descriptive only (not Friedman-tested): the "
                  f"question is whether Baseline/Adaptive-Generic are always exactly 0.\n")
    lines2.append("| script | n | mean transitions/incident | std | max | incidents with >0 | "
                  "always 0? | mean restricted-phase length |")
    lines2.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for s in SCRIPTS:
        d = osc_info["per_script"][s]
        lines2.append(f"| {LABELS[s]} | {d['n']} | {d['mean']:.4g} | {d['std']:.4g} | "
                      f"{d['max']:.0f} | {d['n_nonzero']} | {d['always_zero']} | "
                      f"{d['mean_restricted_phase_len']:.4g} |")
    lines2.append("")
    for s in (SCRIPT_BASELINE, SCRIPT_AG):
        d = osc_info["per_script"][s]
        verdict = "CONFIRMED" if d["always_zero"] else "NOT confirmed"
        lines2.append(f"{LABELS[s]}: {verdict} -- 0 transitions on {d['n'] - d['n_nonzero']}/{d['n']} incidents.")
    lines2.append("")

    osc_report = "\n".join(lines2)
    with open(OSC_SUMMARY_PATH, "w") as f:
        f.write(osc_report)
    print(osc_report)
    print(f"\nSaved to {OSC_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
