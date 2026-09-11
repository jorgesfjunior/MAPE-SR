"""
analyze_graduated_25seed.py

Statistical analysis of results/graduated_25seed_raw.csv (the 25-seed x
3-driver run produced by harness/run_graduated_25seed.py):
quiet_throughput (fraction of quiet-phase requests admitted -- action ==
"ALLOW") and first_allow_event (1-indexed position of the first such
admission), Baseline vs Adaptive-Generic vs MAPE-SR.

Same statistical pipeline as analysis/analyze_admission_metrics_25seed.py:
Friedman omnibus across the 3 paired (by-seed) groups, then (only if
significant) pairwise Wilcoxon signed-rank + Vargha-Delaney A12 for the
three pairs, Holm-Bonferroni corrected within that follow-up family.
Reuses _holm_bonferroni/_friedman_safe/_descriptive/_pairwise/ALPHA
directly from analyze_results_3way.py (same functions, not a
reimplementation) so the methodology is provably identical -- only
SCRIPTS/LABELS/PAIRS differ, since the MAPE-SR row in
results/graduated_25seed_raw.csv is keyed under the historical script id
"test_nist_graduated" (see harness/run_graduated_25seed.py's module
docstring for why) rather than "test_nist_only" as used everywhere else
in this repository.

Run: python analysis/analyze_graduated_25seed.py
"""
import csv
import os

import numpy as np

from analyze_results_3way import (
    SCRIPT_BASELINE, SCRIPT_AG,
    ALPHA, _holm_bonferroni, _friedman_safe, _descriptive, _pairwise,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "results", "graduated_25seed_raw.csv")
SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "graduated_25seed_statistical_summary.md")

SCRIPT_MAPESR_GRAD = "test_nist_graduated"
SCRIPTS = [SCRIPT_BASELINE, SCRIPT_AG, SCRIPT_MAPESR_GRAD]
LABELS = {
    SCRIPT_BASELINE: "Baseline",
    SCRIPT_AG: "Adaptive-Generic",
    SCRIPT_MAPESR_GRAD: "MAPE-SR (graduated)",
}
PAIRS = [
    (SCRIPT_BASELINE, SCRIPT_AG),
    (SCRIPT_AG, SCRIPT_MAPESR_GRAD),
    (SCRIPT_BASELINE, SCRIPT_MAPESR_GRAD),
]

METRICS = ["quiet_throughput", "first_allow_event"]


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
    paired sample. A blank first_allow_event means that (seed, script)
    never got an ALLOW within the 50-probe quiet-phase budget (see
    closed_within_budget in the raw CSV) -- excluded here, never imputed."""
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


def analyze_metric_family(rows, metrics):
    per_metric = {}
    for metric in metrics:
        seeds, vals = _grouped_by_seed(rows, metric)
        if not seeds:
            per_metric[metric] = {"metric": metric, "seeds": [], "applicable": False}
            continue
        pvalue, note = _friedman_safe(vals, scripts=SCRIPTS)
        per_metric[metric] = {
            "metric": metric, "seeds": seeds, "vals": vals, "applicable": pvalue is not None,
            "friedman_p": pvalue, "friedman_note": note,
            "desc": {s: _descriptive(vals[s]) for s in SCRIPTS},
        }

    testable = [m for m in metrics if per_metric[m]["applicable"]]
    raw_ps = [per_metric[m]["friedman_p"] for m in testable]
    adj_ps = _holm_bonferroni(raw_ps) if raw_ps else []
    for m, adj in zip(testable, adj_ps):
        per_metric[m]["friedman_p_holm"] = adj
        per_metric[m]["significant"] = adj < ALPHA

    for m in testable:
        info = per_metric[m]
        if not info["significant"]:
            continue
        vals = info["vals"]
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

    return per_metric


def closure_coverage_summary(rows):
    per_script = {s: {"closed": 0, "not_closed": 0, "not_closed_seeds": []} for s in SCRIPTS}
    for row in rows:
        s = row["script"]
        if s not in per_script:
            continue
        seed = int(row["seed"])
        if _to_bool(row["closed_within_budget"]):
            per_script[s]["closed"] += 1
        else:
            per_script[s]["not_closed"] += 1
            per_script[s]["not_closed_seeds"].append(seed)
    return per_script


def fmt_p(p):
    return "n/a" if p is None else f"{p:.3g}"


def main():
    rows = _load_rows(CSV_PATH)
    n_seeds_total = len(set(int(r["seed"]) for r in rows))

    cov = closure_coverage_summary(rows)
    fam = analyze_metric_family(rows, METRICS)

    lines = []
    lines.append("# Graduated MAPE-SR 25-seed statistical analysis: "
                  "quiet-phase throughput and time-to-first-ALLOW\n")
    lines.append(f"Source: `results/graduated_25seed_raw.csv`, {n_seeds_total} seeds "
                 f"(seeds 1-25). Workload: 20 baseline + 30 attack (DrDoS_NTP-calibrated) "
                 f"+ 50 quiet-closure probes -- workload/generate.py.\n")

    lines.append("## closed_within_budget coverage (before any pairing)\n")
    lines.append("| script | closed (n) | not closed (n) | not-closed seeds |")
    lines.append("|---|---:|---:|---|")
    for s in SCRIPTS:
        d = cov[s]
        lines.append(f"| {LABELS[s]} | {d['closed']} | {d['not_closed']} | "
                     f"{', '.join(map(str, d['not_closed_seeds'])) or '-'} |")
    lines.append("")

    for metric in METRICS:
        info = fam[metric]
        lines.append(f"## {metric}\n")
        if not info["applicable"]:
            lines.append(f"Not applicable: {info.get('friedman_note', 'no paired seeds')}\n")
            continue
        n_pairs = len(info["seeds"])
        lines.append(f"n paired seeds = {n_pairs} "
                     f"(excluded: {sorted(set(range(1, n_seeds_total+1)) - set(info['seeds']))})\n")
        lines.append("| script | n | mean | std | 95% CI |")
        lines.append("|---|---:|---:|---:|---|")
        for s in SCRIPTS:
            d = info["desc"][s]
            lines.append(f"| {LABELS[s]} | {d['n']} | {d['mean']:.6g} | {d['std']:.6g} | "
                         f"[{d['ci_lo']:.6g}, {d['ci_hi']:.6g}] |")
        lines.append("")
        lines.append(f"Friedman p = {fmt_p(info['friedman_p'])}, "
                     f"Holm-corrected p = {fmt_p(info.get('friedman_p_holm'))}, "
                     f"significant @ alpha=0.05: {info.get('significant', False)}\n")

        if info.get("significant"):
            lines.append("| pair | mean diff | 95% CI | Wilcoxon p | Holm p | sig | "
                         "A12 | magnitude |")
            lines.append("|---|---:|---|---:|---:|---|---:|---|")
            for (s1, s2) in PAIRS:
                r = info["pairwise"][(s1, s2)]
                lines.append(f"| {LABELS[s1]} vs {LABELS[s2]} | {r['mean_diff']:.6g} | "
                             f"[{r['ci_lo']:.6g}, {r['ci_hi']:.6g}] | {fmt_p(r['pvalue'])} | "
                             f"{fmt_p(r['pvalue_holm'])} | {r['significant']} | "
                             f"{r['a12']:.4f} | {r['a12_magnitude']} |")
            lines.append("")

    report = "\n".join(lines)
    with open(SUMMARY_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
