"""
analyze_results_3way.py

Shared statistics helpers, trimmed to exactly the names imported by
analyze_admission_metrics_25seed.py and analyze_graduated_25seed.py --
the two scripts that reproduce the paper's reported admission-metric
statistics (quiet-phase throughput, cumulative admission, admission
oscillation) from the CSVs in results/: SCRIPT_BASELINE, SCRIPT_AG,
SCRIPT_MAPESR, SCRIPTS, LABELS, PAIRS, ALPHA, _holm_bonferroni,
_friedman_safe, _descriptive, _pairwise.

METHOD (per-metric, for the metrics common to all three scripts) --
this is the exact pipeline analyze_admission_metrics_25seed.py and
analyze_graduated_25seed.py apply to their own metrics:
  1. scipy.stats.friedmanchisquare as the omnibus test across the three
     paired (by-seed) groups -- the right choice for 3+ related samples,
     instead of running three pairwise Wilcoxon tests directly and
     inflating the family-wise false-positive rate.
  2. Holm-Bonferroni correction #1: across every metric that received a
     Friedman test (only metrics with an actual, non-degenerate p-value
     enter this family -- see _friedman_safe()).
  3. For metrics significant after that correction: pairwise Wilcoxon
     signed-rank + Vargha-Delaney A12 for the three pairs (Baseline vs
     Adaptive-Generic, Adaptive-Generic vs MAPE-SR, Baseline vs MAPE-SR),
     with Holm-Bonferroni correction #2 applied WITHIN that metric across
     just its three pairwise p-values.
  4. For metrics NOT significant after correction #1: descriptive stats
     only, no pairwise follow-up -- not fishing for significance the
     omnibus test didn't support.

This file originally also analyzed results/raw_runs_3way.csv (the
broader 3-way comparison covering detection-quality metrics, PSI, and
figure generation) directly via its own main(); that part is not
included here since results/raw_runs_3way.csv is outside this release's
scope (see the repository root README). Only the shared statistical
primitives and constants remain.
"""
import warnings

import numpy as np
from scipy.stats import friedmanchisquare, t as t_dist

from analyze_results import _paired_ci, _vda12, _vda12_magnitude, _wilcoxon_safe

SCRIPT_BASELINE = "test_pure_mapek"
SCRIPT_AG = "test_adaptive_generic"
SCRIPT_MAPESR = "test_nist_only"
SCRIPTS = [SCRIPT_BASELINE, SCRIPT_AG, SCRIPT_MAPESR]
LABELS = {SCRIPT_BASELINE: "Baseline", SCRIPT_AG: "Adaptive-Generic", SCRIPT_MAPESR: "MAPE-SR"}

ALPHA = 0.05
PAIRS = [
    (SCRIPT_BASELINE, SCRIPT_AG),
    (SCRIPT_AG, SCRIPT_MAPESR),
    (SCRIPT_BASELINE, SCRIPT_MAPESR),
]


def _holm_bonferroni(pvals):
    """Standard Holm step-down correction. `pvals` is a list; returns the
    adjusted p-values in the SAME order (not sorted)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min((m - rank) * pvals[i], 1.0)
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted


def _friedman_safe(vals_by_script, scripts=SCRIPTS):
    """Runs friedmanchisquare across the 3 groups; handles the degenerate
    all-identical case (chi2 undefined/0-variance ranks) the same way
    analyze_results.py's _wilcoxon_safe handles all-zero paired diffs --
    never crashes, returns (None, note) instead."""
    arrs = [vals_by_script[s] for s in scripts]
    if all(np.array_equal(arrs[0], a) for a in arrs[1:]):
        return None, "no variation across the three groups (identical values) -- test not applicable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _stat, pvalue = friedmanchisquare(*arrs)
    except Exception as e:
        return None, f"Friedman test not applicable ({type(e).__name__})"
    if pvalue is not None and np.isnan(pvalue):
        return None, "Friedman returned NaN -- not applicable"
    return pvalue, None


def _descriptive(arr):
    n = len(arr)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else 0.0
    if n > 1 and std > 0:
        tc = t_dist.ppf(0.975, df=n - 1)
        half = tc * std / np.sqrt(n)
        ci = (mean - half, mean + half)
    else:
        ci = (mean, mean)
    return {"n": n, "mean": mean, "std": std, "ci_lo": ci[0], "ci_hi": ci[1]}


def _pairwise(a_vals, b_vals):
    diffs = a_vals - b_vals
    ci_lo, ci_hi = _paired_ci(diffs)
    pvalue, note = _wilcoxon_safe(a_vals, b_vals)
    a12 = _vda12(a_vals, b_vals)
    return {
        "mean_diff": diffs.mean(), "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pvalue": pvalue, "note": note, "a12": a12, "a12_magnitude": _vda12_magnitude(a12),
    }
