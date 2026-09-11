"""
analyze_results.py

Shared statistics helpers, trimmed to exactly the functions imported by
analyze_results_3way.py (and, through it, by
analyze_admission_metrics_25seed.py and analyze_graduated_25seed.py --
the two scripts that reproduce the paper's reported admission-metric
statistics): _paired_ci, _vda12, _vda12_magnitude, _wilcoxon_safe.

This file originally also analyzed results/raw_runs.csv (a 2-way
Baseline vs MAPE-SR comparison) directly via its own main(); that part
is not included here since results/raw_runs.csv is outside this
release's scope (see the repository root README). Only the shared
statistical primitives remain.
"""
import warnings

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon


def _paired_ci(diffs, confidence=0.95):
    """Paired t-interval for the mean difference."""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean = diffs.mean()
    sem = diffs.std(ddof=1) / np.sqrt(n)
    if sem == 0:
        return mean, mean
    tc = t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return mean - tc * sem, mean + tc * sem


def _vda12(a_vals, b_vals):
    """Vargha-Delaney A12 effect size (Vargha & Delaney, 2000): the
    probability that a randomly drawn value from `a_vals` exceeds a
    randomly drawn value from `b_vals`, with ties counted as half a win.
    Computed via the full n1*n2 cross-product (the standard/original
    definition -- a stochastic-dominance measure between the two
    distributions, not a per-pair statistic), which also gives a
    well-defined, correct value (0.5, no effect) in the degenerate case
    where every paired difference is zero and Wilcoxon is not applicable.

    0.5 = no difference. Common thresholds (Vargha & Delaney 2000; Hess &
    Kromrey 2004): negligible <0.56, small 0.56-0.64, medium 0.64-0.71,
    large >0.71 (mirrored below 0.5 the same way).
    """
    a = np.asarray(a_vals)
    b = np.asarray(b_vals)
    wins = (a[:, None] > b[None, :]).sum()
    ties = (a[:, None] == b[None, :]).sum()
    n1, n2 = len(a), len(b)
    return (wins + 0.5 * ties) / (n1 * n2)


def _vda12_magnitude(a12):
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def _wilcoxon_safe(a_vals, b_vals):
    """Runs scipy.stats.wilcoxon on the paired values, handling the
    degenerate all-zero-difference case gracefully regardless of how the
    installed scipy version manifests it (raising, or returning a NaN
    p-value with a RuntimeWarning) -- never crashes the analysis.

    Returns (pvalue_or_None, note_or_None).
    """
    diffs = a_vals - b_vals
    if np.all(diffs == 0):
        return None, "no variation across paired runs (identical values) -- test not applicable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _stat, pvalue = wilcoxon(a_vals, b_vals)
    except Exception as e:
        return None, (f"no variation across paired runs (identical values) -- "
                       f"test not applicable ({type(e).__name__})")
    if pvalue is not None and np.isnan(pvalue):
        return None, "no variation across paired runs (identical values) -- test not applicable"
    return pvalue, None
