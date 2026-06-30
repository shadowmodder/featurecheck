"""Population stability, distribution drift, and schema checks."""
from __future__ import annotations
import numpy as np


def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index using quantile bins from the expected sample."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = e_counts / max(e_counts.sum(), 1) + eps
    a_perc = a_counts / max(a_counts.sum(), 1) + eps
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def check_schema(records, schema):
    """records: list[dict]; schema: {column: 'int'|'float'|'str'|'bool'}."""
    type_map = {"int": int, "float": (int, float), "str": str, "bool": bool}
    cols = set()
    for r in records:
        cols.update(r.keys())
    expected = set(schema)
    n = len(records) or 1
    issues, null_rates = [], {}
    for col, tname in schema.items():
        nulls = mism = 0
        for r in records:
            v = r.get(col)
            if v is None:
                nulls += 1
            elif not isinstance(v, type_map[tname]):
                mism += 1
        null_rates[col] = nulls / n
        if mism:
            issues.append({"column": col, "type_mismatches": mism, "expected": tname})
    return {"missing_columns": sorted(expected - cols),
            "extra_columns": sorted(cols - expected),
            "type_issues": issues, "null_rates": null_rates, "n": len(records)}


def chi2_drift(expected_counts, actual_counts, eps=1e-6):
    """Chi-squared drift between two categorical count distributions.

    Args:
        expected_counts: dict {category: count} or list/array of counts.
        actual_counts:   same shape as expected_counts.

    Returns dict with chi2, dof, chi2_per_dof, and level ('ok'/'warn'/'alert').
    Thresholds (chi2/dof > 2 → warn, > 5 → alert) are guidelines, not p-values.
    """
    import numpy as np  # already imported at module level, re-stated for clarity
    if isinstance(expected_counts, dict):
        categories = sorted(set(expected_counts) | set(actual_counts))
        e = np.array([expected_counts.get(c, 0) for c in categories], dtype=float)
        a = np.array([actual_counts.get(c, 0) for c in categories], dtype=float)
    else:
        e = np.asarray(expected_counts, dtype=float)
        a = np.asarray(actual_counts, dtype=float)
    # Scale expected to the same total as actual
    e = e / (e.sum() + eps) * (a.sum() + eps)
    e = np.maximum(e, eps)
    stat = float(np.sum((a - e) ** 2 / e))
    dof = max(len(e) - 1, 1)
    ratio = stat / dof
    level = "alert" if ratio >= 5.0 else "warn" if ratio >= 2.0 else "ok"
    return {"chi2": stat, "dof": dof, "chi2_per_dof": ratio, "level": level}


def drift_report(expected, actual, psi_warn=0.1, psi_alert=0.25):
    val = psi(expected, actual)
    level = "alert" if val >= psi_alert else "warn" if val >= psi_warn else "ok"
    return {"psi": val, "ks": ks_statistic(expected, actual), "level": level}
