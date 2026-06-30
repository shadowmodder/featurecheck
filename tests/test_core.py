import numpy as np
from featurecheck import psi, ks_statistic, check_schema, drift_report, chi2_drift


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)
    assert psi(a, b) > 0.25
    assert drift_report(a, b)["level"] == "alert"


def test_ks_bounds():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 1000)
    assert 0.0 <= ks_statistic(a, a) < 1e-9


def test_schema_issues():
    recs = [{"age": 30, "name": "a"}, {"age": "oops"}]
    out = check_schema(recs, {"age": "int", "name": "str"})
    assert out["null_rates"]["name"] == 0.5
    assert any(i["column"] == "age" for i in out["type_issues"])


def test_chi2_identical():
    counts = {"cat": 100, "dog": 80, "bird": 40}
    result = chi2_drift(counts, counts)
    assert result["chi2"] < 1e-6
    assert result["level"] == "ok"


def test_chi2_heavy_shift():
    expected = {"A": 500, "B": 500}
    actual = {"A": 950, "B": 50}
    result = chi2_drift(expected, actual)
    assert result["level"] in ("warn", "alert")
