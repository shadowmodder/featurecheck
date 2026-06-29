# featurecheck

Most production ML incidents are data incidents. `featurecheck` is a tiny, NumPy-only guardrail you can run in training, batch scoring, or CI to catch feature drift and schema breakage early.

## Install
```bash
pip install -e .
```

## Usage
```python
from featurecheck import drift_report, psi, ks_statistic, check_schema

# Distribution drift between a reference and a live sample
drift_report(reference_values, live_values)   # -> {"psi":..., "ks":..., "level":"ok|warn|alert"}

# Schema / null / dtype validation on records
check_schema(records, {"age": "int", "email": "str"})
```

- **PSI** with quantile binning from the reference sample
- **KS** two-sample statistic
- **Schema check**: missing/extra columns, per-column null rates, dtype mismatches

MIT licensed.
