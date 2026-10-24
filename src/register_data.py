# Step 1 - data registration
# checks that data/engine_data.csv is what we expect before anything else uses it
# prints a small report and saves data/registry.json
# exits with code 1 if a check fails, so the pipeline stops here

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "engine_data.csv"
REGISTRY_PATH = ROOT / "data" / "registry.json"

# columns I expect, with a min and max value that makes sense for each sensor
# (kept the limits wide so only clearly wrong values like negatives get caught)
EXPECTED_COLUMNS = {
    "Engine rpm":       (0, 5000),
    "Lub oil pressure": (0, 15),
    "Fuel pressure":    (0, 40),
    "Coolant pressure": (0, 15),
    "lub oil temp":     (0, 200),
    "Coolant temp":     (0, 250),
    "Engine Condition": (0, 1),
}
RAW_TARGET = "Engine Condition"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    # sha256 hash of the file - if the csv changes, this value changes
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_data(path=DATA_PATH, registry_path=REGISTRY_PATH):
    # runs all the checks, prints a small report and saves registry.json
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    data = pd.read_csv(path)
    checks = []

    # check 1 - are all the columns there?
    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    extra = [c for c in data.columns if c not in EXPECTED_COLUMNS]
    checks.append(("All expected columns present", not missing, f"missing={missing}"))
    checks.append(("No unexpected columns", not extra, f"extra={extra}"))
    if missing:
        raise ValueError(f"Registration failed: missing columns {missing}")

    # check 2 - numbers only, values in range, target is 0/1, enough rows
    non_numeric = [c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(data[c])]
    checks.append(("All columns numeric", not non_numeric, f"non_numeric={non_numeric}"))
    out_of_range = {c: int(((data[c] < lo) | (data[c] > hi)).sum()) for c, (lo, hi) in EXPECTED_COLUMNS.items()}
    out_of_range = {c: n for c, n in out_of_range.items() if n}
    checks.append(("Values within plausible bounds", not out_of_range, f"{out_of_range}"))
    checks.append(("Target is binary {0,1}", set(data[RAW_TARGET].unique()) <= {0, 1},
                   f"values={sorted(data[RAW_TARGET].unique())}"))
    checks.append(("Minimum row count (>= 1,000)", len(data) >= 1000, f"rows={len(data)}"))

    # nulls and duplicates - just counting here, cleaning is done later in data prep
    n_null = int(data.isna().sum().sum())
    n_dup = int(data.duplicated().sum())

    print("=" * 64)
    print("DATASET REGISTRATION REPORT")
    print("=" * 64)
    print(f"File           : {path.name}")
    print(f"Rows x Columns : {data.shape[0]:,} x {data.shape[1]}")
    print(f"Null values    : {n_null}")
    print(f"Duplicate rows : {n_dup}")
    print("Class balance  : " + ", ".join(
        f"{k}={v:.2%}" for k, v in data[RAW_TARGET].value_counts(normalize=True).sort_index().items()))
    print("-" * 64)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {detail}"))
    print("-" * 64)
    print(data.describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())

    passed = all(ok for _, ok, _ in checks)
    registry = {
        "dataset": path.name,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(path),
        "rows": int(data.shape[0]), "columns": list(data.columns),
        "nulls": n_null, "duplicates": n_dup,
        "target_distribution": {str(k): int(v) for k, v in data[RAW_TARGET].value_counts().sort_index().items()},
        "validation_passed": passed,
    }
    registry_path.write_text(json.dumps(registry, indent=2))
    print("-" * 64)
    print(f"Registry written to data/registry.json  |  status: {'VALID' if passed else 'INVALID'}")
    return passed


if __name__ == "__main__":
    log("starting: data registration")
    ok = register_data()
    log("finished: data registration")
    sys.exit(0 if ok else 1)
