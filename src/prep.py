# Step 2 - data preparation
# loads the csv from the data folder, cleans it, splits 80/20 and saves train.csv and test.csv
# in GitHub Actions the output folder (data/processed) is uploaded as the artifact "train-test-data"

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "engine_data.csv"     # read straight from the data folder
PROCESSED_DIR = ROOT / "data" / "processed"       # train and test files are saved here
TEST_SIZE, SEED = 0.20, 42
CLEAN_TARGET = "engine_condition"
RENAME = {"Engine rpm": "engine_rpm", "Lub oil pressure": "lub_oil_pressure",
          "Fuel pressure": "fuel_pressure", "Coolant pressure": "coolant_pressure",
          "lub oil temp": "lub_oil_temp", "Coolant temp": "coolant_temp",
          "Engine Condition": CLEAN_TARGET}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_data(raw_path=RAW_PATH, out_dir=PROCESSED_DIR):
    # cleans the data, splits it 80/20 and saves train.csv and test.csv
    data = pd.read_csv(raw_path)
    info = {"raw_rows": len(data)}

    # 1. rename columns (no spaces, all lowercase)
    data = data.rename(columns=RENAME)

    # 2. drop duplicates and rows with missing values
    info["duplicates_removed"] = int(data.duplicated().sum())
    data = data.drop_duplicates()
    info["null_rows_removed"] = int(data.isna().any(axis=1).sum())
    data = data.dropna()

    # 3. drop columns we don't need (constant columns, id columns, anything extra)
    constant = [c for c in data.columns if data[c].nunique() <= 1]
    id_like = [c for c in data.columns if c != CLEAN_TARGET and data[c].nunique() == len(data)
               and pd.api.types.is_integer_dtype(data[c])]
    unknown = [c for c in data.columns if c not in RENAME.values()]
    drop_cols = sorted(set(constant + id_like + unknown))
    data = data.drop(columns=drop_cols)
    info["columns_dropped"] = drop_cols

    # 4. drop impossible rows (rpm 0 or negative values)
    features = [c for c in data.columns if c != CLEAN_TARGET]
    impossible = (data["engine_rpm"] <= 0) | (data[features] < 0).any(axis=1)
    info["impossible_rows_removed"] = int(impossible.sum())
    data = data[~impossible]

    # 5. count outliers but don't remove them - EDA showed they are often faulty engines
    q1, q3 = data[features].quantile(0.25), data[features].quantile(0.75)
    iqr = q3 - q1
    out_mask = (data[features] < q1 - 1.5 * iqr) | (data[features] > q3 + 1.5 * iqr)
    info["rows_with_any_iqr_outlier_kept"] = int(out_mask.any(axis=1).sum())

    # 6. split 80/20 (stratify keeps the same % faulty in both) and save
    X, y = data[features], data[CLEAN_TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([X_tr, y_tr], axis=1).to_csv(out_dir / "train.csv", index=False)
    pd.concat([X_te, y_te], axis=1).to_csv(out_dir / "test.csv", index=False)

    info.update({"clean_rows": len(data), "features": features, "target": CLEAN_TARGET,
                 "train_rows": len(X_tr), "test_rows": len(X_te),
                 "train_positive_rate": round(float(y_tr.mean()), 4),
                 "test_positive_rate": round(float(y_te.mean()), 4),
                 "test_size": TEST_SIZE, "random_state": SEED})
    (out_dir / "prep_summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("Saved train/test splits to data/processed/")
    return info


if __name__ == "__main__":
    log("starting: data preparation")
    prepare_data()
    log("finished: data preparation")
