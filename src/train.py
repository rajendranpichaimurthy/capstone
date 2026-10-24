# Step 3 - model building with experiment tracking
# loads train/test from the workflow artifact, tunes 6 models with RandomizedSearchCV,
# logs every try in MLflow and saves the best model in models/ (the pipeline commits this folder)

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# in GitHub Actions the artifact is downloaded to artifacts/train-test-data
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts" / "train-test-data"))
if not ARTIFACT_DIR.exists():                     # running locally right after prep.py
    ARTIFACT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
SEED, N_ITER, CV_FOLDS = 42, 20, 5
CLEAN_TARGET = "engine_condition"
REFIT = "roc_auc"   # using roc_auc to pick the best - f1 is already 0.77 if we just say "faulty" for everything
EXPERIMENT = "engine-predictive-maintenance"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# clean up old mlflow runs first so every run starts fresh (has to happen before mlflow is imported)
for p in [ROOT / "mlflow.db", ROOT / "mlruns"]:
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()

import joblib
import logging
import mlflow
import mlflow.sklearn
from scipy.stats import randint, uniform, loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.getLogger("mlflow").setLevel(logging.ERROR)   # mlflow prints a lot of warnings, hiding them
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment(EXPERIMENT)

# the 6 models and the parameters to try for each
# n_jobs=1 inside the models because the search below already runs in parallel
MODELS = {
    "DecisionTree": (DecisionTreeClassifier(random_state=SEED), {
        "max_depth": randint(2, 15), "min_samples_split": randint(2, 200),
        "min_samples_leaf": randint(1, 100), "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"]}),
    "Bagging": (BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=SEED, n_jobs=1), {
        "n_estimators": randint(50, 200), "max_samples": uniform(0.3, 0.7),
        "max_features": uniform(0.5, 0.5), "bootstrap": [True, False],
        "estimator__max_depth": randint(3, 12), "estimator__min_samples_leaf": randint(1, 50)}),
    "RandomForest": (RandomForestClassifier(random_state=SEED, n_jobs=1), {
        "n_estimators": randint(100, 300), "max_depth": randint(3, 16),
        "min_samples_leaf": randint(1, 60), "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced", "balanced_subsample"]}),
    "AdaBoost": (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=SEED), {
        "n_estimators": randint(50, 300), "learning_rate": loguniform(0.01, 1.0),
        "estimator__max_depth": randint(1, 4)}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {
        "n_estimators": randint(100, 300), "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(2, 6), "subsample": uniform(0.6, 0.4),
        "min_samples_leaf": randint(1, 80)}),
    "XGBoost": (XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss", tree_method="hist"), {
        "n_estimators": randint(100, 400), "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(2, 8), "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 20),
        "gamma": uniform(0, 5), "reg_lambda": loguniform(0.1, 10)}),
}
SCORING = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision", "accuracy": "accuracy"}


def evaluate(model, X, y):
    # get all the scores on the test set (roc_auc needs the probabilities)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}, pred


def clean(v):
    # round long decimals before logging
    return round(float(v), 6) if isinstance(v, float) else v


def main():
    # load train and test from the artifact folder
    train_art = pd.read_csv(ARTIFACT_DIR / "train.csv")
    test_art = pd.read_csv(ARTIFACT_DIR / "test.csv")
    X_tr, y_tr = train_art.drop(columns=CLEAN_TARGET), train_art[CLEAN_TARGET]
    X_te, y_te = test_art.drop(columns=CLEAN_TARGET), test_art[CLEAN_TARGET]
    print(f"Loaded artifact from {ARTIFACT_DIR}: train={X_tr.shape}, test={X_te.shape}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results, fitted = [], {}

    # baseline model that always says 'faulty', to compare the real models against
    log("starting: baseline model")
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    dm, dpred = evaluate(dummy, X_te, y_te)
    tn, fp, fn, tp = confusion_matrix(y_te, dpred).ravel()
    with mlflow.start_run(run_name="Baseline-MajorityClass") as r:
        mlflow.set_tags({"algorithm": "Dummy", "stage": "final"})
        mlflow.log_param("strategy", "most_frequent")
        mlflow.log_metrics({f"test_{k}": v for k, v in dm.items()})
    results.append({"model": "Baseline (always faulty)", "cv_roc_auc": 0.5, "cv_roc_auc_std": 0.0,
                    "train_roc_auc": 0.5, "cv_f1": dm["f1"], **{f"test_{k}": v for k, v in dm.items()},
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp, "tuning_seconds": 0.0,
                    "best_params": "{}", "mlflow_run_id": r.info.run_id})
    log("finished: baseline model")

    # tune each model and log everything to mlflow
    log(f"starting: model tuning ({len(MODELS)} models x {N_ITER} tries x {CV_FOLDS} folds)")
    for name, (estimator, space) in MODELS.items():
        log(f"starting: tuning {name} ...")
        t0 = time.time()
        search = RandomizedSearchCV(estimator, space, n_iter=N_ITER, scoring=SCORING, refit=REFIT,
                                    cv=cv, random_state=SEED, n_jobs=-1, return_train_score=True)
        search.fit(X_tr, y_tr)
        secs = time.time() - t0
        cvr = pd.DataFrame(search.cv_results_)
        b = search.best_index_

        with mlflow.start_run(run_name=name) as parent:
            mlflow.set_tags({"algorithm": name, "stage": "final", "search": "RandomizedSearchCV"})
            mlflow.log_params({"cv_folds": CV_FOLDS, "n_iter": N_ITER, "refit_metric": REFIT, "seed": SEED})
            # log each of the 20 tries as a child run so all the parameters are saved
            for i, row in cvr.iterrows():
                with mlflow.start_run(run_name=f"{name}-cand{i:02d}", nested=True):
                    mlflow.log_params({k: clean(v) for k, v in row["params"].items()})
                    mlflow.log_metrics({f"cv_{m}_mean": row[f"mean_test_{m}"] for m in SCORING} |
                                       {f"cv_{m}_std": row[f"std_test_{m}"] for m in SCORING} |
                                       {"train_roc_auc_mean": row["mean_train_roc_auc"],
                                        "rank_roc_auc": int(row["rank_test_roc_auc"])})
            best_est = search.best_estimator_
            test_m, pred_m = evaluate(best_est, X_te, y_te)
            tn, fp, fn, tp = confusion_matrix(y_te, pred_m).ravel()
            mlflow.log_params({f"best_{k}": clean(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()} |
                               {"cv_roc_auc_mean": cvr.loc[b, "mean_test_roc_auc"],
                                "cv_roc_auc_std": cvr.loc[b, "std_test_roc_auc"],
                                "cv_f1_mean": cvr.loc[b, "mean_test_f1"],
                                "train_roc_auc": cvr.loc[b, "mean_train_roc_auc"],
                                "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
                                "tuning_seconds": secs})
            mlflow.sklearn.log_model(best_est, name="model", serialization_format="cloudpickle")
            run_id = parent.info.run_id

        fitted[name] = best_est
        results.append({"model": name, "cv_roc_auc": cvr.loc[b, "mean_test_roc_auc"],
                        "cv_roc_auc_std": cvr.loc[b, "std_test_roc_auc"],
                        "train_roc_auc": cvr.loc[b, "mean_train_roc_auc"], "cv_f1": cvr.loc[b, "mean_test_f1"],
                        **{f"test_{k}": v for k, v in test_m.items()},
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "tuning_seconds": round(secs, 1),
                        "best_params": json.dumps({k: clean(v) for k, v in search.best_params_.items()}, default=str),
                        "mlflow_run_id": run_id})
        print(f"{name:<17} cv_auc={cvr.loc[b, 'mean_test_roc_auc']:.4f}  test_auc={test_m['roc_auc']:.4f}  "
              f"test_f1={test_m['f1']:.4f}  test_recall={test_m['recall']:.4f}  "
              f"test_acc={test_m['accuracy']:.4f}  ({secs:.0f}s)", flush=True)
        log(f"finished: {name} in {secs:.0f}s")
    log("finished: model tuning")

    # sort models by cv roc_auc and save the best one in models/
    log("starting: ranking models and saving the best one")
    res = pd.DataFrame(results)
    res = pd.concat([res[res.model != "Baseline (always faulty)"].sort_values("cv_roc_auc", ascending=False),
                     res[res.model == "Baseline (always faulty)"]]).reset_index(drop=True)
    MODEL_DIR.mkdir(exist_ok=True)
    res.to_csv(MODEL_DIR / "experiment_results.csv", index=False)

    best_name = res.loc[0, "model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    meta = {"best_model": best_name, "selection_metric": "cv_roc_auc (5-fold, training set)",
            "features": list(X_tr.columns), "target": CLEAN_TARGET, "decision_threshold": 0.5,
            "best_params": json.loads(res.loc[0, "best_params"]),
            "cv_roc_auc": round(float(res.loc[0, "cv_roc_auc"]), 4),
            "test_metrics": {k: round(float(res.loc[0, f"test_{k}"]), 4)
                             for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
            "mlflow_experiment": EXPERIMENT, "mlflow_run_id": res.loc[0, "mlflow_run_id"]}
    (MODEL_DIR / "best_model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Best model: {best_name}  -> saved to models/best_model.joblib")
    print(json.dumps(meta, indent=2))
    log("finished: ranking models and saving the best one")


if __name__ == "__main__":
    main()
