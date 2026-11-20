"""Classical baseline: handcrafted color/texture features + scikit-learn classifier.

Model selection uses the validation split only; the test split is scored once at the end.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features
from .images import load_batch
from .metrics import classification_metrics, environment, macro_f1, plot_confusion, save_json


def make_model(name: str, cfg: dict):
    b, seed = cfg["baseline"], cfg["seed"]
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=b["logreg_C"], max_iter=5000, class_weight="balanced", random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=b["rf_n_estimators"], class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown baseline model: {name}")


def feature_matrix(manifest: pd.DataFrame, cfg: dict) -> np.ndarray:
    root = Path(cfg["data"]["root"])
    images = load_batch([root / p for p in manifest["path"]], cfg["data"]["image_size"])
    return features.extract_batch(images)


def run(manifest: pd.DataFrame, cfg: dict) -> dict:
    classes = cfg["data"]["classes"]
    out_dir = Path(cfg["paths"]["reports"]) / "baseline"
    X = feature_matrix(manifest, cfg)
    y = manifest["label_id"].to_numpy()
    split = manifest["split"].to_numpy()
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    val_scores = {}
    for name in cfg["baseline"]["candidates"]:
        model = make_model(name, cfg).fit(X[tr], y[tr])
        val_scores[name] = float(macro_f1(y[va], model.predict(X[va])))
    best = max(val_scores, key=val_scores.get)

    final = make_model(best, cfg).fit(X[tr | va], y[tr | va])
    metrics = classification_metrics(y[te], final.predict(X[te]), classes,
                                     cfg["evaluate"]["bootstrap_samples"], cfg["seed"])
    metrics.update({"model": best, "val_macro_f1_by_candidate": val_scores, "n_features": int(X.shape[1]),
                    "environment": environment()})

    save_json(metrics, out_dir / "metrics.json")
    plot_confusion(metrics["confusion_matrix"], classes, out_dir / "confusion_matrix.png",
                   f"Baseline ({best}) - test")
    art = Path(cfg["paths"]["artifacts"]) / "baseline"
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, art / "model.joblib")
    return metrics
