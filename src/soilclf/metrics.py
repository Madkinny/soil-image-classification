"""Classification metrics with bootstrap confidence intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI. Small test sets make point estimates noisy, so always report this."""
    rng = np.random.default_rng(seed)
    m = len(y_true)
    stats = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, m, m)
        stats[b] = metric(y_true[idx], y_pred[idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def classification_metrics(y_true, y_pred, class_names: list[str], n_bootstrap: int = 2000,
                           seed: int = 0) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ids = list(range(len(class_names)))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=ids, target_names=class_names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "class_names": class_names,
    }
    if n_bootstrap:
        out["accuracy_ci95"] = bootstrap_ci(y_true, y_pred, accuracy_score, n_bootstrap, seed)
        out["macro_f1_ci95"] = bootstrap_ci(y_true, y_pred, macro_f1, n_bootstrap, seed)
    return out


def plot_confusion(cm, class_names: list[str], path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def environment() -> dict:
    """Library versions that can change results (recorded next to every metrics file)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    env = {"python": platform.python_version()}
    for pkg in ("numpy", "scikit-learn", "scikit-image", "opencv-python-headless", "pillow", "tensorflow"):
        try:
            env[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return env


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def summary_line(name: str, m: dict) -> str:
    s = f"{name}: accuracy {m['accuracy']:.3f}"
    if "accuracy_ci95" in m:
        lo, hi = m["accuracy_ci95"]
        s += f" (95% CI {lo:.3f}-{hi:.3f})"
    s += f", macro-F1 {m['macro_f1']:.3f}, n={m['n']}"
    return s
