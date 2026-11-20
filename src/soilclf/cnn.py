"""MobileNetV2 transfer learning: two-phase training (frozen head, then partial fine-tune).

Preprocessing (scaling to [-1, 1]) is inside the model, so callers always pass raw RGB
uint8/float images from `soilclf.images.load_rgb`. TensorFlow is imported lazily so the
rest of the package (audit, baseline, tests) works without it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .images import load_batch
from .metrics import classification_metrics, plot_confusion, save_json, summary_line


def _tf():
    import keras  # noqa: F401  (Keras 3, installed with TensorFlow >= 2.16)
    import tensorflow as tf

    return tf, keras


def set_seed(seed: int) -> None:
    tf, keras = _tf()
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # some GPU kernels have no deterministic implementation
        pass


def build_model(num_classes: int, image_size: int, dropout: float, weights: str | None = "imagenet"):
    _, keras = _tf()
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")  # raw RGB in [0, 255]
    x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    base = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                          input_shape=(image_size, image_size, 3))
    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm in inference mode, also during fine-tuning
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="soil_mobilenetv2"), base


def augmenter(seed: int):
    _, keras = _tf()
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        keras.layers.RandomRotation(0.1, fill_mode="reflect", seed=seed),
        keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=seed),
        keras.layers.RandomContrast(0.15, seed=seed),
    ], name="augment")


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, train: bool):
    tf, _ = _tf()
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y.astype("int32")))
    if train:
        aug = augmenter(seed)
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True).batch(batch_size)
        ds = ds.map(lambda a, b: (tf.clip_by_value(aug(a, training=True), 0.0, 255.0), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_split(manifest: pd.DataFrame, cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    rows = manifest[manifest["split"] == split]
    root = Path(cfg["data"]["root"])
    x = load_batch([root / p for p in rows["path"]], cfg["data"]["image_size"])
    return x, rows["label_id"].to_numpy()


def _class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return {i: float(len(y) / (n_classes * c)) for i, c in enumerate(counts) if c > 0}


def train(manifest: pd.DataFrame, cfg: dict, weights: str | None = "imagenet") -> Path:
    _, keras = _tf()
    t, d, seed = cfg["train"], cfg["data"], cfg["seed"]
    set_seed(seed)
    classes = d["classes"]
    xtr, ytr = load_split(manifest, cfg, "train")
    xva, yva = load_split(manifest, cfg, "val")
    ds_tr = make_dataset(xtr, ytr, t["batch_size"], seed, train=True)
    ds_va = make_dataset(xva, yva, t["batch_size"], seed, train=False)
    cw = _class_weights(ytr, len(classes))

    model, base = build_model(len(classes), d["image_size"], t["dropout"], weights)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stopping_patience"],
                                         restore_best_weights=True)
    history: dict[str, list] = {}

    def fit(epochs: int, lr: float, phase: str):
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, class_weight=cw,
                      callbacks=[stop], verbose=2)
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)
        history.setdefault("phase", []).extend([phase] * len(h.history["loss"]))

    start = time.perf_counter()
    fit(t["epochs_head"], t["lr_head"], "head")
    if t["epochs_finetune"] > 0 and t["finetune_layers"] > 0:
        base.trainable = True
        for layer in base.layers[:-t["finetune_layers"]]:
            layer.trainable = False
        fit(t["epochs_finetune"], t["lr_finetune"], "finetune")
    train_seconds = time.perf_counter() - start

    out = Path(cfg["paths"]["artifacts"]) / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.keras"
    model.save(model_path)
    (out / "labels.json").write_text(json.dumps(classes))
    rep = Path(cfg["paths"]["reports"]) / "cnn"
    save_json({"history": history, "train_seconds": train_seconds,
               "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
               "total_params": int(model.count_params())}, rep / "training.json")
    _plot_history(history, rep / "training_curves.png")
    return model_path


def _plot_history(history: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(history.get(key, []), label=f"train {key}")
        ax.plot(history.get(f"val_{key}", []), label=f"val {key}")
        n_head = history.get("phase", []).count("head")
        if 0 < n_head < len(history.get(key, [])):
            ax.axvline(n_head - 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_model(path: str | Path):
    _, keras = _tf()
    return keras.models.load_model(path)


def predict_proba(model, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
    return model.predict(images.astype("float32"), batch_size=batch_size, verbose=0)


def evaluate(manifest: pd.DataFrame, cfg: dict, model_path: str | Path | None = None) -> dict:
    classes = cfg["data"]["classes"]
    model_path = model_path or Path(cfg["paths"]["artifacts"]) / "cnn" / "model.keras"
    model = load_model(model_path)
    rep = Path(cfg["paths"]["reports"]) / "cnn"
    results = {}
    for split in ("val", "test"):
        x, y = load_split(manifest, cfg, split)
        probs = predict_proba(model, x)
        m = classification_metrics(y, probs.argmax(1), classes, cfg["evaluate"]["bootstrap_samples"], cfg["seed"])
        thr = cfg["evaluate"]["uncertain_threshold"]
        m["share_below_confidence_threshold"] = float((probs.max(1) < thr).mean())
        m["confidence_threshold"] = thr
        results[split] = m
        print(summary_line(f"CNN {split}", m))
    save_json(results, rep / "metrics.json")
    plot_confusion(results["test"]["confusion_matrix"], classes, rep / "confusion_matrix.png", "CNN - test")
    return results
