"""Model compression (TFLite fp32 / fp16 / dynamic-range int8) and inference benchmarking.

Every variant is scored on the same test split, so size and latency gains are reported
together with any accuracy cost.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cnn import _tf, load_model, load_split
from .metrics import classification_metrics, save_json

VARIANTS = ("fp32", "fp16", "int8_dynamic")


def convert(saved_model_dir: Path, variant: str) -> bytes:
    tf, _ = _tf()
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if variant in ("fp16", "int8_dynamic"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if variant == "fp16":
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


class TFLiteRunner:
    def __init__(self, path: Path, image_size: int, threads: int = 1):
        tf, _ = _tf()
        self.interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]
        self.interp.resize_tensor_input(self.inp, [1, image_size, image_size, 3])
        self.interp.allocate_tensors()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        self.interp.set_tensor(self.inp, image[None].astype(np.float32))
        self.interp.invoke()
        return self.interp.get_tensor(self.out)[0]


def latency_ms(fn, x: np.ndarray, warmup: int, runs: int) -> dict:
    for i in range(warmup):
        fn(x[i % len(x)])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(x[i % len(x)])
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def run(manifest: pd.DataFrame, cfg: dict) -> dict:
    tf, _ = _tf()
    classes, size = cfg["data"]["classes"], cfg["data"]["image_size"]
    b = cfg["benchmark"]
    art = Path(cfg["paths"]["artifacts"]) / "cnn"
    keras_path = art / "model.keras"
    model = load_model(keras_path)
    x, y = load_split(manifest, cfg, "test")
    xf = x.astype(np.float32)

    def score(probs):
        m = classification_metrics(y, np.asarray(probs).argmax(1), classes, n_bootstrap=0)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    rows = {}
    keras_single = lambda img: model(img[None], training=False).numpy()[0]  # noqa: E731
    rows["keras"] = {"size_mb": keras_path.stat().st_size / 1e6,
                     **score([keras_single(img) for img in xf]),
                     **latency_ms(keras_single, xf, b["warmup"], b["runs"])}

    saved = art / "saved_model"
    model.export(str(saved))
    for v in VARIANTS:
        path = art / f"model_{v}.tflite"
        path.write_bytes(convert(saved, v))
        runner = TFLiteRunner(path, size)
        rows[f"tflite_{v}"] = {"size_mb": path.stat().st_size / 1e6,
                               **score([runner(img) for img in xf]),
                               **latency_ms(runner, xf, b["warmup"], b["runs"])}

    report = {"n_test": int(len(y)), "threads": 1, "variants": rows,
              "tensorflow_version": tf.__version__}
    rep = Path(cfg["paths"]["reports"]) / "export"
    save_json(report, rep / "report.json")
    (rep / "report.md").write_text(to_markdown(report))
    print(to_markdown(report))
    return report


def to_markdown(report: dict) -> str:
    lines = ["| Variant | Size (MB) | Test accuracy | Macro-F1 | Latency p50 (ms) | Latency p95 (ms) |",
             "|---|---|---|---|---|---|"]
    for name, r in report["variants"].items():
        lines.append(f"| {name} | {r['size_mb']:.2f} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                     f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} |")
    return "\n".join(lines) + f"\n\nSingle-image CPU latency, {report['threads']} thread, n_test={report['n_test']}.\n"
