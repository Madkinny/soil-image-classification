"""Command-line interface: soilclf {audit,baseline,train,evaluate,export,predict}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import apply_frozen_split, audit_markdown, build_manifest
from .metrics import save_json, summary_line


def manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["reports"]) / "manifest.csv"


def get_manifest(cfg: dict) -> pd.DataFrame:
    path = manifest_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `soilclf audit` first.")
    return pd.read_csv(path)


def cmd_audit(cfg: dict, args) -> None:
    manifest, report = build_manifest(cfg)
    existing = manifest_path(cfg)
    if existing.exists() and not args.resplit:
        manifest, report = apply_frozen_split(manifest, report, pd.read_csv(existing))
        print(f"Reusing the frozen split in {existing} (use --resplit to regenerate).")
    else:
        report["split_source"] = "new split"
    rep = Path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path(cfg), index=False)
    save_json(report, rep / "data_audit.json")
    md = audit_markdown(report)
    (rep / "data_audit.md").write_text(md)
    print(md)


def cmd_baseline(cfg: dict, args) -> None:
    from . import baseline

    m = baseline.run(get_manifest(cfg), cfg)
    print(f"Validation macro-F1 by candidate: {m['val_macro_f1_by_candidate']}")
    print(summary_line(f"Baseline ({m['model']}) test", m))


def cmd_train(cfg: dict, args) -> None:
    from . import cnn

    path = cnn.train(get_manifest(cfg), cfg, weights=None if args.no_pretrained else "imagenet")
    print(f"Saved model to {path}")


def cmd_evaluate(cfg: dict, args) -> None:
    from . import cnn

    cnn.evaluate(get_manifest(cfg), cfg, args.model)


def cmd_export(cfg: dict, args) -> None:
    from . import export

    export.run(get_manifest(cfg), cfg)


def cmd_predict(cfg: dict, args) -> None:
    import numpy as np

    from .images import load_rgb

    classes, size = cfg["data"]["classes"], cfg["data"]["image_size"]
    model_path = Path(args.model or Path(cfg["paths"]["artifacts"]) / "cnn" / "model.keras")
    if model_path.suffix == ".tflite":
        from .export import TFLiteRunner

        runner = TFLiteRunner(model_path, size)
        predict = lambda img: runner(img.astype(np.float32))  # noqa: E731
    else:
        from .cnn import load_model, predict_proba

        model = load_model(model_path)
        predict = lambda img: predict_proba(model, img[None])[0]  # noqa: E731
    thr = cfg["evaluate"]["uncertain_threshold"]
    for image in args.images:
        probs = predict(load_rgb(image, size))
        top = int(np.argmax(probs))
        print(json.dumps({
            "image": str(image),
            "prediction": classes[top],
            "confidence": round(float(probs[top]), 4),
            "uncertain": bool(probs[top] < thr),
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        }))


def parse_set(values: list[str]) -> dict:
    out = {}
    for kv in values or []:
        key, _, raw = kv.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="soilclf", description=__doc__)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value (repeatable), e.g. --set train.epochs_head=5")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="Scan, de-duplicate, and split the dataset; write the manifest and audit report")
    a.add_argument("--resplit", action="store_true",
                   help="Ignore the committed manifest and create a new split")
    sub.add_parser("baseline", help="Train and test the handcrafted-feature baseline")
    t = sub.add_parser("train", help="Train the MobileNetV2 transfer-learning model")
    t.add_argument("--no-pretrained", action="store_true", help="Random init (for smoke tests, no download)")
    e = sub.add_parser("evaluate", help="Evaluate the CNN on val and test splits")
    e.add_argument("--model", default=None)
    sub.add_parser("export", help="Export TFLite variants and benchmark size, accuracy, latency")
    pr = sub.add_parser("predict", help="Predict soil type for one or more images")
    pr.add_argument("images", nargs="+")
    pr.add_argument("--model", default=None, help=".keras or .tflite path")

    args = p.parse_args(argv)
    cfg = load_config(args.config, parse_set(args.set))
    handlers = {"audit": cmd_audit, "baseline": cmd_baseline, "train": cmd_train,
                "evaluate": cmd_evaluate, "export": cmd_export, "predict": cmd_predict}
    handlers[args.command](cfg, args)


if __name__ == "__main__":
    main(sys.argv[1:])
