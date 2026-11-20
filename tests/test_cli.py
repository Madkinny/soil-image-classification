import json
from pathlib import Path

import pytest

from soilclf.cli import main


def run(cfg, *args):
    sets = [f"data.root={cfg['data']['root']}", f"data.image_size={cfg['data']['image_size']}",
            f"paths.artifacts={cfg['paths']['artifacts']}", f"paths.reports={cfg['paths']['reports']}",
            "evaluate.bootstrap_samples=100", "baseline.rf_n_estimators=50",
            "train.epochs_head=1", "train.epochs_finetune=1", "train.batch_size=8", "benchmark.runs=3"]
    main([a for kv in sets for a in ("--set", kv)] + list(args))


def test_audit_then_baseline_end_to_end(cfg):
    run(cfg, "audit")
    reports = Path(cfg["paths"]["reports"])
    assert (reports / "manifest.csv").exists()
    assert json.loads((reports / "data_audit.json").read_text())["groups_shared_between_splits"] == 0
    run(cfg, "baseline")
    m = json.loads((reports / "baseline" / "metrics.json").read_text())
    assert m["accuracy"] >= 0.8  # synthetic classes are separable by color
    assert (reports / "baseline" / "confusion_matrix.png").exists()


def test_cnn_train_evaluate_export_predict(cfg, capsys):
    pytest.importorskip("tensorflow")
    run(cfg, "audit")
    run(cfg, "train", "--no-pretrained")
    run(cfg, "evaluate")
    run(cfg, "export")
    art = Path(cfg["paths"]["artifacts"]) / "cnn"
    img = next(Path(cfg["data"]["root"]).rglob("*.png"))
    capsys.readouterr()
    run(cfg, "predict", str(img), "--model", str(art / "model_int8_dynamic.tflite"))
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["prediction"] in cfg["data"]["classes"]
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
