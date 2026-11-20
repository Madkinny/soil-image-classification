import numpy as np
from PIL import Image

from soilclf import features
from soilclf.images import load_rgb
from soilclf.metrics import classification_metrics


def test_load_rgb_handles_alpha_and_grayscale(tmp_path):
    Image.new("RGBA", (40, 30), (10, 20, 30, 0)).save(tmp_path / "a.png")
    Image.new("L", (30, 40), 128).save(tmp_path / "g.png")
    for name in ("a.png", "g.png"):
        x = load_rgb(tmp_path / name, 32)
        assert x.shape == (32, 32, 3) and x.dtype == np.uint8


def test_features_fixed_length_finite_deterministic():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
    f = features.extract_batch(imgs)
    assert f.shape[0] == 3 and np.isfinite(f).all()
    assert np.array_equal(f, features.extract_batch(imgs))
    flat = np.full((64, 64, 3), 128, np.uint8)
    assert np.isfinite(features.extract(flat)).all()


def test_metrics_perfect_and_ci_bounds():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, y, ["a", "b", "c"], n_bootstrap=100, seed=0)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["accuracy_ci95"] == (1.0, 1.0)
    m = classification_metrics(y, np.zeros_like(y), ["a", "b", "c"], n_bootstrap=100, seed=0)
    lo, hi = m["accuracy_ci95"]
    assert 0.0 <= lo <= m["accuracy"] <= hi <= 1.0
    assert np.array(m["confusion_matrix"]).sum() == len(y)
