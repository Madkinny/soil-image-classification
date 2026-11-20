import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from soilclf.config import load_config

CLASSES = ["Alluvial soil", "Black Soil", "Clay soil"]
TINTS = {"Alluvial soil": (200, 170, 120), "Black Soil": (50, 40, 35), "Clay soil": (170, 90, 60)}


def make_image(path: Path, tint, seed: int, size: int = 64) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 35, (size, size, 3))
    img = np.clip(np.array(tint, dtype=float) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Synthetic 3-class dataset with known duplicates and one cross-label conflict."""
    root = tmp_path / "Datasets"
    seed = 0
    for label in CLASSES:
        d = root / label
        d.mkdir(parents=True)
        for i in range(14):
            make_image(d / f"{label[:3]}_{i}.png", TINTS[label], seed)
            seed += 1
    # exact copies within a class (should be removed)
    shutil.copy(root / "Black Soil" / "Bla_0.png", root / "Black Soil" / "Copy of Bla_0.png")
    shutil.copy(root / "Clay soil" / "Cla_0.png", root / "Clay soil" / "Copy of Cla_0.png")
    # the same photo filed under two labels (should be flagged and dropped)
    shutil.copy(root / "Alluvial soil" / "All_0.png", root / "Clay soil" / "mislabeled.png")
    return root


@pytest.fixture
def cfg(dataset: Path, tmp_path: Path) -> dict:
    return load_config(overrides={
        "data.root": str(dataset),
        "data.image_size": 64,
        "paths.artifacts": str(tmp_path / "artifacts"),
        "paths.reports": str(tmp_path / "reports"),
        "evaluate.bootstrap_samples": 200,
        "baseline.rf_n_estimators": 50,
    })
