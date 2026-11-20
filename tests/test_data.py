import numpy as np

from soilclf.data import build_manifest, dedupe_exact, make_splits, scan


def test_scan_and_exact_dedupe(cfg):
    d = cfg["data"]
    samples = scan(d["root"], d["classes"], d["extensions"])
    assert len(samples) == 3 * 14 + 3
    kept, removed = dedupe_exact(samples)
    assert len(removed) == 2  # the two "Copy of" files; cross-class copy is not a within-class duplicate
    assert all(dup.path.name.startswith("Copy of") for dup, _ in removed)


def test_manifest_flags_conflict_and_has_no_leakage(cfg):
    manifest, report = build_manifest(cfg)
    assert report["exact_duplicates_removed"] == 2
    assert len(report["label_conflicts"]) == 1
    assert report["dropped_for_label_conflict"] == 2
    assert len(manifest) == 3 * 14 + 3 - 2 - 2
    assert set(manifest["split"]) == {"train", "val", "test"}
    for g, rows in manifest.groupby("group"):
        assert rows["split"].nunique() == 1, f"group {g} spans splits"
    for split in ("val", "test"):
        assert manifest.loc[manifest["split"] == split, "label"].nunique() == 3


def test_splits_are_deterministic_and_disjoint():
    labels = np.repeat([0, 1, 2], 20)
    groups = np.arange(60) // 2  # pairs of near-duplicates
    a = make_splits(labels, groups, 0.15, 0.15, seed=7)
    b = make_splits(labels, groups, 0.15, 0.15, seed=7)
    for k in a:
        assert np.array_equal(a[k], b[k])
    all_idx = np.concatenate(list(a.values()))
    assert len(all_idx) == len(set(all_idx)) == 60
    assert not set(groups[a["train"]]) & set(groups[a["test"]])
