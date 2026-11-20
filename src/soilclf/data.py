"""Dataset scanning, auditing, de-duplication, and leakage-free splitting."""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    md5: str


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def scan(root: str | Path, classes: list[str], extensions: list[str]) -> list[Sample]:
    """List every image under root/<class>/ in a deterministic order."""
    root = Path(root)
    exts = {e.lower() for e in extensions}
    samples = []
    for label in classes:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                samples.append(Sample(p, label, file_md5(p)))
    return samples


def dedupe_exact(samples: list[Sample]) -> tuple[list[Sample], list[tuple[Sample, Sample]]]:
    """Drop byte-identical copies within the same class. Returns (kept, [(duplicate, kept_original)])."""
    first: dict[tuple[str, str], Sample] = {}
    kept, removed = [], []
    for s in samples:
        key = (s.label, s.md5)
        if key in first:
            removed.append((s, first[key]))
        else:
            first[key] = s
            kept.append(s)
    return kept, removed


def difference_hash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
    """dHash: sign of horizontal gradients on a small grayscale thumbnail (hash_size**2 bits)."""
    a = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).ravel()


def thumbnail_vector(img: Image.Image, size: int = 32) -> np.ndarray:
    """Standardized RGB thumbnail; the dot product / length is a Pearson-style correlation."""
    a = np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def group_near_duplicates(samples: list[Sample], strict_bits: int, loose_bits: int,
                          min_corr: float) -> np.ndarray:
    """Union-find over image fingerprints. Returns one group id per sample.

    Two images are near-duplicates if their dHash differs in <= strict_bits bits, or in
    <= loose_bits bits *and* their thumbnails correlate >= min_corr (re-shot or re-cropped
    photos of the same scene). Thresholds were set by inspecting pairs in the real dataset:
    true duplicates sat at <= 15 bits (or 32 bits with r = 0.985); unrelated photos at >= 30
    bits with r < 0.82.
    """
    n = len(samples)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_md5: dict[str, int] = {}
    for i, s in enumerate(samples):
        if s.md5 in by_md5:
            union(i, by_md5[s.md5])
        else:
            by_md5[s.md5] = i

    hashes, thumbs = [], []
    for s in samples:
        with Image.open(s.path) as img:
            hashes.append(difference_hash(img))
            thumbs.append(thumbnail_vector(img))
    if n:
        H, T = np.stack(hashes), np.stack(thumbs)
        for i in range(n - 1):
            dist = (H[i] != H[i + 1:]).sum(axis=1)
            corr = T[i + 1:] @ T[i] / T.shape[1]
            match = (dist <= strict_bits) | ((dist <= loose_bits) & (corr >= min_corr))
            for j in np.flatnonzero(match):
                union(i, i + 1 + int(j))

    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def label_conflicts(samples: list[Sample], groups: np.ndarray) -> list[list[Sample]]:
    """Groups of (near-)duplicate images that carry more than one class label."""
    out = []
    for g in np.unique(groups):
        members = [samples[i] for i in np.flatnonzero(groups == g)]
        if len({m.label for m in members}) > 1:
            out.append(members)
    return out


def make_splits(labels: np.ndarray, groups: np.ndarray, test_size: float, val_size: float,
                seed: int) -> dict[str, np.ndarray]:
    """Stratified, group-aware train/val/test split. No group appears in two splits."""
    idx = np.arange(len(labels))
    n_test = max(2, round(1 / test_size))
    outer = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest, test = next(outer.split(idx, labels, groups))
    n_val = max(2, round((1 - test_size) / val_size))
    inner = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr, va = next(inner.split(rest, labels[rest], groups[rest]))
    return {"train": np.sort(rest[tr]), "val": np.sort(rest[va]), "test": np.sort(test)}


def build_manifest(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Scan, audit, clean, and split the dataset. Returns (manifest, audit_report)."""
    d = cfg["data"]
    root = Path(d["root"])
    raw = scan(root, d["classes"], d["extensions"])
    kept, removed = dedupe_exact(raw)
    groups = group_near_duplicates(kept, d["near_duplicate_bits"], d["near_duplicate_loose_bits"],
                                   d["near_duplicate_min_corr"])
    conflicts = label_conflicts(kept, groups)

    drop = set()
    if d.get("drop_label_conflicts", True):
        drop = {s.path for grp in conflicts for s in grp}
    keep_idx = [i for i, s in enumerate(kept) if s.path not in drop]
    samples = [kept[i] for i in keep_idx]
    groups = groups[keep_idx]

    class_to_id = {c: i for i, c in enumerate(d["classes"])}
    labels = np.array([class_to_id[s.label] for s in samples])
    splits = make_splits(labels, groups, d["test_size"], d["val_size"], cfg["seed"])

    split_of = np.empty(len(samples), dtype=object)
    for name, ids in splits.items():
        split_of[ids] = name
    manifest = pd.DataFrame({
        "path": [s.path.relative_to(root).as_posix() for s in samples],
        "label": [s.label for s in samples],
        "label_id": labels,
        "group": groups,
        "split": split_of,
        "md5": [s.md5 for s in samples],
    })

    rel = lambda p: Path(p).relative_to(root).as_posix()  # noqa: E731
    report = {
        "root": str(root),
        "files_found": len(raw),
        "files_per_class": dict(Counter(s.label for s in raw)),
        "exact_duplicates_removed": len(removed),
        "exact_duplicates": [{"duplicate": rel(a.path), "original": rel(b.path)} for a, b in removed],
        "near_duplicate_groups": int(sum(1 for c in Counter(groups.tolist()).values() if c > 1)),
        "label_conflicts": [[{"path": rel(s.path), "label": s.label} for s in grp] for grp in conflicts],
        "dropped_for_label_conflict": len(drop),
        "images_used": len(samples),
        "images_used_per_class": dict(Counter(s.label for s in samples)),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "split_class_counts": {
            k: dict(Counter(manifest.loc[v, "label"])) for k, v in splits.items()
        },
        "groups_shared_between_splits": 0,
    }
    shared = set(groups[splits["train"]]) & (set(groups[splits["val"]]) | set(groups[splits["test"]]))
    shared |= set(groups[splits["val"]]) & set(groups[splits["test"]])
    report["groups_shared_between_splits"] = len(shared)
    if shared:
        raise AssertionError("Split leakage: a duplicate group appears in more than one split")
    return manifest, report


def apply_frozen_split(manifest: pd.DataFrame, report: dict, frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reuse the split assignments of a committed manifest.

    StratifiedGroupKFold output can change between scikit-learn versions, so the committed
    manifest, not the library, is the source of truth for which image is in which split.
    The cleaned data must match the manifest exactly (same files, same bytes).
    """
    new = manifest.set_index("path")
    old = frozen.set_index("path")
    if set(new.index) != set(old.index):
        missing, extra = sorted(set(old.index) - set(new.index)), sorted(set(new.index) - set(old.index))
        raise ValueError(f"Dataset differs from the frozen manifest (missing {missing[:5]}, extra {extra[:5]}). "
                         "Re-run with --resplit to create a new split.")
    changed = new.index[new["md5"] != old.loc[new.index, "md5"]]
    if len(changed):
        raise ValueError(f"Image contents changed since the manifest was written: {list(changed[:5])}. "
                         "Re-run with --resplit to create a new split.")
    new["split"] = old.loc[new.index, "split"]
    out = new.reset_index()
    for g, rows in out.groupby("group"):
        if rows["split"].nunique() > 1:
            raise AssertionError(f"Split leakage in frozen manifest: duplicate group {g} spans splits")
    report = dict(report)
    report["split_sizes"] = {k: int((out["split"] == k).sum()) for k in ("train", "val", "test")}
    report["split_class_counts"] = {k: dict(Counter(out.loc[out["split"] == k, "label"]))
                                    for k in ("train", "val", "test")}
    report["split_source"] = "frozen manifest"
    return out, report


def audit_markdown(report: dict) -> str:
    lines = [
        "# Data audit",
        "",
        "| Class | Files found | Used after cleaning |",
        "|---|---|---|",
    ]
    for c, n in report["files_per_class"].items():
        lines.append(f"| {c} | {n} | {report['images_used_per_class'].get(c, 0)} |")
    lines += [
        f"| **Total** | **{report['files_found']}** | **{report['images_used']}** |",
        "",
        f"- Exact duplicates removed: **{report['exact_duplicates_removed']}**",
        f"- Near-duplicate groups kept together in one split: **{report['near_duplicate_groups']}**",
        f"- Images dropped because the same photo had different labels: **{report['dropped_for_label_conflict']}**",
        f"- Split sizes: {report['split_sizes']}",
        f"- Duplicate groups shared between splits: **{report['groups_shared_between_splits']}**",
    ]
    if report["label_conflicts"]:
        lines += ["", "## Label conflicts", ""]
        for grp in report["label_conflicts"]:
            lines.append("- " + "; ".join(f"`{m['path']}` ({m['label']})" for m in grp))
    return "\n".join(lines) + "\n"
