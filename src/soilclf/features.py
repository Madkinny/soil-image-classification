"""Handcrafted color and texture features for the classical baseline."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

HSV_BINS = 16
LBP_POINTS, LBP_RADIUS = 8, 1
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 3)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")


def color_histogram(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ranges = ((0, 180), (0, 256), (0, 256))
    feats = []
    for ch, r in enumerate(ranges):
        h = cv2.calcHist([hsv], [ch], None, [HSV_BINS], list(r)).ravel()
        feats.append(h / max(h.sum(), 1.0))
    return np.concatenate(feats)


def color_moments(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    mean, std = lab.mean(0), lab.std(0)
    skew = ((lab - mean) ** 3).mean(0) / np.maximum(std, 1e-6) ** 3
    return np.concatenate([mean / 255.0, std / 255.0, np.tanh(skew)])


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return h / max(h.sum(), 1)


def glcm_features(gray: np.ndarray) -> np.ndarray:
    q = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    glcm = graycomatrix(q, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    # average over angles -> rotation-tolerant texture descriptors per distance
    return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in GLCM_PROPS])


def extract(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.concatenate([
        color_histogram(rgb),
        color_moments(rgb),
        lbp_histogram(gray),
        glcm_features(gray),
    ]).astype(np.float32)


def extract_batch(images: np.ndarray) -> np.ndarray:
    return np.stack([extract(img) for img in images])
