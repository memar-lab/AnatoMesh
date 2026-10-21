"""Two-stage SDF smoothing of label volumes."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from engine.mesh import voxel_hex
from engine.paths import READ_WRITE_ROOT, assert_write_path, outputs_root
from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

THREADS = voxel_hex.THREADS


CSF_LABEL = 100
BRAIN_LABEL = 1
REMAINING_BRAIN_LABEL = 101
SIGMA_OUTER_MM = 0.3
ITERATIONS_OUTER = 3
SIGMA_INNER_MM = 0.4
ITERATIONS_INNER = 5
MESH_ITERATIONS = 3


def add_csf(segmentation: np.ndarray, csf: np.ndarray) -> np.ndarray:
    """adding the CSF"""
    out = np.asarray(csf).copy()
    idx = np.nonzero(np.asarray(segmentation) != 0)
    inside = np.all([i < n for i, n in zip(idx, out.shape)], axis=0)
    idx = tuple(i[inside] for i in idx)
    out[idx] = np.asarray(segmentation)[idx].astype(out.dtype)
    return out


def fill_holes(seg: np.ndarray, value: int = CSF_LABEL) -> np.ndarray:
    """filling holes"""
    clusters, _ = ndi.label(seg == 0, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(clusters.ravel())
    sizes[0] = 0
    out = np.asarray(seg).copy()
    out[(clusters > 0) & (clusters != sizes.argmax())] = value
    return out


def combine_brain(seg: np.ndarray) -> np.ndarray:
    """combining the brain regions"""
    out = np.asarray(seg).copy()
    out[(out > 0) & (out < CSF_LABEL)] = BRAIN_LABEL
    return out


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """`imgaussfilt3` default kernel"""
    radius = int(np.ceil(2 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    h = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return h / h.sum()


def imgaussfilt3(volume: np.ndarray, sigma_vox) -> np.ndarray:
    """Separable Gaussian in single precision with replicate padding, as `imgaussfilt3`."""
    out = np.asarray(volume, dtype=np.float32)
    for axis, sigma in enumerate(np.broadcast_to(np.asarray(sigma_vox, dtype=np.float64), (3,))):
        out = ndi.correlate1d(out, _gaussian_kernel(float(sigma)).astype(np.float32),
                              axis=axis, mode="nearest", output=np.float32)
    return out


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """`bwdist(~mask) - bwdist(mask)` in single"""
    with ThreadPoolExecutor(2) as pool:
        inside, outside = pool.map(lambda m: fast_kernels.distance_transform_edt(m).astype(np.float32),
                                   (mask, ~mask))
    return inside - outside


def smooth_segmentation_sdf(seg: np.ndarray, voxel_mm, *,
                            sigma_outer_mm: float = SIGMA_OUTER_MM,
                            iterations_outer: int = ITERATIONS_OUTER,
                            sigma_inner_mm: float = SIGMA_INNER_MM,
                            iterations_inner: int = ITERATIONS_INNER) -> tuple[np.ndarray, dict]:
    """SDF segmentation smoothing.

    Computed on the box around the nonzero voxels, grown by how far the smoothing can reach
    (one kernel radius per outer pass, one for the inner pass, plus two): every voxel that can
    end up nonzero sees the same distances and the same kernel values as on the whole lattice
    (the voxels outside the box are background, and background a kernel radius past the box
    edge stays background), so the result is identical; a full head fills about half the
    256^3 lattice."""
    original = np.asarray(seg)
    voxel_mm = np.asarray(voxel_mm, dtype=np.float64)
    nonzero = np.argwhere(original != 0)
    if len(nonzero):
        def reach(sigma_mm, passes):
            if sigma_mm <= 0 or passes <= 0:
                return np.zeros(3, dtype=int)
            return np.array([int(np.ceil(2 * s)) for s in
                             np.broadcast_to(sigma_mm / voxel_mm, (3,))]) * passes
        margin = reach(sigma_outer_mm, iterations_outer) + reach(sigma_inner_mm, 1) + 2
        lo = np.maximum(nonzero.min(axis=0) - margin, 0)
        hi = np.minimum(nonzero.max(axis=0) + margin + 1, original.shape)
        if np.prod(hi - lo) < 0.9 * original.size:
            box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            inner, report = _smooth_segmentation_sdf(
                original[box], voxel_mm, sigma_outer_mm=sigma_outer_mm,
                iterations_outer=iterations_outer, sigma_inner_mm=sigma_inner_mm,
                iterations_inner=iterations_inner)
            out = np.zeros_like(original)
            out[box] = inner
            return out, report
    return _smooth_segmentation_sdf(original, voxel_mm, sigma_outer_mm=sigma_outer_mm,
                                    iterations_outer=iterations_outer,
                                    sigma_inner_mm=sigma_inner_mm,
                                    iterations_inner=iterations_inner)


def _label_counts(volume: np.ndarray) -> dict:
    values, counts = np.unique(volume[volume != 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _smooth_segmentation_sdf(original, voxel_mm, *, sigma_outer_mm, iterations_outer,
                             sigma_inner_mm, iterations_inner):
    original = np.asarray(original)
    voxel_mm = np.asarray(voxel_mm, dtype=np.float64)
    union = original != 0
    mask = union.copy()
    passes = [("input", _label_counts(original))]
    if sigma_outer_mm > 0 and iterations_outer > 0:
        for k in range(iterations_outer):
            mask = imgaussfilt3(_signed_distance(mask), sigma_outer_mm / voxel_mm) > 0
            passes.append((f"outer {k + 1}", {0: int(mask.sum())}))

    after_outer = original.copy()
    lost = union & ~mask
    gained = ~union & mask
    after_outer[lost] = 0
    if gained.any():
        _, nearest = ndi.distance_transform_edt(~union, return_indices=True)
        after_outer[gained] = original[tuple(n[gained] for n in nearest)]

    current = after_outer
    passes.append(("after outer", _label_counts(after_outer)))
    if sigma_inner_mm > 0 and iterations_inner > 0:
        for k in range(iterations_inner):
            best_value = np.full(current.shape, -np.inf, dtype=np.float32)
            best_label = current.copy()
            labels = [label for label in np.unique(current) if label != 0]
            sigma = sigma_inner_mm / voxel_mm
            with ThreadPoolExecutor(max(1, THREADS // 2)) as pool:
                fields = pool.map(lambda label: imgaussfilt3(_signed_distance(current == label), sigma),
                                  labels)
                for label, smoothed in zip(labels, fields):      # combined in label order
                    win = mask & (smoothed > best_value)
                    best_value[win] = smoothed[win]
                    best_label[win] = label
            best_label[~mask] = 0
            current = best_label
            passes.append((f"inner {k + 1}", _label_counts(current)))

    report = {"voxels_gained": int(gained.sum()), "voxels_lost": int(lost.sum()),
              "internal_voxels_changed": int((mask & (after_outer != current)).sum()),
              "passes": passes}
    return current, report


def overlay_regions(smoothed: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """restoring the brain regions"""
    if smoothed.shape != regions.shape:
        raise ValueError(f"volumes differ in shape: {smoothed.shape} vs {regions.shape}")
    out = np.asarray(smoothed).copy()
    out[out == BRAIN_LABEL] = REMAINING_BRAIN_LABEL
    take = (out == REMAINING_BRAIN_LABEL) & (regions > 0) & (regions < 99)
    out[take] = regions[take]
    return out
