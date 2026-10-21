"""Seat the finished head on its own midsagittal plane."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage

from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

_MAX_SAMPLES = 20000

_SLOPE_SPAN = 0.15
_OFFSET_SPAN_MM = 15.0

_MIN_SYMMETRY = 0.55


def _brain_mask(labels: np.ndarray) -> np.ndarray:
    """Cerebrum if there is one, whole brain otherwise."""
    from engine.mesh.membranes import _BRAIN, _CEREBRUM

    mask = np.isin(labels, _CEREBRUM)
    if mask.sum() < 1000:
        mask = np.isin(labels, _BRAIN)
    return mask


def _plane_normal(a: float, b: float) -> np.ndarray:
    n = np.array([1.0, -a, -b])
    return n / np.linalg.norm(n)


def _cache_key(labels: np.ndarray, affine: np.ndarray) -> tuple:
    labels = np.ascontiguousarray(labels)
    return (labels.shape, labels.dtype.str, hash(labels.tobytes()),
            np.asarray(affine, dtype=float).tobytes())


_PLANE_CACHE: dict[tuple, Optional[tuple[np.ndarray, np.ndarray, float]]] = {}


def midsagittal_plane(labels: np.ndarray,
                      affine: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Best mirror-symmetry sagittal plane, in world mm."""
    labels = np.asarray(labels)
    key = _cache_key(labels, affine)
    if key not in _PLANE_CACHE:
        _PLANE_CACHE[key] = _fit_midsagittal_plane(labels, affine)
    return _PLANE_CACHE[key]


def _fit_midsagittal_plane(
        labels: np.ndarray,
        affine: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    labels = np.asarray(labels)
    affine = np.asarray(affine, dtype=float)
    mask = _brain_mask(labels)
    if not mask.any():
        return None

    vox = np.argwhere(mask)
    if len(vox) > _MAX_SAMPLES:
        vox = vox[:: int(np.ceil(len(vox) / _MAX_SAMPLES))]
    pts = vox @ affine[:3, :3].T + affine[:3, 3]
    centre = pts.mean(axis=0)

    smooth = ndimage.gaussian_filter(mask.astype(np.float32), 1.0)
    inv = np.linalg.inv(affine[:3, :3])
    origin = affine[:3, 3]

    def symmetry(a: float, b: float, c: float) -> float:
        n = _plane_normal(a, b)
        p0 = np.array([c, centre[1], centre[2]])
        d = (pts - p0) @ n
        mirrored = pts - 2.0 * d[:, None] * n
        idx = (mirrored - origin) @ inv.T
        return float(fast_kernels.sample_trilinear(smooth, idx.T).mean())

    best = (symmetry(0.0, 0.0, float(centre[0])), 0.0, 0.0, float(centre[0]))
    for a in np.arange(-_SLOPE_SPAN, _SLOPE_SPAN + 1e-9, 0.03):
        for b in np.arange(-_SLOPE_SPAN, _SLOPE_SPAN + 1e-9, 0.03):
            for c in np.arange(centre[0] - _OFFSET_SPAN_MM,
                               centre[0] + _OFFSET_SPAN_MM + 1e-9, 2.0):
                s = symmetry(float(a), float(b), float(c))
                if s > best[0]:
                    best = (s, float(a), float(b), float(c))

    from scipy.optimize import minimize

    polished = minimize(lambda p: -symmetry(p[0], p[1], p[2]),
                        x0=np.array(best[1:]), method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 600})
    if -float(polished.fun) > best[0]:
        best = (-float(polished.fun), *(float(v) for v in polished.x))

    score, a, b, c = best
    logger.info("midsagittal symmetry plane: yaw %+.3f, roll %+.3f, x=%.2fmm, symmetry %.3f",
                a, b, c, score)
    return np.array([c, centre[1], centre[2]]), _plane_normal(a, b), score


def _rotation_onto_x(normal: np.ndarray) -> np.ndarray:
    """Smallest rotation taking `normal` onto +x."""
    target = np.array([1.0, 0.0, 0.0])
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if n[0] < 0:
        n = -n
    v = np.cross(n, target)
    s, c = float(np.linalg.norm(v)), float(n @ target)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def upright_transform(labels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """4x4 rigid transform putting this subject's midsagittal plane on x=0."""
    found = midsagittal_plane(labels, affine)
    if found is None:
        return np.eye(4)
    point, normal, score = found
    if score < _MIN_SYMMETRY:
        logger.warning("midsagittal symmetry only %.3f, under the %.2f floor; leaving the head "
                       "in the scanner frame", score, _MIN_SYMMETRY)
        return np.eye(4)

    rot = _rotation_onto_x(normal)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[0, 3] = -float((rot @ point)[0])
    return transform


def apply_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points
    return points @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def apply_vectors(transform: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Directions (fibre axes, normals) rotate and do not translate."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.size == 0:
        return vectors
    return vectors @ np.asarray(transform)[:3, :3].T


def report(transform: np.ndarray, labels: np.ndarray, affine: np.ndarray) -> dict:
    """What the transform did and what is left over, for `metrics.json`."""
    transform = np.asarray(transform, dtype=float)
    rot = transform[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))
    out = {
        "applied": bool(not np.allclose(transform, np.eye(4))),
        "rotation_deg": round(angle, 3),
        "shift_mm": round(float(transform[0, 3]), 3),
    }
    found = midsagittal_plane(labels, transform @ np.asarray(affine, dtype=float))
    if found is not None:
        point, normal, score = found
        out["residual_offset_mm"] = round(float(point[0]), 3)
        out["residual_tilt_deg"] = round(
            float(np.degrees(np.arccos(np.clip(abs(float(normal[0])), -1.0, 1.0)))), 3)
        out["symmetry"] = round(float(score), 4)
    return out


def demo() -> None:
    """A deliberately mis-seated symmetric phantom has to come back straight."""
    shape = (96, 96, 96)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    centre = np.array([48.0, 48.0, 48.0])
    scaled = (grid - centre) * np.array([1.0, 0.7, 1.0])
    r = np.linalg.norm(scaled, axis=-1)
    mask = (r < 30) & (np.abs(grid[..., 0] - 48) > 2) & (grid[..., 1] > 24)
    labels = np.where(mask, 3, 0).astype(np.int16)

    yaw = np.deg2rad(5.0)
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    affine = np.eye(4)
    affine[:3, :3] = rot
    affine[:3, 3] = rot @ (-centre) + np.array([6.0, 0.0, 0.0])

    transform = upright_transform(labels, affine)
    assert not np.allclose(transform, np.eye(4)), "a mis-seated phantom must move"
    stats = report(transform, labels, affine)
    assert abs(stats["residual_offset_mm"]) < 1.0, stats
    assert stats["residual_tilt_deg"] < 1.0, stats

    pts = np.random.default_rng(0).normal(size=(50, 3)) * 20.0
    moved = apply_points(transform, pts)
    assert np.allclose(np.linalg.norm(pts[1:] - pts[:-1], axis=1),
                       np.linalg.norm(moved[1:] - moved[:-1], axis=1)), "must be rigid"
    assert np.allclose(upright_transform(np.zeros(shape, np.int16), np.eye(4)), np.eye(4))
    print("orientation demo ok:", stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
