"""numba building blocks shared by the engines.

Volumes are indexed [i, j, k] (array order, as nibabel returns them). "Index space" means
continuous array indices; voxel centres sit on integers.
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit, prange, set_num_threads

THREADS = int(os.environ.get("ANATOMESH_THREADS", min(8, os.cpu_count() or 1)))
set_num_threads(max(1, THREADS))


@njit(cache=True, inline="always")
def trilinear(vol, x, y, z, outside):
    """Trilinear value of vol at index (x, y, z); `outside` when any corner is off the grid."""
    nx, ny, nz = vol.shape
    if not (x >= 0.0 and y >= 0.0 and z >= 0.0 and x <= nx - 1 and y <= ny - 1 and z <= nz - 1):
        return outside
    i0, j0, k0 = int(x), int(y), int(z)
    i1, j1, k1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1), min(k0 + 1, nz - 1)
    fx, fy, fz = x - i0, y - j0, z - k0
    c00 = vol[i0, j0, k0] * (1 - fz) + vol[i0, j0, k1] * fz
    c01 = vol[i0, j1, k0] * (1 - fz) + vol[i0, j1, k1] * fz
    c10 = vol[i1, j0, k0] * (1 - fz) + vol[i1, j0, k1] * fz
    c11 = vol[i1, j1, k0] * (1 - fz) + vol[i1, j1, k1] * fz
    return (c00 * (1 - fy) + c01 * fy) * (1 - fx) + (c10 * (1 - fy) + c11 * fy) * fx


@njit(cache=True, parallel=True)
def sample_points(vol, idx, outside):
    """Trilinear samples of vol at index points idx (n, 3)."""
    out = np.empty(idx.shape[0], dtype=np.float64)
    for p in prange(idx.shape[0]):
        out[p] = trilinear(vol, idx[p, 0], idx[p, 1], idx[p, 2], outside)
    return out


@njit(cache=True, parallel=True)
def _resample_linear(vol, m, shape, outside, out):
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x = m[0, 0] * i + m[0, 1] * j + m[0, 2] * k + m[0, 3]
                y = m[1, 0] * i + m[1, 1] * j + m[1, 2] * k + m[1, 3]
                z = m[2, 0] * i + m[2, 1] * j + m[2, 2] * k + m[2, 3]
                out[i, j, k] = trilinear(vol, x, y, z, outside)


@njit(cache=True, parallel=True)
def _resample_nearest(vol, m, shape, outside, out):
    nx, ny, nz = vol.shape
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x = m[0, 0] * i + m[0, 1] * j + m[0, 2] * k + m[0, 3]
                y = m[1, 0] * i + m[1, 1] * j + m[1, 2] * k + m[1, 3]
                z = m[2, 0] * i + m[2, 1] * j + m[2, 2] * k + m[2, 3]
                a, b, c = int(np.floor(x + 0.5)), int(np.floor(y + 0.5)), int(np.floor(z + 0.5))
                if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
                    out[i, j, k] = vol[a, b, c]
                else:
                    out[i, j, k] = outside


def resample(vol: np.ndarray, vol_affine: np.ndarray, target_affine: np.ndarray,
             target_shape, *, world_to_world: np.ndarray | None = None,
             order: int = 1, outside: float = 0.0, dtype=np.float32) -> np.ndarray:
    """vol (on vol_affine) sampled on the target grid.

    world_to_world maps a target-space world point to the matching vol-space world point
    (identity when None). order 1 = trilinear, 0 = nearest.
    """
    w2w = np.eye(4) if world_to_world is None else np.asarray(world_to_world, float)
    m = np.linalg.inv(np.asarray(vol_affine, float)) @ w2w @ np.asarray(target_affine, float)
    shape = tuple(int(n) for n in target_shape)
    if order == 0:
        out = np.empty(shape, dtype=vol.dtype)
        _resample_nearest(np.ascontiguousarray(vol), m, np.array(shape), vol.dtype.type(outside), out)
        return out.astype(dtype, copy=False)
    out = np.empty(shape, dtype=np.float32)
    _resample_linear(np.ascontiguousarray(vol, dtype=np.float32), m, np.array(shape),
                     np.float32(outside), out)
    return out.astype(dtype, copy=False)


@njit(cache=True, inline="always")
def splitmix64(state):
    """Next value of a counter-based generator (Steele, Lea and Flood 2014): the same seed
    gives the same numbers whatever thread runs it."""
    z = (state + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def uniform01(seed, counter):
    """A uniform number in [0, 1) for (seed, counter)."""
    v = splitmix64(np.uint64(seed) * np.uint64(0x100000001B3) + np.uint64(counter))
    return (v >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold (Otsu 1979): the cut that maximises the between-class variance."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centres)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[int(np.argmax(between)) + 1])
