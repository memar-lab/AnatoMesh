"""Deterministic streamline tractography (replaces DSI Studio's fiber tracking).

Method: streamline propagation along the principal diffusion direction (Mori et al. 1999,
doi:10.1002/1531-8249(199902)45:2<265::AID-ANA21>3.0.CO;2-3; Basser et al. 2000,
doi:10.1002/1522-2594(200010)44:4<625::AID-MRM17>3.0.CO;2-O), with the direction at a point
interpolated from the eight surrounding voxels after each voxel's axis is turned to agree with
the incoming direction, and voxels that are under threshold or would turn too sharply left out
(the interpolation of Yeh et al. 2013, doi:10.1371/journal.pone.0080713). Termination: FA under
threshold, turn over the angle limit, outside the mask, or longer than the maximum. The default
FA threshold is a share of Otsu's threshold of FA in the mask (the DSI Studio convention).

Seeds are random points in the seed region, drawn in fixed-size batches from a seeded generator
and tracked in parallel; accepted streamlines are kept in seed order, so the same settings give
the same tractogram on any number of threads.

Coordinates: `e1` is in the array-axis frame (x along the first array index) with millimetre
units, i.e. the frame of a tensor fitted with array-frame b-vectors (engine.engines.dti). Points are
tracked in array index space and returned in world millimetres through the affine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange

from engine.engines import kernels

logger = logging.getLogger(__name__)


@dataclass
class TrackingSettings:
    tract_count: int = 100_000          # accepted streamlines wanted
    seed_limit_factor: float = 50.0     # give up after tract_count x this many seeds
    step_mm: float = 1.0                # integration step
    angle_deg: float = 45.0             # largest turn per step
    fa_threshold: Optional[float] = None  # None: otsu_scale x Otsu threshold of FA in the mask
    otsu_scale: float = 0.6             # share of Otsu's threshold used when fa_threshold is None
    min_length_mm: float = 20.0         # shorter streamlines are discarded
    max_length_mm: float = 300.0        # longer streamlines are discarded
    smoothing: float = 0.0              # share of the previous direction kept at each step (0..1)
    integrator: str = "euler"           # "euler" or "rk2" (midpoint)
    seed: int = 0                       # random seed; same seed, same tractogram
    batch_mb: float = 64.0              # working memory per seed batch


@njit(cache=True, inline="always")
def _direction(dirs, fa, mask, x, y, z, vx, vy, vz, thr, cos_max, use_mask):
    """Interpolated direction at index (x, y, z) agreeing with (vx, vy, vz); also the trilinear
    FA there. Returns (dx, dy, dz, fa, ok)."""
    nx, ny, nz = fa.shape
    if not (x >= 0.0 and y >= 0.0 and z >= 0.0 and x <= nx - 1 and y <= ny - 1 and z <= nz - 1):
        return 0.0, 0.0, 0.0, 0.0, False
    i0, j0, k0 = int(x), int(y), int(z)
    fx, fy, fz = x - i0, y - j0, z - k0
    sx = sy = sz = 0.0
    fa_here = 0.0
    wsum = 0.0
    for c in range(8):
        di, dj, dk = (c >> 2) & 1, (c >> 1) & 1, c & 1
        i, j, k = min(i0 + di, nx - 1), min(j0 + dj, ny - 1), min(k0 + dk, nz - 1)
        w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy) * (fz if dk else 1.0 - fz)
        f = fa[i, j, k]
        fa_here += w * f
        if w <= 0.0 or f < thr:
            continue
        if use_mask and not mask[i, j, k]:
            continue
        ex, ey, ez = dirs[i, j, k, 0], dirs[i, j, k, 1], dirs[i, j, k, 2]
        dot = ex * vx + ey * vy + ez * vz
        if dot < 0.0:
            ex, ey, ez, dot = -ex, -ey, -ez, -dot
        if dot < cos_max:
            continue
        sx += w * ex
        sy += w * ey
        sz += w * ez
        wsum += w
    norm = np.sqrt(sx * sx + sy * sy + sz * sz)
    if wsum <= 0.0 or norm <= 1e-12 or fa_here < thr:
        return 0.0, 0.0, 0.0, fa_here, False
    return sx / norm, sy / norm, sz / norm, fa_here, True


@njit(cache=True)
def _half(dirs, fa, mask, px, py, pz, vx, vy, vz, inv_vox, step, thr, cos_max, use_mask,
          smoothing, rk2, max_steps, out, sign_slot, centre):
    """Track from (px, py, pz) along (vx, vy, vz); writes points at centre + sign_slot * n.
    Returns the number of steps taken, or -1 when the half ran past max_steps."""
    n = 0
    nx, ny, nz = fa.shape
    while True:
        if n >= max_steps:
            return -1
        if rk2:
            ax, ay, az, _f, ok = _direction(dirs, fa, mask, px, py, pz, vx, vy, vz,
                                            thr, cos_max, use_mask)
            if not ok:
                return n
            mx = px + 0.5 * step * ax * inv_vox[0]
            my = py + 0.5 * step * ay * inv_vox[1]
            mz = pz + 0.5 * step * az * inv_vox[2]
            dx, dy, dz, _f, ok = _direction(dirs, fa, mask, mx, my, mz, ax, ay, az,
                                            thr, cos_max, use_mask)
        else:
            dx, dy, dz, _f, ok = _direction(dirs, fa, mask, px, py, pz, vx, vy, vz,
                                            thr, cos_max, use_mask)
        if not ok:
            return n
        if smoothing > 0.0:
            dx = (1.0 - smoothing) * dx + smoothing * vx
            dy = (1.0 - smoothing) * dy + smoothing * vy
            dz = (1.0 - smoothing) * dz + smoothing * vz
            norm = np.sqrt(dx * dx + dy * dy + dz * dz)
            dx, dy, dz = dx / norm, dy / norm, dz / norm
        if dx * vx + dy * vy + dz * vz < cos_max:
            return n
        px = px + step * dx * inv_vox[0]
        py = py + step * dy * inv_vox[1]
        pz = pz + step * dz * inv_vox[2]
        if use_mask:
            a, b, c = int(np.floor(px + 0.5)), int(np.floor(py + 0.5)), int(np.floor(pz + 0.5))
            if a < 0 or b < 0 or c < 0 or a >= nx or b >= ny or c >= nz or not mask[a, b, c]:
                return n
        n += 1
        slot = centre + sign_slot * n
        out[slot, 0], out[slot, 1], out[slot, 2] = px, py, pz
        vx, vy, vz = dx, dy, dz


@njit(cache=True, parallel=True)
def _track_batch(dirs, fa, mask, seeds, inv_vox, step, thr, cos_max, use_mask, smoothing, rk2,
                 max_steps, min_steps, out, lo, hi):
    centre = max_steps
    for s in prange(seeds.shape[0]):
        px, py, pz = seeds[s, 0], seeds[s, 1], seeds[s, 2]
        lo[s] = 0
        hi[s] = 0
        a, b, c = int(np.floor(px + 0.5)), int(np.floor(py + 0.5)), int(np.floor(pz + 0.5))
        vx, vy, vz = dirs[a, b, c, 0], dirs[a, b, c, 1], dirs[a, b, c, 2]
        if vx * vx + vy * vy + vz * vz < 0.5:
            continue
        # the seed's own direction, interpolated, decides both halves
        dx, dy, dz, _f, ok = _direction(dirs, fa, mask, px, py, pz, vx, vy, vz,
                                        thr, -1.0, use_mask)
        if not ok:
            continue
        o = out[s]
        o[centre, 0], o[centre, 1], o[centre, 2] = px, py, pz
        nf = _half(dirs, fa, mask, px, py, pz, dx, dy, dz, inv_vox, step, thr, cos_max,
                   use_mask, smoothing, rk2, max_steps, o, 1, centre)
        if nf < 0:
            continue
        nb = _half(dirs, fa, mask, px, py, pz, -dx, -dy, -dz, inv_vox, step, thr, cos_max,
                   use_mask, smoothing, rk2, max_steps, o, -1, centre)
        if nb < 0 or nf + nb < min_steps or nf + nb > max_steps:
            continue
        lo[s] = centre - nb
        hi[s] = centre + nf + 1


def fa_threshold(fa: np.ndarray, mask: Optional[np.ndarray], settings: TrackingSettings) -> float:
    """The FA threshold in use: the setting, or a share of Otsu's threshold of FA in the mask."""
    if settings.fa_threshold is not None:
        return float(settings.fa_threshold)
    values = fa[mask] if mask is not None else fa[fa > 0]
    return float(settings.otsu_scale) * kernels.otsu_threshold(values[values > 0])


def track(fa: np.ndarray, e1: np.ndarray, affine: np.ndarray,
          mask: Optional[np.ndarray] = None, seed_mask: Optional[np.ndarray] = None,
          settings: Optional[TrackingSettings] = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Whole-brain deterministic tracking.

    Returns (points (n, 3) world mm, lengths (n_streamlines,) points per streamline, report).
    """
    settings = settings or TrackingSettings()
    t0 = time.time()
    fa = np.ascontiguousarray(fa, dtype=np.float32)
    e1 = np.ascontiguousarray(e1, dtype=np.float32)
    if e1.shape[:3] != fa.shape:
        raise ValueError(f"e1 {e1.shape[:3]} does not match FA {fa.shape}")
    affine = np.asarray(affine, dtype=float)
    vox = np.linalg.norm(affine[:3, :3], axis=0)
    norms = np.linalg.norm(e1, axis=-1, keepdims=True)
    dirs = np.divide(e1, norms, out=np.zeros_like(e1), where=norms > 1e-8)
    use_mask = mask is not None
    mask_arr = (np.ascontiguousarray(mask, dtype=np.bool_) if use_mask
                else np.ones((1, 1, 1), dtype=np.bool_))
    thr = fa_threshold(fa, mask if use_mask else None, settings)
    seedable = fa >= thr
    if use_mask:
        seedable &= mask_arr
    if seed_mask is not None:
        seedable &= np.asarray(seed_mask, bool)
    voxels = np.argwhere(seedable).astype(np.float64)
    step = float(settings.step_mm)
    max_steps = int(np.floor(settings.max_length_mm / step))
    min_steps = int(np.ceil(settings.min_length_mm / step))
    cos_max = float(np.cos(np.deg2rad(settings.angle_deg)))
    report = {"engine": "native", "method": "deterministic, interpolated principal direction",
              "fa_threshold": round(thr, 5), "seedable_voxels": int(len(voxels)),
              "settings": {k: getattr(settings, k) for k in settings.__dataclass_fields__}}
    if not len(voxels):
        report["streamlines"] = 0
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), report

    width = 2 * max_steps + 1
    batch = int(max(256, min(65536, settings.batch_mb * 1e6 / (width * 3 * 4))))
    out = np.empty((batch, width, 3), dtype=np.float32)
    lo = np.empty(batch, dtype=np.int64)
    hi = np.empty(batch, dtype=np.int64)
    rng = np.random.default_rng(int(settings.seed))
    want = int(settings.tract_count)
    limit = int(max(want * settings.seed_limit_factor, batch))
    kept_pts, kept_len = [], []
    accepted = seeds_used = 0
    inv_vox = (1.0 / vox).astype(np.float64)
    while accepted < want and seeds_used < limit:
        pick = rng.integers(0, len(voxels), size=batch)
        seeds = voxels[pick] + rng.uniform(-0.5, 0.5, size=(batch, 3))
        _track_batch(dirs, fa, mask_arr, seeds, inv_vox, step, thr, cos_max, use_mask,
                     float(settings.smoothing), settings.integrator == "rk2",
                     max_steps, min_steps, out, lo, hi)
        good = np.flatnonzero(hi > lo)
        if len(good) > want - accepted:
            good = good[:want - accepted]
        for s in good:
            kept_pts.append(out[s, lo[s]:hi[s]].astype(np.float64))
            kept_len.append(int(hi[s] - lo[s]))
        accepted += len(good)
        seeds_used += batch
    points = (np.concatenate(kept_pts) if kept_pts else np.zeros((0, 3)))
    points = points @ affine[:3, :3].T + affine[:3, 3]
    lengths = np.asarray(kept_len, dtype=np.int64)
    report.update(streamlines=int(accepted), seeds=int(seeds_used),
                  acceptance=round(accepted / max(seeds_used, 1), 4),
                  seconds=round(time.time() - t0, 2))
    if accepted < want:
        logger.warning("tracking: %d of %d streamlines after %d seeds (seed limit)",
                       accepted, want, seeds_used)
    logger.info("tracking: %d streamlines from %d seeds in %.1f s (FA threshold %.3f)",
                accepted, seeds_used, report["seconds"], thr)
    return points, lengths, report


def split(points: np.ndarray, lengths: np.ndarray) -> list:
    """Ragged (points, lengths) as a list of (n_i, 3) arrays (views)."""
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    return [points[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def density_map(points: np.ndarray, lengths: np.ndarray, affine: np.ndarray, shape) -> np.ndarray:
    """Streamline count per voxel (each streamline counted once per voxel it visits)."""
    inv = np.linalg.inv(np.asarray(affine, float))
    idx = np.rint(points @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
    shape = tuple(int(n) for n in shape)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    owner = np.repeat(np.arange(len(lengths)), lengths)
    flat = np.ravel_multi_index(tuple(idx[ok].T), shape)
    pairs = np.unique(np.stack([owner[ok], flat], axis=1), axis=0)
    return np.bincount(pairs[:, 1], minlength=int(np.prod(shape))).reshape(shape)


def demo() -> None:
    """Synthetic check: a straight bundle and a 90-degree arc are followed to their ends."""
    shape = (40, 40, 40)
    fa = np.zeros(shape, np.float32)
    e1 = np.zeros(shape + (3,), np.float32)
    fa[18:22, 18:22, 2:38] = 0.8
    e1[18:22, 18:22, 2:38, 2] = 1.0
    pts, lens, rep = track(fa, e1, np.eye(4), settings=TrackingSettings(
        tract_count=200, fa_threshold=0.3, step_mm=0.5, min_length_mm=10, max_length_mm=100))
    lines = split(pts, lens)
    spans = np.array([abs(l[-1, 2] - l[0, 2]) for l in lines])
    assert len(lines) == 200, len(lines)
    assert spans.min() > 33.0 and spans.max() < 37.5, (spans.min(), spans.max())
    # quarter circle of radius 15 around (20, 20) in the x-z plane
    fa2 = np.zeros(shape, np.float32)
    e2 = np.zeros(shape + (3,), np.float32)
    ii, kk = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    r = np.hypot(ii - 5.0, kk - 5.0)
    ring = (np.abs(r - 20) < 2.5) & (ii >= 5) & (kk >= 5)
    for j in range(18, 23):
        fa2[:, j, :][ring] = 0.8
        e2[:, j, :, 0][ring] = (-(kk - 5.0) / np.maximum(r, 1e-6))[ring]
        e2[:, j, :, 2][ring] = ((ii - 5.0) / np.maximum(r, 1e-6))[ring]
    pts2, lens2, _ = track(fa2, e2, np.eye(4), settings=TrackingSettings(
        tract_count=100, fa_threshold=0.3, step_mm=0.5, min_length_mm=20, max_length_mm=100,
        integrator="rk2"))
    arc = split(pts2, lens2)
    radii = np.concatenate([np.hypot(l[:, 0] - 5.0, l[:, 2] - 5.0) for l in arc])
    assert len(arc) == 100 and np.abs(radii - 20).max() < 2.6, np.abs(radii - 20).max()
    again, _l, _r = track(fa2, e2, np.eye(4), settings=TrackingSettings(
        tract_count=100, fa_threshold=0.3, step_mm=0.5, min_length_mm=20, max_length_mm=100,
        integrator="rk2"))
    assert np.array_equal(again, pts2), "tracking is not repeatable"
    print("tracking.demo ok:", len(lines), "straight,", len(arc), "arc, repeatable")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
