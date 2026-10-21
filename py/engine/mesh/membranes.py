"""Falx cerebri and tentorium cerebelli as structured quad sheets."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from engine.mesh.hex_mesh import _unique_rows
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from engine.mesh import shell_repair
from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

SHEET_REPAIR_REPORTS: dict[str, dict] = {}
FINISHED_SHEETS: set[str] = set()  # sheets build_membranes meshed to final quality (no repair passes)

_GRAY, _WHITE, _BRAINSTEM, _CEREBELLUM, _DEEP, _CSF = 1, 2, 3, 4, 5, 6
_CEREBRUM = (_GRAY, _WHITE, _DEEP)
_BRAIN = (_GRAY, _WHITE, _BRAINSTEM, _CEREBELLUM, _DEEP)
_INTRACRANIAL = _BRAIN + (_CSF,)

_BAND_EPS_MM = 1e-3

FALX_HEIGHT_MM = 34.9
FALX_AREA_CM2 = 56.5
TENTORIUM_WIDTH_MM = 99.64
TENTORIUM_AREA_CM2 = 57.6

FALX_REF_AREA_CM2 = 95.9
FALX_REF_QUADS = 9310
TENTORIUM_REF_AREA_CM2 = 72.6
TENTORIUM_REF_QUADS = 9815

_TENT_HALF_WIDTH_PROFILE = np.array([
    0.486, 0.580, 0.685, 0.747, 0.803, 0.853, 0.895, 0.937, 0.972,
    0.988, 0.991, 0.996, 1.000, 0.986, 0.950, 0.932, 0.883, 0.828,
    0.764, 0.665, 0.611, 0.574, 0.527, 0.486, 0.403,
])

_FALX_FREE_MARGIN_GAP_MM = 10.0  # cingulate gyrus depth under the free margin
_FALX_CC_CLEARANCE_AT_SPLENIUM_MM = 3.0
_FALX_BLOCK_FILL_MM = 8.0
_FALX_HEIGHT_MIN_MM = 2.0
_FALX_SPAN_MIN_MM = 8.0
_FALX_HEIGHT_MAX_MM = 130.0
_FALX_END_ROUND_MM = (12.0, 4.0)  # posterior, anterior (the crista galli tip stays sharp)
_FALX_OPEN_RUN_DROP_MM = 4.0
_FALX_FREE_SMOOTH_MM = (10.0, 2.0)
_FALX_CC_ARC_BLEND_MM = 8.0
_FALX_CC_ARC_SMOOTH_MM = 10.0

_TENTORIUM_AMPLITUDE_MM = 10.0
_TENT_FREE_EDGE_LIFT_MM = 1.5
_TENT_SADDLE_DEPTH_MM = 8.0
_TENT_AP_SMOOTH_MM = 3.0
_TENT_CEILING_TAUT_MM = 8.0
_TENT_FREE_EDGE_DECAY_MM = 9.0
_TENT_FREE_EDGE_RAMP_MM = 12.0
_FALX_STEM_SEAT_MM = 0.0
_TENT_WRAP_START_FRAC = 0.55
_TENT_WRAP_END_FRAC = 1.15
_TENT_POSTERIOR_MIN_FRAC = 0.35
_TENT_POSTERIOR_ROUND_MM = 18.0
_TENT_POSTERIOR_CORNER = 0.62
_TENT_ANTERIOR_ROUND_MM = 16.0
_TENT_ANTERIOR_CORNER = 0.5
_TENT_RIM_SMOOTH_MM = 8.0
_TENT_NOTCH_OPEN_MM = 12.0
_TENTORIUM_SLOPE_EXP = 1.6
_INCISURA_CLEARANCE_MM = 2.0
_TENTORIUM_CLEARANCE_MM = 1.5

_TILT_MAX_SAMPLES = 20000

MAX_GROWTH_RATIO = 1.2
_GROWTH_LIMIT = 1.12


class _Frame:
    """Anatomical (R, A, S) view of a voxel array, from its affine."""

    def __init__(self, affine: np.ndarray, shape: tuple[int, ...]):
        self.affine = np.asarray(affine, dtype=float)
        self.shape = tuple(int(s) for s in shape)
        axes, signs = [], []
        for world_axis in range(3):
            col = int(np.argmax(np.abs(self.affine[world_axis, :3])))
            axes.append(col)
            signs.append(1 if self.affine[world_axis, col] >= 0 else -1)
        if sorted(axes) != [0, 1, 2]:
            raise ValueError(f"affine does not map array axes to distinct world axes: {axes}")
        self.axes = axes  # canonical axis  ->  original array axis
        self.signs = signs
        self.spacing = np.array([float(np.linalg.norm(self.affine[:3, a])) for a in axes])
        self.spacing[self.spacing <= 0] = 1e-6
        self.canon_shape = tuple(self.shape[a] for a in axes)

    def view(self, volume: np.ndarray) -> np.ndarray:
        out = np.transpose(volume, self.axes)
        return out[tuple(slice(None, None, s) for s in self.signs)]

    def to_index(self, points_world: np.ndarray) -> np.ndarray:
        """Inverse of `to_world`"""
        pts = np.atleast_2d(np.asarray(points_world, dtype=float))
        vox = (pts - self.affine[:3, 3]) @ np.linalg.inv(self.affine[:3, :3]).T
        out = np.empty_like(vox)
        for w in range(3):
            axis, sign = self.axes[w], self.signs[w]
            out[:, w] = vox[:, axis] if sign > 0 else (self.shape[axis] - 1 - vox[:, axis])
        return out

    def to_world(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        vox = np.empty_like(points)
        for w in range(3):
            axis, sign = self.axes[w], self.signs[w]
            vox[:, axis] = points[:, w] if sign > 0 else (self.shape[axis] - 1 - points[:, w])
        return vox @ self.affine[:3, :3].T + self.affine[:3, 3]


def _soft_max(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """`maximum(a, b)` rounded over a width of `k`, with no corner."""
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k)) - 0.5 * k


def _soft_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """`minimum(a, b)` rounded over a width of `k`."""
    return -_soft_max(-a, -b, k)


def _smooth_curve(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Interpolate across gaps, then smooth along the curve."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return values
    idx = np.arange(len(values), dtype=float)
    out = values.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    if sigma > 0:
        out = ndimage.gaussian_filter1d(out, sigma, mode="nearest")
    return out


def _fit_arch(values: np.ndarray, valid: np.ndarray, degree: int = 4) -> np.ndarray:
    """Least-squares polynomial through a margin curve, evaluated everywhere."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() <= degree + 1:
        return values
    idx = np.arange(len(values), dtype=float)
    span = max(float(np.ptp(idx[valid])), 1.0)
    t = (idx - idx[valid].mean()) / span
    coeff = np.polynomial.polynomial.polyfit(t[valid], values[valid], degree)
    return np.polynomial.polynomial.polyval(t, coeff)


def _limit_growth(values: np.ndarray, floor: float,
                  max_ratio: float = _GROWTH_LIMIT) -> np.ndarray:
    """Clamp a positive curve so neighbours never differ by more than max_ratio."""
    v = np.maximum(np.asarray(values, dtype=float), floor)
    if len(v) < 2:
        return v
    lo = np.broadcast_to(np.asarray(floor, dtype=float), v.shape)
    for i in range(1, len(v)):
        v[i] = max(min(max(v[i], v[i - 1] / max_ratio), v[i - 1] * max_ratio), lo[i])
    for i in range(len(v) - 2, -1, -1):
        v[i] = max(min(max(v[i], v[i + 1] / max_ratio), v[i + 1] * max_ratio), lo[i])
    return v


_TENT_ENVELOPE_MM = 8.0         # cerebellum top / cerebrum underside: span bridged by the envelope
_TENT_ENVELOPE_SMOOTH_MM = 3.0  # ... and its smoothing


def _smooth_envelope(height: np.ndarray, spacing_xy: tuple[float, float], upper: bool,
                     span_mm: float = _TENT_ENVELOPE_MM,
                     sigma_mm: float = _TENT_ENVELOPE_SMOOTH_MM) -> np.ndarray:
    """Smooth envelope of a height field (voxel z per x, y column): over its peaks (`upper`,
    a grey closing that bridges the valleys between folia) or under its dips (an opening)."""
    size = tuple(max(int(round(span_mm / s)) | 1, 1) for s in spacing_xy)
    env = (ndimage.grey_closing if upper else ndimage.grey_opening)(height, size=size, mode="nearest")
    return ndimage.gaussian_filter(env, [sigma_mm / s for s in spacing_xy], mode="nearest")


def _column_top(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest Z index occupied in each (X, Y) column, and which columns exist."""
    present = mask.any(axis=2)
    z = np.arange(mask.shape[2])
    top = np.where(mask, z[None, None, :], -1).max(axis=2)
    return np.maximum(top, 0).astype(float), present


def _nearest_fill(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Extend a 2D field into its invalid region from the nearest valid cell."""
    if valid.all():
        return field
    if not valid.any():
        return np.zeros_like(field)
    _dist, idx = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
    return field[tuple(idx)]


def _grid_quads(nu: int, nv: int, keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Quad connectivity of a structured (nu x nv) vertex grid, row-major."""
    if nu < 2 or nv < 2:
        return np.zeros((0, 4), dtype=np.int64)
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    if keep is not None:
        i, j = i[keep], j[keep]
    i, j = i.ravel(), j.ravel()
    n0 = i * nv + j
    return np.stack([n0, n0 + nv, n0 + nv + 1, n0 + 1], axis=1).astype(np.int64)


def weld_vertices(verts: np.ndarray, quads: np.ndarray,
                  tol_mm: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that sit in the same place, so one sheet is one sheet."""
    if not len(quads):
        return verts, quads
    quantized = np.rint(verts / max(tol_mm, 1e-9)).astype(np.int64)
    _uniq, first, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    if len(first) == len(verts):
        return verts, quads
    logger.info("weld_vertices: merged %d duplicated vertices", len(verts) - len(first))
    return verts[first], inverse.reshape(-1)[quads]


def _prune_unused(verts: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no surviving quad references, and reindex."""
    if not len(quads):
        return np.zeros((0, 3)), quads
    used = np.unique(quads)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[quads]


def _keep_main_piece(verts: np.ndarray, quads: np.ndarray, name: str = "sheet"):
    """Drop quads that share no node with the main sheet (a stray rim quad left behind by the
    finishing pass would be a free-floating element in the deck)."""
    if len(quads) < 2:
        return verts, quads
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.repeat(np.arange(len(quads)), 4)
    incidence = coo_matrix((np.ones(len(rows)), (rows, quads.ravel())),
                           shape=(len(quads), len(verts))).tocsr()
    n, piece = connected_components(incidence @ incidence.T, directed=False)
    if n == 1:
        return verts, quads
    keep = piece == np.argmax(np.bincount(piece))
    logger.warning("%s: dropped %d quad(s) in %d piece(s) detached from the sheet",
                   name, int((~keep).sum()), n - 1)
    return _prune_unused(verts, quads[keep])


def _sheet_area_cm2(verts: np.ndarray, quads: np.ndarray) -> float:
    """One-sided area of a quad sheet, cm^2 (verts in mm)."""
    if not len(quads):
        return 0.0
    p = verts[quads]
    a = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    b = np.linalg.norm(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=1)
    return float(0.5 * (a + b).sum() / 100.0)


def _sample_counts(span_mm: float, target_mm: float, lo: int, hi: int) -> int:
    """Vertex count spanning span_mm at roughly target_mm per element."""
    return int(np.clip(round(span_mm / max(target_mm, 1e-3)) + 1, lo, hi))


def _arc_samples(y_lo: int, y_hi: int, y_index: np.ndarray, z_curve: np.ndarray,
                 mm_y: float, mm_z: float, target_mm: float,
                 lo: int, hi: int,
                 z_other: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Sample a margin at uniform ARC length rather than uniform AP index."""
    ys = np.arange(y_lo, y_hi + 1, dtype=float)
    zs = np.interp(ys, y_index, z_curve)
    steps = np.hypot(np.diff(ys) * mm_y, np.diff(zs) * mm_z)
    if z_other is not None:
        other = np.interp(ys, y_index, z_other)
        steps = np.maximum(steps, np.hypot(np.diff(ys) * mm_y, np.diff(other) * mm_z))
    dist = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(dist[-1])
    n = _sample_counts(total, target_mm, lo, hi)
    return np.interp(np.linspace(0.0, total, n), dist, ys), total


def sheet_components(verts: np.ndarray, quads: np.ndarray) -> int:
    """How many separate sheets these quads form."""
    if not len(quads):
        return 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    used, local = np.unique(quads, return_inverse=True)
    local = local.reshape(quads.shape)
    rows = local.ravel()
    cols = local[:, [1, 2, 3, 0]].ravel()
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(len(used), len(used)))
    return int(connected_components(graph, directed=False, return_labels=False))


def sheet_quality(verts: np.ndarray, quads: np.ndarray) -> dict:
    """Edge-length spread and worst neighbour growth ratio on a quad sheet."""
    if not len(quads):
        return {"n_quads": 0, "min_edge_mm": 0.0, "max_edge_mm": 0.0, "max_growth_ratio": 1.0}
    p = verts[quads]
    edges = np.stack([np.linalg.norm(p[:, (i + 1) % 4] - p[:, i], axis=1) for i in range(4)], axis=1)
    size = edges.mean(axis=1)

    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    owner = np.tile(np.arange(len(quads)), 4)
    key = np.sort(pairs, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    a, b = owner[:-1][same], owner[1:][same]
    ratio = 1.0
    if len(a):
        ratio = float(np.max(np.maximum(size[a] / size[b], size[b] / size[a])))
    return {
        "n_quads": int(len(quads)),
        "min_edge_mm": round(float(edges.min()), 3),
        "max_edge_mm": round(float(edges.max()), 3),
        "max_growth_ratio": round(ratio, 3),
    }


def _report_quality(name: str, verts: np.ndarray, quads: np.ndarray) -> dict:
    """Measure a sheet and warn if it misses the growth-ratio target."""
    quality = sheet_quality(verts, quads)
    if quality["max_growth_ratio"] > MAX_GROWTH_RATIO:
        logger.warning("%s: worst neighbour growth ratio %.2f exceeds the %.2f target "
                       "(edges %.2f-%.2fmm)", name, quality["max_growth_ratio"],
                       MAX_GROWTH_RATIO, quality["min_edge_mm"], quality["max_edge_mm"])
    quality["n_components"] = sheet_components(verts, quads)
    if quality["n_components"] > 1:
        logger.warning("%s came out as %d separate sheets, not one. The reference decks "
                       "are single sheets; check the incisura cut and the sliver drop",
                       name, quality["n_components"])
    return quality


def _taubin_pass(out: np.ndarray, src: np.ndarray, dst: np.ndarray,
                 movable: np.ndarray, degree: np.ndarray, factor: float) -> None:
    """One weighted Laplacian displacement, in place, over the given graph."""
    total = np.zeros_like(out)
    np.add.at(total, src, out[dst])
    mean = total[movable] / degree[movable][:, None]
    out[movable] += factor * (mean - out[movable])


def smooth_cell_mask(keep: np.ndarray, radius: int = 1) -> np.ndarray:
    """Take the one-cell teeth out of a structured grid's cell mask."""
    if radius <= 0 or keep.ndim != 2 or not keep.any():
        return keep
    window = np.ones((3, 3), dtype=np.int8)
    out = keep
    for _ in range(radius):
        neighbours = ndimage.convolve(out.astype(np.int8), window,
                                      mode="constant", cval=0) - out
        thinned = out & (neighbours >= 1)
        if thinned.any():
            out = thinned
        neighbours = ndimage.convolve(out.astype(np.int8), window,
                                      mode="constant", cval=0) - out
        out = out | (~out & (neighbours >= 6))
    return out


def largest_cell_island(keep: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected island of a structured cell mask."""
    if not keep.any():
        return keep
    islands, n = ndimage.label(keep)
    if n <= 1:
        return keep
    sizes = ndimage.sum(keep, islands, index=range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    logger.info("dropped %d cell(s) in %d island(s) off the main sheet",
                int(keep.sum() - sizes[biggest - 1]), n - 1)
    return islands == biggest


@shell_repair.timed
def relax_sheet(verts: np.ndarray, quads: np.ndarray, iterations: int = 40,
                lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin-smooth a quad sheet, holding its outline fixed."""
    if iterations <= 0 or not len(quads):
        return verts
    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    key = np.sort(pairs, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    pinned = np.zeros(len(verts), dtype=bool)
    pinned[np.unique(uniq[counts == 1])] = True

    src = np.concatenate([uniq[:, 0], uniq[:, 1]])
    dst = np.concatenate([uniq[:, 1], uniq[:, 0]])
    degree = np.bincount(src, minlength=len(verts)).astype(float)
    movable = (~pinned) & (degree > 0)
    if not movable.any():
        return verts

    out = verts.copy()
    for _ in range(iterations):
        for factor in (lam, mu):
            _taubin_pass(out, src, dst, movable, degree, factor)
    return out


@shell_repair.timed
def relax_sheet_guarded(
    verts: np.ndarray, quads: np.ndarray, iterations: int = 20,
    max_displacement_mm: float = 0.5, lam: float = 0.5, mu: float = -0.53,
) -> tuple[np.ndarray, dict]:
    """Taubin-smooth a sheet under a quality guard and a displacement cap."""
    if iterations <= 0 or not len(quads):
        return verts, {"accepted": 0, "passes": 0}
    from engine.export.validate import _quad_scaled_jacobian

    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    key = np.sort(pairs, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    pinned = np.zeros(len(verts), dtype=bool)
    pinned[np.unique(uniq[counts == 1])] = True

    src = np.concatenate([uniq[:, 0], uniq[:, 1]])
    dst = np.concatenate([uniq[:, 1], uniq[:, 0]])
    degree = np.bincount(src, minlength=len(verts)).astype(float)
    movable = (~pinned) & (degree > 0)
    if not movable.any():
        return verts, {"accepted": 0, "passes": 0}
    degree[degree == 0] = 1.0

    origin = verts
    out = verts.copy()
    accepted_total, passes = 0, 0
    for _ in range(iterations):
        accepted_round = 0
        for factor in (lam, mu):
            proposed = out.copy()
            total = np.zeros_like(out)
            np.add.at(total, src, out[dst])
            mean = total / degree[:, None]
            proposed[movable] += factor * (mean[movable] - out[movable])
            delta = proposed - origin
            far = np.linalg.norm(delta, axis=1)
            over = far > max_displacement_mm
            if over.any():
                proposed[over] = (origin[over]
                                  + delta[over] * (max_displacement_mm / far[over])[:, None])

            before = _quad_scaled_jacobian(out[quads])
            worst_before = np.full(len(out), np.inf)
            after = _quad_scaled_jacobian(proposed[quads])
            worst_after = np.full(len(out), np.inf)
            for corner in range(4):
                np.minimum.at(worst_before, quads[:, corner], before)
                np.minimum.at(worst_after, quads[:, corner], after)
            keep = movable & (worst_after >= worst_before)
            if keep.any():
                out[keep] = proposed[keep]
                accepted_round += int(keep.sum())
            passes += 1
        accepted_total += accepted_round
        if accepted_round == 0:
            break
    return out, {"accepted": accepted_total, "passes": passes,
                 "max_displacement_mm": round(float(max_displacement_mm), 3)}


def _independent_sets(quads: np.ndarray, n_verts: int) -> list[np.ndarray]:
    """Greedy colouring of the shares-a-quad graph, as arrays of vertex ids."""
    if not len(quads):
        return []
    pairs = np.concatenate(
        [quads[:, [i, j]] for i in range(4) for j in range(i + 1, 4)], axis=0)
    key = np.unique(np.sort(pairs, axis=1), axis=0)
    src = np.concatenate([key[:, 0], key[:, 1]])
    dst = np.concatenate([key[:, 1], key[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    start = np.searchsorted(src, np.arange(n_verts + 1))

    colour = np.full(n_verts, -1, dtype=np.int64)
    degree = np.diff(start)
    for v in np.argsort(-degree, kind="stable"):
        if degree[v] == 0:
            continue
        taken = colour[dst[start[v]:start[v + 1]]]
        c = 0
        while c in taken:
            c += 1
        colour[v] = c
    return [np.nonzero(colour == c)[0] for c in range(int(colour.max()) + 1)] if (
        colour >= 0).any() else []


_UNTANGLE_SHEET_STAGNANT_PASSES = 3


@shell_repair.timed
def untangle_sheet(verts: np.ndarray, quads: np.ndarray, min_sj: float = 0.05,
                   iterations: int = 100, max_disp_mm: float | None = None,
                   ) -> tuple[np.ndarray, dict]:
    """Pull folded quads back out, without deleting any."""
    if not len(quads):
        return verts, {"repaired": 0, "remaining": 0}
    from engine.export.validate import _quad_scaled_jacobian

    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    key = np.sort(pairs, axis=1)
    uniq = np.unique(key, axis=0)
    src = np.concatenate([uniq[:, 0], uniq[:, 1]])
    dst = np.concatenate([uniq[:, 1], uniq[:, 0]])
    degree = np.bincount(src, minlength=len(verts)).astype(float)
    degree[degree == 0] = 1.0

    colours = _independent_sets(quads, len(verts))
    incident = np.argsort(quads.ravel(), kind="stable") // 4        # quads per vertex, CSR
    incident_start = np.searchsorted(np.sort(quads.ravel()), np.arange(len(verts) + 1))

    out = verts.copy()
    before = int((_quad_scaled_jacobian(out[quads]) < min_sj).sum())
    stagnant = 0
    best_bad, best_worst = before, -np.inf
    passes = 0
    for _ in range(iterations):
        passes += 1
        sj = _quad_scaled_jacobian(out[quads])
        bad = sj < min_sj
        if not bad.any():
            break

        n_bad, worst = int(bad.sum()), float(sj.min())
        if n_bad < best_bad or worst > best_worst + 1e-6:
            best_bad, best_worst = min(n_bad, best_bad), max(worst, best_worst)
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= _UNTANGLE_SHEET_STAGNANT_PASSES:
                break
        moving = np.zeros(len(out), dtype=bool)
        moving[np.unique(quads[bad])] = True

        moved = np.zeros(len(out), dtype=bool)
        for colour in colours:
            in_colour = np.zeros(len(out), dtype=bool)
            in_colour[colour] = True
            candidates = np.flatnonzero(moving & in_colour & ~moved)
            if not len(candidates):
                continue
            # only the quads around the vertices that may move can change the decision
            near = np.unique(incident[np.concatenate(
                [np.arange(incident_start[v], incident_start[v + 1]) for v in candidates])])
            near_quads = quads[near]
            sj_now = _quad_scaled_jacobian(out[near_quads])
            worst_before = np.full(len(out), np.inf)
            for corner in range(4):
                np.minimum.at(worst_before, near_quads[:, corner], sj_now)
            edges = np.flatnonzero(np.isin(src, candidates))
            total = np.zeros_like(out)
            np.add.at(total, src[edges], out[dst[edges]])
            target = total / degree[:, None]

            for damping in (1.0, 0.6, 0.3, 0.15, 0.05):
                pending = moving & in_colour & ~moved
                if not pending.any():
                    break
                candidate = out.copy()
                candidate[pending] += damping * (target[pending] - out[pending])
                if max_disp_mm is not None:
                    travel = np.linalg.norm(candidate - verts, axis=1)
                    pending &= travel <= max_disp_mm
                    if not pending.any():
                        continue
                worst_after = np.full(len(out), np.inf)
                after = _quad_scaled_jacobian(candidate[near_quads])
                for corner in range(4):
                    np.minimum.at(worst_after, near_quads[:, corner], after)
                keep = pending & (worst_after > worst_before)
                if keep.any():
                    out[keep] = candidate[keep]
                    moved |= keep
        if not moved.any():
            break

    final_sj = _quad_scaled_jacobian(out[quads])
    remaining = int((final_sj < min_sj).sum())
    negative = int((final_sj <= 0.0).sum())
    report = {"repaired": before - remaining, "remaining": remaining,
              "n_negative": negative, "min_sj": round(float(final_sj.min()), 4),
              "passes": passes}
    if negative:
        logger.warning("untangle_sheet: %d of %d cells are INVERTED (Scaled Jacobian "
                       "<= 0, worst %.4f) after %d passes", negative, len(quads),
                       float(final_sj.min()), passes)
    elif remaining:
        logger.warning("untangle_sheet: %d of %d cells are still below Scaled "
                       "Jacobian %.2f after %d passes (none inverted, worst %.4f)",
                       remaining, len(quads), min_sj, passes, float(final_sj.min()))
    elif before:
        logger.info("untangle_sheet: untangled %d folded cells", before)
    return out, report


def _edge_keys(quads: np.ndarray) -> np.ndarray:
    """One int64 key per quad side, low node id first."""
    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return lo * (np.int64(quads.max()) + 1) + hi


def _euler_characteristic(quads: np.ndarray) -> int:
    """V - E + F over the cells given, counting only the vertices they use."""
    if not len(quads):
        return 0
    n_edges = len(np.unique(_edge_keys(quads)))
    return int(len(np.unique(quads)) - n_edges + len(quads))


@shell_repair.timed
def _drop_slivers(verts: np.ndarray, quads: np.ndarray, min_edge_mm: float,
                  max_aspect: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Drop cells that are degenerate"""
    if not len(quads):
        return verts, quads
    p = verts[quads]
    edges = np.stack([np.linalg.norm(p[:, (i + 1) % 4] - p[:, i], axis=1) for i in range(4)], axis=1)
    shortest = edges.min(axis=1)
    keep = (shortest >= min_edge_mm) & (shortest * max_aspect >= edges.max(axis=1))
    if keep.all():
        return verts, quads

    before = sheet_components(verts, quads)
    before_chi = _euler_characteristic(quads)
    aspect = edges.max(axis=1) / np.maximum(shortest, 1e-12)
    candidates = np.nonzero(~keep)[0]
    _uniq, inverse, shared = np.unique(_edge_keys(quads), return_inverse=True,
                                       return_counts=True)
    on_rim = (shared[inverse] == 1).reshape(4, len(quads)).any(axis=0)
    candidates = candidates[on_rim[candidates]]
    candidates = candidates[np.argsort(-aspect[candidates])]
    keep = np.ones(len(quads), dtype=bool)
    restored = 0

    def holds(mask: np.ndarray) -> bool:
        return (sheet_components(verts, quads[mask]) <= before
                and _euler_characteristic(quads[mask]) >= before_chi)

    groups = [list(candidates)]
    while groups:
        group = groups.pop()
        keep[group] = False
        if holds(keep):
            continue
        keep[group] = True
        if len(group) == 1:
            restored += 1
            continue
        mid = len(group) // 2
        groups.append(group[mid:])
        groups.append(group[:mid])
    if keep.all():
        return verts, quads
    logger.info("dropped %d sliver quad(s) of %d (edge < %.2fmm or aspect > %.0f)%s",
                int((~keep).sum()), len(quads), min_edge_mm, max_aspect,
                f", kept {restored} that were holding the sheet together" if restored else "")
    return _prune_unused(verts, quads[keep])


def drop_inverted_without_holing(verts: np.ndarray, quads: np.ndarray,
                                 name: str = "sheet") -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove inverted cells, but only where removal cannot open a new hole."""
    from engine.export.validate import _quad_scaled_jacobian

    bad = _quad_scaled_jacobian(verts[quads]) <= 0.0
    report = {"n_inverted": int(bad.sum()), "n_dropped": 0, "n_kept_to_avoid_a_hole": 0}
    if not bad.any():
        return _unpinch(verts, quads, report, name)

    edges = np.sort(np.stack([quads[:, [0, 1]], quads[:, [1, 2]],
                              quads[:, [2, 3]], quads[:, [3, 0]]], axis=1), axis=2)
    flat = edges.reshape(-1, 2)
    _uniq, inverse, counts = np.unique(flat, axis=0, return_inverse=True,
                                       return_counts=True)
    on_rim = (counts[inverse] == 1).reshape(len(quads), 4).any(axis=1)

    drop = bad & on_rim
    report["n_dropped"] = int(drop.sum())
    report["n_kept_to_avoid_a_hole"] = int((bad & ~on_rim).sum())
    if report["n_kept_to_avoid_a_hole"]:
        logger.warning("%s: keeping %d inverted cell(s) whose removal would punch a "
                       "hole in a closed covering; %d dropped from the rim",
                       name, report["n_kept_to_avoid_a_hole"], report["n_dropped"])
    if drop.any():
        verts, quads = _prune_unused(verts, quads[~drop])
    return _unpinch(verts, quads, report, name)


def _unpinch(verts: np.ndarray, quads: np.ndarray, report: dict,
             name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split any node where two parts of the rim touch."""
    verts, quads, n_split = shell_repair.split_pinch_nodes(verts, quads)
    report["n_pinches_split"] = int(n_split)
    if n_split:
        logger.info("%s: split %d pinched rim node(s)", name, n_split)
    return verts, quads, report


def _sample_mask(mask: np.ndarray, points_vox: np.ndarray) -> np.ndarray:
    """Nearest-voxel lookup of a boolean mask at index-space points."""
    idx = np.rint(points_vox).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=1)
    out = np.zeros(len(points_vox), dtype=bool)
    sel = idx[inside]
    out[inside] = mask[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def _spread_displacement(verts: np.ndarray, moved: np.ndarray, quads: np.ndarray,
                         pushed: np.ndarray, allowed: np.ndarray, affine: np.ndarray,
                         iterations: int = 24) -> np.ndarray:
    """Carry the ring around each pushed vertex along with it."""
    n = len(verts)
    disp = moved - verts
    if not len(quads) or not len(pushed):
        return moved
    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    uniq = np.unique(np.sort(pairs, axis=1), axis=0)
    src = np.concatenate([uniq[:, 0], uniq[:, 1]])
    dst = np.concatenate([uniq[:, 1], uniq[:, 0]])
    degree = np.bincount(src, minlength=n).astype(float)
    degree[degree == 0] = 1.0

    held = disp[pushed].copy()
    for _ in range(iterations):
        disp = np.stack([np.bincount(src, weights=disp[dst, k], minlength=n)
                         for k in range(3)], axis=1) / degree[:, None]
        disp[pushed] = held

    out = verts + disp
    inv = np.linalg.inv(affine)
    ok = _sample_mask(allowed, out @ inv[:3, :3].T + inv[:3, 3])
    ok[pushed] = True
    return np.where(ok[:, None], out, moved)


RIM_BONE_TOLERANCE_MM = 1.0   # an attached rim may sit this far into the bone (the dura's band)
_RETRACT_MIN_SJ = 0.45        # a retract step may not push any quad under the ADAPT bar
_RETRACT_PER_QUAD = False     # True: judged per quad (each stays at its own level), offending nodes held
_RETRACT_RIM_WIDTH = 8        # rim nodes on each side a retract move is spread over ...
_RETRACT_RIM_DECAY = 0.8      # ... falling off by this factor per node


TISSUE_TOLERANCE_MM = 0.5  # a node exactly on a voxel face reads 0.5 mm deep (linear sampling)
_RETRACT_GAIN_MAX = 2.5     # per pass: a lone deep node is cleared over several passes, not one big bulge


def retract_sheet_from(verts: np.ndarray, quads: np.ndarray, solid: np.ndarray,
                       affine: np.ndarray, tolerance_mm: float = TISSUE_TOLERANCE_MM,
                       passes: int = 30, spread: int = 6,
                       name: str = "sheet") -> tuple[np.ndarray, dict]:
    """Ease every node deeper than `tolerance_mm` in `solid` back out, in 3D, with smooth moves.

    A deep node moves down the depth gradient by its excess depth; the move is spread over
    `spread` rings of neighbours (a broad dent, not a spike) and a pass that would push any
    quad under the quality bar is halved, then dropped. Unlike push_sheet_out_of, nothing
    jumps to a voxel centre, so a smooth sheet stays smooth."""
    from engine.export.validate import _quad_scaled_jacobian

    verts = np.asarray(verts, dtype=float).copy()
    report = {"deep_before": 0, "deep_after": 0, "moved_nodes": 0, "max_move_mm": 0.0}
    if not len(quads) or not solid.any():
        return verts, report
    n = len(verts)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    # a node on a voxel face reads half a voxel deep: never ask for less than that
    tolerance_mm = max(float(tolerance_mm), 0.5 * float(spacing.min()))
    rot = affine[:3, :3] / spacing
    inv = np.linalg.inv(affine)
    depth = fast_kernels.distance_transform_edt(solid, sampling=spacing)
    depth = ndimage.gaussian_filter(depth, 0.5)
    grads = np.gradient(depth, *spacing)
    pairs = np.unique(np.sort(np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)]),
                              axis=1), axis=0)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    degree = np.maximum(np.bincount(src, minlength=n), 1).astype(float)
    target = 0.5 * tolerance_mm
    start = verts.copy()

    def sample(field, pts):
        return ndimage.map_coordinates(field, (pts @ inv[:3, :3].T + inv[:3, 3]).T,
                                       order=1, mode="nearest")

    report["deep_before"] = int((sample(depth, verts) > tolerance_mm).sum())
    for _ in range(passes):
        d = sample(depth, verts)
        deep = np.where(d > tolerance_mm)[0]
        if not len(deep):
            break
        g = np.stack([sample(gk, verts[deep]) for gk in grads], axis=1) @ rot.T
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        need = (d[deep] - target)
        disp = np.zeros_like(verts)
        disp[deep] = -need[:, None] * g
        # one smooth field: blur the wanted moves over the sheet (no node held, so no spike),
        # then scale it so every deep node still gets at least its own move
        for _k in range(spread * spread):
            avg = np.stack([np.bincount(src, weights=disp[dst, k], minlength=n)
                            for k in range(3)], axis=1) / degree[:, None]
            disp = 0.5 * disp + 0.5 * avg
        along = -(disp[deep] * g).sum(1)
        gain = float(np.clip(np.max(need / np.maximum(along, 1e-9)), 1.0, _RETRACT_GAIN_MAX))
        disp *= gain
        touched = np.linalg.norm(disp, axis=1)[quads].max(axis=1) > 1e-6
        before = _quad_scaled_jacobian(verts[quads[touched]])
        accepted = False
        if _RETRACT_PER_QUAD:
            # each quad only has to stay at its own level (or the bar): one bad quad already in
            # the sheet (a sliver at a leaf tip) no longer vetoes every move; nodes of quads
            # that would still get worse are held and the rest moves
            floor_q = np.minimum(_RETRACT_MIN_SJ, before) - 1e-3
            for scale in (1.0, 0.5, 0.25):
                step = scale * disp
                for _hold in range(4):
                    bad = _quad_scaled_jacobian((verts + step)[quads[touched]]) < floor_q
                    if not bad.any():
                        break
                    step[np.unique(quads[touched][bad])] = 0.0
                if not bad.any() and np.abs(step).max() > 1e-9:
                    verts, accepted = verts + step, True
                    break
        else:
            for scale in (1.0, 0.5, 0.25):
                trial = verts + scale * disp
                if _quad_scaled_jacobian(trial[quads[touched]]).min() >= min(_RETRACT_MIN_SJ, before.min()):
                    verts, accepted = trial, True
                    break
        if not accepted:
            break
    moved = np.linalg.norm(verts - start, axis=1)
    report.update(deep_after=int((sample(depth, verts) > tolerance_mm).sum()),
                  moved_nodes=int((moved > 1e-6).sum()), max_move_mm=round(float(moved.max()), 3))
    if report["deep_before"]:
        logger.info("%s: eased %d of %d nodes out of solid tissue (%d moved, max %.2f mm)",
                    name, report["deep_before"] - report["deep_after"], report["deep_before"],
                    report["moved_nodes"], report["max_move_mm"])
    return verts, report


def _rim_edges(quads: np.ndarray) -> np.ndarray:
    edges = np.sort(np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    return unique[count == 1]


def retract_rim_from_bone(verts: np.ndarray, quads: np.ndarray, depth: np.ndarray,
                          affine: np.ndarray, tolerance_mm: float = RIM_BONE_TOLERANCE_MM,
                          passes: int = 12) -> tuple[np.ndarray, dict]:
    """Pull rim nodes that sit deeper than `tolerance_mm` in bone back along the sheet.

    Each deep rim node moves down the bone-depth gradient, projected into the sheet's tangent
    plane, so the sheet keeps its shape and plane; the move is spread over its neighbours
    (rim and interior) so the edge stays a smooth curve instead of a notched one. A step that
    would push any quad under the quality bar is halved, then dropped."""
    from engine.export.validate import _quad_scaled_jacobian

    verts = np.asarray(verts, dtype=float).copy()
    n = len(verts)
    report = {"moved_nodes": 0, "max_move_mm": 0.0}
    if not len(quads):
        return verts, report
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    rot = affine[:3, :3] / spacing  # index axes -> world unit directions
    grads = np.gradient(depth, *spacing)
    inv = np.linalg.inv(affine)
    rim_e = _rim_edges(quads)
    rim = np.unique(rim_e)
    pairs = np.unique(np.sort(np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)]),
                              axis=1), axis=0)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    degree = np.maximum(np.bincount(src, minlength=n), 1).astype(float)
    target = max(tolerance_mm - 0.25, 0.0)
    start = verts.copy()

    for _ in range(passes):
        index = (verts @ inv[:3, :3].T + inv[:3, 3]).T
        d = ndimage.map_coordinates(depth, index, order=1, mode="nearest")
        deep = rim[d[rim] > tolerance_mm]
        if not len(deep):
            break
        g = np.stack([ndimage.map_coordinates(gk, index[:, deep], order=1, mode="nearest")
                      for gk in grads], axis=1) @ rot.T
        # vertex normals from the quads around each node
        qn = np.cross(verts[quads[:, 2]] - verts[quads[:, 0]], verts[quads[:, 3]] - verts[quads[:, 1]])
        vn = np.zeros_like(verts)
        for k in range(4):
            np.add.at(vn, quads[:, k], qn)
        nrm = vn[deep] / np.maximum(np.linalg.norm(vn[deep], axis=1, keepdims=True), 1e-12)
        g -= (g * nrm).sum(1, keepdims=True) * nrm
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        step = (d[deep] - target)[:, None] * -g
        disp = np.zeros_like(verts)
        disp[deep] = step
        # widen each move along the rim first (a broad dent, not a one-node kink): diffuse it
        # over the rim edges, keeping at every node the largest move it has seen
        mag = np.zeros(n)
        mag[deep] = np.linalg.norm(step, axis=1)
        for _k in range(_RETRACT_RIM_WIDTH):
            spread = np.zeros(n)
            np.maximum.at(spread, rim_e[:, 0], mag[rim_e[:, 1]] * _RETRACT_RIM_DECAY)
            np.maximum.at(spread, rim_e[:, 1], mag[rim_e[:, 0]] * _RETRACT_RIM_DECAY)
            grow = (spread > mag) & np.isin(np.arange(n), rim)
            if not grow.any():
                break
            # direction: the nearest deep node's, taken from the neighbour that passed it on
            for a, b in ((0, 1), (1, 0)):
                take = grow[rim_e[:, a]] & (mag[rim_e[:, b]] * _RETRACT_RIM_DECAY >= spread[rim_e[:, a]] - 1e-12)
                ia, ib = rim_e[take, a], rim_e[take, b]
                unit = disp[ib] / np.maximum(np.linalg.norm(disp[ib], axis=1, keepdims=True), 1e-12)
                disp[ia] = unit * spread[ia, None]
            mag = np.maximum(mag, spread)
        held_idx = np.where(mag > 1e-9)[0]
        deep = held_idx
        held = disp[deep].copy()
        for _k in range(16):
            disp = np.stack([np.bincount(src, weights=disp[dst, k], minlength=n)
                             for k in range(3)], axis=1) / degree[:, None]
            disp[deep] = held
        before = _quad_scaled_jacobian(verts[quads])
        touched = np.linalg.norm(disp, axis=1)[quads].max(axis=1) > 1e-6
        accepted = False
        for scale in (1.0, 0.5, 0.25):
            trial = verts + scale * disp
            sj = _quad_scaled_jacobian(trial[quads[touched]])
            if sj.min() >= min(_RETRACT_MIN_SJ, before[touched].min()):
                verts, accepted = trial, True
                break
        if not accepted:
            break
    moved = np.linalg.norm(verts - start, axis=1)
    report.update(moved_nodes=int((moved > 1e-6).sum()), max_move_mm=round(float(moved.max()), 3))
    return verts, report


def trim_rim_in_bone(verts: np.ndarray, quads: np.ndarray, bone: np.ndarray, affine: np.ndarray,
                     tolerance_mm: Optional[float] = None, name: str = "sheet",
                     retract: Optional[bool] = None):
    """Keep the rim within `tolerance_mm` of bone: retract it smoothly, then trim what is left.

    The rim is first pulled back along the sheet (retract_rim_from_bone), which keeps the edge
    a smooth curve. Rim quads that still have a node deeper than `tolerance_mm` are then
    removed, one ring at a time, so no hole opens (the reference adult head: the tentorium's back rim reached
    3 mm into the occipital bone, where the cerebellum touches it). The default tolerance is
    RIM_BONE_TOLERANCE_MM scaled with the head of the last build_membranes call."""
    if tolerance_mm is None:
        tolerance_mm = RIM_BONE_TOLERANCE_MM * LAST_BUILD_SCALE
    if retract is None:
        retract = _RIM_RETRACT
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    depth = fast_kernels.distance_transform_edt(bone, sampling=spacing)
    retract_report: dict = {}
    if retract:
        verts, retract_report = retract_rim_from_bone(verts, quads, depth, affine, tolerance_mm)
    inv = np.linalg.inv(affine)
    deep = ndimage.map_coordinates(depth, (verts @ inv[:3, :3].T + inv[:3, 3]).T,
                                   order=1, mode="nearest") > tolerance_mm
    removed = 0
    while True:
        edges = np.sort(np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
        where, counts = _unique_rows(edges)
        on_rim = (counts[where.ravel()] == 1).reshape(4, -1).any(axis=0)
        drop = on_rim & deep[quads].any(axis=1)
        if not drop.any():
            break
        quads = quads[~drop]
        removed += int(drop.sum())
    used, local = np.unique(quads, return_inverse=True)
    report = {"removed_quads": removed, "nodes_deeper_left": int(deep[used].sum()),
              "tolerance_mm": tolerance_mm, "retract": retract_report}
    if removed:
        logger.info("%s: trimmed %d rim quads that reached more than %.1f mm into bone",
                    name, removed, tolerance_mm)
    return verts[used], local.reshape(-1, 4), report


@shell_repair.timed
def push_sheet_out_of(
    verts: np.ndarray, quads: np.ndarray, allowed: np.ndarray, affine: np.ndarray,
    max_push_mm: float = 6.0, relax_iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Move sheet vertices out of forbidden tissue, keeping every quad."""
    if not len(quads):
        return verts, quads
    inv = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    vox_mm = float(np.mean(spacing))

    vox = verts @ inv[:3, :3].T + inv[:3, 3]
    bad = ~_sample_mask(allowed, vox)
    if not bad.any():
        return verts, quads

    if not allowed.any():
        logger.warning("push_sheet_out_of: nothing is allowed anywhere; sheet left as built")
        return verts, quads

    _dist, nearest = ndimage.distance_transform_edt(
        ~allowed, sampling=spacing, return_indices=True)

    moved = verts.copy()
    n_started = int(bad.sum())
    active = np.nonzero(bad)[0]
    here = np.clip(np.rint(vox[active]).astype(np.int64), 0, np.array(allowed.shape) - 1)
    target_idx = np.stack([axis[here[:, 0], here[:, 1], here[:, 2]] for axis in nearest],
                          axis=1).astype(float)
    target = target_idx @ affine[:3, :3].T + affine[:3, 3]

    delta = target - moved[active]
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-12)[:, None]
    wanted = np.minimum(length + 0.5 * vox_mm, max_push_mm)
    moved[active] += direction * wanted[:, None]
    moved = _spread_displacement(verts, moved, quads, active, allowed, affine)

    bad[:] = False
    bad[active] = ~_sample_mask(allowed, moved[active] @ inv[:3, :3].T + inv[:3, 3])

    travelled = np.linalg.norm(moved - verts, axis=1)
    stuck = int(bad.sum())
    logger.info("push_sheet_out_of: moved %d of %d vertices out of solid tissue "
                "(max %.1fmm, %d still inside)", n_started - stuck, n_started,
                float(travelled.max()) if len(travelled) else 0.0, stuck)
    if stuck:
        logger.warning("%d membrane vertices could not clear solid tissue within %.0fmm; "
                       "they are kept in place rather than deleted, so the sheet stays "
                       "whole -- check the label volume around them", stuck, max_push_mm)

    if relax_iterations:
        relaxed = relax_sheet(moved, quads, iterations=relax_iterations)
        ok = _sample_mask(allowed, relaxed @ inv[:3, :3].T + inv[:3, 3])
        moved[ok] = relaxed[ok]
    return moved, quads


def _sample_field(field: np.ndarray, points_vox: np.ndarray) -> np.ndarray:
    """Trilinear lookup of a scalar field at index-space points."""
    return ndimage.map_coordinates(field, points_vox.T, order=1, mode="nearest")


def center_in_gap(
    verts: np.ndarray, quads: np.ndarray, allowed: np.ndarray, affine: np.ndarray,
    iterations: int = 6, max_step_mm: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Slide sheet vertices toward the middle of the space they are allowed in."""
    if not len(quads) or not allowed.any():
        return verts, {}
    inv = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)

    clearance = fast_kernels.distance_transform_edt(allowed, sampling=spacing)
    grad = np.gradient(clearance, *spacing)

    p = verts[quads]
    edge_len = np.stack([np.linalg.norm(p[:, (i + 1) % 4] - p[:, i], axis=1)
                         for i in range(4)], axis=1)
    local = np.full(len(verts), np.inf)
    for corner in range(4):
        np.minimum.at(local, quads[:, corner], edge_len[:, corner])
        np.minimum.at(local, quads[:, (corner + 1) % 4], edge_len[:, corner])
    budget = np.where(np.isfinite(local), 0.4 * local, max_step_mm)

    out = verts.copy()
    vox = out @ inv[:3, :3].T + inv[:3, 3]
    before = _sample_field(clearance, vox)
    current = before.copy()
    for _ in range(iterations):
        vox = out @ inv[:3, :3].T + inv[:3, 3]
        direction = np.stack([_sample_field(g, vox) for g in grad], axis=1)
        length = np.linalg.norm(direction, axis=1)
        live = length > 1e-9
        if not live.any():
            break
        step = np.zeros_like(out)
        step[live] = direction[live] / length[live][:, None] * max_step_mm
        candidate = out + step
        c_vox = candidate @ inv[:3, :3].T + inv[:3, 3]
        keep = (_sample_mask(allowed, c_vox)
                & (_sample_field(clearance, c_vox) > current)
                & (np.linalg.norm(candidate - verts, axis=1) <= budget))
        if not keep.any():
            break
        out[keep] = candidate[keep]
        current[keep] = _sample_field(clearance, c_vox[keep])

    report = {
        "min_clearance_mm_before": round(float(before.min()), 3),
        "min_clearance_mm_after": round(float(current.min()), 3),
        "mean_clearance_mm_after": round(float(current.mean()), 3),
        "moved_mm_max": round(float(np.linalg.norm(out - verts, axis=1).max()), 3),
    }
    logger.info("center_in_gap: clearance from tissue %.2f -> %.2fmm worst, %.2fmm mean",
                report["min_clearance_mm_before"], report["min_clearance_mm_after"],
                report["mean_clearance_mm_after"])
    return out, report


def project_sheet_to_offset(
    verts: np.ndarray, quads: np.ndarray, field: np.ndarray, affine: np.ndarray,
    lo_mm: float, hi_mm: Optional[float] = None, iterations: int = 12,
    name: str = "sheet", max_move_mm: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """Walk sheet vertices along `field`'s gradient until `lo <= field <= hi`."""
    if not len(quads):
        return verts, {}
    inv = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    grad = np.gradient(field, *spacing)
    one_sided = hi_mm is None
    span = float(np.max(spacing)) if one_sided else float(hi_mm) - float(lo_mm)
    target = float(lo_mm) + 0.5 * span if one_sided else 0.5 * (lo_mm + hi_mm)
    upper = np.inf if one_sided else float(hi_mm)

    p = verts[quads]
    edge = np.stack([np.linalg.norm(p[:, (i + 1) % 4] - p[:, i], axis=1)
                     for i in range(4)], axis=1)
    local = np.full(len(verts), np.inf)
    for corner in range(4):
        np.minimum.at(local, quads[:, corner], edge[:, corner])
        np.minimum.at(local, quads[:, (corner + 1) % 4], edge[:, corner])
    budget = np.where(np.isfinite(local), 0.4 * local, span)
    if max_move_mm is not None:
        budget = np.maximum(budget, float(max_move_mm))

    out = verts.copy()
    before = _sample_field(field, out @ inv[:3, :3].T + inv[:3, 3])
    n_before = int(((before < lo_mm) | (before > upper)).sum())

    for _ in range(iterations):
        vox = out @ inv[:3, :3].T + inv[:3, 3]
        value = _sample_field(field, vox)
        off = (value < lo_mm + _BAND_EPS_MM) | (value > upper - _BAND_EPS_MM)
        if not off.any():
            break
        direction = np.stack([_sample_field(g, vox) for g in grad], axis=1)
        length = np.linalg.norm(direction, axis=1)
        live = off & (length > 1e-9)
        if not live.any():
            break
        want = np.clip((target - value) / np.maximum(length, 1e-9), -span, span)
        step = np.zeros_like(out)
        step[live] = direction[live] / length[live][:, None] * want[live][:, None]
        candidate = out + step
        c_value = _sample_field(field, candidate @ inv[:3, :3].T + inv[:3, 3])
        keep = (live
                & (np.abs(c_value - target) < np.abs(value - target))
                & (np.linalg.norm(candidate - verts, axis=1) <= budget))
        if not keep.any():
            break
        out[keep] = candidate[keep]

    after = _sample_field(field, out @ inv[:3, :3].T + inv[:3, 3])
    n_after = int(((after < lo_mm) | (after > upper)).sum())
    report = {
        "interval_mm": [round(float(lo_mm), 3),
                        None if one_sided else round(float(hi_mm), 3)],
        "outside_before": n_before,
        "outside_after": n_after,
        "value_min_mm": round(float(after.min()), 3),
        "value_max_mm": round(float(after.max()), 3),
        "moved_mm_max": round(float(np.linalg.norm(out - verts, axis=1).max()), 3),
    }
    logger.info("project_sheet_to_offset: %s into [%.2f, %s]mm -- %d of %d vertices "
                "outside it, now %d; range %.2f to %.2fmm, worst move %.2fmm",
                name, lo_mm, "inf" if one_sided else f"{hi_mm:.2f}",
                n_before, len(verts), n_after,
                report["value_min_mm"], report["value_max_mm"], report["moved_mm_max"])
    return out, report


@shell_repair.timed
def carve_slots(
    labels: np.ndarray, affine: np.ndarray,
    sheets: dict[str, tuple[np.ndarray, np.ndarray]],
    csf_label: int = _CSF, thickness_mm: float = 1.0,
    carvable: tuple[int, ...] = _BRAIN,
) -> tuple[np.ndarray, dict]:
    """Open a CSF slot in the label volume along each membrane sheet."""
    out = np.asarray(labels).copy()
    report: dict = {}
    if not sheets:
        return out, report
    inv = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    can_carve = np.isin(out, list(carvable))

    for name, (verts, quads) in sheets.items():
        if not len(quads):
            continue
        p = verts[quads]
        edge = max(float(np.linalg.norm(p[:, 1] - p[:, 0], axis=1).max()),
                   float(np.linalg.norm(p[:, 3] - p[:, 0], axis=1).max()))
        n = int(np.clip(np.ceil(edge / (0.5 * float(np.mean(spacing)))) + 1, 2, 16))
        s = np.linspace(0.0, 1.0, n)
        su, sv = np.meshgrid(s, s, indexing="ij")
        su, sv = su.ravel()[None, :, None], sv.ravel()[None, :, None]
        pts = ((1 - su) * (1 - sv) * p[:, 0][:, None, :]
               + su * (1 - sv) * p[:, 1][:, None, :]
               + su * sv * p[:, 2][:, None, :]
               + (1 - su) * sv * p[:, 3][:, None, :]).reshape(-1, 3)

        vox = np.rint(pts @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
        keep = np.all((vox >= 0) & (vox < np.array(out.shape)), axis=1)
        vox = vox[keep]
        slot = np.zeros(out.shape, dtype=bool)
        slot[vox[:, 0], vox[:, 1], vox[:, 2]] = True

        extra = int(round(0.5 * thickness_mm / float(np.mean(spacing))))
        if extra > 0:
            slot = ndimage.binary_dilation(slot, iterations=extra)
        slot &= can_carve
        out[slot] = csf_label
        report[name] = int(slot.sum())
        logger.info("carve_slots: %s opened a %.1fmm CSF slot through %d brain voxels",
                    name, thickness_mm, report[name])
    return out, report


_VOID_CLEARANCE_MM = 0.5
_VOID_MAX_FRACTION = 0.25


def carve_element_void(
    points: np.ndarray,
    membrane_blocks: dict[str, np.ndarray],
    solid_blocks: dict[str, np.ndarray],
    *,
    host_parts: tuple[str, ...] = ("csf",),
    clearance_mm: float = _VOID_CLEARANCE_MM,
    max_fraction: float = _VOID_MAX_FRACTION,
    chunk: int = 20_000,
) -> tuple[dict[str, np.ndarray], dict]:
    """Delete the host elements each membrane sheet passes through."""
    from scipy.spatial import cKDTree

    from engine.export.validate import _tri_aabb_overlap

    out = dict(solid_blocks)
    report: dict = {"clearance_mm": float(clearance_mm), "carved": {}, "refused": {}}
    hosts = [n for n in out
             if str(n).lower() in {str(h).lower() for h in host_parts}
             and np.asarray(out[n]).size]
    sheets = {n: np.asarray(q) for n, q in membrane_blocks.items()
              if np.asarray(q).size}
    if not hosts or not sheets:
        return out, report

    points = np.asarray(points, dtype=float)
    for part in hosts:
        block = np.asarray(out[part])
        corners = points[block]
        low, high = corners.min(axis=1), corners.max(axis=1)
        centroids = 0.5 * (low + high)
        half = 0.5 * (high - low) + float(clearance_mm)
        tree = cKDTree(centroids)
        reach = float(np.linalg.norm(half, axis=1).max())

        doomed: set[int] = set()
        per_sheet: dict[str, int] = {}
        for name, quads in sheets.items():
            before = len(doomed)
            quad_pts = points[quads]
            tris = np.concatenate([quad_pts[:, [0, 1, 2]], quad_pts[:, [0, 2, 3]]], axis=0)
            centre = tris.mean(axis=1)
            radius = np.linalg.norm(tris - centre[:, None, :], axis=2).max(axis=1)
            for start in range(0, len(tris), chunk):
                sl = slice(start, start + chunk)
                near = tree.query_ball_point(centre[sl], radius[sl] + reach)
                pair_t, pair_h = [], []
                for i, candidates in enumerate(near):
                    if candidates:
                        pair_t.append(np.full(len(candidates), start + i))
                        pair_h.append(np.asarray(candidates, dtype=int))
                if not pair_t:
                    continue
                pair_t = np.concatenate(pair_t)
                pair_h = np.concatenate(pair_h)
                hit = _tri_aabb_overlap(tris[pair_t], centroids[pair_h], half[pair_h])
                doomed.update(int(h) for h in pair_h[hit])
            per_sheet[name] = len(doomed) - before

        if not doomed:
            continue
        if len(doomed) > max_fraction * len(block):
            report["refused"][part] = {
                "would_carve": len(doomed), "of": int(len(block)),
                "max_fraction": float(max_fraction)}
            logger.error("carve_element_void: REFUSED to carve %d of %d %s elements "
                         "(over the %.0f%% ceiling). A sheet asking for that much of a "
                         "part is misplaced; carving around it would hide that",
                         len(doomed), len(block), part, 100 * max_fraction)
            continue
        keep = np.ones(len(block), dtype=bool)
        keep[np.fromiter(doomed, dtype=np.int64, count=len(doomed))] = False
        out[part] = block[keep]
        report["carved"][part] = {"removed": int((~keep).sum()), "of": int(len(block)),
                                  "by_sheet": per_sheet}
        logger.info("carve_element_void: %s lost %d of %d elements to a %.1fmm void "
                    "along %s", part, int((~keep).sum()), len(block), clearance_mm,
                    ", ".join(per_sheet))
    return out, report


def extrude_sheet(
    verts: np.ndarray, quads: np.ndarray, thickness_mm: float, weld_tol_mm: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a quad sheet into one layer of hexes, half the thickness each side."""
    if not len(quads):
        return verts, np.zeros((0, 8), dtype=np.int64)

    p = verts[quads]
    face_n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    length = np.linalg.norm(face_n, axis=1)
    keep = length > 1e-12
    if not keep.all():
        logger.info("extrude_sheet: dropped %d degenerate quad(s)", int((~keep).sum()))
        quads, p, face_n, length = quads[keep], p[keep], face_n[keep], length[keep]
        if not len(quads):
            return verts, np.zeros((0, 8), dtype=np.int64)
    face_n /= length[:, None]

    half = 0.5 * thickness_mm
    offset = half * face_n[:, None, :]  # (K, 1, 3)
    corners = np.concatenate([p - offset, p + offset], axis=1).reshape(-1, 3)

    quantized = np.rint(corners / weld_tol_mm).astype(np.int64)
    _uniq, first, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    hexes = inverse.reshape(len(quads), 8).astype(np.int64)
    return corners[first], hexes


def midsagittal_plane(frame: _Frame, canon: np.ndarray,
                      pin: Optional[float] = None) -> Optional[tuple[float, int, int, int]]:
    """The one sagittal plane the falx sits on, in canonical voxel index."""
    cerebrum = np.isin(canon, _CEREBRUM)
    if not cerebrum.any():
        return None
    guess = int(round(np.median(np.where(cerebrum)[0])))
    reach = max(1, int(round(5.0 / frame.spacing[0])))
    if pin is not None:
        guess = int(round(float(pin)))
    lo = max(guess - reach, 0)
    hi = min(guess + reach + 1, canon.shape[0])
    x_mid = lo + int(np.argmin(cerebrum[lo:hi].sum(axis=(1, 2))))
    if pin is not None:
        return float(pin), x_mid, lo, hi

    intracranial = np.isin(canon, _INTRACRANIAL)
    if not intracranial.any():
        intracranial = np.isin(canon, _BRAIN)
    band = intracranial.any(axis=0)
    if not band.any():
        return float(x_mid), x_mid, lo, hi
    clear = fast_kernels.distance_transform_edt(~cerebrum, sampling=frame.spacing)
    yy, zz = np.nonzero(band)
    yy, zz = yy.astype(float), zz.astype(float)
    candidates = np.arange(float(lo), float(hi - 1) + 1e-6, 0.25)
    scores = [float(ndimage.map_coordinates(
        clear, np.stack([np.full(len(yy), xc), yy, zz]), order=1, mode="nearest").mean())
        for xc in candidates]
    x_plane = float(candidates[int(np.argmax(scores))])
    logger.info("midsagittal plane: x=%.2f voxels (bracket %d-%d, fissure slice %d), mean "
                "clearance from cerebrum %.2fmm", x_plane, lo, hi - 1, x_mid, max(scores))
    return x_plane, x_mid, lo, hi


def subject_plane(labels: np.ndarray,
                  affine: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """This subject's midsagittal plane in world mm, as `(point, unit normal)`."""
    from engine.anatomy import orientation

    found = orientation.midsagittal_plane(labels, affine)
    if found is None:
        return None
    point, normal, score = found
    if score < orientation._MIN_SYMMETRY:
        logger.warning("midsagittal symmetry only %.3f, under the %.2f floor; membranes stay "
                       "on the scanner's sagittal planes", score, orientation._MIN_SYMMETRY)
        return None
    return np.asarray(point, dtype=float), np.asarray(normal, dtype=float)


def align_to_midsagittal(
        labels: np.ndarray,
        affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Resample onto a grid whose sagittal planes are this subject's own."""
    labels = np.asarray(labels)
    affine = np.asarray(affine, dtype=float)
    found = subject_plane(labels, affine)
    if found is None:
        return labels, affine, None
    point, n_world = found

    frame = _Frame(affine, labels.shape)
    canon_to_world = affine[:3, :3][:, frame.axes] * np.asarray(frame.signs, dtype=float)
    x_world = canon_to_world[:, 0]
    x_world = x_world / np.linalg.norm(x_world)
    if float(n_world @ x_world) < 0:
        n_world = -n_world

    v = np.cross(x_world, n_world)
    s, c = float(np.linalg.norm(v)), float(x_world @ n_world)
    if s < 1e-9:
        return labels, affine, float(frame.to_index(point[None, :])[0, 0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    rot = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    body = np.isin(labels, _CEREBRUM)
    if not body.any():
        body = np.isin(labels, _BRAIN)
    centre_vox = np.array(ndimage.center_of_mass(body))
    centre = affine[:3, :3] @ centre_vox + affine[:3, 3]
    affine_rot = affine.copy()
    affine_rot[:3, :3] = rot @ affine[:3, :3]
    affine_rot[:3, 3] = rot @ (affine[:3, 3] - centre) + centre

    m = np.linalg.inv(affine) @ affine_rot
    rotated = ndimage.affine_transform(labels, m[:3, :3], offset=m[:3, 3],
                                       output_shape=labels.shape, order=0,
                                       mode="constant", cval=0)
    frame_rot = _Frame(affine_rot, rotated.shape)
    x_plane = float(frame_rot.to_index(point[None, :])[0, 0])
    logger.info("midsagittal alignment: head turned %.1f degrees onto its symmetry plane; "
                "membranes on a grid turned to match, plane at x=%.2f voxels",
                np.degrees(np.arccos(np.clip(c, -1.0, 1.0))), x_plane)
    return rotated, affine_rot, x_plane


def _falx_curves(frame: _Frame, canon: np.ndarray,
                 pin: Optional[float] = None) -> Optional[dict]:
    """Attached and free margins of the falx, per AP position, for this subject."""
    cerebrum = np.isin(canon, _CEREBRUM)
    if not cerebrum.any():
        return None
    intracranial = np.isin(canon, _INTRACRANIAL)
    if not intracranial.any():
        intracranial = np.isin(canon, _BRAIN)

    plane = midsagittal_plane(frame, canon, pin)
    if plane is None:
        return None
    x_falx, x_mid, lo, hi = plane

    half = max(1, int(round(2.0 / frame.spacing[0])))
    slab = slice(max(x_mid - half, 0), min(x_mid + half + 1, canon.shape[0]))

    vault = intracranial[slab].any(axis=0)  # (Y, Z)
    # the cavity on the falx plane itself: the slab union above reaches past the bone of the
    # plane, so an outline drawn on it ends up in the skull and the rim trim notches it
    yy, zz = np.meshgrid(np.arange(canon.shape[1]), np.arange(canon.shape[2]), indexing="ij")
    vault_plane = ndimage.map_coordinates(
        intracranial.astype(np.float32), np.stack([np.full(yy.shape, float(x_falx)), yy, zz]),
        order=1, mode="nearest") >= 0.5
    vault_plane &= vault
    z = np.arange(vault.shape[1])
    top_raw = np.where(vault, z[None, :], -1).max(axis=1).astype(float)
    present = top_raw >= 0
    if present.sum() < 8:
        return None
    top = _smooth_curve(top_raw, present, sigma=3.0)
    bottom_raw = np.where(vault, z[None, :], vault.shape[1] + 1).min(axis=1).astype(float)
    bottom = _smooth_curve(bottom_raw, present, sigma=3.0)

    vault_peak = float(top.max())

    top = _fit_arch(top, present)
    top = np.clip(top, bottom + 1.0, vault_peak)
    logger.info("falx: attached margin fitted to a degree-4 arch on the inner "
                "vault (peak z %.1f against the vault's %.1f)",
                float(top.max()), vault_peak)

    midline_tissue = np.isin(canon, (_WHITE, _DEEP))[slab]
    blocking = midline_tissue.all(axis=0)
    block_raw = np.where(blocking, z[None, :], -1).max(axis=1).astype(float)
    measured = block_raw >= 0
    has_block = measured.copy()
    gaps, n_gaps = ndimage.label(~measured)
    for gap in range(1, n_gaps + 1):
        idx = np.where(gaps == gap)[0]
        interior = idx[0] > 0 and idx[-1] < len(measured) - 1
        if interior and len(idx) * frame.spacing[1] <= _FALX_BLOCK_FILL_MM:
            has_block[idx] = True
    block = _smooth_curve(block_raw, block_raw >= 0, sigma=4.0)

    block_hard = _smooth_curve(block_raw, block_raw >= 0, sigma=0.0)

    room = present & ((top - bottom) * frame.spacing[2] >= _FALX_SPAN_MIN_MM)
    runs, n_runs = ndimage.label(room)
    if not n_runs:
        return None
    ys = np.where(runs == 1 + np.argmax([(runs == r).sum() for r in range(1, n_runs + 1)]))[0]

    cav, _n_cav = ndimage.label(present)
    tag = cav[ys[0]]
    full = np.where(cav == tag)[0] if tag else ys
    y_post, y_ant = int(ys.min()), int(ys.max())
    open_run = np.zeros_like(has_block)
    span = np.zeros_like(has_block)
    span[y_post:y_ant + 1] = True
    components, n_runs = ndimage.label(span & ~has_block)
    for run in range(1, n_runs + 1):
        idx = np.where(components == run)[0]
        if idx[0] <= y_post or idx[-1] >= y_ant:
            open_run[idx] = True

    return {"x_mid": x_mid, "x_falx": x_falx, "y_post": y_post,
            "y_ant": y_ant, "y_first": int(full.min()), "y_last": int(full.max()),
            "top": top, "bottom": bottom,
            "block": block, "block_hard": block_hard,
            "has_block": has_block, "open_run": open_run,
            "vault": vault, "vault_plane": vault_plane, "blocking": blocking}


def _main_run(has_block: np.ndarray) -> np.ndarray:
    """Station indices of the callosal run"""
    runs, n_runs = ndimage.label(np.asarray(has_block, dtype=bool))
    if not n_runs:
        return np.zeros(0, dtype=np.int64)
    main = 1 + int(np.argmax([(runs == r).sum() for r in range(1, n_runs + 1)]))
    return np.where(runs == main)[0]


def _cc_clearance(has_block: np.ndarray, mm_y: float, gap_mm: float) -> np.ndarray:
    """Clearance the free margin keeps above the callosum, per AP station, in mm."""
    n = len(has_block)
    lo = float(_FALX_CC_CLEARANCE_AT_SPLENIUM_MM)
    out = np.full(n, float(gap_mm))
    idx = _main_run(has_block)
    if len(idx) < 2:
        return out
    y0, y1 = int(idx[0]), int(idx[-1])
    span = (y1 - y0) * mm_y
    if span <= 0:
        return out
    t = np.clip((np.arange(n, dtype=float) - y0) * mm_y / span, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    return lo + s * (float(gap_mm) - lo)


def _narrow_front(free: np.ndarray, vault: np.ndarray, genu: int,
                  y_end: int, mm_y: float, mm_z: float) -> np.ndarray:
    """In front of the genu the falx is a band along the vault and frontal bone, narrowing
    to the crista galli, not a sheet filling the fissure down to the skull base."""
    if genu >= y_end:
        return free
    # crista galli: the most anterior-inferior corner of the vault in front of the genu
    yy, zz = np.nonzero(vault[genu + 1:y_end + 1])
    if not len(yy):
        return free
    corner = int(np.argmax(yy * mm_y - zz * mm_z))
    y_c, z_c = float(yy[corner] + genu + 1), float(zz[corner])
    z_g = float(free[genu])
    out = np.array(free, dtype=float)
    # quadratic Bezier from the free margin over the genu (P_g) to the crista galli (P_c),
    # control point (y_c, z_g): runs forward level with the genu, then turns down the
    # frontal bone, leaving the subfrontal fissure below it open
    for y in range(genu + 1, min(int(round(y_c)), len(out) - 1) + 1):
        s = 1.0 - np.sqrt(max(y_c - y, 0.0) / max(y_c - genu, 1e-6))
        out[y] = max(out[y], z_g - s * s * (z_g - z_c))
    if int(round(y_c)) + 1 < len(out):
        out[int(round(y_c)) + 1:y_end + 1] = np.maximum(out[int(round(y_c)) + 1:y_end + 1], z_c)
    logger.info("falx: anterior free margin curves from the genu (z %.1f) down to the crista "
                "galli at y %.1f, z %.1f (voxels)", z_g, y_c, z_c)
    return out


def sheet_plane(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit plane of a sheet, as `(point, unit normal)`."""
    centre = verts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(verts - centre, full_matrices=False)
    return centre, vt[2]


def flatten_to_plane(verts: np.ndarray, plane: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Project vertices onto a plane, removing any out-of-plane warp."""
    point, normal = plane
    return verts - ((verts - point) @ normal)[:, None] * normal[None, :]


def _offset_margin(curves: dict, clearance_mm: np.ndarray,
                   mm_y: float, mm_z: float) -> np.ndarray:
    """Free margin as a level set of the offset field around the midline mass."""
    blocking = np.asarray(curves["blocking"], dtype=bool)
    vault = np.asarray(curves["vault"], dtype=bool)
    n_y, n_z = vault.shape
    z = np.arange(n_z)

    if blocking.any():
        dist = fast_kernels.distance_transform_edt(~blocking, sampling=(mm_y, mm_z))
    else:
        dist = np.full(vault.shape, np.inf)
    room = vault & (dist >= clearance_mm[:, None])
    floor = np.where(curves["has_block"], curves["block_hard"], -np.inf)
    room &= z[None, :] >= floor[:, None]

    label, n_label = ndimage.label(room)
    if n_label:
        rows = np.arange(n_y)
        seat = np.clip(np.rint(curves["top"]).astype(np.int64) - 1, 0, n_z - 1)
        tags = np.unique(label[rows, seat])
        tags = tags[tags > 0]
        if len(tags):
            room = np.isin(label, tags)

    reached = room.any(axis=1)
    free = np.where(room, z[None, :], n_z + 1).min(axis=1).astype(float)
    return np.where(reached, free, curves["bottom"])


@shell_repair.timed
def falx_sheet(
    labels: np.ndarray, affine: np.ndarray, target_mm: float = 1.0,
    tentorium_ridge: Optional[tuple[np.ndarray, np.ndarray]] = None,
    cc_gap_mm: float = _FALX_FREE_MARGIN_GAP_MM,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Build the falx cerebri as a structured quad sheet."""
    labels, affine, x_pin = align_to_midsagittal(labels, affine)
    frame = _Frame(affine, labels.shape)
    canon = frame.view(np.asarray(labels))
    curves = _falx_curves(frame, canon, x_pin)
    if curves is None:
        return None

    x_mid, y_post, y_ant = curves["x_mid"], curves["y_post"], curves["y_ant"]
    top, bottom = curves["top"], curves["bottom"]
    mm_y, mm_z = frame.spacing[1], frame.spacing[2]

    y_index = np.arange(len(top), dtype=float)
    clearance = _cc_clearance(curves["has_block"], mm_y, float(cc_gap_mm))
    free = _offset_margin(curves, clearance, mm_y, mm_z)

    floor_curve = curves["bottom"].astype(float)
    ridge_at_y = None
    if tentorium_ridge is not None and len(tentorium_ridge[0]) >= 2:
        ridge_y, ridge_z = tentorium_ridge
        on_tent = y_index <= ridge_y.max()
        ridge_at_y = np.interp(y_index, ridge_y, ridge_z) - _FALX_STEM_SEAT_MM / mm_z
        floor_curve = np.where(on_tent, np.maximum(floor_curve, ridge_at_y),
                               floor_curve)
    if ridge_at_y is not None:
        free = np.where(free < floor_curve, floor_curve, free)

    _wide_mm, tight_mm = _FALX_FREE_SMOOTH_MM
    free = ndimage.gaussian_filter1d(free, max(tight_mm / mm_y, 1.0), mode="nearest")
    reach_u = max(int(round(target_mm / mm_y)), 1)
    blocked = np.where(curves["has_block"],
                       curves["block_hard"] + clearance / mm_z, -np.inf)
    callosal_floor = ndimage.maximum_filter1d(blocked, 2 * reach_u + 1, mode="nearest")
    if ridge_at_y is not None:
        joined = on_tent & curves["open_run"]
        runs, n_joined = ndimage.label(joined)
        stations = np.zeros(len(joined), dtype=float)
        anchor = np.array(free, dtype=float)
        for run in range(1, n_joined + 1):
            idx = np.where(runs == run)[0]
            reach = np.full(len(idx), np.inf)
            seat = None
            if idx[0] > 0 and curves["has_block"][idx[0] - 1]:
                reach = np.minimum(reach, idx - (idx[0] - 1))
                seat = idx[0] - 1
            if idx[-1] + 1 < len(joined) and curves["has_block"][idx[-1] + 1]:
                far = (idx[-1] + 1) - idx
                if seat is None or far.min() < reach.min():
                    seat = idx[-1] + 1
                reach = np.minimum(reach, far)
            stations[idx] = reach
            if seat is not None:
                anchor[idx] = free[seat]
        ramp = np.clip(stations * mm_y / _FALX_OPEN_RUN_DROP_MM, 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        free = np.where(joined, ramp * ridge_at_y + (1.0 - ramp) * anchor, free)
    if ridge_at_y is not None:
        callosal_floor = np.where(joined, -np.inf, callosal_floor)
    free = np.maximum(free, callosal_floor)

    cc_base = ndimage.gaussian_filter1d(curves["block_hard"],
                                        max(_FALX_CC_ARC_SMOOTH_MM / mm_y, 1.0),
                                        mode="nearest")
    cc_arc = _main_run(curves["has_block"])
    if len(cc_arc) >= 8:
        y_run = y_index[cc_arc]
        arc = cc_base[cc_arc] + clearance[cc_arc] / mm_z
        lift = float(np.max(callosal_floor[cc_arc] - arc))
        if np.isfinite(lift) and lift > 0.0:
            arc = arc + lift
        blend = np.clip(np.minimum(y_run - y_run[0], y_run[-1] - y_run) * mm_y
                        / _FALX_CC_ARC_BLEND_MM, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        free[cc_arc] = blend * arc + (1.0 - blend) * free[cc_arc]

    if ridge_at_y is not None:
        free = np.where(on_tent, np.maximum(free, ridge_at_y), free)
    genu = int(cc_arc[-1]) if len(cc_arc) >= 8 else int(y_ant)
    free = _narrow_front(free, curves["vault"], genu, int(curves["y_last"]), mm_y, mm_z)
    free = np.clip(free, bottom, top)

    available = np.maximum(top - bottom, 0.0)
    raw_height = np.minimum(np.clip(top - free, 0.0, _FALX_HEIGHT_MAX_MM / mm_z), available)

    u, arc_mm = _arc_samples(curves["y_first"], curves["y_last"], y_index, top,
                             mm_y, mm_z, target_mm, 8, 320, z_other=free)
    nu = len(u)
    top_u = np.interp(u, y_index, top)
    raw_u = np.interp(u, y_index, raw_height)
    avail_u = np.interp(u, y_index, available)

    core_u = (u >= y_post) & (u <= y_ant)
    if ridge_at_y is not None:
        core_u |= np.interp(u, y_index,
                            (on_tent & curves["open_run"]).astype(float)) > 0.5
    height_u = raw_u.copy()
    if core_u.sum() >= 2:
        floor_u = np.minimum(_FALX_HEIGHT_MIN_MM / mm_z, avail_u[core_u])
        height_u[core_u] = _limit_growth(raw_u[core_u], floor=floor_u)
    height_u = np.minimum(height_u, avail_u)
    core_idx = np.where(core_u)[0]
    if len(core_idx):
        for i in range(core_idx[-1] + 1, nu):
            height_u[i] = min(height_u[i], height_u[i - 1] * _GROWTH_LIMIT)
        for i in range(core_idx[0] - 1, -1, -1):
            height_u[i] = min(height_u[i], height_u[i + 1] * _GROWTH_LIMIT)

    cut = np.zeros_like(u)
    for d, r in (((u - u[0]) * mm_y, _FALX_END_ROUND_MM[0]),
                 ((u[-1] - u) * mm_y, _FALX_END_ROUND_MM[1])):
        near = d < r
        cut[near] = np.maximum(cut[near],
                               r - np.sqrt(np.maximum(r * r - (r - d[near]) ** 2, 0.0)))
    if ridge_at_y is not None:
        joined = (on_tent & curves["open_run"]).astype(float)
        cut = cut * (1.0 - np.clip(np.interp(u, y_index, joined), 0.0, 1.0))
    if ridge_at_y is not None:
        pinned = np.interp(u, y_index, on_tent.astype(float)) > 0.5
        height_u = np.where(pinned, np.minimum(raw_u, avail_u), height_u)
    free_u = top_u - height_u + cut / mm_z
    top_u = top_u - cut / mm_z
    height_u = np.maximum(top_u - free_u, 0.0)

    cc_run = np.zeros(len(y_index), dtype=float)
    cc_run[cc_arc] = 1.0
    cc_u = np.where(np.interp(u, y_index, cc_run) > 0.5)[0]
    if len(cc_u) >= 2:
        base_u = np.interp(u[cc_u], y_index, cc_base)
        free_u[cc_u] = base_u + np.maximum.accumulate(free_u[cc_u] - base_u)
        height_u = np.maximum(top_u - free_u, 0.0)

    height_mm = float(np.mean(height_u) * mm_z)
    height_max_mm = float(np.max(height_u) * mm_z)
    nv = _sample_counts(max(float(height_u.max()) * mm_z, target_mm), target_mm, 4, 96)
    t = np.linspace(0.0, 1.0, nv)[None, :]
    zz = free_u[:, None] + t * height_u[:, None]

    yy = np.repeat(u[:, None], nv, axis=1)
    smooth = shell_repair.winslow_smooth(np.stack([yy * mm_y, zz * mm_z], axis=-1))
    yy, zz = smooth[..., 0] / mm_y, smooth[..., 1] / mm_z
    if ridge_at_y is not None:
        seated = np.maximum(zz[:, 0], np.interp(yy[:, 0], y_index, ridge_at_y))
        zz[:, 0] = np.where(yy[:, 0] <= ridge_y.max(), seated, zz[:, 0])
    xx = np.full_like(zz, curves["x_falx"])
    verts = frame.to_world(np.stack([xx, yy, zz], axis=-1).reshape(-1, 3))

    tall = height_u >= _FALX_HEIGHT_MIN_MM / mm_z
    keep = np.repeat((tall[:-1] & tall[1:])[:, None], nv - 1, axis=1)
    verts, quads = _prune_unused(verts, _grid_quads(nu, nv, keep=keep))
    verts, quads = _drop_slivers(verts, quads, 0.05 * target_mm)
    verts = relax_sheet(verts, quads)
    verts, _ = untangle_sheet(verts, quads)
    ends = frame.to_world(np.array([[curves["x_falx"], 0.0, 0.0], [curves["x_falx"] + 1.0, 0.0, 0.0]]))
    normal = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    verts = flatten_to_plane(verts, (ends[0], normal))

    quality = _report_quality("falx", verts, quads)
    logger.info("falx: %d quads (reference deck %d), height %.1fmm max / %.1fmm mean "
                "(published body height %.1f+/-3.9), area %.1fcm^2 "
                "(reference deck %.1f, published %.1f+/-7.7), growth ratio %.2f",
                len(quads), FALX_REF_QUADS, height_max_mm, height_mm, FALX_HEIGHT_MM,
                _sheet_area_cm2(verts, quads), FALX_REF_AREA_CM2, FALX_AREA_CM2,
                quality["max_growth_ratio"])
    return verts, quads


from engine.paths import DATA_DIR
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_FALX = REFERENCE_DIR / "falx_reference.k"
REFERENCE_TENTORIUM = REFERENCE_DIR / "tentorium_reference.k"
_HUMAN_VAULT_MIN_MM = 100.0  # the reference sheets are adult human; smaller vaults keep the old builder
_SEAT_TOUCH_MM = 3.0  # reference falx boundary nodes this close to the reference tentorium are its seat
_FALX_OUTLINE_SMOOTH_MM = 2.5  # attached margin smoothing along the vault, on top of the field
                               # smoothing in _smooth_outline (6 mm here alone shrank the arch)
_FALX_OUTLINE_RESMOOTH = 0.5  # after each push back inside the vault, smooth again at this share
                              # of the outline smoothing (0.25 left a kink at each push)
_FALX_FREE_SMOOTH_SIGMA_MM = 4.0
_FALX_BONE_BAND_MM = 1.0  # the attached margin may reach this far past the cavity on its plane
_FALX_OUTLINE_FIELD_MM = 3.0  # smoothing of the cavity's signed distance the margin is traced on
_FALX_OUTLINE_INSET_MM = 0.4  # ... and how far inside it
# free margin clearance from the convex envelope of the callosum (mm), against position along
# the CC (0 = back of the splenium, 1 = front of the genu): almost touching at the splenium,
# then drawing steadily away toward the genu (user, 2026-10-08: the posterior CC must be the
# closest, the anterior the farthest; the falx is deepest behind and narrows in front)
_FALX_CLEARANCE_KNOTS = ((-0.1, 3.5), (0.05, 3.0), (0.35, 3.5), (0.65, 6.0), (0.95, 9.0),
                         (1.08, 10.0))
_FALX_UNDER_GENU_DEG = 35.0  # under the genu the margin leaves the envelope once it turns this far back
_FALX_DROP_DEG = 35.0  # ... heads down and back at this angle from vertical, and curls round to
                       # meet the anterior fossa floor head-on
_FALX_STEM_THICK_MM = 6.0  # height of the stem resting on the straight sinus
_FALX_STEM_TIP_MM = 6.0  # the stem tip rounds off over this length
_FALX_CORNER_ROUND_MM = 4.0  # fillet where the attached margin meets the tentorium


def read_k_shells(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and 4-node shells of an LS-DYNA keyword file (free or fixed format)."""
    nodes: dict[int, list[float]] = {}
    shells: list[list[int]] = []
    block = None
    with open(path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("*"):
                block = line.strip().upper()
                continue
            if line.startswith("$") or not line.strip():
                continue
            if block == "*NODE":
                if "," in line:
                    parts = line.split(",")
                    nodes[int(parts[0])] = [float(p) for p in parts[1:4]]
                else:
                    nodes[int(line[:8])] = [float(line[8 + 16 * k:24 + 16 * k]) for k in range(3)]
            elif block == "*ELEMENT_SHELL":
                parts = line.replace(",", " ").split()
                shells.append([int(p) for p in parts[2:6]])
    ids = np.array(sorted(nodes))
    lookup = {n: k for k, n in enumerate(ids)}
    verts = np.array([nodes[n] for n in ids], dtype=float)
    quads = np.array([[lookup[n] for n in s] for s in shells], dtype=np.int64)
    return verts, quads


def boundary_loop(quads: np.ndarray) -> np.ndarray:
    """The single boundary loop of a disc-like quad sheet, in order."""
    edges = np.sort(np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]],
                                    quads[:, [2, 3]], quads[:, [3, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    neighbours: dict[int, list[int]] = {}
    for a, b in unique[count == 1]:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    start = min(neighbours)
    loop, previous = [start], None
    while True:
        nxt = [n for n in neighbours[loop[-1]] if n != previous]
        if not nxt or nxt[0] == start:
            break
        previous = loop[-1]
        loop.append(nxt[0])
        if len(loop) > len(neighbours):
            # the outline touches itself (a boundary node with more than two boundary
            # neighbours): the walk circles a sub-loop that never returns to the start
            pinch = [n for n, nb in neighbours.items() if len(nb) > 2]
            raise ValueError(f"boundary_loop: outline touches itself at node(s) {pinch[:5]}")
    return np.array(loop)


def _loop_slice(loop_pts: np.ndarray, i: int, j: int) -> np.ndarray:
    """Points from index i to j walking forward around a closed loop."""
    n = len(loop_pts)
    idx = [(i + k) % n for k in range(((j - i) % n) + 1)]
    return loop_pts[idx]


_REF_CACHE: dict[str, dict] = {}
FALX_DEBUG: dict[str, np.ndarray] = {}  # last subject's falx curves and 2D nodes (mm), for checking


def _reference_falx() -> Optional[dict]:
    """The reference falx as a 2D mesh (AP, SI up) with its three landmarks and edges."""
    if "falx" in _REF_CACHE:
        return _REF_CACHE["falx"]
    if not (REFERENCE_FALX.exists() and REFERENCE_TENTORIUM.exists()):
        return None
    verts, quads = read_k_shells(REFERENCE_FALX)
    tent, _tq = read_k_shells(REFERENCE_TENTORIUM)
    centre = verts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(verts - centre, full_matrices=False)
    # in-plane axes: the reference decks are x anterior, z inferior (tentorium has larger z)
    ap = np.array([1.0, 0.0, 0.0]) - vt[2] * vt[2][0]
    ap /= np.linalg.norm(ap)
    up = np.cross(vt[2], ap)
    if (tent.mean(axis=0) - centre) @ up > 0:
        up = -up
    flat = np.stack([(verts - centre) @ ap, (verts - centre) @ up], axis=1)
    tent_flat_dist = np.min(np.linalg.norm(verts[:, None, :] - tent[None, ::7, :], axis=2), axis=1)

    loop = boundary_loop(quads)
    pts = flat[loop]
    if np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]) < 0:
        loop, pts = loop[::-1], pts[::-1]  # counter-clockwise
    touching = tent_flat_dist[loop] <= _SEAT_TOUCH_MM
    runs, n_runs = ndimage.label(np.r_[touching, touching])  # doubled to see wrapped runs
    longest = max(range(1, n_runs + 1), key=lambda r: (runs == r).sum())
    seat_idx = np.where(runs == longest)[0] % len(loop)
    # seat runs posterior -> anterior; its anterior end is the apex
    s0, s1 = seat_idx[0], seat_idx[-1]
    if pts[s0, 0] > pts[s1, 0]:
        s0, s1 = s1, s0
    # tip: most anterior-inferior point of the rest of the boundary
    rest = np.ones(len(loop), bool)
    rest[seat_idx] = False
    score = np.where(rest, pts[:, 0] - pts[:, 1], -np.inf)
    # the crista galli corner is the sharpest point near the anterior-inferior end
    step = 6
    a = np.roll(pts, step, 0) - pts
    b = np.roll(pts, -step, 0) - pts
    angle = np.degrees(np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1)
                                                          * np.linalg.norm(b, axis=1)), -1, 1)))
    front = rest & (pts[:, 0] > np.percentile(pts[:, 0], 60)) & (pts[:, 1] < np.median(pts[:, 1]))
    tip = int(np.argmin(np.where(front, angle, np.inf))) if front.any() else int(np.argmax(score))
    # three arcs between the landmarks: seat (corner-apex), free margin (apex-tip), attached
    # margin (tip-corner, over the vault); each returned posterior -> anterior
    n = len(loop)

    def walk(i, j):
        return [(i + k) % n for k in range((j - i) % n + 1)]

    if s0 not in walk(s1, tip):
        outer, seat, free = pts[walk(tip, s0)][::-1], pts[walk(s0, s1)], pts[walk(s1, tip)]
        ids = {"outer": loop[walk(tip, s0)][::-1], "seat": loop[walk(s0, s1)], "free": loop[walk(s1, tip)]}
    else:
        outer, seat, free = pts[walk(s0, tip)], pts[walk(s1, s0)][::-1], pts[walk(tip, s1)][::-1]
        ids = {"outer": loop[walk(s0, tip)], "seat": loop[walk(s1, s0)][::-1], "free": loop[walk(tip, s1)][::-1]}
    ref = {"flat": flat, "quads": quads, "outer": outer, "seat": seat, "free": free, "ids": ids,
           "tip": pts[tip], "apex": pts[s1], "corner": pts[s0],
           # how far the crista galli sits behind the front of the falx opening, as a share
           # of the corner -> frontal pole length
           "tip_behind_opening_frac": float((free[:, 0].max() - pts[tip][0])
                                            / max(outer[:, 0].max() - pts[s0][0], 1e-6)),
           "area_cm2": _sheet_area_cm2(verts, quads)}
    _REF_CACHE["falx"] = ref
    return ref


def _vault_outline(vault: np.ndarray) -> np.ndarray:
    """Longest closed contour of the 2D vault mask, in (y, z) voxel index."""
    from skimage import measure

    padded = np.pad(vault.astype(np.float32), 1)
    contours = measure.find_contours(ndimage.gaussian_filter(padded, 0.7), 0.5)
    return max(contours, key=len) - 1.0


def _smooth_outline(mask: np.ndarray, to_mm: np.ndarray, sigma_mm: float,
                    inset_mm: float) -> np.ndarray:
    """Longest contour `inset_mm` inside a 2D mask, taken on its signed distance smoothed by
    `sigma_mm`: smooth like a smoothed curve, but it does not shrink round convex arcs the
    way smoothing the polyline does, so the margin stays against the bone."""
    from skimage import measure

    padded = np.pad(mask, 2)
    sdf = (fast_kernels.distance_transform_edt(padded, sampling=to_mm)
           - fast_kernels.distance_transform_edt(~padded, sampling=to_mm))
    sdf = ndimage.gaussian_filter(sdf, sigma_mm / to_mm)
    contours = measure.find_contours(sdf, inset_mm)
    return max(contours, key=len) - 2.0


def _smooth_polyline(points: np.ndarray, sigma_mm: float, to_mm: np.ndarray,
                     step_mm: float = 0.5, hold_mm: Optional[float] = None) -> np.ndarray:
    """Resample an open polyline at a fixed step and smooth it, ends held (the smoothing fades
    in over `hold_mm` from each end, 2 sigma by default)."""
    mm = points * to_mm
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(mm, axis=0), axis=1))])
    keep = np.r_[True, np.diff(d) > 1e-9]
    mm, d = mm[keep], d[keep]
    t = np.linspace(0.0, d[-1], max(int(d[-1] / step_mm), 3))
    out = np.stack([np.interp(t, d, mm[:, k]) for k in range(2)], axis=1)
    if sigma_mm > 0 and len(out) > 4:
        smooth = ndimage.gaussian_filter1d(out, sigma_mm / step_mm, axis=0, mode="nearest")
        hold = 2.0 * sigma_mm if hold_mm is None else max(float(hold_mm), step_mm)
        w = np.clip(np.minimum(t, t[-1] - t) / hold, 0.0, 1.0)[:, None]
        out = w * smooth + (1.0 - w) * out
    return out / to_mm


def _smooth_inside(points: np.ndarray, inside: np.ndarray, sigma_mm: float, to_mm: np.ndarray,
                   margin_mm: float = 0.3, resmooth: float = 0.25, passes: int = 6,
                   hold_mm: Optional[float] = None) -> np.ndarray:
    """Smooth an open polyline (ends held) without letting it leave the `inside` mask."""
    sdf = (fast_kernels.distance_transform_edt(inside, sampling=to_mm)
           - fast_kernels.distance_transform_edt(~inside, sampling=to_mm))
    sdf = ndimage.gaussian_filter(sdf, 1.0)
    grad = np.gradient(sdf, *to_mm)
    out = _smooth_polyline(points, sigma_mm, to_mm, hold_mm=hold_mm)
    for _ in range(int(passes)):
        depth = ndimage.map_coordinates(sdf, out.T, order=1, mode="nearest")
        push = np.clip(margin_mm - depth, 0.0, None)
        push[[0, -1]] = 0.0  # ends are held: they join other curves
        if push.max() < 0.05:
            break
        g = np.stack([ndimage.map_coordinates(gk, out.T, order=1, mode="nearest") for gk in grad], 1)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        out = out + push[:, None] * g / to_mm
        out = _smooth_polyline(out, sigma_mm * resmooth, to_mm, hold_mm=hold_mm)
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# Rater profiles (lab templates only): each lab sheet's outline was drawn by a different rater
# at their own distance from the anatomy . A profile is a
# callable taking the outline's points (world mm) and returning how far (mm) to move each one
# outward in the sheet's plane; None (default) leaves the anatomical outline as built.
_FALX_OUTLINE_ADJUST = None
_TENT_OUTLINE_ADJUST = None
_TENT_HEIGHT_ADJUST = None   # callable: tent points (world mm) -> lift (mm, +superior)
_FALX_PLANE = None   # (point, unit normal) in world mm: the rater's falx plane, not the midsagittal one
_FALX_PLANE_COLUMN = None  # falx plane at this index of the label grid's L-R axis (lab templates: 128.5)


def _offset_outline(poly_mm: np.ndarray, to_world, adjust) -> np.ndarray:
    """Move a closed outline (2D, mm) outward by adjust(world points) mm, smoothly."""
    d = np.asarray(adjust(to_world(poly_mm)), dtype=float)
    if not np.any(d):
        return poly_mm
    seg = np.linalg.norm(np.diff(np.vstack([poly_mm, poly_mm[:1]]), axis=0), axis=1)
    step = max(float(np.median(seg)), 1e-3)
    d = ndimage.gaussian_filter1d(d, max(1.0 / step, 1.0), mode="wrap")
    t = np.roll(poly_mm, -1, axis=0) - np.roll(poly_mm, 1, axis=0)
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    x, y = poly_mm[:, 0], poly_mm[:, 1]
    if 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:   # clockwise: flip
        n = -n
    return poly_mm + d[:, None] * n


# Opening of the falx as one smooth arch over the callosal mass instead of a clearance that
# follows its outline (the lab templates' free margin is an ellipse to 1.3-1.5 mm rms): centred
# on the callosal envelope front to back, reaching _FALX_ARCH_PAD_MM past each end, its top
# _FALX_ARCH_TOP_MM above the envelope's top and its half-height _FALX_ARCH_HEIGHT_FRAC of the
# envelope's height; joined to a band _FALX_ARCH_MIN_CLEAR_MM round the envelope, so behind the
# splenium the margin drops close round it to the stem. None: the clearance knots below.
_FALX_ARCH_PAD_MM = None
_FALX_ARCH_TOP_MM = 5.5
_FALX_ARCH_HEIGHT_FRAC = 0.43
_FALX_ARCH_MIN_CLEAR_MM = 2.5
_FALX_ARCH_FRONT_PAD_MM = None  # reach past the genu, if not _FALX_ARCH_PAD_MM
_FALX_ARCH_FRONT_CLEAR_MM = None  # band round the genu, widening to it from the splenium's (None: the same)
_FALX_ARCH_BAND_CLOSE_MM = None  # band round the callosum closed by this radius (None: its convex hull)
_FALX_STEM_REACH_MM = None  # stem carried this far forward under the splenium past the tentorial ridge (None: off)
_FALX_TIP_ROUND_MM = None  # fillet where the attached margin meets the free margin at the front (None: sharp)


def _callosal_arch(hull: np.ndarray, to_mm: np.ndarray,
                   blocking: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """_callosal_ring with the arch rule: (closed ring, signed field in mm, >= 0 outside)."""
    from skimage import measure

    ij = np.argwhere(hull)
    y0, y1 = ij[:, 0].min() * to_mm[0], ij[:, 0].max() * to_mm[0]
    z0, z1 = ij[:, 1].min() * to_mm[1], ij[:, 1].max() * to_mm[1]
    front = _FALX_ARCH_PAD_MM if _FALX_ARCH_FRONT_PAD_MM is None else _FALX_ARCH_FRONT_PAD_MM
    lo, hi = y0 - _FALX_ARCH_PAD_MM, y1 + front     # +y is anterior
    a = 0.5 * (hi - lo)
    b = _FALX_ARCH_HEIGHT_FRAC * (z1 - z0)
    cy, cz = 0.5 * (lo + hi), z1 + _FALX_ARCH_TOP_MM - b
    yy = np.arange(hull.shape[0])[:, None] * to_mm[0]
    zz = np.arange(hull.shape[1])[None, :] * to_mm[1]
    ellipse = ((yy - cy) / a) ** 2 + ((zz - cz) / b) ** 2 <= 1.0
    if _FALX_ARCH_BAND_CLOSE_MM is not None and blocking is not None:
        # the band follows the callosum itself (closed over small dents), not its convex hull,
        # so behind and under the splenium the margin hooks in and the stem cradles it
        r = max(int(round(_FALX_ARCH_BAND_CLOSE_MM / float(min(to_mm)))), 1)
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        shape = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(blocking, r), disk)[r:-r, r:-r])
    else:
        shape = hull
    clear = np.full((hull.shape[0], 1), float(_FALX_ARCH_MIN_CLEAR_MM))
    if _FALX_ARCH_FRONT_CLEAR_MM is not None:
        # the band widens from the splenium to the genu (the free margin keeps further off the
        # genu and rostrum than off the splenium)
        t = np.clip((yy - y0) / max(y1 - y0, 1e-6), 0.0, 1.0)
        clear = _FALX_ARCH_MIN_CLEAR_MM + (_FALX_ARCH_FRONT_CLEAR_MM - _FALX_ARCH_MIN_CLEAR_MM) * t
    near = fast_kernels.distance_transform_edt(~shape, sampling=to_mm) <= clear
    opening = ellipse | near
    level = (fast_kernels.distance_transform_edt(~opening, sampling=to_mm)
             - fast_kernels.distance_transform_edt(opening, sampling=to_mm))
    ring = max(measure.find_contours(level, 0.0), key=len)
    return ring, level


def _callosal_ring(blocking: np.ndarray, to_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Opening of the falx: a level set around the convex envelope of the callosal mass, wider
    behind (over the splenium) than in front (over the genu). Returns the closed ring and the
    clearance field (mm, >= 0 outside the opening)."""
    from skimage import measure
    from skimage.morphology import convex_hull_image

    hull = convex_hull_image(blocking)
    y_cc = np.where(blocking.any(axis=1))[0]
    if _FALX_ARCH_PAD_MM is not None:
        return _callosal_arch(hull, to_mm, blocking)
    t = (np.arange(blocking.shape[0]) - y_cc.min()) / max(float(np.ptp(y_cc)), 1.0)
    knots = np.asarray(_FALX_CLEARANCE_KNOTS, dtype=float)
    clear = PchipInterpolator(knots[:, 0], knots[:, 1], extrapolate=False)(t)
    clear = np.where(t < knots[0, 0], knots[0, 1], np.where(t > knots[-1, 0], knots[-1, 1], clear))
    level = fast_kernels.distance_transform_edt(~hull, sampling=to_mm) - clear[:, None]
    ring = max(measure.find_contours(level, 0.0), key=len)
    return ring, level


def _march(start: np.ndarray, heading: np.ndarray, mask: np.ndarray, to_mm: np.ndarray,
           reach_mm: float = 200.0) -> np.ndarray:
    """Step from `start` along `heading` (unit, mm) in 0.5 mm steps while inside `mask`."""
    pt = np.array(start, dtype=float)
    for _ in range(int(reach_mm / float(min(to_mm)))):
        nxt = pt + 0.5 * heading / to_mm
        i, j = int(round(nxt[0])), int(round(nxt[1]))
        if not (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and mask[i, j]):
            break
        pt = nxt
    return pt


def _falx_front_end(ahead: np.ndarray, vault: np.ndarray, to_mm: np.ndarray):
    """Under the genu the free margin leaves the callosal envelope once it has turned
    _FALX_UNDER_GENU_DEG back, then runs down and back to the anterior fossa floor (crista
    galli). Returns (index on `ahead`, that point, the bend, the floor point)."""
    n_r = len(ahead)
    k_front = int(np.argmax(ahead[:, 0]))
    step = ndimage.gaussian_filter1d(np.gradient(ahead * to_mm, axis=0), 2.0, axis=0, mode="wrap")
    turned = np.degrees(np.arctan2(-step[:, 0], -step[:, 1]))
    past = np.where((np.arange(n_r) > k_front) & (turned > _FALX_UNDER_GENU_DEG))[0]
    k_under = int(past[0]) if len(past) else k_front
    under = ahead[k_under]
    angle = np.radians(_FALX_DROP_DEG)
    heading = np.array([-np.sin(angle), -np.cos(angle)])
    drop_mm = (under[1] - _march(under, np.array([0.0, -1.0]), vault, to_mm)[1]) * to_mm[1]
    bend = _march(under, heading, vault, to_mm)
    reach = np.linalg.norm((bend - under) * to_mm)
    bend = under + heading * min(reach, 0.5 * drop_mm / np.cos(angle)) / to_mm
    floor_pt = _march(bend, np.array([0.0, -1.0]), vault, to_mm)
    return k_under, under, bend, floor_pt


_FALX_WEDGE_DEG = 25.0  # two margins leaving a corner closer than this ...
_FALX_WEDGE_GAP_MM = 2.5  # ... are cut back to where they are this far apart (a paver can't fill a sliver)


def _cut_wedge(a: np.ndarray, b: np.ndarray, to_mm: np.ndarray, gap_mm: float = _FALX_WEDGE_GAP_MM):
    """Two polylines leaving the same corner (a[0] ~ b[0]): if they leave at a sliver angle,
    drop both up to where they are _FALX_WEDGE_GAP_MM apart. Returns (a, b) cut, unchanged
    when the corner is open enough."""
    am, bm = a * to_mm, b * to_mm

    def along(p):
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])

    da, db = along(am), along(bm)
    if da[-1] < 1e-6 or db[-1] < 1e-6:
        return a, b
    reach = min(8.0, 0.5 * da[-1], 0.5 * db[-1])
    va = am[np.searchsorted(da, reach)] - am[0]
    vb = bm[np.searchsorted(db, reach)] - bm[0]
    cos = float(va @ vb / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
    if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) >= _FALX_WEDGE_DEG:
        return a, b
    from scipy.spatial import cKDTree

    gap, _ = cKDTree(bm).query(am)
    open_ = np.where((gap >= gap_mm) & (da <= 0.6 * da[-1]))[0]
    if not len(open_):
        return a, b
    i = int(open_[0])
    j = int(np.argmin(np.linalg.norm(bm - am[i], axis=1)))
    return a[i:], b[j:]


def _ring_branches(ring: np.ndarray):
    """The callosal ring split at its top: (ahead: top -> genu -> under it ..., behind: top ->
    splenium -> under it ...)."""
    n_r = len(ring)
    top_r = int(np.argmax(ring[:, 1]))
    front_r = int(np.argmax(ring[:, 0]))
    fwd = 1 if (front_r - top_r) % n_r < n_r / 2 else -1
    return (ring[(top_r + fwd * np.arange(n_r)) % n_r],
            ring[(top_r - fwd * np.arange(n_r)) % n_r])


_FALX_SEAT_GAP_MM = 1.0  # open-back species: the falx base sits this far above the cerebellum / tectum
_FALX_OPEN_BASE_MM = 3.0  # ... and its free margin comes down to this height above that base


def _infratentorial_seat(canon, curves, vault, blocking, to_mm):
    """Base for a falx with no midline tentorium (pig, rat, mouse; Klintworth 1968): the top of
    the cerebellum and brainstem (tectum) on the falx plane, a small gap above it, from the back
    of the cavity forward to the back of the callosal envelope. Returned in the same form as the
    tentorial ridge, (y, z) in voxel index, so the falx is drawn the same way on it."""
    x_falx = curves["x_falx"]
    sl = canon[int(round(x_falx))]
    infra = np.isin(sl, (_CEREBELLUM, _BRAINSTEM)) & vault
    if not infra.any():
        return None
    ring, _level = _callosal_ring(blocking, to_mm)
    _ahead, behind = _ring_branches(ring)
    y_front = float(behind[:max(len(behind) * 3 // 5, 2), 0].min())   # back of the envelope
    z = np.arange(sl.shape[1])
    top = np.where(infra, z[None, :], -1).max(axis=1).astype(float)
    cols = np.where(top >= 0)[0]
    cols = cols[cols <= y_front]
    if len(cols) < 4:
        return None
    ys = np.arange(cols.min(), int(np.floor(y_front)) + 1)
    zs = np.interp(ys, cols, top[cols])
    zs = ndimage.gaussian_filter1d(zs, max(3.0 / to_mm[0], 1.0), mode="nearest")
    zs = zs + 0.5 + _FALX_SEAT_GAP_MM / to_mm[1]
    # only the stretch with cavity above it that reaches the callosum: behind it the vault comes
    # down to the cerebellum, or (rodents) the colliculi and cerebellum reach the dorsal surface
    # between the hemispheres, which is where the falx ends
    lift = int(np.ceil(_FALX_STEM_THICK_MM / to_mm[1]))
    room = np.array([vault[int(y), int(min(max(round(zz) + lift, 0), vault.shape[1] - 1))]
                     for y, zz in zip(ys, zs)])
    if not room[-1] or room.sum() < 4:
        return None
    first = len(room) - int(np.argmin(room[::-1])) if not room.all() else 0
    if len(ys) - first < 4:
        return None
    return ys[first:].astype(float), zs[first:]

def falx_on_midline(
    labels: np.ndarray, affine: np.ndarray, target_mm: float = 1.0,
    tentorium_ridge: Optional[tuple[np.ndarray, np.ndarray]] = None,
    cc_gap_mm: float = _FALX_FREE_MARGIN_GAP_MM,
    open_back: bool = False, seat_from_tissue: bool = False,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The falx as one flat sickle on the subject's midsagittal plane, meshed with planar quads.

    `open_back` is for species whose tentorium does not reach the midline (pig, rat, mouse;
    Klintworth 1968): there is no tentorial ridge to rest on, so behind the splenium the free
    margin turns down and back to the vault above the cerebellum, and the falx has no stem.

    Every curve comes from this subject's labels:
    - attached margin: the inner vault on the midline from the posterior tentorial corner over
      the vertex, down the frontal bone and back along the anterior fossa floor, smoothed into
      one arch and held inside the vault;
    - base: the whole tentorial ridge (straight sinus), so the stem runs forward under the
      splenium to the front of the tentorium at the midline;
    - free margin: from the stem tip back along the top of the stem, then a smooth arch around
      the convex envelope of the corpus callosum (wide behind, close over the genu), curling
      under the genu and down to the anterior fossa floor.
    The outline is meshed with square quads and one boundary layer (mesh/planar_quads.py).
    """
    grid_plane = None
    if _FALX_PLANE_COLUMN is not None:
        # the lab templates: the falx is two voxel columns at a fixed index of the label grid
        # (128 and 129 of 256), not a plane fitted to the brain (LAB_SHAPE_SOURCES.md)
        a0 = np.asarray(affine, dtype=float)
        col = int(np.argmax(np.abs(a0[0, :3])))
        idx = np.zeros(3); idx[col] = float(_FALX_PLANE_COLUMN)
        n_g = a0[:3, col] / np.linalg.norm(a0[:3, col])
        grid_plane = (a0[:3, :3] @ idx + a0[:3, 3], n_g)
    from engine.mesh import planar_quads

    if not (open_back or seat_from_tissue) and (tentorium_ridge is None or len(tentorium_ridge[0]) < 4):
        return None
    labels, affine, x_pin = align_to_midsagittal(labels, affine)
    frame = _Frame(affine, labels.shape)
    canon = frame.view(np.asarray(labels))
    curves = _falx_curves(frame, canon, x_pin)
    if curves is None:
        return None
    mm_y, mm_z = frame.spacing[1], frame.spacing[2]
    to_mm = np.array([mm_y, mm_z])
    vault = curves["vault"]
    if np.ptp(np.where(vault.any(axis=1))[0]) * mm_y < _HUMAN_VAULT_MIN_MM:
        return None
    # outline, march and hold-inside use the cavity on the falx plane itself, grown by the band
    # the attached margin may sit in bone (the superior sagittal sinus and the dura's band);
    # the slab cavity reached further and left rim quads deep in bone that the trim notched
    plane_parts, n_plane = ndimage.label(curves["vault_plane"])
    if n_plane and _FALX_PLANE_CAVITY:
        plane_vault = ndimage.binary_fill_holes(
            plane_parts == 1 + int(np.argmax(np.bincount(plane_parts.ravel())[1:])))
        band = fast_kernels.distance_transform_edt(~plane_vault, sampling=to_mm) <= _FALX_BONE_BAND_MM
        vault = vault & band

    blocking = np.asarray(curves["blocking"], dtype=bool)
    parts, n_parts = ndimage.label(blocking)
    if not n_parts:
        return None
    blocking = parts == 1 + int(np.argmax(np.bincount(parts.ravel())[1:]))

    outline = _smooth_outline(vault, to_mm, _FALX_OUTLINE_FIELD_MM, _FALX_OUTLINE_INSET_MM)
    if open_back or seat_from_tissue:
        # no midline tentorial ridge to rest on (open midline, or a tentorium built in a
        # pitched frame): the base follows the top of the cerebellum and tectum
        tentorium_ridge = _infratentorial_seat(canon, curves, vault, blocking, to_mm)
        if tentorium_ridge is None:
            return None
    ridge_y, ridge_z = (np.asarray(a, dtype=float) for a in tentorium_ridge)
    order = np.argsort(ridge_y)
    ridge_y, ridge_z = ridge_y[order], ridge_z[order]
    seat_vox = np.stack([ridge_y, ridge_z], axis=1)

    # attached margin: walk the vault outline from the tentorial corner over the vertex
    corner_k = int(np.argmin(np.linalg.norm((outline - seat_vox[0]) * to_mm, axis=1)))
    n_out = len(outline)
    top_k = int(np.argmax(outline[:, 1]))
    direction = 1 if (top_k - corner_k) % n_out < n_out / 2 else -1
    walked = outline[(corner_k + direction * np.arange(n_out)) % n_out]
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(walked * to_mm, axis=0), axis=1))])
    pole = int(np.argmax(np.where(dist < 0.75 * dist[-1], walked[:, 0], -np.inf)))

    # opening around the corpus callosum
    ring, level = _callosal_ring(blocking, to_mm)
    n_r = len(ring)
    top_r = int(np.argmax(ring[:, 1]))
    front_r = int(np.argmax(ring[:, 0]))
    fwd = 1 if (front_r - top_r) % n_r < n_r / 2 else -1
    ahead = ring[(top_r + fwd * np.arange(n_r)) % n_r]   # top -> genu -> under the genu ...
    behind = ring[(top_r - fwd * np.arange(n_r)) % n_r]  # top -> splenium -> under it ...

    k_under, under, bend, floor_pt = _falx_front_end(ahead, vault, to_mm)
    past_pole = np.arange(len(walked)) > pole
    tip_i = int(np.argmin(np.where(past_pole & (dist < 0.95 * dist[-1]),
                                   np.linalg.norm((walked - floor_pt) * to_mm, axis=1),
                                   np.inf)))
    outer = _smooth_inside(walked[:tip_i + 1], vault, _FALX_OUTLINE_SMOOTH_MM, to_mm,
                           resmooth=_FALX_OUTLINE_RESMOOTH, passes=12)
    tip = outer[-1]

    # lowest point of the opening above the ridge, per A-P column (for a stem under the splenium)
    z_cols = np.arange(level.shape[1], dtype=float)
    # ... or the lowest brain tissue on the falx plane there (the tectum under the splenium)
    plane_tissue = np.isin(canon[int(round(curves["x_falx"]))], _BRAIN)
    no_room = (level < 0.5) | (fast_kernels.distance_transform_edt(~plane_tissue, sampling=to_mm) < 1.0)

    def opening_floor(y):
        yi = np.clip(np.rint(np.atleast_1d(y)).astype(int), 0, level.shape[0] - 1)
        base = np.interp(np.atleast_1d(y), ridge_y, ridge_z)
        inside = no_room[yi] & (z_cols[None, :] > base[:, None] + 0.5)
        return np.where(inside.any(axis=1), np.argmax(inside, axis=1).astype(float), np.inf)

    if _FALX_STEM_REACH_MM and not open_back:
        # the stem goes on forward under the splenium past the end of the tentorial ridge (along
        # the great cerebral vein), cradling it, while there is room under the opening
        ext = ridge_y[-1] + np.arange(0.5, float(_FALX_STEM_REACH_MM) + 0.25, 0.5) / mm_y
        room = (opening_floor(ext) - ridge_z[-1]) * mm_z
        ok = np.cumprod(room >= 2.5).astype(bool)
        if ok.any():
            ridge_y = np.concatenate([ridge_y, ext[ok]])
            ridge_z = np.concatenate([ridge_z, np.full(int(ok.sum()), ridge_z[-1])])
            seat_vox = np.stack([ridge_y, ridge_z], axis=1)

    # stem: the base runs along the whole tentorial ridge; its top edge sits a stem height above
    stem_tip = seat_vox[-1]

    def stem_top(y):
        y = np.minimum(y, stem_tip[0])
        rise = 0.6 + 0.4 * _smoothstep((stem_tip[0] - y) * mm_y / _FALX_STEM_TIP_MM)
        # no straight sinus without a midline tentorium: the base is a thin rim, not a stem
        thick = _FALX_STEM_THICK_MM if not open_back else _FALX_OPEN_BASE_MM
        top = np.interp(y, ridge_y, ridge_z) + rise * thick / mm_z
        if _FALX_STEM_REACH_MM and not open_back:   # never into the opening round the callosum
            top = np.minimum(top, opening_floor(y).reshape(np.shape(y)) - 0.5 / mm_z)
        return top

    above = behind[:, 1] - stem_top(behind[:, 0])
    low = np.where(above <= 0.0)[0]
    k_meet = int(low[0]) if len(low) else n_r - 1
    meet = behind[k_meet]
    # rounded tip: a half-disc cap across the stem height, its front at the end of the ridge
    cap_z = float(np.interp(stem_tip[0], ridge_y, ridge_z))
    r_mm = 0.5 * (float(stem_top(stem_tip[0])) - cap_z) * mm_z
    centre = np.array([stem_tip[0] - r_mm / mm_y, cap_z + r_mm / mm_z])
    # dense (about 0.3 mm) so the round tip is a curve, not a polygon; it is kept out of the
    # free-margin smoothing below, whose 4 mm sigma is wider than the cap and made it lumpy
    ang = np.radians(np.linspace(-80.0, 90.0, max(int(np.pi * r_mm / 0.3), 12)))
    cap = centre + np.stack([np.cos(ang) * r_mm / mm_y, np.sin(ang) * r_mm / mm_z], axis=1)
    ys = np.arange(centre[0] - 0.5 / mm_y, meet[0], -0.5 / mm_y)
    stem_edge = np.concatenate([cap[-1:], np.stack([ys, stem_top(ys)], axis=1)]) if len(ys) else cap[-1:]
    ridge_keep = ridge_y < centre[0]
    seat_vox = np.concatenate([seat_vox[ridge_keep], [[centre[0], float(np.interp(centre[0], ridge_y, ridge_z))]]])

    s_drop = np.linspace(0.0, 1.0, max(int(2 * np.linalg.norm((tip - under) * to_mm)), 3))[1:-1, None]
    down = (1 - s_drop) ** 2 * under + 2 * s_drop * (1 - s_drop) * bend + s_drop ** 2 * tip
    raw = np.concatenate([stem_edge, behind[:k_meet + 1][::-1], ahead[1:k_under + 1], down, tip[None, :]])
    raw = _smooth_polyline(raw, 0.0, to_mm)
    # smoothing must not pull the margin into the opening's clearance: it is held outside it
    # the same way the attached margin is held inside the vault (falling back to the raw
    # margin point by point put a step wherever the check flipped)
    free = _smooth_inside(raw, level >= -0.5, _FALX_FREE_SMOOTH_SIGMA_MM, to_mm,
                          margin_mm=0.0, resmooth=0.5, passes=12, hold_mm=1.5)
    free = np.concatenate([cap[:-1], free])  # the round tip joins at its top ...
    free = _smooth_polyline(free, 1.0, to_mm, hold_mm=0.5)  # ... with a 1 mm fillet there

    seat = _smooth_polyline(np.concatenate([outer[:1], seat_vox[1:]]), 2.0, to_mm)

    def joint(p, q):
        n_j = max(int(np.linalg.norm((q - p) * to_mm) / 0.5), 2)
        return p + np.linspace(0.0, 1.0, n_j)[:, None] * (q - p)

    # sliver corners (a steep tentorium along the vault, a base just under the vault, a drop
    # to the floor along the frontal floor) are cut back to a fillable width
    wedge_mm = max(_FALX_WEDGE_GAP_MM, 2.0 * float(target_mm))
    outer_c, seat_c = _cut_wedge(outer, seat, to_mm, wedge_mm)
    if len(outer_c) < len(outer):
        outer, seat = np.concatenate([joint(seat_c[0], outer_c[0])[:-1], outer_c]), seat_c
    tip_o, tip_f = _cut_wedge(outer[::-1], free[::-1], to_mm, wedge_mm)
    if len(tip_o) < len(outer):
        outer = tip_o[::-1]
        free = np.concatenate([tip_f[::-1], joint(tip_f[0], tip_o[0])[1:]])
    def along(line):
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line * to_mm, axis=0), axis=1))])

    # a free margin that nearly touches the vault (rat: the callosal envelope reaches up to the
    # vault) pinches the falx into two lobes joined by a sliver: keep the front lobe, which
    # then ends at the pinch
    from scipy.spatial import cKDTree

    gap, near_o = cKDTree(outer * to_mm).query(free * to_mm)
    d_free = along(free)
    pinched = np.where((gap < max(_FALX_WEDGE_GAP_MM, 1.5 * float(target_mm))) & (d_free > 0.15 * d_free[-1])
                       & (d_free < 0.85 * d_free[-1]))[0]
    if len(pinched) and along(outer[int(near_o[pinched[-1]]):])[-1] > 0.5 * along(outer)[-1]:
        kf = int(pinched[-1])
        o2, f2 = _cut_wedge(outer[int(near_o[kf]):], free[kf:], to_mm, wedge_mm)
        logger.info("falx: free margin pinches against the vault; the falx ends there "
                    "(%.1f mm of the attached margin dropped)", along(outer)[-1] - along(o2)[-1])
        outer, free, seat = o2, f2, np.zeros((0, 2))
        polygon = np.vstack([outer, free[::-1][1:], joint(free[0], outer[0])[1:-1]]) * to_mm
    else:
        if _FALX_TIP_ROUND_MM:
            # round the front tip (crista galli), where the attached margin coming back along
            # the anterior fossa floor meets the free margin coming down from the genu
            reach = min(float(_FALX_TIP_ROUND_MM), 0.4 * along(outer)[-1], 0.4 * along(free)[-1])
            i_o = len(outer) - 1 - int(np.searchsorted(along(outer[::-1]), reach))
            i_f = len(free) - 1 - int(np.searchsorted(along(free[::-1]), reach))
            n_t = max(int(2.0 * reach / 0.5), 6)
            s_t = np.linspace(0.0, 1.0, n_t)[1:-1, None]
            tip_arc = (1 - s_t) ** 2 * outer[i_o] + 2 * s_t * (1 - s_t) * outer[-1] + s_t ** 2 * free[i_f]
            outer = np.concatenate([outer[:i_o + 1], tip_arc, free[i_f][None, :]])
            free = free[:i_f + 1]
        # round the posterior corner, where the vault meets the tentorium at an acute angle
        i_out = int(np.searchsorted(along(outer), _FALX_CORNER_ROUND_MM))
        i_seat = int(np.searchsorted(along(seat), _FALX_CORNER_ROUND_MM))
        s_c = np.linspace(0.0, 1.0, 9)[1:-1, None]
        fillet = (1 - s_c) ** 2 * seat[i_seat] + 2 * s_c * (1 - s_c) * outer[0] + s_c ** 2 * outer[i_out]
        outer = np.concatenate([seat[i_seat][None, :], fillet, outer[i_out:]])
        seat = seat[i_seat:]
        polygon = np.vstack([outer, free[::-1][1:], seat[::-1][:-1]]) * to_mm
    if _FALX_OUTLINE_ADJUST is not None:
        polygon = _offset_outline(polygon, lambda yz_mm: frame.to_world(np.column_stack(
            [np.full(len(yz_mm), curves["x_falx"]), yz_mm / to_mm])), _FALX_OUTLINE_ADJUST)
    nodes_mm, quads, _on = planar_quads.pave(polygon, float(target_mm))
    if planar_quads.quad_sj(nodes_mm[quads]).min() < 0.5:
        # tight corners (small heads, cut wedges): local search on the worst quads, outline
        # nodes sliding along the outline, judged flat
        nodes_mm = planar_quads.improve_lifted(
            nodes_mm, quads, polygon, _on,
            lambda xy: np.column_stack([xy, np.zeros(len(xy))]), float(target_mm), target=0.6)
    FALX_DEBUG.update(outer=outer * to_mm, seat=seat * to_mm, free=free * to_mm, yz_mm=nodes_mm,
                      ring=ring * to_mm, quads=quads, to_mm=to_mm, x_falx=curves["x_falx"],
                      vault_slab=curves["vault"], vault_hold=vault, blocking=blocking,
                      level=level, outline=outline * to_mm, ridge=np.stack([ridge_y, ridge_z], 1) * to_mm,
                      slice=canon[int(round(curves["x_falx"]))], frame=frame,
                      drop=np.stack([under, bend, floor_pt, tip]) * to_mm)

    yz = nodes_mm / to_mm
    verts = frame.to_world(np.column_stack([np.full(len(yz), curves["x_falx"]), yz]))
    ends = frame.to_world(np.array([[curves["x_falx"], 0.0, 0.0], [curves["x_falx"] + 1.0, 0.0, 0.0]]))
    normal = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
    plane_used = _FALX_PLANE if _FALX_PLANE is not None else (grid_plane if grid_plane is not None
                                                              else (ends[0], normal))
    verts = flatten_to_plane(verts, plane_used)
    quality = _report_quality("falx", verts, quads)
    logger.info("falx: %d planar quads on the midline, area %.1fcm^2 (reference %s %.1f), "
                "growth ratio %.2f", len(quads), _sheet_area_cm2(verts, quads),
                REFERENCE_FALX.name, FALX_REF_AREA_CM2, quality["max_growth_ratio"])
    return verts, quads


TENTORIUM_TISSUE_REPORT: dict[str, dict] = {}  # last build's retract_sheet_from report
TENT_DEBUG: dict = {}  # last paved build: incisura apex (rule, final), tectum, need start (voxel y)
TENTORIUM_PAVED: set[str] = set()  # holds "tentorium" when the last build used the paved planform
_TENT_PAVE_MIN_WIDTH_MM = 60.0  # human-sized tentoria get the paved planform
_TENT_NOTCH_HALF_FRAC = 0.36  # incisura half-width at the front, share of the half-width
                              # (reference Tentorium_male.k: ~40 mm across of ~110 mm)
_TENT_NOTCH_APEX_FRAC = 0.72  # incisura apex, share of the tentorium's length from the back
_TENT_FISSURE_CLOSE_MM = 4.0  # half-width of the left-right closing across the fissure
_TENT_LEAF_TIP_MM = 8.0  # narrowest leaf width at the anterior tips (reference ~8-10 mm)
_TENT_LIFT_SMOOTH_MM = 2.5  # smoothing of the tentorial surface the paved planform is lifted onto
_TENT_POST_SMOOTH_MM = 3.0  # smoothing after the floor/ceiling clip (0 = off)
_TENT_RIM_INSET_MM = 0.0  # attached margin held this far inside the cavity (0: at its wall)
_TENT_CEREBRUM_GAP_MM = None  # tent below the cerebrum's underside (None: _TENTORIUM_CLEARANCE_MM)
_TENT_TIP_ROUND = False  # leaf tips rounded forward instead of cut square across
_TENT_TIP_FRONT_MM = None  # leaf tips end this far ahead of the midbrain's front at the sheet's height (None: off)
_TENT_TIP_BLOCK_MM = None  # straight front: leaf tips as a block this wide beside the incisura (None: as the rim)
_TENT_NOTCH_BS_GAP_MM = None  # incisura = midbrain section grown by this gap (None: the U from the apex share)
_TENT_NOTCH_BAND_MM = 2.0  # ... the section taken this far above and below the tent
_TENT_NOTCH_BS_MODE = "full"  # "full": the U is the grown section; "apex": only its bottom comes from it
_TENT_STRAIGHT_FRONT = False  # anterolateral margin straight from the widest point to the leaf tip
_TENT_STRAIGHT_TIP_MM = 6.0  # ... ending this far short of the tip
_TENT_FRONT_SAG_MM = 0.0  # ... bowed in this far at its middle (mm; the lab templates' margin is concave)
_TENT_FRONT_SAG_R_MM = None  # ... on the right (+x) side only (None: _TENT_FRONT_SAG_MM)
_TENT_FRONT_SAG_L_MM = None  # ... on the left side only
_TENT_STRAIGHT_START_FRAC = None  # ... leaving the rim this share of the way from the widest point to the tip (None: at it)
_TENT_RIDGE_HOLD = False  # hold the midline ridge at its gap after the surface smoothing (which flattens it)
_TENT_RIDGE_GAP_MM = None  # gap under the cerebrum on the midline (None: the gap above everywhere)
_TENT_RIDGE_HALF_MM = 10.0  # ... widening to the lateral gap over this distance from the midline
_TENT_GAP_FRACTION = None  # landmark rule: tent this share of the way up from the cerebellum top to the cerebrum underside (overrides the gap above)
_TENT_APEX_TECTUM_MM = None  # incisura apex this far behind the top of the midbrain (tectum) on the midline (None: _TENT_NOTCH_APEX_FRAC)
_TENT_SHARE_MID = None  # tent at this share of the cerebellum-to-cerebrum gap on the midline (None: off)
_TENT_SHARE_SIDE = 0.33  # ... and this share away from it (lab templates: 0.50 / 0.29-0.39, both sexes)
_TENT_SHARE_HALF_MM = 10.0  # ... changing over this distance from the midline
_TENT_RIM_LINE_W = None  # outline along the gap-share line this far (mm) from the wall (None: the inset cavity footprint)
_TENT_RIM_LINE_SHARE = 0.43  # ... at this cerebellum-cerebrum gap share
_TENT_RIM_LINE_CB_MM = 9.0  # ... and only where the cerebellum is within this distance (not the middle fossa)
_TENT_LINE_BACK_ONLY = False  # with the line outline: only behind the widest point (front corners as before)
_TENT_CB_GROW_MM = None  # outline = cerebellum footprint grown this far (mm), the lab's way (None: off)
_TENT_CB_GROW_WALL_MM = 2.0  # ... kept this far off the cavity wall at the sheet's height
_TENT_CB_GROW_FRONT_MM = None  # ... grown this far in front instead (None: the same)
_TENT_LINE_HULL = False  # ... the line's region taken convex behind where one side ends: one back margin
                         # across the midline (sinuses to the confluence), no step where the sides end apart
_TENT_OUTLINE_FROM_LINE = False  # True (with _TENT_RIM_LINE_W): each side's margin is the line itself (no reference
                                 # half-width profile, no corner rounding, back where the line closes)
_TENT_CORNER_SHARE = None  # gap share over the anterolateral corners only (lab attached margin: 0.41-0.43; None: off)
_TENT_CORNER_FROM_FRAC = 0.6  # ... from this share of the cerebellum's A-P length forward
_TENT_CORNER_FROM_MM = 25.0  # ... and this far off the midline
_TENT_CB_GUARD_MM = None  # after all smoothing the tent stays this far over the cerebellum's top (None: off)
_TENT_CB_GUARD_MID_MM = None  # ... this far on the midline instead (None: the same)
_TENT_CB_GUARD_MID_HALF_MM = 8.0  # ... blending to the side value over this distance
_TENT_ROLL_MM = 0.0  # side-to-side tilt: lift (mm) at 50 mm toward +x, the same drop toward -x
_TENT_HUG_ALL = False  # the surface hugs the cerebellum's top everywhere (not under the cerebrum near the midline)
_TENT_RIM_DROOP_MM = 0.0  # the sheet bends down this far (mm) at its attached margin ...
_TENT_RIM_DROOP_DECAY_MM = 6.0  # ... fading over this distance inward
_TENT_RIM_DROOP_BACK_ONLY = False  # ... behind the widest point only
_TENT_LINE_TIPS_PROFILE = False  # line outline: rows past the line's front keep the profile (no finger tips)
_TENT_RIM_BAND_MM = 3.0  # the rim follows the cavity within this height of the tent surface (the attached margin)
_TENT_APEX_NEED_MM = -2.0  # the apex never lies further forward than this past the first row where the midbrain
                           # needs an opening (that row is already grown by _INCISURA_CLEARANCE_MM)
_TENT_APEX_OVER_TECTUM = False  # True: the apex is not pulled back behind the midbrain; the sheet passes over the
                                # back of the tectum, lifted clear of it (the lab templates: ~2.5 mm above it)


def _paved_tentorium(frame, x_mid, y0, y1, y_index, cb_left, cb_right, posterior_cap,
                     anterior_cap, floor_half, notch_required, half_widest, mid, ceiling,
                     clearance, free_edge, target_mm, floor, open_frac=0.0, ridge_hold=None,
                     notch_bs=None, tectum_y=None, has_cb=None, share_target=None,
                     share_weight=None):
    """Tentorium planform (outer rim + U-shaped incisura) paved with quads, then lifted onto
    the tentorial surface `mid`. Returns world verts, quads and the midline ridge. With
    `open_frac` > 0 (pig, rat, mouse) it is two lateral leaves instead, each from the rim in to
    `open_frac` of its half-width, with no incisura."""
    from scipy.spatial import cKDTree

    from engine.mesh import planar_quads

    mm_x, mm_y, mm_z = frame.spacing
    ys = np.arange(y0, y1 + 1, dtype=float)

    def half(edge):
        h = np.abs(np.interp(ys, y_index, edge) - x_mid)
        if _TENT_OUTLINE_FROM_LINE and (_TENT_RIM_LINE_W is not None or _TENT_CB_GROW_MM is not None):
            # the line is already smooth; a ratio limit grown from its one-voxel back tip cut
            # the back corners off diagonally
            h = np.maximum(h, 2.0 / mm_x)
        else:
            h = _limit_growth(h, floor=2.0 / mm_x)
        return np.maximum(
            h * (1.0 - _TENT_POSTERIOR_CORNER * (1.0 - np.interp(ys, y_index, posterior_cap)))
            * (1.0 - _TENT_ANTERIOR_CORNER * (1.0 - np.interp(ys, y_index, anterior_cap))),
            floor_half)

    right, left = half(cb_right), half(cb_left)
    need = np.interp(ys, np.arange(len(notch_required)), notch_required)
    # incisura apex (straight sinus / vein of Galen): at the reference's share of the length,
    # but never behind the start of the opening the midbrain needs
    apex_y = y0 + _TENT_NOTCH_APEX_FRAC * (y1 - y0)
    if tectum_y is not None and _TENT_APEX_TECTUM_MM is not None:
        # the apex (vein of Galen into the straight sinus) sits behind the colliculi
        apex_y = float(tectum_y) - _TENT_APEX_TECTUM_MM / mm_y
    if notch_bs is not None and _TENT_NOTCH_BS_MODE == "apex" and (notch_bs > 0).any():
        # the U's bottom at the back of the midbrain (plus the gap), not a share of the length
        apex_y = float(np.argmax(notch_bs > 0))
        notch_bs = None
    apex_rule = apex_y
    if (need > 0).any() and not _TENT_APEX_OVER_TECTUM:
        apex_y = min(apex_y, float(ys[np.argmax(need > 0)]) + _TENT_APEX_NEED_MM / mm_y)
    TENT_DEBUG.update(apex_rule_y=apex_rule, apex_y=apex_y, tectum_y=tectum_y, y0=y0, y1=y1,
                      frame=frame, x_mid=x_mid, ys=ys, need=need, mid=mid,
                      need_start=float(ys[np.argmax(need > 0)]) if (need > 0).any() else None)
    frac = np.clip((ys - apex_y) / max(y1 - apex_y, 1.0), 0.0, 1.0)
    hw = _TENT_NOTCH_HALF_FRAC * half_widest * np.sqrt(frac)
    hw = np.where(ys > apex_y, np.maximum(hw, need), 0.0)
    # a smooth U that only widens toward the front
    hw = ndimage.gaussian_filter1d(hw, max(3.0 / mm_y, 1.0), mode="nearest")
    hw = np.where(ys > apex_y, np.maximum.accumulate(hw), 0.0)
    if notch_bs is not None and _TENT_NOTCH_BS_MODE == "cap" and (notch_bs > 0).any():
        # the lab templates' incisura: the U as above, but its walls never wider than the
        # midbrain section plus the gap (parallel walls forward to the tips, on the brainstem)
        hb = np.interp(ys, np.arange(len(notch_bs), dtype=float), notch_bs)
        hb = ndimage.gaussian_filter1d(hb, max(1.5 / mm_y, 1.0), mode="nearest")
        hw = np.where((ys > apex_y) & (hb > 0), np.minimum(hw, np.maximum(hb, need)), hw)
        notch_bs = None
    if notch_bs is not None and (notch_bs > 0).any():
        # the U is the grown midbrain section: apex at its back, widening with it to the front
        hw = np.interp(ys, np.arange(len(notch_bs), dtype=float), notch_bs)
        apex_y = float(ys[np.argmax(hw > 0)]) if (hw > 0).any() else apex_y
        hw = ndimage.gaussian_filter1d(hw, max(1.5 / mm_y, 1.0), mode="nearest")
        hw = np.where(ys > apex_y, np.maximum.accumulate(np.maximum(hw, need)), 0.0)
    min_leaf = _TENT_LEAF_TIP_MM / mm_x
    # the leaves' anterior tips stay beside the open incisura (reference), so the rim never
    # closes in on it: widen the rim where the U needs room, smoothly
    sigma = max(4.0 / mm_y, 1.0)
    right = np.maximum(right, ndimage.gaussian_filter1d(hw + min_leaf, sigma, mode="nearest"))
    left = np.maximum(left, ndimage.gaussian_filter1d(hw + min_leaf, sigma, mode="nearest"))
    if _TENT_STRAIGHT_FRONT:
        # anterolateral margin along the petrous ridge: a straight edge from the widest point to
        # the leaf tip beside the incisura (the lab templates), never outside the rim built above
        def straight(h, sag_mm=0.0):
            iw = int(np.argmax(h))
            if _TENT_STRAIGHT_START_FRAC is not None:   # leaves the rim this share of the way to the tip
                iw = int(np.searchsorted(ys, ys[iw] + _TENT_STRAIGHT_START_FRAC * (ys[-1] - ys[iw])))
            # ... ending short of the tip, which keeps its rounding (a sharp tip folds its quads)
            ke = int(np.searchsorted(ys, ys[-1] - _TENT_STRAIGHT_TIP_MM / mm_y))
            if iw >= ke - 2:
                return h
            end = h[ke] if _TENT_TIP_BLOCK_MM is None else hw[ke] + _TENT_TIP_BLOCK_MM / mm_x
            line = np.interp(ys, [ys[iw], ys[ke]], [h[iw], end])
            if sag_mm:
                # the lab templates' anterolateral margin bows in (along the temporal lobe)
                tt = np.clip((ys - ys[iw]) / max(ys[ke] - ys[iw], 1e-6), 0.0, 1.0)
                line = line - float(sag_mm) / mm_x * 4.0 * tt * (1.0 - tt)
            if _TENT_TIP_BLOCK_MM is None:
                line = np.where(ys > ys[ke], h, line)
            else:
                # the lab templates' leaf tip: a block this wide beside the incisura, its
                # outer edge carried forward to the square front
                line = np.where(ys > ys[ke], hw + _TENT_TIP_BLOCK_MM / mm_x, line)
            # ... and never inside the incisura plus a leaf (that would narrow the incisura,
            # and with it the free-edge lift along the ridge)
            line = np.maximum(line, ndimage.gaussian_filter1d(hw + min_leaf, sigma, mode="nearest"))
            if _TENT_TIP_BLOCK_MM is None:
                out = np.where(ys > ys[iw], np.minimum(h, line), h)
            else:
                # where the rim built from the anatomy has run out (the line outline ends
                # before the tips) the straight edge wins
                spent = h <= hw + min_leaf + 0.5 / mm_x
                out = np.where(ys > ys[iw], np.where(spent, line, np.minimum(h, line)), h)
                return ndimage.gaussian_filter1d(out, max(1.5 / mm_y, 1.0), mode="nearest")
            # no corner where the straight edge leaves the rim
            return np.minimum(ndimage.gaussian_filter1d(out, sigma, mode="nearest"), h)
        sag_r = _TENT_FRONT_SAG_MM if _TENT_FRONT_SAG_R_MM is None else _TENT_FRONT_SAG_R_MM
        sag_l = _TENT_FRONT_SAG_MM if _TENT_FRONT_SAG_L_MM is None else _TENT_FRONT_SAG_L_MM
        right, left = straight(right, sag_r), straight(left, sag_l)
    hw = np.maximum(np.minimum(hw, np.minimum(right, left) - min_leaf), 0.0)
    front = ys >= apex_y

    def tip_cap(xa, xb):
        """Rounded leaf tip from the rim end (xa) to the incisura end (xb) at the front row,
        bulging forward by half the tip's width (empty: the square cut)."""
        if not _TENT_TIP_ROUND:
            return np.zeros((0, 2))
        s = np.linspace(0.0, 1.0, 9)[1:-1, None]
        bulge = 0.5 * abs(xb - xa) * mm_x / mm_y
        p0, p2 = np.array([xa, ys[-1]]), np.array([xb, ys[-1]])
        p1 = np.array([0.5 * (xa + xb), ys[-1] + 2.0 * bulge])
        return (1 - s) ** 2 * p0 + 2 * s * (1 - s) * p1 + s ** 2 * p2

    polygon = np.vstack([
        np.stack([x_mid + right[::-1], ys[::-1]], axis=1),           # right rim, front -> back
        np.stack([x_mid - left, ys], axis=1),                         # left rim, back -> front
        tip_cap(x_mid - left[-1], x_mid - hw[-1]),                    # left leaf tip
        np.stack([x_mid - hw[front][::-1], ys[front][::-1]], axis=1),  # incisura, front -> apex
        np.stack([x_mid + hw[front][1:], ys[front][1:]], axis=1),      # apex -> front
        tip_cap(x_mid + hw[-1], x_mid + right[-1]),                   # right leaf tip
    ]) * np.array([mm_x, mm_y])
    if open_frac > 0:
        inner_r, inner_l = open_frac * right, open_frac * left
        polygons = [np.vstack([np.stack([x_mid + right[::-1], ys[::-1]], axis=1),
                               np.stack([x_mid + inner_r, ys], axis=1)]),
                    np.vstack([np.stack([x_mid - left, ys], axis=1),
                               np.stack([x_mid - inner_l[::-1], ys[::-1]], axis=1)])]
        polygons = [p * np.array([mm_x, mm_y]) for p in polygons]
        hw = np.zeros_like(hw)
    else:
        polygons = [polygon]
    polygons = [p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]] for p in polygons]
    if _TENT_OUTLINE_ADJUST is not None:
        def _tent_world(xy_mm):
            xx, yy = xy_mm[:, 0] / mm_x, xy_mm[:, 1] / mm_y
            zz = ndimage.map_coordinates(mid, np.stack([xx, yy]), order=1, mode="nearest")
            return frame.to_world(np.stack([xx, yy, zz], axis=1))
        polygons = [_offset_outline(p, _tent_world, _TENT_OUTLINE_ADJUST) for p in polygons]
    paved = [planar_quads.pave(p, float(target_mm)) for p in polygons]

    # a smooth tent: steep steps in the surface would shear the lifted quads
    sigma = (_TENT_LIFT_SMOOTH_MM / mm_x, _TENT_LIFT_SMOOTH_MM / mm_y)
    surface = ndimage.gaussian_filter(mid, sigma=sigma, mode="nearest")
    surface = np.clip(surface, floor + clearance, np.maximum(ceiling - clearance, floor + clearance))
    room = np.maximum(ceiling - clearance, mid)
    if _TENT_POST_SMOOTH_MM > 0:
        # the cerebellum top and cerebrum underside are voxel staircases, and the clip puts the
        # steps back as creases (the reference adult head: 1028 quads warped over 30 degrees, worst SJ -0.24; after
        # this 5 and 0.65, the tent moving 0.18 mm on average). The CSF slot is carved around
        # the finished sheet, so clearance is restored there.
        post = (_TENT_POST_SMOOTH_MM / mm_x, _TENT_POST_SMOOTH_MM / mm_y)
        surface = ndimage.gaussian_filter(surface, sigma=post, mode="nearest")
        room = ndimage.gaussian_filter(room, sigma=post, mode="nearest")
    if ridge_hold is not None:
        target, weight = ridge_hold
        surface = surface + weight * np.maximum(target - surface, 0.0)
        surface = ndimage.gaussian_filter(surface, sigma=(1.0 / mm_x, 1.0 / mm_y), mode="nearest")
        room = np.maximum(room, surface)
    if share_target is not None:
        # the gap share sets the height wherever the cerebellum is under the sheet
        surface = surface + np.clip(share_weight, 0.0, 1.0) * (share_target - surface)
        room = np.maximum(room, surface)
    if _TENT_CB_GUARD_MM is not None:
        # last word: the smoothing above pulls the hump over the vermis down onto (or into) the
        # cerebellum; the tent keeps this gap over its top (the lab templates: ~1.5-2 mm)
        g_mm = np.full(surface.shape[0], float(_TENT_CB_GUARD_MM))
        if _TENT_CB_GUARD_MID_MM is not None:
            # higher over the vermis (the lab templates: 2.8 mm on the midline, 1.1-1.4 over the sides)
            off_g = np.abs(np.arange(surface.shape[0], dtype=float) - float(x_mid)) * mm_x
            g_mm = g_mm + (_TENT_CB_GUARD_MID_MM - g_mm) * (1.0 - _smoothstep(off_g / _TENT_CB_GUARD_MID_HALF_MM))
        guard = _soft_max(surface, floor + g_mm[:, None] / mm_z, 0.5 / mm_z)
        if has_cb is not None:   # only over the cerebellum itself (floor is filled in beyond it)
            over = ndimage.gaussian_filter(has_cb.astype(float), sigma=(1.5 / mm_x, 1.5 / mm_y), mode="nearest")
            guard = surface + np.clip(over, 0.0, 1.0) * (guard - surface)
        surface = guard
        room = np.maximum(room, surface)
    incisura = np.vstack([np.stack([x_mid - hw[front], ys[front]], axis=1),
                          np.stack([x_mid + hw[front], ys[front]], axis=1)]) * np.array([mm_x, mm_y])
    if open_frac > 0:   # the leaves' medial free edges
        incisura = np.vstack([np.stack([x_mid + inner_r, ys], axis=1),
                              np.stack([x_mid - inner_l, ys], axis=1)]) * np.array([mm_x, mm_y])
    free_tree = cKDTree(incisura)
    rim_tree = None
    if _TENT_RIM_DROOP_MM:
        # the attached margin only: outline points away from the incisura
        outer = np.concatenate(polygons)
        d_inc, _ = free_tree.query(outer)
        outer = outer[d_inc > 2.0]
        if _TENT_RIM_DROOP_BACK_ONLY:   # the back (transverse sinus) only: behind the widest row
            outer = outer[outer[:, 1] < float(ys[int(np.argmax(right + left))]) * mm_y]
        rim_tree = cKDTree(outer) if len(outer) else None

    def height(xy_mm):
        """Tentorial height (voxels) under planform points (mm)."""
        xx, yy = xy_mm[:, 0] / mm_x, xy_mm[:, 1] / mm_y
        zz = ndimage.map_coordinates(surface, np.stack([xx, yy]), order=1, mode="nearest")
        d_free, _ = free_tree.query(xy_mm)
        lift = (_TENT_FREE_EDGE_LIFT_MM / mm_z * np.exp(-(d_free / _TENT_FREE_EDGE_DECAY_MM) ** 2)
                * np.interp(yy, np.arange(len(free_edge), dtype=float), free_edge))
        head_room = ndimage.map_coordinates(room, np.stack([xx, yy]), order=1, mode="nearest")
        out = _soft_min(zz + lift, head_room, 1.0)
        if rim_tree is not None:
            # the lab templates' sheet bends down to its attachment (transverse sinus, petrous ridge)
            d_rim, _ = rim_tree.query(xy_mm)
            out = out - float(_TENT_RIM_DROOP_MM) * np.exp(-d_rim / float(_TENT_RIM_DROOP_DECAY_MM)) / mm_z
        if _TENT_ROLL_MM:
            # the lab templates' slab is painted tilted side to side (on the cerebellum on
            # one side, up in the gap on the other): this many mm at 50 mm off the midline
            out = out + float(_TENT_ROLL_MM) * (xx - float(x_mid)) * mm_x / 50.0 / mm_z
        if _TENT_HEIGHT_ADJUST is not None:
            out = out + np.asarray(_TENT_HEIGHT_ADJUST(frame.to_world(np.stack([xx, yy, out], axis=1))),
                                   dtype=float) / mm_z
        return out

    def lifted_mm(xy_mm):
        return np.column_stack([xy_mm, height(xy_mm) * mm_z])

    all_nodes, all_quads, offset = [], [], 0
    for poly, (nodes_mm, quads, on_outline) in zip(polygons, paved):
        nodes_mm = planar_quads.improve_lifted(nodes_mm, quads, poly, on_outline, lifted_mm,
                                               float(target_mm))
        all_nodes.append(nodes_mm)
        all_quads.append(quads + offset)
        offset += len(nodes_mm)
    nodes_mm, quads = np.concatenate(all_nodes), np.concatenate(all_quads)
    xx, yy = nodes_mm[:, 0] / mm_x, nodes_mm[:, 1] / mm_y
    verts = frame.to_world(np.stack([xx, yy, height(nodes_mm)], axis=1))

    ridge_y = ys[ys <= apex_y]
    ridge_z = ndimage.map_coordinates(surface, np.stack([np.full(len(ridge_y), float(x_mid)), ridge_y]),
                                      order=1, mode="nearest")
    TENT_RIDGE_FRAME.update(frame=frame, x_mid=float(x_mid))   # to read the ridge off the finished sheet
    return verts, quads, (ridge_y, ridge_z)


@shell_repair.timed
def tentorium_sheet(
    labels: np.ndarray, affine: np.ndarray, target_mm: float = 1.0,
    fused: bool = True, open_frac: float = 0.35,
) -> Optional[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]:
    """Build the tentorium.

    `fused` (human, primates, carnivores): one crescent with the incisura cut from the midbrain
    and the straight-sinus ridge (paved planform when it is wide enough). Not fused (pig, rat,
    mouse; Klintworth 1968): two separate leaves over the lateral part of the cerebrum and
    cerebellum only, their medial free edge `open_frac` of the half-width from the midline, no
    incisura and no straight sinus."""
    labels, affine, x_pin = align_to_midsagittal(labels, affine)
    frame = _Frame(affine, labels.shape)
    canon = frame.view(np.asarray(labels))
    cerebellum = canon == _CEREBELLUM
    if not cerebellum.any():
        return None
    cerebrum = np.isin(canon, _CEREBRUM)
    intracranial = np.isin(canon, _INTRACRANIAL)
    if not intracranial.any():
        intracranial = np.isin(canon, _BRAIN)
    mm_x, mm_y, mm_z = frame.spacing

    plane = midsagittal_plane(frame, canon, x_pin)
    x_plane = plane[0] if plane is not None else (
        float(x_pin) if x_pin is not None else float(np.median(np.where(cerebellum)[0])))
    x_mid = int(round(x_plane))

    floor_raw, has_cb = _column_top(cerebellum)
    floor = _nearest_fill(floor_raw, has_cb)

    z = np.arange(canon.shape[2])
    # the tentorium lies under the occipital lobes, not up the interhemispheric fissure:
    # close the fissure left-right before reading the cerebrum's underside
    reach = max(int(round(_TENT_FISSURE_CLOSE_MM / mm_x)), 1)
    occipital = cerebrum | ndimage.binary_closing(
        cerebrum, structure=np.ones((2 * reach + 1, 1, 1), dtype=bool))
    above = occipital & (z[None, None, :] > floor[:, :, None])
    has_ceiling = above.any(axis=2)
    ceiling_raw = np.where(above, z[None, None, :], canon.shape[2] + 1).min(axis=2).astype(float)
    clearance = _TENTORIUM_CLEARANCE_MM / mm_z
    ceiling = _nearest_fill(np.where(has_ceiling, ceiling_raw, 0.0), has_ceiling)
    # every clip below is against these two heights; as voxel tops they carry the cerebellar
    # folia and the occipital gyri, and each clip printed them back into the tent as bumps.
    # Clip against smooth envelopes instead: over the folia (closing) and under the gyri
    # (opening). The CSF slot is carved around the finished sheet, so any voxel it grazes
    # becomes CSF.
    if _TENT_ENVELOPE_MM > 0:
        floor = _smooth_envelope(floor, (mm_x, mm_y), upper=True, span_mm=_TENT_ENVELOPE_MM)
        ceiling = _smooth_envelope(ceiling, (mm_x, mm_y), upper=False, span_mm=_TENT_ENVELOPE_MM)

    taut = ndimage.grey_opening(ceiling, size=(1, max(int(round(
        _TENT_CEILING_TAUT_MM / mm_y)), 1)), mode="nearest")
    taut = ndimage.gaussian_filter1d(taut, max(3.0 / mm_y, 1.0), axis=1, mode="nearest")
    gap_cerebrum = (clearance if _TENT_CEREBRUM_GAP_MM is None
                    else _TENT_CEREBRUM_GAP_MM / mm_z)
    ridge_hold = None
    if _TENT_RIDGE_GAP_MM is not None:
        # the straight sinus: the falx holds the midline up, closer under the cerebrum
        off_mid = np.abs(np.arange(floor.shape[0], dtype=float) - x_plane) * mm_x
        gap_cerebrum = (_TENT_RIDGE_GAP_MM / mm_z + (gap_cerebrum - _TENT_RIDGE_GAP_MM / mm_z)
                        * _smoothstep(off_mid / _TENT_RIDGE_HALF_MM))[:, None]
        if _TENT_RIDGE_HOLD:   # (target height, weight) for the paved surface, over the cerebellum
            weight = (1.0 - _smoothstep(off_mid / _TENT_RIDGE_HALF_MM))[:, None] \
                * has_cb[int(round(x_plane))][None, :]
            ridge_hold = [taut - _TENT_RIDGE_GAP_MM / mm_z, weight]
    share_target = share_weight = None
    if _TENT_SHARE_MID is not None or _TENT_CORNER_SHARE is not None:
        # the lab templates sit at a share of the CSF gap between the cerebellum and the
        # cerebrum (in 3D: distance to the cerebellum over the sum of the distances to both, the
        # way the lab sheets were measured): on average half on the midline, about a third over
        # the sides, and near half along the attached margin, on both templates. Per column the
        # height is the lowest above the cerebellum's top where the share is reached; it is the
        # last word on the paved surface's height, and the planform is cut as before.
        nx_, ny_ = floor.shape
        off = np.abs(np.arange(nx_, dtype=float) - x_plane) * mm_x
        share2 = np.zeros((nx_, ny_))
        weight2 = np.zeros((nx_, ny_))
        if _TENT_SHARE_MID is not None:
            share2 += (_TENT_SHARE_SIDE + (_TENT_SHARE_MID - _TENT_SHARE_SIDE)
                       * (1.0 - _smoothstep(off / _TENT_SHARE_HALF_MM)))[:, None]
            weight2 = ndimage.gaussian_filter(has_cb.astype(float), sigma=(2.0 / mm_x, 2.0 / mm_y), mode="nearest")
        if _TENT_CORNER_SHARE is not None:
            # the anterolateral corners (petrous ridges): from the front share of the
            # cerebellum's length and this far off the midline
            cby = np.where(has_cb.any(axis=0))[0]
            y_c = cby.min() + _TENT_CORNER_FROM_FRAC * (cby.max() - cby.min())
            cy = _smoothstep((np.arange(ny_, dtype=float) - y_c) * mm_y / 10.0)
            cx = _smoothstep((off - _TENT_CORNER_FROM_MM) / 10.0)
            corner = np.clip(cx[:, None] * cy[None, :], 0.0, 1.0)
            share2 = share2 * (1.0 - corner) + _TENT_CORNER_SHARE * corner
            weight2 = np.maximum(weight2, corner)
        d_cb = fast_kernels.distance_transform_edt(~cerebellum, sampling=(mm_x, mm_y, mm_z)).astype(np.float32)
        d_cr = fast_kernels.distance_transform_edt(~cerebrum, sampling=(mm_x, mm_y, mm_z)).astype(np.float32)
        frac = d_cb / np.maximum(d_cb + d_cr, 1e-3)
        del d_cb, d_cr
        zz = np.arange(canon.shape[2], dtype=np.float32)[None, None, :]
        reached = (zz > floor[:, :, None] - 1.0) & (frac >= share2[:, :, None])
        k = np.argmax(reached, axis=2)
        ok = reached.any(axis=2) & (weight2 > 1e-3)
        # sub-voxel: interpolate between the last voxel under the share and the first at it
        xi, yi = np.indices(k.shape)
        k0 = np.maximum(k - 1, 0)
        f0, f1 = frac[xi, yi, k0], frac[xi, yi, k]
        t = np.clip((share2 - f0) / np.maximum(f1 - f0, 1e-3), 0.0, 1.0)
        level = np.where(k > 0, k0 + t, k).astype(float)
        del frac, reached
        by_share = _nearest_fill(np.where(ok, level, 0.0), ok) if ok.any() else floor + clearance
        share_target = np.clip(by_share, floor + clearance, np.maximum(ceiling - clearance, floor + clearance))
        share_target = ndimage.gaussian_filter(share_target, sigma=(1.5 / mm_x, 1.5 / mm_y), mode="nearest")
        share_weight = weight2
        under_cerebrum = np.maximum(taut - gap_cerebrum, floor + clearance)
    elif _TENT_GAP_FRACTION is not None:
        under_cerebrum = floor + clearance + _TENT_GAP_FRACTION * np.maximum(taut - floor - clearance, 0.0)
    else:
        under_cerebrum = np.maximum(taut - gap_cerebrum, floor + clearance)
    hugging = floor + clearance
    cb_cols = np.where(has_cb.any(axis=1))[0]
    half_width = max(float(np.max(np.abs(cb_cols - x_plane))), 1.0) if len(cb_cols) else 1.0
    lateral = np.abs(np.arange(floor.shape[0], dtype=float) - x_plane) / half_width
    w = np.clip((lateral - _TENT_WRAP_START_FRAC)
                / max(_TENT_WRAP_END_FRAC - _TENT_WRAP_START_FRAC, 1e-6), 0.0, 1.0)
    w = (w * w * (3.0 - 2.0 * w))[:, None]
    if _TENT_HUG_ALL:
        # the lab templates' slab is painted on the cerebellum's top everywhere
        w = np.ones_like(w)
    mid = (1.0 - w) * under_cerebrum + w * hugging
    if _TENT_CORNER_SHARE is not None and share_target is not None:
        # over the corners the planform is cut at the share height too, so the attached margin
        # reaches out along the petrous ridge instead of stopping under the temporal lobe
        mid = mid * (1.0 - corner) + share_target * corner

    sigma_xy = max(2.0 / mm_x, 1.0)
    mid = ndimage.gaussian_filter(mid, sigma=(sigma_xy, sigma_xy), mode="nearest")
    mid = np.clip(mid, floor + clearance, np.maximum(ceiling - clearance, floor + clearance))

    ix = np.arange(mid.shape[0])
    left = ix <= x_plane
    for half, order in ((left, slice(None, None, -1)), (~left, slice(None))):
        cols = ix[half][order]
        if len(cols) > 1:
            mid[cols] = np.minimum.accumulate(mid[cols], axis=0)
    edge_lo = np.where(has_cb.any(axis=0), np.argmax(has_cb, axis=0), 0)
    edge_hi = np.where(has_cb.any(axis=0),
                       has_cb.shape[0] - 1 - np.argmax(has_cb[::-1], axis=0),
                       mid.shape[0] - 1)
    yi_all = np.arange(mid.shape[1])
    skirt = np.where(left[:, None], mid[edge_lo, yi_all][None, :],
                     mid[edge_hi, yi_all][None, :])
    outboard = (ix[:, None] < edge_lo[None, :]) | (ix[:, None] > edge_hi[None, :])
    mid = np.where(outboard, skirt, mid)
    xi, yi = np.indices(mid.shape)
    band = max(0, int(round(_TENT_RIM_BAND_MM / mm_z)))
    zi = np.rint(mid).astype(np.int64)
    intra_plane = np.zeros(mid.shape, dtype=bool)
    for dz in range(-band, band + 1):
        intra_plane |= intracranial[xi, yi, np.clip(zi + dz, 0, canon.shape[2] - 1)]

    if _TENT_RIM_INSET_MM > 0:
        # the attached margin this far inside the cavity (a CSF label that takes in the dura and
        # sinus band: template segmentations)
        intra_plane = fast_kernels.distance_transform_edt(
            intra_plane, sampling=(mm_x, mm_y)) > _TENT_RIM_INSET_MM

    if _TENT_RIM_LINE_W is not None:
        # the attached margin runs where the cerebellum-cerebrum gap meets the wall (transverse
        # sinus, petrous ridge): a column is inside the tentorium when, at the height where the
        # gap share reaches _TENT_RIM_LINE_SHARE, the cavity wall is more than W mm away (3D).
        # The lab templates' outlines follow this line (the share 0.43 on both, W 4.3 / 2.5 mm)
        dcb3 = (fast_kernels.distance_transform_edt(~cerebellum, sampling=(mm_x, mm_y, mm_z))
                .astype(np.float32))
        dc = fast_kernels.distance_transform_edt(~cerebrum, sampling=(mm_x, mm_y, mm_z)).astype(np.float32)
        fr = dcb3 / np.maximum(dcb3 + dc, 1e-3)
        del dc
        zz3 = np.arange(canon.shape[2], dtype=np.float32)[None, None, :]
        hit = (zz3 > floor[:, :, None] - 1.0) & (fr >= _TENT_RIM_LINE_SHARE)
        z_line = np.argmax(hit, axis=2)
        has_line = hit.any(axis=2)
        del fr, hit
        wall3 = fast_kernels.distance_transform_edt(intracranial, sampling=(mm_x, mm_y, mm_z)).astype(np.float32)
        xi3, yi3 = np.indices(z_line.shape)
        # the tentorium only spans the gap over the cerebellum: in front of the petrous ridges
        # (middle fossa, under the temporal lobes) the share is met too but the cerebellum is far
        # (judged from above: within this distance of the cerebellum's footprint, so a deep
        # CSF space over the cerebellum is never cut out)
        # (the footprint closed first: the notch between the cerebellar hemispheres at the back,
        # where the falx cerebelli sits, is under the tentorium too)
        r_c = max(int(round(10.0 / mm_x)), 1)
        disk = np.hypot(*np.mgrid[-r_c:r_c + 1, -r_c:r_c + 1]) <= r_c
        cb_fill = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(has_cb, r_c), disk)[r_c:-r_c, r_c:-r_c])
        near_cb = fast_kernels.distance_transform_edt(~cb_fill, sampling=(mm_x, mm_y)) <= _TENT_RIM_LINE_CB_MM
        wall_at = wall3[xi3, yi3, z_line]
        inside_line = has_line & (wall_at > _TENT_RIM_LINE_W) & near_cb
        TENT_DEBUG.update(rim_has_line=has_line, rim_wall=wall_at, rim_near_cb=near_cb,
                          rim_z=z_line, rim_floor=floor.copy(), rim_has_cb=has_cb.copy(), x_mid0=x_mid)
        del wall3, dcb3
        # keep the single piece round the midline (no islands under the temporal lobes)
        lab_, n_ = ndimage.label(inside_line)
        if n_:
            keep = lab_[x_mid][lab_[x_mid] > 0]
            if len(keep):
                inside_line = lab_ == np.bincount(keep).argmax()
        line_plane = ndimage.binary_fill_holes(inside_line)
        if _TENT_LINE_HULL and line_plane.any():
            # only behind the row where both sides still have the line (elsewhere the line's
            # own concavities are the lab's outline)
            from skimage.morphology import convex_hull_image
            rows_l = line_plane[:x_mid].any(axis=0)
            rows_r = line_plane[x_mid + 1:].any(axis=0)
            if rows_l.any() and rows_r.any():
                y_both = max(int(np.argmax(rows_l)), int(np.argmax(rows_r)))
                hull = convex_hull_image(line_plane)
                line_plane[:, :y_both] = hull[:, :y_both]
        if not _TENT_LINE_BACK_ONLY:
            intra_plane = line_plane
    else:
        line_plane = None
    if _TENT_CB_GROW_MM is not None:
        # the lab templates' outline : the
        # cerebellum's footprint from above, grown outward by a set distance and rounded (male
        # 1.5-4.4, female 3-8.5 mm), kept a set distance off the cavity wall at the sheet's height
        r_c = max(int(round(10.0 / mm_x)), 1)
        disk = np.hypot(*np.mgrid[-r_c:r_c + 1, -r_c:r_c + 1]) <= r_c
        cb_fill = ndimage.binary_fill_holes(
            ndimage.binary_closing(np.pad(has_cb, r_c), disk)[r_c:-r_c, r_c:-r_c])
        grow = np.full(cb_fill.shape, float(_TENT_CB_GROW_MM))
        if _TENT_CB_GROW_FRONT_MM is not None and cb_fill.any():
            # further out in front (the leaves run forward along the petrous ridges): blended in
            # from the footprint's centre row to 7.5 mm ahead of it (+y is anterior)
            yc = float(np.argwhere(cb_fill)[:, 1].mean())
            fw = np.clip((np.arange(cb_fill.shape[1], dtype=float) - yc) * mm_y / 15.0 + 0.5, 0.0, 1.0)
            grow = grow + (float(_TENT_CB_GROW_FRONT_MM) - grow) * fw[None, :]
        grown = fast_kernels.distance_transform_edt(~cb_fill, sampling=(mm_x, mm_y)) <= grow
        at_height = np.zeros(mid.shape, dtype=bool)
        for dz in range(-band, band + 1):
            at_height |= intracranial[xi, yi, np.clip(zi + dz, 0, canon.shape[2] - 1)]
        wall_ok = fast_kernels.distance_transform_edt(at_height, sampling=(mm_x, mm_y)) > _TENT_CB_GROW_WALL_MM
        inside_line = grown & wall_ok
        lab_, n_ = ndimage.label(inside_line)
        if n_:
            keep = lab_[x_mid][lab_[x_mid] > 0]
            if len(keep):
                inside_line = lab_ == np.bincount(keep).argmax()
        line_plane = ndimage.binary_fill_holes(inside_line)
        line_plane = ndimage.binary_opening(line_plane, iterations=max(int(round(2.0 / mm_x)), 1))
        if not _TENT_LINE_BACK_ONLY:
            intra_plane = line_plane

    intra_soft = ndimage.gaussian_filter(intra_plane.astype(np.float32),
                                         sigma=max(3.0 / mm_x, 1.0), mode="nearest")

    brainstem = canon == _BRAINSTEM
    notch_bs = None
    if fused and brainstem.any() and _TENT_NOTCH_BS_GAP_MM is not None:
        # incisura from the midbrain itself: its cross-section at the tent's height (taken before
        # the tent is lifted over it below), grown by the gap the free edge keeps from it;
        # half-width per A-P row from the midline
        zz = np.arange(canon.shape[2], dtype=float)
        band = np.abs(zz[None, None, :] - mid[:, :, None]) <= _TENT_NOTCH_BAND_MM / mm_z
        section = (brainstem & band).any(axis=2)
        if section.any():
            grown = fast_kernels.distance_transform_edt(~section, sampling=(mm_x, mm_y)) \
                <= _TENT_NOTCH_BS_GAP_MM
            off = np.abs(np.arange(grown.shape[0], dtype=float) - x_plane)[:, None]
            notch_bs = np.where(grown, off, 0.0).max(axis=0)
    if fused and brainstem.any():
        bs_top_pre, has_bs_pre = _column_top(brainstem)
        need = np.where(has_bs_pre, bs_top_pre + _INCISURA_CLEARANCE_MM / mm_z,
                        mid - 10.0)
        cap = np.maximum(ceiling - clearance, mid)
        mid = _soft_min(_soft_max(mid, need, 1.0), cap, 1.0)
        mid = ndimage.gaussian_filter(mid, sigma=(max(2.0 / mm_x, 1.0),
                                                  max(2.0 / mm_y, 1.0)), mode="nearest")
        mid = np.clip(mid, floor + clearance,
                      np.maximum(ceiling - clearance, floor + clearance))
        bs_top, has_bs = _column_top(brainstem)
        straddles = has_bs & (bs_top + _INCISURA_CLEARANCE_MM / mm_z
                              > np.maximum(ceiling - clearance, floor + clearance))
        pad = max(1, int(round(_INCISURA_CLEARANCE_MM / mm_x)))
        notch_core = ndimage.binary_dilation(straddles, iterations=pad)
        notch_half = np.zeros(notch_core.shape[1])
        cols = np.where(notch_core.any(axis=0))[0]
        for y in cols:
            xr = np.where(notch_core[:, y])[0]
            notch_half[y] = max(abs(xr.max() - x_mid), abs(x_mid - xr.min()))
        notch_required = notch_half.copy()
        notch_half = np.maximum.accumulate(notch_half)
        notch_half = ndimage.gaussian_filter1d(notch_half, max(6.0 / mm_y, 1.0),
                                               mode="nearest")
        notch_half[:cols.min()] = 0.0
        ramp = np.clip((np.arange(len(notch_half), dtype=float) - cols.min())
                       * mm_y / _TENT_NOTCH_OPEN_MM, 0.0, 1.0)
        notch_half *= ramp * ramp * (3.0 - 2.0 * ramp)
        notch_half = np.maximum(notch_half, notch_required)
        free_edge = np.clip((np.arange(len(notch_half), dtype=float)
                             - (cols.min() - _TENT_FREE_EDGE_RAMP_MM / mm_y))
                            * mm_y / _TENT_FREE_EDGE_RAMP_MM, 0.0, 1.0)
        free_edge = free_edge * free_edge * (3.0 - 2.0 * free_edge)

        mid_cb = np.where(has_cb[x_mid])[0]
        apex_y = float(cols.min())
        start_y = (float(mid_cb[np.argmax(floor_raw[x_mid][mid_cb])])
                   if len(mid_cb) else apex_y - 1.0)
        span = max(apex_y - start_y, 1.0)
        dive = np.clip((np.arange(mid.shape[1], dtype=float) - start_y) / span, 0.0, 1.0)
        dive = (dive * dive * (3.0 - 2.0 * dive))[None, :]
        mid_pre_dive = mid[x_mid].copy()
        border = np.where(has_bs_pre, bs_top_pre + _INCISURA_CLEARANCE_MM / mm_z, mid)
        dived = mid + (1.0 - w) * dive * (np.minimum(border, mid) - mid)
        start_col = int(np.clip(round(start_y), 0, mid.shape[1] - 1))
        floor_saddle = float(mid[x_mid, start_col]) - _TENT_SADDLE_DEPTH_MM / mm_z
        mid = np.minimum(mid, np.maximum(dived, floor_saddle))
        mid = np.clip(mid, floor + clearance,
                      np.maximum(ceiling - clearance, floor + clearance))
        need_after = need
        descend_from = int(round(start_y))
        if ridge_hold is not None:   # the hold stops at the vermis top: in front the tent dips to the apex
            ap_w = 1.0 - _smoothstep((np.arange(mid.shape[1], dtype=float) - start_y) * mm_y / 6.0)
            ridge_hold[1] = ridge_hold[1] * ap_w[None, :]
    else:
        straddles = np.zeros_like(has_cb)
        notch_core = np.zeros_like(has_cb)
        notch_half = np.zeros(has_cb.shape[1])
        notch_required = notch_half.copy()
        free_edge = np.zeros(has_cb.shape[1])
        need_after = None
        descend_from = None
        mid_pre_dive = mid[x_mid].copy()

    ap_sigma = max(_TENT_AP_SMOOTH_MM / mm_y, 1.0)
    mid = ndimage.gaussian_filter1d(mid, ap_sigma, axis=1, mode="nearest")
    apex_col = (descend_from if descend_from is not None
                else int(np.argmax(mid[x_mid])))
    apex_col = int(np.clip(apex_col, 0, mid.shape[1] - 1))
    if apex_col > 0:
        back = np.arange(apex_col + 1)[::-1]
        mid[:, back] = np.minimum.accumulate(mid[:, back], axis=1)
    if need_after is not None:
        mid = _soft_max(mid, need_after, 1.0)
    mid = np.clip(mid, floor + clearance,
                  np.maximum(ceiling - clearance, floor + clearance))

    cb_y = np.where(has_cb.any(axis=0))[0]
    intra_y = np.where(intra_plane.any(axis=0))[0]
    y0 = int(cb_y.min())
    y1 = int(cb_y.max())
    if notch_core.any():
        y1 = max(y1, int(np.where(notch_core.any(axis=0))[0].max()))
    y1 = min(y1, int(np.where(intra_plane.any(axis=0))[0].max()))
    if _TENT_TIP_FRONT_MM is not None and fused:
        # the lab templates' leaf tips reach the front of the midbrain at the sheet's height
        bs_c = canon == _BRAINSTEM
        if bs_c.any():
            zz_ = np.arange(canon.shape[2])[None, None, :]
            sec = (bs_c & (np.abs(zz_ - mid[:, :, None]) <= 3.0 / mm_z)).any(axis=2)
            if sec.any():
                y_f = int(np.where(sec.any(axis=0))[0].max()) + int(round(_TENT_TIP_FRONT_MM / mm_y))
                y1 = min(max(y_f, y0 + 4), int(np.where(intracranial.any(axis=(0, 2)))[0].max()))
    if y1 - y0 < 4:
        return None

    n_y = has_cb.shape[1]
    cb_lo = np.full(n_y, np.nan)
    cb_hi = np.full(n_y, np.nan)
    intra_lo = np.full(n_y, np.nan)
    intra_hi = np.full(n_y, np.nan)
    margin = 2.0 / mm_x
    for y in range(min(int(intra_y.min()), y0), y1 + 1):
        icol = np.where(intra_plane[:, y])[0]
        if len(icol):
            intra_lo[y], intra_hi[y] = float(icol.min()), float(icol.max())
        col = np.where(has_cb[:, y])[0]
        if len(col):
            cb_lo[y] = float(col.min()) - margin
            cb_hi[y] = float(col.max()) + margin
    have_intra = ~np.isnan(intra_lo)
    if have_intra.sum() < 4:
        return None
    intra_lo = _smooth_curve(np.nan_to_num(intra_lo), have_intra, sigma=3.0)
    intra_hi = _smooth_curve(np.nan_to_num(intra_hi), have_intra, sigma=3.0)
    have_cb = ~np.isnan(cb_lo)
    cb_lo = _smooth_curve(np.nan_to_num(cb_lo), have_cb, sigma=3.0)
    cb_hi = _smooth_curve(np.nan_to_num(cb_hi), have_cb, sigma=3.0)

    reach = np.where(have_intra, np.minimum(x_mid - intra_lo, intra_hi - x_mid), np.inf)
    measured = np.isfinite(reach)
    widest = float(np.max(reach[y0:y1 + 1][measured[y0:y1 + 1]])) \
        if measured[y0:y1 + 1].any() else 0.0
    floor_half = 4.0 / mm_x
    y_back = y0
    while (y_back > int(intra_y.min()) and measured[y_back - 1]
           and reach[y_back - 1] >= max(_TENT_POSTERIOR_MIN_FRAC * widest, floor_half)
           and reach[y_back - 1] <= reach[y_back]):
        y_back -= 1
    y0 = y_back
    from_line = _TENT_OUTLINE_FROM_LINE and (_TENT_RIM_LINE_W is not None or _TENT_CB_GROW_MM is not None)
    if from_line:   # the back of the outline is where the line closes behind the cerebellum
        y0 = int(np.where(line_plane.any(axis=0))[0].min())

    ap_n = np.clip((np.arange(n_y, dtype=float) - y0) / max(y1 - y0, 1), 0.0, 1.0)
    profile = np.interp(ap_n, np.linspace(0.0, 1.0, len(_TENT_HALF_WIDTH_PROFILE)),
                        _TENT_HALF_WIDTH_PROFILE)

    ap_span = np.zeros(n_y, dtype=bool)
    ap_span[y0:y1 + 1] = True
    ref_half = np.minimum(profile * widest, reach)

    cb_left = np.clip(np.where(have_cb, np.minimum(x_mid - ref_half, cb_lo),
                               x_mid - ref_half), intra_lo, float(x_mid))
    cb_right = np.clip(np.where(have_cb, np.maximum(x_mid + ref_half, cb_hi),
                                x_mid + ref_half), float(x_mid), intra_hi)
    rim_sigma = max(_TENT_RIM_SMOOTH_MM / mm_y, 1.0)
    cb_left = ndimage.gaussian_filter1d(cb_left, rim_sigma, mode="nearest")
    cb_right = ndimage.gaussian_filter1d(cb_right, rim_sigma, mode="nearest")
    cb_left = np.clip(cb_left, intra_lo, float(x_mid))
    cb_right = np.clip(cb_right, float(x_mid), intra_hi)
    if from_line:
        # each side's attached margin is the anatomical line itself (lab templates: median
        # 0.6-0.7 mm from it), not the reference half-width profile, and no corner rounding.
        # _TENT_LINE_BACK_ONLY: only behind the widest point; the front corners keep the
        # profile/straight-front construction (the line there turns along the petrous apex)
        if _TENT_LINE_BACK_ONLY:
            l_lo, l_hi = np.full(n_y, np.nan), np.full(n_y, np.nan)
            for y in range(n_y):
                c = np.where(line_plane[:, y])[0]
                if len(c):
                    l_lo[y], l_hi[y] = float(c.min()), float(c.max())
            have_l = ~np.isnan(l_lo)
            l_lo = _smooth_curve(np.nan_to_num(l_lo), have_l, sigma=3.0)
            l_hi = _smooth_curve(np.nan_to_num(l_hi), have_l, sigma=3.0)
            line_l = np.where(have_l, l_lo, float(x_mid))
            line_r = np.where(have_l, l_hi, float(x_mid))
        else:
            # rows in front of the line's end (the leaf tips) keep the profile construction
            # with _TENT_LINE_TIPS_PROFILE, instead of collapsing to the incisura
            fill_l = cb_left if _TENT_LINE_TIPS_PROFILE else float(x_mid)
            fill_r = cb_right if _TENT_LINE_TIPS_PROFILE else float(x_mid)
            line_l = np.where(have_intra, intra_lo, fill_l)
            line_r = np.where(have_intra, intra_hi, fill_r)
        if _TENT_LINE_BACK_ONLY:
            yw = int(np.argmax(np.where(have_l, line_r - line_l, -1.0)))
            t_b = 1.0 - _smoothstep((np.arange(n_y, dtype=float) - yw) * mm_y / 10.0 + 0.5)
            cb_left = t_b * line_l + (1.0 - t_b) * cb_left
            cb_right = t_b * line_r + (1.0 - t_b) * cb_right
        else:
            cb_left, cb_right = line_l, line_r

    cap_r = max(_TENT_POSTERIOR_ROUND_MM / mm_y, 1.0)
    t_cap = np.clip((np.arange(n_y, dtype=float) - y0) / cap_r, 0.0, 1.0)
    posterior_cap = np.maximum(1.0 - (1.0 - t_cap) ** 2, 0.0) ** 0.75
    cap_f = max(_TENT_ANTERIOR_ROUND_MM / mm_y, 1.0)
    t_ant = np.clip((y1 - np.arange(n_y, dtype=float)) / cap_f, 0.0, 1.0)
    anterior_cap = np.maximum(1.0 - (1.0 - t_ant) ** 2, 0.0) ** 0.75
    if from_line:
        posterior_cap = np.ones(n_y)
        if not _TENT_LINE_BACK_ONLY:
            anterior_cap = np.ones(n_y)

    width_mm = float(np.max((cb_right - cb_left)[ap_span]) * mm_x)

    if os.environ.get("ANATOMESH_TENT_DEBUG"):
        np.savez(os.environ["ANATOMESH_TENT_DEBUG"], x_mid=x_mid, mm=frame.spacing,
                 floor=floor[x_mid], ceiling=ceiling[x_mid], taut=taut[x_mid],
                 mid=mid[x_mid], notch_half=notch_half, free_edge=free_edge,
                 has_cb=has_cb[x_mid], y0=y0, y1=y1,
                 need=(need_after[x_mid] if need_after is not None
                       else np.zeros_like(mid[x_mid])),
                 mid_pre_dive=mid_pre_dive, cb_left=cb_left, cb_right=cb_right,
                 ref_half=ref_half, ap_n=ap_n, cb_lo=cb_lo, cb_hi=cb_hi,
                 intra_lo=intra_lo, intra_hi=intra_hi, posterior_cap=posterior_cap, anterior_cap=anterior_cap)

    y_index = np.arange(len(cb_left), dtype=float)
    ridge_curve = mid[x_mid]
    u, _arc = _arc_samples(y0, y1, y_index, ridge_curve, mm_y, mm_z, target_mm, 8, 256)
    nu = len(u)

    widest_y = int(np.argmax(np.where(ap_span, np.abs(cb_right - cb_left), -1.0)))
    half_widest = 0.5 * float(abs(cb_right[widest_y] - cb_left[widest_y]))
    fine = np.linspace(0.0, 1.0, 257)
    prof_x = fine * max(half_widest * mm_x, target_mm)
    prof_z = ndimage.map_coordinates(
        mid, np.stack([x_mid + fine * half_widest, np.full(len(fine), float(widest_y))]),
        order=1, mode="nearest") * mm_z
    prof_d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(prof_x), np.diff(prof_z)))])
    nv = _sample_counts(float(prof_d[-1]), target_mm, 4, 128)
    t = np.interp(np.linspace(0.0, prof_d[-1], nv), prof_d, fine)

    tectum_y = None
    if _TENT_APEX_TECTUM_MM is not None and fused and brainstem.any():
        # top of the midbrain on the midline (the colliculi): A-P of its highest voxels
        r = max(int(round(3.0 / mm_x)), 1)
        top = np.argwhere(brainstem[max(x_mid - r, 0):x_mid + r + 1])
        if len(top):
            tectum_y = float(top[top[:, 2] >= top[:, 2].max() - max(1.0 / mm_z, 1.0), 1].mean())

    TENTORIUM_PAVED.clear()
    if (not fused) or (width_mm >= _TENT_PAVE_MIN_WIDTH_MM and straddles.any()):
        try:
            paved = _paved_tentorium(
                frame, x_mid, int(y0), int(y1), y_index, cb_left, cb_right, posterior_cap,
                anterior_cap, floor_half, notch_required, half_widest, mid, ceiling,
                clearance, free_edge, target_mm, floor,
                open_frac=0.0 if fused else float(open_frac), ridge_hold=ridge_hold,
                notch_bs=notch_bs, tectum_y=tectum_y, has_cb=has_cb,
                share_target=share_target, share_weight=share_weight)
        except Exception as exc:  # noqa: BLE001 - fall back to the two structured leaves
            logger.warning("paved tentorium failed (%s); using structured leaves", exc)
            paved = None
        if paved is not None:
            verts, quads, ridge = paved
            TENTORIUM_PAVED.add("tentorium")
            quality = _report_quality("tentorium", verts, quads)
            logger.info("tentorium: %d planar-paved quads lifted onto the tentorial surface, "
                        "width %.1fmm (published %.1f+/-4.8), area %.1fcm^2 (reference %s %.1f), "
                        "incisura U from the straight-sinus apex", len(quads), width_mm,
                        TENTORIUM_WIDTH_MM, _sheet_area_cm2(verts, quads),
                        REFERENCE_TENTORIUM.name, TENTORIUM_REF_AREA_CM2)
            return verts, quads, ridge

    all_verts: list[np.ndarray] = []
    all_quads: list[np.ndarray] = []
    offset = 0
    for edge, sign in ((cb_left, -1.0), (cb_right, 1.0)):
        half_u = _limit_growth(np.abs(np.interp(u, y_index, edge) - x_mid),
                               floor=2.0 / mm_x)
        half_u = np.maximum(
            half_u * (1.0 - _TENT_POSTERIOR_CORNER
                      * (1.0 - np.interp(u, y_index, posterior_cap)))
                   * (1.0 - _TENT_ANTERIOR_CORNER
                      * (1.0 - np.interp(u, y_index, anterior_cap))),
            floor_half)
        min_leaf = 2.0 / mm_x
        inner_u = np.minimum(np.interp(u, np.arange(len(notch_half)), notch_half),
                             np.maximum(half_u - min_leaf, 0.0))
        if not fused:   # open at the midline: the leaf covers the lateral part only
            inner_u = np.maximum(inner_u, np.minimum(float(open_frac) * half_u,
                                                     np.maximum(half_u - min_leaf, 0.0)))
        leaf_u = half_u - inner_u
        xx = x_mid + sign * (inner_u[:, None] + t[None, :] * leaf_u[:, None])
        yy = np.repeat(u[:, None], nv, axis=1)
        planform = shell_repair.winslow_smooth(
            np.stack([xx * mm_x, yy * mm_y], axis=-1))
        xx, yy = planform[..., 0] / mm_x, planform[..., 1] / mm_y
        zz = ndimage.map_coordinates(mid, np.stack([xx.ravel(), yy.ravel()]),
                                     order=1, mode="nearest").reshape(xx.shape)
        lift = (_TENT_FREE_EDGE_LIFT_MM / mm_z
                * np.exp(-((t[None, :] * leaf_u[:, None] * mm_x)
                           / _TENT_FREE_EDGE_DECAY_MM) ** 2)
                * np.interp(u, np.arange(len(free_edge), dtype=float),
                            free_edge)[:, None])
        head_room = ndimage.map_coordinates(
            np.maximum(ceiling - clearance, mid), np.stack([xx.ravel(), yy.ravel()]),
            order=1, mode="nearest").reshape(xx.shape)
        zz = _soft_min(zz + lift, head_room, 1.0)

        pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        keep = np.ones((nu - 1, nv - 1), dtype=bool)
        wide = leaf_u >= max(2.0 / mm_x, 0.15 * float(leaf_u.max()))
        keep &= (wide[:-1] & wide[1:])[:, None]
        keep = largest_cell_island(keep)
        quads = _grid_quads(nu, nv, keep=keep)
        if not len(quads):
            continue
        v, q = _prune_unused(pts, quads)
        all_verts.append(v)
        all_quads.append(q + offset)
        offset += len(v)

    if not all_quads:
        return None
    verts = frame.to_world(np.concatenate(all_verts, axis=0))
    quads = np.concatenate(all_quads, axis=0)
    verts, quads = weld_vertices(verts, quads, tol_mm=1e-3 * max(target_mm, 1.0))
    verts, quads = _drop_slivers(verts, quads, 0.05 * target_mm)
    verts = relax_sheet(verts, quads)
    verts, _ = untangle_sheet(verts, quads)

    ridge_y = u
    ridge_z = ndimage.map_coordinates(
        mid, np.stack([np.full(nu, float(x_mid)), u]), order=1, mode="nearest")
    ridge_ceiling = ndimage.map_coordinates(
        np.maximum(ceiling - clearance, mid),
        np.stack([np.full(nu, float(x_mid)), u]), order=1, mode="nearest")
    ridge_z = np.minimum(
        ridge_z + _TENT_FREE_EDGE_LIFT_MM / mm_z
        * np.interp(u, np.arange(len(free_edge), dtype=float), free_edge),
        ridge_ceiling)
    idx = frame.to_index(verts)
    near = np.abs(idx[:, 0] - float(x_mid)) <= 1.5
    if near.any():
        order = np.argsort(idx[near, 1])
        ys_n, zs_n = idx[near, 1][order], idx[near, 2][order]
        measured = np.array([
            zs_n[np.abs(ys_n - uu) <= 1.5].max() if np.any(np.abs(ys_n - uu) <= 1.5)
            else -np.inf for uu in u])
        ridge_z = np.maximum(ridge_z, np.where(np.isfinite(measured), measured, ridge_z))

    quality = _report_quality("tentorium", verts, quads)
    logger.info("tentorium: %d quads (reference deck %d), width %.1fmm (published "
                "%.1f+/-4.8), area %.1fcm^2 (reference deck %.1f, published %.1f+/-5.8), "
                "growth ratio %.2f, incisura %s",
                len(quads), TENTORIUM_REF_QUADS, width_mm, TENTORIUM_WIDTH_MM,
                _sheet_area_cm2(verts, quads), TENTORIUM_REF_AREA_CM2, TENTORIUM_AREA_CM2,
                quality["max_growth_ratio"],
                "cut from the midbrain at the tentorial plane" if straddles.any()
                else "absent (no brainstem label)")
    return verts, quads, (ridge_y, ridge_z)


# Membrane profiles by species . "fused": one tentorial
# crescent joined by the falx over a straight sinus (human, rhesus monkey; Klintworth 1968);
# "carnivore": the same plus a bony posteromedial tentorium (mink; ferret tentorium osseum,
# He et al. 2002); "lateral": two lateral tentorial leaves open at the midline, no straight
# sinus, falx without a stem (pig, rat, mouse; Klintworth 1968).
MEMBRANE_PROFILES = ("fused", "carnivore", "lateral")
SPECIES_MEMBRANE_PROFILE = {
    "human": "fused", "primate": "fused", "macaque": "fused", "baboon": "fused", "papio": "fused",
    "ferret": "carnivore",
    "swine": "lateral", "pig": "lateral", "rat": "lateral", "rodent": "lateral", "mouse": "lateral",
    "mouse_dsurqe": "lateral", "mouse_fused": "lateral", "mouse_fused_allen": "lateral",
    "mouse_allen": "lateral",
}
HUMAN_CAVITY_MM = 165.0      # intracranial length front to back, the reference adult head: the builders' native size
LAST_BUILD_SCALE = 1.0       # scale of the last build_membranes call (rim tolerances follow it)
TENTORIUM_OSSEOUS_FRAC = 0.33  # carnivore: share of length and half-width that is bone
LAST_TENT_FRAME: dict = {}     # pitched tentorium build: rotation and centre (build-scale world)
TENT_RIDGE_FRAME: dict = {}    # paved tentorium build: its canonical frame and midline column
_RIDGE_FROM_SHEET = False      # the falx rests on the finished tentorium's midline (after the tissue
                               # ease lifted it), not the surface it was lifted from


def membrane_profile(species: str, requested: str = "auto") -> str:
    """The membrane profile for a species ("auto" looks it up; unknown species are "fused")."""
    if requested and requested != "auto":
        if requested not in MEMBRANE_PROFILES:
            raise ValueError(f"unknown membrane profile {requested!r}; use one of {MEMBRANE_PROFILES}")
        return requested
    key = (species or "human").lower()
    if key.startswith("mouse"):
        return "lateral"
    return SPECIES_MEMBRANE_PROFILE.get(key, "fused")


def cavity_length_mm(labels: np.ndarray, affine: np.ndarray) -> float:
    """Front-to-back length of the intracranial space (brain + CSF), in mm."""
    frame = _Frame(affine, np.asarray(labels).shape)
    canon = frame.view(np.asarray(labels))
    ys = np.where(np.isin(canon, _INTRACRANIAL).any(axis=(0, 2)))[0]
    return float(np.ptp(ys) + 1) * float(frame.spacing[1]) if len(ys) else 0.0


def osseous_tentorium_mask(verts: np.ndarray, quads: np.ndarray, affine: np.ndarray,
                           shape: tuple[int, ...],
                           frac: float = TENTORIUM_OSSEOUS_FRAC) -> np.ndarray:
    """Quads of the posteromedial tentorium, the part carnivores ossify (Klintworth 1968:
    "limited to the posteromedial portion" in dogs and minks): within `frac` of the length
    from the back and within `frac` of the half-width of the midline."""
    if not len(quads):
        return np.zeros(0, dtype=bool)
    frame = _Frame(affine, shape)
    pts = np.asarray(verts, dtype=float)
    if LAST_TENT_FRAME:   # an upright tentorium: judge back and middle in its own (level) frame
        k = LAST_TENT_FRAME.get("scale", 1.0)
        centre = affine[:3, 3] + (np.asarray(LAST_TENT_FRAME["centre"]) - affine[:3, 3]) * k
        pts = centre + (pts - centre) @ LAST_TENT_FRAME["R"].T
    idx = frame.to_index(pts[quads].reshape(-1, 3)).reshape(len(quads), 4, 3)
    c = idx.mean(axis=1) * frame.spacing
    x_mid = 0.5 * (c[:, 0].min() + c[:, 0].max())
    half = max(0.5 * float(np.ptp(c[:, 0])), 1e-9)
    y_back, length = float(c[:, 1].min()), max(float(np.ptp(c[:, 1])), 1e-9)
    return (np.abs(c[:, 0] - x_mid) <= frac * half) & (c[:, 1] - y_back <= frac * length)


def build_membranes(
    labels: np.ndarray, affine: np.ndarray, target_mm: float = 1.0,
    clip_to: Optional[np.ndarray] = None,
    falx_cc_gap_mm: float = _FALX_FREE_MARGIN_GAP_MM,
    profile: str = "fused", scale: Optional[float] = 1.0,
    lateral_open_frac: float = 0.35,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build both membranes for a species profile, at any head size.

    The builders' lengths are tuned on the adult human (the reference adult head). Any other head is built in a
    frame scaled by 1 / `scale`, so every length in the falx and tentorium builders scales with
    the head, and the result is scaled back. `scale=None` measures it (cavity length /
    HUMAN_CAVITY_MM; human-sized cavities, >= 120 mm, stay at 1); the default 1.0 is the human
    build, unchanged."""
    global LAST_BUILD_SCALE
    if profile not in MEMBRANE_PROFILES:
        raise ValueError(f"unknown membrane profile {profile!r}")
    affine = np.asarray(affine, dtype=float)
    if scale is None:
        length = cavity_length_mm(labels, affine)
        scale = 1.0 if length >= 120.0 or length <= 0.0 else length / HUMAN_CAVITY_MM
    scale = float(scale)
    LAST_BUILD_SCALE = scale
    if abs(scale - 1.0) < 1e-9:
        return _build_membranes(labels, affine, target_mm, clip_to, falx_cc_gap_mm, profile,
                                lateral_open_frac)
    logger.info("membranes: %s profile, built at human size (scale %.3f) and scaled back",
                profile, scale)
    origin = affine[:3, 3].copy()
    scaled = affine.copy()
    scaled[:3, :3] /= scale
    out = _build_membranes(labels, scaled, target_mm / scale, clip_to, falx_cc_gap_mm / scale,
                           profile, lateral_open_frac)
    return {name: (origin + (np.asarray(v) - origin) * scale, q) for name, (v, q) in out.items()}


_FINISH_SJ = 0.45   # the quality bar (Li, Zhou, Kleiven 2021)
_FINISH_RIM_SJ = 0.45  # rim quads still under the bar after the in-surface pass are dropped
_FINISH_LC_RATIO = 0.3  # ... and rim slivers whose characteristic length is under this share
                        # of the median edge (they would set the explicit time step)


def _finish_sheet(verts, quads, name):
    """Last quality step for a lifted sheet: slide nodes in its own surface; rim quads that
    stay under _FINISH_RIM_SJ (a fold at a twisted leaf edge) are then dropped, ring by ring,
    never opening a hole. Nothing happens to a sheet already at the bar."""
    from engine.export.validate import _quad_scaled_jacobian

    from engine.mesh import sheet_quality

    def lc_ratio(v, q):
        """LS-DYNA characteristic length (area / longest edge) over the median edge."""
        c = v[q]
        edges = np.linalg.norm(c - np.roll(c, -1, axis=1), axis=2)
        area = 0.5 * np.linalg.norm(np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]), axis=1)
        return area / np.maximum(edges.max(axis=1), 1e-12) / max(float(np.median(edges)), 1e-12)

    report = {"sj_before": round(float(_quad_scaled_jacobian(verts[quads]).min()), 3),
              "lc_ratio_before": round(float(lc_ratio(verts, quads).min()), 3)}
    if report["sj_before"] >= _FINISH_SJ and report["lc_ratio_before"] >= _FINISH_LC_RATIO:
        return verts, quads, report
    verts, _r = sheet_quality.improve_sheet_in_surface(verts, quads, name=name)
    dropped = 0
    for _ in range(3):
        sj = _quad_scaled_jacobian(verts[quads])
        rim = np.zeros(len(verts), dtype=bool)
        rim[np.unique(_rim_edges(quads))] = True
        # under the bar, or a sliver that would set the explicit time step
        drop = (((sj < _FINISH_RIM_SJ) | (lc_ratio(verts, quads) < _FINISH_LC_RATIO))
                & rim[quads].any(axis=1))
        if not drop.any():
            break
        verts, quads = _prune_unused(verts, quads[~drop])
        dropped += int(drop.sum())
        verts, _r = sheet_quality.improve_sheet_in_surface(verts, quads, name=name)
    report.update(dropped_rim_quads=dropped,
                  sj_after=round(float(_quad_scaled_jacobian(verts[quads]).min()), 3),
                  lc_ratio_after=round(float(lc_ratio(verts, quads).min()), 3))
    logger.info("%s: finishing pass, worst SJ %.3f -> %.3f, smallest Lc/edge %.3f -> %.3f, "
                "%d rim quads dropped", name, report["sj_before"], report["sj_after"],
                report["lc_ratio_before"], report["lc_ratio_after"], dropped)
    return verts, quads, report


_TENT_PITCH_MIN_DEG = 30.0  # rotate the tentorium frame when the cerebellum->cerebrum axis tilts more
_TISSUE_EASE = True         # ease the tentorium out of brain tissue (False: leave it as built)
_RIM_RETRACT = True         # ease rims out of bone before trimming (False: trim rim quads only)
_FALX_PLANE_CAVITY = True   # falx outline on the cavity of its own plane (False: the +/-2 mm slab)
_LEGACY_FALX_CLEARANCE_KNOTS = ((-0.1, 4.0), (0.2, 9.5), (0.6, 3.0), (0.9, 3.0), (1.08, 6.0))

# Every method choice made on 2026-10-08, selectable: new (default) or conventional (before).
MEMBRANE_OPTIONS = {
    "tentorium_orientation": ("auto", "level"),      # follow the cerebellum / always level
    "tentorium_surface": ("smooth", "voxel"),        # smooth envelopes, 3 mm / voxel clip, 1 mm
    "tentorium_tissue_ease": (True, False),
    "membrane_rim_mode": ("retract", "trim"),        # ease rims out of bone / delete rim quads
    "falx_cc_profile": ("posterior_close", "legacy"),
    "falx_cavity": ("plane", "slab"),
    # "lab_templates": the anatomical rules fitted to adult male and female template heads;
    # "2026-10-09": the rules before. mesh_agent picks "lab_templates" for human runs and "2026-10-09" for every other
    # species ("auto"), so animal decks do not change; "auto" reaching here (a direct caller of
    # build_head_mesh) keeps this module's values.
    "membrane_rules": ("auto", "lab_templates", "2026-10-09"),
}
_ANATOMICAL = False
RULE_ASEG = None   # (aseg array, affine) for membrane_rules="anatomical"; set by mesh_agent
# what "lab_templates" changes (each constant's 2026-10-09 value is its definition above)
LAB_TEMPLATE_RULES = {
    "_FALX_DROP_DEG": 55.0,            # 35: the margin heads down and further back to the crista
    "_TENT_NOTCH_APEX_FRAC": 0.76,     # 0.72
    "_TENT_CEREBRUM_GAP_MM": 4.0,      # None (= the 1.5 mm clearance): room under the cerebrum
    "_TENTORIUM_CLEARANCE_MM": 1.05,   # 1.5: closer over the cerebellum
    "_TENT_SADDLE_DEPTH_MM": 13.0,     # 8: the midline drops further toward the incisura
    "_TENT_LEAF_TIP_MM": 5.0,          # 8: narrower anterior leaf tips
    "_TENT_WRAP_START_FRAC": 0.45,     # 0.55: the sides start hugging the cerebellum nearer the middle
    "_TENT_INSET_PER_CSF": 1.4,        # None: margins move in with the CSF layer on the wall
    "_FALX_INSET_PER_CSF": 0.8,        # (see wall_csf_thickness_mm)
}
# numeric settings carried the same way (MeshingConfig field -> this module's constant): how far
# inside the cavity the attached margins sit. 0.4 / 0 on a subject's own segmentation (the
# margin on the bone); a CSF label that takes in the dura and sinus band (the lab's reference
# template segmentations) needs ~2 / ~3 mm 
MEMBRANE_NUMERIC = {
    "falx_wall_inset_mm": "_FALX_OUTLINE_INSET_MM",
    "tentorium_wall_inset_mm": "_TENT_RIM_INSET_MM",
}


class membrane_settings:
    """Context manager applying MEMBRANE_OPTIONS choices to this module, restoring them after.
    Unknown keys or values raise, so a typo never silently picks the default."""

    def __init__(self, **choices):
        for key, value in choices.items():
            if key in MEMBRANE_NUMERIC:
                if value is not None and float(value) < 0:
                    raise ValueError(f"membrane option {key}={value!r} must be >= 0")
                continue
            if key not in MEMBRANE_OPTIONS or value not in MEMBRANE_OPTIONS[key]:
                raise ValueError(f"membrane option {key}={value!r}; choose from "
                                 f"{ {k: v for k, v in MEMBRANE_OPTIONS.items()} }")
        self.choices = choices
        self.saved: dict = {}

    def __enter__(self):
        g = globals()
        names = ("_TENT_PITCH_MIN_DEG", "_TENT_ENVELOPE_MM", "_TENT_POST_SMOOTH_MM", "_TISSUE_EASE",
                 "_RIM_RETRACT", "_FALX_CLEARANCE_KNOTS", "_FALX_PLANE_CAVITY", "_ANATOMICAL",
                 *MEMBRANE_NUMERIC.values(), *LAB_TEMPLATE_RULES)
        self.saved = {n: g[n] for n in names}
        c = self.choices
        if c.get("membrane_rules") == "lab_templates":
            g.update(LAB_TEMPLATE_RULES)
        if c.get("membrane_rules") == "anatomical":
            g["_ANATOMICAL"] = True
        for key, constant in MEMBRANE_NUMERIC.items():
            if c.get(key) is not None:
                g[constant] = float(c[key])
        if c.get("tentorium_orientation") == "level":
            g["_TENT_PITCH_MIN_DEG"] = 1e9
        if c.get("tentorium_surface") == "voxel":
            g["_TENT_ENVELOPE_MM"], g["_TENT_POST_SMOOTH_MM"] = 0.0, 1.0
        if c.get("tentorium_tissue_ease") is False:
            g["_TISSUE_EASE"] = False
        if c.get("membrane_rim_mode") == "trim":
            g["_RIM_RETRACT"] = False
        if c.get("falx_cc_profile") == "legacy":
            g["_FALX_CLEARANCE_KNOTS"] = _LEGACY_FALX_CLEARANCE_KNOTS
        if c.get("falx_cavity") == "slab":
            g["_FALX_PLANE_CAVITY"] = False
        return self

    def __exit__(self, *exc):
        globals().update(self.saved)
        return False


def _tentorial_axis(labels: np.ndarray, affine: np.ndarray) -> tuple[float, np.ndarray]:
    """Pitch (deg, about left-right) of the direction from the cerebellum to the cerebrum across
    their interface, from vertical; and the brain centre (world). Human ~0-20 deg (the cerebrum
    sits on the tentorium); rodents and carnivores ~60-90 deg (the cerebellum sits behind it,
    and the tentorial folds stand upright in the transverse fissure)."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    cerebellum = labels == _CEREBELLUM
    cerebrum = np.isin(labels, _CEREBRUM)
    if not cerebellum.any() or not cerebrum.any():
        return 0.0, affine[:3, 3]
    reach = 4.0 * float(spacing.max())
    a = cerebrum & (fast_kernels.distance_transform_edt(~cerebellum, sampling=spacing) <= reach)
    b = cerebellum & (fast_kernels.distance_transform_edt(~cerebrum, sampling=spacing) <= reach)
    if not a.any() or not b.any():
        return 0.0, affine[:3, 3]
    to_world = lambda m: np.argwhere(m).mean(axis=0) @ affine[:3, :3].T + affine[:3, 3]  # noqa: E731
    d = to_world(a) - to_world(b)
    centre = to_world(np.isin(labels, _BRAIN))
    return float(np.degrees(np.arctan2(d[1], d[2]))), centre


def _pitched(labels: np.ndarray, affine: np.ndarray, pitch_deg: float, centre: np.ndarray):
    """Labels resampled so the cerebellum->cerebrum axis points up (+z), on a padded grid with
    the SAME affine convention; returns (labels', affine', R), true = c + (pretend - c) @ R."""
    phi = np.radians(pitch_deg)
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    pad = int(0.35 * max(labels.shape))
    padded = np.pad(np.asarray(labels), pad)
    A = affine[:3, :3]
    t = affine[:3, 3] - A @ np.full(3, pad)
    Ainv = np.linalg.inv(A)
    M = Ainv @ R.T @ A
    offset = Ainv @ (R.T @ (t - centre) + centre - t)
    out = ndimage.affine_transform(padded, M, offset=offset, order=0, mode="constant", cval=0)
    aff = affine.copy()
    aff[:3, 3] = t
    return out.astype(np.asarray(labels).dtype), aff, R


# The attached margins sit on the cavity wall. Where the segmentation's CSF label also takes in
# the dura and venous-sinus band (template segmentations: 4 mm of CSF at the
# wall against 1.4-3 mm on segmentations made from the subject's own MRI), the margin belongs
# inside that band. Landmark: the median thickness of the CSF layer at the cavity wall, measured
# from the labels themselves; the margin moves in by k x (thickness - the subarachnoid
# thickness). None: off (the margin on the wall).
_WALL_CSF_REF_MM = 1.5
_TENT_INSET_PER_CSF = None
_FALX_INSET_PER_CSF = None
LAST_WALL_CSF_MM = 0.0


def wall_csf_thickness_mm(labels: np.ndarray, affine: np.ndarray) -> float:
    """Median thickness of the CSF layer on the cavity wall: distance from each cavity-wall
    voxel to the nearest brain voxel."""
    lab = np.asarray(labels)
    intra = np.isin(lab, _INTRACRANIAL)
    brain = np.isin(lab, _BRAIN)
    if not intra.any() or not brain.any():
        return 0.0
    sp = tuple(float(np.linalg.norm(affine[:3, a])) for a in range(3))
    wall = intra & ~ndimage.binary_erosion(intra)
    if not wall.any():
        return 0.0
    # box round the cavity: the distance field is only read on its wall
    idx = np.where(intra)
    sl = tuple(slice(max(int(i.min()) - 2, 0), int(i.max()) + 3) for i in idx)
    d = fast_kernels.distance_transform_edt(~brain[sl], sampling=sp)
    return float(np.median(d[wall[sl]]))


def _build_membranes(labels, affine, target_mm, clip_to, falx_cc_gap_mm, profile,
                     lateral_open_frac):
    """build_membranes with the margin insets read from the wall CSF band when asked."""
    global _TENT_RIM_INSET_MM, _FALX_OUTLINE_INSET_MM, LAST_WALL_CSF_MM
    saved = (_TENT_RIM_INSET_MM, _FALX_OUTLINE_INSET_MM)
    if _TENT_INSET_PER_CSF is not None or _FALX_INSET_PER_CSF is not None:
        band = wall_csf_thickness_mm(labels, np.asarray(affine, dtype=float))
        LAST_WALL_CSF_MM = band
        excess = max(band - _WALL_CSF_REF_MM, 0.0)
        if _TENT_INSET_PER_CSF is not None:
            _TENT_RIM_INSET_MM = max(_TENT_RIM_INSET_MM, _TENT_INSET_PER_CSF * excess)
        if _FALX_INSET_PER_CSF is not None:
            _FALX_OUTLINE_INSET_MM = max(_FALX_OUTLINE_INSET_MM, _FALX_INSET_PER_CSF * excess)
        logger.info("membranes: CSF layer on the cavity wall %.2f mm -> tentorium rim inset "
                    "%.2f mm, falx margin inset %.2f mm", band, _TENT_RIM_INSET_MM,
                    _FALX_OUTLINE_INSET_MM)
    try:
        return _build_membranes_core(labels, affine, target_mm, clip_to, falx_cc_gap_mm, profile,
                                     lateral_open_frac)
    finally:
        _TENT_RIM_INSET_MM, _FALX_OUTLINE_INSET_MM = saved


def _build_membranes_core(labels, affine, target_mm, clip_to, falx_cc_gap_mm, profile,
                          lateral_open_frac):
    """build_membranes at the builders' native (human) size."""
    if _ANATOMICAL:
        if RULE_ASEG is None:
            logger.warning("membrane_rules='anatomical' needs the run's aseg; using the 2026-10-09 builders")
        else:
            try:
                return _build_anatomical(labels, affine, target_mm)
            except Exception as exc:  # noqa: BLE001
                logger.warning("anatomical membranes failed (%s); using the 2026-10-09 builders", exc)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ridge = None
    fused = profile != "lateral"
    pitch, centre = _tentorial_axis(np.asarray(labels), np.asarray(affine, dtype=float))
    TENTORIUM_TISSUE_REPORT["pitch_deg"] = round(pitch, 1)
    LAST_TENT_FRAME.clear()
    TENT_RIDGE_FRAME.clear()
    pitched = abs(pitch) > _TENT_PITCH_MIN_DEG
    if pitched:
        # the tentorium is built as a height field over the plane it spans; where the
        # cerebellum sits behind the cerebrum, that plane is upright: build it in a frame
        # turned so it lies level, then turn the sheet back
        logger.info("tentorium: cerebellum->cerebrum axis %.0f deg from vertical; built in a "
                    "frame pitched to it", pitch)
        lab_p, aff_p, R = _pitched(labels, affine, pitch, centre)
        tent = tentorium_sheet(lab_p, aff_p, target_mm, fused=fused, open_frac=lateral_open_frac)
        if tent is not None:
            v_p, q_p, _ridge_p = tent
            tent = (centre + (np.asarray(v_p) - centre) @ R, q_p, None)
            LAST_TENT_FRAME.update(R=R, centre=centre, scale=LAST_BUILD_SCALE)
    else:
        tent = tentorium_sheet(labels, affine, target_mm, fused=fused, open_frac=lateral_open_frac)
    if tent is not None:
        verts, quads, ridge = tent
        # the smoothed tent may graze the cerebellum or the temporal lobes beside the
        # incisura: ease those nodes out with smooth dents (the slot carve would otherwise
        # turn the grazed cortex into CSF)
        if _TISSUE_EASE:
            verts, TENTORIUM_TISSUE_REPORT["tentorium"] = retract_sheet_from(
                verts, quads, np.isin(np.asarray(labels), _BRAIN), affine, name="tentorium")
        verts, quads, TENTORIUM_TISSUE_REPORT["finish"] = _finish_sheet(verts, quads, "tentorium")
        out["tentorium"] = (verts, quads)
        if _RIDGE_FROM_SHEET and ridge is not None and not pitched and TENT_RIDGE_FRAME:
            idx = TENT_RIDGE_FRAME["frame"].to_index(np.asarray(verts))
            near = idx[np.abs(idx[:, 0] - TENT_RIDGE_FRAME["x_mid"]) <= 1.0]
            ry, rz = (np.asarray(a, dtype=float) for a in ridge)
            top = np.array([near[np.abs(near[:, 1] - y) <= 1.0, 2].max()
                            if np.any(np.abs(near[:, 1] - y) <= 1.0) else np.nan for y in ry])
            ridge = (ry, np.where(np.isfinite(top), top, rz))
    falx = None
    try:
        falx = falx_on_midline(labels, affine, target_mm,
                               tentorium_ridge=ridge if fused else None,
                               cc_gap_mm=falx_cc_gap_mm, open_back=not fused,
                               seat_from_tissue=fused and ridge is None)
    except Exception as exc:  # noqa: BLE001 - fall back to the column builder
        logger.warning("falx_on_midline failed (%s); using the column builder", exc)
    finished = {"falx"} if falx is not None else set()  # meshed in its plane, needs no repair
    if "tentorium" in out:
        finished |= TENTORIUM_PAVED
    FINISHED_SHEETS.clear()
    FINISHED_SHEETS.update(finished)
    if falx is None:
        falx = falx_sheet(labels, affine, target_mm, tentorium_ridge=ridge,
                          cc_gap_mm=falx_cc_gap_mm)
    planes = {}
    if falx is not None:
        planes["falx"] = sheet_plane(falx[0])
        if "falx" in finished:   # planar paved sheet: same finishing pass, kept exactly flat
            fv, fq, _rep = _finish_sheet(falx[0], falx[1], "falx")
            falx = (flatten_to_plane(fv, planes["falx"]), fq)
        out["falx"] = falx
    if clip_to is not None:
        for name, (verts, quads) in list(out.items()):
            if name in finished:
                continue
            verts, quads = push_sheet_out_of(verts, quads, clip_to, affine)
            verts, _clearance = center_in_gap(verts, quads, clip_to, affine)
            if name in planes:
                verts = flatten_to_plane(verts, planes[name])
            out[name] = (verts, quads)

    for name, (verts, quads) in list(out.items()):
        if name in finished:
            continue
        verts, report = shell_repair.improve_sheet(verts, quads, name=name,
                                                   allowed=clip_to, affine=affine)
        if name in planes:
            verts = flatten_to_plane(verts, planes[name])
        out[name] = (verts, quads)
        SHEET_REPAIR_REPORTS[name] = report
    for name, (verts, quads) in list(out.items()):
        out[name] = _keep_main_piece(np.asarray(verts), np.asarray(quads), name)
    return out


def _build_anatomical(labels, affine, target_mm):
    """Falx and tentorium from mesh/membrane_rules (both paved to final quality in their own
    frames: finished sheets, no repair passes; the pipeline still eases the tentorium off the
    bone and trims rims in bone)."""
    from engine.mesh import membrane_rules
    sheets = membrane_rules.build(labels, affine, RULE_ASEG[0], RULE_ASEG[1], target_mm,
                                  intracranial=_INTRACRANIAL)
    out = {}
    for name, (verts, quads) in sheets.items():
        raw = (np.asarray(verts), np.asarray(quads))
        if name == "tentorium" and _TISSUE_EASE:
            verts, TENTORIUM_TISSUE_REPORT["tentorium"] = retract_sheet_from(
                verts, quads, np.isin(np.asarray(labels), _BRAIN), affine, name="tentorium")
        verts, quads, rep = _finish_sheet(np.asarray(verts), np.asarray(quads), name)
        if rep.get("sj_after", rep["sj_before"]) < _FINISH_SJ and name == "tentorium" and _TISSUE_EASE:
            # the tissue ease folded a quad the finishing pass cannot open: keep the rule
            # sheet as built (it is in CSF by construction) and finish that instead
            v2, q2, rep2 = _finish_sheet(*raw, name)
            if rep2.get("sj_after", rep2["sj_before"]) > rep.get("sj_after", rep["sj_before"]):
                logger.info("tentorium: ease left SJ %.3f; using the sheet without it (SJ %.3f)",
                            rep.get("sj_after", rep["sj_before"]), rep2.get("sj_after", rep2["sj_before"]))
                verts, quads, rep = v2, q2, rep2
        if name == "falx":
            verts = flatten_to_plane(verts, sheet_plane(verts))
        out[name] = _keep_main_piece(np.asarray(verts), np.asarray(quads), name)
    FINISHED_SHEETS.clear()
    FINISHED_SHEETS.update(out)
    TENTORIUM_TISSUE_REPORT["anatomical"] = dict(membrane_rules.REPORT)
    return out


@shell_repair.timed
def free_standing_sheet(
    nodes: np.ndarray, quads: np.ndarray, away_from: np.ndarray, allowed: np.ndarray,
    affine: np.ndarray, offset_mm: float, name: str = "sheet",
    relax_iterations: int = 40, strict_forbidden: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lift an interface-quad sheet off the volume mesh and into the fluid."""
    if not len(quads):
        return np.zeros((0, 3)), quads, {}
    verts, quads = _prune_unused(np.asarray(nodes, dtype=float), np.asarray(quads))

    inv = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    report: dict = {"n_quads": int(len(quads)), "offset_mm": round(float(offset_mm), 3)}

    if away_from.any() and offset_mm > 0:
        _dist, nearest = ndimage.distance_transform_edt(
            ~away_from, sampling=spacing, return_indices=True)
        vox = verts @ inv[:3, :3].T + inv[:3, 3]
        here = np.clip(np.rint(vox).astype(np.int64), 0, np.array(away_from.shape) - 1)
        src_idx = np.stack([axis[here[:, 0], here[:, 1], here[:, 2]] for axis in nearest],
                           axis=1).astype(float)
        src = src_idx @ affine[:3, :3].T + affine[:3, 3]
        direction = verts - src
        length = np.linalg.norm(direction, axis=1)
        live = length > 1e-9
        step = np.zeros_like(verts)
        step[live] = direction[live] / length[live][:, None] * offset_mm
        candidate = verts + step
        ok = _sample_mask(allowed, candidate @ inv[:3, :3].T + inv[:3, 3])
        verts[ok] = candidate[ok]
        report["moved"] = int(ok.sum())
        report["held"] = int(len(verts) - ok.sum())

    verts, relax = relax_sheet_guarded(verts, quads, iterations=relax_iterations,
                                       max_displacement_mm=max(offset_mm, 0.3))
    verts, untangle = untangle_sheet(verts, quads)
    report["relax"] = relax
    report["untangle"] = untangle

    verts, repair = shell_repair.improve_sheet(verts, quads, name=name,
                                               allowed=allowed, affine=affine)
    report["repair"] = repair
    SHEET_REPAIR_REPORTS[name] = repair

    verts, quads, drop_report = drop_inverted_without_holing(verts, quads, name=name)
    report["inverted_dropped"] = drop_report
    report["n_components"] = sheet_components(verts, quads)
    report["area_cm2"] = round(_sheet_area_cm2(verts, quads), 1)
    vox_out = verts @ inv[:3, :3].T + inv[:3, 3]
    outside = int((~_sample_mask(allowed, vox_out)).sum())
    report["vertices_outside_the_slot"] = outside
    if strict_forbidden is not None:
        report["vertices_inside_solid_tissue"] = int(_sample_mask(strict_forbidden, vox_out).sum())
    logger.info("%s: %d quads lifted off the volume mesh, %.1f cm^2, %d component(s), "
                "%d inverted cell(s), %d of %d vertices outside the meningeal slot",
                name, len(quads), report["area_cm2"], report["n_components"],
                untangle.get("n_negative", 0), outside, len(verts))
    return verts, quads, report


@shell_repair.timed
def mask_shell_quads(
    mask: np.ndarray, affine: np.ndarray, block: int = 1, grow: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned quad shell of a boolean voxel mask, optionally coarsened."""
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=float)
    block = max(int(block), 1)

    if block > 1:
        pad = [(0, (-s) % block) for s in mask.shape]
        padded = np.pad(mask, pad, mode="constant", constant_values=False)
        shape = tuple(s // block for s in padded.shape)
        votes = padded.reshape(shape[0], block, shape[1], block, shape[2], block)
        counts = votes.sum(axis=(1, 3, 5))
        mask = counts > 0 if grow else counts > (block ** 3) // 2
        mask = ndimage.binary_fill_holes(mask)
        islands, n_islands = ndimage.label(mask)
        if n_islands > 1:
            sizes = ndimage.sum(mask, islands, index=range(1, n_islands + 1))
            keep = 1 + int(np.argmax(sizes))
            logger.info("mask_shell_quads: dropped %d island(s), %d voxels, off the main body",
                        n_islands - 1, int(mask.sum() - sizes[keep - 1]))
            mask = islands == keep
        affine = affine.copy()
        affine[:3, :3] = affine[:3, :3] * block
        affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.full(3, 0.5 * (1.0 - 1.0 / block))

    if not mask.any():
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64), affine

    from engine.mesh.hex_topology import clean_topology

    cleaned, pinch_report = clean_topology(mask.astype(np.int8), priority=[1, 0],
                                           seal_background=True)
    mask = cleaned.astype(bool)
    mask = ndimage.binary_fill_holes(mask)
    islands, n_islands = ndimage.label(mask)
    if n_islands > 1:
        sizes = ndimage.sum(mask, islands, index=range(1, n_islands + 1))
        mask = islands == 1 + int(np.argmax(sizes))
    logger.info("mask_shell_quads: unpinched %s voxel(s), residual pinches %s",
                pinch_report.get("voxels_changed"), pinch_report.get("after"))

    nx, ny, nz = mask.shape
    strides = np.array([(ny + 1) * (nz + 1), nz + 1, 1], dtype=np.int64)

    quads: list[np.ndarray] = []
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    for axis in range(3):
        lo = padded[1:-1, 1:-1, 1:-1]
        hi = np.roll(padded, -1, axis=axis)[1:-1, 1:-1, 1:-1]
        for inside, sign in ((lo & ~hi, 1), (hi & ~lo, -1)):
            idx = np.argwhere(inside)
            if not len(idx):
                continue
            corner = idx.copy()
            corner[:, axis] += 1
            u, v = [a for a in range(3) if a != axis]
            offsets = [(0, 0), (1, 0), (1, 1), (0, 1)][::sign]
            ids = []
            for du, dv in offsets:
                c = corner.copy()
                c[:, u] += du
                c[:, v] += dv
                ids.append(c @ strides)
            quads.append(np.stack(ids, axis=1))
    if not quads:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64), affine
    quads_arr = np.concatenate(quads, axis=0)

    a, b, c = np.unravel_index(np.arange((nx + 1) * (ny + 1) * (nz + 1)),
                               (nx + 1, ny + 1, nz + 1))
    lattice = np.stack([a, b, c], axis=1).astype(float) - 0.5
    verts = lattice @ affine[:3, :3].T + affine[:3, 3]
    verts, quads_arr = _prune_unused(verts, quads_arr)
    logger.info("mask_shell_quads: %d quads at block %d (%d voxels in the mask)",
                len(quads_arr), block, int(mask.sum()))
    return verts, quads_arr, affine


def demo() -> None:
    """Self-check on a synthetic head."""
    shape = (120, 140, 120)
    labels = np.zeros(shape, dtype=np.int16)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")

    def ball(cx, cy, cz, r):
        return (zz - cx) ** 2 + (yy - cy) ** 2 + (xx - cz) ** 2 < r ** 2

    labels[ball(60, 80, 70, 42)] = _GRAY
    labels[ball(60, 80, 70, 30)] = _WHITE
    labels[ball(60, 45, 40, 22)] = _CEREBELLUM
    labels[ball(60, 62, 60, 9)] = _BRAINSTEM
    affine = np.eye(4)

    sheets = build_membranes(labels, affine, target_mm=3.0)
    assert set(sheets) == {"falx", "tentorium"}, sheets

    for name, (verts, quads) in sheets.items():
        assert 60 < len(quads) < 20000, (name, len(quads))
        assert quads.max() < len(verts)
        assert all(len(set(q)) == 4 for q in quads), name
        assert _sheet_area_cm2(verts, quads) > 1.0, name
        keys = {tuple(sorted(q)) for q in quads}
        assert len(keys) == len(quads), f"{name} is not manifold"
        assert sheet_components(verts, quads) == 1, \
            f"{name} came out in {sheet_components(verts, quads)} pieces"
        q = sheet_quality(verts, quads)
        assert q["max_growth_ratio"] <= 2.4, (name, q)

    toothed = np.zeros((12, 12), dtype=bool)
    toothed[3:9, 3:9] = True
    toothed[1, 10] = True  # an isolated cell, clear of the body
    toothed[5, 5] = False  # a one-cell bite
    cleaned = smooth_cell_mask(toothed)
    assert not cleaned[1, 10], "an isolated cell survived"
    assert cleaned[5, 5], "the bite was not filled"
    assert cleaned[3:9, 3:9].all(), "the body was eroded"
    edge = np.zeros((6, 6), dtype=bool)
    edge[:, 0:3] = True
    assert smooth_cell_mask(edge)[:, 0].all(), "the border column was eroded"

    tv = sheets["tentorium"][0]
    dist = np.linalg.norm(tv[:, [0, 1]] - np.array([60.0, 62.0]), axis=1)
    assert dist.min() > 8.0, f"tentorium enters the brainstem: {dist.min():.1f}mm"

    carved, carve_report = carve_slots(labels, affine, sheets, thickness_mm=2.0)
    assert all(v > 0 for v in carve_report.values()), carve_report
    for name, (verts, quads) in sheets.items():
        inside_tissue = _sample_mask(np.isin(carved, _BRAIN), verts)
        assert not inside_tissue.any(), (
            f"{name}: {int(inside_tissue.sum())} of {len(verts)} vertices are still "
            f"inside solid tissue after carving")

    fv = sheets["falx"][0]
    assert fv[:, 2].min() <= tv[:, 2].max() + 2.0, "falx does not meet the tentorium"

    hv, hexes = extrude_sheet(*sheets["falx"], thickness_mm=1.0)
    assert len(hexes) == len(sheets["falx"][1]) and hv.shape[1] == 3

    print(f"membranes.demo OK: falx {len(sheets['falx'][1])} quads, "
          f"tentorium {len(sheets['tentorium'][1])} quads")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
