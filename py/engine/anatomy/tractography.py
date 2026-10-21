from __future__ import annotations

import gzip
import io
import logging
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

_TT_SUBVOXEL = 32.0

FA_BINS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

BIN_PART_NAMES = tuple(f"axon_fa{int(round(edge * 100)):02d}" for edge in FA_BINS)


def load_tt(path: Path) -> tuple[list, dict]:

    with gzip.open(str(path), "rb") as handle:
        mat = loadmat(io.BytesIO(handle.read()))
    header = {
        "dimension": np.asarray(mat["dimension"]).ravel().astype(int),
        "voxel_size": np.asarray(mat["voxel_size"]).ravel().astype(float),
    }
    if "trans_to_mni" in mat:
        header["trans_to_mni"] = np.asarray(mat["trans_to_mni"], dtype=float).reshape(4, 4)
    return _decode_tt_buffer(np.ascontiguousarray(mat["track"]).ravel()), header


def _decode_tt_buffer(track: np.ndarray) -> list:
    """Unpack .tt records (int32 count, int32 first point, int8 deltas, in 1/32 voxel)."""
    buf = track.astype(np.uint8).tobytes()
    out: list = []
    pos, end = 0, len(buf)
    while pos + 16 <= end:
        n = int(np.frombuffer(buf, dtype="<i4", count=1, offset=pos)[0])
        if n < 3:
            logger.warning("tractography: .tt stream desynchronised at byte %d (n=%d); "
                           "keeping the %d tracks read so far", pos, n, len(out))
            break
        first = np.frombuffer(buf, dtype="<i4", count=3, offset=pos + 4).astype(np.float64)
        n_delta = n - 3
        deltas = np.frombuffer(buf, dtype=np.int8, count=n_delta,
                               offset=pos + 16).astype(np.float64)
        pos += 16 + n_delta
        points = np.empty((n_delta // 3 + 1, 3))
        points[0] = first
        if n_delta:
            points[1:] = first + np.cumsum(deltas.reshape(-1, 3), axis=0)
        out.append(points / _TT_SUBVOXEL)
    return out


def streamlines_to_t1(streamlines_vox: list, dwi_affine: np.ndarray,
                      invtransforms: list, t1_brain_path: Path,
                      *, min_inside: float = 0.8) -> tuple[list, dict]:
    """DWI array voxels -> T1 world (RAS) mm, through ants.registration(fixed=T1, moving=b0).

    Points move opposite to images in ANTs: fwdtransforms map T1 points to DWI points, so DWI
    points reach the T1 through invtransforms with every affine (.mat) inverted (the
    old call applied the forward rigid and put the reference adult head tracts ~4 mm off).
    """
    import ants
    import pandas as pd

    t1 = nib.load(str(t1_brain_path))
    brain = np.asarray(t1.dataobj) > 0
    inv_t1 = np.linalg.inv(np.asarray(t1.affine, dtype=float))
    lengths = [len(s) for s in streamlines_vox]
    stacked = np.concatenate(streamlines_vox, axis=0) if streamlines_vox else np.zeros((0, 3))
    to_lps = np.array([-1.0, -1.0, 1.0])
    frame = pd.DataFrame(_voxel_to_world(stacked, dwi_affine) * to_lps, columns=["x", "y", "z"])
    names = [str(n) for n in invtransforms]
    moved = ants.apply_transforms_to_points(3, frame, transformlist=names,
                                            whichtoinvert=[n.endswith(".mat") for n in names])
    world = np.asarray(moved[["x", "y", "z"]], dtype=float) * to_lps
    voxel = np.rint((inv_t1 @ np.c_[world, np.ones(len(world))].T).T[:, :3]).astype(int)
    ok = np.all((voxel >= 0) & (voxel < np.array(brain.shape)), axis=1)
    inside = float(brain[voxel[ok, 0], voxel[ok, 1], voxel[ok, 2]].sum()) / max(len(world), 1)
    if inside < min_inside:
        raise RuntimeError(
            f"the tractogram does not land on this subject's T1: {100 * inside:.1f}% "
            f"of its nodes are inside the brain, under the {100 * min_inside:.0f}% floor. "
            f"Shipping it would put every beam somewhere plausible and nowhere real.")
    out, at = [], 0
    for n in lengths:
        out.append(world[at:at + n])
        at += n
    return out, {"inside_brain": round(inside, 4)}


def mask_on_grid(mask: np.ndarray, mask_affine: np.ndarray,
                 target_affine: np.ndarray, target_shape) -> np.ndarray:

    target_shape = tuple(int(n) for n in target_shape)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == target_shape and np.allclose(mask_affine, target_affine):
        return mask
    grid = np.indices(target_shape, dtype=float).reshape(3, -1)
    world = np.asarray(target_affine, dtype=float) @ np.vstack([grid, np.ones(grid.shape[1])])
    voxel = (np.linalg.inv(np.asarray(mask_affine, dtype=float)) @ world)[:3]
    return map_coordinates(mask.astype(np.uint8), voxel, order=0,
                           mode="constant").reshape(target_shape) > 0


def mask_test(mask: np.ndarray, affine: np.ndarray):
    """`f(points) -> bool`, True where a world point falls in a voxel of `mask`."""
    mask = np.asarray(mask, dtype=bool)
    inv = np.linalg.inv(np.asarray(affine, dtype=float))

    def inside(pts: np.ndarray) -> np.ndarray:
        voxel = (inv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        index = np.rint(voxel).astype(int)
        ok = np.all((index >= 0) & (index < np.array(mask.shape)), axis=1)
        ok[ok] = mask[tuple(index[ok].T)]
        return ok

    return inside


def clip_to_envelope(streamlines_world: list, mask: Optional[np.ndarray] = None,
                     affine: Optional[np.ndarray] = None,
                     *, min_length_mm: float = 20.0, inside=None,
                     tract_ids: Optional[np.ndarray] = None) -> tuple:
    """Keep the runs of each streamline that stay inside the tissue.

    `inside` (a points -> bool callable) overrides `mask`/`affine`. With `tract_ids` the
    return is (kept, report, kept_ids) so each run keeps the tract it came from.
    """
    if inside is None:
        inside = mask_test(mask, affine)
    kept: list = []
    kept_ids: list = []
    nodes_in = nodes_out = 0
    lines = [np.asarray(p, dtype=float) for p in streamlines_world]
    flags = inside(np.concatenate(lines)) if lines else np.zeros(0, bool)
    bounds = np.r_[0, np.cumsum([len(p) for p in lines])]
    for i, pts in enumerate(lines):
        if len(pts) < 2:
            continue
        ok = flags[bounds[i]:bounds[i + 1]]
        nodes_in += int(ok.sum())
        nodes_out += int((~ok).sum())
        edges = np.flatnonzero(np.diff(ok.astype(np.int8)))
        for a, b in zip(np.r_[0, edges + 1], np.r_[edges + 1, len(ok)]):
            if not ok[a] or b - a < 2:
                continue
            run = pts[a:b]
            if float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum()) >= min_length_mm:
                kept.append(run)
                if tract_ids is not None:
                    kept_ids.append(int(tract_ids[i]))
    total = nodes_in + nodes_out
    report = {
        "streamlines_in": len(streamlines_world),
        "streamlines_out": len(kept),
        "nodes_inside": nodes_in,
        "nodes_outside": nodes_out,
        "fraction_inside": round(nodes_in / total, 4) if total else 0.0,
    }
    logger.info("tractography: envelope clip kept %d of %d streamlines; %.2f%% of "
                "nodes were inside the tissue", len(kept), len(streamlines_world),
                100.0 * report["fraction_inside"])
    if tract_ids is not None:
        return kept, report, np.asarray(kept_ids, dtype=int)
    return kept, report


def beam_length_study(streamlines_world: list, inside, spacings_mm,
                      tract_ids: Optional[np.ndarray] = None,
                      tract_names: Optional[list] = None,
                      samples_per_beam: int = 5) -> dict:
    """Unclipped beams at each spacing, scored against `inside` (the final mesh's brain).

    A node outside means the tract itself leaves the brain there; a segment point outside
    with both end nodes inside means the beam is too long and cuts a corner of the surface.
    """
    t = np.linspace(0.0, 1.0, samples_per_beam + 2)[1:-1]
    study: dict = {}
    for spacing in spacings_mm:
        nodes_out = nodes = chords = chord_points = cut_corners = 0
        per_tract: dict = {}
        lines = [(i, _resample_polyline(np.asarray(p, dtype=float), float(spacing)))
                 for i, p in enumerate(streamlines_world)]
        lines = [(i, p) for i, p in lines if len(p) >= 2]
        if not lines:
            continue
        node_ok = inside(np.concatenate([p for _i, p in lines]))
        mid_all = np.concatenate([(p[:-1, None, :] + t[None, :, None] * (p[1:] - p[:-1])[:, None, :])
                                  .reshape(-1, 3) for _i, p in lines])
        mid_ok_all = inside(mid_all)
        at_node = at_mid = 0
        for i, pts in lines:
            ok = node_ok[at_node:at_node + len(pts)]
            at_node += len(pts)
            a = pts[:-1]
            mid_ok = mid_ok_all[at_mid:at_mid + len(a) * len(t)].reshape(len(a), len(t))
            at_mid += len(a) * len(t)
            both = ok[:-1] & ok[1:]
            corner = both & ~mid_ok.all(axis=1)
            nodes += len(ok)
            nodes_out += int((~ok).sum())
            chords += int(both.sum())
            chord_points += int(mid_ok.size)
            cut_corners += int(corner.sum())
            if tract_ids is not None:
                name = tract_names[int(tract_ids[i])] if tract_names else str(int(tract_ids[i]))
                entry = per_tract.setdefault(name, [0, 0, 0])
                entry[0] += len(ok)
                entry[1] += int((~ok).sum())
                entry[2] += int(corner.sum())
        study[f"{float(spacing):g}"] = {
            "nodes": nodes, "nodes_outside_pct": round(100.0 * nodes_out / max(nodes, 1), 3),
            "segments_with_both_ends_inside": chords,
            "corner_cutting_segments": cut_corners,
            "corner_cutting_pct": round(100.0 * cut_corners / max(chords, 1), 3),
            "per_tract_nodes_outside_pct": {
                k: round(100.0 * v[1] / max(v[0], 1), 2) for k, v in sorted(per_tract.items())},
        }
        logger.info("beam length study %.3gmm: %.2f%% of nodes outside the brain, %.2f%% of "
                    "inside segments cut a corner", spacing,
                    study[f"{float(spacing):g}"]["nodes_outside_pct"],
                    study[f"{float(spacing):g}"]["corner_cutting_pct"])
    return study


_DYAD_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _dyad_volume(e1: np.ndarray) -> np.ndarray:

    return np.stack([e1[..., i] * e1[..., j] for i, j in _DYAD_PAIRS],
                    axis=-1).astype(np.float32)


def _box_bounds(interesting: np.ndarray, margin: int = 2) -> tuple:

    where = np.argwhere(interesting)
    if not len(where):
        return tuple(slice(0, n) for n in interesting.shape)
    low = np.maximum(where.min(axis=0) - margin, 0)
    high = np.minimum(where.max(axis=0) + margin + 1, interesting.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def track_e1_field(e1: np.ndarray, fa: np.ndarray, affine: np.ndarray,
                   *, seed_count: int = 20_000, fa_min: float = 0.15,
                   step_mm: float = 0.5, angle_max_deg: float = 60.0,
                   max_steps: int = 600, rng_seed: int = 0,
                   mask: Optional[np.ndarray] = None,
                   seed_voxels: Optional[np.ndarray] = None,
                   persistence: float = 0.0,
                   persistence_fa: float = 0.45) -> list:
 
    rng = np.random.default_rng(rng_seed)
    seedable = fa >= fa_min
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != fa.shape:
            raise ValueError(f"tract mask {mask.shape} does not match FA {fa.shape}")
        seedable = seedable & mask
    if seed_voxels is not None:
        seedable = seedable & np.asarray(seed_voxels, dtype=bool)
    candidates = np.argwhere(seedable)
    if len(candidates) == 0:
        return []
    if len(candidates) > seed_count:
        candidates = candidates[rng.choice(len(candidates), seed_count, replace=False)]
    elif seed_count > len(candidates):
        plures = candidates[rng.choice(len(candidates), seed_count - len(candidates),
                                       replace=True)]
        candidates = np.concatenate([candidates,
                                     plures + rng.uniform(-0.5, 0.5, plures.shape)])

    bbox = _box_bounds(fa >= fa_min)
    origo = np.array([s.start for s in bbox], dtype=float)
    fa = fa[bbox]
    e1 = e1[bbox]
    dyads = _dyad_volume(e1)
    if mask is not None:
        mask = mask[bbox]
    candidates = candidates - origo

    voxel_mm = float(np.linalg.norm(np.asarray(affine)[:3, :3], axis=0).mean())
    step_vox = step_mm / max(voxel_mm, 1e-6)
    cos_max = np.cos(np.deg2rad(angle_max_deg))
    shape = np.array(fa.shape, dtype=float)

    def _sample_dyad(points):
        return np.stack([map_coordinates(dyads[..., k], points.T, order=1, mode="constant")
                         for k in range(6)], axis=1)

    def _principal_axis(points, prev):
        """Principal direction of the interpolated dyadic, nearest to `prev`."""
        c = _sample_dyad(points)
        tensor = np.empty((len(points), 3, 3))
        tensor[:, 0, 0], tensor[:, 0, 1], tensor[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
        tensor[:, 1, 1], tensor[:, 1, 2], tensor[:, 2, 2] = c[:, 3], c[:, 4], c[:, 5]
        tensor[:, 1, 0], tensor[:, 2, 0], tensor[:, 2, 1] = c[:, 1], c[:, 2], c[:, 4]
        vec = np.einsum("nij,nj->ni", tensor, prev)
        return np.einsum("nij,nj->ni", tensor, vec)

    def _unit(vec):
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.where(norms > 1e-12, norms, 1.0), norms.ravel()

    halves = []
    for sign in (1.0, -1.0):
        pos = candidates.astype(float)
        seed_mask = np.stack([map_coordinates(e1[..., c], pos.T, order=0, mode="constant")
                            for c in range(3)], axis=1)
        prev, norm0 = _unit(seed_mask * sign)
        alive = norm0 > 1e-8
        trail = [pos.copy()]
        alive_history = [alive.copy()]
        for _ in range(max_steps):
            if not alive.any():
                break
            pos = pos + prev * step_vox
            inside = np.all((pos >= 0) & (pos <= shape - 1), axis=1)
            here = map_coordinates(fa, pos.T, order=1, mode="constant")
            direction, norms = _unit(_principal_axis(pos, prev))
            flip = np.sign(np.einsum("ij,ij->i", direction, prev))
            flip[flip == 0] = 1.0
            direction = direction * flip[:, None]
            turn_ok = np.einsum("ij,ij->i", direction, prev) >= cos_max
            alive = alive & inside & (here >= fa_min) & turn_ok & (norms > 1e-12)
            if mask is not None:
                alive = alive & (map_coordinates(mask.astype(np.uint8), pos.T,
                                                 order=0, mode="constant") > 0)
            if persistence:
                fides = np.clip((here - fa_min) / max(persistence_fa - fa_min, 1e-6), 0.0, 1.0)
                fides = fides + (1.0 - fides) * (1.0 - persistence)
                direction, _ = _unit(direction * fides[:, None]
                                     + prev * (1.0 - fides)[:, None])
            prev = np.where(alive[:, None], direction, prev)
            trail.append(pos.copy())
            alive_history.append(alive.copy())
        halves.append((np.stack(trail, axis=0), np.stack(alive_history, axis=0)))

    forward, backward = halves
    streamlines = []
    for i in range(len(candidates)):
        f_len = int(forward[1][:, i].sum()) + 1
        b_len = int(backward[1][:, i].sum()) + 1
        if f_len + b_len < 4:
            continue
        pts = np.concatenate([backward[0][:b_len, i][::-1], forward[0][1:f_len, i]], axis=0)
        streamlines.append(pts + origo)
    world = [_voxel_to_world(p, affine) for p in streamlines]
    logger.info("tractography: built-in tracker produced %d streamlines from %d seeds",
                len(world), len(candidates))
    return world


def _voxel_to_world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts_h = np.c_[points, np.ones(len(points))]
    return (np.asarray(affine, dtype=float) @ pts_h.T).T[:, :3]


def orient_streamline(points: np.ndarray) -> np.ndarray:

    span = points[-1] - points[0]
    axis = int(np.argmax(np.abs(span)))
    return points[::-1] if span[axis] < 0 else points


def sample_scalar(volume: np.ndarray, affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Trilinearly sample a scalar volume at world-space points."""
    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    pts_h = np.c_[points, np.ones(len(points))]
    voxel = (inv @ pts_h.T).T[:, :3]
    return map_coordinates(volume, voxel.T, order=1, mode="constant")


def _resample_polyline(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Even arc-length samples along a polyline, both endpoints kept."""
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total <= spacing_mm:
        return points[[0, -1]]
    n = max(int(round(total / spacing_mm)), 1)
    want = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(want, arc, points[:, c]) for c in range(3)], axis=1)


def _inside_runs(polylines: list, inside) -> tuple[list, int]:
    """Split each (points, tract) polyline into runs whose nodes and segment midpoints are all
    inside; one batched `inside` call for every point."""
    if not polylines:
        return [], 0
    pts_all = np.concatenate([p for p, _t in polylines])
    mids_all = np.concatenate([0.5 * (p[:-1] + p[1:]) for p, _t in polylines])
    ok_nodes = inside(pts_all)
    ok_mids = inside(mids_all)
    out, dropped, a, m = [], 0, 0, 0
    for pts, tract in polylines:
        n = len(pts)
        node_ok = ok_nodes[a:a + n]
        seg_ok = node_ok[:-1] & node_ok[1:] & ok_mids[m:m + n - 1]
        dropped += int((~seg_ok).sum())
        a, m = a + n, m + n - 1
        edges = np.flatnonzero(np.diff(np.r_[0, seg_ok.astype(np.int8), 0]))
        for start, stop in zip(edges[0::2], edges[1::2]):
            out.append((pts[start:stop + 1], tract))
    return out, dropped


def beam_budget(volume_mm3: float, beam_mm: float, volume_per_beam_mm2: float) -> int:
    """Axon beam elements N with volume / (N * beam length) = volume_per_beam_mm2."""
    return max(int(round(float(volume_mm3) / (float(volume_per_beam_mm2) * float(beam_mm)))), 1)


def streamline_pool(streamlines: list, beam_mm: float, beams: int, least: int = 2000) -> int:
    """Streamlines to pick before clipping so `beams` beams can be met: four times the
    count the mean streamline length suggests, never under `least`."""
    if not len(streamlines):
        return least
    mean = float(np.mean([np.linalg.norm(np.diff(np.asarray(s, float), axis=0), axis=1).sum()
                          for s in streamlines]))
    return max(int(4 * beams / max(mean / float(beam_mm), 1.0)), least)


def select_beam_budget(beams: np.ndarray, tract_ids: np.ndarray, budget: int, seed: int = 0,
                       always: Optional[np.ndarray] = None) -> np.ndarray:
    """Mask of streamlines whose beams add up to about `budget`: every tract gives the same
    fraction of its beams, whole streamlines only, at least one streamline per tract so none
    disappears; `always` lines are kept and counted. Fixed-seed order."""
    beams = np.asarray(beams, dtype=int)
    tract_ids = np.asarray(tract_ids, dtype=int)
    fixed = (np.zeros(len(beams), dtype=bool) if always is None
             else np.asarray(always, dtype=bool).copy())
    left = int(budget) - int(beams[fixed].sum())
    free = ~fixed
    if beams[free].sum() <= left:
        return np.ones(len(beams), dtype=bool)
    rng = np.random.default_rng(seed)
    order = {int(t): rng.permutation(np.flatnonzero(free & (tract_ids == t)))
             for t in np.unique(tract_ids[free])}
    totals = {t: float(beams[idx].sum()) for t, idx in order.items()}

    def pick(fraction):
        keep = fixed.copy()
        for t, idx in order.items():
            goal, got = fraction * totals[t], 0
            for i in idx:
                b = int(beams[i])
                if got == 0 or abs(got + b - goal) < abs(got - goal):
                    keep[i] = True
                    got += b
        return keep

    # the one-per-tract floor and whole streamlines overshoot a little: rescale and keep the
    # closest of a few passes
    fraction = max(left, 0) / max(float(beams[free].sum()), 1.0)
    best = None
    for _ in range(8):
        keep = pick(fraction)
        got = int(beams[keep].sum())
        if best is None or abs(got - budget) < abs(int(beams[best].sum()) - budget):
            best = keep
        if got == budget or got == 0:
            break
        fraction *= budget / got
    # close the last gap one streamline at a time: add the free line nearest the shortfall, or
    # drop the nearest overshoot from a tract that keeps another line
    for _ in range(200):
        gap = int(budget) - int(beams[best].sum())
        if gap == 0:
            break
        if gap > 0:
            cand = np.flatnonzero(~best)
            change = beams[cand]
        else:
            per_tract = np.bincount(np.searchsorted(np.unique(tract_ids), tract_ids),
                                    weights=best.astype(float))
            spare = per_tract[np.searchsorted(np.unique(tract_ids), tract_ids)] > 1
            cand = np.flatnonzero(best & free & spare)
            change = -beams[cand]
        if not len(cand):
            break
        j = int(np.argmin(np.abs(gap - change)))
        if abs(gap - change[j]) >= abs(gap):
            break
        best = best.copy()
        best[cand[j]] = not best[cand[j]]
    return best


def _largest_remainder(count: int, weights: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Split `count` in proportion to `weights`, no entry above its `capacity`."""
    share = np.zeros(len(weights), dtype=int)
    left = int(count)
    while left > 0:
        room = capacity - share
        open_ = room > 0
        if not open_.any():
            break
        exact = left * np.where(open_, weights, 0) / max(float(weights[open_].sum()), 1e-12)
        step = np.minimum(np.floor(exact).astype(int), room)
        if step.sum() == 0:
            order = np.argsort(-np.where(open_, exact, -1.0), kind="stable")
            step = np.zeros_like(share)
            step[order[:left]] = 1
            step = np.minimum(step, room)
        share += step
        left = int(count - share.sum())
    return share


def select_per_tract(tract_ids: np.ndarray, count: int, seed: int = 0,
                     floor: int = 30) -> np.ndarray:
    """`count` streamline indices shared across tracts in proportion to their size, every
    tract keeping at least `floor` (or all it has) so none disappears; fixed-seed sampling."""
    tract_ids = np.asarray(tract_ids, dtype=int)
    if count >= len(tract_ids):
        return np.arange(len(tract_ids))
    tracts, sizes = np.unique(tract_ids, return_counts=True)
    least = np.minimum(sizes, floor)
    if least.sum() > count:  # too few for the floor: one each, then by size
        least = np.minimum(sizes, 1 if count >= len(tracts) else 0)
    # tracts whose fair share is under the floor are held at it; the rest share what is left
    # at one common fraction
    held = np.zeros(len(sizes), dtype=bool)
    while True:
        free_total = float(sizes[~held].sum())
        left = count - int(least[held].sum())
        fair = np.where(held, least, left * sizes / max(free_total, 1.0))
        newly = ~held & (fair < least)
        if not newly.any():
            break
        held |= newly
    share = np.where(held, least, 0)
    share = share + np.where(held, 0, _largest_remainder(
        count - int(share.sum()), np.where(held, 0.0, sizes.astype(float)),
        np.where(held, 0, sizes)))
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(tract_ids == t), size=int(n), replace=False)
            for t, n in zip(tracts, share) if n > 0]
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


def build_axon_beams(streamlines_world: list, fa_path: Optional[Path],
                     *, spacing_mm: float = 3.0, min_length_mm: float = 20.0,
                     max_length_mm: float = 300.0,
                     max_tracts: int = 5000, engine: str = "unknown",
                     parameters: Optional[dict] = None,
                     envelope: Optional[tuple] = None,
                     min_fraction_inside: float = 0.5,
                     parc_data: Optional[tuple] = None,
                     keep_mask: Optional[list] = None,
                     inside=None,
                     tract_ids: Optional[np.ndarray] = None,
                     tract_names: Optional[list] = None,
                     max_streamlines: Optional[int] = None,
                     max_beams: Optional[int] = None,
                     ) -> tuple[dict, dict]:
    """Axon beams binned by FA. `inside` (points -> bool) clips against the final mesh
    instead of `envelope`; `tract_ids`/`tract_names` give a per-tract census.
    `max_streamlines` picks that many streamlines before clipping, the same
    fraction from every tract (or the longest, without tract ids). `max_beams` then keeps
    whole clipped streamlines until the beam elements add up to it, the same fraction of every
    tract's beams (`select_beam_budget`)."""
    report: dict = {"engine": engine, "parameters": dict(parameters or {}),
                    "streamlines_in": len(streamlines_world)}
    ids = (np.asarray(tract_ids, dtype=int) if tract_ids is not None
           else np.full(len(streamlines_world), -1, dtype=int))
    if max_streamlines is not None and len(streamlines_world) > max_streamlines:
        if tract_ids is not None:
            pick = select_per_tract(ids, int(max_streamlines))
            report["selection"] = "same fraction of every tract"
        else:
            length = [float(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum())
                      for p in streamlines_world]
            pick = np.sort(np.argsort(length, kind="stable")[::-1][:int(max_streamlines)])
            report["selection"] = "longest"
        streamlines_world = [streamlines_world[i] for i in pick]
        ids = ids[pick]
    report["streamlines_selected"] = len(streamlines_world)
    fa_volume = fa_affine = None
    if fa_path is not None and Path(fa_path).exists():
        fa_img = nib.load(str(fa_path))
        fa_volume = np.asarray(fa_img.dataobj, dtype=np.float32)
        fa_affine = np.asarray(fa_img.affine, dtype=float)
    else:
        logger.warning("tractography: no FA volume; every tract lands in the lowest bin "
                       "and no FA is reported")

    if inside is not None or envelope is not None:
        streamlines_world, clip, source = clip_to_envelope(
            streamlines_world, *(envelope if inside is None else (None, None)),
            min_length_mm=min_length_mm, inside=inside, tract_ids=np.arange(len(ids)))
        ids = ids[source] if len(source) else np.zeros(0, dtype=int)
        report["streamlines_surviving_clip"] = int(len(np.unique(source)))
        clip["against"] = "final mesh brain elements" if inside is not None else "atlas labels"
        report["envelope"] = clip
        report["envelope"]["passed"] = clip["fraction_inside"] >= min_fraction_inside
        report["envelope"]["min_fraction_inside"] = min_fraction_inside
    else:
        report["envelope"] = {"status": "not clipped; tracts are not checked against anatomy"}

    kept = []
    too_long = 0
    for pts, tract in zip(streamlines_world, ids):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if length < min_length_mm:
            continue
        if length > max_length_mm:
            too_long += 1
            continue
        kept.append((length, orient_streamline(pts), int(tract)))
    report["streamlines_long_enough"] = len(kept)
    report["streamlines_too_long"] = too_long
    if not kept:
        report["status"] = "no streamline reached the minimum length"
        return {}, report

    kept.sort(key=lambda item: item[0], reverse=True)
    if len(kept) > max_tracts:
        logger.info("tractography: %d streamlines over the %d cap; keeping the longest",
                    len(kept), max_tracts)
        report["streamlines_dropped_by_cap"] = len(kept) - max_tracts
        kept = kept[:max_tracts]

    if keep_mask:
        removed_mask = []
        for pts in keep_mask:
            pts = np.asarray(pts, dtype=float)
            if len(pts) < 2:
                continue
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if min_length_mm <= length <= max_length_mm:
                removed_mask.append((length, orient_streamline(pts), -1))
        report["streamlines_praeservatae"] = len(removed_mask)
        kept = removed_mask + kept
    n_exempt = int(report.get("streamlines_praeservatae", 0))

    # each run carries (tract, index into kept) so the budget can keep whole streamlines
    resampled = [(_resample_polyline(pts, spacing_mm), (tract, k))
                 for k, (_length, pts, tract) in enumerate(kept)]
    if inside is not None:
        # a beam whose end nodes are in the brain can still cut a corner of the surface: drop
        # those segments (and any node outside) and keep the runs either side
        resampled, dropped = _inside_runs(resampled, inside)
        report["corner_segments_dropped"] = dropped
    if max_beams is not None:
        line_beams = np.zeros(len(kept), dtype=int)
        line_tract = np.array([tract for _length, _pts, tract in kept], dtype=int)
        for pts, (_tract, k) in resampled:
            line_beams[k] += max(len(pts) - 1, 0)
        report["beams_before_budget"] = int(line_beams.sum())
        report["streamlines_before_budget"] = int((line_beams > 0).sum())
        has = line_beams > 0
        keep = np.zeros(len(kept), dtype=bool)
        keep[has] = select_beam_budget(line_beams[has], line_tract[has], int(max_beams),
                                       always=(np.arange(len(kept)) < n_exempt)[has])
        resampled = [(pts, key) for pts, key in resampled if keep[key[1]]]
        kept = [item if keep[k] else None for k, item in enumerate(kept)]
        report["beam_budget"] = int(max_beams)
        report["streamlines_after_budget"] = int(keep.sum())
    resampled = [(pts, tract) for pts, (tract, _k) in resampled]

    if parc_data is not None:
        from engine.anatomy import tract_bundles

        report["tract_bundles"] = tract_bundles.tract_census(
            [item[1] for item in kept if item is not None], parc_data[0], parc_data[1])

    bins: dict = {name: {"points": [], "segments": [], "fa": [], "tangent": [], "tract": [],
                         "n": 0}
                  for name in BIN_PART_NAMES}
    census: dict = {}
    for pts, tract in resampled:
        if len(pts) < 2:
            continue
        if fa_volume is not None:
            node_fa = sample_scalar(fa_volume, fa_affine, pts)
            mean_fa = float(np.nanmean(node_fa))
        else:
            node_fa = np.full(len(pts), np.nan)
            mean_fa = FA_BINS[0]
        index = min(int(np.searchsorted(FA_BINS, mean_fa, side="right") - 1), len(FA_BINS) - 1)
        if index < 0:
            continue
        bucket = bins[BIN_PART_NAMES[index]]
        offset = bucket["n"]
        bucket["points"].append(pts)
        bucket["segments"].append(
            np.c_[np.arange(len(pts) - 1), np.arange(1, len(pts))] + offset)
        bucket["n"] = offset + len(pts)
        bucket["fa"].append(0.5 * (node_fa[:-1] + node_fa[1:]))
        step = np.diff(pts, axis=0)
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        bucket["tangent"].append(step / np.where(norms > 1e-9, norms, 1.0))
        bucket["tract"].append(np.full(len(pts) - 1, tract, dtype=np.int32))
        if tract >= 0:
            name = tract_names[tract] if tract_names else str(tract)
            entry = census.setdefault(name, {"streamlines": 0, "beams": 0, "length_mm": 0.0})
            entry["streamlines"] += 1
            entry["beams"] += len(pts) - 1
            entry["length_mm"] += float(norms.sum())

    parts: dict = {}
    summary: dict = {}
    for name, bucket in bins.items():
        if not bucket["points"]:
            continue
        points = np.concatenate(bucket["points"], axis=0)
        segments = np.concatenate(bucket["segments"], axis=0)
        beam_fa = np.concatenate(bucket["fa"], axis=0)
        tangent = np.concatenate(bucket["tangent"], axis=0)
        parts[name] = (points, segments, beam_fa, tangent)
        report.setdefault("beam_tract", {})[name] = np.concatenate(bucket["tract"])
        lengths = np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]], axis=1)
        has_fa = bool(np.isfinite(beam_fa).any())
        summary[name] = {
            "tracts": len(bucket["points"]), "beams": int(len(segments)),
            "nodes": int(len(points)), "length_mm": round(float(lengths.sum()), 1),
            "shortest_beam_mm": round(float(lengths.min()), 3),
            "fa_mean": round(float(np.nanmean(beam_fa)), 3) if has_fa else None,
            "mean_direction": [round(float(v), 3) for v in tangent.mean(axis=0)],
        }
    report["parts"] = summary
    if tract_ids is not None:
        report["tracts"] = {k: {**v, "length_mm": round(v["length_mm"], 1)}
                            for k, v in sorted(census.items())}
        if tract_names:
            report["tracts_missing"] = sorted(set(tract_names) - set(census))
    report["status"] = "ok" if parts else "every streamline fell below the lowest FA bin"
    return parts, report


def write_fa_sidecar(path: Path, parts: dict, first_element_id: dict) -> Path:

    lines = ["element_id,part,fa,dir_x,dir_y,dir_z"]
    if os.environ.get("ANATOMESH_PY_FORMAT"):  # reference path; fastfmt must match it
        for name, (_points, segments, beam_fa, tangent) in sorted(parts.items()):
            start = int(first_element_id.get(name, 0))
            for i in range(len(segments)):
                fa = beam_fa[i]
                fa_text = "" if not np.isfinite(fa) else f"{float(fa):.4f}"
                t = tangent[i]
                lines.append(f"{start + i},{name},{fa_text},"
                             f"{t[0]:.4f},{t[1]:.4f},{t[2]:.4f}")
    else:
        from engine.export import fastfmt

        for name, (_points, segments, beam_fa, tangent) in sorted(parts.items()):
            n = len(segments)
            if not n:
                continue
            lines.append(fastfmt.fa_rows(int(first_element_id.get(name, 0)), name,
                                         beam_fa[:n], tangent[:n]).decode("ascii"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("tractography: wrote %d per-beam FA rows to %s", len(lines) - 1, path.name)
    return path


def demo() -> None:
    """Self-check on synthetic data, with no subject or disk I/O."""
    first = np.array([100, 200, 300], dtype="<i4")
    deltas = np.array([1, 2, 3, -1, -2, -3], dtype=np.int8)
    record = (np.array([3 + len(deltas)], dtype="<i4").tobytes()
              + first.tobytes() + deltas.tobytes())
    got = _decode_tt_buffer(np.frombuffer(record * 2, dtype=np.uint8))
    assert len(got) == 2, len(got)
    assert len(got[0]) == 3, got[0].shape
    assert np.allclose(got[0][0], first / _TT_SUBVOXEL)
    assert np.allclose(got[0][1], (first + [1, 2, 3]) / _TT_SUBVOXEL)
    assert np.allclose(got[0][2], first / _TT_SUBVOXEL)

    down = np.array([[0.0, 0, 10], [0, 0, 5], [0, 0, 0]])
    assert orient_streamline(down)[0][2] == 0.0
    assert np.allclose(orient_streamline(down[::-1]), down[::-1])

    line = np.array([[0.0, 0, 0], [0, 0, 30]])
    out = _resample_polyline(line, 3.0)
    assert len(out) == 11 and np.allclose(out[-1], line[-1]), out.shape

    shape = (12, 12, 40)
    e1 = np.zeros(shape + (3,), dtype=np.float32)
    e1[..., 2] = 1.0
    fa = np.zeros(shape, dtype=np.float32)
    fa[4:8, 4:8, 2:38] = 0.9
    lines = track_e1_field(e1, fa, np.eye(4), seed_count=20, step_mm=0.5, max_steps=200)
    assert lines, "the tracker found nothing in a field that is nothing but fibre"
    spans = [abs(line[-1][2] - line[0][2]) for line in lines]
    assert max(spans) > 25, max(spans)
    assert max(spans) < 40, max(spans)

    parts, report = build_axon_beams(lines, None, spacing_mm=2.0, min_length_mm=10.0,
                                     engine="demo")
    assert parts, report
    name, (points, segments, beam_fa, tangent) = next(iter(parts.items()))
    assert name in BIN_PART_NAMES
    assert segments.max() < len(points), (segments.max(), len(points))
    assert len(beam_fa) == len(segments) == len(tangent)
    assert abs(report["parts"][name]["mean_direction"][2]) > 0.9, report["parts"][name]
    short = [np.array([[0.0, 0, 0], [0, 0, 5]])]
    _, short_report = build_axon_beams(short, None, min_length_mm=20.0)
    assert short_report["streamlines_long_enough"] == 0, short_report
    print("tractography.demo ok:", {k: v["beams"] for k, v in report["parts"].items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
