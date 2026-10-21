"""Voxel topology cleanup using foreground and complement connectivity."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage

from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

def _nonmanifold_tables() -> tuple[np.ndarray, np.ndarray]:
    """Flag invalid octant patterns and select a minimal valid filling."""
    adj = [[b2 for b2 in range(8) if bin(b1 ^ b2).count("1") == 1] for b1 in range(8)]

    def connected(pattern: int) -> bool:
        bits = [b for b in range(8) if pattern >> b & 1]
        if len(bits) <= 1:
            return True
        seen = {bits[0]}
        stack = [bits[0]]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if pattern >> nb & 1 and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(bits)

    bad = np.zeros(256, dtype=bool)
    fix = np.zeros(256, dtype=np.uint8)
    for pattern in range(256):
        if connected(pattern) and connected(255 ^ pattern):
            continue
        bad[pattern] = True
        free = [b for b in range(8) if not pattern >> b & 1]
        best: Optional[int] = None
        for size in range(1, len(free) + 1):
            for combo in _combinations(free, size):
                add = 0
                for b in combo:
                    add |= 1 << b
                if connected(pattern | add) and connected(255 ^ (pattern | add)):
                    best = add
                    break
            if best is not None:
                break
        fix[pattern] = best if best is not None else 0
    return bad, fix


def _combinations(items: list[int], size: int):
    """Yield index combinations, smallest bit pattern first (itertools-free)."""
    if size == 0:
        yield ()
        return
    for i in range(len(items) - size + 1):
        for rest in _combinations(items[i + 1:], size - 1):
            yield (items[i],) + rest


_NONMANIFOLD_BAD, _NONMANIFOLD_FIX = _nonmanifold_tables()


def _background_seal_table() -> np.ndarray:
    """For every pinched pattern, the bits of every clump but the largest."""
    seal = np.zeros(256, dtype=np.uint8)
    adj = [[b2 for b2 in range(8) if bin(b1 ^ b2).count("1") == 1] for b1 in range(8)]
    for pattern in range(256):
        if not _NONMANIFOLD_BAD[pattern]:
            continue
        bits = [b for b in range(8) if pattern >> b & 1]
        seen: set[int] = set()
        clumps: list[list[int]] = []
        for start in bits:
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adj[cur]:
                    if pattern >> nb & 1 and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            clumps.append(comp)
        clumps.sort(key=len, reverse=True)
        mask = 0
        for comp in clumps[1:]:
            for b in comp:
                mask |= 1 << b
        seal[pattern] = mask
    return seal


_BACKGROUND_SEAL = _background_seal_table()


def _pattern_codes(mask: np.ndarray) -> np.ndarray:
    """2x2x2 occupancy pattern at every interior node of `mask`."""
    nx, ny, nz = mask.shape
    code = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.uint8)
    for u in (0, 1):
        for v in (0, 1):
            for w in (0, 1):
                code |= mask[u:nx - 1 + u, v:ny - 1 + v, w:nz - 1 + w].astype(np.uint8) \
                    << np.uint8(4 * u + 2 * v + w)
    return code


def count_nonmanifold(labels: np.ndarray, values: Optional[list[int]] = None) -> dict[int, int]:
    """Count pinched nodes per material, without modifying anything."""
    if values is None:
        values = [int(v) for v in np.unique(labels)]
    padded = np.pad(labels, 1, constant_values=0)
    out: dict[int, int] = {}
    if fast_kernels.HAVE_NUMBA and len(values) and min(values) >= 0:
        counts = fast_kernels.pinch_counts(padded, int(max(values)) + 1, _NONMANIFOLD_BAD)
        return {int(v): int(counts[v]) for v in values if counts[v]}
    for value in values:
        n = int(_NONMANIFOLD_BAD[_pattern_codes(padded == value)].sum())
        if n:
            out[value] = n
    return out


def seal_background_pinches(labels: np.ndarray, max_passes: int = 12) -> tuple[np.ndarray, dict]:
    """Close residual background pinches by filling them, not by perforating."""
    out = np.asarray(labels).copy()
    before = int(_NONMANIFOLD_BAD[_pattern_codes(np.pad(out, 1, constant_values=0) == 0)].sum())
    filled_total = 0
    passes = 0
    for passes in range(1, max_passes + 1):
        padded = np.pad(out, 1, constant_values=0)
        code = _pattern_codes(padded == 0)
        bad = _NONMANIFOLD_BAD[code]
        if not bad.any():
            passes -= 1
            break
        seal = np.where(bad, _BACKGROUND_SEAL[code], np.uint8(0))
        nx, ny, nz = padded.shape
        take = np.zeros(padded.shape, dtype=bool)
        for u in (0, 1):
            for v in (0, 1):
                for w in (0, 1):
                    bit = np.uint8(4 * u + 2 * v + w)
                    take[u:nx - 1 + u, v:ny - 1 + v, w:nz - 1 + w] |= \
                        ((seal >> bit) & np.uint8(1)) != 0
        take = take[1:-1, 1:-1, 1:-1] & (out == 0)
        if not take.any():
            break
        fill = _majority_neighbour(out, take)
        got = take & (fill > 0)
        if not got.any():
            break
        out[got] = fill[got]
        filled_total += int(got.sum())

    after = int(_NONMANIFOLD_BAD[_pattern_codes(np.pad(out, 1, constant_values=0) == 0)].sum())
    if before:
        logger.info("seal_background_pinches: %d background pinch(es) -> %d, %d voxel(s) "
                    "filled in %d pass(es)", before, after, filled_total, passes)
    return out, {"passes": passes, "voxels_filled": filled_total,
                 "before": before, "after": after}


def _majority_neighbour(labels: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Most common non-zero 6-neighbour label of every voxel in `where`."""
    counts: dict[int, np.ndarray] = {}
    for value in (int(v) for v in np.unique(labels) if v):
        mask = (labels == value).astype(np.int16)
        total = np.zeros(labels.shape, dtype=np.int16)
        for axis in range(3):
            for shift in (-1, 1):
                total += np.roll(mask, shift, axis=axis)
        counts[value] = total
    best = np.zeros(labels.shape, dtype=labels.dtype)
    best_n = np.zeros(labels.shape, dtype=np.int16)
    for value, total in counts.items():
        take = where & (total > best_n)
        best[take] = value
        best_n[take] = total[take]
    return best


def _claims(work: np.ndarray, winner: np.ndarray, winner_fix: np.ndarray, rank: np.ndarray,
            priority: list[int]) -> np.ndarray:
    """Voxel -> material that claims it (-1 none): of the materials whose pinch fix wants the
    voxel, the highest-priority one, when it outranks the voxel's current material.

    Same claims as walking `priority` in order and letting each material take the wanted
    voxels nobody took yet (a material that is refused by the rank test means every lower one
    is refused too), but it touches only the pinched nodes instead of sweeping the whole
    lattice 8 times per material (~10 s a call on a 256^3 head)."""
    claim = np.full(work.shape, -1, dtype=np.int16)
    pi, qi, ri = np.nonzero(winner >= 0)
    if not len(pi):
        return claim
    value_rank = rank[winner[pi, qi, ri]].astype(np.int64)
    fix = winner_fix[pi, qi, ri]
    none = len(priority) + 1
    best = np.full(work.size, none, dtype=np.int64)
    for u in (0, 1):
        for v in (0, 1):
            for w in (0, 1):
                bit = np.uint8(4 * u + 2 * v + w)
                sel = ((fix >> bit) & 1) != 0
                ti, tj, tk = pi[sel] + u - 1, qi[sel] + v - 1, ri[sel] + w - 1
                ok = ((ti >= 0) & (ti < work.shape[0]) & (tj >= 0) &
                      (tj < work.shape[1]) & (tk >= 0) & (tk < work.shape[2]))
                flat = np.ravel_multi_index((ti[ok], tj[ok], tk[ok]), work.shape)
                np.minimum.at(best, flat, value_rank[sel][ok])
    hit = np.flatnonzero(best < none)
    take = hit[rank[work.ravel()[hit]] > best[hit]]
    order = np.asarray(priority, dtype=np.int16)
    claim.ravel()[take] = order[best[take]]
    return claim


def clean_topology(
    labels: np.ndarray, max_passes: int = 12, priority: Optional[list[int]] = None,
    seal_background: bool = False,
    preserve_connectivity: bool = True,
) -> tuple[np.ndarray, dict]:
    """Repair pinched (non-manifold) voxel configurations, in place-safe form."""
    unique_values, unique_counts = np.unique(labels, return_counts=True)
    values = [int(v) for v in unique_values]
    counts_before = {int(v): int(c) for v, c in zip(unique_values, unique_counts)}
    if priority is None:
        priority = sorted(v for v in values if v != 0) + ([0] if 0 in values else [])
    rank = np.full(int(max(values + list(priority))) + 2, len(priority), dtype=np.int16)
    for i, value in enumerate(priority):
        rank[value] = i

    occ = labels != 0
    if not occ.any():
        return labels.copy(), {"passes": 0, "voxels_changed": 0, "before": {},
                               "after": {}, "converged": True,
                               "voxel_count_before": counts_before,
                               "voxel_count_after": counts_before.copy(),
                               "voxel_count_delta": {v: 0 for v in values},
                               "connectivity_rejections": 0, "termination": "empty",
                               "topology_resolved": True}
    idx = np.argwhere(occ)
    lo = np.maximum(idx.min(axis=0) - 2, 0)
    hi = np.minimum(idx.max(axis=0) + 3, np.array(labels.shape))
    view = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    full = labels.copy()
    out = full[view].copy()
    original = out.copy()

    before = count_nonmanifold(out, values)
    changed_total = 0
    changed_values: set[int] = set()
    rejected_connectivity = 0
    converged = False
    passes = 0
    scan: tuple[slice, slice, slice] = (slice(None), slice(None), slice(None))
    interior_only = False
    for passes in range(1, max_passes + 1):
        work = out[scan]
        padded = np.pad(work, 1, constant_values=0)
        node_shape = tuple(s - 1 for s in padded.shape)
        if fast_kernels.HAVE_NUMBA:     # every value in one compiled pass, same winners
            winner, winner_fix = fast_kernels.pinch_winners(
                padded, rank, len(priority), _NONMANIFOLD_BAD, _NONMANIFOLD_FIX)
        else:
            winner = np.full(node_shape, -1, dtype=np.int16)
            winner_fix = np.zeros(node_shape, dtype=np.uint8)
            for value in priority:
                code = _pattern_codes(padded == value)
                take = _NONMANIFOLD_BAD[code] & (winner < 0)
                if not take.any():
                    continue
                winner[take] = value
                winner_fix[take] = _NONMANIFOLD_FIX[code][take]
        if not (winner >= 0).any():
            converged = True
            break

        claim = _claims(work, winner, winner_fix, rank, priority)
        if interior_only:
            claim[0, :, :] = claim[-1, :, :] = -1
            claim[:, 0, :] = claim[:, -1, :] = -1
            claim[:, :, 0] = claim[:, :, -1] = -1
        apply = claim >= 0
        if preserve_connectivity:
            from engine.mesh.label_connectivity import connectivity_safe_claims

            accepted = connectivity_safe_claims(work, claim)
            rejected_connectivity += int(np.count_nonzero(apply & ~accepted))
            apply &= accepted
        if not apply.any():
            converged = True
            break
        changed_values.update(int(v) for v in np.unique(work[apply]))
        changed_values.update(int(v) for v in np.unique(claim[apply]))
        work[apply] = claim[apply]
        changed_total += int(apply.sum())

        base = np.array([s.start or 0 for s in scan])
        written = np.argwhere(apply)
        lo2 = np.maximum(base + written.min(axis=0) - 2, 0)
        hi2 = np.minimum(base + written.max(axis=0) + 3, np.array(out.shape))
        scan = tuple(slice(int(a), int(b)) for a, b in zip(lo2, hi2))
        interior_only = True

    full[view] = out
    seal_report: dict = {}
    if seal_background:
        out, seal_report = seal_background_pinches(out)
        full[view] = out
    alternative_report = {}
    if preserve_connectivity and count_nonmanifold(out, values):
        from engine.mesh.topology_alternatives import repair_alternative_fillings

        alternative, alternative_report = repair_alternative_fillings(out, priority)
        changed = alternative != out
        changed_values.update(int(v) for v in np.unique(out[changed]))
        changed_values.update(int(v) for v in np.unique(alternative[changed]))
        changed_total += int(np.count_nonzero(changed))
        out = alternative
        full[view] = out
    after = count_nonmanifold(out, values)
    termination = "stationary" if converged else "iteration_limit"

    from scipy.ndimage import label as cc_label
    components = {}
    for value in values:
        if value == 0 or value not in changed_values:
            continue  # unchanged mask, component count can't have moved
        n0 = int(cc_label(original == value)[1])
        n1 = int(cc_label(out == value)[1])
        components[value] = (n0, n1)
        if n1 > n0:
            logger.warning("clean_topology: label %d components increased %d -> %d; "
                           "anatomical continuity requires review", value, n0, n1)
    if after:
        logger.info("clean_topology: %d passes, %d voxels changed, residual pinches %s "
                    "(stationary=%s) -- unrepairable without perforating a higher-priority "
                    "material; strict domain topology validation is still required",
                    passes, changed_total, after, converged)
    report = {"passes": passes, "voxels_changed": changed_total, "before": before,
              "after": after, "components": components, "converged": converged,
              "connectivity_rejections": rejected_connectivity, "termination": termination,
              "topology_resolved": not bool(after), "alternative_fillings": alternative_report}
    final_values, final_counts = np.unique(full, return_counts=True)
    counts_after = {int(v): int(c) for v, c in zip(final_values, final_counts)}
    report.update(voxel_count_before=counts_before, voxel_count_after=counts_after,
                  voxel_count_delta={v: counts_after.get(v, 0) - counts_before.get(v, 0)
                                     for v in sorted(set(counts_before) | set(counts_after))})
    if seal_report:
        report["seal_background"] = seal_report
    return full, report
