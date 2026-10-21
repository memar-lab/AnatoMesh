"""Raise the quality of a quad shell by sliding its nodes within its own surface.

Smoothing (Taubin) and offsetting (to the mid-thickness of a layer) give a sheet its shape but
also shear its quads. Shear is in-plane: moving every node only along the sheet, and putting it
back on the sheet's own surface after every move, removes the shear and leaves the shape alone.

The surface each node is held to is the sheet as it came in, as four triangles per quad around
the quad centre. A move is kept only when the worst quad around the node does not get worse and
the node's quads get better overall (quad scaled Jacobian of export.validate, the number the
deck report uses). Nodes on an open rim stay put. Nodes that share a quad never move in the
same step.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from engine.mesh.hex_quality import _closest_on_triangles, _edges_on_surface

logger = logging.getLogger(__name__)

SCALES = (1.0, 0.5, 0.25, 0.1)
MEAN_WEIGHT = 0.25


def _quad_sj(c):
    from engine.export.validate import _quad_scaled_jacobian

    return _quad_scaled_jacobian(c)


def _incidence(quads, n):
    flat = quads.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n)
    width = int(counts.max())
    rank = np.arange(len(flat)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    inc = np.full((n, width), -1, dtype=np.int64)
    loc = np.full((n, width), -1, dtype=np.int64)
    inc[flat[order], rank] = order // 4
    loc[flat[order], rank] = order % 4
    return inc, loc


def _colour(quads, inc, n):
    colour = np.full(n, -1, dtype=np.int64)
    for v in range(n):
        q = inc[v][inc[v] >= 0]
        used = set(colour[quads[q]].ravel().tolist())
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    return colour


def _rim(quads, n):
    edges = np.sort(np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    rim = np.zeros(n, dtype=bool)
    rim[np.unique(uniq[counts == 1])] = True
    return rim


class _Surface:
    """The incoming sheet around each node (its own quads, four triangles each); a node is only
    ever projected onto those, so it cannot jump to a neighbouring fold of the sheet."""

    def __init__(self, verts, quads):
        from engine.mesh.hex_quality import MAX_UMBRELLA_FACES, _umbrellas

        n = len(verts)
        members = np.arange(n)
        self.tris, _normal, bend = _umbrellas(np.asarray(verts, dtype=float), quads, members)
        counts = np.bincount(np.asarray(quads).ravel(), minlength=n)
        self.too_many = (counts > MAX_UMBRELLA_FACES) | (bend >= 180.0)
        from engine.mesh.hex_quality import _interface_neighbours

        self.nb, self.tol = _interface_neighbours(np.asarray(verts, dtype=float),
                                                  np.asarray(quads), members)

    def project(self, p, rows):
        return _closest_on_triangles(p, self.tris[rows])

PROTECTED = (0.9, 0.95, 0.99)


def _crosses(before, after):
    """True per node where a move would push one of its elements below 0.9, 0.95 or 0.99 from
    above, so no element band ever loses members to make room for another."""
    out = np.zeros(len(before), dtype=bool)
    for level in PROTECTED:
        out |= np.any((before >= level) & (after < level), axis=1)
    return out


def _local(verts, quads, inc, loc, rows, positions):
    q, lc = inc[rows], loc[rows]
    valid = q >= 0
    c = verts[quads[np.maximum(q, 0)]]                        # (n, k, 4, 3)
    n, k = q.shape
    c[np.repeat(np.arange(n), k), np.tile(np.arange(k), n), np.maximum(lc, 0).ravel()] = \
        np.repeat(positions, k, axis=0)
    sj = _quad_sj(c.reshape(n * k, 4, 3)).reshape(n, k)
    low = np.where(valid, sj, np.inf).min(1)
    mean = np.where(valid, sj, 0.0).sum(1) / np.maximum(valid.sum(1), 1)
    return low, mean, np.where(valid, sj, np.nan)


def _targets(verts, quads, inc, loc, rows):
    """Parallelogram completion over the node's quads, and the mean of its edge neighbours."""
    q, lc = inc[rows], loc[rows]
    valid = q >= 0
    c = quads[np.maximum(q, 0)]
    lc = np.maximum(lc, 0)
    take = lambda off: verts[np.take_along_axis(c, ((lc + off) % 4)[..., None], 2)[..., 0]]  # noqa
    nxt, opp, prv = take(1), take(2), take(3)
    para = nxt + prv - opp
    lap = 0.5 * (nxt + prv)
    w = valid[..., None].astype(float)
    n = np.maximum(w.sum(1), 1.0)
    return (para * w).sum(1) / n, (lap * w).sum(1) / n


def _normals(verts, quads, inc, rows):
    q = inc[rows]
    c = verts[quads[np.maximum(q, 0)]]
    nrm = np.cross(c[:, :, 2] - c[:, :, 0], c[:, :, 3] - c[:, :, 1])
    nrm = np.where((q >= 0)[..., None], nrm, 0.0)
    ref = nrm[:, 0]
    sign = np.sign(np.einsum("nkj,nj->nk", nrm, ref))
    out = (nrm * np.where(sign == 0, 1.0, sign)[..., None]).sum(1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def flip_edges(verts, quads, *, min_new_sj: float = 0.3, passes: int = 20):
    """Re-pair quads to cancel irregular nodes; same quads count, same node positions.

    Two quads sharing an edge make a hexagon y s1 s2 x t1 t2 (shared edge y-x). It can be cut
    into two quads along y-x, s1-t1 or s2-t2. A cut is changed when the sum of (valence - 4)^2
    over the hexagon's nodes falls, both new quads have a scaled Jacobian above `min_new_sj`,
    and the new diagonal is not already an edge. On a closed sheet the number of 3-valent
    minus 5-valent nodes is fixed by the topology (sum of 4 - valence = 4 x Euler
    characteristic), so flips cancel irregular nodes in pairs.
    """
    quads = [list(map(int, q)) for q in np.asarray(quads)]
    n = len(verts)
    valence = np.zeros(n, dtype=np.int64)
    edges: dict = {}
    for qi, q in enumerate(quads):
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            key = (min(a, b), max(a, b))
            edges.setdefault(key, []).append(qi)
    for (a, b) in edges:
        valence[a] += 1
        valence[b] += 1
    flips = 0
    for _ in range(passes):
        changed = 0
        for key in list(edges):
            owners = edges.get(key)
            if owners is None or len(owners) != 2:
                continue
            owners = list(owners)                       # edges[key] is edited below
            q1, q2 = quads[owners[0]], quads[owners[1]]
            x, y = key
            i = q1.index(x)
            if q1[(i + 1) % 4] != y:                    # want x -> y along q1
                x, y = y, x
                i = q1.index(x)
                if q1[(i + 1) % 4] != y:
                    continue
            j = q1.index(y)
            s1, s2 = q1[(j + 1) % 4], q1[(j + 2) % 4]
            k = q2.index(x)
            if q2[(k + 3) % 4] != y:
                continue                                # orientations disagree: leave it
            t1, t2 = q2[(k + 1) % 4], q2[(k + 2) % 4]
            hexagon = [y, s1, s2, x, t1, t2]
            if len(set(hexagon)) != 6:
                continue
            def energy(add, remove):                   # noqa: E306
                v = valence[hexagon].copy()
                for node in add:
                    v[hexagon.index(node)] += 1
                for node in remove:
                    v[hexagon.index(node)] -= 1
                return int(((v - 4) ** 2).sum())
            best = (energy((), ()), None)
            for a, b, qa, qb in ((s1, t1, [s1, s2, x, t1], [t1, t2, y, s1]),
                                 (s2, t2, [s2, x, t1, t2], [t2, y, s1, s2])):
                if (min(a, b), max(a, b)) in edges:
                    continue
                e = energy((a, b), (x, y))
                if e >= best[0]:
                    continue
                sj = _quad_sj(np.asarray(verts)[np.array([qa, qb])])
                if sj.min() < min_new_sj:
                    continue
                best = (e, (a, b, qa, qb))
            if best[1] is None:
                continue
            a, b, qa, qb = best[1]
            for q in (q1, q2):                          # drop the old edges of both quads
                for m in range(4):
                    ek = (min(q[m], q[(m + 1) % 4]), max(q[m], q[(m + 1) % 4]))
                    lst = edges[ek]
                    lst.remove(owners[0] if q is q1 else owners[1])
                    if not lst:
                        del edges[ek]
            quads[owners[0]], quads[owners[1]] = qa, qb
            for qi in owners:
                q = quads[qi]
                for m in range(4):
                    ek = (min(q[m], q[(m + 1) % 4]), max(q[m], q[(m + 1) % 4]))
                    edges.setdefault(ek, []).append(qi)
            valence[x] -= 1
            valence[y] -= 1
            valence[a] += 1
            valence[b] += 1
            changed += 1
        flips += changed
        if not changed:
            break
    return np.asarray(quads, dtype=np.int64), {"flips": flips,
                                               "valence": {int(k): int(c) for k, c in
                                                           enumerate(np.bincount(valence)) if c}}


def improve_sheet_in_surface(verts, quads, *, sweeps: int = 60, target: float = 0.98,
                             name: str = "sheet", surface_quads=None):
    """Slide nodes within the sheet's own surface to raise quad quality; returns (verts, report)."""
    t0 = time.time()
    verts = np.array(verts, dtype=float, copy=True)
    quads = np.asarray(quads, dtype=np.int64)
    n = len(verts)
    surface = _Surface(verts, quads if surface_quads is None else np.asarray(surface_quads))
    inc, loc = _incidence(quads, n)
    colour = _colour(quads, inc, n)
    movable_all = ~_rim(quads, n) & (inc[:, 0] >= 0) & ~surface.too_many
    sj = _quad_sj(verts[quads])
    before = {"min": float(sj.min()), "mean": float(sj.mean()),
              "below_0p7": int((sj < 0.7).sum()), "below_0p9": int((sj < 0.9).sum())}
    dirty = np.ones(n, dtype=bool)
    total = float(sj.sum())
    sweep = -1
    for sweep in range(sweeps):
        movable = np.zeros(n, dtype=bool)
        movable[np.unique(quads[sj < target])] = True
        movable &= movable_all & dirty
        dirty = np.zeros(n, dtype=bool)
        moved = 0
        for c in np.unique(colour):
            rows = np.flatnonzero(movable & (colour == c))
            if not len(rows):
                continue
            low0, mean0, each0 = _local(verts, quads, inc, loc, rows, verts[rows])
            score0 = low0 + MEAN_WEIGHT * mean0
            best_pos, best_score = verts[rows].copy(), score0.copy()
            normal = _normals(verts, quads, inc, rows)
            for goal in _targets(verts, quads, inc, loc, rows):
                step = goal - verts[rows]
                step -= (step * normal).sum(1, keepdims=True) * normal
                for s in SCALES:
                    cand = surface.project(verts[rows] + s * step, rows)
                    low, mean, each = _local(verts, quads, inc, loc, rows, cand)
                    score = low + MEAN_WEIGHT * mean
                    take = (low >= low0 - 1e-12) & (score > best_score + 1e-9) & ~_crosses(each0, each)
                    take &= _edges_on_surface(cand, verts, surface.nb[rows], surface.tris[rows],
                                              surface.tol[rows])
                    best_pos[take], best_score[take] = cand[take], score[take]
            take = best_score > score0 + 1e-9
            verts[rows[take]] = best_pos[take]
            moved += int(take.sum())
            touched = np.unique(inc[rows[take]])
            touched = touched[touched >= 0]
            if len(touched):
                sj[touched] = _quad_sj(verts[quads[touched]])
                dirty[np.unique(quads[touched])] = True
        gain, total = float(sj.sum()) - total, float(sj.sum())
        first_gain = gain if sweep == 0 else first_gain
        if moved == 0 or gain < max(1e-6 * len(sj), 0.05 * first_gain):
            break
    off = np.linalg.norm(surface.project(verts, np.arange(n)) - verts, axis=1).max()
    report = {"before": before,
              "after": {"min": float(sj.min()), "mean": float(sj.mean()),
                        "below_0p7": int((sj < 0.7).sum()), "below_0p9": int((sj < 0.9).sum())},
              "sweeps": sweep + 1, "off_surface_max_mm": float(off),
              "seconds": round(time.time() - t0, 1)}
    logger.info("%s in-surface repair: worst %.3f -> %.3f, mean %.4f -> %.4f, %d -> %d quads "
                "under 0.9, nodes at most %.2g mm off the sheet", name, before["min"],
                report["after"]["min"], before["mean"], report["after"]["mean"],
                before["below_0p9"], report["after"]["below_0p9"], off)
    return verts, report
