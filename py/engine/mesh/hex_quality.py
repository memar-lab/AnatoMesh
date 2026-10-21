"""Raise the scaled Jacobian of a smoothed hex mesh without changing what it looks like.

After the reference smoothing the shape of every tissue is fixed by where its interfaces are. This
pass only moves nodes in ways that leave those interfaces where they are:

  free nodes      (inside one tissue)              move anywhere
  surface nodes   (on one interface between two    slide in the interface, and every position
                   regions, or on the outside)      is projected back onto the ORIGINAL faces
  feature nodes   (three or more regions meet)     do not move

A move is kept only if the worst element around the node gets better (ties broken by the
mean), so the global minimum never drops and inverted elements are untangled first. Nodes
that share an element never move in the same step (lattice parity colouring), so every
accepted step is measured exactly.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from engine.mesh.hex_mesh import _unique_rows

from engine.mesh import fast_kernels
from engine.mesh.voxel_hex import rows_in_parallel

logger = logging.getLogger(__name__)

# corner c of a hex: its three edge neighbours (same order as export.validate)
_CORNER_NEIGHBOURS = np.array([(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
                               (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)])
_HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [3, 7, 6, 2],
                       [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5]])


def _face_opposite():
    """For corner c and each pair of its neighbours, the corner that closes that face."""
    faces = [set(f) for f in _HEX_FACES]
    out = np.zeros((8, 3), dtype=int)
    for c in range(8):
        a, b, d = _CORNER_NEIGHBOURS[c]
        for k, (p, q) in enumerate(((a, b), (a, d), (b, d))):
            face = next(f for f in faces if {c, p, q} <= f)
            out[c, k] = (face - {c, p, q}).pop()
    return out


_FACE_OPPOSITE = _face_opposite()
_HEX_TETS_VOLUME = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [3, 4, 6, 7],
                             [1, 4, 5, 6]])
SCORE_MEAN_WEIGHT = 0.25
CREASE_DEG = 30.0
MAX_UMBRELLA_FACES = 8
SCALES = (1.0, 0.5, 0.25, 0.1)


def _incidence(hexes: np.ndarray, n_nodes: int):
    """(n_nodes, k) incident hexes and the node's corner index in each, padded with -1."""
    flat = hexes.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_nodes)
    width = int(counts.max())
    start = np.r_[0, np.cumsum(counts)[:-1]]
    rank = np.arange(len(flat)) - np.repeat(start, counts)
    inc = np.full((n_nodes, width), -1, dtype=np.int64)
    loc = np.full((n_nodes, width), -1, dtype=np.int64)
    inc[flat[order], rank] = order // 8
    loc[flat[order], rank] = order % 8
    return inc, loc, counts


def classify_nodes(hexes: np.ndarray, part: np.ndarray, n_nodes: int, incidence=None):
    """0 free, 1 surface (two sides), 2 feature (three or more sides)."""
    inc, _loc, counts = incidence if incidence is not None else _incidence(hexes, n_nodes)
    parts = np.where(inc >= 0, part[np.maximum(inc, 0)], np.iinfo(np.int64).max)
    parts = np.sort(parts, axis=1)
    valid = parts != np.iinfo(np.int64).max
    distinct = valid[:, 0].astype(int) + (valid[:, 1:] & (np.diff(parts, axis=1) != 0)).sum(1)
    sides = distinct + (counts < 8)
    kind = np.where(sides <= 1, 0, np.where(sides == 2, 1, 2))
    kind[counts == 0] = 2
    return kind, inc


def _interface_faces(hexes: np.ndarray, part: np.ndarray, with_sides: bool = False):
    """Quad faces on an interface between two parts or on the outside."""
    faces = hexes[:, _HEX_FACES].reshape(-1, 4)
    owner = np.repeat(np.arange(len(hexes)), 6)
    key = np.sort(faces, axis=1)
    inverse, counts = _unique_rows(key)
    inverse = inverse.ravel()
    single = counts[inverse] == 1
    order = np.argsort(inverse, kind="stable")
    paired = order[counts[inverse[order]] == 2].reshape(-1, 2)
    differ = part[owner[paired[:, 0]]] != part[owner[paired[:, 1]]]
    keep = np.zeros(len(faces), dtype=bool)
    keep[single] = True
    keep[paired[differ, 0]] = True
    if not with_sides:
        return faces[keep]
    other = np.full(len(faces), -1, dtype=np.int64)        # -1 = outside
    other[paired[:, 0]] = part[owner[paired[:, 1]]]
    other[paired[:, 1]] = part[owner[paired[:, 0]]]
    mine = part[owner]
    sides = np.sort(np.stack([mine, other], axis=1), axis=1)
    return faces[keep], sides[keep]


def _feature_curves(faces: np.ndarray, sides: np.ndarray, n_nodes: int):
    """Edges where two different interfaces meet, and for each node on exactly two of them its
    two curve neighbours (-1 elsewhere)."""
    edges = np.concatenate([faces[:, [i, (i + 1) % 4]] for i in range(4)])
    pair = np.tile(sides, (4, 1))
    edges = np.sort(edges, axis=1)
    code = pair[:, 0] * 1_000_003 + pair[:, 1]
    # edges with two or more distinct side pairs, in lexicographic edge order (the order the
    # curve neighbours are filled in below); 1-D keys instead of np.unique(axis=0) (~10 s)
    eid, _n = _unique_rows(edges)                            # group ids in lexicographic order
    code = code - code.min()
    distinct = np.unique(eid * (int(code.max()) + 1) + code) // (int(code.max()) + 1)
    n_pairs = np.bincount(distinct, minlength=int(eid.max()) + 1 if len(eid) else 0)
    first = np.full(len(n_pairs), -1, dtype=np.int64)
    first[eid[::-1]] = np.arange(len(eid))[::-1]
    feature = edges[first[np.flatnonzero(n_pairs >= 2)]]
    nb = np.full((n_nodes, 2), -1, dtype=np.int64)
    count = np.zeros(n_nodes, dtype=np.int64)
    for a, b in feature:
        for u, v in ((a, b), (b, a)):
            if count[u] < 2:
                nb[u, count[u]] = v
            count[u] += 1
    nb[count != 2] = -1
    return nb


def _umbrellas(nodes: np.ndarray, faces: np.ndarray, members: np.ndarray):
    """Triangles of the original interface around each surface node: (n, t, 3, 3)."""
    corners = nodes[faces]
    centre = corners.mean(axis=1)
    tris = np.stack([np.stack([corners[:, i], corners[:, (i + 1) % 4], centre], axis=1)
                     for i in range(4)], axis=1)             # (F, 4, 3, 3)
    flat = faces.ravel()
    which = np.repeat(np.arange(len(faces)), 4)
    wanted = np.zeros(len(nodes), dtype=bool)
    wanted[members] = True
    sel = wanted[flat]
    flat, which = flat[sel], which[sel]
    order = np.argsort(flat, kind="stable")
    flat, which = flat[order], which[order]
    row = np.searchsorted(members, flat)
    counts = np.bincount(row, minlength=len(members))
    # a smooth surface puts at most 8 faces round a node; more is a pinch, which stays put
    width = max(min(int(counts.max()), MAX_UMBRELLA_FACES), 1)
    rank = np.arange(len(flat)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    face_of = np.full((len(members), width), -1, dtype=np.int64)
    fits = rank < width
    face_of[row[fits], rank[fits]] = which[fits]
    first = np.where(face_of[:, :1] >= 0, face_of[:, :1], 0)
    face_of = np.where(face_of >= 0, face_of, first)
    out = tris[face_of].reshape(len(members), width * 4, 3, 3)
    normal = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
    unit = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    # faces come from hexes on either side, so their windings disagree: align each face with
    # the node's first face before adding them up
    around = normal[face_of]                                    # (n, w, 3)
    sign = np.sign(np.einsum("nkj,nj->nk", around, around[:, 0]))
    sign[sign == 0] = 1.0
    node_normal = (around * sign[..., None]).sum(axis=1)
    node_normal /= np.maximum(np.linalg.norm(node_normal, axis=1, keepdims=True), 1e-30)
    # the sharpest bend between any face around the node and the node's mean normal
    bend = np.degrees(np.arccos(np.clip(
        np.abs(np.einsum("nkj,nj->nk", unit[face_of], node_normal)), 0.0, 1.0))).max(axis=1)
    bend[counts > MAX_UMBRELLA_FACES] = 180.0
    return out, node_normal, bend


def _closest_on_triangles(p: np.ndarray, tri: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Closest point to p (n, 3) on the union of triangles tri (n, t, 3, 3); rows are
    independent, so big calls run in row chunks on the shared thread pool (same result)."""
    from engine.mesh.voxel_hex import THREADS, pool

    if fast_kernels.HAVE_NUMBA:
        return fast_kernels.closest_on_triangles(p, tri)
    if len(p) <= 2 * chunk or THREADS < 2:
        return _closest_on_triangles_rows(p, tri)
    starts = range(0, len(p), chunk)
    return np.concatenate(list(pool().map(
        lambda i: _closest_on_triangles_rows(p[i:i + chunk], tri[i:i + chunk]), starts)))


def _closest_on_triangles_rows(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
    q = p[:, None, :]
    ab, ac, ap = b - a, c - a, q - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = q - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = q - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom
        w = vc / denom
    res = a + ab * v[..., None] + ac * w[..., None]
    # Voronoi regions of the vertices and edges (Ericson, Real-Time Collision Detection 5.1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    on_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    res = np.where(on_bc[..., None], b + (c - b) * t_bc[..., None], res)
    on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    res = np.where(on_ac[..., None], a + ac * t_ac[..., None], res)
    on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    res = np.where(on_ab[..., None], a + ab * t_ab[..., None], res)
    res = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, res)
    res = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, res)
    res = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, res)
    dist = ((res - q) ** 2).sum(-1)
    best = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    return res[np.arange(len(p)), best]


def _targets(nodes, hexes, inc, loc, rows):
    """Two step targets per node: the mean parallelepiped completion over its hexes, and the
    mean of its edge neighbours."""
    h, lc = inc[rows], loc[rows]
    valid = h >= 0
    hh, ll = np.maximum(h, 0), np.maximum(lc, 0)
    corner = hexes[hh]                                   # (n, k, 8)
    pick = lambda idx: nodes[np.take_along_axis(corner, idx, axis=2)]  # noqa: E731
    nb = _CORNER_NEIGHBOURS[ll]                          # (n, k, 3)
    op = _FACE_OPPOSITE[ll]
    a, b, c = (pick(nb[..., i:i + 1])[:, :, 0] for i in range(3))
    dab, dac, dbc = (pick(op[..., i:i + 1])[:, :, 0] for i in range(3))
    para = (2.0 * (a + b + c) - (dab + dac + dbc)) / 3.0
    lap = (a + b + c) / 3.0
    w = valid[..., None].astype(float)
    n = np.maximum(w.sum(axis=1), 1.0)
    return (para * w).sum(axis=1) / n, (lap * w).sum(axis=1) / n


def hex_sj(c: np.ndarray) -> np.ndarray:
    """The scaled Jacobian the deck reports (export.validate: min over the eight corners)."""
    if fast_kernels.HAVE_NUMBA:
        return fast_kernels.hex_scaled_jacobian(c)
    from engine.export.validate import _hex_scaled_jacobian

    return rows_in_parallel(_hex_scaled_jacobian, np.asarray(c, dtype=float))


PROTECTED = (0.9, 0.95, 0.99)
MIN_MEAN_GAIN = 1e-4     # a sweep that raises the mean scaled Jacobian by less ends the pass


def _crosses(before, after):
    """True per node where a move would push one of its elements below 0.9, 0.95 or 0.99 from
    above, so no element band ever loses members to make room for another."""
    out = np.zeros(len(before), dtype=bool)
    for level in PROTECTED:
        out |= np.any((before >= level) & (after < level), axis=1)
    return out


def _local_quality(nodes, hexes, inc, loc, rows, positions):
    """(min, mean) scaled Jacobian of each node's hexes with the node placed at `positions`."""
    if fast_kernels.HAVE_NUMBA:     # same values, without the (n, k, 8, 3) corner gather
        return fast_kernels.local_quality(nodes, hexes, inc, loc, rows, positions)
    h, lc = inc[rows], loc[rows]
    valid = h >= 0
    coords = nodes[hexes[np.maximum(h, 0)]]              # (n, k, 8, 3)
    n, k = h.shape
    ii = np.repeat(np.arange(n), k)
    jj = np.tile(np.arange(k), n)
    coords[ii, jj, np.maximum(lc, 0).ravel()] = np.repeat(positions, k, axis=0)
    sj = hex_sj(coords.reshape(n * k, 8, 3)).reshape(n, k)
    sj = np.where(np.isnan(sj), -1.0, sj)
    low = np.where(valid, sj, np.inf).min(axis=1)
    mean = np.where(valid, sj, 0.0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    return low, mean, np.where(valid, sj, np.nan)


def improve_hex_quality(nodes: np.ndarray, hexes: np.ndarray, part: np.ndarray,
                        colour: np.ndarray | None = None, *, target: float = 0.98,
                        sweeps: int = 60, min_moved: int = 0,
                        fixed: np.ndarray | None = None,
                        progress=None) -> tuple[np.ndarray, dict]:
    """Node moves that raise the scaled Jacobian and leave every interface where it is."""
    t0 = time.time()
    nodes = np.array(nodes, dtype=float, copy=True)
    hexes = np.asarray(hexes, dtype=np.int64)
    part = np.asarray(part)
    original = nodes.copy()
    inc, loc, _counts = _incidence(hexes, len(nodes))
    kind, _ = classify_nodes(hexes, part, len(nodes), incidence=(inc, loc, _counts))
    if fixed is not None:
        kind = np.where(np.asarray(fixed, dtype=bool), 2, kind)
    surface = np.flatnonzero(kind == 1)
    faces, sides = _interface_faces(hexes, part, with_sides=True)
    umbrella, surf_normal, bend = _umbrellas(original, faces, surface)
    # on a crease (faces bending more than CREASE_DEG around the node) sliding would move the
    # crease itself, so those nodes stay put
    crease = bend > CREASE_DEG
    kind[surface[crease]] = 2
    surface = surface[~crease]
    umbrella, surf_normal = umbrella[~crease], surf_normal[~crease]
    # where three regions meet along a curve, slide along that curve (kind 3)
    curve_nb = _feature_curves(faces, sides, len(nodes))
    on_curve = (kind == 2) & (curve_nb[:, 0] >= 0)
    u = original - original[np.maximum(curve_nb[:, 0], 0)]
    w = original[np.maximum(curve_nb[:, 1], 0)] - original
    cos_turn = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1),
                                           1e-30)
    on_curve &= cos_turn > np.cos(np.radians(CREASE_DEG))   # a corner of the curve stays put
    if fixed is not None:
        on_curve &= ~np.asarray(fixed, dtype=bool)
    kind[on_curve] = 3
    seg_a = original[np.maximum(curve_nb[:, 0], 0)]
    seg_b = original[np.maximum(curve_nb[:, 1], 0)]
    surf_row = np.full(len(nodes), -1, dtype=np.int64)
    surf_row[surface] = np.arange(len(surface))
    surf_nb, surf_tol = _interface_neighbours(original, faces, surface)
    if colour is None:
        colour = _greedy_colour(hexes, inc, len(nodes))
    colours = np.unique(colour)

    sj = hex_sj(nodes[hexes])
    before = _summary(sj)
    say = progress or (lambda m: logger.info("hex quality: %s", m))
    say(f"start: {before}; {int((kind == 0).sum())} free, {len(surface)} surface, "
        f"{int((kind == 3).sum())} curve, {int((kind == 2).sum())} fixed nodes")
    # a node that found no better place keeps failing until something around it moves, so
    # each sweep only revisits nodes whose neighbourhood changed (same result, less work)
    dirty = np.ones(len(nodes), dtype=bool)
    total = float(sj.sum())
    sweep = -1
    for sweep in range(sweeps):
        bad = sj < target
        movable = np.zeros(len(nodes), dtype=bool)
        movable[hexes[bad].ravel()] = True          # marking, no sort needed
        movable &= (kind != 2) & dirty
        dirty = np.zeros(len(nodes), dtype=bool)
        moved = 0
        for c in colours:
            rows = np.flatnonzero(movable & (colour == c))
            if not len(rows):
                continue
            # the node's current quality is the cached per-hex sj (kept current below)
            h0 = inc[rows]
            valid0 = h0 >= 0
            each0 = np.where(valid0, sj[np.maximum(h0, 0)], np.nan)
            low0 = np.where(valid0, each0, np.inf).min(axis=1)
            mean0 = np.where(valid0, each0, 0.0).sum(axis=1) / np.maximum(valid0.sum(axis=1), 1)
            score0 = low0 + SCORE_MEAN_WEIGHT * mean0
            kind_r = kind[rows]
            if fast_kernels.HAVE_NUMBA:     # free nodes get their targets in the kernel
                para = np.zeros((len(rows), 3))
                lap = np.zeros((len(rows), 3))
                rest = np.flatnonzero(kind_r != 0)
                if len(rest):
                    para[rest], lap[rest] = _targets(nodes, hexes, inc, loc, rows[rest])
            else:
                para, lap = _targets(nodes, hexes, inc, loc, rows)
            best_pos, best_score = nodes[rows].copy(), score0.copy()
            # backtracking line search: each node takes the first trial (full parallelepiped step,
            # halved, ..., then the Laplacian steps) that improves it; later trials are only
            # evaluated for the nodes still waiting
            open_ = np.ones(len(rows), dtype=bool)
            if fast_kernels.HAVE_NUMBA:
                # interior nodes in the compiled kernel (same trials, same acceptance rules)
                free = np.flatnonzero(kind_r == 0)
                if len(free):
                    pos, ok = fast_kernels.free_node_moves(
                        nodes, hexes, inc, loc, rows[free], sj, None, None,
                        SCALES, SCORE_MEAN_WEIGHT, PROTECTED, opposite=_FACE_OPPOSITE)
                    best_pos[free[ok]] = pos[ok]
                    best_score[free[ok]] = score0[free[ok]] + 1.0   # accepted: improves
                    open_[free] = False
            for goal in (para, lap):
                step_all = goal - nodes[rows]
                for scale in SCALES:
                    sub = np.flatnonzero(open_)
                    if not len(sub):
                        break
                    r = rows[sub]
                    step = step_all[sub]
                    is_surf = kind_r[sub] == 1
                    is_curve = kind_r[sub] == 3
                    srows = surf_row[r[is_surf]]
                    crow = r[is_curve]
                    if is_surf.any():
                        nrm = surf_normal[srows]
                        step[is_surf] -= (step[is_surf] * nrm).sum(1, keepdims=True) * nrm
                    if is_curve.any():
                        tan = seg_b[crow] - seg_a[crow]
                        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-30)
                        step[is_curve] = (step[is_curve] * tan).sum(1, keepdims=True) * tan
                    cand = nodes[r] + scale * step
                    if is_surf.any():
                        cand[is_surf] = _closest_on_triangles(cand[is_surf], umbrella[srows])
                    if is_curve.any():
                        cand[is_curve] = _closest_on_curve(cand[is_curve], seg_a[crow],
                                                           original[crow], seg_b[crow])
                    low, mean, each = _local_quality(nodes, hexes, inc, loc, r, cand)
                    score = low + SCORE_MEAN_WEIGHT * mean
                    take = ((low >= low0[sub] - 1e-12) & (score > score0[sub] + 1e-9)
                            & ~_crosses(each0[sub], each))
                    if is_surf.any():   # the faces between nodes must stay on the surface too
                        if fast_kernels.HAVE_NUMBA:
                            take[is_surf] &= fast_kernels.edges_on_surface(
                                cand[is_surf], nodes, surf_nb, srows, umbrella, surf_tol)
                        else:
                            take[is_surf] &= _edges_on_surface(
                                cand[is_surf], nodes, surf_nb[srows], umbrella[srows],
                                surf_tol[srows])
                    best_pos[sub[take]], best_score[sub[take]] = cand[take], score[take]
                    open_[sub[take]] = False
            take = best_score > score0 + 1e-9
            nodes[rows[take]] = best_pos[take]
            moved += int(take.sum())
            hit = inc[rows[take]].ravel()
            touched_mask = np.zeros(len(hexes), dtype=bool)
            touched_mask[hit[hit >= 0]] = True
            touched = np.flatnonzero(touched_mask)
            if len(touched):
                sj[touched] = (fast_kernels.hex_scaled_jacobian_of(nodes, hexes, touched)
                               if fast_kernels.HAVE_NUMBA else hex_sj(nodes[hexes[touched]]))
                dirty[hexes[touched].ravel()] = True
        say(f"sweep {sweep + 1}: {moved} nodes moved, min {sj.min():.4f}, mean {sj.mean():.4f}, "
            f">=0.9 {100 * (sj >= 0.9).mean():.2f}%, >={target:g} {100 * (sj >= target).mean():.2f}%")
        gain = float(sj.sum()) - total
        total = float(sj.sum())
        first_gain = gain if sweep == 0 else first_gain
        # stop when a sweep buys nothing measurable: under MIN_MEAN_GAIN on the mean scaled
        # Jacobian (every human run: sweeps 3-5 bought 1e-4 each for ~5 min each), or under 5%
        # of what the first sweep bought
        if (moved <= min_moved or gain < max(MIN_MEAN_GAIN * len(sj), 0.05 * first_gain)):
            break
    after = _summary(sj)
    shift = np.linalg.norm(nodes - original, axis=1)
    report = {"before": before, "after": after, "sweeps": sweep + 1,
              "nodes": {"free": int((kind == 0).sum()), "surface": int(len(surface)),
                        "curve": int((kind == 3).sum()), "fixed": int((kind == 2).sum())},
              "max_node_shift": float(shift.max()) if len(shift) else 0.0,
              "seconds": round(time.time() - t0, 1)}
    if len(surface):
        off = surface_deviation(original, nodes, faces, surface, umbrella)
        report["surface_node_off_original_surface_max"] = off
    say(f"done in {report['seconds']} s: {before['min']:.4f} -> {after['min']:.4f} min, "
        f"{before['mean']:.4f} -> {after['mean']:.4f} mean")
    return nodes, report


def _closest_on_curve(p, a, v, b):
    """Closest point to p on the two original segments a-v and v-b of a junction curve."""
    best = None
    for s, e in ((a, v), (v, b)):
        d = e - s
        t = np.clip(((p - s) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0.0, 1.0)
        q = s + t[:, None] * d
        dist = ((q - p) ** 2).sum(1)
        best = (q, dist) if best is None else (np.where((dist < best[1])[:, None], q, best[0]),
                                               np.minimum(dist, best[1]))
    return best[0]


FIDELITY_FRACTION = 0.02


def _interface_neighbours(original, faces, surface):
    """Per surface node: its neighbours along interface faces (padded with -1, at most
    MAX_UMBRELLA_FACES) and its tolerance, FIDELITY_FRACTION of its mean interface edge."""
    edges = np.concatenate([faces[:, [i, (i + 1) % 4]] for i in range(4)])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    both = np.concatenate([edges, edges[:, ::-1]])
    row = np.full(original.shape[0], -1, dtype=np.int64)
    row[surface] = np.arange(len(surface))
    both = both[row[both[:, 0]] >= 0]
    both = both[np.argsort(both[:, 0], kind="stable")]
    r = row[both[:, 0]]
    counts = np.bincount(r, minlength=len(surface))
    rank = np.arange(len(both)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    width = max(min(int(counts.max()) if len(counts) else 1, MAX_UMBRELLA_FACES), 1)
    nb = np.full((len(surface), width), -1, dtype=np.int64)
    keep = rank < width
    nb[r[keep], rank[keep]] = both[keep, 1]
    length = np.linalg.norm(original[both[:, 1]] - original[both[:, 0]], axis=1)
    mean_edge = np.bincount(r, weights=length, minlength=len(surface)) / np.maximum(counts, 1)
    return nb, FIDELITY_FRACTION * mean_edge


def _edges_on_surface(cand, nodes, nb, umbrella, tol):
    """True where every edge midpoint between the moved node and its interface neighbours
    stays within tol of the original interface."""
    ok = np.ones(len(cand), dtype=bool)
    for j in range(nb.shape[1]):
        has = nb[:, j] >= 0
        if not has.any():
            continue
        mid = 0.5 * (cand[has] + nodes[nb[has, j]])
        dev = np.linalg.norm(_closest_on_triangles(mid, umbrella[has]) - mid, axis=1)
        ok[has] &= dev <= tol[has]
    return ok


def surface_deviation(original, nodes, faces, surface, umbrella) -> float:
    """Largest distance of a moved surface node from the original interface (0 by construction
    up to round-off)."""
    p = nodes[surface]
    q = _closest_on_triangles(p, umbrella)
    return float(np.linalg.norm(p - q, axis=1).max())


def _summary(sj: np.ndarray) -> dict:
    return {"min": float(sj.min()), "mean": float(sj.mean()),
            "p01": float(np.percentile(sj, 1)),
            "pct_above_0p9": float(100 * (sj >= 0.9).mean()),
            "pct_above_0p95": float(100 * (sj >= 0.95).mean()),
            "pct_above_0p98": float(100 * (sj >= 0.98).mean()),
            "n_negative": int((sj <= 0).sum())}


def _greedy_colour(hexes, inc, n_nodes) -> np.ndarray:
    """Colours with no two nodes of one hex sharing a colour (when no lattice parity exists)."""
    colour = np.full(n_nodes, -1, dtype=np.int64)
    for v in range(n_nodes):
        h = inc[v][inc[v] >= 0]
        used = set(colour[hexes[h]].ravel().tolist())
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    return colour
