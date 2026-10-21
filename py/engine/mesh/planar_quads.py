"""All-quad mesh of a planar polygon: a square core plus one boundary layer, then smoothing.

Interior cells are squares of the target size. One layer of quads joins the stepped edge of
the core to the true outline, with a node on every sharp corner. Smoothing moves interior
nodes toward their neighbours and slides boundary nodes along the outline, never accepting a
move that makes a cell worse once it is below the quality floor.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as _MplPath
from scipy import ndimage

_CORNER_DEG = 140.0  # outline corners sharper than this get their own node
_CORE_CLEARANCE = 0.45  # core cell corners stay this many cells inside the outline


def quad_sj(points: np.ndarray) -> np.ndarray:
    """Scaled Jacobian of planar quads (K, 4, 2), worst corner, positive for CCW quads."""
    nxt = np.roll(points, -1, axis=1) - points
    prv = np.roll(points, 1, axis=1) - points
    cross = nxt[..., 0] * prv[..., 1] - nxt[..., 1] * prv[..., 0]
    norm = np.linalg.norm(nxt, axis=2) * np.linalg.norm(prv, axis=2)
    return (cross / np.maximum(norm, 1e-12)).min(axis=1)


def _first_best(values: np.ndarray, here: float):
    """Index the sequential scan `if value > best + 1e-6: best = value` ends on (None: no
    candidate beats `here` by 1e-6)."""
    best, pick = here, None
    for i in np.flatnonzero(values > here + 1e-6):
        if values[i] > best + 1e-6:
            best, pick = values[i], int(i)
    return pick


def quad_sj3(points: np.ndarray) -> np.ndarray:
    """Scaled Jacobian of 3D quads (K, 4, 3) about each quad's mean normal, worst corner."""
    nxt = np.roll(points, -1, axis=1) - points
    prv = np.roll(points, 1, axis=1) - points
    cross = np.cross(nxt, prv)
    normal = cross.sum(axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    signed = np.einsum("kcd,kd->kc", cross, normal)
    norm = np.linalg.norm(nxt, axis=2) * np.linalg.norm(prv, axis=2)
    return (signed / np.maximum(norm, 1e-12)).min(axis=1)


def improve_lifted(nodes: np.ndarray, quads: np.ndarray, polygon: np.ndarray, on_outline: np.ndarray,
                   lift, h: float, rounds: int = 80, target: float = 0.7) -> np.ndarray:
    """Local search on a planar mesh judged by its lifted 3D quads: `lift(xy) -> xyz`.
    Interior nodes stay inside the outline, outline nodes slide along it."""
    outline = _Outline(_resample_closed(np.asarray(polygon, dtype=float), h / 4.0))
    inside = _MplPath(outline.points)
    nodes = np.array(nodes, dtype=float)
    corner = np.zeros(len(nodes), bool)
    incident = [[] for _ in range(len(nodes))]
    for k, q in enumerate(quads):
        for v in q:
            incident[v].append(k)
    rng = np.random.default_rng(1)
    xyz = lift(nodes)
    sign = 1.0 if np.median(quad_sj3(xyz[quads])) >= 0 else -1.0
    for _ in range(rounds):
        sj = sign * quad_sj3(xyz[quads])
        bad = np.where(sj < target)[0]
        if not len(bad):
            break
        improved = False
        for v in np.unique(quads[bad].ravel()):
            if corner[v]:
                continue
            inc = np.array(incident[v])
            here = (sign * quad_sj3(xyz[quads[inc]])).min()
            span = min(np.linalg.norm(nodes[quads[inc]] - nodes[v], axis=2).max() * 0.4, 1.5 * h)
            cand = nodes[v] + rng.uniform(-span, span, size=(48, 2))
            cand = np.vstack([cand, nodes[quads[inc]].reshape(-1, 2).mean(0)])
            if on_outline[v]:
                cand = outline.at(outline.project(cand))
            else:
                cand = cand[inside.contains_points(cand)]
            if not len(cand):
                continue
            lifted = lift(cand)
            base = xyz[quads[inc]]
            trials = np.repeat(base[None], len(cand), axis=0)
            trials[:, quads[inc] == v] = lifted[:, None, :]
            qs = (sign * quad_sj3(trials.reshape(-1, 4, 3))).reshape(len(cand), -1).min(axis=1)
            pick = _first_best(qs, here)
            if pick is not None:
                nodes[v], xyz[v] = cand[pick], lifted[pick]
                improved = True
        if not improved:
            break
    return nodes


def _resample_closed(poly: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = seg > 1e-9
    closed = np.vstack([closed[:-1][keep], closed[:1]])
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = max(int(np.ceil(d[-1] / step)), 8)
    t = np.linspace(0.0, d[-1], n, endpoint=False)
    return np.stack([np.interp(t, d, closed[:, k]) for k in range(2)], axis=1)


_ROUND_DEG = 75.0  # outline corners sharper than this are rounded over about one element


def _corner_angles(poly: np.ndarray, step: int = 3) -> np.ndarray:
    prev = np.roll(poly, step, axis=0) - poly
    nextp = np.roll(poly, -step, axis=0) - poly
    cosang = (prev * nextp).sum(1) / np.maximum(np.linalg.norm(prev, axis=1) * np.linalg.norm(nextp, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def _round_sharp(poly: np.ndarray, h: float) -> np.ndarray:
    """Replace each corner sharper than _ROUND_DEG by a quadratic arc spanning ~0.8 element per
    side, so the single quad at that corner keeps a usable angle."""
    spacing = float(np.median(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
    reach = max(int(round(0.8 * h / max(spacing, 1e-9))), 2)
    angle = _corner_angles(poly)
    sharp = np.where(angle < _ROUND_DEG)[0]
    if not len(sharp):
        return poly
    out = poly.copy()
    for group in np.split(sharp, np.where(np.diff(sharp) > 2 * reach)[0] + 1):
        k = int(group[np.argmin(angle[group])])
        a, c = out[(k - reach) % len(out)], out[(k + reach) % len(out)]
        s = np.linspace(0.0, 1.0, 2 * reach + 1)[:, None]
        arc = (1 - s) ** 2 * a + 2 * s * (1 - s) * out[k] + s * s * c
        idx = (k - reach + np.arange(2 * reach + 1)) % len(out)
        out[idx] = arc
    return out


class _Outline:
    """Closed polyline with arc-length parametrisation."""

    def __init__(self, poly: np.ndarray):
        self.points = np.asarray(poly, dtype=float)
        closed = np.vstack([self.points, self.points[:1]])
        self.seg = np.diff(closed, axis=0)
        self.cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(self.seg, axis=1))])
        self.length = float(self.cum[-1])

    def at(self, t: np.ndarray) -> np.ndarray:
        t = np.mod(t, self.length)
        k = np.clip(np.searchsorted(self.cum, t, side="right") - 1, 0, len(self.seg) - 1)
        f = (t - self.cum[k]) / np.maximum(np.linalg.norm(self.seg[k], axis=1), 1e-12)
        return self.points[k] + f[:, None] * self.seg[k]

    def _nearest(self, pts: np.ndarray):
        """Nearest outline point: segment index, fraction along it, squared distance."""
        from scipy.spatial import cKDTree

        if not hasattr(self, "_tree"):
            self._tree = cKDTree(self.points)
        n = len(self.points)
        _d, near = self._tree.query(pts, k=min(4, n), workers=-1 if len(pts) > 2000 else 1)
        near = np.atleast_2d(near)
        segs = np.concatenate([near, (near - 1) % n], axis=1)  # segments starting at or ending at each
        a = self.points[segs]
        ab = self.seg[segs]
        rel = pts[:, None, :] - a
        f = np.clip((rel * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        d2 = ((rel - f[..., None] * ab) ** 2).sum(-1)
        best = np.argmin(d2, axis=1)
        rows = np.arange(len(pts))
        return segs[rows, best], f[rows, best], d2[rows, best]

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Arc-length parameter of the nearest outline point."""
        k, f, _d2 = self._nearest(np.asarray(pts, dtype=float))
        return self.cum[k] + f * np.linalg.norm(self.seg[k], axis=1)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return np.sqrt(self._nearest(np.asarray(pts, dtype=float))[2])


def _core_cells(outline: _Outline, h: float):
    lo = outline.points.min(axis=0) - h
    hi = outline.points.max(axis=0) + h
    nx, ny = (np.ceil((hi - lo) / h).astype(int) + 1)
    gx, gy = np.meshgrid(lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny), indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = _MplPath(outline.points).contains_points(nodes)
    clear = np.zeros(len(nodes), bool)
    clear[inside] = outline.distance(nodes[inside]) >= _CORE_CLEARANCE * h
    ok = clear.reshape(nx, ny)
    cells = ok[:-1, :-1] & ok[1:, :-1] & ok[1:, 1:] & ok[:-1, 1:]
    return cells, lo, (nx, ny)


def _make_manifold(cells: np.ndarray) -> np.ndarray:
    """Largest 4-connected piece, holes filled, no two cells meeting only at a corner."""
    cells = cells.copy()
    for _ in range(200):
        lab, n = ndimage.label(cells)
        if n == 0:
            return cells
        cells = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
        cells = ndimage.binary_fill_holes(cells)
        a = cells[:-1, :-1]
        b = cells[1:, :-1]
        c = cells[:-1, 1:]
        d = cells[1:, 1:]
        diag1 = a & d & ~b & ~c
        diag2 = b & c & ~a & ~d
        if not (diag1.any() or diag2.any()):
            # also no node whose cells form a pinched fan (handled by the diagonal test on a grid)
            return cells
        i, j = np.nonzero(diag1)
        cells[i, j] = False  # drop one of the two cells meeting at a corner
        i, j = np.nonzero(diag2)
        cells[i + 1, j] = False
    return cells


def _loop_of(cells: np.ndarray) -> list[tuple[int, int]]:
    """Ordered boundary node loop (grid indices) of a manifold cell set, counter-clockwise."""
    edges = {}
    nx, ny = cells.shape
    for i, j in zip(*np.nonzero(cells)):
        quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]  # CCW
        for k in range(4):
            a, b = quad[k], quad[(k + 1) % 4]
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = True
    nxt = {a: b for a, b in edges}
    start = min(nxt)
    loop = [start]
    while True:
        b = nxt[loop[-1]]
        if b == start:
            break
        loop.append(b)
    return loop


def pave(polygon: np.ndarray, h: float, smooth_passes: int = 200, floor: float = 0.8):
    """Quad mesh of a closed planar polygon (N, 2), CCW or CW. Returns nodes (M, 2), quads (K, 4)
    counter-clockwise, and a mask of the nodes that lie on the outline."""
    poly = np.asarray(polygon, dtype=float)
    if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
        poly = poly[::-1]
    poly = _round_sharp(_resample_closed(poly, h / 4.0), h)
    outline = _Outline(poly)

    cells, lo, (nx, ny) = _core_cells(outline, h)
    cells = _make_manifold(cells)
    if not cells.any():
        raise ValueError("polygon too thin for a core at this element size")
    node_id = -np.ones((nx, ny), dtype=np.int64)
    used = np.zeros((nx, ny), bool)
    ci, cj = np.nonzero(cells)
    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
        used[ci + di, cj + dj] = True
    node_id[used] = np.arange(used.sum())
    gi, gj = np.nonzero(used)
    nodes = [np.stack([lo[0] + h * gi, lo[1] + h * gj], axis=1)]
    quads = [np.stack([node_id[ci, cj], node_id[ci + 1, cj], node_id[ci + 1, cj + 1],
                       node_id[ci, cj + 1]], axis=1)]
    n_core = int(used.sum())

    loop = _loop_of(cells)
    inner = np.array([node_id[a] for a in loop])
    inner_xy = nodes[0][inner]
    m = len(inner)

    # outline parameter for each layer node: nearest point, made monotone, spread evenly
    t = outline.project(inner_xy)
    d = np.diff(np.r_[t, t[0] + outline.length])
    d = (d + outline.length / 2) % outline.length - outline.length / 2
    d = np.maximum(d, 0.05 * outline.length / m)  # never backwards along the outline
    d *= outline.length / d.sum()
    t = t[0] + np.r_[0.0, np.cumsum(d[:-1])]
    # even out the spacing locally only, so each node stays near its nearest outline point
    for _ in range(20):
        t_wrapped = np.r_[t[-1] - outline.length, t, t[0] + outline.length]
        t = 0.5 * t + 0.25 * (t_wrapped[:-2] + t_wrapped[2:])
    # a node on every sharp outline corner
    prev = np.roll(poly, 1, axis=0) - poly
    nextp = np.roll(poly, -1, axis=0) - poly
    for step in (3,):
        prev = np.roll(poly, step, axis=0) - poly
        nextp = np.roll(poly, -step, axis=0) - poly
    cosang = (prev * nextp).sum(1) / np.maximum(np.linalg.norm(prev, axis=1) * np.linalg.norm(nextp, axis=1), 1e-12)
    angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    corner_pts = np.where(angle < _CORNER_DEG)[0]
    corner_t = []
    if len(corner_pts):
        groups = np.split(corner_pts, np.where(np.diff(corner_pts) > 3)[0] + 1)
        for g in groups:
            k = g[np.argmin(angle[g])]
            corner_t.append(outline.cum[k])
    pinned = np.zeros(m, bool)
    tm = np.mod(t, outline.length)
    for tc in corner_t:
        gap = np.abs((tm - tc + outline.length / 2) % outline.length - outline.length / 2)
        k = int(np.argmin(np.where(pinned, np.inf, gap)))
        shift = (tc - tm[k] + outline.length / 2) % outline.length - outline.length / 2
        t[k] += shift
        pinned[k] = True
    outer_xy = outline.at(t)
    outer = n_core + np.arange(m)
    nodes.append(outer_xy)
    layer = np.stack([inner, outer, np.roll(outer, -1), np.roll(inner, -1)], axis=1)
    nodes = np.vstack(nodes)
    quads = np.vstack(quads + [layer])
    # orientation: make every quad CCW
    if np.median(quad_sj(nodes[quads])) < 0:
        quads = quads[:, ::-1]

    on_outline = np.zeros(len(nodes), bool)
    on_outline[outer] = True
    fixed = np.zeros(len(nodes), bool)
    fixed[outer[pinned]] = True
    nodes = smooth(nodes, quads, outline, outer, fixed, passes=smooth_passes, floor=floor, h=h)
    return nodes, quads, on_outline


def smooth(nodes, quads, outline: _Outline, outer, fixed, passes=200, step=0.5, floor=0.8, h=1.0):
    from scipy.sparse import coo_matrix

    nodes = np.array(nodes, dtype=float)
    n = len(nodes)
    edges = np.unique(np.sort(np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]],
                                              quads[:, [2, 3]], quads[:, [3, 0]]]), axis=1), axis=0)
    adj = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]],
                                               np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    is_outer = np.zeros(n, bool)
    is_outer[outer] = True
    # boundary nodes average only their two outline neighbours (keeps spacing even)
    m = len(outer)
    ring_prev, ring_next = np.roll(outer, 1), np.roll(outer, -1)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    for _ in range(passes):
        sj = quad_sj(nodes[quads])
        target = (adj @ nodes) / np.maximum(deg, 1)[:, None]
        target[outer] = 0.5 * (nodes[ring_prev] + nodes[ring_next])
        moving = ~fixed.copy()
        for _guard in range(10):
            cand = nodes.copy()
            cand[moving] += step * (target[moving] - nodes[moving])
            mv = moving & is_outer
            if mv.any():
                cand[mv] = outline.at(outline.project(cand[mv]))
            new = quad_sj(cand[quads])
            worse = (new < sj - 1e-9) & (new < floor)
            if not worse.any():
                break
            moving[quads[worse].ravel()] = False
        nodes = cand if moving.any() else nodes
        if not moving.any():
            break
    return _optimise_worst(nodes, quads, outline, is_outer, fixed, h=h)


def _optimise_worst(nodes, quads, outline, is_outer, fixed, rounds=30, target=0.6, h=1.0):
    """Local search on nodes of the worst quads: try positions around each, keep the best.
    Interior nodes stay inside the outline; boundary nodes stay on it."""
    inside = _MplPath(outline.points)
    nodes = nodes.copy()
    incident = [[] for _ in range(len(nodes))]
    for k, q in enumerate(quads):
        for v in q:
            incident[v].append(k)
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        sj = quad_sj(nodes[quads])
        bad = np.where(sj < target)[0]
        if not len(bad):
            break
        improved = False
        for v in np.unique(quads[bad].ravel()):
            if fixed[v]:
                continue
            inc = np.array(incident[v])
            here = quad_sj(nodes[quads[inc]]).min()
            span = min(np.linalg.norm(nodes[quads[inc]] - nodes[v], axis=2).max() * 0.35, 1.5 * h)
            cand = nodes[v] + rng.uniform(-span, span, size=(40, 2))
            cand = np.vstack([cand, nodes[quads[inc]].reshape(-1, 2).mean(0)])
            if is_outer[v]:
                cand = outline.at(outline.project(cand))
            else:
                cand = cand[inside.contains_points(cand)]
            if not len(cand):
                continue
            base = nodes[quads[inc]]
            trials = np.repeat(base[None], len(cand), axis=0)
            trials[:, quads[inc] == v] = cand[:, None, :]
            qs = quad_sj(trials.reshape(-1, 4, 2)).reshape(len(cand), -1).min(axis=1)
            pick = _first_best(qs, here)
            if pick is not None:
                nodes[v] = cand[pick]
                improved = True
        if not improved:
            break
    return nodes
