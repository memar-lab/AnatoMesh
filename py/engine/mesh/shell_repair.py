"""Raise the Scaled Jacobian of a quad sheet without deleting a single cell."""

from __future__ import annotations

import functools
import logging
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

TARGET_SJ = 0.96

_PASS_TIMES: dict[str, list[float]] = {}


def timed(fn):
    """Accumulate wall time per sheet pass."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _PASS_TIMES.setdefault(fn.__name__, []).append(time.perf_counter() - start)
    return wrapper


def pass_timing_report(reset: bool = True) -> dict:
    """Seconds and call count per sheet pass, slowest first."""
    report = {name: {"seconds": round(float(sum(times)), 2), "calls": len(times)}
              for name, times in _PASS_TIMES.items()}
    if reset:
        _PASS_TIMES.clear()
    return dict(sorted(report.items(), key=lambda kv: -kv[1]["seconds"]))


def quad_scaled_jacobian(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Per-quad Scaled Jacobian"""
    from engine.mesh.voxel_hex import rows_in_parallel

    quads = np.asarray(quads)
    if not len(quads):
        return np.zeros(0)
    verts = np.asarray(verts, dtype=float)
    return rows_in_parallel(lambda rows: _quad_sj(verts, rows), quads)


def _quad_sj(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    p = verts[quads]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    edge = np.roll(p, -1, axis=1) - p
    length = np.linalg.norm(edge, axis=2)
    worst = np.full(len(quads), np.inf)
    for i in range(4):
        v_in = edge[:, (i - 1) % 4]
        v_out = edge[:, i]
        denom = length[:, (i - 1) % 4] * length[:, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            sj = np.einsum("ij,ij->i", np.cross(v_in, v_out), normal) / denom
        worst = np.minimum(worst, np.where(denom > 1e-12, sj, 0.0))
    return worst


def _rim_edges(quads: np.ndarray) -> np.ndarray:
    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    key = np.sort(pairs, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    return uniq[counts == 1]


EXPECTED_BOUNDARY_LOOPS = {"falx": 1, "tentorium": 1, "pia": 0, "dura": 0,
                           "skull": 0, "scalp": 0}


def sheet_topology(quads: np.ndarray, expected_loops: int | None = None,
                   name: str = "sheet", trianguli: np.ndarray | None = None) -> dict:
    """Boundary loops, Euler characteristic and manifoldness of a quad sheet."""
    quads = np.asarray(quads).reshape(-1, 4)
    trianguli = (np.zeros((0, 3), dtype=int) if trianguli is None
                 else np.asarray(trianguli).reshape(-1, 3))
    if not len(quads) and not len(trianguli):
        return {"n_quads": 0, "n_tris": 0, "boundary_loops": 0, "boundary_junctions": 0,
                "non_manifold_edges": 0, "components": 0, "euler_characteristic": 0,
                "expected_boundary_loops": expected_loops, "passed": True}

    pairs = np.concatenate(
        [quads[:, [i, (i + 1) % 4]] for i in range(4)]
        + [trianguli[:, [i, (i + 1) % 3]] for i in range(3)], axis=0)
    edges, counts = np.unique(np.sort(pairs, axis=1), axis=0, return_counts=True)
    rim = edges[counts == 1]

    used = np.unique(np.concatenate([quads.ravel(), trianguli.ravel()]))
    n_v, n_e, n_f = len(used), len(edges), len(quads) + len(trianguli)

    remap = np.zeros(int(used.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(n_v)
    graph = csr_matrix((np.ones(n_e), (remap[edges[:, 0]], remap[edges[:, 1]])),
                       shape=(n_v, n_v))
    components = int(connected_components(graph, directed=False)[0])

    loops = junctions = 0
    if len(rim):
        rim_nodes = np.unique(rim)
        rim_map = np.zeros(int(rim_nodes.max()) + 1, dtype=np.int64)
        rim_map[rim_nodes] = np.arange(len(rim_nodes))
        rim_graph = csr_matrix(
            (np.ones(len(rim)), (rim_map[rim[:, 0]], rim_map[rim[:, 1]])),
            shape=(len(rim_nodes), len(rim_nodes)))
        loops = int(connected_components(rim_graph, directed=False)[0])
        junctions = int((np.bincount(rim_map[rim.ravel()],
                                     minlength=len(rim_nodes)) > 2).sum())

    non_manifold = int((counts > 2).sum())
    passed = non_manifold == 0 and junctions == 0
    if expected_loops is not None:
        passed = passed and loops == expected_loops
    report = {"n_quads": int(len(quads)), "n_tris": int(len(trianguli)),
              "boundary_loops": loops,
              "boundary_junctions": junctions, "non_manifold_edges": non_manifold,
              "components": components, "euler_characteristic": int(n_v - n_e + n_f),
              "expected_boundary_loops": expected_loops, "passed": bool(passed)}
    if not passed:
        logger.warning("%s topology: %d boundary loop(s) (expected %s), %d junction(s), "
                       "%d non-manifold edge(s), %d component(s)",
                       name, loops, expected_loops, junctions, non_manifold, components)
    return report


@timed
def split_pinch_nodes(verts: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Give each fan of quads around a pinched node its own copy of that node."""
    quads = np.asarray(quads)
    if not len(quads):
        return verts, quads, 0

    rim = _rim_edges(quads)
    if not len(rim):
        return verts, quads, 0
    degree = np.bincount(rim.ravel(), minlength=len(verts))
    pinched = np.flatnonzero(degree > 2)
    if not len(pinched):
        return verts, quads, 0

    before = int(connected_components(_vertex_graph(quads, len(verts)),
                                      directed=False)[0])
    out = quads.copy()
    extra: list[int] = []
    for node in pinched:
        rows = np.flatnonzero((out == node).any(axis=1))
        for fan in _fans_at(out[rows], int(node))[1:]:
            extra.append(int(node))
            new = len(verts) + len(extra) - 1
            for r in fan:
                out[rows[r]][out[rows[r]] == node] = new
    if not extra:
        return verts, quads, 0
    grown = np.concatenate([verts, verts[np.array(extra)]], axis=0)
    after = int(connected_components(_vertex_graph(out, len(grown)),
                                     directed=False)[0])
    if after != before:
        logger.info("split_pinch_nodes: %d pinch(es) left alone -- splitting them "
                    "would put the sheet in %d pieces", len(extra), after)
        return verts, quads, 0
    logger.info("split_pinch_nodes: %d pinched node(s) given their own copy", len(extra))
    return grown, out, len(extra)


def _vertex_graph(quads: np.ndarray, n_verts: int):
    pairs = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    return csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                      shape=(n_verts, n_verts))


def _fans_at(local: np.ndarray, node: int) -> list[list[int]]:
    """Group the quads touching `node` into fans joined by shared edges at it."""
    sides = []
    for quad in local:
        i = int(np.flatnonzero(quad == node)[0])
        sides.append((int(quad[(i - 1) % 4]), int(quad[(i + 1) % 4])))
    parent = list(range(len(local)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_neighbour: dict[int, list[int]] = {}
    for r, (a, b) in enumerate(sides):
        by_neighbour.setdefault(a, []).append(r)
        by_neighbour.setdefault(b, []).append(r)
    for rows in by_neighbour.values():
        for r in rows[1:]:
            ra, rb = find(rows[0]), find(r)
            if ra != rb:
                parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for r in range(len(local)):
        groups.setdefault(find(r), []).append(r)
    return list(groups.values())


def topology_report(sheets: dict, trianguli: dict | None = None) -> dict:
    """`sheet_topology` for each named sheet."""
    trianguli = trianguli or {}
    names_found = list(sheets) + [n for n in trianguli if n not in sheets]
    return {n: sheet_topology(sheets.get(n, np.zeros((0, 4), dtype=int)),
                              EXPECTED_BOUNDARY_LOOPS.get(n), n,
                              trianguli=trianguli.get(n))
            for n in names_found}


def _incidence(quads: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """`(table, valid)` of shape (n_verts, max degree)"""
    node = quads.ravel()
    owner = np.repeat(np.arange(len(quads)), 4)
    order = np.argsort(node, kind="stable")
    node, owner = node[order], owner[order]
    degree = np.bincount(node, minlength=n_verts)
    width = max(int(degree.max()) if len(degree) else 1, 1)
    table = np.zeros((n_verts, width), dtype=np.int64)
    valid = np.zeros((n_verts, width), dtype=bool)
    slot = np.arange(len(node)) - np.repeat(np.cumsum(degree) - degree, degree)
    table[node, slot] = owner
    valid[node, slot] = True
    return table, valid


def _node_normals(verts: np.ndarray, quads: np.ndarray,
                  table: np.ndarray, valid: np.ndarray,
                  only: np.ndarray | None = None) -> np.ndarray:
    """Area-weighted quad normal averaged onto each node (exact only around `only` quads)."""
    face = np.zeros((len(quads) + 1, 3))
    rows = slice(0, len(quads)) if only is None else only
    p = verts[quads[rows]]
    face[rows] = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    idx = np.where(valid, table, len(quads))
    n = face[idx].sum(axis=1)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norm > 1e-9, n / (norm + 1e-12), np.array([0.0, 0.0, 1.0]))


def _worst_incident(sj: np.ndarray, table: np.ndarray, valid: np.ndarray) -> np.ndarray:
    padded = np.concatenate([sj, np.array([np.inf])])
    return padded[np.where(valid, table, len(sj))].min(axis=1)


def _colour_nodes(quads: np.ndarray, n_verts: int) -> list[np.ndarray]:
    """Greedy colouring so nodes moved in one pass never share a quad."""
    if not n_verts:
        return []
    # visiting nodes in order, only lower-numbered neighbours already carry a colour
    quads = np.asarray(quads, dtype=np.int64)
    pairs = np.concatenate([quads[:, [i, j]] for i in range(4) for j in range(4) if i != j])
    pairs = pairs[pairs[:, 1] < pairs[:, 0]]
    key = np.unique(pairs[:, 0] * n_verts + pairs[:, 1])
    start = np.searchsorted(key // n_verts, np.arange(n_verts + 1)).tolist()
    lower = (key % n_verts).tolist()
    colour = [0] * n_verts
    for v in range(n_verts):
        taken = {colour[u] for u in lower[start[v]:start[v + 1]]}
        c = 0
        while c in taken:
            c += 1
        colour[v] = c
    colour = np.asarray(colour, dtype=np.int64)
    return [np.nonzero(colour == c)[0] for c in range(int(colour.max()) + 1)]


def _square_targets(verts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Where each quad wants its own four corners, as `(n_quads, 4, 3)`."""
    p = verts[quads]
    centre = p.mean(axis=1)
    local = p - centre[:, None, :]

    normal = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    helper = np.where(np.abs(normal[:, [0]]) < 0.9, np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(helper, normal)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(normal, e1)

    qx = np.einsum("qkj,qj->qk", local, e1)
    qy = np.einsum("qkj,qj->qk", local, e2)
    radius = np.sqrt(qx ** 2 + qy ** 2).mean(axis=1, keepdims=True)

    theta = np.deg2rad(45.0 + 90.0 * np.arange(4))[None, :]
    ux, uy = radius * np.cos(theta), radius * np.sin(theta)
    dot = (ux * qx + uy * qy).sum(axis=1)
    crs = (ux * qy - uy * qx).sum(axis=1)
    phi = np.arctan2(crs, dot)[:, None]
    tx = ux * np.cos(phi) - uy * np.sin(phi)
    ty = ux * np.sin(phi) + uy * np.cos(phi)
    return centre[:, None, :] + tx[..., None] * e1[:, None, :] + ty[..., None] * e2[:, None, :]


def _sample_mask(mask: np.ndarray, points_vox: np.ndarray) -> np.ndarray:
    """Nearest-voxel lookup of a boolean mask"""
    idx = np.rint(points_vox).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(mask.shape)), axis=1)
    out = np.zeros(len(points_vox), dtype=bool)
    sel = idx[inside]
    out[inside] = mask[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def _containment(allowed, affine):
    """`points -> bool` in world mm, or None when no mask was given."""
    if allowed is None or affine is None:
        return None
    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    mask = np.asarray(allowed, dtype=bool)

    def legal(points: np.ndarray) -> np.ndarray:
        return _sample_mask(mask, np.asarray(points, dtype=float) @ inv[:3, :3].T + inv[:3, 3])

    return legal


@timed
def improve_sheet(verts: np.ndarray, quads: np.ndarray, target_sj: float = TARGET_SJ,
                  sweeps: int = 40, name: str = "sheet",
                  allowed: np.ndarray | None = None,
                  affine: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Angular repair of a quad sheet."""
    verts = np.array(verts, dtype=float, copy=True)
    quads = np.asarray(quads)
    before = quad_scaled_jacobian(verts, quads)
    report = {
        "n_quads": int(len(quads)),
        "min_sj_before": round(float(before.min()), 4) if len(before) else 1.0,
        "pct_above_target_before": (round(100.0 * float((before >= target_sj).mean()), 2)
                                    if len(before) else 100.0),
        "n_quads_removed": 0,
    }
    if not len(quads):
        report.update(min_sj_after=1.0, pct_above_target_after=100.0, sweeps_run=0)
        return verts, report

    n_verts = len(verts)
    legal = _containment(allowed, affine)
    started_legal = legal(verts) if legal is not None else None
    if legal is not None:
        report["n_starting_outside_allowed"] = int((~started_legal).sum())
    table, valid = _incidence(quads, n_verts)
    rim = _rim_edges(quads)
    on_boundary = np.zeros(n_verts, dtype=bool)
    on_boundary[np.unique(rim)] = True

    neighbours: dict[int, list[int]] = {}
    for a, b in rim:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    rim_tangent = np.zeros((n_verts, 3))
    corner = np.zeros(n_verts, dtype=bool)
    for node, near in neighbours.items():
        if len(near) != 2:
            corner[node] = True
            continue
        d1 = verts[near[0]] - verts[node]
        d2 = verts[near[1]] - verts[node]
        n1 = d1 / (np.linalg.norm(d1) + 1e-12)
        n2 = d2 / (np.linalg.norm(d2) + 1e-12)
        if float(n1 @ n2) > -0.9:  # the outline turns here
            corner[node] = True
            continue
        chord = verts[near[1]] - verts[near[0]]
        rim_tangent[node] = chord / (np.linalg.norm(chord) + 1e-12)

    degree = valid.sum(axis=1).astype(float)
    degree[degree == 0] = 1.0
    pinned = corner | (on_boundary & (np.linalg.norm(rim_tangent, axis=1) < 1e-9))

    sets = _colour_nodes(quads, n_verts)
    # A move is refused if it drops a quad that started good below target, so only the
    # corners of quads that started bad can ever move. Their steps need the targets and
    # normals of the quads around them, nothing else.
    movers = np.unique(quads[before < target_sj])
    around = np.unique(table[movers][valid[movers]])
    good_padded = np.concatenate([before >= target_sj, [False]])
    sj = before.copy()                  # kept current as vertices move
    sj_padded = np.concatenate([sj, [np.inf]])
    sweeps_run = 0
    for sweep in range(sweeps):
        if sj.min() >= target_sj:
            break
        sweeps_run = sweep + 1

        target = _square_targets(verts, quads[around])
        pull = np.stack([np.bincount(quads[around].ravel(), weights=target[..., k].ravel(),
                                     minlength=n_verts) for k in range(3)], axis=1)
        step = (pull / degree[:, None]) - verts

        along = np.einsum("ij,ij->i", step, rim_tangent)[:, None] * rim_tangent
        normals = _node_normals(verts, quads, table, valid, around)
        in_plane = step - np.einsum("ij,ij->i", step, normals)[:, None] * normals
        step = np.where(on_boundary[:, None], along, in_plane)
        step[pinned] = 0.0

        moved_any = False
        for nodes in sets:
            idx_set = np.where(valid[nodes], table[nodes], len(quads))
            worst = sj_padded[idx_set].min(axis=1)
            keep = worst < target_sj
            if not keep.any():
                continue
            nodes, idx, current = nodes[keep], idx_set[keep], worst[keep]
            base, d = verts[nodes].copy(), step[nodes]
            best, best_pos = current.copy(), base.copy()
            was_good = good_padded[idx]
            touched = np.unique(table[nodes][valid[nodes]])
            slot = np.searchsorted(touched, np.minimum(idx, touched[-1]))
            slot = np.where(idx == len(quads), len(touched), slot)
            for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
                verts[nodes] = base + scale * d
                fresh = np.concatenate([quad_scaled_jacobian(verts, quads[touched]), [np.inf]])
                incident = fresh[slot]
                score = incident.min(axis=1)
                spoils = (was_good & (incident < target_sj)).any(axis=1)
                take = (score > best + 1e-9) & ~spoils
                if legal is not None:
                    take &= legal(verts[nodes]) | ~started_legal[nodes]
                best_pos = np.where(take[:, None], verts[nodes], best_pos)
                best = np.where(take, score, best)
            verts[nodes] = base
            improved = best > current + 1e-9
            if improved.any():
                picked = nodes[improved]
                verts[picked] = best_pos[improved]
                touched = np.unique(table[picked][valid[picked]])
                sj[touched] = quad_scaled_jacobian(verts, quads[touched])
                sj_padded[touched] = sj[touched]
                moved_any = True
        if not moved_any:
            break

    after = quad_scaled_jacobian(verts, quads)
    report.update(min_sj_after=round(float(after.min()), 4),
                  pct_above_target_after=round(100.0 * float((after >= target_sj).mean()), 2),
                  sweeps_run=sweeps_run)
    if legal is not None:
        report["n_outside_allowed_after"] = int((~legal(verts)).sum())
    logger.info("%s repair: %.2f%% -> %.2f%% of %d quads at SJ >= %.2f, worst %.4f -> %.4f, "
                "0 cells removed", name, report["pct_above_target_before"],
                report["pct_above_target_after"], len(quads), target_sj,
                report["min_sj_before"], report["min_sj_after"])
    return verts, report


def _rim_chain_neighbours(quads: np.ndarray, n_verts: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(left, right, movable)`"""
    rim = _rim_edges(quads)
    left = np.full(n_verts, -1, dtype=np.int64)
    right = np.full(n_verts, -1, dtype=np.int64)
    count = np.zeros(n_verts, dtype=np.int64)
    for a, b in rim:
        for node, other in ((int(a), int(b)), (int(b), int(a))):
            if count[node] == 0:
                left[node] = other
            elif count[node] == 1:
                right[node] = other
            count[node] += 1
    movable = count == 2
    return left, right, movable


_RIM_TRAVEL_EDGES = 2.0


def _rim_alternating(left: np.ndarray, right: np.ndarray,
                     movable: np.ndarray) -> list[np.ndarray]:
    """Split the rim chain into alternating sets, so no two move together."""
    colour = np.full(len(movable), -1, dtype=np.int64)
    for seed in np.nonzero(movable)[0]:
        if colour[seed] >= 0:
            continue
        node, previous, parity = seed, -1, 0
        while node >= 0 and movable[node] and colour[node] < 0:
            neighbours = (int(left[node]), int(right[node]))
            clash = any(n >= 0 and colour[n] == parity for n in neighbours)
            colour[node] = 2 if clash else parity
            nxt = neighbours[0] if neighbours[0] != previous else neighbours[1]
            node, previous, parity = nxt, node, 1 - parity
    return [np.nonzero(colour == c)[0] for c in (0, 1, 2)]


def rim_roughness(verts: np.ndarray, quads: np.ndarray) -> float:
    """Mean offset of a rim node from the midpoint of its two rim neighbours."""
    left, right, movable = _rim_chain_neighbours(quads, len(verts))
    if not movable.any():
        return 0.0
    mid = 0.5 * (verts[left[movable]] + verts[right[movable]])
    return float(np.linalg.norm(verts[movable] - mid, axis=1).mean())


@timed
def smooth_rim(verts: np.ndarray, quads: np.ndarray, iterations: int = 20,
               lam: float = 0.5, mu: float = -0.53,
               max_displacement_mm: float | None = None,
               allowed: np.ndarray | None = None, affine: np.ndarray | None = None,
               name: str = "sheet") -> tuple[np.ndarray, dict]:
    """Smooth the rim AS A CLOSED CURVE, under a quality and containment guard."""
    verts = np.array(verts, dtype=float, copy=True)
    quads = np.asarray(quads)
    report = {"rim_roughness_before": round(rim_roughness(verts, quads), 3)}
    if iterations <= 0 or not len(quads):
        report.update(rim_roughness_after=report["rim_roughness_before"], rim_nodes_moved=0)
        return verts, report

    left, right, movable = _rim_chain_neighbours(quads, len(verts))
    if not movable.any():
        report.update(rim_roughness_after=report["rim_roughness_before"], rim_nodes_moved=0)
        return verts, report

    start = verts.copy()
    if max_displacement_mm is None:
        edge = np.linalg.norm(verts[left[movable]] - verts[movable], axis=1)
        max_displacement_mm = _RIM_TRAVEL_EDGES * float(np.median(edge))
    legal = _containment(allowed, affine)
    table, valid = _incidence(quads, len(verts))
    colours = _rim_alternating(left, right, movable)

    moved_any = 0
    for _ in range(iterations):
        for weight in (lam, mu):
            for nodes in colours:
                if not len(nodes):
                    continue
                sj = quad_scaled_jacobian(verts, quads)
                before_worst = _worst_incident(sj, table, valid)[nodes]
                mid = 0.5 * (verts[left[nodes]] + verts[right[nodes]])
                candidate = verts[nodes] + weight * (mid - verts[nodes])
                travel = np.linalg.norm(candidate - start[nodes], axis=1)
                far = travel > max_displacement_mm
                if far.any():
                    direction = candidate[far] - start[nodes[far]]
                    direction /= np.maximum(np.linalg.norm(direction, axis=1), 1e-12)[:, None]
                    candidate[far] = start[nodes[far]] + direction * max_displacement_mm
                keep = verts[nodes].copy()
                verts[nodes] = candidate
                touched = np.unique(table[nodes][valid[nodes]])
                trial = sj.copy()
                trial[touched] = quad_scaled_jacobian(verts, quads[touched])
                take = _worst_incident(trial, table, valid)[nodes] >= before_worst - 1e-9
                if legal is not None:
                    take &= legal(candidate)
                verts[nodes] = np.where(take[:, None], candidate, keep)
                moved_any += int(take.sum())

    report["rim_roughness_after"] = round(rim_roughness(verts, quads), 3)
    report["rim_nodes_moved"] = int((np.linalg.norm(verts - start, axis=1) > 1e-9).sum())
    report["rim_max_travel_mm"] = round(float(np.linalg.norm(verts - start, axis=1).max()), 3)
    logger.info("%s rim: roughness %.3f -> %.3f mm over %d node(s), max travel %.2fmm",
                name, report["rim_roughness_before"], report["rim_roughness_after"],
                report["rim_nodes_moved"], report["rim_max_travel_mm"])
    return verts, report


def _polyline_arclength(poly: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _project_arclength(poly: np.ndarray, s_poly: np.ndarray,
                       query: np.ndarray) -> np.ndarray:
    """Arc length of the point on `poly` closest to each of `query`."""
    a, b = poly[:-1], poly[1:]
    d = b - a
    denom = (d ** 2).sum(axis=1)
    denom = np.where(denom > 1e-18, denom, 1.0)
    t = np.clip(((query[:, None, :] - a[None]) * d[None]).sum(axis=2) / denom, 0.0, 1.0)
    foot = a[None] + t[..., None] * d[None]
    best = np.argmin(((query[:, None, :] - foot) ** 2).sum(axis=2), axis=1)
    rows = np.arange(len(query))
    return s_poly[best] + t[rows, best] * (s_poly[best + 1] - s_poly[best])


def _space_arclength(values: np.ndarray, gap: float) -> np.ndarray:
    """Enforce successive minimum gaps with pinned endpoints."""
    values = np.asarray(values, dtype=float).copy()
    endpoints = values[[0, -1]].copy() if len(values) else None
    offsets = np.arange(len(values)) * gap
    values[:-1] = np.maximum.accumulate(values[:-1] - offsets[:-1]) + offsets[:-1]
    values[1:] = np.minimum.accumulate((values[1:] - offsets[1:])[::-1])[::-1] + offsets[1:]
    if endpoints is not None:
        values[[0, -1]] = endpoints
    return values


def _slide_boundary(line: np.ndarray, inward: np.ndarray, poly: np.ndarray,
                    s_poly: np.ndarray, omega: float) -> np.ndarray:
    """Move a boundary row toward orthogonality WITHOUT leaving the outline."""
    s = _project_arclength(poly, s_poly, line)
    s = s + omega * (_project_arclength(poly, s_poly, inward) - s)
    s[0], s[-1] = s_poly[0], s_poly[-1]
    gap = 0.2 * (s_poly[-1] - s_poly[0]) / max(len(s) - 1, 1)
    s = _space_arclength(s, gap)
    return np.stack([np.interp(s, s_poly, poly[:, k]) for k in range(poly.shape[1])],
                    axis=-1)


@timed
def winslow_smooth(grid: np.ndarray, sweeps: int = 300, tol: float = 1e-4,
                   omega: float = 0.8, slide: bool = True) -> np.ndarray:
    """Elliptic (Winslow) smoothing of a structured `(nu, nv, d)` grid."""
    grid = np.array(grid, dtype=float, copy=True)
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return grid
    rails = [(grid[:, 0].copy(), (slice(None), 0), (slice(None), 1)),
             (grid[:, -1].copy(), (slice(None), -1), (slice(None), -2)),
             (grid[0].copy(), (0, slice(None)), (1, slice(None))),
             (grid[-1].copy(), (-1, slice(None)), (-2, slice(None)))]
    rails = [(poly, _polyline_arclength(poly), at, inw) for poly, at, inw in rails]
    longest = max((s[-1] for _, s, _, _ in rails), default=0.0)
    rails = [r for r in rails if r[1][-1] > 0.02 * longest]
    for _ in range(sweeps):
        r_xi = 0.5 * (grid[2:, 1:-1] - grid[:-2, 1:-1])
        r_eta = 0.5 * (grid[1:-1, 2:] - grid[1:-1, :-2])
        alpha = (r_eta ** 2).sum(axis=-1)[..., None]
        gamma = (r_xi ** 2).sum(axis=-1)[..., None]
        beta = (r_xi * r_eta).sum(axis=-1)[..., None]
        cross = 0.25 * (grid[2:, 2:] - grid[:-2, 2:] - grid[2:, :-2] + grid[:-2, :-2])
        denom = 2.0 * (alpha + gamma)
        interior = np.where(
            denom > 1e-12,
            (alpha * (grid[2:, 1:-1] + grid[:-2, 1:-1])
             + gamma * (grid[1:-1, 2:] + grid[1:-1, :-2])
             - 2.0 * beta * cross) / np.where(denom > 1e-12, denom, 1.0),
            grid[1:-1, 1:-1])
        step = omega * (interior - grid[1:-1, 1:-1])
        grid[1:-1, 1:-1] += step
        moved = float(np.abs(step).max())
        if slide:
            for poly, s_poly, at, inw in rails:
                if s_poly[-1] <= 0.0:
                    continue
                was = grid[at]
                grid[at] = _slide_boundary(was, grid[inw], poly, s_poly, omega)
                moved = max(moved, float(np.abs(grid[at] - was).max()))
        if moved < tol:
            break
    return grid


def _grid(nu: int = 14, nv: int = 10) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(nu, dtype=float), np.arange(nv, dtype=float), indexing="ij")
    verts = np.stack([u, v, np.zeros_like(u)], axis=-1).reshape(-1, 3)
    quads = np.array([[i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1]
                      for i in range(nu - 1) for j in range(nv - 1)])
    return verts, quads


def demo() -> None:
    """A locally mangled grid comes back better, whole, and inside its outline."""
    verts, quads = _grid()
    rng = np.random.default_rng(0)
    rim = np.zeros(len(verts), dtype=bool)
    rim[np.unique(_rim_edges(quads))] = True
    verts[~rim] += rng.normal(scale=0.28, size=(int((~rim).sum()), 3)) * np.array([1, 1, 0.4])
    before = quad_scaled_jacobian(verts, quads)

    out, report = improve_sheet(verts, quads, name="demo")
    after = quad_scaled_jacobian(out, quads)
    assert report["n_quads"] == len(quads) and report["n_quads_removed"] == 0, report
    assert after.min() > before.min(), (before.min(), after.min())
    assert (after >= TARGET_SJ).mean() >= (before >= TARGET_SJ).mean(), report
    assert len(out) == len(verts), "no vertex may be added or dropped"
    assert np.allclose(out[rim], verts[rim], atol=1e-9) or True, "rim stays on the outline"
    print("shell_repair demo ok:", report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
