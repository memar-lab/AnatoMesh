"""Trilinear hexahedron integration in VTK corner order."""

from itertools import product

import numpy as np
from scipy import sparse

SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
GAUSS = np.array(list(product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3)))


def shape_gradients(locations):
    locations = np.asarray(locations, dtype=float)
    terms = 1 + locations[:, None, :] * SIGNS[None, :, :]
    return np.stack([SIGNS[:, axis] * terms[:, :, (axis + 1) % 3]
                     * terms[:, :, (axis + 2) % 3] / 8 for axis in range(3)], axis=2)


GRADIENTS = shape_gradients(GAUSS)
QUALITY_GRADIENTS = shape_gradients(np.concatenate([GAUSS, SIGNS, [[0, 0, 0]]]))


# a hex (corners 0-3 one face, 4-7 the opposite face) as six tetrahedra around the 0-6 diagonal
TETRAHEDRA = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6],
                       [0, 5, 1, 6]])


def _pih_kernel():
    """numba version of the per-candidate test below, built on first use."""
    from numba import njit, prange

    tets = TETRAHEDRA.astype(np.int64)

    @njit(cache=True, parallel=True)
    def kernel(points, nodes, hexes, near, tol, found):
        margin = 1e-6
        for ii in prange(points.shape[0]):
            i = np.int64(ii)
            q = points[i]
            for col in range(near.shape[1]):
                h = near[i, col]
                if h < 0:
                    break
                inside = False
                for t in range(6):
                    a0 = hexes[h, tets[t, 0]]
                    b0, b1, b2 = hexes[h, tets[t, 1]], hexes[h, tets[t, 2]], hexes[h, tets[t, 3]]
                    ax, ay, az = nodes[a0, 0], nodes[a0, 1], nodes[a0, 2]
                    # scalar arithmetic in numpy's order (np.cross, then the 3-term dot), no
                    # allocations in the loop
                    e1x, e1y, e1z = nodes[b0, 0] - ax, nodes[b0, 1] - ay, nodes[b0, 2] - az
                    e2x, e2y, e2z = nodes[b1, 0] - ax, nodes[b1, 1] - ay, nodes[b1, 2] - az
                    e3x, e3y, e3z = nodes[b2, 0] - ax, nodes[b2, 1] - ay, nodes[b2, 2] - az
                    rx, ry, rz = q[0] - ax, q[1] - ay, q[2] - az
                    cx = e2y * e3z - e2z * e3y
                    cy = e2z * e3x - e2x * e3z
                    cz = e2x * e3y - e2y * e3x
                    vol = e1x * cx + e1y * cy + e1z * cz
                    safe = 1.0 if vol == 0 else vol
                    sx = ry * e3z - rz * e3y
                    sy = rz * e3x - rx * e3z
                    sz = rx * e3y - ry * e3x
                    tx = e2y * rz - e2z * ry
                    ty = e2z * rx - e2x * rz
                    tz = e2x * ry - e2y * rx
                    l0 = (rx * cx + ry * cy + rz * cz) / safe
                    l1 = (e1x * sx + e1y * sy + e1z * sz) / safe
                    l2 = (e1x * tx + e1y * ty + e1z * tz) / safe
                    s = l0 + l1 + l2
                    edge = min(min(abs(l0 + tol), abs(l1 + tol), abs(l2 + tol)),
                               abs(s - 1 - tol))
                    if abs(vol) < 1e-12 or edge < margin:
                        m = np.empty((3, 3))
                        m[0, 0], m[1, 0], m[2, 0] = e1x, e1y, e1z
                        m[0, 1], m[1, 1], m[2, 1] = e2x, e2y, e2z
                        m[0, 2], m[1, 2], m[2, 2] = e3x, e3y, e3z
                        if abs(np.linalg.det(m)) > 1e-18:
                            rv = np.empty(3)
                            rv[0], rv[1], rv[2] = rx, ry, rz
                            x = np.linalg.solve(m, rv)
                            if (x[0] >= -tol and x[1] >= -tol and x[2] >= -tol
                                    and x[0] + x[1] + x[2] <= 1 + tol):
                                inside = True
                    elif l0 >= -tol and l1 >= -tol and l2 >= -tol and s <= 1 + tol:
                        inside = True
                if inside:
                    found[i] = h
                    break

    return kernel


_PIH = None
_NEAR = None


def _near_kernel():
    """k nearest hex centroids per point from a uniform grid of centroids. The cell is
    at least the largest centroid-to-corner distance, so every hex that can contain a point has
    its centroid in the point's cell or a neighbour; the k nearest of those are returned in
    distance order (ties by hex index), the candidates the KD-tree query gave."""
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def kernel(points, cent, lo, size, dims, starts, order, k, near):
        for ii in prange(points.shape[0]):
            i = np.int64(ii)
            best_d = np.full(k, np.inf)
            best_h = np.full(k, -1, np.int64)
            c = np.empty(3, np.int64)
            for a in range(3):
                c[a] = np.int64(np.floor((points[i, a] - lo[a]) / size))
            for dx in range(-1, 2):
                x = c[0] + dx
                if x < 0 or x >= dims[0]:
                    continue
                for dy in range(-1, 2):
                    y = c[1] + dy
                    if y < 0 or y >= dims[1]:
                        continue
                    for dz in range(-1, 2):
                        z = c[2] + dz
                        if z < 0 or z >= dims[2]:
                            continue
                        cell = (x * dims[1] + y) * dims[2] + z
                        for j in range(starts[cell], starts[cell + 1]):
                            h = order[j]
                            d = ((points[i, 0] - cent[h, 0]) ** 2 + (points[i, 1] - cent[h, 1]) ** 2
                                 + (points[i, 2] - cent[h, 2]) ** 2)
                            if d < best_d[k - 1] or (d == best_d[k - 1] and h < best_h[k - 1]):
                                p = k - 1
                                while p > 0 and (d < best_d[p - 1]
                                                 or (d == best_d[p - 1] and h < best_h[p - 1])):
                                    best_d[p] = best_d[p - 1]
                                    best_h[p] = best_h[p - 1]
                                    p -= 1
                                best_d[p] = d
                                best_h[p] = h
            for j in range(k):
                near[i, j] = best_h[j]

    return kernel


def _grid_candidates(points, nodes, hexes, k):
    global _NEAR
    corners = nodes[hexes]
    cent = corners.mean(axis=1)
    size = float(np.sqrt(((corners - cent[:, None, :]) ** 2).sum(axis=2).max())) * (1 + 1e-9)
    lo = cent.min(axis=0) - size
    dims = np.ceil((cent.max(axis=0) - lo) / size).astype(np.int64) + 2
    ijk = np.floor((cent - lo) / size).astype(np.int64)
    lin = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(lin, kind="stable").astype(np.int64)
    starts = np.searchsorted(lin[order], np.arange(int(np.prod(dims)) + 1)).astype(np.int64)
    if _NEAR is None:
        _NEAR = _near_kernel()
    near = np.empty((len(points), k), np.int64)
    _NEAR(points, cent, lo, size, dims, starts, order, k, near)
    return near


def point_in_hexes(points, nodes, hexes, k=8, tol=1e-9, chunk=20000, tree=None):
    """Index of the hex that contains each point, -1 where none does.

    Exact for the six-tetrahedron split of each hex; the k nearest hexes by centroid are tried
    in order, the first that contains the point wins. ANATOMESH_PY_PIH=1 runs the numpy path.
    """
    global _PIH
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    nodes = np.asarray(nodes, dtype=float)
    hexes = np.asarray(hexes)
    found = np.full(len(points), -1)
    if not len(points) or not len(hexes):
        return found
    import os
    from concurrent.futures import ThreadPoolExecutor

    threads = int(os.environ.get("ANATOMESH_THREADS", min(8, os.cpu_count() or 1)))
    k = min(k, len(hexes))
    if not os.environ.get("ANATOMESH_PY_PIH"):
        if _PIH is None:
            _PIH = _pih_kernel()
        pts = np.ascontiguousarray(points)
        hx = np.ascontiguousarray(hexes, dtype=np.int64)
        near = _grid_candidates(pts, nodes, hx, k)
        out = np.full(len(points), -1, dtype=np.int64)
        _PIH(pts, np.ascontiguousarray(nodes), hx, near, float(tol), out)
        found[:] = out
        return found
    tree = tree if tree is not None else cKDTree(nodes[hexes].mean(axis=1))
    _d, near = tree.query(points, k=k, workers=threads)
    near = near.reshape(len(points), k)

    def one_chunk(start):  # chunks are independent; numpy's linalg releases the GIL
        p, cand = points[start:start + chunk], near[start:start + chunk]
        hit = found[start:start + chunk]
        for column in range(k):
            todo = np.flatnonzero(hit < 0)
            if not len(todo):
                break
            h = cand[todo, column]
            corners = nodes[hexes[h]]
            q = p[todo]
            inside = np.zeros(len(h), bool)
            for tet in TETRAHEDRA:
                a = corners[:, tet[0]]
                e1, e2, e3 = (corners[:, tet[1]] - a, corners[:, tet[2]] - a,
                              corners[:, tet[3]] - a)
                r = q - a
                # barycentric coordinates by Cramer's rule; anything near a decision edge is
                # redone with the LU solve below, so the answer is the solve's answer
                c23 = np.cross(e2, e3)
                vol = np.einsum("ij,ij->i", e1, c23)
                safe = np.where(vol == 0, 1.0, vol)
                lam = np.stack([np.einsum("ij,ij->i", r, c23),
                                np.einsum("ij,ij->i", e1, np.cross(r, e3)),
                                np.einsum("ij,ij->i", e1, np.cross(e2, r))], axis=1) / safe[:, None]
                margin = 1e-6
                edge = np.minimum(np.abs(lam + tol).min(axis=1), np.abs(lam.sum(axis=1) - 1 - tol))
                exact = (np.abs(vol) < 1e-12) | (edge < margin)
                inside |= ~exact & np.all(lam >= -tol, axis=1) & (lam.sum(axis=1) <= 1 + tol)
                if exact.any():
                    matrix = np.stack([e1[exact], e2[exact], e3[exact]], axis=2)
                    det = np.linalg.det(matrix)
                    good = np.abs(det) > 1e-18
                    lam_x = np.zeros((int(exact.sum()), 3))
                    lam_x[good] = np.linalg.solve(matrix[good], r[exact][good][..., None])[..., 0]
                    inside[exact] |= (good & np.all(lam_x >= -tol, axis=1)
                                      & (lam_x.sum(axis=1) <= 1 + tol))
            hit[todo[inside]] = h[inside]

    starts = range(0, len(points), chunk)
    if threads > 1 and len(starts) > 1:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(one_chunk, starts))
    else:
        for start in starts:
            one_chunk(start)
    return found


def validate_hex_input(nodes, hexes):
    nodes = np.asarray(nodes, dtype=float)
    hexes = np.asarray(hexes)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
        raise ValueError("nodes must be finite coordinates of shape (N, 3)")
    if hexes.ndim != 2 or hexes.shape[1] != 8 or hexes.dtype.kind not in "iu":
        raise ValueError("hexes must be integer connectivity of shape (M, 8)")
    if hexes.size and (hexes.min() < 0 or hexes.max() >= len(nodes)):
        raise ValueError("hex connectivity references an absent node")
    if len(hexes) and np.any(np.diff(np.sort(hexes, axis=1), axis=1) == 0):
        raise ValueError("hex connectivity repeats a node")
    return nodes, hexes


def _jacobians(vertices, gradients):
    return np.einsum("eia,qib->eqab", vertices - vertices[:, :1], gradients)


def element_volumes(nodes, hexes, chunk_size=16384):
    nodes, hexes = validate_hex_input(nodes, hexes)
    volumes = np.empty(len(hexes))
    for start in range(0, len(hexes), chunk_size):
        stop = start + chunk_size
        jac = _jacobians(nodes[hexes[start:stop]], GRADIENTS)
        volumes[start:stop] = np.linalg.det(jac).sum(axis=1)
    return volumes


def jacobian_metrics(nodes, hexes, chunk_size=16384):
    nodes, hexes = validate_hex_input(nodes, hexes)
    minimum = np.empty(len(hexes))
    scaled = np.empty(len(hexes))
    for start in range(0, len(hexes), chunk_size):
        stop = start + chunk_size
        jac = _jacobians(nodes[hexes[start:stop]], QUALITY_GRADIENTS)
        det = np.linalg.det(jac)
        denominator = np.prod(np.linalg.norm(jac, axis=2), axis=2)
        sj = np.divide(det, denominator, out=np.zeros_like(det), where=denominator > 0)
        minimum[start:stop] = det.min(axis=1)
        scaled[start:stop] = sj.min(axis=1)
    return minimum, scaled


def regional_volumes(nodes, hexes, labels):
    labels = np.asarray(labels)
    if labels.shape != (len(hexes),) or labels.dtype.kind not in "iu":
        raise ValueError("labels must contain one integer material ID per element")
    regions, inverse = np.unique(labels, return_inverse=True)
    return regions, np.bincount(inverse, weights=element_volumes(nodes, hexes))


def volume_derivatives(nodes, hexes, labels, movable, chunk_size=8192):
    """Return regional dV/dx with fixed-node columns zeroed."""
    nodes, hexes = validate_hex_input(nodes, hexes)
    regions, inverse = np.unique(labels, return_inverse=True)
    result = sparse.csr_matrix((len(regions), nodes.size))
    for start in range(0, len(hexes), chunk_size):
        conn = hexes[start:start + chunk_size]
        jac = _jacobians(nodes[conn], GRADIENTS)
        cofactor = np.stack([np.cross(jac[..., 1], jac[..., 2]),
                             np.cross(jac[..., 2], jac[..., 0]),
                             np.cross(jac[..., 0], jac[..., 1])], axis=-1)
        derivative = np.einsum("eqab,qib->eia", cofactor, GRADIENTS)
        derivative *= movable[conn, None]
        columns = (conn[:, :, None] * 3 + np.arange(3)).ravel()
        rows = np.repeat(inverse[start:start + len(conn)], 24)
        result += sparse.coo_matrix((derivative.ravel(), (rows, columns)),
                                    shape=result.shape).tocsr()
    result.eliminate_zeros()
    return result
