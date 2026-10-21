"""Compiled (numba) kernels for the hot loops; every function has a NumPy twin elsewhere and
the callers fall back to it when numba is not installed (HAVE_NUMBA False).

hex_scaled_jacobian   same values as export.validate._hex_scaled_jacobian (worst of the eight
                      corner Jacobians, each det(e1, e2, e3) / (|e1| |e2| |e3|))
free_node_moves       the hex quality pass for interior nodes: try parallelepiped then
                      Laplacian targets at decreasing step scales, keep the first trial that
                      raises the node's worst hex (score = min + w * mean) without pushing any
                      hex below a protected band (mesh.hex_quality rules)
"""
from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - the NumPy path is used instead
    HAVE_NUMBA = False

# corner c: its three edge neighbours, in export.validate order
CORNER_NEIGHBOURS = np.array([(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
                              (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)], dtype=np.int64)
# mesh.voxel_hex: edge vectors L (b - a; L2 and L10 are flipped back after), the eight corner
# matrices as (L row, sign) x 3, and the edge / diagonal pairs of the element quality metrics
_SJ_PAIRS = np.array([(0, 1), (1, 2), (3, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7),
                      (4, 5), (5, 6), (7, 6), (4, 7)], dtype=np.int64)
_SJ_ROWS = np.array([(0, 1, 3, 1, 4, 1), (1, 1, 0, -1, 5, 1), (2, 1, 1, -1, 6, 1),
                     (3, -1, 2, -1, 7, 1), (11, 1, 8, 1, 4, -1), (8, -1, 9, 1, 5, -1),
                     (9, -1, 10, 1, 6, -1), (10, -1, 11, -1, 7, -1)], dtype=np.int64)
_E_PAIRS = np.array([(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7),
                     (4, 5), (5, 6), (6, 7), (4, 7)], dtype=np.int64)
_D_PAIRS = np.array([(0, 6), (1, 7), (2, 4), (3, 5)], dtype=np.int64)

if HAVE_NUMBA:

    @njit(cache=True, parallel=True)
    def _trilinear(vol, idx, out):
        """Trilinear samples of `vol` at idx (3, n), NaN where a corner is outside."""
        nx, ny, nz = vol.shape
        for p in prange(idx.shape[1]):
            x, y, z = idx[0, p], idx[1, p], idx[2, p]
            if not (x >= 0.0 and y >= 0.0 and z >= 0.0
                    and x <= nx - 1 and y <= ny - 1 and z <= nz - 1):
                out[p] = np.nan
                continue
            i0, j0, k0 = int(x), int(y), int(z)
            i1, j1, k1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1), min(k0 + 1, nz - 1)
            fx, fy, fz = x - i0, y - j0, z - k0
            c00 = vol[i0, j0, k0] * (1 - fz) + vol[i0, j0, k1] * fz
            c01 = vol[i0, j1, k0] * (1 - fz) + vol[i0, j1, k1] * fz
            c10 = vol[i1, j0, k0] * (1 - fz) + vol[i1, j0, k1] * fz
            c11 = vol[i1, j1, k0] * (1 - fz) + vol[i1, j1, k1] * fz
            out[p] = (c00 * (1 - fy) + c01 * fy) * (1 - fx) + (c10 * (1 - fy) + c11 * fy) * fx

    @njit(cache=True, fastmath=False, inline="always")
    def _sj8(c, nb):
        worst = 1e30
        for k in range(8):
            i, j, m = nb[k, 0], nb[k, 1], nb[k, 2]
            ax, ay, az = c[i, 0] - c[k, 0], c[i, 1] - c[k, 1], c[i, 2] - c[k, 2]
            bx, by, bz = c[j, 0] - c[k, 0], c[j, 1] - c[k, 1], c[j, 2] - c[k, 2]
            dx, dy, dz = c[m, 0] - c[k, 0], c[m, 1] - c[k, 1], c[m, 2] - c[k, 2]
            det = ax * (by * dz - bz * dy) - ay * (bx * dz - bz * dx) + az * (bx * dy - by * dx)
            norms = (np.sqrt(ax * ax + ay * ay + az * az) * np.sqrt(bx * bx + by * by + bz * bz)
                     * np.sqrt(dx * dx + dy * dy + dz * dz))
            value = det / norms if norms > 0 else 0.0
            if value < worst:
                worst = value
        return worst

    @njit(cache=True, parallel=True)
    def _sj_batch(coords, nb):
        out = np.empty(coords.shape[0])
        for e in prange(coords.shape[0]):
            out[e] = _sj8(coords[e], nb)
        return out

    @njit(cache=True, parallel=True)
    def _sj_indexed(nodes, hexes, which, nb):
        out = np.empty(which.shape[0])
        for t in prange(which.shape[0]):
            c = np.empty((8, 3))
            h = which[t]
            for q in range(8):
                for d in range(3):
                    c[q, d] = nodes[hexes[h, q], d]
            out[t] = _sj8(c, nb)
        return out

    @njit(cache=True, parallel=True)
    def _free_moves(nodes, hexes, inc, loc, rows, sj, para, lap, scales, weight, levels, nb,
                    opposite, own_targets):
        n = rows.shape[0]
        new_pos = np.empty((n, 3))
        moved = np.zeros(n, dtype=np.bool_)
        for t in prange(n):
            r = rows[t]
            k = 0
            while k < inc.shape[1] and inc[r, k] >= 0:
                k += 1
            if k == 0:
                continue
            if own_targets:
                # mesh.hex_quality._targets for this node: mean parallelepiped completion and
                # mean of its edge neighbours over its hexes
                px_ = py_ = pz_ = 0.0
                lx_ = ly_ = lz_ = 0.0
                for a in range(k):
                    h = inc[r, a]
                    c0 = loc[r, a]
                    sx = sy = sz = 0.0
                    ox = oy = oz = 0.0
                    for q in range(3):
                        m = hexes[h, nb[c0, q]]
                        sx += nodes[m, 0]
                        sy += nodes[m, 1]
                        sz += nodes[m, 2]
                        m = hexes[h, opposite[c0, q]]
                        ox += nodes[m, 0]
                        oy += nodes[m, 1]
                        oz += nodes[m, 2]
                    px_ += (2.0 * sx - ox) / 3.0
                    py_ += (2.0 * sy - oy) / 3.0
                    pz_ += (2.0 * sz - oz) / 3.0
                    lx_ += sx / 3.0
                    ly_ += sy / 3.0
                    lz_ += sz / 3.0
                para[t, 0], para[t, 1], para[t, 2] = px_ / k, py_ / k, pz_ / k
                lap[t, 0], lap[t, 1], lap[t, 2] = lx_ / k, ly_ / k, lz_ / k
            low0 = 1e30
            sum0 = 0.0
            for a in range(k):
                v = sj[inc[r, a]]
                sum0 += v
                if v < low0:
                    low0 = v
            score0 = low0 + weight * sum0 / k
            px, py, pz = nodes[r, 0], nodes[r, 1], nodes[r, 2]
            c = np.empty((8, 3))
            done = False
            for g in range(2):
                if done:
                    break
                gx = para[t, 0] if g == 0 else lap[t, 0]
                gy = para[t, 1] if g == 0 else lap[t, 1]
                gz = para[t, 2] if g == 0 else lap[t, 2]
                for s in range(scales.shape[0]):
                    sc = scales[s]
                    cx, cy, cz = px + sc * (gx - px), py + sc * (gy - py), pz + sc * (gz - pz)
                    low = 1e30
                    total = 0.0
                    ok = True
                    for a in range(k):
                        h = inc[r, a]
                        for q in range(8):
                            for d in range(3):
                                c[q, d] = nodes[hexes[h, q], d]
                        corner = loc[r, a]
                        c[corner, 0], c[corner, 1], c[corner, 2] = cx, cy, cz
                        v = _sj8(c, nb)
                        if v != v:          # NaN counts as inverted
                            v = -1.0
                        before = sj[h]
                        for lv in range(levels.shape[0]):
                            if before >= levels[lv] and v < levels[lv]:
                                ok = False
                        total += v
                        if v < low:
                            low = v
                    if not ok:
                        continue
                    score = low + weight * total / k
                    if low >= low0 - 1e-12 and score > score0 + 1e-9:
                        new_pos[t, 0], new_pos[t, 1], new_pos[t, 2] = cx, cy, cz
                        moved[t] = True
                        done = True
                        break
        return new_pos, moved


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _closest_tri(px, py, pz, t):
        """Closest point on triangle t (3, 3) to p (Ericson, Real-Time Collision Detection 5.1.5)."""
        ax, ay, az = t[0, 0], t[0, 1], t[0, 2]
        abx, aby, abz = t[1, 0] - ax, t[1, 1] - ay, t[1, 2] - az
        acx, acy, acz = t[2, 0] - ax, t[2, 1] - ay, t[2, 2] - az
        apx, apy, apz = px - ax, py - ay, pz - az
        d1 = abx * apx + aby * apy + abz * apz
        d2 = acx * apx + acy * apy + acz * apz
        if d1 <= 0.0 and d2 <= 0.0:
            return ax, ay, az
        bpx, bpy, bpz = px - t[1, 0], py - t[1, 1], pz - t[1, 2]
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0.0 and d4 <= d3:
            return t[1, 0], t[1, 1], t[1, 2]
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return ax + v * abx, ay + v * aby, az + v * abz
        cpx, cpy, cpz = px - t[2, 0], py - t[2, 1], pz - t[2, 2]
        d5 = abx * cpx + aby * cpy + abz * cpz
        d6 = acx * cpx + acy * cpy + acz * cpz
        if d6 >= 0.0 and d5 <= d6:
            return t[2, 0], t[2, 1], t[2, 2]
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return ax + w * acx, ay + w * acy, az + w * acz
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            bx, by, bz = t[1, 0], t[1, 1], t[1, 2]
            return bx + w * (t[2, 0] - bx), by + w * (t[2, 1] - by), bz + w * (t[2, 2] - bz)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w

    @njit(cache=True, parallel=True)
    def _closest_batch(p, tri):
        n, m = tri.shape[0], tri.shape[1]
        out = np.empty((n, 3))
        for i in prange(n):
            best = 1e300
            bx = by = bz = 0.0
            for j in range(m):
                qx, qy, qz = _closest_tri(p[i, 0], p[i, 1], p[i, 2], tri[i, j])
                d = (qx - p[i, 0]) ** 2 + (qy - p[i, 1]) ** 2 + (qz - p[i, 2]) ** 2
                if d < best:
                    best, bx, by, bz = d, qx, qy, qz
            out[i, 0], out[i, 1], out[i, 2] = bx, by, bz
        return out


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _det_unit_rows(ax, ay, az, bx, by, bz, cx, cy, cz):
        """det of the 3x3 matrix with rows a, b, c after mesh.voxel_hex's normalisation,
        which divides each COLUMN (x, y, z across the three vectors) by its norm
        (`A / sqrt((A * A).sum(axis=2))` on (n, 9, 3, 3)); NaN for a zero column, as 0/0."""
        nx = np.sqrt(ax * ax + bx * bx + cx * cx)
        ny = np.sqrt(ay * ay + by * by + cy * cy)
        nz = np.sqrt(az * az + bz * bz + cz * cz)
        if nx == 0.0 or ny == 0.0 or nz == 0.0:
            return np.nan
        ax, bx, cx = ax / nx, bx / nx, cx / nx
        ay, by, cy = ay / ny, by / ny, cy / ny
        az, bz, cz = az / nz, bz / nz, cz / nz
        return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)

    @njit(cache=True, inline="always")
    def _matlab_sj(c):
        """the element quality metrics scaled Jacobian of one hex (mesh.voxel_hex._scaled_jacobian)."""
        L = np.empty((12, 3))
        for e in range(12):
            a, b = _SJ_PAIRS[e, 0], _SJ_PAIRS[e, 1]
            for d in range(3):
                L[e, d] = c[b, d] - c[a, d]
        # L[2] = c3 - c2 in voxel_hex._edges; pairs above give c2 - c3, so flip it back
        for d in range(3):
            L[2, d] = -L[2, d]
            L[10, d] = -L[10, d]
        x = np.empty((3, 3))
        for d in range(3):
            x[0, d] = (c[1, d] - c[0, d]) + (c[2, d] - c[3, d]) + (c[5, d] - c[4, d]) + (c[6, d] - c[7, d])
            x[1, d] = (c[3, d] - c[0, d]) + (c[2, d] - c[1, d]) + (c[7, d] - c[4, d]) + (c[6, d] - c[5, d])
            x[2, d] = (c[4, d] - c[0, d]) + (c[5, d] - c[1, d]) + (c[6, d] - c[2, d]) + (c[7, d] - c[3, d])
        # (row index into L, sign) for the three rows of the eight corner matrices
        best = np.inf
        seen = False
        for m in range(8):
            i, si, j = _SJ_ROWS[m, 0], _SJ_ROWS[m, 1], _SJ_ROWS[m, 2]
            sj, k, sk = _SJ_ROWS[m, 3], _SJ_ROWS[m, 4], _SJ_ROWS[m, 5]
            v = _det_unit_rows(si * L[i, 0], si * L[i, 1], si * L[i, 2],
                               sj * L[j, 0], sj * L[j, 1], sj * L[j, 2],
                               sk * L[k, 0], sk * L[k, 1], sk * L[k, 2])
            if v == v:
                seen = True
                if v < best:
                    best = v
        v = _det_unit_rows(x[0, 0], x[0, 1], x[0, 2], x[1, 0], x[1, 1], x[1, 2],
                           x[2, 0], x[2, 1], x[2, 2])
        if v == v:
            seen = True
            if v < best:
                best = v
        return best if seen else np.nan

    @njit(cache=True, parallel=True)
    def _matlab_metrics(coords, want_ratios):
        n = coords.shape[0]
        jac = np.empty(n)
        edge = np.empty(n)
        diag = np.empty(n)
        for t in prange(n):
            c = coords[t]
            jac[t] = _matlab_sj(c)
            if want_ratios:
                lo, hi = np.inf, 0.0
                for e in range(12):
                    a, b = _E_PAIRS[e, 0], _E_PAIRS[e, 1]
                    s = 0.0
                    for d in range(3):
                        s += (c[b, d] - c[a, d]) ** 2
                    s = np.sqrt(s)
                    lo = min(lo, s)
                    hi = max(hi, s)
                edge[t] = hi / lo
                lo, hi = np.inf, 0.0
                for e in range(4):
                    a, b = _D_PAIRS[e, 0], _D_PAIRS[e, 1]
                    s = 0.0
                    for d in range(3):
                        s += (c[b, d] - c[a, d]) ** 2
                    s = np.sqrt(s)
                    lo = min(lo, s)
                    hi = max(hi, s)
                diag[t] = lo / hi
        return jac, edge, diag


def reference_metrics(coords: np.ndarray):
    """(scaled Jacobian, edge ratio, diagonal ratio) as mesh.voxel_hex._compute_metrics."""
    return _matlab_metrics(np.ascontiguousarray(coords, dtype=np.float64), True)


def reference_scaled_jacobian(coords: np.ndarray) -> np.ndarray:
    """mesh.voxel_hex._scaled_jacobian (compiled)."""
    return _matlab_metrics(np.ascontiguousarray(coords, dtype=np.float64), False)[0]


if HAVE_NUMBA:

    @njit(cache=True, parallel=True)
    def _pinch_winners(padded, rank, n_ranked, bad, fix):
        """mesh.hex_topology.clean_topology's per-value loop in one pass: at every node of the
        padded label volume, the highest-priority (lowest rank) label whose 2x2x2 occupancy
        pattern is non-manifold, and that pattern's fix bits."""
        nx, ny, nz = padded.shape[0] - 1, padded.shape[1] - 1, padded.shape[2] - 1
        winner = np.full((nx, ny, nz), -1, dtype=np.int16)
        winner_fix = np.zeros((nx, ny, nz), dtype=np.uint8)
        for i in prange(nx):
            labs = np.empty(8, dtype=np.int64)
            for j in range(ny):
                for k in range(nz):
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                labs[4 * u + 2 * v + w] = padded[i + u, j + v, k + w]
                    best_rank = n_ranked
                    best_val = -1
                    best_fix = 0
                    for a in range(8):
                        value = labs[a]
                        seen = False
                        for b in range(a):
                            if labs[b] == value:
                                seen = True
                                break
                        if seen or value < 0 or value >= rank.shape[0]:
                            continue
                        r = rank[value]
                        if r >= best_rank:
                            continue
                        code = 0
                        for b in range(8):
                            if labs[b] == value:
                                code |= 1 << b
                        if bad[code]:
                            best_rank, best_val, best_fix = r, value, fix[code]
                    if best_val >= 0:
                        winner[i, j, k] = best_val
                        winner_fix[i, j, k] = best_fix
        return winner, winner_fix

    @njit(cache=True, parallel=True)
    def _pinch_counts(padded, n_values, bad):
        """Per label, the number of nodes whose 2x2x2 pattern of that label is non-manifold
        (mesh.hex_topology.count_nonmanifold for every value at once)."""
        nx, ny, nz = padded.shape[0] - 1, padded.shape[1] - 1, padded.shape[2] - 1
        per_row = np.zeros((nx, n_values), dtype=np.int64)
        for i in prange(nx):
            labs = np.empty(8, dtype=np.int64)
            for j in range(ny):
                for k in range(nz):
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                labs[4 * u + 2 * v + w] = padded[i + u, j + v, k + w]
                    for a in range(8):
                        value = labs[a]
                        seen = False
                        for b in range(a):
                            if labs[b] == value:
                                seen = True
                                break
                        if seen or value < 0 or value >= n_values:
                            continue
                        code = 0
                        for b in range(8):
                            if labs[b] == value:
                                code |= 1 << b
                        if bad[code]:
                            per_row[i, value] += 1
        return per_row.sum(axis=0)


def pinch_winners(padded: np.ndarray, rank: np.ndarray, n_ranked: int, bad: np.ndarray,
                  fix: np.ndarray):
    """(winner, winner_fix) of hex_topology.clean_topology for every node at once."""
    return _pinch_winners(np.ascontiguousarray(padded), np.asarray(rank, dtype=np.int64),
                          int(n_ranked), np.asarray(bad, dtype=np.bool_),
                          np.asarray(fix, dtype=np.uint8))


def pinch_counts(padded: np.ndarray, n_values: int, bad: np.ndarray) -> np.ndarray:
    """Non-manifold node count per label value 0..n_values-1."""
    return _pinch_counts(np.ascontiguousarray(padded), int(n_values),
                         np.asarray(bad, dtype=np.bool_))


try:
    import edt as _edt          # exact Euclidean distance transform, multithreaded (seung-lab)

    HAVE_EDT = True
except ImportError:  # pragma: no cover
    HAVE_EDT = False


def distance_transform_edt(mask, sampling=None) -> np.ndarray:
    """scipy.ndimage.distance_transform_edt(mask, sampling) (distances only): the distance of
    every nonzero voxel to the nearest zero voxel, float64. Same values (exact Euclidean, no
    implicit zero border), about 8x faster on a 256^3 head through the `edt` package."""
    from scipy import ndimage

    arr = np.asarray(mask)
    unit = sampling is None or bool(np.all(np.asarray(sampling, dtype=float) == 1.0))
    # edt works in float32; squared distances in voxel units are whole numbers, exact in
    # float32 up to 2**24 (4096 voxels), so sqrt in float64 equals scipy's value bit for bit.
    # Non-unit or anisotropic spacing stays on scipy (its rounding there is not reproducible).
    if not (HAVE_EDT and unit and arr.ndim in (2, 3) and max(arr.shape) < 4096):
        return ndimage.distance_transform_edt(mask, sampling=sampling)
    fg = np.ascontiguousarray(arr != 0)
    if fg.all() or not fg.any():   # no zero / no foreground: scipy's own conventions
        return ndimage.distance_transform_edt(mask, sampling=sampling)
    import os

    sq = _edt.edtsq(fg, anisotropy=(1.0,) * fg.ndim, black_border=False,
                    parallel=min(8, os.cpu_count() or 1))
    return np.sqrt(sq.astype(np.float64))


def closest_on_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Closest point to each p (n, 3) on its own triangles tri (n, t, 3, 3) (compiled)."""
    return _closest_batch(np.ascontiguousarray(p, dtype=np.float64),
                          np.ascontiguousarray(tri, dtype=np.float64))


def hex_scaled_jacobian(coords: np.ndarray) -> np.ndarray:
    """(m, 8, 3) corners -> (m,) scaled Jacobian (compiled)."""
    return _sj_batch(np.ascontiguousarray(coords, dtype=np.float64), CORNER_NEIGHBOURS)


def hex_scaled_jacobian_of(nodes: np.ndarray, hexes: np.ndarray, which: np.ndarray) -> np.ndarray:
    """Scaled Jacobian of hexes[which] without building the (m, 8, 3) corner array."""
    return _sj_indexed(np.ascontiguousarray(nodes, dtype=np.float64),
                       np.ascontiguousarray(hexes, dtype=np.int64),
                       np.ascontiguousarray(which, dtype=np.int64), CORNER_NEIGHBOURS)


def free_node_moves(nodes, hexes, inc, loc, rows, sj, para, lap, scales, weight, levels,
                    opposite=None):
    """(new positions, moved mask) for interior nodes `rows` (one colour: no shared hexes).
    With `opposite` (mesh.hex_quality._FACE_OPPOSITE) and para/lap None, the targets are
    computed in the kernel (same formula as hex_quality._targets)."""
    own = para is None
    if own:
        para = np.empty((len(rows), 3))
        lap = np.empty((len(rows), 3))
        opposite = np.asarray(opposite, dtype=np.int64)
    else:
        opposite = np.zeros((8, 3), dtype=np.int64)
    return _free_moves(np.ascontiguousarray(nodes, dtype=np.float64),
                       np.ascontiguousarray(hexes, dtype=np.int64),
                       np.ascontiguousarray(inc, dtype=np.int64),
                       np.ascontiguousarray(loc, dtype=np.int64),
                       np.ascontiguousarray(rows, dtype=np.int64),
                       np.ascontiguousarray(sj, dtype=np.float64),
                       np.ascontiguousarray(para, dtype=np.float64),
                       np.ascontiguousarray(lap, dtype=np.float64),
                       np.asarray(scales, dtype=np.float64), float(weight),
                       np.asarray(levels, dtype=np.float64), CORNER_NEIGHBOURS,
                       opposite, own)


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _numpy_row_sum(v, k):
        """Sum of v[:k] in NumPy's pairwise order for a contiguous row (numpy pairwise_sum:
        plain loop under 8 values, eight running sums and a fixed tree from 8 up to 128), so
        a compiled mean matches `x.sum(axis=1)` to the last bit."""
        if k < 8:
            res = 0.0
            for i in range(k):
                res += v[i]
            return res
        r0, r1, r2, r3, r4, r5, r6, r7 = v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
        stop = k - (k % 8)
        for i in range(8, stop, 8):
            r0 += v[i]
            r1 += v[i + 1]
            r2 += v[i + 2]
            r3 += v[i + 3]
            r4 += v[i + 4]
            r5 += v[i + 5]
            r6 += v[i + 6]
            r7 += v[i + 7]
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        for i in range(stop, k):
            res += v[i]
        return res

    @njit(cache=True, parallel=True)
    def _local_quality(nodes, hexes, inc, loc, rows, positions, nb):
        n, width = rows.shape[0], inc.shape[1]
        low = np.empty(n)
        mean = np.empty(n)
        each = np.full((n, width), np.nan)
        for t in prange(n):
            r = rows[t]
            c = np.empty((8, 3))
            padded = np.zeros(width)
            lo = np.inf
            count = 0
            for a in range(width):
                h = inc[r, a]
                if h < 0:
                    continue
                for q in range(8):
                    for d in range(3):
                        c[q, d] = nodes[hexes[h, q], d]
                corner = loc[r, a]
                c[corner, 0], c[corner, 1], c[corner, 2] = positions[t, 0], positions[t, 1], positions[t, 2]
                v = _sj8(c, nb)
                if v != v:
                    v = -1.0
                each[t, a] = v
                padded[a] = v
                count += 1
                if v < lo:
                    lo = v
            low[t] = lo
            mean[t] = _numpy_row_sum(padded, width) / max(count, 1)
        return low, mean, each

    @njit(cache=True, parallel=True)
    def _edges_on_surface(cand, nodes, nb, rows_umb, umbrella, tol):
        n, width = cand.shape[0], nb.shape[1]
        ok = np.ones(n, dtype=np.bool_)
        for i in prange(n):
            u = rows_umb[i]
            for j in range(width):
                m = nb[u, j]
                if m < 0:
                    continue
                mx = 0.5 * (cand[i, 0] + nodes[m, 0])
                my = 0.5 * (cand[i, 1] + nodes[m, 1])
                mz = 0.5 * (cand[i, 2] + nodes[m, 2])
                best = 1e300
                bx = by = bz = 0.0
                for s in range(umbrella.shape[1]):
                    qx, qy, qz = _closest_tri(mx, my, mz, umbrella[u, s])
                    d = (qx - mx) ** 2 + (qy - my) ** 2 + (qz - mz) ** 2
                    if d < best:
                        best, bx, by, bz = d, qx, qy, qz
                ex, ey, ez = bx - mx, by - my, bz - mz
                if not np.sqrt(0.0 + ex * ex + ey * ey + ez * ez) <= tol[u]:
                    ok[i] = False
                    break
        return ok


def local_quality(nodes, hexes, inc, loc, rows, positions):
    """mesh.hex_quality._local_quality compiled: (min, mean, per-hex) scaled Jacobian of each
    node's hexes with the node at `positions`, same values to the last bit."""
    return _local_quality(np.ascontiguousarray(nodes, dtype=np.float64),
                          np.ascontiguousarray(hexes, dtype=np.int64),
                          np.ascontiguousarray(inc, dtype=np.int64),
                          np.ascontiguousarray(loc, dtype=np.int64),
                          np.ascontiguousarray(rows, dtype=np.int64),
                          np.ascontiguousarray(positions, dtype=np.float64), CORNER_NEIGHBOURS)


def edges_on_surface(cand, nodes, nb, rows_umb, umbrella, tol):
    """mesh.hex_quality._edges_on_surface compiled, reading the umbrellas by row instead of
    copying umbrella[rows] (the copy was most of its time)."""
    return _edges_on_surface(np.ascontiguousarray(cand, dtype=np.float64),
                             np.ascontiguousarray(nodes, dtype=np.float64),
                             np.ascontiguousarray(nb, dtype=np.int64),
                             np.ascontiguousarray(rows_umb, dtype=np.int64),
                             np.ascontiguousarray(umbrella, dtype=np.float64),
                             np.ascontiguousarray(tol, dtype=np.float64))

def sample_trilinear(vol: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`scipy.ndimage.map_coordinates(vol, idx, order=1, mode="constant", cval=0.0)` for idx of
    shape (3, n); the compiled path handles points inside the array (agreeing to rounding), the
    rest go to SciPy."""
    from scipy import ndimage

    if not HAVE_NUMBA:
        return ndimage.map_coordinates(vol, idx, order=1, mode="constant", cval=0.0)
    idx = np.ascontiguousarray(idx, dtype=np.float64)
    out = np.empty(idx.shape[1], dtype=np.float64)
    _trilinear(np.ascontiguousarray(vol), idx, out)
    bad = np.isnan(out)
    if bad.any():
        out[bad] = ndimage.map_coordinates(vol, idx[:, bad], order=1, mode="constant", cval=0.0)
    return out.astype(vol.dtype) if np.issubdtype(vol.dtype, np.floating) else out
