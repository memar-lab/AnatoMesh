"""Conforming multi-material hexahedral mesh generation from a label volume."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

HEX_CORNER_OFFSETS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)

HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)


def _unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`np.unique(rows, axis=0, return_inverse=True, return_counts=True)[1:]`, faster."""
    k, cols = rows.shape
    if k == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    bits = int(int(rows.max()) + 1).bit_length()
    if cols % 2 or bits * 2 > 62:
        _u, inverse, counts = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
        return inverse.ravel(), counts
    keys = [(rows[:, i].astype(np.int64) << bits) | rows[:, i + 1]
            for i in range(0, cols, 2)]
    order = np.lexsort(tuple(reversed(keys)))  # last key is primary
    boundary = np.zeros(k, dtype=bool)
    boundary[0] = True
    if k > 1:
        differs = np.zeros(k - 1, dtype=bool)
        for key in keys:
            sorted_key = key[order]
            differs |= sorted_key[1:] != sorted_key[:-1]
        boundary[1:] = differs
    group = np.cumsum(boundary) - 1
    inverse = np.empty(k, dtype=np.int64)
    inverse[order] = group
    return inverse, np.bincount(group)


def _corner_order_for(affine: np.ndarray) -> np.ndarray:
    """Corner offsets wound so hexes come out with a POSITIVE Jacobian."""
    if np.linalg.det(affine[:3, :3]) >= 0:
        return HEX_CORNER_OFFSETS
    swap = [0, 3, 2, 1, 4, 7, 6, 5]
    return HEX_CORNER_OFFSETS[swap]


def conforming_hex_mesh(
    labels: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One conforming all-hex mesh over every material in `labels` at once."""
    occ = labels != 0
    if not occ.any():
        return (np.zeros((0, 3)), np.zeros((0, 8), dtype=np.int64),
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=bool))

    idx = np.argwhere(occ)
    lo = idx.min(axis=0) - 1  # background pad
    hi = idx.max(axis=0) + 2
    pad_lo = np.maximum(-lo, 0)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(hi, np.array(labels.shape))
    sub = np.zeros(tuple(hi - lo), dtype=np.int64)
    sub[pad_lo[0]:pad_lo[0] + (src_hi[0] - src_lo[0]),
        pad_lo[1]:pad_lo[1] + (src_hi[1] - src_lo[1]),
        pad_lo[2]:pad_lo[2] + (src_hi[2] - src_lo[2])] = \
        labels[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]

    occ_sub = sub != 0
    nx, ny, nz = sub.shape

    used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                used[dx:dx + nx, dy:dy + ny, dz:dz + nz] |= occ_sub

    node_id = np.full(used.shape, -1, dtype=np.int64)
    n_nodes = int(used.sum())
    node_id[used] = np.arange(n_nodes, dtype=np.int64)

    corner_offsets = _corner_order_for(affine)
    vi, vj, vk = np.nonzero(occ_sub)
    hexes = np.stack(
        [node_id[vi + dx, vj + dy, vk + dz] for dx, dy, dz in corner_offsets], axis=1)
    hex_labels = sub[vi, vj, vk]

    label_bits = np.zeros(used.shape, dtype=np.int64)
    for value in np.unique(sub):
        vmask = sub == value
        touch = np.zeros(used.shape, dtype=bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    touch[dx:dx + nx, dy:dy + ny, dz:dz + nz] |= vmask
        label_bits[touch] |= np.int64(1) << np.int64(value)

    node_bits = label_bits[used]
    interface = np.zeros(n_nodes, dtype=bool)
    b = node_bits.copy()
    first_cleared = b & (b - 1)
    interface[first_cleared != 0] = True

    coords = np.argwhere(used) + lo - 0.5
    world = coords @ affine[:3, :3].T + affine[:3, 3]
    return world, hexes, hex_labels, node_bits, interface


def interface_quads(
    hexes: np.ndarray, hex_labels: np.ndarray, inner: set[int], outer: set[int],
    return_owners: bool = False,
):
    """Quad faces where material `inner` meets material `outer`."""
    faces = np.concatenate([hexes[:, f] for f in HEX_FACES], axis=0)
    owner = np.tile(hex_labels, len(HEX_FACES))
    key = np.sort(faces, axis=1)
    inverse, counts = _unique_rows(key)

    total = np.zeros(len(counts), dtype=np.int64)
    np.add.at(total, inverse, owner)
    partner = np.where(counts[inverse] == 2, total[inverse] - owner, 0)

    inner_arr = np.array(sorted(inner), dtype=np.int64)
    outer_arr = np.array(sorted(outer), dtype=np.int64)
    sel = np.isin(owner, inner_arr) & np.isin(partner, outer_arr)
    if return_owners:
        return faces[sel], owner[sel]
    return faces[sel]


def drop_nonmanifold_hexes(
    hexes: np.ndarray, hex_labels: np.ndarray, max_passes: int = 32
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove hexes that leave the boundary surface non-manifold."""
    keep = np.ones(len(hexes), dtype=bool)
    dropped = 0
    for _ in range(max_passes):
        live = hexes[keep]
        if not len(live):
            break
        faces = np.concatenate([live[:, f] for f in HEX_FACES], axis=0)
        owner = np.tile(np.nonzero(keep)[0], len(HEX_FACES))
        key = np.sort(faces, axis=1)
        inverse, counts = _unique_rows(key)

        single = counts[inverse] == 1
        bfaces, bowner = faces[single], owner[single]
        edges = np.concatenate([bfaces[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
        eowner = np.tile(bowner, 4)
        ekey = np.sort(edges, axis=1)
        einv, ecounts = _unique_rows(ekey)
        bad = ecounts[einv] != 2
        if not bad.any():
            break
        guilty = np.unique(eowner[bad])
        keep[guilty] = False
        dropped += len(guilty)
    else:
        logger.warning("boundary surface is still non-manifold after %d passes; "
                       "the mesh will fail the structural gate", max_passes)
    if dropped:
        level = logger.warning if dropped > 0.02 * len(hexes) else logger.info
        level("dropped %d non-manifold hex(es) of %d (%.2f%%) to make the boundary "
              "surface manifold", dropped, len(hexes), 100.0 * dropped / len(hexes))
    return hexes[keep], hex_labels[keep], dropped


def conformity_report(nodes: np.ndarray, hexes: np.ndarray,
                       on_lattice: bool = True) -> dict:
    """Check that the mesh is a conforming hex complex."""
    if not len(hexes):
        return {"n_nodes": 0, "n_hexes": 0, "n_faces_interior": 0,
                "n_faces_boundary": 0, "n_faces_overshared": 0,
                "n_boundary_edges_unpaired": 0, "n_hanging_nodes": 0,
                "n_hanging_nodes_on_lattice": 0, "conforming": True}
    faces = np.concatenate([hexes[:, f] for f in HEX_FACES], axis=0)
    key = np.sort(faces, axis=1)
    inverse, counts = _unique_rows(key)
    n_over = int((counts > 2).sum())

    single = counts[inverse] == 1
    bfaces = faces[single]
    bedges = np.concatenate(
        [bfaces[:, [i, (i + 1) % 4]] for i in range(4)], axis=0)
    _beinv, be_counts = _unique_rows(np.sort(bedges, axis=1))
    n_unpaired = int((be_counts != 2).sum())

    if not on_lattice:
        return {
            "n_nodes": int(len(nodes)),
            "n_hexes": int(len(hexes)),
            "n_faces_interior": int((counts == 2).sum()),
            "n_faces_boundary": int((counts == 1).sum()),
            "n_faces_overshared": n_over,
            "n_boundary_edges_unpaired": n_unpaired,
            "n_hanging_nodes": None,
            "conforming": bool(n_over == 0 and n_unpaired == 0),
        }

    scale = 1.0
    sample = hexes[:min(len(hexes), 20000)]
    diffs = np.abs(nodes[sample[:, 1]] - nodes[sample[:, 0]])
    positive = diffs[diffs > 1e-9]
    if positive.size:
        scale = float(np.median(positive)) / 8.0

    lo = np.round(nodes / scale).astype(np.int64).min(axis=0)
    span = (np.round(nodes / scale).astype(np.int64) - lo).max(axis=0) + 1

    def _pack_at(points: np.ndarray) -> np.ndarray:
        q = np.round(points / scale).astype(np.int64) - lo
        inside = ((q >= 0) & (q < span)).all(axis=1)
        packed = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
        return np.where(inside, packed, -1)

    node_keys = np.sort(_pack_at(nodes))
    hanging_keys: list[np.ndarray] = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)):
        p0, p1 = nodes[hexes[:, a]], nodes[hexes[:, b]]
        for t in (0.25, 0.5, 0.75):
            keys = _pack_at(p0 + (p1 - p0) * t)
            pos = np.searchsorted(node_keys, keys)
            pos = np.clip(pos, 0, len(node_keys) - 1)
            hit = (node_keys[pos] == keys) & (keys >= 0)
            if hit.any():
                hanging_keys.append(keys[hit])
    hanging = np.unique(np.concatenate(hanging_keys)) if hanging_keys else np.zeros(0)

    return {
        "n_nodes": int(len(nodes)),
        "n_hexes": int(len(hexes)),
        "n_faces_interior": int((counts == 2).sum()),
        "n_faces_boundary": int((counts == 1).sum()),
        "n_faces_overshared": n_over,
        "n_boundary_edges_unpaired": n_unpaired,
        "n_hanging_nodes": int(len(hanging)),
        "conforming": bool(n_over == 0 and n_unpaired == 0),
    }
