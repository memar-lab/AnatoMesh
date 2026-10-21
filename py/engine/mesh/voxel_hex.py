"""Voxel hex mesher with Laplacian smoothing and targeted element repair."""

from __future__ import annotations

import heapq
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# numpy's det and scipy's distance transforms release the GIL, so big arrays split across threads
THREADS = int(os.environ.get("ANATOMESH_THREADS", min(8, os.cpu_count() or 1)))
_CHUNK = 100_000

JACOBIAN_THRESHOLD = 0.5
LAMBDA_VALUE = 0.3
MU_VALUE = -0.45  # declared in the reference implementation, never used
LAYER_INCREMENTS = (0.1, 0.2, 0.3, 0.4, 0.5)
MAX_ADJUST_ITERATIONS = 20
ADJUST_TOLERANCE = 0.02
SHELL_PART_ID = 102
REF_DIST = float(np.linalg.norm(np.zeros(3) - np.full(3, 0.5)))

_FACES = np.array([[1, 4, 3, 2], [5, 6, 7, 8], [4, 8, 7, 3],
                   [1, 2, 6, 5], [1, 5, 8, 4], [2, 3, 7, 6]]) - 1


def _edges(c: np.ndarray) -> np.ndarray:
    """The twelve edge vectors `L` of the element quality metrics, shape `(n, 12, 3)`."""
    return np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 1], c[:, 3] - c[:, 2], c[:, 3] - c[:, 0],
                     c[:, 4] - c[:, 0], c[:, 5] - c[:, 1], c[:, 6] - c[:, 2], c[:, 7] - c[:, 3],
                     c[:, 5] - c[:, 4], c[:, 6] - c[:, 5], c[:, 7] - c[:, 6], c[:, 7] - c[:, 4]],
                    axis=1)


def _metric_matrices(c: np.ndarray) -> np.ndarray:
    """The nine 3x3 matrices of the element quality metrics for coords `(n, 8, 3)`."""
    L = _edges(c)
    x1 = (c[:, 1] - c[:, 0]) + (c[:, 2] - c[:, 3]) + (c[:, 5] - c[:, 4]) + (c[:, 6] - c[:, 7])
    x2 = (c[:, 3] - c[:, 0]) + (c[:, 2] - c[:, 1]) + (c[:, 7] - c[:, 4]) + (c[:, 6] - c[:, 5])
    x3 = (c[:, 4] - c[:, 0]) + (c[:, 5] - c[:, 1]) + (c[:, 6] - c[:, 2]) + (c[:, 7] - c[:, 3])
    rows = ((L[:, 0], L[:, 3], L[:, 4]),
            (L[:, 1], -L[:, 0], L[:, 5]),
            (L[:, 2], -L[:, 1], L[:, 6]),
            (-L[:, 3], -L[:, 2], L[:, 7]),
            (L[:, 11], L[:, 8], -L[:, 4]),
            (-L[:, 8], L[:, 9], -L[:, 5]),
            (-L[:, 9], L[:, 10], -L[:, 6]),
            (-L[:, 10], -L[:, 11], -L[:, 7]),
            (x1, x2, x3))
    return np.stack([np.stack(r, axis=1) for r in rows], axis=1)


_POOL: ThreadPoolExecutor | None = None


def pool() -> ThreadPoolExecutor:
    """One shared worker pool for per-element work split into row chunks."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(THREADS)
    return _POOL


def rows_in_parallel(function, rows: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """`function(rows)` for a per-row function, split across threads when `rows` is big."""
    if len(rows) <= 2 * chunk or THREADS < 2:
        return function(rows)
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    return np.concatenate(list(pool().map(function, parts)))


def _by_chunks(function, c: np.ndarray):
    """Apply a per-element function to `(n, 8, 3)` coords in parallel row chunks."""
    if len(c) <= _CHUNK or THREADS < 2:
        return function(c)
    parts = [c[i:i + _CHUNK] for i in range(0, len(c), _CHUNK)]
    with ThreadPoolExecutor(THREADS) as pool:
        results = list(pool.map(function, parts))
    if isinstance(results[0], tuple):
        return tuple(np.concatenate(r) for r in zip(*results))
    return np.concatenate(results)


def compute_metrics(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the element quality metrics for `(n, 8, 3)` coords"""
    from engine.mesh import fast_kernels

    if fast_kernels.HAVE_NUMBA:     # same numbers, compiled (mesh.fast_kernels)
        return fast_kernels.reference_metrics(coords)
    return _by_chunks(_compute_metrics, np.asarray(coords, dtype=np.float64))


def _compute_metrics(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jac = _scaled_jacobian(c)
    L = _edges(c)
    lengths = np.linalg.norm(L, axis=2)
    D = np.stack([c[:, 6] - c[:, 0], c[:, 7] - c[:, 1], c[:, 4] - c[:, 2], c[:, 5] - c[:, 3]],
                 axis=1)
    diag = np.linalg.norm(D, axis=2)
    return jac, lengths.max(1) / lengths.min(1), diag.min(1) / diag.max(1)


def scaled_jacobian(coords: np.ndarray) -> np.ndarray:
    """Only the Jacobian of `compute_metrics`."""
    from engine.mesh import fast_kernels

    if fast_kernels.HAVE_NUMBA:
        return fast_kernels.reference_scaled_jacobian(coords)
    return _by_chunks(_scaled_jacobian, np.asarray(coords, dtype=np.float64))


def _scaled_jacobian(c: np.ndarray) -> np.ndarray:
    A = _metric_matrices(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        dets = np.linalg.det(A / np.sqrt((A * A).sum(axis=2, keepdims=True)))
    finite = np.where(np.isnan(dets), np.inf, dets)
    return np.where(np.isnan(dets).all(axis=1), np.nan, finite.min(axis=1))


def adjust_hex_coords(coords: np.ndarray, lambda_value: float = LAMBDA_VALUE,
                      ref_dist: float = REF_DIST) -> np.ndarray:
    """the element reshaping"""
    adjusted = coords.copy()
    centroid = adjusted.mean(axis=0)
    scale = lambda_value + 0.05
    for i in range(8):
        v = adjusted[i] - centroid
        d = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if d > ref_dist:
            alpha = -scale * (d - ref_dist) / ref_dist
        else:
            alpha = +scale * (ref_dist - d) / ref_dist
        adjusted[i] = adjusted[i] + alpha * v
    return adjusted


def _column_major_voxels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0-based (row, col, depth) of `find(grid)` in the reference implementation's column-major order."""
    d, c, r = np.nonzero(np.transpose(grid, (2, 1, 0)))
    return r, c, d


def generate_mesh(labels: np.ndarray, iterations: int = 3, *,
                  part_labels: np.ndarray | None = None,
                  lambda_value: float = LAMBDA_VALUE,
                  targeted_adjustment: bool = True,
                  keep_odd_passes: bool = True,
                  repair_every_element: bool = False,
                  jacobian_threshold: float = JACOBIAN_THRESHOLD,
                  target_min_jacobian: float | None = None,
                  max_iterations: int = 10,
                  progress: Callable[[str], None] | None = None) -> dict:
    """Mesh a label volume the way the voxel mesher does."""
    say = progress or (lambda message: logger.info(message))
    data = np.asarray(labels).copy()
    data[data == 50] = 5
    grid = data > 0
    n_rows, n_cols, n_depth = grid.shape
    plane = (n_cols + 1) * (n_depth + 1)
    n_nodes = (n_rows + 1) * plane

    index = np.arange(n_nodes)
    X = index // plane
    Y = (index // (n_depth + 1)) % (n_cols + 1)
    Z = index % (n_depth + 1)
    nodes = np.stack([X, Y, Z], axis=1).astype(np.float64)

    neighbors = np.zeros((n_nodes, 6), dtype=np.int32 if n_nodes < 2**31 - 1 else np.int64)
    for k, (dx, dy, dz) in enumerate(((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                                      (0, 1, 0), (0, 0, -1), (0, 0, 1))):
        xx, yy, zz = X + dx, Y + dy, Z + dz
        ok = ((xx >= 0) & (xx <= n_rows) & (yy >= 0) & (yy <= n_cols)
              & (zz >= 0) & (zz <= n_depth))
        neighbors[ok, k] = xx[ok] * plane + yy[ok] * (n_depth + 1) + zz[ok] + 1

    rows, cols, depths = _column_major_voxels(grid)
    base = rows * plane + cols * (n_depth + 1) + depths + 1
    offsets = np.array([0, plane, plane + (n_depth + 1), n_depth + 1,
                        1, plane + 1, plane + (n_depth + 1) + 1, (n_depth + 1) + 1])
    elements = base[:, None] + offsets[None, :]
    values = data[rows, cols, depths]
    if part_labels is None:
        part_ids = values.copy()
    else:
        parts_volume = np.asarray(part_labels).copy()
        parts_volume[parts_volume == 50] = 5
        if not np.array_equal(parts_volume > 0, grid):
            raise ValueError("part_labels must be nonzero on exactly the voxels labels is")
        part_ids = parts_volume[rows, cols, depths]
    say(f"{len(elements)} elements on a {n_rows}x{n_cols}x{n_depth} lattice")

    node_class_1 = np.ones(n_nodes + 1, dtype=np.int64)
    unique_values = np.unique(values)
    label_count = np.zeros(n_nodes + 1, dtype=np.int64)
    for value in unique_values:
        label_count[np.unique(elements[values == value])] += 1
    node_class_1[label_count > 1] = 2
    occurrences = np.bincount(elements.ravel(), minlength=n_nodes + 1)
    node_class_1[(occurrences > 0) & (occurrences < 8)] = 3
    node_class_1[occurrences == 0] = 0
    node_class = node_class_1[1:]  # view

    neighbor_class = np.where(neighbors != 0, node_class_1[neighbors], 0)
    use = (neighbor_class == node_class[:, None]) & (node_class[:, None] != 0)
    use[node_class == 1, :] = True
    count = use.sum(axis=1)
    movable = count > 0
    count[count == 0] = 1

    boundary = np.isin(node_class, (2, 3))
    last_group = values == unique_values[-1]
    stale_limit = int(boundary[elements[last_group] - 1].any(axis=1).sum())

    touches = (neighbor_class >= 2).any(axis=1)
    for layer in range(101, 106):
        if layer > 101:
            neighbor_class = np.where(neighbors != 0, node_class_1[neighbors], 0)
            touches = (neighbor_class == layer - 1).any(axis=1)
        node_class[(node_class == 1) & touches] = layer
    weight_odd = np.full(n_nodes, lambda_value)
    for layer, increment in enumerate(LAYER_INCREMENTS, start=101):
        weight_odd[node_class == layer] = lambda_value + increment
    weight_even = weight_odd.copy()
    weight_even[(node_class == 2) | (node_class == 3)] = 0

    gather = np.where(use, neighbors - 1, 0)
    mask = use.astype(np.float64)
    gather_movable, mask_movable, count_movable = gather[movable], mask[movable], count[movable]

    order = np.argsort(elements.ravel(), kind="stable")
    incidence_elements = order // 8
    incidence_start = np.searchsorted(elements.ravel()[order], np.arange(n_nodes + 2))

    nodes_per_iteration: dict[int, np.ndarray] = {}
    stats: dict[int, dict] = {}
    last_step, step = 2 * iterations, 0
    while True:
        step += 1
        if step > last_step:
            if (target_min_jacobian is None or last_step >= 2 * max_iterations
                    or stats[last_step]["jacobian_min"] >= target_min_jacobian):
                break
            last_step += 2
            say(f"scaled Jacobian {stats[last_step - 2]['jacobian_min']:.4f} is below the target "
                f"{target_min_jacobian:.2f}: smoothing iteration {last_step // 2}")
        weight = weight_odd if step % 2 == 1 else weight_even
        updated = nodes.copy()
        for axis in range(3):
            layer = nodes[gather_movable, axis] * mask_movable
            total = layer[:, 0] + layer[:, 1] + layer[:, 2] + layer[:, 3] + layer[:, 4] + layer[:, 5]
            mean = total / count_movable
            updated[movable, axis] = ((1 - weight[movable]) * nodes[movable, axis]
                                      + weight[movable] * mean)
        nodes = updated
        adjusted = 0
        if step % 2 == 1 and targeted_adjustment:
            limit = len(elements) if step == 1 or repair_every_element else stale_limit
            adjusted = _targeted_adjustment(nodes, elements, limit, lambda_value,
                                            incidence_elements, incidence_start,
                                            jacobian_threshold)
        if keep_odd_passes or step % 2 == 0:
            nodes_per_iteration[step] = nodes.copy()
        jac, edge, diag = compute_metrics(nodes[elements - 1])
        stats[step] = {"adjusted_elements": adjusted,
                       "jacobian_min": float(np.nanmin(jac)), "jacobian_max": float(np.nanmax(jac)),
                       "jacobian_mean": float(np.nanmean(jac)),
                       "edge_ratio_min": float(edge.min()), "edge_ratio_max": float(edge.max()),
                       "edge_ratio_mean": float(edge.mean()),
                       "diagonal_ratio_min": float(diag.min()),
                       "diagonal_ratio_max": float(diag.max()),
                       "diagonal_ratio_mean": float(diag.mean())}
        say(f"iteration {step}: scaled Jacobian min {stats[step]['jacobian_min']:.4f} "
            f"mean {stats[step]['jacobian_mean']:.4f}, {adjusted} elements adjusted")

    return {"elements": elements, "part_ids": part_ids, "node_class": node_class.copy(),
            "nodes_per_iteration": nodes_per_iteration, "stats": stats,
            "shape": grid.shape}


def _targeted_adjustment(nodes: np.ndarray, elements: np.ndarray, limit: int,
                         lambda_value: float, incidence_elements: np.ndarray,
                         incidence_start: np.ndarray,
                         threshold: float = JACOBIAN_THRESHOLD) -> int:
    """The reference implementation's sequential `for elem = 1:numElements` repair, in place."""
    jac = scaled_jacobian(nodes[elements[:limit] - 1])
    heap = np.flatnonzero(jac < threshold).tolist()
    heapq.heapify(heap)
    queued = set(heap)
    adjusted = 0
    while heap:
        element = heapq.heappop(heap)
        queued.discard(element)
        node_index = elements[element] - 1
        coords = nodes[node_index].copy()
        jac_old = scaled_jacobian(coords[None])[0]
        if not jac_old < threshold:
            continue
        adjusted += 1
        for _ in range(MAX_ADJUST_ITERATIONS):
            candidate = adjust_hex_coords(coords, lambda_value)
            jac_new = scaled_jacobian(candidate[None])[0]
            converged = jac_new >= threshold or abs(jac_new - jac_old) < ADJUST_TOLERANCE
            coords, jac_old = candidate, jac_new
            if converged:
                break
        nodes[node_index] = coords

        touched = np.unique(np.concatenate(
            [incidence_elements[incidence_start[n + 1]:incidence_start[n + 2]] for n in node_index]))
        touched = touched[(touched > element) & (touched < limit)]
        if len(touched):
            for other, value in zip(touched.tolist(),
                                    scaled_jacobian(nodes[elements[touched] - 1]).tolist()):
                if value < threshold and other not in queued:
                    heapq.heappush(heap, other)
                    queued.add(other)
    return adjusted


def surface_faces(elements: np.ndarray, node_class: np.ndarray) -> np.ndarray:
    """Outer quads"""
    faces = np.concatenate([elements[:, face] for face in _FACES], axis=0)
    faces = faces[(node_class[faces - 1] == 3).all(axis=1)]
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def write_key_file(path: Path, nodes: np.ndarray, elements: np.ndarray,
                   part_ids: np.ndarray, shells: np.ndarray, scale: float = 1.0) -> Path:
    """Write one `Key_File_Iter_<n>.k` in the reference implementation's exact layout."""
    referenced = np.unique(elements)
    coordinates = nodes[referenced - 1]
    if scale != 1:
        coordinates = coordinates * scale
    new_id = np.zeros(len(nodes) + 1, dtype=np.int64)
    new_id[referenced] = np.arange(1, len(referenced) + 1)
    solids = np.column_stack([np.arange(1, len(elements) + 1),
                              np.asarray(part_ids, dtype=np.int64), new_id[elements]])
    first_shell = len(elements) + 1
    shell_rows = np.column_stack([np.arange(first_shell, first_shell + len(shells)),
                                  np.full(len(shells), SHELL_PART_ID), new_id[shells],
                                  np.zeros((len(shells), 4), dtype=np.int64)])
    header = "$#   eid     pid      n1      n2      n3      n4      n5      n6      n7      n8\n"
    row = "%8d%8d%8d%8d%8d%8d%8d%8d%8d%8d\n"
    with open(path, "w", newline="\n") as handle:
        handle.write("*KEYWORD\n*ELEMENT_SOLID\n" + header)
        handle.write("".join(row % tuple(r) for r in solids.tolist()))
        handle.write("*ELEMENT_SHELL\n" + header)
        handle.write("".join(row % tuple(r) for r in shell_rows.tolist()))
        handle.write("*NODE\n$#   nid               x               y               z      tc      rc\n")
        handle.write("".join("%8d%16.6f%16.6f%16.6f%8d%8d\n" % (i, x, y, z, 0, 0)
                             for i, (x, y, z) in enumerate(coordinates.tolist(), 1)))
        handle.write("*END\n")
    return Path(path)


def write_key_files(result: dict, out_dir: Path,
                    path_guard: Callable[[Path], Path] = lambda p: p,
                    scale: float = 1.0) -> dict[int, Path]:
    """Write a key file for every even pass, as the reference implementation's final loop does."""
    shells = surface_faces(result["elements"], result["node_class"])
    written = {}
    for step, nodes in sorted(result["nodes_per_iteration"].items()):
        if step % 2:
            continue
        path = path_guard(Path(out_dir) / f"Key_File_Iter_{step}.k")
        written[step] = write_key_file(path, nodes, result["elements"], result["part_ids"],
                                       shells, scale)
    return written
