"""Voxel hex mesh: smoothing, targeted element adjustment, quality metrics, report, .k export."""

import gzip
import heapq
import math
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np


# =============================================================================================
# SETTINGS: change these
# =============================================================================================


DEFAULT_ITERATIONS = 3                # Smoothing Iterations (the loop runs 2 * Iterations steps)
DEFAULT_LAMBDA = 0.3                  # Lambda_value of the reference implementation
DEFAULT_TARGETED = True               # TargetedElementAdjustment
DEFAULT_LEGACY_LIMIT = True           # True (default) = same as the reference implementation: from the 2nd pass on only the first N elements are visited; False = repair all elements
JACOBIAN_THRESHOLD = 0.5              # elements below this scaled Jacobian are reshaped (jacobianThreshold)
MAX_ADJUST_ITERATIONS = 20            # per element (max_iterations)
CONVERGENCE_REL_DIFF = 0.02           # rel_diff < 0.02 stops the reshaping of an element
DELTA_T_CONSTANT = 2                  # delta_t_constant
SHELL_PART_ID = 102                   # part id of the surface shell elements in the .k file
INNER_LAYER_STEP = 0.1                # layers 101..105 use Lambda + 0.1, 0.2, ... 0.5
KEEP_LEGACY_DIAGONAL_PLUS_ONE = True  # the reference implementation adds 1 to the longest diagonal (triu(ones)); True reproduces it exactly

GO_BLUE, GO_BLUE_HOVER, GO_BLUE_BUSY = "#8fd0f5", "#a8dcf8", "#d3e9f7"      # GENERATE MESH button: idle, hover, running
                                      # (lower = smoother rotation but blockier while moving; the full mesh returns on release)

# =============================================================================================
# NIfTI reading (NIfTI-1, .nii or .nii.gz)
# =============================================================================================

DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8,
          512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def read_nifti(path):
    """Returns (data in storage order, voxel size (3,) in mm)."""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read()
    for endian in ("<", ">"):
        if struct.unpack(endian + "i", raw[:4])[0] == 348:
            break
    else:
        raise ValueError("Not a NIfTI-1 file (header size is not 348).")
    dim = struct.unpack(endian + "8h", raw[40:56])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    if datatype not in DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype code {datatype}.")
    shape = tuple(dim[1:1 + dim[0]])
    if dim[0] != 3 and not (dim[0] > 3 and all(s == 1 for s in shape[3:])):
        raise ValueError(f"Expected a 3D volume, found dim = {dim[:dim[0] + 1]}.")
    shape = shape[:3]
    dtype = np.dtype(DTYPES[datatype]).newbyteorder(endian)
    data = np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape)), offset=vox_offset).reshape(shape, order="F")
    return np.ascontiguousarray(data.astype(dtype.newbyteorder("="), copy=False)), np.array(pixdim[1:4], dtype=float)


# =============================================================================================
# Hexahedral element quality (the element quality metrics), vectorised over elements
# =============================================================================================

# face_definitions of the reference implementation (1-based there), corner order = cube corners 000 100 110 010 001 101 111 011
FACE_DEFS = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [3, 7, 6, 2], [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5]])
EDGE_PAIRS = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [4, 7]])
DIAGONAL_PAIRS = np.array([[0, 6], [1, 7], [2, 4], [3, 5]])
ALL_PAIRS = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)])


def scaled_jacobian(c):
    """the element quality metrics 'Jacobian' for coordinates (n, 8, 3): minimum over 9 normalised determinants."""
    L = np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 1], c[:, 3] - c[:, 2], c[:, 3] - c[:, 0],
                  c[:, 4] - c[:, 0], c[:, 5] - c[:, 1], c[:, 6] - c[:, 2], c[:, 7] - c[:, 3],
                  c[:, 5] - c[:, 4], c[:, 6] - c[:, 5], c[:, 7] - c[:, 6], c[:, 7] - c[:, 4]], axis=1)
    x1 = (c[:, 1] - c[:, 0]) + (c[:, 2] - c[:, 3]) + (c[:, 5] - c[:, 4]) + (c[:, 6] - c[:, 7])
    x2 = (c[:, 3] - c[:, 0]) + (c[:, 2] - c[:, 1]) + (c[:, 7] - c[:, 4]) + (c[:, 6] - c[:, 5])
    x3 = (c[:, 4] - c[:, 0]) + (c[:, 5] - c[:, 1]) + (c[:, 6] - c[:, 2]) + (c[:, 7] - c[:, 3])
    rows = ((L[:, 0], L[:, 3], L[:, 4]), (L[:, 1], -L[:, 0], L[:, 5]), (L[:, 2], -L[:, 1], L[:, 6]),
            (-L[:, 3], -L[:, 2], L[:, 7]), (L[:, 11], L[:, 8], -L[:, 4]), (-L[:, 8], L[:, 9], -L[:, 5]),
            (-L[:, 9], L[:, 10], -L[:, 6]), (-L[:, 10], -L[:, 11], -L[:, 7]), (x1, x2, x3))
    A = np.stack([np.stack(r, axis=1) for r in rows], axis=1)          # (n, 9, 3, 3), rows = vectors
    with np.errstate(invalid="ignore", divide="ignore"):
        dets = np.linalg.det(A / np.sqrt((A * A).sum(axis=2, keepdims=True)))      # columns normalised
    return np.nanmin(np.where(np.isnan(dets), np.inf, dets), axis=1)


def hex_volume(c):
    """Volume of the convex hull of the 8 corners (convhull in the reference implementation).
    Centroid fan: for every quad face both triangulations are tried and the larger (outer) one is kept."""
    centre = c.mean(axis=1)
    total = np.zeros(len(c))
    for q in FACE_DEFS:
        p = [c[:, i] - centre for i in q]
        tet = lambda a, b, d: np.abs(np.einsum("ij,ij->i", a, np.cross(b, d))) / 6.0
        total += np.maximum(tet(p[0], p[1], p[2]) + tet(p[0], p[2], p[3]),
                            tet(p[0], p[1], p[3]) + tet(p[1], p[2], p[3]))
    return total


def element_quality(c):
    """Scaled Jacobian, edge ratio, diagonal ratio, volume, characteristic length and time step of (n, 8, 3) elements."""
    jac = scaled_jacobian(c)
    edges = np.linalg.norm(c[:, EDGE_PAIRS[:, 1]] - c[:, EDGE_PAIRS[:, 0]], axis=2)
    diagonals = np.linalg.norm(c[:, DIAGONAL_PAIRS[:, 1]] - c[:, DIAGONAL_PAIRS[:, 0]], axis=2)
    volume = hex_volume(c)
    longest = np.linalg.norm(c[:, ALL_PAIRS[:, 1]] - c[:, ALL_PAIRS[:, 0]], axis=2).max(axis=1)
    if KEEP_LEGACY_DIAGONAL_PLUS_ONE:
        longest = longest + 1.0
    characteristic = volume * math.sqrt(2) / (3 * 1000 * longest ** 2)
    return dict(jacobian=jac, edge_ratio=edges.max(axis=1) / edges.min(axis=1),
                diagonal_ratio=diagonals.min(axis=1) / diagonals.max(axis=1), volume=volume,
                characteristic=characteristic, time_step=characteristic / DELTA_T_CONSTANT)


def quality_of(nodes, elements, chunk=100000):
    parts = [element_quality(nodes[elements[i:i + chunk]]) for i in range(0, len(elements), chunk)]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def adjust_hex(c, ref_dist, lam):
    """the element reshaping: pull / push every corner toward the reference cube distance from the centroid."""
    centre = c.mean(axis=0)
    v = c - centre
    d = np.linalg.norm(v, axis=1)
    scale = lam + 0.05
    alpha = np.where(d > ref_dist, -scale * (d - ref_dist) / ref_dist, scale * (ref_dist - d) / ref_dist)
    return c + alpha[:, None] * v


# =============================================================================================
# Optional gap sealing (seal_layers_standalone.py)
# =============================================================================================


def seal_layers(labels, brain, outer_layers):
    """Empty voxels touching anything inside a layer become that layer, layer after layer (innermost first).
    26-connected: faces, edges and corners count. Layers that are not in the volume are skipped.
    Returns (sealed labels, {layer: voxels added})."""
    from scipy import ndimage
    out = np.asarray(labels).copy()
    present = set(np.unique(out).tolist())
    inner = set(brain)
    cube = np.ones((3, 3, 3), dtype=bool)
    filled = {}
    for layer in outer_layers:
        if layer not in present:
            continue
        exposed = (out == 0) & ndimage.binary_dilation(np.isin(out, sorted(inner)), cube)
        out[exposed] = layer
        filled[layer] = int(exposed.sum())
        inner.add(layer)
    return out, filled


def exposed_brain_voxels(labels, brain):
    """Brain voxels that have empty space among their 26 neighbours."""
    from scipy import ndimage
    empty = ndimage.binary_dilation(labels == 0, np.ones((3, 3, 3), dtype=bool))
    return int((empty & np.isin(labels, list(brain))).sum())


def parse_labels(text):
    return tuple(int(v) for v in text.replace(",", " ").split())


# =============================================================================================
# Mesh: elements, node classes, neighbours
# =============================================================================================


class Mesh:
    """Voxel mesh. Elements follow the reference implementation's find(grid) order (x fastest) and node ids the ndgrid order (z fastest)."""

    def __init__(self, data, spacing):
        data = data.copy()
        data[data == 50] = 5                                       # "Change voxels with value 50 to 5"
        self.shape = data.shape
        self.spacing = np.asarray(spacing, float)
        self.cube = float(self.spacing.mean())                     # cube_size
        nx, ny, nz = self.shape
        self.lattice = (nx + 1, ny + 1, nz + 1)
        self.N = int(np.prod(self.lattice))
        flat = np.flatnonzero((data > 0).ravel(order="F"))
        ix, iy, iz = np.unravel_index(flat, self.shape, order="F")
        self.ijk = np.stack([ix, iy, iz], axis=1)
        self.labels = data[ix, iy, iz].astype(np.int32)           # elementsValue
        s1, s2 = (ny + 1) * (nz + 1), nz + 1
        base = ix * s1 + iy * s2 + iz
        offsets = np.array([0, s1, s1 + s2, s2, 1, s1 + 1, s1 + s2 + 1, s2 + 1])
        self.E = (base[:, None] + offsets[None, :]).astype(np.int64)     # 0-based node ids, n x 8
        self.n = len(self.E)
        X, Y, Z = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
        self.nodes0 = np.stack([X.ravel() * self.spacing[0], Y.ravel() * self.spacing[1],
                                Z.ravel() * self.spacing[2]], axis=1)    # lattice scaled to mm
        self.region_grid = self.region_labels = None                  # original regions (set by generate when step 1 combined)
        self.label_grid = data * (data > 0)
        self.element_grid = -np.ones(self.shape, dtype=np.int64)
        self.element_grid[ix, iy, iz] = np.arange(self.n)
        self.classify()

    def classify(self):
        """NodeClass 0 / 1 / 2 / 3 exactly as in the reference implementation."""
        count = np.bincount(self.E.ravel(), minlength=self.N)             # elements touching each node
        parts = np.zeros(self.N, dtype=np.int16)
        for label in np.unique(self.labels):
            touched = np.zeros(self.N, dtype=bool)
            touched[self.E[self.labels == label].ravel()] = True
            parts += touched
        cls = np.ones(self.N, dtype=np.int16)
        cls[parts > 1] = 2                                                # shared by two regions
        cls[(count > 0) & (count < 8)] = 3                                # touches the background
        cls[count == 0] = 0                                               # unreferenced
        self.cls = cls
        self.count = count

    def neighbours(self, nodes_ids):
        """6 lattice neighbours (-x +x -y +y -z +z) of the given node ids; -1 outside the lattice."""
        i, j, k = np.unravel_index(nodes_ids, self.lattice)
        out = np.full((len(nodes_ids), 6), -1, dtype=np.int64)
        for d, (axis, step) in enumerate([(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]):
            idx = [i, j, k]
            idx[axis] = idx[axis] + step
            ok = (idx[axis] >= 0) & (idx[axis] < self.lattice[axis])
            lin = np.ravel_multi_index([np.clip(idx[0], 0, self.lattice[0] - 1), np.clip(idx[1], 0, self.lattice[1] - 1),
                                        np.clip(idx[2], 0, self.lattice[2] - 1)], self.lattice)
            out[ok, d] = lin[ok]
        return out


class Smoother:
    """Holds everything that does not change between smoothing steps (neighbour lists, weights, layer classes)."""

    def __init__(self, mesh, lam, log):
        self.mesh = mesh
        cls = mesh.cls
        self.active = np.flatnonzero(cls > 0)
        self.nb = mesh.neighbours(self.active)
        padded = np.concatenate([cls, [0]]).astype(np.int16)             # index -1 -> class 0
        cls_active = cls[self.active]
        nb_class = padded[self.nb]
        same = (nb_class == cls_active[:, None]) & (cls_active[:, None] != 0)
        same[cls_active == 1, :] = True                                    # class 1 uses all neighbours
        self.same = same
        self.count_original = same.sum(axis=1)
        self.divisor = np.maximum(self.count_original, 1)
        self.valid = self.count_original > 0

        # five inner layers 101..105 counted inwards from the class 2 / 3 nodes (layer 1 uses classes >= 2)
        layered = cls.astype(np.int32).copy()
        lookup = np.full(mesh.N, -1, dtype=np.int64)
        lookup[self.active] = np.arange(len(self.active))
        class1 = np.flatnonzero(cls == 1)
        pos1 = lookup[class1]
        nb1 = self.nb[pos1]                                                # class 1 nodes always have 6 real neighbours
        for layer in range(1, 6):
            cur = layered[nb1]
            hit = (cur >= 2).any(axis=1) if layer == 1 else (cur == 100 + layer - 1).any(axis=1)
            mark = (layered[class1] == 1) & hit
            layered[class1[mark]] = 100 + layer
        self.layered = layered
        weight = np.full(mesh.N, lam)
        for layer in range(1, 6):
            weight[layered == 100 + layer] = lam + INNER_LAYER_STEP * layer
        weight1 = weight.copy()
        weight1[cls == 2] = 0
        weight1[cls == 3] = 0
        self.weight_odd = weight[self.active]
        self.weight_even = weight1[self.active]
        counts = {int(c): int((layered == c).sum()) for c in (0, 1, 2, 3, 101, 102, 103, 104, 105)}
        log("Node classes:  0 (unused) = {0:,}   1 (interior) = {1:,}   2 (between regions) = {2:,}   3 (surface) = {3:,}".format(
            counts[0], counts[1], counts[2], counts[3]))
        log("Inner layers:  " + "   ".join(f"{c} = {counts[c]:,}" for c in (101, 102, 103, 104, 105)))
        self.class_counts = counts

    def step(self, nodes, odd):
        """One Laplacian step (all nodes move simultaneously, means use the coordinates before the step)."""
        total = np.zeros((len(self.active), 3))
        for d in range(6):
            total += nodes[self.nb[:, d]] * self.same[:, d, None]
        mean = total / self.divisor[:, None]
        w = (self.weight_odd if odd else self.weight_even)[:, None]
        new = nodes.copy()
        current = nodes[self.active]
        updated = (1 - w) * current + w * mean
        new[self.active[self.valid]] = updated[self.valid]
        return new


def targeted_adjustment(mesh, nodes, node_to_elements, threshold, lam, ref_dist, limit=None):
    """Elements with scaled Jacobian < threshold are reshaped, one after the other in element order.
    Each element sees the nodes already moved by the earlier ones, exactly like the sequential loop of the reference implementation.
    limit: only elements with index < limit are visited (None = all elements)."""
    E = mesh.E
    jac = quality_of_jac(nodes, E)
    heap = [int(e) for e in np.flatnonzero(jac < threshold) if limit is None or e < limit]
    heapq.heapify(heap)
    queued = set(heap)
    last = -1
    reshaped = 0
    while heap:
        e = heapq.heappop(heap)
        if e <= last:
            continue
        last = e
        ids = E[e]
        coords = nodes[ids]
        jac_old = float(scaled_jacobian(coords[None])[0])
        if jac_old >= threshold:
            continue
        for _ in range(MAX_ADJUST_ITERATIONS):
            adjusted = adjust_hex(coords, ref_dist, lam)
            jac_new = float(scaled_jacobian(adjusted[None])[0])
            converged = jac_new >= threshold or abs(jac_new - jac_old) < CONVERGENCE_REL_DIFF
            coords, jac_old = adjusted, jac_new
            if converged:
                break
        nodes[ids] = coords
        reshaped += 1
        touched = np.unique(np.concatenate([node_to_elements(n) for n in ids]))
        touched = touched[(touched > e) & ((touched < limit) if limit is not None else True)]
        if len(touched):
            new_jac = scaled_jacobian(nodes[E[touched]])
            for t in touched[new_jac < threshold]:
                if int(t) not in queued:
                    queued.add(int(t))
                    heapq.heappush(heap, int(t))
    return reshaped


def quality_of_jac(nodes, E, chunk=100000):
    return np.concatenate([scaled_jacobian(nodes[E[i:i + chunk]]) for i in range(0, len(E), chunk)])


def node_element_lookup(E, N):
    order = np.argsort(E.ravel(), kind="stable")
    sorted_nodes = E.ravel()[order]
    owner = order // 8
    starts = np.searchsorted(sorted_nodes, np.arange(N + 1))
    return lambda n: owner[starts[n]:starts[n + 1]]


# =============================================================================================
# Generation
# =============================================================================================


def summarize(q):
    return dict(jac_min=q["jacobian"].min(), jac_mean=q["jacobian"].mean(), jac_max=q["jacobian"].max(),
                vol_min=q["volume"].min(), vol_mean=q["volume"].mean(), vol_max=q["volume"].max(),
                er_mean=q["edge_ratio"].mean(), er_max=q["edge_ratio"].max(),
                dr_mean=q["diagonal_ratio"].mean(), dr_min=q["diagonal_ratio"].min())


def generate(data, spacing, iterations, lam, targeted, threshold, legacy_limit=False, seal=None, regions=None, log=print):
    """Runs the whole mesh generation. Returns a dict with the mesh, nodes of every step, quality of every step.

    legacy_limit reproduces a quirk of the reference implementation: its variable numElements is overwritten by the per-label report loop
    (numElements = size(elements_group, 1)), so from the 2nd targeted pass on only the first N elements are visited,
    N = number of interface / surface elements of the last label. Off (default) = every element is visited.

    seal = (brain labels, outer layers innermost first) closes gaps in the outer layers before the mesh is built.
    regions = RegionMap of step 1 (or None): every element of the combined label also gets its original region label."""
    t0 = time.time()
    sealed = None
    if seal is not None:
        brain, outer = seal
        before = exposed_brain_voxels(data, brain)
        data, filled = seal_layers(data, brain, outer)
        after = exposed_brain_voxels(data, brain)
        sealed = dict(brain=tuple(brain), outer=tuple(outer), filled=filled, exposed_before=before, exposed_after=after)
        log(f"Gap sealing: voxels added {filled if filled else '(nothing to seal)'}; "
            f"brain voxels touching empty space {before:,} -> {after:,}")
    mesh = Mesh(data, spacing)
    log(f"Lattice {mesh.shape[0]} x {mesh.shape[1]} x {mesh.shape[2]} voxels of {mesh.spacing[0]:g} mm: "
        f"{mesh.n:,} elements, {int((mesh.count > 0).sum()):,} nodes")
    for label in np.unique(mesh.labels):
        log(f"  label {int(label)}: {int((mesh.labels == label).sum()):,} elements")
    if regions is not None:
        mesh.region_grid = regions.relabel(mesh.label_grid, mesh.spacing)
        mesh.region_labels = mesh.region_grid[mesh.ijk[:, 0], mesh.ijk[:, 1], mesh.ijk[:, 2]].astype(np.int32)
        log(f"Original regions of the combined label {regions.combined}: "
            + ", ".join(f"{int(l)}: {int((mesh.region_labels == l).sum()):,}" for l in regions.selected if (mesh.region_labels == l).any())
            + " elements")
    smoother = Smoother(mesh, lam, log)
    ref_dist = mesh.cube * math.sqrt(3) / 2                        # distance corner - centroid of the reference cube
    lookup = node_element_lookup(mesh.E, mesh.N) if targeted else None
    limit_later = None
    if targeted and legacy_limit:
        groups = region_groups(mesh)
        limit_later = len(groups[max(groups)][0])
        log(f"Legacy mode: from the 2nd targeted pass only the first {limit_later:,} elements are visited (as in the reference implementation)")

    nodes = mesh.nodes0.copy()
    nodes_per_step = {0: nodes.copy()}
    quality_per_step = {0: summarize(quality_of(nodes, mesh.E))}
    reshaped_per_step = {}
    for step in range(1, 2 * iterations + 1):
        odd = step % 2 == 1
        nodes = smoother.step(nodes, odd)
        note = ""
        if odd and targeted:
            reshaped_per_step[step] = targeted_adjustment(mesh, nodes, lookup, threshold, lam, ref_dist,
                                                          limit=limit_later if step > 1 else None)
            note = f", {reshaped_per_step[step]:,} elements reshaped"
        nodes_per_step[step] = nodes.copy()
        quality_per_step[step] = summarize(quality_of(nodes, mesh.E))
        s = quality_per_step[step]
        log(f"Step {step} ({'smoothing' if odd else 'propagation'}): min Jacobian {s['jac_min']:.4f}, "
            f"mean {s['jac_mean']:.4f}{note}")
    log(f"Mesh generated in {time.time() - t0:.1f} s")
    return dict(mesh=mesh, nodes=nodes_per_step, quality=quality_per_step, reshaped=reshaped_per_step,
                smoother=smoother, iterations=iterations, lam=lam, targeted=targeted, threshold=threshold,
                legacy_limit=legacy_limit, sealed=sealed, regions=regions,
                seconds=time.time() - t0)


# =============================================================================================
# Reports (min / max / mean and percentiles like the Stats_Report files of the reference implementation)
# =============================================================================================


def percentile(values, p):
    return float(np.percentile(values, p, method="hazen"))        # Hazen percentile definition


def stats_lines(title, values):
    lines = [f"{title} Statistics:",
             f"  Minimum: {np.min(values):.4f}", f"  Maximum: {np.max(values):.4f}", f"  Mean: {np.mean(values):.4f}"]
    lines += [f"  {p}th Percentile: {percentile(values, p):.4f}" for p in list(range(10, 100, 10)) + [99]]
    return lines


def gaussian_curvature(nodes, faces):
    """(2 pi - sum of face angles at the node) / (4 * sum of face areas), per unique node of the quad faces."""
    P = nodes[faces]                                                  # (m, 4, 3)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) \
        + 0.5 * np.linalg.norm(np.cross(P[:, 3] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    unique, inverse = np.unique(faces.ravel(), return_inverse=True)
    angle_sum = np.zeros(len(unique))
    area_sum = np.zeros(len(unique))
    for c in range(4):
        v1 = P[:, c] - P[:, (c - 1) % 4]
        v2 = P[:, c] - P[:, (c + 1) % 4]
        cosine = np.abs(np.einsum("ij,ij->i", v1, v2)) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        ang = np.arccos(np.clip(cosine, -1, 1))
        idx = inverse.reshape(-1, 4)[:, c]
        angle_sum += np.bincount(idx, weights=ang, minlength=len(unique))
        area_sum += np.bincount(idx, weights=area, minlength=len(unique))
    return (2 * np.pi - angle_sum) / (4 * area_sum)


def region_groups(mesh):
    """Per label: elements with at least one class 2 / 3 node and their faces whose four nodes are all class 2 / 3."""
    groups = {}
    for label in np.unique(mesh.labels):
        elements = mesh.E[mesh.labels == label]
        valid = (mesh.cls[elements] >= 2).any(axis=1)
        elements = elements[valid]
        faces = np.concatenate([elements[:, f] for f in FACE_DEFS])
        faces = faces[mesh.cls[faces].min(axis=1) >= 2]
        groups[int(label)] = (np.flatnonzero(mesh.labels == label)[valid], faces)
    return groups


def build_report(result, step, log=None):
    mesh, nodes = result["mesh"], result["nodes"][step]
    q = quality_of(nodes, mesh.E)
    lines = ["MESH GENERATION REPORT", "=" * 60,
             f"Elements: {mesh.n:,}   Nodes used: {int((mesh.count > 0).sum()):,}   Voxel size: {mesh.cube:g} mm",
             f"Smoothing iterations: {result['iterations']} ({2 * result['iterations']} steps)   Lambda: {result['lam']}",
             f"Targeted element adjustment: {'ON' if result['targeted'] else 'OFF'}"
             + (f" (Jacobian threshold {result['threshold']}"
                + (", legacy: only the first N elements after the 1st pass)" if result["legacy_limit"] else ", all elements)")
                if result["targeted"] else ""),
             ("Gap sealing: ON, brain labels %s, outer layers %s, voxels added %s, brain voxels touching empty space %d -> %d"
              % (result["sealed"]["brain"], result["sealed"]["outer"], result["sealed"]["filled"] or "none",
                 result["sealed"]["exposed_before"], result["sealed"]["exposed_after"]))
             if result["sealed"] else "Gap sealing: OFF",
             *([f"Combined regions (step 1): original labels {list(result['regions'].selected)} -> label {result['regions'].combined}"]
               if result.get("regions") is not None else []),
             "Node classes: " + ", ".join(f"{k}: {v:,}" for k, v in result["smoother"].class_counts.items()),
             "", "QUALITY AFTER EVERY STEP (all elements)",
             f"{'step':>4} {'kind':>12} {'minJac':>8} {'meanJac':>8} {'<thr':>7} {'minVol':>9} {'meanVol':>9} "
             f"{'maxEdgeR':>9} {'minDiagR':>9} {'reshaped':>9}"]
    for s in range(1, 2 * result["iterations"] + 1):
        qs = result["quality"][s]
        below = ""
        lines.append(f"{s:>4} {'smoothing' if s % 2 else 'propagation':>12} {qs['jac_min']:>8.4f} {qs['jac_mean']:>8.4f} "
                     f"{'':>7} {qs['vol_min']:>9.5f} {qs['vol_mean']:>9.5f} {qs['er_max']:>9.4f} {qs['dr_min']:>9.4f} "
                     f"{result['reshaped'].get(s, ''):>9}")
    lines += ["", f"ALL ELEMENTS AT STEP {step}"]
    for title, key in (("Scaled Jacobian", "jacobian"), ("Element Volume", "volume"), ("Edge Ratio", "edge_ratio"),
                       ("Diagonal Ratio", "diagonal_ratio")):
        lines += stats_lines(title, q[key])
    n_bad = int((q["jacobian"] < result["threshold"]).sum())
    lines.append(f"Elements with scaled Jacobian < {result['threshold']}: {n_bad:,} ({100 * n_bad / mesh.n:.3f} %)")
    lines.append(f"Elements with scaled Jacobian <= 0 (inverted or degenerate): {int((q['jacobian'] <= 0).sum()):,}")

    if mesh.region_labels is not None:
        lines += ["", f"ORIGINAL REGIONS (the combined label {result['regions'].combined} turned back into the original labels, "
                      f"step {step})", f"{'region':>8} {'elements':>10} {'share %':>8} {'minJac':>8} {'meanJac':>8} {'meanVol':>9}"]
        combined_n = int((mesh.labels == result["regions"].combined).sum())
        for region in np.unique(mesh.region_labels):
            sel = mesh.region_labels == region
            in_combined = int((sel & (mesh.labels == result["regions"].combined)).sum())
            share = f"{100 * in_combined / max(1, combined_n):>8.2f}" if in_combined else f"{'-':>8}"
            lines.append(f"{int(region):>8} {int(sel.sum()):>10,} {share} {q['jacobian'][sel].min():>8.4f} "
                         f"{q['jacobian'][sel].mean():>8.4f} {q['volume'][sel].mean():>9.5f}")
        lines.append("(the statistics below use the combined label, which is what the smoothing worked on)")

    for label, (index, faces) in region_groups(mesh).items():
        lines += ["", f"LABEL {label}: {int((mesh.labels == label).sum()):,} elements, "
                      f"{len(index):,} of them touch a region interface or the surface (the 'group' of the reference implementation)"]
        for title, key in (("Scaled Jacobian", "jacobian"), ("Element Volume", "volume"), ("Edge Ratio", "edge_ratio"),
                           ("Diagonal Ratio", "diagonal_ratio"), ("Characteristic Length", "characteristic"),
                           ("Timestep", "time_step")):
            lines += stats_lines(title, q[key][index])
        if len(faces):
            lines += stats_lines("Gaussian Curvature", gaussian_curvature(nodes, faces))
    return "\n".join(lines), q


# =============================================================================================
# LS-DYNA key file export (same layout as the reference implementation)
# =============================================================================================


def surface_shell_faces(mesh):
    """Faces whose four nodes are class 3 and that belong to a single element, in the order of the reference implementation."""
    all_faces = np.concatenate([mesh.E[:, f] for f in FACE_DEFS])          # face-definition major, like cell2mat
    keep = (mesh.cls[all_faces] == 3).all(axis=1)
    faces = all_faces[keep]
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def check_shell_part(shell_part, labels):
    """The shell part id must be a positive whole number that no solid part uses."""
    if int(shell_part) <= 0:
        raise ValueError("The shell part id must be larger than 0.")
    used = sorted(int(v) for v in np.unique(labels))
    if int(shell_part) in used:
        raise ValueError(f"The shell part id {int(shell_part)} is already used by a solid part (solid part ids: "
                         + (str(used) if len(used) <= 12 else f"{len(used)} ids from {used[0]} to {used[-1]}")
                         + "). Choose another id for the shell elements.")


def write_key_file(path, mesh, nodes, labels=None, shells=True, shell_part=SHELL_PART_ID):
    """labels = part id of every solid element (default: the mesh labels; the original regions when they are restored).
    shells=False leaves out the *ELEMENT_SHELL section (the surface shell elements that cover the whole model);
    shell_part = their part id."""
    labels = mesh.labels if labels is None else labels
    referenced = np.unique(mesh.E)                                          # sorted original ids
    new_id = np.zeros(mesh.N, dtype=np.int64)
    new_id[referenced] = np.arange(1, len(referenced) + 1)
    elements = new_id[mesh.E]
    shell_nodes = new_id[surface_shell_faces(mesh)] if shells else np.zeros((0, 4), dtype=np.int64)
    with open(path, "w", newline="\n") as f:
        f.write("*KEYWORD\n*ELEMENT_SOLID\n$#   eid     pid      n1      n2      n3      n4      n5      n6      n7      n8\n")
        eid = np.arange(1, mesh.n + 1)
        np.savetxt(f, np.column_stack([eid, labels, elements]), fmt="%8d" * 10)
        if shells:
            f.write("*ELEMENT_SHELL\n$#   eid     pid      n1      n2      n3      n4      n5      n6      n7      n8\n")
            shell_eid = np.arange(mesh.n + 1, mesh.n + 1 + len(shell_nodes))
            np.savetxt(f, np.column_stack([shell_eid, np.full(len(shell_nodes), int(shell_part)), shell_nodes,
                                           np.zeros((len(shell_nodes), 4), int)]), fmt="%8d" * 10)
        f.write("*NODE\n$#   nid               x               y               z      tc      rc\n")
        coords = nodes[referenced]
        out = np.empty(len(referenced), dtype=[("i", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8"), ("a", "i8"), ("b", "i8")])
        out["i"], out["x"], out["y"], out["z"], out["a"], out["b"] = np.arange(1, len(referenced) + 1), coords[:, 0], coords[:, 1], coords[:, 2], 0, 0
        np.savetxt(f, out, fmt="%8d%16.6f%16.6f%16.6f%8d%8d")
        f.write("*END\n")
    return len(shell_nodes), len(referenced)


def export_mesh(folder, result, step, report_text, log=print, restore=False, shells=True, shell_part=SHELL_PART_ID):
    """restore=True writes the original region labels as part ids (when step 1 combined regions).
    shells=False leaves out the surface shell elements; shell_part is their part id (checked against the solid part ids)."""
    mesh, nodes = result["mesh"], result["nodes"][step]
    labels = mesh.region_labels if (restore and mesh.region_labels is not None) else mesh.labels
    if shells:
        check_shell_part(shell_part, labels)                                # refuse before anything is written
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"Key_File_Iter_{step}.k"
    shell_count, used = write_key_file(path, mesh, nodes, labels, shells, shell_part)
    centroids = nodes[mesh.E].mean(axis=1)                                  # centroids_Many_decimals.csv
    np.savetxt(folder / "centroids_Many_decimals.csv", np.column_stack([centroids, labels]), delimiter=",", fmt="%.15g")
    (folder / "MetricReport.txt").write_text(report_text, encoding="utf-8")
    log(f"Exported {path} ({mesh.n:,} solid elements, "
        + (f"{shell_count:,} shell elements, " if shells else "no shell elements, ") + f"{used:,} nodes)")
    log("   part ids: " + ("ORIGINAL regions " if labels is not mesh.labels else "combined labels ")
        + str([int(v) for v in np.unique(labels)])
        + (f", shell elements {int(shell_part)}" if shells else ", surface shell elements not exported"))
    return path


# =============================================================================================
# 3D surface of the mesh (exposed element faces, drawn with the smoothed node coordinates)
# =============================================================================================
