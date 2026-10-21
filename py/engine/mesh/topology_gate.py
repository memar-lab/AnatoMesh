"""Closed-domain topology checks for VTK-order hexahedra."""

import numpy as np
from engine.mesh.hex_mesh import _unique_rows
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from engine.mesh.hex_geometry import validate_hex_input


FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                  [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


def boundary_topology(faces):
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 4 or faces.dtype.kind not in "iu":
        raise ValueError("boundary faces must be integer quadrilaterals")
    if not len(faces):
        return {"open_edges": 0, "nonmanifold_edges": 0, "nonmanifold_vertices": 0}
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("boundary face repeats a vertex")
    edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    inverse, counts = _unique_rows(keys)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = starts[counts == 2]
    a, b = order[paired], order[paired + 1]
    corner = np.arange(faces.size).reshape(-1, 4)
    endpoints = np.stack([corner, np.roll(corner, -1, axis=1)], axis=-1).reshape(-1, 2)
    normalized = np.take_along_axis(endpoints, np.argsort(edges, axis=1), axis=1)
    rows, cols = normalized[a].ravel(), normalized[b].ravel()
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(faces.size, faces.size)).tocsr()
    _, components = connected_components(graph, directed=False)
    fans = np.unique(np.column_stack([faces.ravel(), components]), axis=0)
    _, fan_counts = np.unique(fans[:, 0], return_counts=True)
    return {"open_edges": int(np.count_nonzero(counts == 1)),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
            "nonmanifold_vertices": int(np.count_nonzero(fan_counts > 1))}


def domain_topology(nodes, hexes, labels):
    _, hexes = validate_hex_input(nodes, hexes)
    labels = np.asarray(labels)
    if labels.shape != (len(hexes),) or labels.dtype.kind not in "iu":
        raise ValueError("one integer domain label is required per element")
    duplicate_elements = int(len(hexes) - len(np.unique(np.sort(hexes, axis=1), axis=0)))
    all_faces = hexes[:, FACES].reshape(-1, 4)
    _, face_counts = np.unique(np.sort(all_faces, axis=1), axis=0, return_counts=True)
    overshared_faces = int(np.count_nonzero(face_counts > 2))
    reports = {}
    for label in np.unique(labels):
        cells = hexes[labels == label]
        faces = cells[:, FACES].reshape(-1, 4)
        _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                     return_index=True, return_counts=True)
        boundary = faces[first[counts == 1]]
        report = boundary_topology(boundary)
        report["overshared_faces"] = int(np.count_nonzero(counts > 2))
        report["duplicate_elements"] = int(len(cells) - len(np.unique(np.sort(cells, axis=1), axis=0)))
        report["boundary_faces"] = int(len(boundary))
        report["passed"] = bool(len(boundary) and not any(report[key] for key in
            ("open_edges", "nonmanifold_edges", "nonmanifold_vertices", "overshared_faces", "duplicate_elements")))
        reports[int(label)] = report
    return {"passed": bool(reports) and not duplicate_elements and not overshared_faces
            and all(r["passed"] for r in reports.values()),
            "duplicate_elements": duplicate_elements, "overshared_faces": overshared_faces,
            "domains": reports}
