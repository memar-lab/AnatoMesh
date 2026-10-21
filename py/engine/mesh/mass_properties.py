"""Integrate HEX8 mass properties from explicit SI tissue densities."""

from itertools import product

import numpy as np

from engine.mesh.hex_geometry import SIGNS, shape_gradients, validate_hex_input
from engine.mesh.jacobian_certificate import certify_hex_jacobians


def mass_properties(nodes, hexes, labels, densities_kg_m3, chunk_size=4096):
    """Return mass in kg, coordinates in mm, and centroidal inertia in kg mm²."""
    nodes, hexes = validate_hex_input(nodes, hexes)
    labels = np.asarray(labels)
    if not len(hexes):
        raise ValueError("mass properties require a nonempty mesh")
    if labels.shape != (len(hexes),) or labels.dtype.kind not in "iu":
        raise ValueError("labels must contain one integer part ID per element")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    regions, inverse = np.unique(labels, return_inverse=True)
    if set(densities_kg_m3) != set(regions):
        raise ValueError("density keys must match mesh part IDs exactly")
    values = [densities_kg_m3[int(region)] for region in regions]
    if any(isinstance(v, (bool, str)) or not np.isscalar(v) for v in values):
        raise ValueError("densities must be positive finite numbers in kg/m³")
    densities = np.asarray(values, dtype=float)
    if not np.isfinite(densities).all() or np.any(densities <= 0):
        raise ValueError("densities must be positive finite numbers in kg/m³")
    if np.any(certify_hex_jacobians(nodes, hexes, chunk_size=chunk_size) != 1):
        raise ValueError("all element Jacobians must be certified positive")
    abscissas, weights = np.polynomial.legendre.leggauss(3)
    indices = np.array(list(product(range(3), repeat=3)))
    locations = abscissas[indices]
    weights = weights[indices].prod(axis=1)
    shape = np.prod(1 + locations[:, None, :] * SIGNS, axis=2) / 8
    gradients = shape_gradients(locations)
    origin = nodes[hexes[0, 0]]
    volumes = np.zeros(len(regions))
    masses = np.zeros(len(regions))
    first = np.zeros((len(regions), 3))

    def quadrature():
        for start in range(0, len(hexes), chunk_size):
            vertices = nodes[hexes[start:start + chunk_size]] - origin
            ids = inverse[start:start + chunk_size]
            jac = np.einsum("eia,qib->eqab", vertices - vertices[:, :1], gradients)
            dv = np.linalg.det(jac) * weights
            if not np.isfinite(dv).all() or np.any(dv <= 0):
                raise ValueError("invalid integration volume")
            yield ids, np.einsum("qi,eia->eqa", shape, vertices), dv

    for ids, points, dv in quadrature():
        dm = dv * densities[ids, None] * 1e-9
        np.add.at(volumes, ids, dv.sum(axis=1))
        np.add.at(masses, ids, dm.sum(axis=1))
        np.add.at(first, ids, np.einsum("eq,eqa->ea", dm, points))
    centers = first / masses[:, None]
    center = first.sum(axis=0) / masses.sum()
    inertia = np.zeros((3, 3))
    regional_inertia = np.zeros((len(regions), 3, 3))

    def tensors(dm, offsets):
        second = np.einsum("eq,eqa,eqb->eab", dm, offsets, offsets)
        return np.trace(second, axis1=1, axis2=2)[:, None, None] * np.eye(3) - second

    for ids, points, dv in quadrature():
        dm = dv * densities[ids, None] * 1e-9
        inertia += tensors(dm, points - center).sum(axis=0)
        np.add.at(regional_inertia, ids, tensors(dm, points - centers[ids, None]))
    if not all(np.isfinite(value).all() for value in
               (volumes, masses, centers, center, inertia, regional_inertia)) or np.any(masses <= 0):
        raise ValueError("mass integration exceeded floating-point range")
    return {
        "units": {"volume": "mm³", "mass": "kg", "center_of_mass": "mm", "inertia": "kg mm²"},
        "volume": float(volumes.sum()), "mass": float(masses.sum()),
        "center_of_mass": (origin + center).tolist(), "inertia": inertia.tolist(),
        "regions": {int(region): {"density_kg_m3": float(densities[i]),
                     "volume": float(volumes[i]), "mass": float(masses[i]),
                     "center_of_mass": (origin + centers[i]).tolist(),
                     "inertia": regional_inertia[i].tolist()}
                    for i, region in enumerate(regions)},
    }
