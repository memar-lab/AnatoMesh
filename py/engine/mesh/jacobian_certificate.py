"""Conservative Bernstein bounds for trilinear hex Jacobian determinants."""

from itertools import product

import numpy as np

from engine.mesh.hex_geometry import shape_gradients, validate_hex_input

LOCATIONS = np.array(list(product((-1., 0., 1.), repeat=3)))
GRADIENTS = shape_gradients(LOCATIONS)
TRANSFORM = np.array([[1., 0., 0.], [-.5, 2., -.5], [0., 0., 1.]])


def bernstein_coefficients(values):
    first = np.einsum("ai,eijk->eajk", TRANSFORM, values)
    second = np.einsum("bj,eajk->eabk", TRANSFORM, first)
    return np.einsum("ck,eabk->eabc", TRANSFORM, second)


def _split(coefficients, axis):
    values = np.moveaxis(coefficients, axis, 0)
    midpoint = (values[0] + 2 * values[1] + values[2]) / 4
    left = np.stack([values[0], (values[0] + values[1]) / 2, midpoint])
    right = np.stack([midpoint, (values[1] + values[2]) / 2, values[2]])
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def _certify(coefficients, tolerance, depth, limit):
    if np.min(coefficients) > tolerance:
        return 1
    if np.min(coefficients[::2, ::2, ::2]) <= 0:
        return -1
    if np.min(coefficients[::2, ::2, ::2]) <= tolerance:
        return 0
    if depth == limit:
        return 0
    results = [_certify(child, tolerance, depth + 1, limit)
               for child in _split(coefficients, depth % 3)]
    return -1 if -1 in results else min(results)


def certify_hex_jacobians(nodes, hexes, max_depth=12, chunk_size=8192):
    """Return 1 for certified positive, -1 for nonpositive, 0 for unresolved."""
    nodes, hexes = validate_hex_input(nodes, hexes)
    if not isinstance(max_depth, (int, np.integer)) or not 0 <= max_depth <= 24:
        raise ValueError("max_depth must be an integer from 0 to 24")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    status = np.zeros(len(hexes), dtype=np.int8)
    for start in range(0, len(hexes), chunk_size):
        vertices = nodes[hexes[start:start + chunk_size]]
        vertices = vertices - vertices[:, :1]
        scale = np.max(np.abs(vertices), axis=(1, 2))
        vertices = np.divide(vertices, scale[:, None, None], out=np.zeros_like(vertices),
                             where=scale[:, None, None] > 0)
        jacobians = np.einsum("eia,qib->eqab", vertices, GRADIENTS)
        values = np.linalg.det(jacobians).reshape(-1, 3, 3, 3)
        coefficients = bernstein_coefficients(values)
        tolerance = 128 * np.finfo(float).eps * np.max(np.abs(coefficients), axis=(1, 2, 3))
        positive = np.min(coefficients, axis=(1, 2, 3)) > tolerance
        current = np.where(positive, 1, 0).astype(np.int8)
        for index in np.flatnonzero(~positive):
            if np.isfinite(coefficients[index]).all():
                current[index] = _certify(coefficients[index], tolerance[index], 0, max_depth)
        status[start:start + len(vertices)] = current
    return status
