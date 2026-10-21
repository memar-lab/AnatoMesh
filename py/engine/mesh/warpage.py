"""Quadrilateral-face warping factors and diagonal triangle angles."""

import numpy as np

from engine.mesh.hex_geometry import validate_hex_input
from engine.mesh.topology_gate import FACES


def hex_warpage(nodes, hexes, chunk_size=8192):
    nodes, hexes = validate_hex_input(nodes, hexes)
    factors, angles = np.empty(len(hexes)), np.empty(len(hexes))
    for start in range(0, len(hexes), chunk_size):
        faces = nodes[hexes[start:start + chunk_size][:, FACES]]
        a, b, c, d = (faces[:, :, i] for i in range(4))
        normal = np.cross(c - a, d - b)
        norm = np.linalg.norm(normal, axis=-1)
        unit = np.divide(normal, norm[..., None], out=np.zeros_like(normal),
                         where=norm[..., None] > 0)
        height = np.abs(np.sum((b - a) * unit, axis=-1))
        factor = np.divide(height, np.sqrt(norm / 2), out=np.full_like(norm, np.inf),
                           where=norm > 0)
        diagonal_angles = []
        for n1, n2 in [(np.cross(b - a, d - a), np.cross(c - b, d - b)),
                       (np.cross(b - a, c - a), np.cross(c - a, d - a))]:
            denominator = np.linalg.norm(n1, axis=-1) * np.linalg.norm(n2, axis=-1)
            cosine = np.divide(np.sum(n1 * n2, axis=-1), denominator,
                               out=np.full_like(denominator, -1), where=denominator > 0)
            diagonal_angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        factors[start:start + len(faces)] = factor.max(axis=1)
        angles[start:start + len(faces)] = np.maximum(*diagonal_angles).max(axis=1)
    return factors, angles
