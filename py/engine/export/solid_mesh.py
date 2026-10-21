"""Deterministic Abaqus and VTK geometry export with strict HEX8 acceptance."""

from pathlib import Path

import numpy as np

from engine.mesh.quality_gate import require_hex_quality
from engine.mesh.topology_gate import FACES
from engine.paths import safe_open_write


def boundary_sets(hexes, labels):
    result = {}
    for part in np.unique(labels):
        elements = np.flatnonzero(labels == part)
        faces = hexes[elements][:, FACES].reshape(-1, 4)
        _, first, count = np.unique(np.sort(faces, axis=1), axis=0,
                                    return_index=True, return_counts=True)
        selected = np.sort(first[count == 1])
        result[int(part)] = {'nodes': np.unique(faces[selected]),
                             'elements': elements[selected // 6],
                             'sides': selected % 6 + 1}
    return result


def write_solid_mesh(path, nodes, hexes, labels, format=None):
    """Write a geometry include"""
    path = Path(path)
    format = format or {'.inp': 'abaqus', '.vtk': 'vtk'}.get(path.suffix.lower())
    if format not in ('abaqus', 'vtk'):
        raise ValueError('format must be abaqus or vtk')
    nodes, hexes, labels = np.asarray(nodes), np.asarray(hexes), np.asarray(labels)
    quality = require_hex_quality(nodes, hexes, labels)
    if np.any(labels <= 0) or np.any(labels > 2147483647):
        raise ValueError('part IDs must be positive signed 32-bit integers')
    used, inverse = np.unique(hexes, return_inverse=True)
    points, cells = nodes[used], inverse.reshape(hexes.shape)
    boundaries = boundary_sets(cells, labels)
    with safe_open_write(path, 'w') as stream:
        if format == 'abaqus':
            stream.write('** AnatoMesh HEX8 geometry; coordinates in mm\n*NODE\n')
            for index, point in enumerate(points, 1):
                stream.write(f'{index}, ' + ', '.join(format_float(v) for v in point) + '\n')
            stream.write('*ELEMENT, TYPE=C3D8\n')
            for index, cell in enumerate(cells, 1):
                stream.write(', '.join(map(str, [index, *(cell + 1)])) + '\n')
            for part, boundary in boundaries.items():
                stream.write(f'*ELSET, ELSET=PART_{part}\n')
                _indices(stream, np.flatnonzero(labels == part) + 1)
                stream.write(f'*NSET, NSET=BOUNDARY_{part}\n')
                _indices(stream, boundary['nodes'] + 1)
                stream.write(f'*SURFACE, TYPE=ELEMENT, NAME=SURFACE_{part}\n')
                for element, side in zip(boundary['elements'], boundary['sides']):
                    stream.write(f'{element + 1}, S{side}\n')
        else:
            stream.write('# vtk DataFile Version 3.0\nAnatoMesh HEX8; coordinates in mm\nASCII\n')
            stream.write(f'DATASET UNSTRUCTURED_GRID\nPOINTS {len(points)} double\n')
            for point in points:
                stream.write(' '.join(format_float(v) for v in point) + '\n')
            stream.write(f'CELLS {len(cells)} {9 * len(cells)}\n')
            for cell in cells:
                stream.write('8 ' + ' '.join(map(str, cell)) + '\n')
            stream.write(f'CELL_TYPES {len(cells)}\n')
            stream.writelines('12\n' for _ in cells)
            stream.write(f'CELL_DATA {len(cells)}\nSCALARS part_id int 1\nLOOKUP_TABLE default\n')
            stream.writelines(f'{part}\n' for part in labels)
            stream.write(f'POINT_DATA {len(points)}\n')
            for part, boundary in boundaries.items():
                membership = np.zeros(len(points), dtype=np.uint8)
                membership[boundary['nodes']] = 1
                stream.write(f'SCALARS boundary_{part} unsigned_char 1\nLOOKUP_TABLE default\n')
                stream.writelines(f'{value}\n' for value in membership)
    return {'format': format, 'nodes': len(points), 'elements': len(cells),
            'original_node_indices': used.tolist(), 'quality': quality,
            'parts': list(boundaries), 'analysis_ready': False}


def format_float(value):
    return f'{float(value):.17g}'


def _indices(stream, values):
    for start in range(0, len(values), 16):
        stream.write(', '.join(map(str, values[start:start + 16])) + '\n')
