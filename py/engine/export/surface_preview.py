from pathlib import Path

import numpy as np

from engine.paths import assert_write_path


FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


def _face_table(hexes):
    faces = hexes[:, FACES].reshape(-1, 4)
    owner = np.repeat(np.arange(len(hexes)), 6)
    from engine.mesh.hex_mesh import _unique_rows   # np.unique(axis=0) took ~50 s on a 4M-hex head

    inverse, counts = _unique_rows(np.sort(faces, axis=1))
    order = np.argsort(inverse, kind="stable")
    shared = order[counts[inverse[order]] == 2]
    partner = np.full(len(faces), -1)
    partner[shared[0::2]] = owner[shared[1::2]]
    partner[shared[1::2]] = owner[shared[0::2]]
    return faces, owner, partner


def part_surfaces(points, blocks):
    hexes = np.concatenate([np.asarray(block) for _, block in blocks])
    parts = np.concatenate([np.full(len(block), pid) for pid, block in blocks])
    faces, owner, partner = _face_table(hexes)
    keep = (partner < 0) | (parts[owner] != parts[np.maximum(partner, 0)])
    exterior = partner[keep] < 0
    used, inverse = np.unique(faces[keep], return_inverse=True)
    return (np.asarray(points, dtype=np.float32)[used], inverse.reshape(-1, 4).astype(np.int32),
            parts[owner[keep]].astype(np.int16), exterior)


def write(points, blocks, names, path):
    vertices, quads, parts, exterior = part_surfaces(points, blocks)
    path = assert_write_path(Path(path))
    np.savez_compressed(path, vertices=vertices, quads=quads, parts=parts, exterior=exterior,
                        part_ids=np.array(sorted(names), dtype=np.int16),
                        part_names=np.array([names[k] for k in sorted(names)]))
    return path
