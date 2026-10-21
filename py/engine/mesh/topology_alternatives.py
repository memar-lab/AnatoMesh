"""Bounded search over alternative octant fillings with connectivity guards."""

from itertools import product

import numpy as np

from engine.mesh.hex_topology import _NONMANIFOLD_BAD, _pattern_codes, count_nonmanifold
from engine.mesh.label_connectivity import connectivity_safe_claims


def repair_alternative_fillings(labels, priority, max_candidates=1000, max_rounds=12):
    out = np.asarray(labels).copy()
    rank = {int(value): index for index, value in enumerate(priority)}
    if out.ndim != 3 or out.dtype.kind not in "iu" or np.any(out < 0):
        raise ValueError("labels must be a nonnegative integer 3D array")
    if not set(map(int, np.unique(out))) <= set(rank) or len(rank) != len(priority):
        raise ValueError("priority must contain every present label without duplicates")
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in (max_candidates, max_rounds)):
        raise ValueError("search limits must be nonnegative integers")
    offsets = np.array(list(product((0, 1), repeat=3))) - 1
    attempted, accepted = 0, 0
    # pattern codes per material, computed once and patched around each accepted change
    # (recomputing every material over the whole lattice after each change cost ~10 s a call)
    cache: dict[int, np.ndarray] = {}
    pinched: dict[int, np.ndarray] = {}   # flat indices of the pinched nodes, C order

    def codes_of(material):
        if material not in cache:
            cache[material] = _pattern_codes(np.pad(out == material, 1))
            pinched[material] = np.flatnonzero(_NONMANIFOLD_BAD[cache[material]])
        return cache[material]

    def patch_codes(lo, hi):
        # voxel i (unpadded) is corner of nodes i and i+1 of the padded node grid
        a = np.maximum(np.asarray(lo), 0)
        b = np.minimum(np.asarray(hi) + 1, np.array(out.shape) + 1)
        for material, codes in cache.items():
            # nodes a .. b-1 read padded voxels a .. b, i.e. unpadded a-1 .. b-1 (zero outside)
            src_lo, src_hi = a - 1, b
            clip_lo, clip_hi = np.maximum(src_lo, 0), np.minimum(src_hi, out.shape)
            block = np.zeros(tuple(src_hi - src_lo), dtype=bool)
            inner = tuple(slice(int(c - s), int(d - s)) for c, d, s in zip(clip_lo, clip_hi, src_lo))
            block[inner] = out[tuple(slice(int(c), int(d)) for c, d in zip(clip_lo, clip_hi))] == material
            box = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
            codes[box] = _pattern_codes(block)
            # the pinched list: drop the box's old entries, add its new ones, keep C order
            old = pinched[material]
            ijk = np.unravel_index(old, codes.shape)
            inside = np.all([(c >= x) & (c < y) for c, x, y in zip(ijk, a, b)], axis=0)
            local = np.argwhere(_NONMANIFOLD_BAD[codes[box]]) + a
            new = np.ravel_multi_index(tuple(local.T), codes.shape) if len(local) else local[:, 0]
            pinched[material] = np.sort(np.concatenate([old[~inside], new]))

    for _ in range(max_rounds):
        changed = False
        for material in priority:
            codes = codes_of(material)
            for node in np.stack(np.unravel_index(pinched[material], codes.shape), axis=1):
                pattern = int(codes[tuple(node)])
                candidates = [bits for bits in range(1, 256) if not bits & pattern
                              and not _NONMANIFOLD_BAD[bits | pattern]]
                for bits in sorted(candidates, key=lambda bits: (bits.bit_count(), bits)):
                    if attempted >= max_candidates:
                        return out, {"attempted": attempted, "accepted": accepted, "budget_exhausted": True}
                    attempted += 1
                    points = node + offsets[[bit for bit in range(8) if bits >> bit & 1]]
                    if np.any(points < 0) or np.any(points >= out.shape):
                        continue
                    donors = out[tuple(points.T)]
                    if any(rank.get(int(v), len(rank)) <= rank[int(material)] for v in donors):
                        continue
                    lo, hi = np.maximum(node - 2, 0), np.minimum(node + 2, out.shape)
                    view = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                    patch = out[view]
                    claims = np.full(patch.shape, -1, dtype=np.int64)
                    claims[tuple((points - lo).T)] = material
                    mask = connectivity_safe_claims(patch, claims)
                    if not np.all(mask[tuple((points - lo).T)]):
                        continue
                    trial = patch.copy()
                    trial[mask] = material
                    if sum(count_nonmanifold(trial).values()) >= sum(count_nonmanifold(patch).values()):
                        continue
                    out[view] = trial
                    patch_codes(lo, hi)
                    accepted += 1
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return out, {"attempted": attempted, "accepted": accepted, "budget_exhausted": False}
