"""Conservative local guards for sequential tissue-label reassignment."""

import numpy as np
from scipy.ndimage import label


def connectivity_safe_claims(grid, claims):
    trial = np.asarray(grid).copy()
    claims = np.asarray(claims)
    if trial.ndim != 3 or claims.shape != trial.shape:
        raise ValueError("grid and claims must have the same 3D shape")
    accepted = np.zeros(trial.shape, dtype=bool)
    steps = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    for point in np.argwhere(claims >= 0):
        index = tuple(point)
        donor, recipient = int(trial[index]), int(claims[index])
        if donor == recipient:
            continue
        lo, hi = np.maximum(point - 1, 0), np.minimum(point + 2, trial.shape)
        patch = trial[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
        neighbors = point + steps
        neighbors = neighbors[((neighbors >= 0) & (neighbors < trial.shape)).all(axis=1)]
        if recipient and not np.any(trial[tuple(neighbors.T)] == recipient):
            continue
        if donor:
            mask = patch == donor
            mask[tuple(point - lo)] = False
            components, _ = label(mask)
            touching = components[tuple((neighbors - lo).T)]
            touching = np.unique(touching[touching > 0])
            if len(touching) != 1:
                continue
        trial[index] = recipient
        accepted[index] = True
    return accepted
