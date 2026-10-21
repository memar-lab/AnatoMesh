"""Subject-specific correction of atlas labels warped from a template.

Tissue posteriors are estimated with a prior-driven Gaussian EM, then used to
re-assign cortical, cerebral white-matter and ventricular labels. The
mixture model follows Ashburner & Friston 2005, "Unified segmentation",
NeuroImage 26:839-851.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CORTEX_BASE_IDS = (3, 42)
CORTEX_RANGES = {
    "aseg": (),
    "aparc": ((1000, 1999), (2000, 2999)),
    "a2009s": ((11100, 11199), (12100, 12199)),
}
CEREBRAL_WM_IDS = (2, 41)
VENTRICLE_IDS = (4, 43, 5, 44, 14, 15)
CSF_LABEL = 24

LH_IDS = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28)
LH_RANGES = ((1000, 1999), (11100, 11199))
RH_IDS = (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)
RH_RANGES = ((2000, 2999), (12100, 12199))
LH_WM_LABEL, RH_WM_LABEL = CEREBRAL_WM_IDS

_TISSUE_KEYS = ("gm", "wm", "csf")
_MAX_ISLAND_PASSES = 10


def tissue_em(
    t1: np.ndarray,
    brain: np.ndarray,
    priors: dict[str, np.ndarray],
    *,
    iterations: int = 20,
    tol: float = 1e-4,
) -> dict[str, np.ndarray]:
    brain = np.asarray(brain, dtype=bool)
    out = {k: np.zeros(t1.shape, dtype=np.float32) for k in _TISSUE_KEYS}
    n = int(brain.sum())
    if n == 0:
        return out

    x = np.asarray(t1, dtype=np.float64)[brain]
    p = np.stack([np.asarray(priors[k], dtype=np.float64)[brain] for k in _TISSUE_KEYS])
    p += 1e-6
    p /= p.sum(axis=0, keepdims=True)

    gvar = float(x.var())
    std = np.sqrt(gvar) if gvar > 0 else 1.0
    floor = 1e-6 * gvar if gvar > 0 else 1e-12

    def moments(w):
        sw = np.maximum(w.sum(axis=1), 1e-12)
        mu = (w @ x) / sw
        var = (w * (x[None, :] - mu[:, None]) ** 2).sum(axis=1) / sw
        return mu, np.maximum(var, floor)

    def e_step(mu, var):
        # log domain avoids underflow for voxels far from every class mean
        logr = (
            np.log(p)
            - 0.5 * np.log(2.0 * np.pi * var)[:, None]
            - 0.5 * (x[None, :] - mu[:, None]) ** 2 / var[:, None]
        )
        logr -= logr.max(axis=0, keepdims=True)
        r = np.exp(logr)
        r /= np.maximum(r.sum(axis=0, keepdims=True), 1e-300)
        return r

    mu, var = moments(p)
    for _ in range(iterations):
        r = e_step(mu, var)
        mu_new, var = moments(r)
        delta = np.max(np.abs(mu_new - mu)) / std
        mu = mu_new
        if delta < tol:
            break
    r = e_step(mu, var)

    for i, k in enumerate(_TISSUE_KEYS):
        out[k][brain] = r[i].astype(np.float32)
    return out


def _mask_of(labels, ids, ranges=()):
    m = np.isin(labels, ids)
    for lo, hi in ranges:
        m |= (labels >= lo) & (labels <= hi)
    return m


def _nearest_index(mask):
    # EDT measures distance to the nearest zero, so invert the source mask
    _, idx = ndi.distance_transform_edt(~mask, return_indices=True)
    return tuple(idx)


def _reassign_islands(out, label_id, structure):
    """Relabel all but the largest 26-connected piece of label_id. Returns True if any."""
    coords = np.nonzero(out == label_id)
    if coords[0].size == 0:
        return False
    lo = [int(c.min()) for c in coords]
    hi = [int(c.max()) + 1 for c in coords]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    comps, ncomp = ndi.label(out[sl] == label_id, structure=structure)
    if ncomp <= 1:
        return False
    keep = int(np.argmax(np.bincount(comps.ravel())[1:])) + 1
    isl = np.nonzero((comps != keep) & (comps > 0))
    pos = [c + o for c, o in zip(isl, lo)]

    nvox = pos[0].size
    nb = np.empty((nvox, 2 * out.ndim), dtype=out.dtype)
    valid = np.zeros((nvox, 2 * out.ndim), dtype=bool)
    col = 0
    for ax in range(out.ndim):
        for step in (-1, 1):
            c = [q.copy() for q in pos]
            c[ax] = c[ax] + step
            inb = (c[ax] >= 0) & (c[ax] < out.shape[ax])
            c[ax] = np.clip(c[ax], 0, out.shape[ax] - 1)
            nb[:, col] = out[tuple(c)]
            # background is never a candidate: an island joins a real neighbouring label
            valid[:, col] = inb & (nb[:, col] != label_id) & (nb[:, col] != 0)
            col += 1

    same = (nb[:, :, None] == nb[:, None, :]) & valid[:, None, :]
    counts = np.where(valid, same.sum(axis=2), 0)
    best = counts.argmax(axis=1)
    chosen = nb[np.arange(nvox), best]
    has = valid.any(axis=1)
    out[tuple(p[has] for p in pos)] = chosen[has]
    return True


def constrain_labels(
    labels: np.ndarray,
    posteriors: dict[str, np.ndarray],
    *,
    scheme: str,
) -> np.ndarray:
    if scheme not in CORTEX_RANGES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {sorted(CORTEX_RANGES)}")

    labels = np.asarray(labels, dtype=np.int32)
    out = labels.copy()

    cort = _mask_of(labels, CORTEX_BASE_IDS, CORTEX_RANGES[scheme])
    cwm = np.isin(labels, CEREBRAL_WM_IDS)
    vent = np.isin(labels, VENTRICLE_IDS)

    post = np.stack([np.asarray(posteriors[k], dtype=np.float32) for k in _TISSUE_KEYS])
    has_post = post.sum(axis=0) > 0
    tissue = post.argmax(axis=0)

    # unlabelled voxels may join only next to the warped labels: posteriors inside a loose
    # brain mask would otherwise grow cortex into dura and skull
    near = ndi.binary_dilation(labels > 0, iterations=2)
    editable = (cort | cwm | vent | ((labels == 0) & near)) & has_post

    rh = _mask_of(labels, RH_IDS, RH_RANGES)
    lateral = rh | _mask_of(labels, LH_IDS, LH_RANGES)

    gm_sel = editable & (tissue == 0)
    if gm_sel.any() and cort.any():
        idx = _nearest_index(cort)
        out[gm_sel] = labels[idx][gm_sel]

    wm_sel = editable & (tissue == 1)
    if wm_sel.any() and lateral.any():
        idx = _nearest_index(lateral)
        is_rh = rh[idx]
        out[wm_sel] = np.where(is_rh, RH_WM_LABEL, LH_WM_LABEL)[wm_sel]

    csf_sel = editable & (tissue == 2)
    if csf_sel.any():
        if vent.any():
            hull = ndi.binary_fill_holes(ndi.binary_closing(vent, iterations=2)) | vent
            idx = _nearest_index(vent)
            near_vent = labels[idx]
            out[csf_sel] = np.where(hull, near_vent, CSF_LABEL)[csf_sel]
        else:
            out[csf_sel] = CSF_LABEL

    structure = ndi.generate_binary_structure(out.ndim, out.ndim)
    for _ in range(_MAX_ISLAND_PASSES):
        present = np.unique(out)
        targets = present[
            _mask_of(present, CORTEX_BASE_IDS, CORTEX_RANGES[scheme])
            | np.isin(present, VENTRICLE_IDS)
        ]
        found = False
        for lab in targets:
            found |= _reassign_islands(out, int(lab), structure)
        if not found:
            break

    return out
