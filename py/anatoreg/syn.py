"""Greedy diffeomorphic registration, multi-resolution, local-CC or Mattes-MI force.

Total map, fixed -> moving (the order ants applies [1Warp, 0GenericAffine]):  y = A (x + u(x)).
The field is kept per level in index units of that level's fixed grid, d(i). Each iteration:
force = dmetric/dw * grad(w) on the warped moving image w, Gaussian-smoothed (flow_sigma voxels), scaled so the
largest step is `step` voxels, composed into d (d <- v + d(. + v)), which keeps the map invertible for small steps.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

from .image import ANTsImage
from .linear import level_image
from .metric import mattes_force
from .transform import AffineTx, DispTx

VERBOSE = bool(os.environ.get("PYANTS_VERBOSE"))
WINDOW = 10


def _resample_disp(u, old: ANTsImage, new: ANTsImage):
    if u.shape[:3] == new.shape and old.origin == new.origin and old.spacing == new.spacing:
        return u
    mo, mn = old.affine_lps(), new.affine_lps()
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in new.shape], indexing="ij")).reshape(3, -1).astype(np.float64)
    idx = np.linalg.inv(mo[:3, :3]) @ (mn[:3, :3] @ ijk + (mn[:3, 3] - mo[:3, 3])[:, None])
    return np.stack([ndimage.map_coordinates(u[..., k], idx, order=1, mode="nearest").reshape(new.shape)
                     for k in range(3)], -1).astype(np.float32)


def _lin(d, img: ANTsImage, inv=False):
    """Displacement between index units (voxels) and LPS mm of img."""
    m = img.affine_lps()[:3, :3]
    return np.einsum("ab,...b->...a", (np.linalg.inv(m) if inv else m).astype(np.float32), d)


def _cc_force(f, w, radius, var_f, var_w, mask_var):
    """Local CC over a (2r+1)^3 box. Returns (mean cc, dcc/dw per voxel)."""
    size = 2 * radius + 1
    mf, mw = (ndimage.uniform_filter(a, size, mode="nearest") for a in (f, w))
    sff = ndimage.uniform_filter(f * f, size, mode="nearest") - mf * mf
    sww = ndimage.uniform_filter(w * w, size, mode="nearest") - mw * mw
    sfw = ndimage.uniform_filter(f * w, size, mode="nearest") - mf * mw
    den = sff * sww + 1e-4 * var_f * var_w
    g = 2.0 * sfw / den * ((f - mf) - (sfw / (sww + 1e-4 * var_w)) * (w - mw))
    g *= (sww > mask_var * var_w) & (sff > mask_var * var_f)     # no texture on one side: no information
    return float((sfw * sfw / den).mean()), g


def register_syn(fixed, moving, affine: AffineTx, *, metric="cc", step=0.25, flow_sigma=3.0, total_sigma=0.0,
                 iters, shrink, sigma, cc_radius=4, mask_var=0.01, conv=1e-6):
    """Returns DispTx on the fixed grid."""
    A, Aoff = affine.matrix, affine.offset
    d = None
    grid = None
    for sh, sg, it in zip(shrink, sigma, iters):
        fl = level_image(fixed, sh, sg)
        d = np.zeros(fl.shape + (3,), np.float32) if d is None else _lin(_resample_disp(_lin(d, grid), grid, fl), fl, inv=True)
        grid = fl
        if it == 0:
            continue
        r = max(1, int(round(float(np.mean(fl.spacing)) / float(np.mean(moving.spacing)))))
        ml = level_image(moving, r, sg * float(np.mean(fixed.spacing)) / float(np.mean(moving.spacing)))
        mf, mm = fl.affine_lps(), ml.affine_lps()
        mm_inv = np.linalg.inv(mm[:3, :3])
        shape = fl.shape
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")).astype(np.float32)
        # moving index of fixed index i with displacement d (voxels): idx = P0 + Q d
        P0 = np.einsum("ab,b...->a...", mm_inv @ A @ mf[:3, :3], ijk) + (mm_inv @ (A @ mf[:3, 3] + Aoff - mm[:3, 3])).reshape(3, 1, 1, 1)
        Q = (mm_inv @ A @ mf[:3, :3]).astype(np.float32)
        P0 = P0.astype(np.float32)
        f = fl.view().astype(np.float32)
        ma = ml.view().astype(np.float32)
        var_f, var_w = float(f.var()) + 1e-12, float(ma.var()) + 1e-12
        rng = (float(f.min()), float(f.max()), float(ma.min()), float(ma.max()))
        hist = []
        for k in range(it):
            idx = P0 + np.einsum("ab,...b->a...", Q, d)
            w = ndimage.map_coordinates(ma, idx, order=1, mode="nearest")
            if metric == "cc":
                val, g = _cc_force(f, w, cc_radius, var_f, var_w, mask_var)
            else:
                val, g = mattes_force(f.ravel().astype(np.float64), w.ravel().astype(np.float64), rng=rng)
                g = g.reshape(shape).astype(np.float32)
            hist.append(val)
            if len(hist) >= WINDOW and np.polyfit(np.arange(WINDOW), hist[-WINDOW:], 1)[0] < conv:
                break
            force = np.stack(np.gradient(w)) * g
            if flow_sigma > 0:
                force = np.stack([ndimage.gaussian_filter(force[a], flow_sigma, mode="nearest") for a in range(3)])
            mx = np.sqrt((force ** 2).sum(0)).max()
            if mx <= 0:
                break
            v = force * (step / mx)
            xi = ijk + v
            dn = np.stack([ndimage.map_coordinates(d[..., a], xi, order=1, mode="nearest") for a in range(3)], -1)
            d = (dn + np.moveaxis(v, 0, -1)).astype(np.float32)
            if total_sigma > 0:
                d = np.stack([ndimage.gaussian_filter(d[..., a], total_sigma, mode="nearest") for a in range(3)], -1)
        if VERBOSE:
            print(f"syn shrink {sh}: {len(hist)} its, {metric} {hist[0]:.4f} -> {hist[-1]:.4f}", flush=True)
    return DispTx(_resample_disp(_lin(d, grid), grid, fixed), fixed)
