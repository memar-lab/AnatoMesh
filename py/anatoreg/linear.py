"""Rigid / affine registration by Mattes MI, multi-resolution, analytic gradient, L-BFGS.

Geometry is LPS mm like the transform files. Transform: y = M (x - c) + c + t, fixed point x -> moving point y.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize

from .image import ANTsImage
from .metric import mattes_force
from .transform import AffineTx

SAMPLE_SEED = 1


def level_image(img: ANTsImage, shrink: int, sigma_vox: float):
    """ITK-style level: Gaussian smoothing (sigma in voxels of the full grid), then shrink by an integer factor."""
    a = img.view().astype(np.float32)
    if sigma_vox > 0:
        a = ndimage.gaussian_filter(a, sigma_vox, mode="nearest")
    if shrink == 1:
        return ANTsImage(a, img.origin, img.spacing, img.direction)
    n = np.asarray(a.shape) // shrink
    off = (shrink - 1) / 2.0
    grids = np.meshgrid(*[np.arange(k) * shrink + off for k in n], indexing="ij")
    if shrink % 2:
        a = a[off_slices(n, shrink, int(off))]
    else:
        a = ndimage.map_coordinates(a, np.stack(grids), order=1, mode="nearest")
    org = img.affine_lps() @ np.append(np.full(3, off), 1.0)
    return ANTsImage(np.ascontiguousarray(a), org[:3], np.asarray(img.spacing) * shrink, img.direction)


def off_slices(n, s, o):
    return tuple(slice(o, o + k * s, s) for k in n)


def center_of_mass(img: ANTsImage):
    a = img.view().astype(np.float64)
    w = a - a.min()
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in a.shape], indexing="ij")).reshape(3, -1)
    c = (idx * w.ravel()).sum(1) / w.sum()
    m = img.affine_lps()
    return m[:3, :3] @ c + m[:3, 3]


def _rot(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _rotvec(R):
    c = (np.trace(R) - 1) / 2
    th = np.arccos(np.clip(c, -1, 1))
    if th < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ax / (2 * np.sin(th)) * th


class _Level:
    """Fixed samples and moving image gradients for one pyramid level."""

    def __init__(self, fixed, moving, rate, center):
        self.f_img, self.m_img = fixed, moving
        a = fixed.view()
        n = a.size
        rs = np.random.RandomState(SAMPLE_SEED)
        sel = np.sort(rs.choice(n, size=max(int(n * rate), 1000), replace=False)) if rate < 1 else np.arange(n)
        ijk = np.stack(np.unravel_index(sel, a.shape))
        mf = fixed.affine_lps()
        self.x = (mf[:3, :3] @ ijk + mf[:3, 3:4])
        self.fv = a.ravel()[sel].astype(np.float64)
        self.q = self.x - center[:, None]
        mm = moving.affine_lps()
        self.mm_inv = np.linalg.inv(mm[:3, :3])
        self.mm_off = mm[:3, 3:4]
        ma = moving.view().astype(np.float32)
        self.ma = ma
        gi = np.gradient(ma)
        bt = np.linalg.inv(mm[:3, :3]).T
        self.g = [sum(bt[a_, b] * gi[b] for b in range(3)) for a_ in range(3)]
        self.rng = (self.fv.min(), self.fv.max(), float(ma.min()), float(ma.max()))
        self.center = center

    def eval(self, M, t):
        y = M @ self.q + (self.center + t)[:, None]
        idx = self.mm_inv @ (y - self.mm_off)
        inside = np.all((idx >= 0) & (idx <= np.asarray(self.ma.shape)[:, None] - 1), axis=0)
        if inside.sum() < 100:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        ii = idx[:, inside]
        mv = ndimage.map_coordinates(self.ma, ii, order=1, mode="nearest")
        mi, g = mattes_force(self.fv[inside], mv.astype(np.float64), rng=self.rng)
        gm = np.stack([ndimage.map_coordinates(self.g[a], ii, order=1, mode="nearest") for a in range(3)])
        gy = gm * g
        dt = gy.sum(1)
        dM = gy @ self.q[:, inside].T
        return mi, dt, dM


def register_linear(fixed, moving, kind, *, shrink, sigma, iters, rate, init: AffineTx | None = None,
                    radius=None, ftol=1e-7):
    """kind 'rigid' | 'affine'. Returns AffineTx (fixed -> moving)."""
    c = np.asarray(fixed.affine_lps()[:3, :3] @ (np.asarray(fixed.shape) - 1) / 2 + fixed.affine_lps()[:3, 3])
    if init is None:
        init = AffineTx(np.eye(3), center_of_mass(moving) - center_of_mass(fixed), np.zeros(3))
    # re-centre the initial transform on c
    M0 = init.matrix.copy()
    t0 = init.offset - c + M0 @ c        # y = M0 (x - c) + c + t0
    t0 = init.apply(c[None])[0] - c
    if radius is None:
        radius = 0.5 * float(np.min(np.asarray(fixed.shape) * np.asarray(fixed.spacing)))
    M, t = M0, t0
    for sh, sg, it in zip(shrink, sigma, iters):
        if it == 0:
            continue
        lv = _Level(level_image(fixed, sh, sg), level_image(moving, sh, sg), rate, c)
        if kind == "rigid":
            p0 = np.concatenate([_rotvec(M) * radius, t])

            def fun(p):
                R = _rot(p[:3] / radius)
                mi, dt, dM = lv.eval(R, p[3:])
                eps = 1e-5
                dR = np.zeros((3, 3, 3))
                for k in range(3):
                    e = np.zeros(3); e[k] = eps
                    dR[k] = (_rot(p[:3] / radius + e) - _rot(p[:3] / radius - e)) / (2 * eps)
                gr = np.array([(dM * dR[k]).sum() for k in range(3)]) / radius
                return -mi, -np.concatenate([gr, dt])
        else:
            p0 = np.concatenate([M.ravel() * radius, t])

            def fun(p):
                mi, dt, dM = lv.eval(p[:9].reshape(3, 3) / radius, p[9:])
                return -mi, -np.concatenate([dM.ravel() / radius, dt])
        res = optimize.minimize(fun, p0, jac=True, method="L-BFGS-B",
                                options={"maxiter": min(it, 150), "ftol": ftol, "gtol": 1e-9})
        p = res.x
        if kind == "rigid":
            M, t = _rot(p[:3] / radius), p[3:]
        else:
            M, t = p[:9].reshape(3, 3) / radius, p[9:]
    return AffineTx(M, t, c)
