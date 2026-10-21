"""Transforms in ITK's file formats (affine .mat, displacement .nii.gz) and image / point resampling.

A fixed -> moving map is T(x) for x a fixed-grid point in LPS mm. Stacks follow ants.apply_transforms:
transformlist[0] is applied to the point last... see apply_stack.
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio
from scipy import ndimage

from .image import ANTsImage, image_read, to_nifti
import nibabel as nib

CHUNK = 1 << 20          # points per block, keeps temporaries small in a browser tab


class AffineTx:
    """y = M (x - c) + c + t, ITK's MatrixOffsetTransformBase parameterisation (LPS mm)."""

    def __init__(self, matrix=None, translation=None, center=None):
        self.matrix = np.eye(3) if matrix is None else np.asarray(matrix, float).reshape(3, 3)
        self.translation = np.zeros(3) if translation is None else np.asarray(translation, float)
        self.center = np.zeros(3) if center is None else np.asarray(center, float)

    @property
    def parameters(self):
        return np.concatenate([self.matrix.ravel(), self.translation])

    @property
    def fixed_parameters(self):
        return self.center

    @property
    def offset(self):
        return self.translation + self.center - self.matrix @ self.center

    def apply(self, pts):
        return pts @ self.matrix.T + self.offset

    def inverse(self):
        mi = np.linalg.inv(self.matrix)
        return AffineTx(mi, -mi @ self.offset, np.zeros(3))

    def as_matrix4(self):
        m = np.eye(4); m[:3, :3] = self.matrix; m[:3, 3] = self.offset
        return m

    def write(self, path):
        sio.savemat(str(path), {"AffineTransform_double_3_3": np.append(self.matrix.ravel(), self.translation)[:, None],
                                "fixed": self.center[:, None]}, format="4")


def read_transform(path, dimension=3):
    m = sio.loadmat(str(path))
    key = next(k for k in m if k.startswith("AffineTransform"))
    p = np.asarray(m[key], float).ravel()
    return AffineTx(p[:9], p[9:12], np.asarray(m["fixed"], float).ravel())


class DispTx:
    """Dense displacement field on `grid` (an ANTsImage); disp has shape grid.shape + (3,), LPS mm."""

    def __init__(self, disp, grid):
        self.disp = np.asarray(disp, np.float32)
        self.grid = grid

    def sample(self, pts):
        idx = self.grid_index(pts)
        return np.stack([ndimage.map_coordinates(self.disp[..., k], idx, order=1, mode="nearest") for k in range(3)], 1)

    def grid_index(self, pts):
        m = self.grid.affine_lps()
        return (np.linalg.inv(m[:3, :3]) @ (pts - m[:3, 3]).T)

    def apply(self, pts):
        return pts + self.sample(pts)

    def write(self, path):
        a = self.disp.copy()
        img = nib.Nifti1Image(a[:, :, :, None, :], to_nifti(self.grid).affine)
        img.header.set_intent("vector")
        nib.save(img, str(path))


def read_warp(path):
    img = nib.load(str(path))
    a = np.asarray(img.dataobj, np.float32)
    return DispTx(a.reshape(a.shape[:3] + (3,)), _grid_of(img))


def _grid_of(img):
    from .image import _RAS2LPS
    m = _RAS2LPS @ img.affine
    sp = np.linalg.norm(m[:3, :3], axis=0)
    return ANTsImage(np.zeros(img.shape[:3], np.float32), m[:3, 3], sp, m[:3, :3] / sp)


def load_transform(path):
    p = str(path)
    if p.endswith(".mat"):
        return read_transform(p)
    if p.endswith(".nii.gz") or p.endswith(".nii"):
        return read_warp(p)
    raise ValueError(f"unsupported transform file {p}")


def apply_stack(txs, pts):
    """Map fixed-space points to moving space; transformlist[0] is applied to the point first
    (ants.registration fwdtransforms are [warp, affine]: the warp acts in fixed space, then the affine)."""
    for t in txs:
        pts = t.apply(pts)
    return pts


def _prep(transformlist, whichtoinvert):
    if isinstance(transformlist, (str, bytes)) or hasattr(transformlist, "__fspath__"):
        transformlist = [transformlist]
    names = [str(p) for p in transformlist]
    if whichtoinvert is None or all(w is None for w in whichtoinvert):
        # antspyx default: (matrix, warp) pair is the inverse stack
        pair = len(names) == 2 and ".mat" in names[0] and ".mat" not in names[1]
        whichtoinvert = [True, False] if pair else [False] * len(transformlist)
    invert = list(whichtoinvert)
    txs = []
    for p, inv in zip(transformlist, invert):
        t = load_transform(p) if isinstance(p, (str, bytes)) or hasattr(p, "__fspath__") else p
        if inv:
            t = t.inverse() if isinstance(t, AffineTx) else invert_disp(t)
        txs.append(t)
    return txs


def invert_disp(t, iters=12):
    """Fixed-point inverse of a displacement field on its own grid: v(x) = -u(x + v(x))."""
    g = t.grid
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in g.shape], indexing="ij")).reshape(3, -1).astype(np.float64)
    m = g.affine_lps()
    pts = (m[:3, :3] @ idx + m[:3, 3:4]).T
    v = np.zeros_like(pts)
    for _ in range(iters):
        v = -t.sample(pts + v)
    return DispTx(v.reshape(g.shape + (3,)), g)


def _grid_points(fixed, z0, z1):
    sh = fixed.shape
    ii, jj, kk = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), np.arange(z0, z1), indexing="ij")
    idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()]).astype(np.float64)
    m = fixed.affine_lps()
    return (m[:3, :3] @ idx + m[:3, 3:4]).T


def _to_index(moving, pts):
    m = moving.affine_lps()
    return np.linalg.inv(m[:3, :3]) @ (pts - m[:3, 3]).T


def _label_vote(a, idx, sigma_vox=1.0):
    """ITK GenericLabel: Gaussian-weighted vote of the neighbouring labels, ties to the lower label."""
    base = np.floor(idx).astype(np.int64)
    frac = idx - base
    out = np.rint(ndimage.map_coordinates(a, idx, order=0, mode="constant", cval=0.0)).astype(a.dtype)
    near = [np.clip(base[k][None] + np.arange(-1, 3)[:, None], 0, a.shape[k] - 1) for k in range(3)]
    w1 = [np.exp(-0.5 * ((np.arange(-1, 3)[:, None] - frac[k][None]) / sigma_vox) ** 2) for k in range(3)]
    labs, wts = [], []
    for i in range(4):
        for j in range(4):
            for l in range(4):
                labs.append(a[near[0][i], near[1][j], near[2][l]])
                wts.append(w1[0][i] * w1[1][j] * w1[2][l])
    labs = np.stack(labs); wts = np.stack(wts)
    best_s = np.full(labs.shape[1], -1.0); best_l = np.zeros(labs.shape[1], a.dtype)
    for u in np.unique(labs):
        s = (wts * (labs == u)).sum(0)
        better = s > best_s
        best_s[better] = s[better]; best_l[better] = u
    inside = np.all((idx >= -0.5) & (idx <= np.asarray(a.shape)[:, None] - 0.5), axis=0)
    return np.where(inside, best_l, 0)


def apply_transforms(fixed, moving, transformlist, interpolator="linear", imagetype=0, whichtoinvert=None,
                     defaultvalue=0, singleprecision=False, **kwargs):
    txs = _prep(transformlist, whichtoinvert)
    a = moving.view().astype(np.float32)
    n = int(np.prod(fixed.shape[:2]))
    step = max(1, CHUNK // n)
    out = np.empty(fixed.shape, np.float32)
    order = {"nearestNeighbor": 0, "linear": 1, "bSpline": 3, "genericLabel": 0, "multiLabel": 0}.get(interpolator, 1)
    for z0 in range(0, fixed.shape[2], step):
        z1 = min(z0 + step, fixed.shape[2])
        idx = _to_index(moving, apply_stack(txs, _grid_points(fixed, z0, z1)))
        if interpolator in ("genericLabel", "multiLabel"):
            vals = _label_vote_fast(a, idx)
        else:
            vals = ndimage.map_coordinates(a, idx, order=order, mode="constant", cval=defaultvalue)
        out[:, :, z0:z1] = vals.reshape(fixed.shape[0], fixed.shape[1], z1 - z0)
    return fixed.new_image_like(out)


def _label_vote_fast(a, idx):
    """Nearest label everywhere; Gaussian vote only where the 2x2x2 neighbourhood disagrees."""
    nn = ndimage.map_coordinates(a, idx, order=0, mode="constant", cval=0.0)
    base = np.floor(idx).astype(np.int64)
    mixed = np.zeros(idx.shape[1], bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = [np.clip(base[0] + dx, 0, a.shape[0] - 1), np.clip(base[1] + dy, 0, a.shape[1] - 1),
                     np.clip(base[2] + dz, 0, a.shape[2] - 1)]
                mixed |= a[c[0], c[1], c[2]] != nn
    if mixed.any():
        sel = np.flatnonzero(mixed)
        nn[sel] = _label_vote(a, idx[:, sel])
    return nn


def apply_transforms_to_points(dim, points, transformlist, whichtoinvert=None, **kwargs):
    """DataFrame-like input (x, y, z columns, LPS mm) -> same shape. Checked against antspyx 0.5.3: the points
    run through the same stack as an image grid (first listed first, flagged matrices inverted)."""
    txs = _prep(transformlist, whichtoinvert)
    p = np.asarray(points[["x", "y", "z"]], float) if hasattr(points, "columns") else np.asarray(points, float)
    p = apply_stack(txs, p)
    if hasattr(points, "columns"):
        out = points.copy(); out[["x", "y", "z"]] = p
        return out
    return p
