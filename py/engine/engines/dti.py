"""Diffusion tensor fit (replaces dipy's TensorModel and DSI Studio's DTI reconstruction).

Method: the single-tensor model S = S0 exp(-b g^T D g) (Basser, Mattiello and LeBihan 1994,
doi:10.1016/S0006-3495(94)80775-1), fitted to log signal by ordinary least squares and then
by weighted least squares with weights equal to the squared OLS prediction (Salvador et al.
2005, doi:10.1002/hbm.20076; Chung, Lu and Henry 2006, doi:10.1016/j.neuroimage.2006.06.019).
Eigenvalues are floored at a minimum diffusivity; FA from the eigenvalues (Basser and Pierpaoli
1996). Every voxel is independent, so the fit runs in parallel over the mask.

Gradient frame: b-vectors are taken in the frame they are given in (for FSL tables, the image
array axes, with x flipped when the affine has a positive determinant). `array_frame_bvecs`
converts an FSL table to the array frame so that e1 is in array-axis coordinates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit, prange

from engine.engines import kernels  # noqa: F401  (sets the thread count)

logger = logging.getLogger(__name__)


@dataclass
class TensorSettings:
    b0_threshold: float = 50.0          # b-values at or under this count as b = 0 (mask, shells)
    zero_small_b: bool = False          # set those b to 0 in the design (dipy keeps them as given)
    max_b: Optional[float] = None       # fit only shells at or under this b (None: all shells)
    fit: str = "wls"                    # "wls" (weighted, the default) or "ols"
    min_diffusivity_scale: float = 1e-6  # eigenvalue floor = this / -min(design), as dipy
    min_signal: float = 1e-4            # signal floor before the log (dipy MIN_POSITIVE_SIGNAL)
    renormalise_bvecs: bool = False     # dipy uses the table as written
    mask_otsu_scale: float = 0.5        # automatic b0 mask: share of the Otsu threshold kept
    extra: dict = field(default_factory=dict)


def read_gradients(bval_path: Path, bvec_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """b-values (n,) and b-vectors (n, 3) from FSL/BIDS text tables, as written."""
    bvals = np.loadtxt(str(bval_path), dtype=np.float64).ravel()
    bvecs = np.loadtxt(str(bvec_path), dtype=np.float64)
    if bvecs.ndim != 2:
        raise ValueError(f"{bvec_path} is not a 3 x n or n x 3 table")
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    elif bvecs.shape[0] == 3 and bvecs.shape[1] == 3 and len(bvals) == 3:
        bvecs = bvecs.T
    if len(bvecs) != len(bvals):
        raise ValueError(f"{len(bvals)} b-values but {len(bvecs)} b-vectors")
    return bvals, bvecs


def unit_bvecs(bvecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    return np.divide(bvecs, norms, out=np.zeros_like(bvecs), where=norms > 1e-8)


def array_frame_bvecs(bvecs: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """FSL b-vectors in the image array frame (FSL flips x when the affine is not radiological)."""
    out = np.array(bvecs, dtype=np.float64, copy=True)
    if np.linalg.det(np.asarray(affine, float)[:3, :3]) > 0:
        out[:, 0] = -out[:, 0]
    return out


def design_matrix(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    """Rows [-b gx^2, -b gy^2, -b gz^2, -2b gx gy, -2b gx gz, -2b gy gz, 1]."""
    g, b = np.asarray(bvecs, float), np.asarray(bvals, float)
    return np.column_stack([-b * g[:, 0] ** 2, -b * g[:, 1] ** 2, -b * g[:, 2] ** 2,
                            -2 * b * g[:, 0] * g[:, 1], -2 * b * g[:, 0] * g[:, 2],
                            -2 * b * g[:, 1] * g[:, 2], np.ones(len(b))])


@njit(cache=True, parallel=True)
def _fit(signal, design, pinv, min_signal, min_diffusivity, weighted,
         fa, md, evals_out, e1, s0):
    n_vox, n_meas = signal.shape
    for v in prange(n_vox):
        y = np.empty(n_meas)
        for m in range(n_meas):
            s = signal[v, m]
            y[m] = np.log(s if s > min_signal else min_signal)
        x = pinv @ y
        if weighted:
            # weights = squared OLS-predicted signal (dipy scales rows by the prediction and
            # takes the pinv; the normal equations give the same fit within 1e-6 in FA)
            pred = design @ x
            w = np.exp(2.0 * pred)
            a = np.zeros((7, 7))
            rhs = np.zeros(7)
            for m in range(n_meas):
                for r in range(7):
                    wr = w[m] * design[m, r]
                    rhs[r] += wr * y[m]
                    for c in range(r, 7):
                        a[r, c] += wr * design[m, c]
            for r in range(7):
                for c in range(r):
                    a[r, c] = a[c, r]
            ok = True
            for r in range(7):
                if not np.isfinite(a[r, r]) or a[r, r] <= 0:
                    ok = False
            if ok:
                x = np.linalg.solve(a, rhs)
        t = np.empty((3, 3))
        t[0, 0], t[1, 1], t[2, 2] = x[0], x[1], x[2]
        t[0, 1] = t[1, 0] = x[3]
        t[0, 2] = t[2, 0] = x[4]
        t[1, 2] = t[2, 1] = x[5]
        if not np.all(np.isfinite(t)):
            fa[v] = 0.0
            md[v] = 0.0
            s0[v] = 0.0
            for c in range(3):
                evals_out[v, c] = 0.0
                e1[v, c] = 0.0
            continue
        w3, vec = np.linalg.eigh(t)          # ascending
        l1 = max(w3[2], min_diffusivity)
        l2 = max(w3[1], min_diffusivity)
        l3 = max(w3[0], min_diffusivity)
        mean = (l1 + l2 + l3) / 3.0
        num = (l1 - mean) ** 2 + (l2 - mean) ** 2 + (l3 - mean) ** 2
        den = l1 * l1 + l2 * l2 + l3 * l3
        fa[v] = np.sqrt(1.5 * num / den) if den > 0 else 0.0
        md[v] = mean
        evals_out[v, 0], evals_out[v, 1], evals_out[v, 2] = l1, l2, l3
        for c in range(3):
            e1[v, c] = vec[c, 2]
        s0[v] = np.exp(x[6])


def b0_mask(dwi: np.ndarray, bvals: np.ndarray, settings: TensorSettings) -> np.ndarray:
    """Brain mask from the mean b0: Otsu threshold (scaled), largest component, holes filled."""
    from scipy import ndimage

    b0 = dwi[..., bvals <= settings.b0_threshold].mean(axis=-1)
    level = kernels.otsu_threshold(b0[b0 > 0]) * float(settings.mask_otsu_scale)
    mask = b0 > level
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == int(np.argmax(sizes))
    return ndimage.binary_fill_holes(mask)


def fit_tensor(dwi: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
               mask: Optional[np.ndarray] = None,
               settings: Optional[TensorSettings] = None) -> dict:
    """Tensor maps for a 4D DWI (x, y, z, n). Returns fa, md, evals (x,y,z,3, descending),
    e1 (x,y,z,3, in the frame of `bvecs`), s0 and the mask used."""
    settings = settings or TensorSettings()
    dwi = np.asarray(dwi)
    bvals = np.asarray(bvals, float)
    bvecs = np.asarray(bvecs, float)
    use = np.ones(len(bvals), bool)
    if settings.max_b is not None:
        use = bvals <= float(settings.max_b) + 1e-6
    b = (np.where(bvals <= settings.b0_threshold, 0.0, bvals) if settings.zero_small_b
         else bvals)[use]
    g = (unit_bvecs(bvecs) if settings.renormalise_bvecs else bvecs)[use]
    if mask is None:
        mask = b0_mask(dwi, bvals, settings)
    mask = np.asarray(mask, bool)
    design = design_matrix(b, g)
    pinv = np.linalg.pinv(design)
    signal = dwi[mask][:, use].astype(np.float64)
    min_signal = float(settings.min_signal)
    min_diffusivity = float(settings.min_diffusivity_scale) / max(-float(design.min()), 1e-12)
    n = len(signal)
    fa, md, s0 = np.zeros(n), np.zeros(n), np.zeros(n)
    evals, e1 = np.zeros((n, 3)), np.zeros((n, 3))
    _fit(np.ascontiguousarray(signal), design, pinv, min_signal, min_diffusivity,
         settings.fit == "wls", fa, md, evals, e1, s0)

    def full(values, tail=()):
        out = np.zeros(mask.shape + tail, dtype=np.float32)
        out[mask] = values
        return out

    return {"fa": full(fa), "md": full(md), "evals": full(evals, (3,)), "e1": full(e1, (3,)),
            "s0": full(s0), "mask": mask,
            "settings": {"fit": settings.fit, "max_b": settings.max_b,
                         "shells": sorted({int(round(v, -1)) for v in b}),
                         "measurements": int(use.sum()), "voxels": int(n)}}


def demo() -> None:
    """Synthetic check: a known tensor is recovered exactly from noise-free signal."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=(30, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    bvals = np.r_[np.zeros(3), np.full(30, 1000.0)]
    bvecs = np.r_[np.zeros((3, 3)), g]
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    d = rot @ np.diag([1.7e-3, 0.3e-3, 0.2e-3]) @ rot.T
    s = 1000.0 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, d, bvecs))
    dwi = np.tile(s, (4, 4, 4, 1)).astype(np.float32)
    out = fit_tensor(dwi, bvals, bvecs, mask=np.ones((4, 4, 4), bool))
    want = 1.7e-3, 0.3e-3, 0.2e-3
    assert np.allclose(out["evals"][1, 1, 1], want, rtol=1e-3), out["evals"][1, 1, 1]
    assert abs(abs(float(out["e1"][1, 1, 1] @ rot[:, 0])) - 1.0) < 1e-4
    print("dti.demo ok: FA", round(float(out["fa"][1, 1, 1]), 4))


if __name__ == "__main__":
    demo()
