"""Mattes mutual information (ITK conventions: fixed zero-order, moving cubic B-spline Parzen window).

mattes_force returns MI and dMI/d(moving intensity) per sample, the only thing both the linear
optimiser and the SyN update need.
"""
from __future__ import annotations

import numpy as np

PAD = 2


def _norm(v, lo, hi, bins):
    return (v - lo) * ((bins - 2 * PAD - 1) / max(hi - lo, 1e-12)) + PAD


def _bspline(u):
    u2, u3 = u * u, u * u * u
    w = np.stack([(1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6])
    d = np.stack([-(1 - u) ** 2 / 2, (9 * u2 - 12 * u) / 6, (-9 * u2 + 6 * u + 3) / 6, u2 / 2])
    return w, d


def mattes_force(f, m, bins=32, rng=None):
    """f, m: 1-D sample arrays (fixed, warped moving). Returns (mi, g) with g = dMI/dm per sample,
    already multiplied by 1/N so that summing g * grad(m) * dy/dp gives the parameter gradient.
    rng = (fmin, fmax, mmin, mmax) to pin the histogram range."""
    n = f.size
    fmin, fmax, mmin, mmax = rng if rng is not None else (f.min(), f.max(), m.min(), m.max())
    fi = np.clip(np.floor(_norm(f, fmin, fmax, bins)), PAD, bins - PAD - 1).astype(np.int64)
    mn = np.clip(_norm(m, mmin, mmax, bins), PAD, bins - PAD - 1e-6)
    j0 = np.floor(mn).astype(np.int64)
    w, dw = _bspline(mn - j0)
    pj = np.zeros(bins * bins)
    base = fi * bins + j0 - 1
    for k in range(4):
        pj += np.bincount(base + k, weights=w[k], minlength=bins * bins)
    p = pj.reshape(bins, bins) / n
    pm = p.sum(0)
    eps = 1e-12
    pf = p.sum(1)
    nz = p > eps
    mi = float((p[nz] * np.log(p[nz] / (pf[:, None] * pm[None, :])[nz])).sum())
    lg = np.log((p + eps) / (pm[None, :] + eps))
    scale = (bins - 2 * PAD - 1) / max(mmax - mmin, 1e-12)
    g = np.zeros(n)
    for k in range(4):
        g += dw[k] * lg[fi, j0 + k - 1]
    return mi, g * (scale / n)
