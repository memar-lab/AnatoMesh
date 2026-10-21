"""N4 bias field correction (Tustison et al. 2010, IEEE TMI 29:1310) on numpy/scipy, following ITK's N4 filter.

Log-domain loop: sharpen the histogram of log(v) - field (200 bins, Wiener deconvolution, FWHM 0.15, noise 0.01),
fit the residual with a cubic B-spline lattice by weighted least squares, add it to the field. The lattice has
`mesh` elements per axis at the first level and doubles per level; one level per entry of convergence["iters"].
The fit runs on the image shrunk by shrink_factor, the field is evaluated at full resolution.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .image import ANTsImage, get_mask

BINS, FWHM, NOISE = 200, 0.15, 0.01


def _sharpen(v, lo, hi):
    """Histogram-sharpened log intensities for the samples v (N3 step, as in ITK's N4)."""
    slope = (hi - lo) / (BINS - 1)
    pos = (v - lo) / slope
    i0 = np.minimum(pos.astype(np.int64), BINS - 2)
    w1 = pos - i0
    h = np.bincount(i0, 1 - w1, BINS) + np.bincount(i0 + 1, w1, BINS)
    n = 1 << int(np.ceil(np.log2(BINS)) + 1)
    off = (n - BINS) // 2
    hp = np.zeros(n); hp[off:off + BINS] = h
    vp = lo + (np.arange(n) - off) * slope
    sf = FWHM / slope
    expf, scale = 4 * np.log(2) / sf ** 2, 2 * np.sqrt(np.log(2) / np.pi) / sf
    k = np.arange(n); k = np.minimum(k, n - k)
    g = scale * np.exp(-k * k * expf)
    gf = np.fft.fft(g)
    u = np.maximum(np.fft.ifft(np.fft.fft(hp) * np.conj(gf) / (np.abs(gf) ** 2 + NOISE)).real, 0)
    num = np.fft.ifft(gf * np.fft.fft(vp * u)).real
    den = np.fft.ifft(gf * np.fft.fft(u)).real
    e = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return np.interp(pos + off, np.arange(n), e)


def _basis(n, elems, offset, step, span):
    """(n, elems + 3) cubic B-spline basis for samples offset + i * step over [0, span]."""
    p = np.clip((offset + np.arange(n) * step) / span, 0, 1) * elems
    s = np.minimum(np.floor(p).astype(int), elems - 1)
    u = p - s
    w = np.stack([(1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                  (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6], 1)
    b = np.zeros((n, elems + 3))
    for j in range(4):
        b[np.arange(n), s + j] = w[:, j]
    return b


def _apply(bs, c):
    """Lattice c -> grid: contract c with the three axis bases."""
    return np.einsum("xi,yj,zk,ijk->xyz", *bs, c, optimize=True)


def _adjoint(bs, f):
    return np.einsum("xi,yj,zk,xyz->ijk", *bs, f, optimize=True)


def _normal(bs, w):
    """A^T W A for the tensor-product basis, contracted one axis at a time."""
    t = np.einsum("xa,xb,xyz->abyz", bs[0], bs[0], w, optimize=True)
    t = np.einsum("yc,yd,abyz->abcdz", bs[1], bs[1], t, optimize=True)
    t = np.einsum("ze,zf,abcdz->abcdef", bs[2], bs[2], t, optimize=True)
    n = int(np.prod([b.shape[1] for b in bs]))
    return t.transpose(0, 2, 4, 1, 3, 5).reshape(n, n)


def n4_bias_field_correction(image, mask=None, rescale_intensities=False, shrink_factor=4, convergence=None,
                             spline_param=None, return_bias_field=False, verbose=False, weight_mask=None):
    """antspyx signature. spline_param: scalar = control-point spacing in mm, sequence = mesh elements per axis;
    default one element per axis at the first level (antspyx default)."""
    conv = convergence or {"iters": [50, 50, 50, 50], "tol": 1e-7}
    iters, tol = list(conv["iters"]), float(conv["tol"])
    a = image.view().astype(np.float32)
    ok = np.ones(a.shape, bool) if mask is None else mask.view() > 0
    ok &= a > 0
    if weight_mask is not None:
        ok &= weight_mask.view() > 0
    s = int(shrink_factor)
    off = s // 2
    sl = tuple(slice(off, None, s) for _ in range(3))
    ash, oksh = a[sl], ok[sl]
    span = np.maximum(np.asarray(a.shape) - 1, 1).astype(float)
    if spline_param is None:
        mesh = np.ones(3, int)
    elif np.ndim(spline_param) == 0:
        mesh = np.maximum(np.ceil(span * np.asarray(image.spacing) / float(spline_param)), 1).astype(int)
    else:
        mesh = np.asarray(spline_param, int)
    logv = np.log(ash[oksh].astype(np.float64))
    lo, hi = logv.min(), logv.max()
    field = np.zeros(ash.shape)                     # log bias field on the shrunk grid
    wts = oksh.astype(np.float64)
    lattices = []
    for lev, n_it in enumerate(iters):
        el = mesh * (1 << lev)
        bs = [_basis(ash.shape[k], el[k], off, s, span[k]) for k in range(3)]
        nm = _normal(bs, wts)
        nm[np.diag_indices_from(nm)] += 1e-6 * np.trace(nm) / nm.shape[0]
        chol = linalg.cho_factor(nm)
        c_lev = np.zeros(nm.shape[0])
        for it in range(n_it):
            lu = np.log(np.maximum(ash, 1e-12)) - field
            res = np.zeros(ash.shape)
            res[oksh] = lu[oksh] - _sharpen(lu[oksh], lu[oksh].min(), lu[oksh].max())
            c = linalg.cho_solve(chol, _adjoint(bs, res * wts).ravel())
            inc = _apply(bs, c.reshape([b.shape[1] for b in bs]))
            field += inc
            c_lev += c
            r = np.exp(inc[oksh])
            cv = r.std() / r.mean()
            if verbose:
                print(f"N4 level {lev} it {it}: cv {cv:.3g}")
            if cv < tol:
                break
        lattices.append((el, c_lev.reshape([b.shape[1] for b in bs])))
    full = np.zeros(a.shape, np.float32)
    for el, c in lattices:
        bs = [_basis(a.shape[k], el[k], 0, 1, span[k]) for k in range(3)]
        full += _apply(bs, c).astype(np.float32)
    bias = np.exp(full)
    if return_bias_field:
        return image.new_image_like(bias)
    out = a / bias
    if rescale_intensities:
        out = np.where(ok, np.interp(out, (out[ok].min(), out[ok].max()), (a[ok].min(), a[ok].max())), out)
    return image.new_image_like(out.astype(np.float32))
