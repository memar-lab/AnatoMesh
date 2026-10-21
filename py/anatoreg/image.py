"""ANTsImage: array + LPS geometry (origin, spacing, direction), the subset of antspyx the engine uses."""
from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage

_RAS2LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


class ANTsImage:
    def __init__(self, array, origin=None, spacing=None, direction=None):
        self._a = np.asarray(array)
        nd = self._a.ndim
        self.origin = tuple(float(v) for v in (origin if origin is not None else (0.0,) * nd))
        self.spacing = tuple(float(v) for v in (spacing if spacing is not None else (1.0,) * nd))
        self.direction = np.eye(nd) if direction is None else np.asarray(direction, dtype=float).reshape(nd, nd)

    shape = property(lambda s: s._a.shape)
    dimension = property(lambda s: s._a.ndim)
    pixeltype = property(lambda s: "float" if s._a.dtype.kind == "f" else "unsigned int")

    def numpy(self):
        return self._a.copy()

    def view(self):
        return self._a

    def clone(self, pixeltype=None):
        return ANTsImage(self._a.copy(), self.origin, self.spacing, self.direction)

    def new_image_like(self, array):
        return ANTsImage(np.asarray(array).reshape(self.shape), self.origin, self.spacing, self.direction)

    def affine_lps(self):
        m = np.eye(4)
        m[:3, :3] = self.direction * np.asarray(self.spacing)
        m[:3, 3] = self.origin
        return m

    def _binop(self, other, op):
        o = other._a if isinstance(other, ANTsImage) else other
        return self.new_image_like(op(self._a, o))

    __mul__ = lambda s, o: s._binop(o, np.multiply)
    __rmul__ = __mul__
    __add__ = lambda s, o: s._binop(o, np.add)
    __sub__ = lambda s, o: s._binop(o, np.subtract)
    __gt__ = lambda s, o: s._binop(o, np.greater)
    __ne__ = lambda s, o: s._binop(o, np.not_equal)

    def __getitem__(self, idx):
        return self._a[idx]


def image_read(path, pixeltype="float"):
    img = nib.load(str(path))
    a = np.asarray(img.dataobj, dtype=np.float32)
    m = _RAS2LPS @ img.affine
    sp = np.linalg.norm(m[:3, :3], axis=0)
    if a.ndim == 4:
        d = np.eye(4); d[:3, :3] = m[:3, :3] / sp
        return ANTsImage(a, tuple(m[:3, 3]) + (0.0,), tuple(sp) + (float(img.header.get_zooms()[3]),), d)
    return ANTsImage(a, m[:3, 3], sp, m[:3, :3] / sp)


def to_nifti(image):
    ras = _RAS2LPS @ image.affine_lps()
    return nib.Nifti1Image(np.ascontiguousarray(image.view()), ras)


def image_write(image, path):
    nib.save(to_nifti(image), str(path))


def from_numpy(data, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False):
    return ANTsImage(np.asarray(data), origin, spacing, direction)


def get_mask(image, low_thresh=None, high_thresh=None, cleanup=2):
    a = image.view().astype(np.float32)
    lo = a.mean() if low_thresh is None else low_thresh
    hi = a.max() if high_thresh is None else high_thresh
    m0 = (a >= lo) & (a <= hi)
    m = m0
    for c in range(int(cleanup), 0, -1):
        # antspyx: erode, keep the largest component, dilate, fill holes (ball radius c); retry smaller if emptied
        ball = np.linalg.norm(np.mgrid[-c:c + 1, -c:c + 1, -c:c + 1], axis=0) <= c
        m = ndimage.binary_erosion(m0, ball)
        lab, n = ndimage.label(m)
        if n:
            m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
            m = ndimage.binary_fill_holes(ndimage.binary_dilation(m, ball))
            break
        m = m0
    return image.new_image_like(m.astype(np.float32))


def slice_image(image, axis, idx, collapse_strategy=0):
    sl = [slice(None)] * image.dimension
    sl[axis] = idx
    keep = [i for i in range(image.dimension) if i != axis]
    d = image.direction[np.ix_(keep, keep)]
    return ANTsImage(image.view()[tuple(sl)], [image.origin[i] for i in keep],
                     [image.spacing[i] for i in keep], d)


def crop_indices(image, lowerind, upperind):
    sl = tuple(slice(int(l), int(u)) for l, u in zip(lowerind, upperind))
    origin = image.affine_lps() @ np.array([*[int(l) for l in lowerind], 1.0])
    return ANTsImage(image.view()[sl].copy(), origin[:3], image.spacing, image.direction)


def resample_image(image, resample_params, use_voxels=False, interp_type=1):
    """interp_type: 0 linear, 1 nearest (antspyx numbering), 2 gaussian, 3 windowed sinc, 4 bspline."""
    a = image.view()
    sp = np.asarray(image.spacing)
    p = np.asarray(resample_params, dtype=float)
    new_shape = np.rint(p).astype(int) if use_voxels else np.maximum(np.rint(np.asarray(a.shape) * sp / p), 1).astype(int)
    new_sp = sp * (np.asarray(a.shape) - 1) / np.maximum(p - 1, 1) if use_voxels else p
    order = {0: 1, 1: 0, 4: 3}.get(interp_type, 1)
    # ITK resampler: same origin, new spacing, output voxel j at input index j * new_sp / sp
    ratio = new_sp / sp
    grids = np.meshgrid(*[np.arange(n) * r for n, r in zip(new_shape, ratio)], indexing="ij")
    out = ndimage.map_coordinates(a.astype(np.float32), np.stack(grids), order=order, mode="nearest")
    return ANTsImage(out.astype(a.dtype), image.origin, new_sp, image.direction)
