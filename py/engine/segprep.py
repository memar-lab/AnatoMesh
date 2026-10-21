"""Label-volume preparation: holes, combined regions, SDF smoothing, voxel size, label tables."""

import gzip
import math
import queue
import struct
import sys
import threading
from pathlib import Path

import numpy as np
from scipy import ndimage


# =============================================================================================
# SETTINGS: change these
# =============================================================================================


DEFAULT_SIGMA_OUTER_MM = 0.3
DEFAULT_ITERATIONS_OUTER = 3
DEFAULT_SIGMA_INTERNAL_MM = 0.4
DEFAULT_ITERATIONS_INTERNAL = 5

                                      # with about this many faces is drawn instead (the full view returns when you stop)

# one colour per label id (labels not listed get the next colour of this list)
DEFAULT_FILL_MAX_VOXELS = 1000        # holes larger than this are not filled (0 = no limit)
DEFAULT_COMBINED_LABEL = 101          # label of the combined region (Stage 0)

# =============================================================================================
# NIfTI reading / writing (NIfTI-1, .nii or .nii.gz)
# =============================================================================================

DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8,
          512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def _open(path, mode):
    return gzip.open(path, mode) if str(path).lower().endswith(".gz") else open(path, mode)


def read_nifti(path):
    """Returns (data in storage order, voxel size (3,) in mm, header info dict used for saving)."""
    with _open(path, "rb") as stream:
        raw = stream.read()
    for endian in ("<", ">"):
        if struct.unpack(endian + "i", raw[:4])[0] == 348:
            break
    else:
        raise ValueError("Not a NIfTI-1 file (header size is not 348).")
    dim = struct.unpack(endian + "8h", raw[40:56])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    if datatype not in DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype code {datatype}.")
    ndim = dim[0]
    shape = tuple(dim[1:1 + ndim])
    if ndim != 3 and not (ndim > 3 and all(s == 1 for s in shape[3:])):
        raise ValueError(f"Expected a 3D volume, found dim = {dim[:ndim + 1]}.")
    shape = shape[:3]
    dtype = np.dtype(DTYPES[datatype]).newbyteorder(endian)
    count = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=vox_offset).reshape(shape, order="F")
    data = np.ascontiguousarray(data.astype(dtype.newbyteorder("="), copy=False))
    info = dict(header=raw[:vox_offset], endian=endian, dtype=dtype, shape=shape,
                voxel=np.array(pixdim[1:4], dtype=float))
    return data, np.array(pixdim[1:4], dtype=float), info


def _updated_header(info, shape, voxel):
    """Input header with dim, pixdim and the qform / sform of a lattice resampled to a new voxel size.
    Voxel k of the new lattice is centred at old index (k + 0.5) * f - 0.5, so the origin moves by M * (0.5 f - 0.5)."""
    e = info["endian"]
    header = bytearray(info["header"])
    old = np.asarray(info["voxel"], float)
    new = np.asarray(voxel, float)
    f = new / old
    shift = 0.5 * f - 0.5                                         # origin shift in old voxel units
    dim = list(struct.unpack(e + "8h", header[40:56]))
    dim[1:4] = [int(n) for n in shape]
    header[40:56] = struct.pack(e + "8h", *dim)
    pixdim = list(struct.unpack(e + "8f", header[76:108]))
    pixdim[1:4] = [float(v) for v in new]
    header[76:108] = struct.pack(e + "8f", *pixdim)

    qform_code, sform_code = struct.unpack(e + "2h", header[252:256])
    if qform_code > 0:                                            # quaternion form
        b, c, d = struct.unpack(e + "3f", header[256:268])
        qx, qy, qz = struct.unpack(e + "3f", header[268:280])
        a = math.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
        R = np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                      [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                      [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - c*c - b*b]])
        qfac = -1.0 if struct.unpack(e + "8f", bytes(info["header"][76:108]))[0] < 0 else 1.0
        M = R @ np.diag([old[0], old[1], qfac * old[2]])
        q = np.array([qx, qy, qz]) + M @ shift
        header[268:280] = struct.pack(e + "3f", *q)
    if sform_code > 0:                                            # affine form
        rows = np.array(struct.unpack(e + "12f", bytes(header[280:328]))).reshape(3, 4)
        shift_t = rows[:, :3] @ shift
        rows[:, :3] *= f                                          # column j scaled by the new / old voxel size
        rows[:, 3] += shift_t
        header[280:328] = struct.pack(e + "12f", *rows.ravel())
    return bytes(header)


def write_nifti(path, data, info, voxel=None):
    """Writes data with the header of the input file (same dtype and geometry).
    If voxel is given and differs from the input voxel size, dim / pixdim / qform / sform are updated."""
    out = np.ascontiguousarray(data).astype(info["dtype"]).ravel(order="F").tobytes()
    header = info["header"]
    if voxel is not None and not np.allclose(voxel, info["voxel"]):
        header = _updated_header(info, data.shape, voxel)
    with _open(path, "wb") as stream:
        stream.write(header)
        stream.write(out)


# =============================================================================================
# Smoothing (two stages: bwdist -> imgaussfilt3 -> threshold / label competition)
# =============================================================================================


def signed_distance(mask):
    """Positive inside, negative outside (bwdist(~mask) - bwdist(mask))."""
    return (ndimage.distance_transform_edt(mask).astype(np.float32)
            - ndimage.distance_transform_edt(~mask).astype(np.float32))


def gaussian3(volume, sigma_vox):
    """imgaussfilt3: separable Gaussian, kernel radius ceil(2 sigma), replicated borders."""
    out = volume
    for axis, sigma in enumerate(sigma_vox):
        if sigma > 0:
            out = ndimage.gaussian_filter1d(out, sigma, axis=axis, mode="nearest",
                                            truncate=math.ceil(2 * sigma) / sigma)
    return out


def bounding_slices(mask, margin):
    """Slices of the mask's bounding box grown by margin voxels (distances stay exact inside it)."""
    slices = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        present = np.nonzero(mask.any(axis=other))[0]
        if present.size == 0:
            slices.append(slice(0, mask.shape[axis]))
        else:
            slices.append(slice(max(0, present[0] - margin), min(mask.shape[axis], present[-1] + 1 + margin)))
    return tuple(slices)


def smooth_segmentation(original, voxel_size, smooth_outer, sigma_outer_mm, iterations_outer,
                        smooth_internal, sigma_internal_mm, iterations_internal, log=print):
    """Returns the smoothed label volume. All printed checks are part of the report."""
    sigma_outer = sigma_outer_mm / voxel_size
    sigma_internal = sigma_internal_mm / voxel_size

    # ---------------- STAGE 1: outer surface (0 vs any non-zero label) ----------------
    original_union = original != 0
    current = original_union
    if smooth_outer:
        box = bounding_slices(original_union, 2 * math.ceil(2 * sigma_outer.max()) + 5)
        cropped = current[box].copy()
        for it in range(1, iterations_outer + 1):
            log(f"Outer iteration {it} of {iterations_outer}...")
            cropped = gaussian3(signed_distance(cropped), sigma_outer) > 0
        current = np.zeros_like(original_union)
        current[box] = cropped
    smoothed_union = current

    after_outer = original.copy()
    lost = original_union & ~smoothed_union
    after_outer[lost] = 0
    gained = ~original_union & smoothed_union
    if gained.any():
        box = bounding_slices(original_union | smoothed_union, 1)
        _, nearest = ndimage.distance_transform_edt(~original_union[box], return_indices=True)
        gained_local = gained[box]
        view = original[box]
        after_box = after_outer[box]
        after_box[gained_local] = view[tuple(n[gained_local] for n in nearest)]
        after_outer[box] = after_box
    internal_changes_1 = int(np.count_nonzero((original != 0) & (after_outer != 0) & (original != after_outer)))
    log(f"Outer surface smoothing: gained {int(gained.sum())} voxels, lost {int(lost.sum())} voxels, "
        f"internal label changes = {internal_changes_1}"
        f"  ({'PASSED' if internal_changes_1 == 0 else 'WARNING: internal labels changed'})")

    # ---------------- STAGE 2: internal label boundaries inside the fixed outer mask ----------------
    fixed = smoothed_union
    current = after_outer
    if smooth_internal:
        box = bounding_slices(fixed, 2 * math.ceil(2 * sigma_internal.max()) + 5)
        fixed_c = fixed[box]
        cur = current[box].copy()
        for it in range(1, iterations_internal + 1):
            labels = [int(v) for v in np.unique(cur) if v != 0]
            log(f"Internal iteration {it} of {iterations_internal}... competing labels {labels}")
            best_val = np.full(cur.shape, -np.inf, dtype=np.float32)
            best_label = cur.copy()
            for label in labels:
                mask = cur == label
                if not mask.any():
                    continue
                smooth = gaussian3(signed_distance(mask), sigma_internal)
                update = fixed_c & (smooth > best_val)
                best_val[update] = smooth[update]
                best_label[update] = label
            best_label[~fixed_c] = 0
            cur = best_label
        current = np.zeros_like(after_outer)
        current[box] = cur
    updated = current

    outer_changes_2 = int(np.count_nonzero((updated != 0) != fixed))
    internal_changes_2 = int(np.count_nonzero(fixed & (after_outer != updated)))
    log(f"Inner surface smoothing: outer-mask voxels changed = {outer_changes_2}"
        f"  ({'PASSED' if outer_changes_2 == 0 else 'WARNING: outer surface changed'}), "
        f"internal voxels changed = {internal_changes_2}"
        f" ({100 * internal_changes_2 / max(1, int(fixed.sum())):.4f} % of tissue)")

    return updated


def resample_labels(seg, voxel_in, voxel_out, log=print):
    """Resample a label volume onto a lattice of voxel_out mm voxels (isotropic).
    Same method as the voxel mesher's label resampling: every label becomes a signed distance field,
    the SDFs are interpolated at the centres of the new voxels and the label with the largest value wins.
    The largest label (background) goes last and only wins where no other label is positive."""
    from scipy.interpolate import RegularGridInterpolator

    voxel_in = np.asarray(voxel_in, float)
    factor = voxel_out / voxel_in                                  # new voxel size in old voxels, per axis
    if np.allclose(factor, 1, atol=1e-9):
        return seg.copy()
    old_size = np.array(seg.shape)
    new_size = np.maximum(1, np.round(old_size / factor).astype(int))
    centres = [((np.arange(new_size[d]) + 0.5) * factor[d]) + 0.5 for d in range(3)]   # in 1-based old index space

    labels, counts = np.unique(seg, return_counts=True)
    labels = labels[np.argsort(counts, kind="stable")]             # ascending voxel count, background last
    best = np.full(tuple(new_size), -np.inf, dtype=np.float32)
    out = np.zeros(tuple(new_size), dtype=seg.dtype)

    for k, label in enumerate(labels):
        is_largest = k == len(labels) - 1
        log(f"Resampling label {int(label)} ({k + 1} of {len(labels)})...")
        inside = seg == label
        region = ~inside if is_largest else inside
        if not region.any():
            if is_largest:
                out[:] = label
            continue
        present = [np.nonzero(region.any(axis=tuple(a for a in range(3) if a != d)))[0] for d in range(3)]
        lo = np.array([max(p[0] - 3, 0) for p in present])         # 0-based inclusive box with a margin of 3
        hi = np.array([min(p[-1] + 3, old_size[d] - 1) for d, p in enumerate(present)])
        box = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
        part = inside[box]
        sdf = (ndimage.distance_transform_edt(part) - ndimage.distance_transform_edt(~part)).astype(np.float32)

        # new voxels whose centres are safely inside the box (1-based index centres)
        keep = []
        for d in range(3):
            lo1, hi1 = lo[d] + 1, hi[d] + 1
            ok = (centres[d] >= lo1 + (1 if lo1 > 1 else 0)) & (centres[d] <= hi1 - (1 if hi1 < old_size[d] else 0))
            keep.append(np.nonzero(ok)[0])
        value = np.full(tuple(new_size), np.inf if is_largest else -np.inf, dtype=np.float32)
        if all(len(kk) for kk in keep):
            grid = [np.arange(lo[d] + 1, hi[d] + 2, dtype=float) for d in range(3)]
            interp = RegularGridInterpolator(grid, sdf, method="linear", bounds_error=False, fill_value=None)
            pts = np.meshgrid(*[centres[d][keep[d]] for d in range(3)], indexing="ij")
            values = interp(np.stack([p.ravel() for p in pts], axis=1)).reshape(pts[0].shape)
            value[np.ix_(*keep)] = values
        win = value > best
        best[win] = value[win]
        out[win] = label
    return out


def element_counts(volume):
    """{label: number of elements}: every non-zero voxel becomes one hexahedral element (label 0 is not meshed)."""
    labels, counts = np.unique(volume, return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts) if l != 0}


def label_table(original, voxel, result, voxel_out, log=print):
    """Elements (non-zero voxels) per label before and after, then the volume per label in mm^3
    (volumes stay comparable when the voxel size changes)."""
    before, after = element_counts(original), element_counts(result)
    labels = sorted(set(before) | set(after))
    log("")
    log("ELEMENTS PER LABEL (label 0 not counted)")
    log(f"{'Label':>8} {'Before':>12} {'After':>12} {'Net':>10} {'Net %':>9}")
    for label in labels:
        b, a = before.get(label, 0), after.get(label, 0)
        pct = f"{100 * (a - b) / b:>+8.3f}%" if b else "      n/a"
        log(f"{label:>8} {b:>12,} {a:>12,} {a - b:>+10,} {pct}")
    b_all, a_all = sum(before.values()), sum(after.values())
    log(f"{'total':>8} {b_all:>12,} {a_all:>12,} {a_all - b_all:>+10,} {100 * (a_all - b_all) / b_all:>+8.3f}%")

    unit_before, unit_after = float(np.prod(voxel)), float(np.prod(voxel_out))
    log("")
    log("VOLUME PER LABEL")
    log(f"{'Label':>8} {'Before mm3':>12} {'After mm3':>12} {'Net mm3':>10} {'Net %':>9}")
    for label in labels:
        vb, va = before.get(label, 0) * unit_before, after.get(label, 0) * unit_after
        pct = f"{100 * (va - vb) / vb:>+8.3f}%" if vb else "      n/a"
        log(f"{label:>8} {vb:>12.2f} {va:>12.2f} {va - vb:>+10.2f} {pct}")


def find_holes(volume):
    """Holes are clusters of label 0 (6-connected, faces only) that do not touch the border of the volume, i.e. that are
    completely enclosed by labelled voxels. Returns (cluster number of every voxel, numbers of the enclosed clusters,
    size of every cluster)."""
    components, count = ndimage.label(volume == 0)
    touches_border = np.zeros(count + 1, dtype=bool)
    for axis in range(3):
        for index in (0, -1):
            touches_border[np.unique(np.take(components, index, axis=axis))] = True
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    holes = [i for i in range(1, count + 1) if not touches_border[i]]
    return components, holes, sizes


def fill_holes(volume, mode, label=0, max_size=0, log=print):
    """Fills the enclosed holes of volume. mode 'neighbour': every hole voxel takes the label of the nearest labelled voxel;
    mode 'label': every hole voxel gets `label`. Holes larger than max_size voxels are left (max_size 0 = no limit)."""
    components, holes, sizes = find_holes(volume)
    out = volume.copy()
    if not holes:
        log("Hole filling: no enclosed unlabelled holes found.")
        return out, dict(found=0, filled=0, voxels=0, skipped=0)
    chosen = [i for i in holes if max_size <= 0 or sizes[i] <= max_size]
    skipped = [i for i in holes if max_size > 0 and sizes[i] > max_size]
    lookup = np.zeros(len(sizes), dtype=bool)
    lookup[chosen] = True
    mask = lookup[components]
    if mask.any():
        if mode == "label":
            out[mask] = label
        else:
            _, nearest = ndimage.distance_transform_edt(volume == 0, return_indices=True)
            out[mask] = volume[tuple(n[mask] for n in nearest)]
    log(f"Hole filling: {len(holes)} enclosed holes ({int(sum(sizes[i] for i in holes)):,} voxels, "
        f"largest {int(max(sizes[i] for i in holes)):,}); filled {len(chosen)} holes = {int(mask.sum()):,} voxels"
        + (f"; left {len(skipped)} larger than {max_size:,} voxels" if skipped else ""))
    if mask.any():
        used, counts = np.unique(out[mask], return_counts=True)
        log("   voxels added per label: " + ", ".join(f"{int(l)}: {int(c):,}" for l, c in zip(used, counts)))
    return out, dict(found=len(holes), filled=len(chosen), voxels=int(mask.sum()), skipped=len(skipped))


def combine_labels(volume, selected, target):
    """Every voxel whose label is in selected gets the label target."""
    out = volume.copy()
    out[np.isin(volume, list(selected))] = target
    return out


class RegionMap:
    """Remembers which regions were combined, so the combined label can be turned back into the original labels
    at ANY voxel size (the smoothed / resampled volume has a different lattice than the original one).

    A voxel of the final volume that carries the combined label is placed in physical space (both lattices start at the
    same corner, so positions agree):
      * if its centre falls inside a voxel of one of the original regions, it takes that region;
      * otherwise (smoothing or resampling moved the boundary a little, so the centre now lies on background, CSF or another
        label) it takes the region of the NEAREST original-region voxel.
    So every combined voxel ends up with exactly one of the original region labels."""

    def __init__(self, original, voxel, selected, combined):
        self.voxel = np.asarray(voxel, float)
        self.selected = tuple(int(v) for v in selected)
        self.combined = int(combined)
        self.original = original                                  # kept by reference, never modified
        mask = np.isin(original, list(self.selected))
        self.points = np.argwhere(mask)                           # voxel indices of all region voxels
        self.point_labels = original[tuple(self.points.T)]
        self.counts = {int(l): int(c) for l, c in zip(*np.unique(original[mask], return_counts=True))}
        self._tree = None

    def prepare(self):
        """Build the nearest-voxel search once (done in the worker thread, so the window does not wait for it)."""
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree((self.points + 0.5) * self.voxel)

    def relabel(self, volume, spacing):
        """volume: labels on a lattice of `spacing` mm voxels (same corner as the original). Returns a copy in which every
        voxel of the combined label carries the label of its original region."""
        spacing = np.asarray(spacing, float)
        shape = self.original.shape
        index = [np.clip(np.floor((np.arange(n) + 0.5) * spacing[a] / self.voxel[a] + 1e-9).astype(int), 0, shape[a] - 1)
                 for a, n in enumerate(volume.shape)]
        direct = self.original[np.ix_(*index)]                    # original label at the voxel that contains each centre
        out = volume.copy()
        combined = volume == self.combined
        inside = combined & np.isin(direct, list(self.selected))
        out[inside] = direct[inside]
        rest = combined & ~inside
        if rest.any():
            self.prepare()
            centres = (np.argwhere(rest) + 0.5) * spacing
            _, nearest = self._tree.query(centres)
            out[rest] = self.point_labels[nearest]
        return out


def process(original, voxel, params, log=print):
    """Runs the enabled stages. Returns (smoothed on the input lattice, final result, final voxel size)."""
    smoothed = original
    if params["outer_on"] or params["int_on"]:
        smoothed = smooth_segmentation(original, voxel, params["outer_on"], params["sigma_out"], params["iter_out"],
                                       params["int_on"], params["sigma_int"], params["iter_int"], log=log)
    else:
        log("Smoothing is OFF: the segmentation is not smoothed.")
    final, voxel_out = smoothed, voxel
    if params["vox_on"]:
        voxel_out = np.full(3, float(params["voxel_out"]))
        log("")
        log(f"Voxel size: resampling {voxel[0]:.4g} x {voxel[1]:.4g} x {voxel[2]:.4g} mm  ->  {voxel_out[0]:.4g} mm voxels")
        final = resample_labels(smoothed, voxel, voxel_out, log=log)
        log(f"New lattice: {final.shape[0]} x {final.shape[1]} x {final.shape[2]}")
    else:
        log("Voxel-size change is OFF: the lattice is not resampled.")
    label_table(original, voxel, final, voxel_out, log)
    return smoothed, final, voxel_out
