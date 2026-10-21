"""Scalp and skull straight out of the T1, instead of offset envelopes."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage
from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

SKULL_MIN_MM = 2.0
SKULL_MAX_MM = 12.0
SKULL_TARGET_MM = 6.1
SKULL_MAX_GAP_FRACTION = 0.6
SCALP_MIN_MM = 2.0
SCALP_MAX_MM = 15.0
HEAD_SEAL_MM = 6.0


def _voxel_mm(affine: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob."""
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _fill_cavities(mask: np.ndarray) -> np.ndarray:
    """Fill background that cannot reach the volume edge."""
    padded = np.pad(~mask, 1, constant_values=True)
    outside = np.zeros_like(padded)
    outside[0, 0, 0] = True
    outside = ndimage.binary_propagation(outside, mask=padded)
    return ~outside[1:-1, 1:-1, 1:-1]


def _fill_cavities_sealed(mask: np.ndarray, seal: int) -> np.ndarray:
    """Fill cavities, sealing channels narrower than `2 * seal` voxels first."""
    if seal <= 0:
        return _fill_cavities(mask)
    grown = ndimage.binary_dilation(mask, iterations=seal)
    outside = ndimage.binary_dilation(~_fill_cavities(grown), iterations=seal)
    return ~outside


def head_mask_from_t1(t1: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """The whole head"""
    from skimage.filters import threshold_otsu

    finite = t1[np.isfinite(t1)]
    if not finite.size:
        raise ValueError("T1 volume is empty")
    lo, hi = np.percentile(finite, (1.0, 99.5))
    norm = np.clip((t1 - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    mask = norm > threshold_otsu(norm[np.isfinite(norm)])
    vox = _voxel_mm(affine)
    seal = max(1, int(round(HEAD_SEAL_MM / vox)))
    mask = _fill_cavities_sealed(mask, seal)
    mask = ndimage.binary_opening(mask, iterations=max(1, int(round(1.5 / vox))))
    mask = _largest_component(mask)
    mask = _fill_cavities_sealed(mask, seal)
    logger.info("head mask from T1: %d voxels (%.0f cm^3)", int(mask.sum()),
                mask.sum() * abs(np.linalg.det(affine[:3, :3])) / 1000.0)
    return mask


def skull_interior_from_t1(
    t1: np.ndarray, affine: np.ndarray, head: np.ndarray, inner: np.ndarray
) -> Optional[np.ndarray]:
    """Everything inside the skull's OUTER surface, from the T1's dark band."""
    vox = _voxel_mm(affine)
    shell = head & ~inner
    if not shell.any():
        logger.warning("no tissue outside the brain envelope; skipping T1 skull")
        return None

    values = t1[shell]
    values = values[np.isfinite(values)]
    if values.size < 100:
        return None
    from skimage.filters import threshold_otsu

    threshold = float(threshold_otsu(values))
    spacing = np.linalg.norm(affine[:3, :3], axis=0)

    out_from_dura = fast_kernels.distance_transform_edt(~inner, sampling=spacing)
    in_from_air = fast_kernels.distance_transform_edt(head, sampling=spacing)
    gap_mm = out_from_dura + in_from_air

    darkness = np.clip((threshold - t1) / max(threshold, 1e-6), 0.0, 1.0)
    darkness = ndimage.uniform_filter(darkness, size=max(3, int(round(5.0 / vox))))
    modulation = 1.0 + 0.8 * (darkness - 0.5)

    wanted_mm = SKULL_TARGET_MM * modulation
    thickness_field = np.minimum(wanted_mm, gap_mm * SKULL_MAX_GAP_FRACTION)
    thickness_field = np.minimum(thickness_field, gap_mm - SCALP_MIN_MM)
    thickness_field = np.clip(thickness_field, SKULL_MIN_MM, SKULL_MAX_MM)

    superior = int(np.argmax(np.abs(affine[2, :3])))
    flip = affine[2, superior] < 0
    shape = [1, 1, 1]
    shape[superior] = -1
    centre = float(np.median(np.argwhere(inner)[:, superior]))
    index = np.arange(interior_shape := shell.shape[superior]).reshape(shape)
    above = (index > centre) if not flip else (index < centre)
    vault = shell & above
    vault_gap = float(np.median(gap_mm[vault])) if vault.any() else 0.0
    if vault_gap < SKULL_MIN_MM + SCALP_MIN_MM:
        logger.warning("only %.1fmm between the dura and the outside of the head over the "
                       "vault, which is less than a %.0fmm skull plus a %.0fmm scalp -- this "
                       "volume looks skull-stripped or cropped. Falling back to the offset "
                       "envelope.", vault_gap, SKULL_MIN_MM, SCALP_MIN_MM)
        return None

    band = shell & (out_from_dura <= thickness_field)
    if not band.any():
        logger.warning("no room for a skull between the dura and the head surface; "
                       "skipping T1 skull")
        return None

    interior = _fill_cavities(inner | band)
    interior |= inner  # never inside the brain envelope
    interior &= head  # never outside the head

    thickness_mm = float(np.median(thickness_field[band & above])) if (band & above).any() \
        else float(np.median(thickness_field[band]))
    logger.info("skull from T1: %d voxels, vault thickness %.1fmm "
                "(Lillie et al. 2016: 6.1+/-1.5), gap available %.1fmm, "
                "range enforced %.0f-%.0f", int((interior & ~inner).sum()),
                thickness_mm, vault_gap, SKULL_MIN_MM, SKULL_MAX_MM)
    return interior


def _surface_voxels(mask: np.ndarray) -> int:
    """Voxel count on a mask's own boundary, for a mean-thickness estimate."""
    return int((mask & ~ndimage.binary_erosion(mask)).sum())


def derive_head_layers(
    t1: np.ndarray, affine: np.ndarray, brain: np.ndarray, meninges_gap_mm: float = 2.5,
    scalp_extent: str = "cranial", scalp_max_mm: float = SCALP_MAX_MM,
) -> dict:
    """Dura, skull and scalp interiors for one subject, all from the T1."""
    from engine.anatomy.anatomy_labels import _offset_mask

    out: dict = {"dura": None, "skull": None, "scalp": None, "sources": {}}
    vox = _voxel_mm(affine)

    head_raw = head_mask_from_t1(t1, affine)

    dura = _offset_mask(brain, meninges_gap_mm, meninges_gap_mm + 5.0, affine)

    room = max(1, int(round((SKULL_TARGET_MM + SCALP_MIN_MM) / vox)))
    clipped = (dura & ndimage.binary_erosion(head_raw, iterations=room)) | brain
    escaped = int((dura & ~clipped).sum())
    if escaped:
        logger.info("dura envelope: clipped %d voxels (%.0f cm^3) back inside the head, "
                    "leaving %.0fmm for bone and scalp", escaped,
                    escaped * abs(np.linalg.det(affine[:3, :3])) / 1000.0,
                    SKULL_MIN_MM + SCALP_MIN_MM)
    dura = _fill_cavities(clipped)
    out["dura"] = dura
    out["sources"]["dura"] = "offset_envelope, clipped inside the T1 head"

    head = head_raw | dura
    skull = skull_interior_from_t1(t1, affine, head, dura)
    if skull is not None:
        out["skull"] = skull
        out["sources"]["skull"] = "t1_dark_band"

    if scalp_extent == "cranial" and skull is not None:
        reach = max(1, int(round(scalp_max_mm / vox)))
        head = head & ndimage.binary_dilation(skull, iterations=reach)
        out["sources"]["scalp"] = f"t1_threshold, capped {scalp_max_mm:.0f}mm over the skull"
    else:
        out["sources"]["scalp"] = "t1_threshold, whole head"
    out["scalp"] = head | (skull if skull is not None else dura)

    for inner_name, outer_name in (("dura", "skull"), ("skull", "scalp")):
        a, b = out[inner_name], out[outer_name]
        if a is None or b is None:
            continue
        leak = int((a & ~b).sum())
        if leak:
            logger.warning("%s leaks %d voxels outside %s; unioning to restore nesting",
                           inner_name, leak, outer_name)
            out[outer_name] = b | a
    return out


SEG_SKULL_MIN_MM = 1.5    # thinnest skull kept (temporal squama is ~1-2 mm)
SEG_SKULL_MAX_MM = 20.0   # dark tissue farther than this from the cavity is not skull
SEG_SCALP_MAX_MM = 30.0   # soft tissue farther than this outside the skull is face or neck
SEG_CLOSE_MM = 4.0        # closes the bright diploe between the two dark tables
SEG_CSF_REACH_MM = 10.0   # grey CSF is taken up to this far outside the brain
SEG_BASE_MARGIN_MM = 10.0  # scalp kept down to this far below the cranial floor above it
SEG_BONE_TOUCH_MM = 3.0   # dark tissue counts as bone only if it reaches this close to the cavity


def skull_base_cut(icv: np.ndarray, affine: np.ndarray, margin_mm: float) -> np.ndarray:
    """Voxels no more than `margin_mm` under the cranial floor above them.

    The floor follows the skull base: under each front-back position, the lowest point of the
    cranial cavity there (anterior fossa high above the orbits, posterior fossa low), so the
    cranial scalp and the occipital muscle over the bone stay and the face drops out."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    superior = int(np.argmax(np.abs(affine[2, :3])))
    anterior = int(np.argmax(np.abs(affine[1, :3])))
    z_axis = np.arange(icv.shape[superior]) * np.sign(affine[2, superior])
    zz = np.broadcast_to(np.expand_dims(z_axis, tuple(a for a in range(3) if a != superior)),
                         icv.shape)
    floor = np.where(icv, zz, np.inf).min(axis=tuple(a for a in range(3) if a != anterior))
    seen = np.isfinite(floor)
    if seen.any():
        idx = np.arange(len(floor))
        floor = np.interp(idx, idx[seen], floor[seen])   # in front of / behind the cavity: its ends
    shape = [1, 1, 1]
    shape[anterior] = -1
    return zz >= floor.reshape(shape) - margin_mm / spacing[superior]


def segment_head_layers(t1: np.ndarray, affine: np.ndarray, brain: np.ndarray,
                        csf: Optional[np.ndarray] = None,
                        scalp_max_mm: float = SEG_SCALP_MAX_MM,
                        skull_min_mm: float = SEG_SKULL_MIN_MM,
                        extent: str = "cranial") -> dict:
    """Intracranial space, skull and scalp segmented from the T1's own intensities.

    Morphological skull and scalp segmentation of T1 MRI (Dogdas, Shattuck and Leahy 2005,
    Hum Brain Mapp 26:273, the BrainSuite method), driven by this subject's tissue levels:
    - air: the background outside the head; CSF: the median of the CSF label (ventricles);
      bone is darker than CSF on a T1, so `bone_threshold` = halfway between air and CSF;
    - intracranial space: the brain plus every voxel brighter than bone connected to it within
      SEG_CSF_REACH_MM (the subarachnoid CSF), filled;
    - outer skull: the intracranial space plus the dark voxels (< bone_threshold) around it,
      closed by SEG_CLOSE_MM so the brighter diploe between the two tables is skull, holes
      filled, then kept at least SEG_SKULL_MIN_MM and at most SEG_SKULL_MAX_MM thick;
    - scalp: the head itself (threshold, cavities filled), out to SEG_SCALP_MAX_MM from the
      skull and no lower than 10 mm under the brain, so the face and neck stay out; with
      `extent="full"` the whole imaged head down to the bottom of the image (the skull is
      still only the cranial vault and base: facial bones are not separable from air here,
      use skull_source="charm" for them).
    No thickness is assumed anywhere else. Returns interiors (dura = intracranial space, skull,
    scalp) like derive_head_layers, plus the measured thicknesses."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    vox = float(spacing.mean())
    t1 = np.where(np.isfinite(t1), t1, 0.0).astype(np.float32)
    head = head_mask_from_t1(t1, affine)
    out: dict = {"dura": None, "skull": None, "scalp": None, "sources": {}}

    air = float(np.median(t1[~ndimage.binary_dilation(head, iterations=3)]))
    csf_core = (ndimage.binary_erosion(csf, iterations=1) if csf is not None and csf.any()
                else None)
    if csf_core is not None and csf_core.sum() > 50:
        csf_level = float(np.median(t1[csf_core]))
    else:   # no CSF label: the darkest decile of the tissue just outside the brain
        rim = ndimage.binary_dilation(brain, iterations=max(1, int(round(3.0 / vox)))) & ~brain
        csf_level = float(np.percentile(t1[rim & head], 25))
    bone_threshold = air + 0.5 * (csf_level - air)
    # CSF is darker than grey matter; diploe marrow and scalp fat are brighter than it, so the
    # cavity takes only CSF-like intensities (between bone and halfway to grey matter)
    grey = brain & ~ndimage.binary_erosion(brain, iterations=max(1, int(round(2.0 / vox))))
    grey_level = float(np.median(t1[grey])) if grey.any() else 2.0 * csf_level
    csf_ceiling = csf_level + 0.5 * max(grey_level - csf_level, 0.0)
    report = {"air": round(air, 2), "csf": round(csf_level, 2), "grey": round(grey_level, 2),
              "bone_threshold": round(bone_threshold, 2), "csf_ceiling": round(csf_ceiling, 2)}

    # intracranial space: brain + connected CSF-like voxels close to it
    near = fast_kernels.distance_transform_edt(~brain, sampling=spacing) <= SEG_CSF_REACH_MM
    soft = (t1 >= bone_threshold) & (t1 <= csf_ceiling) & near & head
    icv = brain | soft
    lab, n = ndimage.label(icv)
    if n > 1:
        keep = np.unique(lab[brain])
        icv = np.isin(lab, keep[keep > 0])
    icv = ndimage.binary_opening(icv, iterations=1) | brain
    icv = _fill_cavities_sealed(icv, max(1, int(round(2.0 / vox))))

    # outer skull: dark tissue around the cavity, diploe closed in
    from_icv = fast_kernels.distance_transform_edt(~icv, sampling=spacing)
    dark = (t1 < bone_threshold) & head & (from_icv <= SEG_SKULL_MAX_MM)
    # bone is dark tissue continuous with the band around the cavity (the inner table), plus
    # pieces within one diploe gap of it (the outer table); the vitreous of the eyes is dark
    # too but sits apart from the orbital roof, behind bright orbital fat
    pieces, _n = ndimage.label(dark)
    inner = np.isin(pieces, np.unique(pieces[dark & (from_icv <= SEG_BONE_TOUCH_MM)]))
    gap = fast_kernels.distance_transform_edt(~inner, sampling=spacing) <= SEG_CLOSE_MM
    dark = np.isin(pieces, np.unique(pieces[dark & gap]))
    close = max(1, int(round(SEG_CLOSE_MM / vox)))
    outer = ndimage.binary_closing(icv | dark, iterations=close) | icv
    outer = ndimage.binary_opening(outer, iterations=max(1, int(round(1.0 / vox)))) | icv
    outer = _fill_cavities(outer)
    outer |= from_icv <= skull_min_mm                # never thinner than the thinnest bone
    outer &= from_icv <= SEG_SKULL_MAX_MM
    in_from_air = fast_kernels.distance_transform_edt(head, sampling=spacing)
    outer &= in_from_air >= 1.0                      # at least a skin layer outside the bone
    outer |= icv

    # the skull base is bone right under the cranial floor; deeper dark tissue (sinuses, the
    # pharynx, the orbits' air and fat planes) is face, not skull
    outer = (outer & skull_base_cut(icv, affine, SEG_SKULL_MAX_MM)) | icv
    from_skull = fast_kernels.distance_transform_edt(~outer, sampling=spacing)
    if extent == "full":
        # the whole head as imaged, down to the bottom of the image: face, jaw and neck too
        scalp = _fill_cavities(head) | outer
    else:
        # the head as imaged, cranial part only
        scalp = (head & (from_skull <= scalp_max_mm)
                 & skull_base_cut(icv, affine, SEG_BASE_MARGIN_MM)) | outer
    scalp = _largest_component(scalp)

    skull_t = fast_kernels.distance_transform_edt(~icv, sampling=spacing)[outer & ~ndimage.binary_erosion(outer)]
    scalp_t = from_skull[scalp & ~ndimage.binary_erosion(scalp)]
    report.update(skull_thickness_mm_p10_p50_p90=[round(float(v), 1) for v in np.percentile(skull_t, [10, 50, 90])],
                  scalp_thickness_mm_p10_p50_p90=[round(float(v), 1) for v in np.percentile(scalp_t, [10, 50, 90])],
                  icv_cm3=round(float(icv.sum() * abs(np.linalg.det(affine[:3, :3])) / 1000.0), 1))
    logger.info("head layers segmented from the T1: %s", report)
    out.update(dura=icv, skull=outer, scalp=scalp, report=report)
    out["sources"] = {"dura": "t1_segmented (CSF brighter than bone)",
                      "skull": "t1_segmented (dark band, diploe closed; Dogdas et al. 2005)",
                      "scalp": f"t1_segmented (head as imaged, up to {scalp_max_mm:.0f} mm out)"}
    return out


BONE_BAND = (0.15, 0.32)  # bone intensity / brain median on the DSURQE T2 (air ~0.1, muscle ~0.35+)
THIN_SKULL_MAX_MM = 0.45
THIN_SKULL_MIN_MM = 0.08
THIN_SKULL_SMOOTH_MM = 0.06


def thin_skull_from_template(image: np.ndarray, affine: np.ndarray, brain: np.ndarray,
                             band: tuple = BONE_BAND, max_mm: float = THIN_SKULL_MAX_MM,
                             min_mm: float = THIN_SKULL_MIN_MM,
                             smooth_mm: float = THIN_SKULL_SMOOTH_MM) -> tuple[np.ndarray, dict]:
    """Skull interior (brain + bone) traced off the dark bone band of an in-skull template.

    For specimens imaged inside the skull (DSURQE mouse). Bone is the voxels within
    `max_mm` of the brain whose intensity, relative to the brain median, sits between air
    and soft tissue, connected to the brain surface. Where no bone shows (foramina, skull
    base) the skull is closed at `min_mm`. The outer surface is smoothed as a signed
    distance so the hexes do not follow single-voxel noise.
    """
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    brain = _fill_cavities(np.asarray(brain, dtype=bool))
    ratio = np.asarray(image, np.float32) / max(float(np.median(image[brain])), 1e-6)
    out_from_brain = fast_kernels.distance_transform_edt(~brain, sampling=spacing)
    candidate = (ratio > band[0]) & (ratio < band[1]) & (out_from_brain > 0) \
        & (out_from_brain <= max_mm)
    parts, _n = ndimage.label(candidate | brain)
    bone = (parts == parts[tuple(np.argwhere(brain)[0])]) & candidate
    bone = ndimage.binary_opening(bone)
    measured = brain | bone
    measured = _fill_cavities(ndimage.binary_closing(measured, iterations=2) | measured)
    floor = out_from_brain <= min_mm
    outer = measured | floor
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:  # scipy's EDT releases the GIL
        away, into = pool.map(lambda m: fast_kernels.distance_transform_edt(m, sampling=spacing),
                              (~outer, outer))
    sdf = away - into
    sdf = ndimage.gaussian_filter(sdf, smooth_mm / spacing)
    outer = ((sdf <= 0) | floor) & (out_from_brain <= max_mm)
    outer = _fill_cavities(outer)
    surface = outer & ~ndimage.binary_erosion(outer)
    thickness = out_from_brain[surface]
    report = {"source": "template_dark_band", "band": list(band), "max_mm": max_mm,
              "min_mm": min_mm, "bone_voxels": int(bone.sum()),
              "thickness_mm_p5_p50_p95": [round(float(np.percentile(thickness, q)), 3)
                                          for q in (5, 50, 95)],
              "surface_without_detected_bone": round(1.0 - float(np.mean(
                  ndimage.binary_dilation(bone, iterations=2)[surface])), 3)}
    logger.info("thin skull from template: %s", report)
    return outer, report


def layer_thickness_mm(inner: np.ndarray, outer: np.ndarray, affine: np.ndarray) -> float:
    """Mean thickness of the shell between two nested masks, in mm."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = float(np.prod(spacing))
    shell = outer & ~inner
    area_mm2 = _surface_voxels(inner) * voxel_volume / _voxel_mm(affine)
    if area_mm2 <= 0:
        return 0.0
    return float(shell.sum()) * voxel_volume / area_mm2


def demo() -> None:
    """Self-check on a synthetic head with KNOWN layer thicknesses."""
    shape = (80, 80, 80)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    r = np.sqrt((zz - 40.0) ** 2 + (yy - 40.0) ** 2 + (xx - 40.0) ** 2)

    t1 = np.zeros(shape, dtype=np.float32)
    t1[r < 34] = 0.9  # scalp
    t1[r < 30] = 0.1  # skull, hypointense
    t1[r < 25] = 0.8  # brain
    t1 += np.random.default_rng(0).normal(0, 0.01, shape).astype(np.float32)
    affine = np.eye(4)
    brain = r < 25

    layers = derive_head_layers(t1, affine, brain, meninges_gap_mm=1.0)
    assert layers["scalp"] is not None and layers["skull"] is not None

    scalp_r = r[layers["scalp"] & ~ndimage.binary_erosion(layers["scalp"])].mean()
    assert 30 < scalp_r < 38, f"scalp surface at r={scalp_r:.1f}, expected ~34"
    assert not (layers["dura"] & ~layers["skull"]).any()
    assert not (layers["skull"] & ~layers["scalp"]).any()
    assert (layers["skull"] & ~layers["dura"]).sum() > 0, "skull has no thickness"

    bone_mm = layer_thickness_mm(layers["dura"], layers["skull"], affine)
    skin_mm = layer_thickness_mm(layers["skull"], layers["scalp"], affine)
    assert 3.0 <= bone_mm <= 7.0, f"skull came out {bone_mm:.1f}mm, phantom truth is 5mm"
    assert 2.0 <= skin_mm <= 6.0, f"scalp came out {skin_mm:.1f}mm, phantom truth is 4mm"

    surface = layers["scalp"] & ~ndimage.binary_erosion(layers["scalp"])
    on_surface = float((surface & layers["skull"]).sum()) / max(surface.sum(), 1)
    assert on_surface < 0.02, f"{on_surface:.0%} of the head surface is skull, not scalp"

    print(f"head_tissue.demo OK: scalp surface r={scalp_r:.1f}, "
          f"skull {bone_mm:.1f}mm (truth 5), scalp {skin_mm:.1f}mm (truth 4), "
          f"{on_surface:.1%} of the surface is bone")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
