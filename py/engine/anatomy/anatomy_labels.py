"""Assemble every head structure into ONE label volume, folds intact."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage
from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

REGION_LABELS: dict[str, int] = {
    "gray_matter": 1,
    "white_matter": 2,
    "brainstem": 3,
    "cerebellum": 4,
    "deep_nuclei": 5,
    "csf": 6,
    "skull": 7,
    "scalp": 8,
}
LABEL_REGIONS: dict[int, str] = {v: k for k, v in REGION_LABELS.items()}

BRAIN_TISSUE = ("gray_matter", "white_matter", "brainstem", "cerebellum", "deep_nuclei")

REPAIR_PRIORITY = [2, 3, 4, 5, 1, 6, 7, 8, 0]


def _voxel_size(affine: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))


def _offset_mask(
    mask: np.ndarray, offset_mm: float, closing_mm: float, affine: np.ndarray
) -> np.ndarray:
    """Grow `mask` outward by `offset_mm`, smoothing over concavities."""
    if not mask.any():
        return np.zeros_like(mask)
    vox = _voxel_size(affine)
    offset_vox = max(1, int(round(offset_mm / vox)))
    closing_vox = max(offset_vox, int(round(closing_mm / vox)))

    # work in the mask's box plus the closing reach, zero-padded where that runs past the
    # volume, so the result does not depend on how much background the volume carries
    margin = closing_vox + 2
    idx = np.argwhere(mask)
    want_lo = idx.min(axis=0) - margin
    want_hi = idx.max(axis=0) + margin + 1
    lo = np.maximum(want_lo, 0)
    hi = np.minimum(want_hi, np.array(mask.shape))
    view = tuple(slice(a, b) for a, b in zip(lo, hi))
    pad = [(int(a), int(b)) for a, b in zip(lo - want_lo, want_hi - hi)]
    region = np.pad(mask[view], pad)

    dilated = ndimage.distance_transform_edt(
        ~region, sampling=np.linalg.norm(affine[:3, :3], axis=0)) <= offset_mm
    envelope = binary_closing_cube(dilated, closing_vox)
    inner = tuple(slice(a, n - b) for (a, b), n in zip(pad, region.shape))
    out = np.zeros_like(mask)
    out[view] = (envelope | dilated)[inner]
    return out


def binary_closing_cube(mask: np.ndarray, iterations: int) -> np.ndarray:
    """`binary_closing` with a 3x3x3 cube repeated `iterations` times, as two sliding filters."""
    size = 2 * int(iterations) + 1
    grown = ndimage.maximum_filter(mask.astype(np.uint8), size=size, mode="constant", cval=0)
    return ndimage.minimum_filter(grown, size=size, mode="constant", cval=0).astype(bool)


BRAIN_FRACTION_OF_ICV = 0.87
GAP_MIN_MM = 0.1
GAP_MAX_MM = 4.0


def subarachnoid_gap_mm(
    brain: np.ndarray, affine: np.ndarray,
    target_fraction: float = BRAIN_FRACTION_OF_ICV,
    sulcal_csf: Optional[np.ndarray] = None,
) -> tuple[float, dict]:
    """Solve for the gap that puts the brain at `target_fraction` of the ICV."""
    spacing = np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)
    voxel_cm3 = float(np.prod(spacing)) / 1000.0
    brain = np.asarray(brain, dtype=bool)
    brain_cm3 = float(brain.sum()) * voxel_cm3
    if not brain_cm3:
        return GAP_MIN_MM, {}

    inside = brain if sulcal_csf is None else (brain | np.asarray(sulcal_csf, dtype=bool))
    distance = fast_kernels.distance_transform_edt(~inside, sampling=spacing)

    def fraction_at(gap: float) -> float:
        envelope = _fill_enclosed_cavities(distance <= gap)
        icv = float(envelope.sum()) * voxel_cm3
        return brain_cm3 / icv if icv else 1.0

    lo, hi = GAP_MIN_MM, GAP_MAX_MM
    if fraction_at(lo) < target_fraction:
        gap = lo  # already too much fluid at the floor
    elif fraction_at(hi) > target_fraction:
        gap = hi
    else:
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if fraction_at(mid) > target_fraction:
                lo = mid
            else:
                hi = mid
        gap = 0.5 * (lo + hi)

    achieved = fraction_at(gap)
    report = {
        "gap_mm": round(float(gap), 3),
        "target_brain_fraction": float(target_fraction),
        "achieved_brain_fraction": round(float(achieved), 4),
        "brain_cm3": round(brain_cm3, 3),
        "icv_cm3": round(brain_cm3 / achieved, 3) if achieved else 0.0,
    }
    logger.info("subarachnoid_gap_mm: %.3fmm puts %.2fcm^3 of brain at %.1f%% of a "
                "%.2fcm^3 cavity (target %.0f%%)", gap, brain_cm3, 100 * achieved,
                report["icv_cm3"], 100 * target_fraction)
    return float(gap), report


def smooth_labels_sdf(
    labels: np.ndarray, affine: np.ndarray, sigma_mm: float, iterations: int = 2,
    preserve_volume: bool = True, sigma_inner_mm: Optional[float] = None,
    iterations_inner: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """De-staircase a multi-label volume in the signed-distance domain."""
    labels = np.asarray(labels)
    if sigma_mm <= 0 or iterations <= 0:
        return labels, {}
    if iterations_inner is None:
        iterations_inner = int(iterations)
    if sigma_inner_mm is None:
        sigma_inner_mm = float(sigma_mm)
    spacing = np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)

    values = [int(v) for v in np.unique(labels)]
    before = {v: int((labels == v).sum()) for v in values}
    current = labels.copy()

    def _sdf(mask: np.ndarray) -> np.ndarray:
        return (fast_kernels.distance_transform_edt(mask, sampling=spacing)
                - fast_kernels.distance_transform_edt(~mask, sampling=spacing))

    def _assign(fields: dict, bias: dict, restrict: Optional[np.ndarray] = None) -> np.ndarray:
        best_val = None
        best_lab = None
        for value, field in fields.items():
            shifted = field + bias.get(value, 0.0)
            if best_val is None:
                best_val = shifted.copy()
                best_lab = np.full(shifted.shape, value, dtype=labels.dtype)
                continue
            upd = shifted > best_val
            if restrict is not None:
                upd &= restrict
            best_val[upd] = shifted[upd]
            best_lab[upd] = value
        if restrict is not None:
            best_lab[~restrict] = 0
        return best_lab

    def _wins(field_shifted: np.ndarray, rivals_max: np.ndarray,
              restrict: np.ndarray) -> int:
        """Voxels where a label's shifted field strictly beats its rivals."""
        return int(((field_shifted > rivals_max) & restrict).sum())

    report_stages: dict = {}

    union = labels != 0
    target_union = int(union.sum())
    sigma_vox_outer = float(sigma_mm) / spacing
    mask = union.copy()
    gained_lost = {"gained": 0, "lost": 0}
    for _ in range(int(iterations)):
        sdf = _sdf(mask).astype(np.float32)
        sdf = ndimage.gaussian_filter(sdf, sigma=sigma_vox_outer)
        if preserve_volume and target_union:
            lo, hi = -2.0, 2.0
            for _step in range(18):
                mid = 0.5 * (lo + hi)
                if int(((sdf + mid) > 0).sum()) < target_union:
                    lo = mid
                else:
                    hi = mid
            sdf = sdf + 0.5 * (lo + hi)
        mask = sdf > 0
    smoothed_union = mask
    gained_lost["lost"] = int((union & ~smoothed_union).sum())
    gained_lost["gained"] = int((~union & smoothed_union).sum())

    after_outer = labels.copy()
    after_outer[union & ~smoothed_union] = 0
    gained = ~union & smoothed_union
    if gained.any():
        _, nearest = ndimage.distance_transform_edt(~union, sampling=spacing,
                                                    return_indices=True)
        after_outer[gained] = labels[tuple(i[gained] for i in nearest)]
    internal_changes_stage1 = int((union & (labels != after_outer)).sum() - gained_lost["lost"])
    report_stages["outer"] = {
        "sigma_mm": round(float(sigma_mm), 3), "iterations": int(iterations),
        "voxels_gained": gained_lost["gained"], "voxels_lost": gained_lost["lost"],
        "internal_label_changes": max(internal_changes_stage1, 0),
    }
    logger.info("smooth_labels_sdf stage 1 (outer): sigma %.2fmm x%d, +%d/-%d voxel(s), "
                "%d internal change(s)", sigma_mm, iterations,
                gained_lost["gained"], gained_lost["lost"], max(internal_changes_stage1, 0))
    current = after_outer

    fixed = smoothed_union
    inner_values = [v for v in values if v and before.get(v)]
    inner_report: dict = {}
    changed = 0
    if inner_values and iterations_inner > 0 and sigma_inner_mm > 0:
        sigma_vox_inner = float(sigma_inner_mm) / spacing
        for _ in range(int(iterations_inner)):
            fields: dict[int, np.ndarray] = {}
            present: list[int] = []
            for value in inner_values:
                m = current == value
                if not m.any():
                    continue
                fields[value] = ndimage.gaussian_filter(
                    _sdf(m).astype(np.float32), sigma=sigma_vox_inner)
                present.append(value)
            if len(fields) < 2:
                break

            bias: dict[int, float] = {v: 0.0 for v in present}
            if preserve_volume:
                for _sweep in range(3):
                    left_max = np.full(current.shape, -np.inf, dtype=np.float32)
                    for vi, value in enumerate(present):
                        rivals = left_max.copy()
                        for u in present[vi + 1:]:
                            np.maximum(rivals, fields[u] + bias[u], out=rivals)
                        want = before.get(value, 0)
                        if not want:
                            continue
                        lo, hi = -2.0, 2.0
                        for _step in range(18):
                            mid = 0.5 * (lo + hi)
                            if _wins(fields[value] + mid, rivals, fixed) < want:
                                lo = mid
                            else:
                                hi = mid
                        bias[value] = 0.5 * (lo + hi)
                        np.maximum(left_max, fields[value] + bias[value], out=left_max)
            nxt = _assign(fields, bias, restrict=fixed)
            changed = int((nxt != current).sum())
            current = nxt
            if changed == 0:
                break
        inner_report = {
            "sigma_mm": round(float(sigma_inner_mm), 3),
            "iterations": int(iterations_inner),
            "labels_competing": len(present),
            "voxels_changed": changed,
        }
        logger.info("smooth_labels_sdf stage 2 (inner): sigma %.2fmm x%d, %d label(s), "
                    "%d voxel(s) reassigned", sigma_inner_mm, iterations_inner,
                    len(present), changed)
    report_stages["inner"] = inner_report

    after = {v: int((current == v).sum()) for v in values}
    drift = {LABEL_REGIONS.get(v, str(v)): round(100.0 * (after[v] / before[v] - 1.0), 2)
             for v in values if before.get(v)}
    report = {
        **report_stages,
        "preserve_volume": bool(preserve_volume),
        "voxels_changed": int((current != labels).sum()),
        "voxels_changed_pct": round(100.0 * float((current != labels).mean()), 3),
        "volume_change_pct": drift,
    }
    logger.info("smooth_labels_sdf: %d voxel(s) reassigned (%.2f%%), volume change %s",
                report["voxels_changed"], report["voxels_changed_pct"], drift)
    return current, report


def _fill_enclosed_cavities(mask: np.ndarray) -> np.ndarray:
    """Fill voids fully enclosed by `mask`, without dilating it."""
    padded = np.pad(~mask, 1, constant_values=True)
    outside = np.zeros_like(padded)
    outside[0, 0, 0] = True
    outside = ndimage.binary_propagation(outside, mask=padded)
    return ~outside[1:-1, 1:-1, 1:-1]


def _drop_small_components(mask: np.ndarray, min_voxels: int = 5) -> np.ndarray:
    """Drop connected components below `min_voxels`, with no dilation."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_voxels
    return keep[labeled]


def _ordered(values: list[int]) -> list[int]:
    """Sort label values into `REPAIR_PRIORITY` order."""
    return [v for v in REPAIR_PRIORITY if v in values]


def _absorb_islands(labels: np.ndarray, min_voxels: int = 8) -> tuple[np.ndarray, dict]:
    """Merge each material's tiny disconnected specks into what surrounds them."""
    out = labels.copy()
    islands = np.zeros(out.shape, dtype=bool)
    absorbed: dict[str, int] = {}
    for value in np.unique(out):
        if value == 0:
            continue
        component, n = ndimage.label(out == value)
        if n <= 1:
            continue
        sizes = np.bincount(component.ravel())
        small = np.nonzero(sizes < min_voxels)[0]
        small = small[small > 0]
        if not len(small):
            continue
        sel = np.isin(component, small)
        islands |= sel
        absorbed[LABEL_REGIONS.get(int(value), str(value))] = int(sel.sum())
    if not islands.any():
        return out, absorbed

    kept = out.copy()
    kept[islands] = 0
    _dist, idx = ndimage.distance_transform_edt(kept == 0, return_indices=True)
    out[islands] = kept[idx[0][islands], idx[1][islands], idx[2][islands]]
    return out, absorbed


def merge_split_components(
    labels: np.ndarray, merge_below_fraction: float = 0.3, min_voxels: int = 1,
    regions: Optional[tuple[str, ...]] = None,
) -> tuple[np.ndarray, dict]:
    """Give each material one continuous body by re-homing its stray fragments."""
    out = np.asarray(labels).copy()
    eligible = {REGION_LABELS[r] for r in (regions or (BRAIN_TISSUE + ("csf",)))
                if r in REGION_LABELS}
    report: dict = {}
    for value in np.unique(out):
        value = int(value)
        if value == 0 or value not in eligible:
            continue
        component, n = ndimage.label(out == value)
        if n <= 1:
            continue
        sizes = np.bincount(component.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        moved = 0
        boxes = ndimage.find_objects(component)
        for cid in range(1, n + 1):
            if cid == largest:
                continue
            if sizes[cid] > min_voxels and sizes[cid] >= merge_below_fraction * sizes[largest]:
                continue
            # the fragment's bounding box grown by one voxel holds the fragment and its whole
            # 6-neighbour ring: the same ring as on the full volume, without a full-volume pass
            box = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, dim))
                        for s, dim in zip(boxes[cid - 1], out.shape))
            piece = component[box] == cid
            ring = ndimage.binary_dilation(piece) & ~piece
            local = out[box]
            neighbours = local[ring]
            neighbours = neighbours[(neighbours != 0) & (neighbours != value)]
            if not neighbours.size:
                continue  # floating in background
            counts = np.bincount(neighbours)
            local[piece] = int(np.argmax(counts))          # writes through to `out`
            moved += int(sizes[cid])
        after = int(ndimage.label(out == value)[1])
        if moved or after != n:
            region = LABEL_REGIONS.get(value, str(value))
            report[region] = {"components_before": n, "components_after": after,
                              "voxels_rehomed": moved}
            logger.info("merge_split_components: %s %d -> %d components, "
                        "%d voxels moved to the material around them",
                        region, n, after, moved)
    return out, report


def scale_regions(
    seg: np.ndarray, affine: np.ndarray, scale: dict[str, float]
) -> tuple[np.ndarray, dict]:
    """Resize named regions about their own centroids, in place in the volume."""
    out = np.asarray(seg).copy()
    voxel_cm3 = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
    report: dict = {}
    active = {r: float(f) for r, f in (scale or {}).items() if abs(float(f) - 1.0) > 1e-6}
    if not active:
        return out, report

    for region, factor in active.items():
        value = REGION_LABELS.get(region)
        if value is None:
            raise ValueError(f"unknown region {region!r}; choose from {sorted(REGION_LABELS)}")
        mask = out == value
        if not mask.any():
            logger.warning("scale_regions: %s is not present, skipping", region)
            continue
        if factor <= 0:
            raise ValueError(f"scale factor for {region!r} must be positive, got {factor}")

        centre = np.array(ndimage.center_of_mass(mask))
        matrix = np.eye(3) / factor
        offset = centre - matrix @ centre
        grown = ndimage.affine_transform(
            mask.astype(np.uint8), matrix, offset=offset, order=0,
            output_shape=mask.shape, mode="constant", cval=0) > 0

        before = int(mask.sum())
        out[mask] = 0  # vacate the old footprint
        out[grown] = value  # and claim the new one
        report[region] = {
            "scale": factor,
            "before_cm3": round(before * voxel_cm3, 1),
            "after_cm3": round(int(grown.sum()) * voxel_cm3, 1),
        }
        logger.info("scale_regions: %s x%.2f, %.1f -> %.1f cm^3", region, factor,
                    report[region]["before_cm3"], report[region]["after_cm3"])

    was_tissue = np.asarray(seg) != 0
    holes = was_tissue & (out == 0)
    if holes.any():
        kept = out.copy()
        _dist, idx = ndimage.distance_transform_edt(kept == 0, return_indices=True)
        out[holes] = kept[idx[0][holes], idx[1][holes], idx[2][holes]]
        report["filled_voxels"] = int(holes.sum())
        logger.info("scale_regions: filled %d vacated voxels from the nearest material",
                    int(holes.sum()))
    return out, report


def build_head_labels(
    seg: np.ndarray,
    affine: np.ndarray,
    meninges_gap_mm: float = 2.5,
    skull_thickness_mm: float = 5.0,
    scalp_thickness_mm: float = 5.0,
    layers: tuple[str, ...] = ("csf", "skull", "scalp"),
    dura_mask: Optional[np.ndarray] = None,
    skull_mask: Optional[np.ndarray] = None,
    scalp_mask: Optional[np.ndarray] = None,
    vault_bridge_mm: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Build the single, whole-head, fold-preserving label volume."""
    from engine.mesh.hex_topology import clean_topology

    seg = np.asarray(seg)
    out = np.zeros(seg.shape, dtype=np.int64)
    report: dict = {"sources": {}, "voxels": {}, "volume_cm3": {}, "repair": {}}

    for region in BRAIN_TISSUE:
        value = REGION_LABELS[region]
        mask = _drop_small_components(seg == value)
        out[mask] = value
    brain = out != 0
    if not brain.any():
        raise ValueError("segmentation contains none of the five brain tissue labels")
    cavity = _fill_enclosed_cavities(brain) & ~brain
    interior = REGION_LABELS["csf"] if "csf" in layers else REGION_LABELS["gray_matter"]
    out[cavity] = interior
    if cavity.any():
        logger.info("build_head_labels: %d enclosed voxels inside the brain filled with %s",
                    int(cavity.sum()), LABEL_REGIONS[interior])

    out, rep = clean_topology(out, priority=_ordered([interior] + [REGION_LABELS[r]
                                                                  for r in BRAIN_TISSUE]) + [0])
    report["repair"]["brain"] = rep
    out, report["merged_components_brain"] = merge_split_components(
        out, regions=BRAIN_TISSUE)
    brain = np.isin(out, [REGION_LABELS[r] for r in BRAIN_TISSUE])

    voxel_cm3 = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
    placed = [REGION_LABELS[r] for r in BRAIN_TISSUE]

    chain = []
    for name in ("csf", "skull", "scalp"):
        if name not in layers:
            break
        chain.append(name)
    if "csf" not in chain:
        logger.info("build_head_labels: brain only, no CSF or outer layers requested")
        out, absorbed = _absorb_islands(out)
        report["absorbed_islands"] = absorbed
        out, report["merged_components"] = merge_split_components(out)
        return _summarize(out, affine, report)

    if dura_mask is None:
        dura = _offset_mask(brain, meninges_gap_mm, meninges_gap_mm + 5.0, affine)
        report["sources"]["dura"] = "offset_envelope"
    else:
        dura = dura_mask | brain  # a real surface must still contain the brain
        report["sources"]["dura"] = "real_surface"
    csf = (dura & ~brain) | ((seg == REGION_LABELS["csf"]) & ~brain)
    out[csf & (out == 0)] = REGION_LABELS["csf"]
    placed.append(REGION_LABELS["csf"])
    out, rep = clean_topology(out, priority=_ordered(placed) + [0])
    report["repair"]["csf"] = rep

    for region, thickness, real_mask in (
        ("skull", skull_thickness_mm, skull_mask),
        ("scalp", scalp_thickness_mm, scalp_mask),
    ):
        if region in chain:
            inner = out != 0
            if real_mask is None and region == "skull" and vault_bridge_mm > 0:
                bridge = max(1, int(round(vault_bridge_mm / _voxel_size(affine))))
                vault = ndimage.binary_closing(
                    inner, structure=np.ones((3, 3, 3)), iterations=bridge)
                vault = _fill_enclosed_cavities(vault | inner)
                spanned = int((vault & ~inner).sum())
                outer = _offset_mask(vault, thickness, thickness + vault_bridge_mm, affine)
                out[vault & (out == 0)] = REGION_LABELS["csf"]
                report["sources"][region] = f"offset_envelope, vault bridged {vault_bridge_mm:.2f}mm"
                report["vault_bridge"] = {
                    "bridge_mm": float(vault_bridge_mm),
                    "voxels_spanned": spanned,
                    "spanned_cm3": round(spanned * voxel_cm3, 3),
                }
                logger.info("build_head_labels: vault bridged at %.2fmm, %d voxel(s) "
                            "(%.2fcm^3) of sulcus spanned and filled with CSF rather than "
                            "traced by bone", vault_bridge_mm, spanned, spanned * voxel_cm3)
            elif real_mask is None:
                outer = _offset_mask(inner, thickness, thickness + 3.0, affine)
                report["sources"][region] = "offset_envelope"
            else:
                outer = real_mask | inner
                report["sources"][region] = "real_surface"
            outer = _fill_enclosed_cavities(outer | ndimage.binary_dilation(inner))
            uncovered = int((outer & (out == 0) & ~(real_mask if real_mask is not None
                                                   else outer)).sum()) if real_mask is not None else 0
            out[outer & (out == 0)] = REGION_LABELS[region]
            if uncovered:
                logger.info("build_head_labels: %s closed over %d voxels the measured "
                            "mask did not reach, so the layer has no gaps in it",
                            region, uncovered)
            placed.append(REGION_LABELS[region])
            out, rep = clean_topology(out, priority=_ordered(placed) + [0])
            report["repair"][region] = rep

    out, absorbed = _absorb_islands(out)
    report["absorbed_islands"] = absorbed
    out, report["merged_components"] = merge_split_components(out, regions=BRAIN_TISSUE)
    out, report["csf_separation"] = _separate_brain_from_bone(out)
    if "scalp" in chain:
        out, report["scalp_closure"] = _close_outer_layer(out, "scalp")
    return _summarize(out, affine, report)


def _close_outer_layer(out: np.ndarray, region: str) -> tuple[np.ndarray, dict]:
    """Guarantee the outermost layer is the only thing the air ever touches."""
    value = REGION_LABELS[region]
    out = np.asarray(out).copy()
    total = 0
    for _round in range(8):
        body = (out != 0) & (out != value)
        exposed = ndimage.binary_dilation(body) & (out == 0)
        n = int(exposed.sum())
        if not n:
            break
        out[exposed] = value
        total += n
    holes_left = int((ndimage.binary_dilation((out != 0) & (out != value)) & (out == 0)).sum())
    if total:
        logger.info("build_head_labels: closed %d voxels of hole in the %s, so nothing "
                    "inside the head faces air except through it", total, region)
    if holes_left:
        logger.warning("build_head_labels: %d voxels of %s are still exposed to air after "
                       "closure", holes_left, region)
    return out, {"voxels_added": total, "still_exposed": holes_left}


def _separate_brain_from_bone(out: np.ndarray) -> tuple[np.ndarray, dict]:
    """Guarantee at least one CSF voxel between the brain and anything outside it."""
    brain = np.isin(out, [REGION_LABELS[r] for r in BRAIN_TISSUE])
    outside = (out != 0) & ~brain & (out != REGION_LABELS["csf"])
    touching = outside & ndimage.binary_dilation(brain)
    n = int(touching.sum())
    if n:
        out = out.copy()
        out[touching] = REGION_LABELS["csf"]
        logger.info("build_head_labels: %d voxels of bone/scalp touched the brain "
                    "with no CSF between them; reassigned to CSF so the pia and "
                    "the dura stay distinct surfaces", n)
    return out, {"voxels_reassigned_to_csf": n}


def _summarize(out: np.ndarray, affine: np.ndarray, report: dict) -> tuple[np.ndarray, dict]:
    """Fill in per-region voxel counts, volumes and component counts."""
    voxel_cm3 = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
    for region, value in REGION_LABELS.items():
        n = int((out == value).sum())
        if n:
            report["voxels"][region] = n
            report["volume_cm3"][region] = round(n * voxel_cm3, 1)
            report.setdefault("components", {})[region] = int(ndimage.label(out == value)[1])
    report["total_voxels"] = int((out != 0).sum())
    logger.info("build_head_labels: %s", report["volume_cm3"])
    return out, report


def cortical_surface_area(labels: np.ndarray, affine: np.ndarray) -> float:
    """Area of the gray-matter/CSF interface, in cm^2."""
    gm = labels == REGION_LABELS["gray_matter"]
    other = np.isin(labels, (0, REGION_LABELS["csf"]))
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    total = 0.0
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(0, -1)
        b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        n = int((gm[a] & other[b]).sum()) + int((gm[b] & other[a]).sum())
        face = float(np.prod(spacing) / spacing[axis])
        total += n * face
    return total / 100.0
