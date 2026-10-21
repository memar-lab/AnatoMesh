"""Label volume in, head mesh out, by the voxel method."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from engine.mesh.hex_mesh import _unique_rows

from engine.anatomy.anatomy_labels import (
    BRAIN_TISSUE,
    LABEL_REGIONS,
    REGION_LABELS,
    build_head_labels,
    scale_regions,
)

logger = logging.getLogger(__name__)

BRAIN_LABELS = {REGION_LABELS[r] for r in BRAIN_TISSUE}
CSF = REGION_LABELS["csf"]
SKULL = REGION_LABELS["skull"]
SCALP = REGION_LABELS["scalp"]


_DURA_OFFSET_MM = 0.6

_PIA_DURA_CLEARANCE_MM = 1.0


def _pia_clearance_mm(grid_mm: float, block_pia: int) -> float:
    """How far off the brain the pial sheet is built, in mm."""
    return max(float(block_pia) * float(grid_mm), float(grid_mm))

from engine.export.validate import _DURA_INNER_TABLE_BAND_MM, dura_inner_table_band_mm  # noqa: E402
from engine.mesh import fast_kernels
from engine.pipeline import timing

_MEMBRANE_SETTLE_ROUNDS = 3

HIERARCHICAL_SMOOTH_REPORTS: list[dict] = []

SDF_SIGMA_OUTER_VOX = 1.2
SDF_SIGMA_INNER_VOX = 0.6


def _ventricular_csf(labels: np.ndarray,
                     seed: Optional[np.ndarray] = None) -> np.ndarray:
    """CSF that is not subarachnoid (ventricles and enclosed pockets)."""
    csf = labels == CSF
    if not csf.any():
        return np.zeros(labels.shape, dtype=bool)
    from scipy import ndimage

    from_seed = (csf & np.asarray(seed, dtype=bool)) if seed is not None \
        else np.zeros(labels.shape, dtype=bool)
    brain = np.isin(labels, sorted(BRAIN_LABELS))
    enclosed = csf & ndimage.binary_fill_holes(brain)

    components, n = ndimage.label(csf)
    unreached = np.zeros(labels.shape, dtype=bool)
    if n:
        outside = ~np.isin(labels, sorted(BRAIN_LABELS | {CSF}))
        touching = np.unique(components[ndimage.binary_dilation(outside) & csf])
        unreached = csf & ~np.isin(components, touching[touching > 0])

    ventricular = from_seed | enclosed | unreached
    if ventricular.any():
        logger.info("pia: excluding %d ventricular CSF voxels (%d from the "
                    "segmentation's own ventricle label, %d inside the brain's hull, "
                    "%d in components that never reach the dura)",
                    int(ventricular.sum()), int(from_seed.sum()),
                    int(enclosed.sum()), int(unreached.sum()))
    return ventricular


def _pad_cropped_volume(
    labels: np.ndarray, affine: np.ndarray, margin_mm: float,
    mantles: Optional[dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[dict[str, np.ndarray]]]:
    """Pad with background where tissue sits too close to the volume edge."""
    occ = labels != 0
    if not occ.any():
        return labels, affine, mantles
    vox = float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
    pad_vox = max(1, int(round(margin_mm / vox)))
    idx = np.argwhere(occ)
    small_vol = [0, 0, 0]
    large_vol = [0, 0, 0]
    for a in range(3):
        lo, hi = int(idx[:, a].min()), int(idx[:, a].max())
        margin_lo = lo
        margin_hi = int(occ.shape[a] - 1 - hi)
        small_vol[a] = max(0, pad_vox - margin_lo)
        large_vol[a] = max(0, pad_vox - margin_hi)
    if not any(small_vol) and not any(large_vol):
        return labels, affine, mantles
    padded_volume = np.pad(labels, [(small_vol[0], large_vol[0]), (small_vol[1], large_vol[1]),
                            (small_vol[2], large_vol[2])],
                   mode="constant", constant_values=0)
    affine_padded = affine.copy()
    affine_padded[:3, 3] = affine[:3, 3] - affine[:3, :3] @ np.array(small_vol)
    mantles_padded = None
    if mantles:
        mantles_padded = {name: np.pad(m, [(small_vol[0], large_vol[0]), (small_vol[1], large_vol[1]),
                                         (small_vol[2], large_vol[2])], mode="constant",
                                     constant_values=False)
                        for name, m in mantles.items()}
    logger.info("padded label volume to %.1fmm background on the faces whose "
                "margin was under the %d-voxel envelope clearance (low %s, high %s); "
                "affine shifted to keep world space",
                margin_mm, pad_vox, small_vol, large_vol)
    return padded_volume, affine_padded, mantles_padded


def build_head_mesh(
    seg: np.ndarray,
    affine: np.ndarray,
    config,
    freesurfer_subject_dir: Optional[Path] = None,
    dura_mask: Optional[np.ndarray] = None,
    skull_mask: Optional[np.ndarray] = None,
    scalp_mask: Optional[np.ndarray] = None,
    progress=None,
) -> dict:
    """Mesh the whole head by the voxel method, with its membranes."""
    timings: dict[str, float] = {}
    phase: list = [None, time.perf_counter()]

    def step(fraction: float, message: str) -> None:
        now = time.perf_counter()
        if phase[0] is not None:
            timings[phase[0]] = round(timings.get(phase[0], 0.0) + now - phase[1], 2)
        phase[0], phase[1] = message, now
        timing.mark(message)
        logger.info("[%3d%%] %s", int(fraction * 100), message)
        if progress is not None:
            progress(fraction, message)

    wanted = set(getattr(config, "auto_boundaries", None) or ())
    element_mm = getattr(config, "element_size_mm", None)
    if element_mm:
        original_affine = affine
        from engine.pipeline.label_volume import resample_labels

        source_mm = float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
        seg = resample_labels(np.asarray(seg), source_mm, float(element_mm))
        affine = affine.copy()
        affine[:3, :3] = affine[:3, :3] * (float(element_mm) / source_mm)
        for name, mask in (("dura", dura_mask), ("skull", skull_mask), ("scalp", scalp_mask)):
            if mask is None:
                continue
            resampled = resample_labels(mask.astype(np.uint8), source_mm,
                                        float(element_mm)).astype(bool)
            if name == "dura":
                dura_mask = resampled
            elif name == "skull":
                skull_mask = resampled
            else:
                scalp_mask = resampled

    gap_mm = float(getattr(config, "meninges_gap_mm", 2.5))
    if wanted & {"skull", "scalp"}:
        layers_mm = float(getattr(config, "shell_thickness_skull", 5.0))
        if "scalp" in wanted:
            layers_mm += float(getattr(config, "shell_thickness_scalp", 5.0))
        voxel_mm = float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
        # a measured skull may reach past the nominal thickness (thin_skull_from_template caps at 0.45 mm)
        margin_mm = gap_mm + max(layers_mm, 0.6) + 4 * voxel_mm
        if voxel_mm >= 0.5:  # human and large-animal grids keep the old padding
            margin_mm = max(6.5, gap_mm + float(getattr(config, "shell_thickness_skull", 5.0)) + 4.0)
    else:
        # only the CSF layer is grown off the brain; the solved gap can reach GAP_MAX_MM
        from engine.anatomy.anatomy_labels import GAP_MAX_MM

        if getattr(config, "subarachnoid_from_physiology", False):
            gap_mm = max(gap_mm, GAP_MAX_MM)
        margin_mm = gap_mm + 4 * float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
    mantles = {}
    if dura_mask is not None:
        mantles["dura"] = dura_mask
    if skull_mask is not None:
        mantles["skull"] = skull_mask
    if scalp_mask is not None:
        mantles["scalp"] = scalp_mask
    seg, affine, mantles_padded = _pad_cropped_volume(seg, affine, margin_mm, mantles or None)
    if mantles_padded:
        dura_mask = mantles_padded.get("dura", dura_mask)
        skull_mask = mantles_padded.get("skull", skull_mask)
        scalp_mask = mantles_padded.get("scalp", scalp_mask)

    region_scale = getattr(config, "region_scale", None)
    if region_scale:
        step(0.03, "scaling regions")
        seg, scale_report = scale_regions(seg, affine, region_scale)

    step(0.05, "assembling head labels")
    ventricle_seed = np.asarray(seg) == REGION_LABELS["csf"] \
        if getattr(config, "csf_ventricles_from_segmentation", True) \
        else np.zeros(np.shape(seg), dtype=bool)
    gap_mm = float(config.meninges_gap_mm)
    if getattr(config, "subarachnoid_from_physiology", False):
        from engine.anatomy.anatomy_labels import subarachnoid_gap_mm

        gap_mm, gap_report = subarachnoid_gap_mm(
            np.isin(seg, sorted(BRAIN_LABELS)), affine,
            sulcal_csf=(np.asarray(seg) == REGION_LABELS["csf"]))
    else:
        gap_report = {}
    labels, report = build_head_labels(
        seg, affine,
        meninges_gap_mm=gap_mm,
        vault_bridge_mm=float(getattr(config, "vault_bridge_mm", 0.0) or 0.0),
        skull_thickness_mm=config.shell_thickness_skull,
        scalp_thickness_mm=config.shell_thickness_scalp,
        layers=tuple(n for n in ("csf", "skull", "scalp") if n in wanted),
        dura_mask=dura_mask, skull_mask=skull_mask, scalp_mask=scalp_mask,
    )
    if gap_report:
        report["subarachnoid"] = gap_report
    if region_scale:
        report["region_scale"] = scale_report

    if (getattr(config, "label_smoothing_sigma_vox", SDF_SIGMA_OUTER_VOX) or 0) > 0 or \
            (getattr(config, "label_smoothing_sigma_inner_vox", SDF_SIGMA_INNER_VOX) or 0) > 0:
        step(0.06, "SDF smoothing of the head labels")
        labels, report["label_smoothing"] = smooth_head_labels(labels, config)

    membrane_sheets: dict = {}
    if {"falx", "tentorium"} & wanted:
        from engine.mesh import membranes

        step(0.08, "building falx and tentorium")
        if getattr(config, "membrane_rules", "") == "anatomical":
            # thin skull plates inside the cavity (dura / sinus walls the skull segmentation took
            # in) become CSF, so the falx and tentorium run through them in one piece
            from engine.mesh import membrane_rules as _mr
            labels, report["thin_skull_to_csf_voxels"] = _mr.thin_skull_to_csf(
                labels, affine, skull_label=SKULL, csf_label=CSF)
            logger.info("membranes: %d thin skull voxels inside the cavity relabelled CSF",
                        report["thin_skull_to_csf_voxels"])
        # the conventional / new method choices (membranes.MEMBRANE_OPTIONS), kept in force
        # until the membranes are finished; restored on the way out of build_head_mesh
        _membrane_choices = membranes.membrane_settings(**{
            k: getattr(config, k) for k in (*membranes.MEMBRANE_OPTIONS, *membranes.MEMBRANE_NUMERIC)
            if hasattr(config, k)})
        _membrane_choices.__enter__()
        import os

        if os.environ.get("ANATOMESH_DUMP_MEMBRANE_INPUT"):  # offline falx/tentorium experiments
            np.savez_compressed(os.environ["ANATOMESH_DUMP_MEMBRANE_INPUT"], labels=labels,
                                affine=affine)
            raise SystemExit("membrane input dumped")
        try:
            membrane_sheets = membranes.build_membranes(
                labels, affine,
                target_mm=getattr(config, "membrane_element_size_mm", 3.0),
                falx_cc_gap_mm=getattr(config, "falx_cc_gap_mm", 10.0),
                profile=membranes.membrane_profile(
                    "human", getattr(config, "membrane_profile", "auto")),
                lateral_open_frac=float(getattr(config, "tentorium_lateral_open_frac", 0.35)),
                scale=getattr(config, "membrane_scale", 1.0))
        except Exception as exc:  # noqa: BLE001 - an approximation must not sink the run
            logger.warning("membrane construction failed (%s); falx and tentorium omitted", exc)
        membrane_sheets = {k: v for k, v in membrane_sheets.items() if k in wanted}
        if membrane_sheets:
            slot_mm = max(float(getattr(config, "membrane_slot_mm", 2.0)),
                          2.0 * float(np.mean(np.linalg.norm(affine[:3, :3], axis=0))))
            labels, carve_report = membranes.carve_slots(
                labels, affine, membrane_sheets, csf_label=CSF, thickness_mm=slot_mm)
            report["membrane_slots"] = carve_report
            from engine.mesh.hex_topology import clean_topology
            from engine.anatomy.anatomy_labels import REPAIR_PRIORITY
            labels, _rep = clean_topology(labels, priority=REPAIR_PRIORITY)
            report["membrane_slot_topology_repair"] = _rep
            allowed = np.isin(labels, (0, CSF))
            for name, (verts, quads) in list(membrane_sheets.items()):
                if name in membranes.FINISHED_SHEETS:
                    # meshed in its own plane (falx_on_midline) and sitting in its slot:
                    # the push/relax/repair passes below would only bend it
                    report.setdefault("membrane_repair_final", {})[name] = {"skipped": "planar mesh"}
                    if name != "falx" and membranes._RIM_RETRACT:
                        # the tentorium meets the occipital bone across its face, where an
                        # in-sheet retract cannot help: ease it off the bone in 3D with smooth
                        # dents first, so the trim below does not notch its back edge
                        verts, report.setdefault("membrane_bone_ease", {})[name] = \
                            membranes.retract_sheet_from(
                                verts, quads, labels == SKULL, affine,
                                tolerance_mm=membranes.RIM_BONE_TOLERANCE_MM
                                * membranes.LAST_BUILD_SCALE, name=f"{name} (bone)")
                    verts, quads, report.setdefault("membrane_rim_bone", {})[name] = \
                        membranes.trim_rim_in_bone(verts, quads, labels == SKULL, affine, name=name)
                    membrane_sheets[name] = (verts, quads)
                    if name == "tentorium" and getattr(config, "hex_quality_pass", False):
                        from engine.mesh import sheet_quality   # in its own surface: shape unchanged
                        verts, report.setdefault("membrane_in_surface", {})[name] = \
                            sheet_quality.improve_sheet_in_surface(verts, quads, name=name)
                        membrane_sheets[name] = (verts, quads)
                    continue
                # the falx is one flat sheet on the midsagittal plane: every pass below may
                # move its vertices within that plane, never out of it
                plane = membranes.sheet_plane(verts) if name == "falx" else None

                def keep_flat(v, plane=plane):
                    return v if plane is None else membranes.flatten_to_plane(v, plane)

                verts, quads = membranes.push_sheet_out_of(verts, quads, allowed, affine)
                verts, clearance = membranes.center_in_gap(verts, quads, allowed, affine)
                verts = keep_flat(verts)
                report.setdefault("membrane_clearance", {})[name] = clearance
                verts, relax = membranes.relax_sheet_guarded(
                    verts, quads, iterations=20, max_displacement_mm=0.3)
                report.setdefault("membrane_relax", {})[name] = relax
                verts, untangle = membranes.untangle_sheet(verts, quads)
                verts = keep_flat(verts)
                report.setdefault("membrane_untangle", {})[name] = untangle

                from engine.export.validate import _quad_scaled_jacobian

                from engine.mesh import shell_repair

                verts, rim_report = shell_repair.smooth_rim(
                    verts, quads, allowed=allowed, affine=affine, name=name)
                verts = keep_flat(verts)
                report.setdefault("membrane_rim", {})[name] = rim_report

                verts, final_repair = shell_repair.improve_sheet(
                    verts, quads, name=f"{name} (post-push)",
                    allowed=allowed, affine=affine)
                verts = keep_flat(verts)
                report.setdefault("membrane_repair_final", {})[name] = final_repair

                verts, quads, drop_report = membranes.drop_inverted_without_holing(
                    verts, quads, name=name)
                report.setdefault("membrane_inverted_dropped", {})[name] = drop_report
                membrane_sheets[name] = (verts, quads)

    from engine.mesh import shell_repair as _shell_repair

    report["membrane_pass_seconds"] = _shell_repair.pass_timing_report()
    logger.info("falx/tentorium pass timings (s): %s", report["membrane_pass_seconds"])

    grid_mm = float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
    step(0.2, f"meshing the head by the voxel method at {grid_mm:.3g}mm")
    labels, report["sealed_voxels"] = seal_layers(labels)
    nodes, hexes, hex_labels, mesh_report = voxel_volume_mesh(labels, affine, config)
    HIERARCHICAL_SMOOTH_REPORTS.append(mesh_report)
    report["voxel_mesh"] = mesh_report
    solid_blocks: dict[str, np.ndarray] = {}
    for value in np.unique(hex_labels):
        region = LABEL_REGIONS.get(int(value))
        if region is not None:
            solid_blocks[region] = hexes[hex_labels == value]
    conformity = {"conforming": True, "blocks": {}, "n_hanging_nodes_on_lattice": 0,
                  "n_boundary_edges_unpaired": 0}
    base_regions = set(solid_blocks)
    tied_pairs: list[tuple[str, str]] = []

    step(0.7, "extracting membrane sheets")
    shell_quads: dict[str, np.ndarray] = {}
    fluid = labels == CSF
    brain = np.isin(labels, sorted(BRAIN_LABELS))
    outer_solid = np.isin(labels, (SKULL, SCALP))
    from scipy import ndimage as _nd

    if fluid.any():
        slot = _nd.binary_dilation(fluid)
        strict = _nd.binary_erosion(brain) | _nd.binary_erosion(outer_solid)
    else:
        slot = strict = np.zeros(labels.shape, dtype=bool)

    from engine.mesh import membranes as _membranes

    meninges_mm = float(getattr(config, "meninges_element_size_mm", 3.0) or 3.0)
    block = max(int(round(meninges_mm / max(grid_mm, 1e-6))), 1)

    ventricular = _ventricular_csf(labels, ventricle_seed)
    conforming = str(getattr(config, "meninges_mode", "conforming")) == "conforming"
    if conforming and wanted & {"pia", "dura"} and fluid.any():
        sheets, report["meninges_sheets"] = conforming_meninges(hexes, hex_labels, ventricular)
        for name, quads in sheets.items():
            if name in wanted and len(quads):
                shell_quads[name] = quads
        base_regions |= set(shell_quads)
    pia_mm = getattr(config, "pia_element_size_mm", None) or grid_mm
    block_pia = max(int(round(float(pia_mm) / max(grid_mm, 1e-6))), 1)
    pia_clearance_mm = _pia_clearance_mm(grid_mm, block_pia)
    clearance_vox = max(1, int(round(pia_clearance_mm / max(grid_mm, 1e-6))))
    logger.info("pia clearance %.3fmm (%d voxel(s) of a %.3fmm grid, block %d)",
                pia_clearance_mm, clearance_vox, grid_mm, block_pia)
    seed = brain | ventricular
    d_tissue = fast_kernels.distance_transform_edt(~seed, sampling=np.linalg.norm(affine[:3, :3], axis=0))
    reach = float(clearance_vox) * grid_mm
    medial = _nd.maximum_filter(d_tissue, size=2 * max(clearance_vox, block_pia) + 1)
    pia_mask = seed | (fluid & (d_tissue <= reach) & (d_tissue < medial))
    pia_clear = _nd.binary_erosion(pia_mask)
    intracranial = _nd.binary_fill_holes(brain | fluid)
    dura_mask = intracranial
    field_placement = bool(getattr(config, "membrane_field_placement", False))
    band_mm = (dura_inner_table_band_mm(intracranial, labels == SKULL,
                                        np.linalg.norm(affine[:3, :3], axis=0))
               if field_placement else _DURA_INNER_TABLE_BAND_MM)
    spacing_mm = np.linalg.norm(affine[:3, :3], axis=0)
    out_from_envelope = out_from_brain = None
    if field_placement:
        out_from_envelope = (fast_kernels.distance_transform_edt(~intracranial, sampling=spacing_mm)
                             - fast_kernels.distance_transform_edt(intracranial, sampling=spacing_mm))
        out_from_brain = (fast_kernels.distance_transform_edt(~brain, sampling=spacing_mm)
                          - fast_kernels.distance_transform_edt(brain, sampling=spacing_mm))
        dura_band = (out_from_envelope <= band_mm) & ~intracranial
    else:
        band_vox = max(1, int(round(band_mm / max(grid_mm, 1e-6))))
        dura_band = _nd.binary_dilation(intracranial, iterations=band_vox) & ~intracranial
    pia_allowed = (~pia_clear) & intracranial
    if field_placement:
        pia_allowed &= ~seed

    for name, source, away, allowed, offset, sheet_block in (
            ("pia", pia_mask, pia_mask, pia_allowed, 0.0, block_pia),
            ("dura", dura_mask, dura_mask,
             dura_band, 0.0, block)):
        if name not in wanted or not slot.any() or conforming:
            continue
        verts, quads, _coarse = _membranes.mask_shell_quads(
            source, affine, block=sheet_block, grow=True)
        if not len(quads):
            continue
        verts, quads = _membranes.push_sheet_out_of(
            verts, quads, ~away, affine, max_push_mm=float(sheet_block) * grid_mm + 2.0)
        verts, quads, sheet_report = _membranes.free_standing_sheet(
            verts, quads, away, allowed, affine, offset, name=name,
            strict_forbidden=strict)
        if not len(quads):
            continue
        verts, quads = _membranes.push_sheet_out_of(
            verts, quads, allowed, affine, max_push_mm=float(sheet_block) * grid_mm)
        verts, final_relax = _membranes.relax_sheet_guarded(
            verts, quads, iterations=20, max_displacement_mm=0.3)
        sheet_report["final_relax"] = final_relax
        verts, final_untangle = _membranes.untangle_sheet(verts, quads)
        sheet_report["final_untangle"] = final_untangle
        from engine.export.validate import _quad_scaled_jacobian

        from engine.mesh import shell_repair

        verts, quads = _membranes.push_sheet_out_of(
            verts, quads, allowed, affine, max_push_mm=float(sheet_block) * grid_mm)

        verts, push_untangle = _membranes.untangle_sheet(verts, quads)
        sheet_report["untangle_after_push"] = push_untangle

        verts, final_repair = shell_repair.improve_sheet(
            verts, quads, name=f"{name} (post-push)", allowed=allowed, affine=affine)
        sheet_report["repair_final"] = final_repair
        for round_index in range(_MEMBRANE_SETTLE_ROUNDS if field_placement else 0):
            if round_index:
                verts, settle_untangle = _membranes.untangle_sheet(verts, quads)
                sheet_report[f"untangle_settle_{round_index}"] = settle_untangle
            field, upper = ((out_from_envelope, band_mm) if name == "dura"
                            else (out_from_brain, None))
            verts, band_report = _membranes.project_sheet_to_offset(
                verts, quads, field, affine, lo_mm=0.5 * grid_mm, hi_mm=upper,
                name=name, max_move_mm=2.0 * grid_mm)
            sheet_report[f"band_projection_{round_index}"] = band_report

        if getattr(config, "hex_quality_pass", False):
            from engine.mesh import sheet_quality

            verts, sheet_report["in_surface_repair"] = sheet_quality.improve_sheet_in_surface(
                verts, quads, name=name)
        verts, quads, drop_report = _membranes.drop_inverted_without_holing(
            verts, quads, name=name)
        sheet_report["inverted_after_push"] = drop_report
        sheet_report["block"] = sheet_block
        inv = np.linalg.inv(affine)
        vox = verts @ inv[:3, :3].T + inv[:3, 3]
        sheet_report["vertices_in_brain"] = int(_membranes._sample_mask(brain, vox).sum())
        sheet_report["vertices_in_csf"] = int(_membranes._sample_mask(fluid, vox).sum())
        membrane_sheets[name] = (verts, quads)
        report.setdefault("meninges_sheets", {})[name] = sheet_report

    report["meninges_pass_seconds"] = _shell_repair.pass_timing_report()
    logger.info("pia/dura pass timings (s): %s", report["meninges_pass_seconds"])

    step(0.8, "meshing complete")
    report["timings"] = dict(sorted(timings.items(), key=lambda kv: -kv[1]))
    if "_membrane_choices" in locals():   # restore the module defaults
        _membrane_choices.__exit__(None, None, None)
    return {
        "lattice_colour": LATTICE_COLOUR[0],
        "nodes": nodes,
        "hexes": hexes,
        "hex_labels": hex_labels,
        "solid_blocks": solid_blocks,
        "shell_quads": shell_quads,
        "head_labels": labels,
        "affine": affine,
        "conformity": conformity,
        "report": report,
        "welded_regions": base_regions,
        "conforming_sheets": set(shell_quads),
        "tied_pairs": tied_pairs,
        "membrane_sheets": membrane_sheets,
        "element_sizes": {region: grid_mm for region in solid_blocks},
    }


def smooth_head_labels(labels: np.ndarray, config) -> tuple[np.ndarray, dict]:
    """Two-stage SDF smoothing of a head label volume."""
    from engine.pipeline import sdf_smoothing, label_volume

    outer = getattr(config, "label_smoothing_sigma_vox", SDF_SIGMA_OUTER_VOX)
    inner = getattr(config, "label_smoothing_sigma_inner_vox", SDF_SIGMA_INNER_VOX)
    combine = bool(getattr(config, "combine_brain_for_smoothing", True))
    smoothed, report = sdf_smoothing.smooth_segmentation_sdf(
        label_volume.combine(labels) if combine else labels, (1.0, 1.0, 1.0),
        sigma_outer_mm=float(outer or 0.0),
        iterations_outer=int(getattr(config, "label_smoothing_iterations", 3)),
        sigma_inner_mm=float(inner or 0.0),
        iterations_inner=int(getattr(config, "label_smoothing_iterations_inner", 5)))
    report.update(sigma_outer_vox=outer, sigma_inner_vox=inner, combined=combine)
    return (label_volume.overlay(smoothed, labels) if combine else smoothed), report


def seal_layers(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy import ndimage

    out = np.asarray(labels).copy()
    present = set(np.unique(out).tolist())
    layers = [sorted(BRAIN_LABELS)] + [[v] for v in (CSF, SKULL, SCALP) if v in present]
    inner = set(layers[0])
    report = {}
    for layer in layers[1:]:
        exposed = (out == 0) & ndimage.binary_dilation(np.isin(out, sorted(inner)),
                                                       np.ones((3, 3, 3), dtype=bool))
        out[exposed] = layer[0]
        report[LABEL_REGIONS.get(layer[0], str(layer[0]))] = int(exposed.sum())
        inner |= set(layer)
    if any(report.values()):
        logger.info("sealed exposed layers with the next layer out: %s", report)
    return out, report


LATTICE_COLOUR: list = [None]   # colour of each node of the last voxel_volume_mesh


def voxel_volume_mesh(labels: np.ndarray, affine: np.ndarray, config
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """the voxel mesher on a head label volume, in world mm."""
    import os

    from engine.mesh import voxel_hex
    from engine.pipeline import label_volume

    if os.environ.get("ANATOMESH_DUMP_LABELS"):  # inputs for offline quality experiments
        np.savez_compressed(os.environ["ANATOMESH_DUMP_LABELS"], labels=labels, affine=affine)
    result = voxel_hex.generate_mesh(
        label_volume.combine(labels), int(getattr(config, "hierarchical_smoothing_iterations", 3)),
        part_labels=labels,
        lambda_value=float(getattr(config, "hierarchical_smoothing_lambda", voxel_hex.LAMBDA_VALUE)),
        targeted_adjustment=bool(getattr(config, "hierarchical_smoothing_repair", True)),
        keep_odd_passes=False, repair_every_element=True,
        jacobian_threshold=float(getattr(config, "repair_jacobian_threshold", voxel_hex.JACOBIAN_THRESHOLD)),
        target_min_jacobian=getattr(config, "smoothing_target_jacobian", None),
        max_iterations=int(getattr(config, "smoothing_max_iterations", 10)),
        progress=lambda message: logger.info("voxel mesh: %s", message))
    last = max(result["nodes_per_iteration"])
    lattice = result["nodes_per_iteration"][last]
    elements = result["elements"]
    used = np.unique(elements)
    new_id = np.full(len(lattice) + 1, -1, dtype=np.int64)
    new_id[used] = np.arange(len(used))
    hexes = new_id[elements]
    nodes = (lattice[used - 1] - 0.5) @ np.asarray(affine[:3, :3], dtype=float).T \
        + np.asarray(affine[:3, 3], dtype=float)
    if np.linalg.det(affine[:3, :3]) < 0:
        hexes = hexes[:, [0, 3, 2, 1, 4, 7, 6, 5]]
    report = {"method": "MeshGeneration_Sep2026_NoDuplicateNodes", "passes": last,
              "n_elements": int(len(hexes)), "n_nodes": int(len(nodes)),
              "iterations": {str(k): v for k, v in result["stats"].items()}}
    final = result["stats"][last]
    logger.info("voxel mesh: %d hexes, scaled Jacobian min %.4f mean %.4f",
                len(hexes), final["jacobian_min"], final["jacobian_mean"])
    part_ids = result["part_ids"].astype(int)
    n_rows, n_cols, n_depth = result["shape"]
    plane = (n_cols + 1) * (n_depth + 1)
    lattice_id = used - 1
    x, y, z = lattice_id // plane, (lattice_id // (n_depth + 1)) % (n_cols + 1), \
        lattice_id % (n_depth + 1)
    # lattice parity: no two nodes of a hex share a colour (mesh.hex_quality moves by colour)
    LATTICE_COLOUR[0] = ((x % 2) * 4 + (y % 2) * 2 + (z % 2)).astype(np.int8)
    LATTICE_IJK[0] = np.stack([x, y, z], axis=1).astype(np.int32)
    return nodes, hexes, part_ids, report


LATTICE_IJK: list = [None]      # lattice index of each node; a hex's lowest corner is its voxel
_HEX_FACES_OUT = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [3, 7, 6, 2],
                           [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5]])  # normals point out


def conforming_meninges(hexes: np.ndarray, hex_labels: np.ndarray,
                        ventricular: np.ndarray) -> tuple[dict, dict]:
    """Pia and dura as faces of the hex mesh, sharing its nodes.

    Pia: brain faces that do not face brain or ventricular CSF. Dura: faces of the
    intracranial hexes (brain and CSF) that face skull, scalp or the outside. This is how
    the KTH/ADAPT (Kleiven 2007; Li et al. 2021), GHBMC (Mao et al. 2013) and SIMon
    (Takhounts et al. 2008) models place them, and it needs no coupling to the solids.
    """
    vox = LATTICE_IJK[0][hexes].min(axis=1)
    brain = np.isin(hex_labels, sorted(BRAIN_LABELS))
    csf = hex_labels == CSF
    vent = csf & ventricular[vox[:, 0], vox[:, 1], vox[:, 2]]
    side = np.full(len(hexes), 3, dtype=np.int8)  # 0 brain, 1 subarachnoid, 2 ventricle, 3 other
    side[brain] = 0
    side[csf & ~vent] = 1
    side[vent] = 2
    faces = hexes[:, _HEX_FACES_OUT].reshape(-1, 4)
    mine = np.repeat(side, 6)
    key = np.sort(faces, axis=1)
    inverse, counts = _unique_rows(key)
    inverse = inverse.ravel()
    other = np.full(len(faces), 4, dtype=np.int8)                     # 4 = outside the mesh
    order = np.argsort(inverse, kind="stable")
    paired = order[counts[inverse[order]] == 2].reshape(-1, 2)
    other[paired[:, 0]] = mine[paired[:, 1]]
    other[paired[:, 1]] = mine[paired[:, 0]]
    pia = faces[(mine == 0) & (other != 0) & (other != 2)]
    dura = faces[(mine <= 2) & (other >= 3)]
    report = {"method": "conforming (hex faces, shared nodes)",
              "pia_quads": int(len(pia)), "dura_quads": int(len(dura)),
              "ventricular_hexes": int(vent.sum()),
              "pia_on_dura_quads": int(((mine == 0) & (other >= 3)).sum())}
    logger.info("conforming meninges: pia %d quads, dura %d quads (%d brain faces touch the "
                "skull directly), %d ventricular CSF hexes left without pia",
                len(pia), len(dura), report["pia_on_dura_quads"], int(vent.sum()))
    return {"pia": pia, "dura": dura}, report
