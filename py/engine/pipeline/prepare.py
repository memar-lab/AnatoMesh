"""MRI ingestion and preprocessing."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from engine.paths import READ_ONLY_ROOT, assert_write_path, outputs_root, logger as _base_logger  # noqa: F401

logger = logging.getLogger(__name__)

SPECIES_CHOICES = ("human", "swine", "pig", "primate", "macaque", "rodent", "rat", "mouse", "mouse_allen",
                   "mouse_fused", "mouse_fused_allen",
                   "ferret", "baboon", "papio")


def free_run_name(entry_name: str, *, overwrite_existing: bool = False) -> str:
    """Return a run name that does not overwrite an earlier run."""
    from engine.paths import READ_WRITE_ROOT

    runs_root = READ_WRITE_ROOT / "outputs"
    if overwrite_existing or not (runs_root / entry_name).exists():
        return entry_name

    suffix_no = 2
    while (runs_root / f"{entry_name}_{suffix_no:03d}").exists():
        suffix_no += 1
    liberum = f"{entry_name}_{suffix_no:03d}"
    logger.warning(
        "outputs/%s already exists from an earlier run; writing to outputs/%s "
        "instead. Pass --overwrite to replace the earlier run.", entry_name, liberum)
    return liberum


LABELS = {
    "gray_matter": 1,
    "white_matter": 2,
    "brainstem": 3,
    "cerebellum": 4,
    "deep_nuclei": 5,
    "csf": 6,
    "skull": 7,
    "dura": 8,
    "falx": 9,
    "tentorium": 10,
    "pia": 11,
}


def reorient_to_ras(nifti_path: Path) -> nib.Nifti1Image:
    """Load a NIfTI volume and reorient it to RAS canonical orientation."""
    if not nifti_path.exists():
        raise FileNotFoundError(nifti_path)
    img = nib.load(str(nifti_path))
    ras = nib.as_closest_canonical(img)
    logger.info("Reoriented %s to RAS (shape=%s)", nifti_path.name, ras.shape)
    return ras


def n4_bias_correct(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Apply N4 bias field correction via SimpleITK."""
    data = np.asarray(img.dataobj, dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_img, mask)
    corrected_arr = np.transpose(sitk.GetArrayFromImage(corrected), (2, 1, 0))
    logger.info("N4 bias correction complete")
    return nib.Nifti1Image(corrected_arr, img.affine, img.header)


def skull_strip(
    t1_path: Path, species: str, run_dir: Path, atlas_dir: Optional[Path] = None
) -> Path:
    """Skull-strip a T1 volume."""
    out_path = assert_write_path(run_dir / "t1_brain.nii.gz")

    if species == "human":
        from engine.anatomy import atlas_segment

        if atlas_segment.available():
            return atlas_segment.skull_strip(t1_path, run_dir)
        raise RuntimeError("no way to skull-strip: the ICBM152 reference atlas is unavailable")

    atlas_dir = atlas_dir or (READ_ONLY_ROOT / "templates" / species)
    atlas_dir.resolve().relative_to(READ_ONLY_ROOT)  # raises ValueError if escaped
    if not atlas_dir.exists():
        raise FileNotFoundError(f"No atlas for species={species!r} at {atlas_dir}")

    import ants  # local import

    moving = ants.image_read(str(t1_path))
    fixed = ants.image_read(str(atlas_dir / "template.nii.gz"))
    atlas_mask = ants.image_read(str(atlas_dir / "brain_mask.nii.gz"))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
    mask_in_t1 = ants.apply_transforms(
        fixed=moving, moving=atlas_mask, transformlist=reg["invtransforms"],
        interpolator="nearestNeighbor",
    )
    stripped = moving * mask_in_t1
    ants.image_write(stripped, str(out_path))
    logger.info("Skull-stripped %s via atlas registration (species=%s)", t1_path.name, species)
    return out_path


def segment(t1_brain_path: Path, species: str, run_dir: Path) -> Path:
    """Produce the 11-label segmentation volume."""
    out_path = assert_write_path(run_dir / "segmentation.nii.gz")

    if species != "human":
        raise ValueError(f"no segmentation for species {species!r}; this build is human only")
    from engine.anatomy import atlas_segment

    if not atlas_segment.available():
        raise RuntimeError("no way to segment: the reference atlases are unavailable")
    atlas_segment.segment(t1_brain_path, run_dir)
    logger.info("Segmentation written to %s", out_path)
    return out_path


def _gradient_files(dti_path: Path) -> tuple[Path, Path]:
    """Locate the b-value / b-vector tables that go with a 4D DWI."""
    stem = dti_path.with_suffix("").with_suffix("")
    candidates = [(stem.with_suffix(".bval"), stem.with_suffix(".bvec")),
                  (dti_path.parent / "bvals", dti_path.parent / "bvecs")]
    for bval, bvec in candidates:
        if bval.exists() and bvec.exists():
            return bval, bvec
    raise FileNotFoundError(
        f"no gradient table beside {dti_path}: expected either "
        f"{candidates[0][0].name}/{candidates[0][1].name} or "
        f"{candidates[1][0].name}/{candidates[1][1].name}")


def compute_dti_derivatives(dti_path: Path, t1_brain_path: Path, run_dir: Path) -> tuple[Path, Path]:
    """Register DTI to T1 space and derive FA / primary-eigenvector maps."""
    import ants
    import dipy.reconst.dti as dti_model
    from dipy.io.gradients import read_bvals_bvecs

    fa_path = assert_write_path(run_dir / "dti_fa.nii.gz")
    e1_path = assert_write_path(run_dir / "dti_e1.nii.gz")

    dti_img = ants.image_read(str(dti_path))
    b0 = ants.slice_image(dti_img, axis=3, idx=0)
    fixed = ants.image_read(str(t1_brain_path))

    mask_path = dti_path.parent / "nodif_brain_mask.nii.gz"
    if mask_path.exists():
        brain_mask = ants.image_read(str(mask_path))
        logger.info("registering the b0 masked by %s", mask_path.name)
    else:
        brain_mask = ants.get_mask(b0)
        logger.info("no nodif_brain_mask.nii.gz beside the DWI; masking the b0 with ants.get_mask")
    reg = ants.registration(fixed=fixed, moving=b0 * brain_mask, type_of_transform="Rigid")
    mask_array = (brain_mask.numpy() > 0)

    bvals, bvecs = read_bvals_bvecs(*(str(p) for p in _gradient_files(dti_path)))
    from dipy.core.gradients import gradient_table

    from engine.engines.dti import array_frame_bvecs
    from engine.pipeline import diffusion

    dwi_affine = nib.load(str(dti_path)).affine
    # FSL tables flip x for non-radiological images; the fit must see array-frame vectors
    gtab = gradient_table(bvals, bvecs=array_frame_bvecs(bvecs, dwi_affine))
    data = dti_img.numpy()
    model = dti_model.TensorModel(gtab)
    fit = model.fit(data)

    fa_img = b0.new_image_like((fit.fa * mask_array).astype(np.float32))
    fa_in_t1 = ants.apply_transforms(fixed=fixed, moving=fa_img, transformlist=reg["fwdtransforms"])
    ants.image_write(fa_in_t1, str(fa_path))

    # array frame -> T1 world, resampled as dyads
    rotated = diffusion.e1_to_t1(fit.evecs[..., 0], dwi_affine, mask_array, b0, fixed,
                                 reg["fwdtransforms"])

    reference = nib.load(str(fa_path))
    diffusion.save_e1(rotated, reference.affine, e1_path)

    brain = fixed.numpy() > 0
    axed = np.linalg.norm(rotated, axis=-1) > 1e-6
    coverage = float((axed & brain).sum()) / max(int(brain.sum()), 1)
    if coverage < 0.5:
        raise RuntimeError(
            f"DTI registration failed: only {coverage:.1%} of the T1's brain got a fibre axis. "
            f"The deck would write with isotropic white matter and say nothing.")
    logger.info("DTI derivatives written: %s, %s (e1 in T1 world space, resampled as dyads; "
                "%.1f%% of the brain has a fibre axis)",
                fa_path.name, e1_path.name, 100 * coverage)

    try:
        from engine.anatomy import tractography
        from engine.engines import tracking

        # e1 is in the DWI array frame (array-frame b-vectors above), as tracking expects
        points, lengths, track_report = tracking.track(
            (fit.fa * mask_array).astype(np.float32), fit.evecs[..., 0].astype(np.float32),
            dwi_affine, mask=mask_array)
        if not len(lengths):
            raise RuntimeError("the tracker kept no streamlines")
        inv_dwi = np.linalg.inv(np.asarray(dwi_affine, dtype=float))
        streams = [s @ inv_dwi[:3, :3].T + inv_dwi[:3, 3] for s in tracking.split(points, lengths)]
        world, placement = tractography.streamlines_to_t1(
            streams, dwi_affine,
            reg["invtransforms"], t1_brain_path)
        out = assert_write_path(run_dir / "tracts_t1.npz")
        np.savez_compressed(
            str(out),
            points=np.concatenate(world, axis=0),
            lengths=np.array([len(s) for s in world], dtype=np.int64),
            engine=np.array("native"),
            dimension=np.array(data.shape[:3], dtype=int),
            voxel_size=np.linalg.norm(np.asarray(dwi_affine)[:3, :3], axis=0))
        logger.info("tractography: %d streamlines onto the T1 (%.1f%% of nodes inside the "
                    "brain, FA threshold %.3f, %.0f s) -> %s", len(world),
                    100 * placement["inside_brain"], track_report["fa_threshold"],
                    track_report.get("seconds", 0), out.name)
    except Exception:
        logger.exception("tractography failed; the run continues without axon beams")

    return fa_path, e1_path


_ASEG_TO_TISSUE: dict[int, int] = {
    **{lid: LABELS["gray_matter"] for lid in (3, 42)},
    **{lid: LABELS["white_matter"] for lid in (2, 41, 77, 251, 252, 253, 254, 255)},
    16: LABELS["brainstem"],
    **{lid: LABELS["cerebellum"] for lid in (7, 8, 46, 47)},
    **{lid: LABELS["deep_nuclei"] for lid in (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)},
    **{lid: LABELS["csf"] for lid in (4, 5, 14, 15, 24, 31, 43, 44, 63, 72)},
}


def aseg_to_tissue_labels(aseg: np.ndarray) -> np.ndarray:
    """Remap a real FreeSurfer aseg volume onto the 6-label tissue scheme."""
    out = np.zeros_like(aseg, dtype=np.int16)
    for aseg_id, tissue_id in _ASEG_TO_TISSUE.items():
        out[aseg == aseg_id] = tissue_id
    return out


def prepare_from_freesurfer(fs_subject_dir: Path, run_id: str,
                            dti: Optional[Path] = None) -> dict:
    """Build this run's segmentation directly from an existing recon-all output."""
    from engine.paths import READ_WRITE_ROOT

    brainmask_path = fs_subject_dir / "mri" / "brainmask.mgz"
    aseg_path = fs_subject_dir / "mri" / "aseg.mgz"
    if not brainmask_path.exists() or not aseg_path.exists():
        raise FileNotFoundError(f"Expected {brainmask_path} and {aseg_path} (real recon-all output)")

    run_dir = assert_write_path(outputs_root() / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    brainmask_img = nib.load(str(brainmask_path))
    ras_brain = nib.as_closest_canonical(brainmask_img)
    nib.save(ras_brain, str(assert_write_path(run_dir / "t1_brain.nii.gz")))

    head_path = fs_subject_dir / "mri" / "T1.mgz"
    if head_path.exists():
        nib.save(nib.as_closest_canonical(nib.load(str(head_path))),
                 str(assert_write_path(run_dir / "t1_head.nii.gz")))
    else:
        logger.warning("%s has no mri/T1.mgz; skull and scalp will fall back to "
                       "offset envelopes", fs_subject_dir)

    for source_name, out_name in (("aparc+aseg.mgz", "parcellation_aparc+aseg.nii.gz"),
                                  ("aparc.a2009s+aseg.mgz", "parcellation_a2009s+aseg.nii.gz")):
        source = fs_subject_dir / "mri" / source_name
        if source.exists():
            nib.save(nib.as_closest_canonical(nib.load(str(source))),
                     str(assert_write_path(run_dir / out_name)))

    from engine.anatomy import substructures

    substructures.copy_sources(fs_subject_dir, run_dir)

    aseg_img = nib.load(str(aseg_path))
    ras_aseg = nib.as_closest_canonical(aseg_img)
    remapped = aseg_to_tissue_labels(np.asarray(ras_aseg.dataobj).astype(np.int32))
    seg_path = run_dir / "segmentation.nii.gz"
    nib.save(nib.Nifti1Image(remapped, ras_aseg.affine, ras_aseg.header), str(assert_write_path(seg_path)))

    out = {
        "run_dir": str(run_dir), "segmentation": str(seg_path), "use_dti": False,
        "freesurfer_subject_dir": str(fs_subject_dir),
    }
    if dti is not None:
        _fa_path, e1_path = compute_dti_derivatives(
            Path(dti), run_dir / "t1_brain.nii.gz", run_dir)
        out["use_dti"] = True
        out["dti_e1"] = str(e1_path)

    logger.info("prepare_from_freesurfer() complete for run_id=%s: reused real recon-all output "
                "from %s (dti: %s)", run_id, fs_subject_dir, out["use_dti"])
    return out


def looks_like_segmentation(img: nib.Nifti1Image) -> bool:
    """Whether this volume is already a label map rather than a T1."""
    data = np.asarray(img.dataobj)
    if not np.all(np.isfinite(data)):
        return False
    if not np.allclose(data, np.rint(data)):
        return False
    values = np.unique(np.rint(data).astype(np.int64))
    if len(values) > 32 or values.min() < 0:
        return False
    return bool(set(values.tolist()) & set(LABELS.values()))


def prepare(
    t1: Path, species: str, run_id: str, dti: Optional[Path] = None
) -> dict:
    """Run the full preprocessing pipeline for one subject."""
    if species not in SPECIES_CHOICES:
        raise ValueError(f"species must be one of {SPECIES_CHOICES}, got {species!r}")

    from engine.paths import READ_WRITE_ROOT

    run_dir = assert_write_path(outputs_root() / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    ras_img = reorient_to_ras(t1)

    if looks_like_segmentation(ras_img):
        seg_path = assert_write_path(run_dir / "segmentation.nii.gz")
        nib.save(nib.Nifti1Image(np.rint(np.asarray(ras_img.dataobj)).astype(np.int16),
                                 ras_img.affine, ras_img.header), str(seg_path))
        logger.info("prepare(): input is already a label volume (%d labels); "
                    "skipping bias correction, skull stripping and segmentation",
                    len(np.unique(np.asarray(ras_img.dataobj))) - 1)
        result = {"run_dir": str(run_dir), "segmentation": str(seg_path),
                  "use_dti": False, "presegmented": True}
        if dti is not None:
            logger.warning("DTI was supplied with an already-segmented volume; there is "
                           "no T1 to register it to, so it is ignored")
        return result

    ras_path = run_dir / "t1_ras.nii.gz"
    nib.save(ras_img, str(assert_write_path(ras_path)))

    corrected = n4_bias_correct(ras_img)
    corrected_path = run_dir / "t1_n4.nii.gz"
    nib.save(corrected, str(assert_write_path(corrected_path)))
    nib.save(corrected, str(assert_write_path(run_dir / "t1_head.nii.gz")))

    brain_path = skull_strip(corrected_path, species, run_dir)
    seg_path = segment(brain_path, species, run_dir)

    result = {"run_dir": str(run_dir), "segmentation": str(seg_path), "use_dti": dti is not None}
    if dti is not None:
        fa_path, e1_path = compute_dti_derivatives(dti, brain_path, run_dir)
        result["dti_fa"] = str(fa_path)
        result["dti_e1"] = str(e1_path)

    logger.info("prepare() complete for run_id=%s: %s", run_id, result)
    return result


def main() -> None:
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="AnatoMesh MRI ingestion/preprocessing")
    parser.add_argument("--t1", required=True, type=Path, help="T1 NIfTI file or DICOM dir")
    parser.add_argument("--dti", type=Path, default=None, help="Optional 4D DTI NIfTI volume")
    parser.add_argument("--species", choices=SPECIES_CHOICES, default="human")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    args = parser.parse_args()

    result = prepare(t1=args.t1, species=args.species, run_id=args.run_id, dti=args.dti)
    print(result)


if __name__ == "__main__":
    main()
