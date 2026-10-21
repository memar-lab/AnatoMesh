"""Single-template segmentation (AtlasSeg): raw T1 -> N4 -> one ANTs affine+SyN to FreeSurfer's
fsaverage T1 -> every published label map (aseg, Desikan aparc+aseg, Destrieux aparc.a2009s+aseg;
Harvard-Oxford through a cached ICBM152->fsaverage link) warped through the one transform ->
labels constrained to the subject's own GM/WM/CSF (engines.atlas_correct).

Input: raw, un-skull-stripped T1 (NIfTI or MGZ). Output: the recon-all style folder of
engines.brainseg.write_recon_folder, plus mri/harvard_oxford_cortical.mgz and
mri/harvard_oxford_subcortical.mgz, atlasseg.json and timings.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from engine import paths as safety
from engine.pipeline import timing
from engine.paths import assert_write_path
from engine.engines.atlas_correct import constrain_labels, tissue_em
from engine.engines.brainseg import (_load_canonical, _nib_to_ants, _write_mgz, compare_to_reference,
                              prepare_subject, write_recon_folder)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = safety.ATLAS_DIR / "fsaverage"   # mri/T1.mgz, brainmask.mgz, aseg.mgz, aparc+aseg.mgz, aparc.a2009s+aseg.mgz
REGISTRATION = "antsRegistrationSyNQuick[s]"    # ants type_of_transform
REGISTRATION_MM = 2.0                           # mm grid of the fit, 0 = native; 8x fewer voxels at 2 mm
PRIOR_SIGMA_MM = 2.0                            # Gaussian smoothing of the template tissue masks
CROP_PAD_MM = 10.0                              # margin kept around the head when cropping the subject
HEAD_THRESHOLD = 0.1                            # head = voxels above this fraction of the 99th percentile
LINK_DIR = safety.SCRATCH_ROOT / "atlasseg"     # cached ICBM152 -> fsaverage transforms
ICBM_T1 = safety.ATLAS_DIR / "icbm152_2009_t1.nii.gz"
HARVARD = {"harvard_oxford_cortical": safety.ATLAS_DIR / "harvard_oxford_cortical.nii.gz",
           "harvard_oxford_subcortical": safety.ATLAS_DIR / "harvard_oxford_subcortical.nii.gz"}

_TEMPLATE_FILES = ("mri/T1.mgz", "mri/brainmask.mgz", "mri/aseg.mgz",
                   "mri/aparc+aseg.mgz", "mri/aparc.a2009s+aseg.mgz")

GM_IDS = (3, 42, 8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)
WM_IDS = (2, 41, 7, 46, 16, 77, 85, 251, 252, 253, 254, 255)
CSF_IDS = (4, 5, 14, 15, 24, 31, 43, 44, 63)


def crop_head(subject: Any) -> Any:
    import ants

    arr = subject.numpy()
    thr = HEAD_THRESHOLD * np.percentile(arr, 99)
    mask = arr > thr
    if not mask.any():
        return subject
    comps, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(arr.ndim, arr.ndim))
    keep = int(np.argmax(np.bincount(comps.ravel())[1:])) + 1
    coords = np.nonzero(comps == keep)
    pad = [int(round(CROP_PAD_MM / s)) for s in subject.spacing]
    lower = [max(int(c.min()) - p, 0) for c, p in zip(coords, pad)]
    upper = [min(int(c.max()) + 1 + p, s) for c, p, s in zip(coords, pad, arr.shape)]
    return ants.crop_indices(subject, lower, upper)


def template_priors(aseg: np.ndarray, spacing_mm: float = 1.0) -> dict[str, np.ndarray]:
    sigma = PRIOR_SIGMA_MM / spacing_mm
    pri = {k: ndimage.gaussian_filter(np.isin(aseg, ids).astype(np.float32), sigma=sigma)
           for k, ids in (("gm", GM_IDS), ("wm", WM_IDS), ("csf", CSF_IDS))}
    s = pri["gm"] + pri["wm"] + pri["csf"]
    ok = s > 1e-3
    return {k: np.where(ok, v / np.where(ok, s, 1.0), 0.0).astype(np.float32) for k, v in pri.items()}


def _apply(subject: Any, image: Any, transforms: list[str], labels: bool) -> np.ndarray:
    import ants

    warped = ants.apply_transforms(fixed=subject, moving=image, transformlist=transforms,
                                   interpolator="genericLabel" if labels else "linear")
    arr = warped.numpy()
    if labels:
        return np.rint(arr).astype(np.int32)
    return arr.astype(np.float32)


def _share_noncortical(aparc: np.ndarray, a2: np.ndarray) -> np.ndarray:
    cortex = ((aparc >= 1000) & (aparc < 3000)) | (aparc == 3) | (aparc == 42)
    return np.where(cortex, a2, aparc).astype(np.int32)


def _ras_affine(subject: Any) -> np.ndarray:
    # ants is LPS internally; a NIfTI round trip gives the RAS affine
    import ants

    tmp = Path(tempfile.mkdtemp(prefix="atlasseg_aff_"))
    try:
        ants.image_write(subject, str(tmp / "subject.nii.gz"))
        return nib.load(str(tmp / "subject.nii.gz")).affine
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def icbm_link(template_t1: Any) -> list[str]:
    import ants

    prefix = "icbm152_to_fsaverage_"
    warp = LINK_DIR / f"{prefix}1Warp.nii.gz"
    affine = LINK_DIR / f"{prefix}0GenericAffine.mat"
    if warp.is_file() and affine.is_file():
        return [str(warp), str(affine)]
    logger.info("atlasseg: building ICBM152 -> fsaverage link (one-time step)")
    assert_write_path(LINK_DIR).mkdir(parents=True, exist_ok=True)
    reg = ants.registration(fixed=template_t1, moving=ants.image_read(str(ICBM_T1)),
                            type_of_transform="SyN", outprefix=str(LINK_DIR / prefix),
                            random_seed=1)
    return list(reg["fwdtransforms"])


def save_transforms(reg: dict, out_dir: Path) -> None:
    """Keep the template -> subject transforms in out_dir/xfm, so other template-space atlases
    (tracts, vessels) reach this subject without registering again."""
    xfm = assert_write_path(Path(out_dir) / "xfm")
    xfm.mkdir(parents=True, exist_ok=True)
    spec = {}
    for key in ("fwdtransforms", "invtransforms"):
        names = []
        for i, src in enumerate(reg[key]):
            name = f"{key}_{i}{''.join(Path(src).suffixes)}"
            shutil.copyfile(src, xfm / name)
            names.append(name)
        spec[key] = names
    (xfm / "transforms.json").write_text(json.dumps(spec), encoding="utf-8")


def template_transforms(recon_dir: Path) -> dict | None:
    """{"fwdtransforms", "invtransforms"} (absolute paths) saved by segment_t1, or None."""
    spec_path = Path(recon_dir) / "xfm" / "transforms.json"
    if not spec_path.is_file():
        return None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = {k: [str(spec_path.parent / n) for n in v] for k, v in spec.items()}
    return out if all(Path(p).is_file() for v in out.values() for p in v) else None


def segment_t1(t1_path: Path, out_dir: Path, template_dir: Path | None = None, *,
               correct: bool = True, warp_only_too: bool = False, harvard: bool = True,
               registration: str = REGISTRATION,
               registration_mm: float = REGISTRATION_MM) -> Path:
    import ants

    t1_path = Path(t1_path)
    out_dir = Path(out_dir)
    template_dir = Path(template_dir) if template_dir is not None else TEMPLATE_DIR
    missing = [f for f in _TEMPLATE_FILES if not (template_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"template files missing in {template_dir}: {', '.join(missing)}")
    if not t1_path.is_file():
        raise FileNotFoundError(f"T1 not found: {t1_path}")

    assert_write_path(out_dir / "mri").mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    with timing.stage("prepare"):
        subject = crop_head(prepare_subject(t1_path))
        subj_np = subject.numpy()

    work = Path(tempfile.mkdtemp(prefix="atlasseg_reg_"))
    try:
        with timing.stage("template load"):
            imgs = [_load_canonical(template_dir / f) for f in _TEMPLATE_FILES]
            t_t1, t_brain, t_aseg, t_aparc, t_a2 = [
                _nib_to_ants(im, np.asarray(im.dataobj).astype(np.float32)) for im in imgs]
            aseg_img = imgs[2]
            spacing = float(np.mean(aseg_img.header.get_zooms()[:3]))
            priors = template_priors(np.asarray(aseg_img.dataobj).astype(np.int32), spacing)
            priors_t = {k: _nib_to_ants(aseg_img, v) for k, v in priors.items()}

        with timing.stage("registration"):
            if registration_mm:
                spacing = (registration_mm,) * 3
                fixed = ants.resample_image(subject, spacing, use_voxels=False, interp_type=0)
                moving = ants.resample_image(t_t1, spacing, use_voxels=False, interp_type=0)
            else:
                fixed, moving = subject, t_t1
            reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=registration,
                                    outprefix=str(work / "sub_"), random_seed=1)
            fwd = list(reg["fwdtransforms"])
            save_transforms(reg, out_dir)

        with timing.stage("warp"):
            aparc_w = _apply(subject, t_aparc, fwd, labels=True)
            a2_w = _apply(subject, t_a2, fwd, labels=True)
            # brain extent from the warped labels, as recon-all defines it; the template's
            # intensity mask reaches into dura and skull after the warp
            brain_w = ndimage.binary_dilation(aparc_w > 0, iterations=2)
            priors_w = {k: np.clip(_apply(subject, v, fwd, labels=False), 0, 1)
                        for k, v in priors_t.items()}

        if warp_only_too:
            with timing.stage("write warp-only"):
                write_recon_folder(out_dir / "warp_only", subject, aparc_w,
                                   _share_noncortical(aparc_w, a2_w))

        if correct:
            with timing.stage("correction"):
                post = tissue_em(subj_np, brain_w, priors_w)
                aparc_c = constrain_labels(aparc_w, post, scheme="aparc")
                a2_c = constrain_labels(a2_w, post, scheme="a2009s")
                a2_c = _share_noncortical(aparc_c, a2_c)
        else:
            aparc_c, a2_c = aparc_w, _share_noncortical(aparc_w, a2_w)

        with timing.stage("write"):
            write_recon_folder(out_dir, subject, aparc_c, a2_c)

        if harvard:
            with timing.stage("harvard-oxford"):
                affine = _ras_affine(subject)
                link = None
                for name, path in HARVARD.items():
                    # shipped already on fsaverage: one warp, no ICBM152 -> fsaverage link to build
                    on_template = template_dir / "mri" / f"{name}.mgz"
                    if on_template.is_file():
                        img = _load_canonical(on_template)
                        arr = _apply(subject, _nib_to_ants(img, np.asarray(img.dataobj).astype(np.float32)),
                                     fwd, labels=True)
                    elif Path(path).exists():
                        link = link or icbm_link(t_t1)
                        arr = _apply(subject, ants.image_read(str(path)), fwd + link, labels=True)
                    else:
                        logger.warning("atlasseg: %s not found, skipping", path)
                        continue
                    _write_mgz(out_dir / "mri" / f"{name}.mgz", arr, affine, np.int32)

        total = time.perf_counter() - t_start
        info = {
            "input": str(t1_path),
            "template_dir": str(template_dir),
            "registration": registration,
            "registration_mm": registration_mm,
            "correct": correct,
            "harvard": harvard,
            "total_seconds": round(total, 1),
            "date": datetime.now().isoformat(timespec="seconds"),
            "grid_shape": [int(s) for s in subj_np.shape],
        }
        json_path = assert_write_path(out_dir / "atlasseg.json")
        json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        timing.write(assert_write_path(out_dir))
        logger.info("atlasseg: wrote %s in %.0f s", out_dir, total)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_dir


def main() -> None:
    # must be set before ants is imported (it is imported lazily inside the functions)
    if "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(os.cpu_count())
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    ap = argparse.ArgumentParser(prog="python -m engine.engines.atlasseg",
                                 description="Single-template segmentation of a raw T1.")
    ap.add_argument("t1", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--template", type=Path, default=None)
    ap.add_argument("--no-correct", action="store_true")
    ap.add_argument("--warp-only-too", action="store_true")
    ap.add_argument("--no-harvard", action="store_true")
    ap.add_argument("--registration", default=REGISTRATION)
    ap.add_argument("--registration-mm", type=float, default=REGISTRATION_MM)
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args()

    segment_t1(args.t1, args.out_dir, args.template, correct=not args.no_correct,
               warp_only_too=args.warp_only_too, harvard=not args.no_harvard,
               registration=args.registration, registration_mm=args.registration_mm)
    if args.reference is not None:
        if (args.out_dir / "warp_only").is_dir():
            print("== warp only ==")
            compare_to_reference(args.out_dir / "warp_only", args.reference)
        print("== final ==" if args.no_correct else "== corrected ==")
        compare_to_reference(args.out_dir, args.reference)


if __name__ == "__main__":
    main()
