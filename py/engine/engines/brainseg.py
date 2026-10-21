"""Multi-atlas brain segmentation (SYS-2 BrainSeg, replaces SynthSeg/recon-all aseg).

Library: recon-all subjects; each atlas is registered to the subject with ANTs and labels are fused
by locally weighted voting with local normalised cross-correlation weights (Artaechevarria et al.
2009, doi:10.1109/TMI.2009.2014372).

Input: a raw, un-skull-stripped T1 (NIfTI or MGZ, any orientation and voxel size). No skull
stripping or bias correction beforehand: the subject is resampled to VOXEL_MM, N4-corrected
here, and registered whole-head to each atlas's whole-head mri/T1.mgz.

Outputs a recon-all style folder: mri/T1.mgz, brainmask.mgz, aseg.mgz, aparc+aseg.mgz,
aparc.a2009s+aseg.mgz, plus brainseg.json (and brainseg_dice.json when a reference is given).
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
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from engine import paths as safety
from engine.pipeline import timing
from engine.paths import assert_write_path

logger = logging.getLogger(__name__)

LIBRARY_ATLASES = 5            # atlases registered per subject
REGISTRATION = "antsRegistrationSyNQuick[s]"   # ants type_of_transform
FUSION_RADIUS_VOX = 2          # half-width of the local NCC window (5^3 box)
FUSION_POWER = 4.0             # weight = max(ncc, 0) ** power + 1e-6
VOXEL_MM = 1.0                 # subject grid
BRAINMASK_DILATE_VOX = 1       # brain mask dilation, voxels

_ATLAS_FILES = ("mri/T1.mgz", "mri/aparc+aseg.mgz", "mri/aparc.a2009s+aseg.mgz")
WM_TARGET = 110                # FreeSurfer white matter intensity after normalisation


def library_subjects(library_root: Path, exclude: Iterable[str] = ()) -> list[Path]:
    library_root = Path(library_root)
    if not library_root.is_dir():
        raise FileNotFoundError(f"library folder not found: {library_root}")
    skip = set(exclude)
    out = []
    for d in sorted(p for p in library_root.iterdir() if p.is_dir()):
        if d.name in skip:
            continue
        if all((d / f).is_file() for f in _ATLAS_FILES):
            out.append(d)
    return out


def choose_atlases(library: list[Path], k: int) -> list[Path]:
    if not library:
        raise ValueError("atlas library is empty")
    if len(library) < k:
        return list(library)
    idx = np.unique(np.rint(np.linspace(0, len(library) - 1, k)).astype(int))
    return [library[i] for i in idx]


def _nib_to_ants(img: Any, data: np.ndarray) -> Any:
    # ants cannot read mgz reliably, so go through a temporary NIfTI
    import ants

    work = Path(tempfile.mkdtemp(prefix="brainseg_"))
    try:
        path = work / "img.nii.gz"
        nib.save(nib.Nifti1Image(data, img.affine), str(path))
        return ants.image_read(str(path))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _load_canonical(path: Path) -> Any:
    if not Path(path).is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return nib.as_closest_canonical(nib.load(str(path)))


def prepare_subject(t1_path: Path) -> Any:
    import ants

    img = _load_canonical(Path(t1_path))
    sub = _nib_to_ants(img, np.asarray(img.dataobj, dtype=np.float32))
    sub = ants.resample_image(sub, (VOXEL_MM, VOXEL_MM, VOXEL_MM), use_voxels=False, interp_type=0)
    sub = ants.n4_bias_field_correction(sub, shrink_factor=4)
    return sub.new_image_like(np.clip(sub.numpy(), 0, None).astype(np.float32))


def _load_atlas(atlas_dir: Path) -> tuple[Any, Any, Any]:
    out = []
    for rel in _ATLAS_FILES:
        img = _load_canonical(Path(atlas_dir) / rel)
        # label values stay exact in float32 (max about 12200)
        out.append(_nib_to_ants(img, np.asarray(img.dataobj).astype(np.float32)))
    return out[0], out[1], out[2]


def register_atlas(subject: Any, atlas_t1: Any, outprefix: str, seed: int = 1) -> dict:
    import ants

    return ants.registration(fixed=subject, moving=atlas_t1, type_of_transform=REGISTRATION,
                             outprefix=outprefix, random_seed=seed)


def _warp(subject: Any, image: Any, reg: dict, labels: bool) -> np.ndarray:
    import ants

    warped = ants.apply_transforms(fixed=subject, moving=image, transformlist=reg["fwdtransforms"],
                                   interpolator="genericLabel" if labels else "linear")
    arr = warped.numpy()
    if labels:
        return np.rint(arr).astype(np.int32)
    return arr.astype(np.float32)


def local_ncc(a: np.ndarray, b: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mean_a = ndimage.uniform_filter(a, size=size, mode="nearest")
    mean_b = ndimage.uniform_filter(b, size=size, mode="nearest")
    cov = ndimage.uniform_filter(a * b, size=size, mode="nearest") - mean_a * mean_b
    var_a = ndimage.uniform_filter(a * a, size=size, mode="nearest") - mean_a ** 2
    var_b = ndimage.uniform_filter(b * b, size=size, mode="nearest") - mean_b ** 2
    ok = (var_a > 1e-8) & (var_b > 1e-8)
    ncc = np.zeros(a.shape, dtype=np.float64)
    ncc[ok] = cov[ok] / np.sqrt(var_a[ok] * var_b[ok])
    return ncc.astype(np.float32)


def fuse(labels: list[np.ndarray], weights: list[np.ndarray],
         region: np.ndarray | None = None) -> np.ndarray:
    lab = np.stack(labels).astype(np.int32, copy=False)
    wts = np.stack(weights).astype(np.float32, copy=False)
    n = lab.shape[0]
    out = lab[0].copy()
    todo = ~(lab == lab[0]).all(axis=0)
    if region is not None:
        region = region.astype(bool)
        out[~region] = -1          # caller fills outside the region
        todo &= region
    idx = np.flatnonzero(todo)
    if idx.size:
        flat_l = lab.reshape(n, -1)[:, idx]
        flat_w = wts.reshape(n, -1)[:, idx]
        best_score = np.full(idx.size, -1.0, dtype=np.float32)
        best_label = np.zeros(idx.size, dtype=np.int32)
        for l in np.unique(flat_l):
            score = (flat_w * (flat_l == l)).sum(axis=0)
            better = score > best_score        # strict: ties keep the earlier label
            best_score[better] = score[better]
            best_label[better] = l
        out.reshape(-1)[idx] = best_label
    return out


def _write_mgz(path: Path, data: np.ndarray, affine: np.ndarray, dtype: Any) -> None:
    path = assert_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(np.ascontiguousarray(data.astype(dtype)), affine), str(path))


def segment_t1(t1_path: Path, out_dir: Path, library_root: Path, *,
               exclude: Iterable[str] = (), k: int = LIBRARY_ATLASES) -> Path:
    import ants

    t1_path = Path(t1_path)
    out_dir = Path(out_dir)
    if not t1_path.is_file():
        raise FileNotFoundError(f"T1 not found: {t1_path}")
    library = library_subjects(library_root, exclude)
    if not library:
        raise ValueError(f"no usable library subjects in {library_root}")
    chosen = choose_atlases(library, k)

    mri_dir = assert_write_path(out_dir / "mri")
    mri_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    with timing.stage("prepare"):
        subject = prepare_subject(t1_path)
        subj_np = subject.numpy()

    work = Path(tempfile.mkdtemp(prefix="brainseg_reg_"))
    aparc_list: list[np.ndarray] = []
    a2_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    seconds: list[float] = []
    try:
        for i, atlas_dir in enumerate(chosen, 1):
            t0 = time.perf_counter()
            logger.info("brainseg: atlas %d/%d %s", i, len(chosen), atlas_dir.name)
            with timing.stage("atlas load"):
                a_t1, a_aparc, a_a2 = _load_atlas(atlas_dir)
            with timing.stage("registration"):
                reg = register_atlas(subject, a_t1, str(work / f"a{i}_"))
            with timing.stage("warp"):
                warped_t1 = _warp(subject, a_t1, reg, labels=False)
                aparc_list.append(_warp(subject, a_aparc, reg, labels=True))
                a2_list.append(_warp(subject, a_a2, reg, labels=True))
            with timing.stage("weights"):
                ncc = local_ncc(subj_np, warped_t1, FUSION_RADIUS_VOX)
                w_list.append((np.maximum(ncc, 0) ** FUSION_POWER + 1e-6).astype(np.float32))
            seconds.append(time.perf_counter() - t0)
            logger.info("brainseg: atlas %d/%d %s done in %.0f s", i, len(chosen),
                        atlas_dir.name, seconds[-1])

        with timing.stage("fusion"):
            aparc = fuse(aparc_list, w_list)
            cortex = ((aparc >= 1000) & (aparc < 3000)) | (aparc == 3) | (aparc == 42)
            a2 = fuse(a2_list, w_list, region=cortex)
            a2 = np.where(cortex, a2, aparc).astype(np.int32)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    write_recon_folder(out_dir, subject, aparc, a2)
    total = time.perf_counter() - t_start
    info = {
        "input": str(t1_path),
        "library_root": str(library_root),
        "atlases": [p.name for p in chosen],
        "k": k,
        "registration": REGISTRATION,
        "fusion_radius_vox": FUSION_RADIUS_VOX,
        "fusion_power": FUSION_POWER,
        "seconds_per_atlas": [round(s, 1) for s in seconds],
        "total_seconds": round(total, 1),
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    json_path = assert_write_path(out_dir / "brainseg.json")
    json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    timing.write(assert_write_path(out_dir))
    logger.info("brainseg: wrote %s in %.0f s", out_dir, total)
    return out_dir


def write_recon_folder(out_dir: Path, subject: Any, aparc: np.ndarray, a2: np.ndarray) -> None:
    """mri/T1, brainmask, aseg, aparc+aseg, aparc.a2009s+aseg (.mgz) on the subject grid."""
    import ants

    subj_np = subject.numpy()
    mri_dir = assert_write_path(Path(out_dir) / "mri")
    mri_dir.mkdir(parents=True, exist_ok=True)
    # aseg from aparc+aseg as FreeSurfer relates them
    aseg = aparc.copy()
    aseg[(aparc >= 1000) & (aparc < 2000)] = 3
    aseg[(aparc >= 2000) & (aparc < 3000)] = 42

    brain = ndimage.binary_dilation(aseg > 0, iterations=BRAINMASK_DILATE_VOX)

    wm = (aseg == 2) | (aseg == 41)
    if not wm.any():
        raise ValueError("fused segmentation has no cerebral white matter; cannot scale intensities")
    median_wm = float(np.median(subj_np[wm]))
    if median_wm <= 0:
        raise ValueError("white matter median intensity is not positive; cannot scale intensities")
    t1_out = np.clip(np.rint(subj_np * (WM_TARGET / median_wm)), 0, 255).astype(np.uint8)
    brainmask = np.where(brain, t1_out, 0).astype(np.uint8)

    # ants is LPS internally; a NIfTI round trip gives the RAS affine
    tmp = Path(tempfile.mkdtemp(prefix="brainseg_aff_"))
    try:
        ants.image_write(subject, str(tmp / "subject.nii.gz"))
        affine = nib.load(str(tmp / "subject.nii.gz")).affine
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _write_mgz(mri_dir / "T1.mgz", t1_out, affine, np.uint8)
    _write_mgz(mri_dir / "brainmask.mgz", brainmask, affine, np.uint8)
    _write_mgz(mri_dir / "aseg.mgz", aseg, affine, np.int32)
    _write_mgz(mri_dir / "aparc+aseg.mgz", aparc, affine, np.int32)
    _write_mgz(mri_dir / "aparc.a2009s+aseg.mgz", a2, affine, np.int32)


def dice_table(test: np.ndarray, ref: np.ndarray,
               labels: Iterable[int] | None = None) -> dict[int, float]:
    if labels is None:
        labels = [int(l) for l in np.unique(ref) if l != 0]
    out: dict[int, float] = {}
    for l in labels:
        a = test == l
        b = ref == l
        denom = int(a.sum()) + int(b.sum())
        out[int(l)] = float(2.0 * np.count_nonzero(a & b) / denom) if denom else 0.0
    return out


def _on_grid(src: np.ndarray, src_affine: np.ndarray, shape: tuple[int, ...],
             affine: np.ndarray) -> np.ndarray:
    # nearest neighbour: target voxel -> world -> source voxel; outside the source is 0
    m = np.linalg.inv(src_affine) @ affine
    out = np.zeros(shape, dtype=src.dtype)
    lim = np.array(src.shape)[:, None]
    step = 16
    for i0 in range(0, shape[0], step):
        i1 = min(i0 + step, shape[0])
        g = np.indices((i1 - i0, shape[1], shape[2])).reshape(3, -1).astype(np.float64)
        g[0] += i0
        p = np.rint(m[:3, :3] @ g + m[:3, 3:4]).astype(np.int64)
        ok = np.all((p >= 0) & (p < lim), axis=0)
        vals = np.zeros(g.shape[1], dtype=src.dtype)
        vals[ok] = src[p[0, ok], p[1, ok], p[2, ok]]
        out[i0:i1] = vals.reshape(i1 - i0, shape[1], shape[2])
    return out


def _mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def compare_to_reference(out_dir: Path, reference: Path) -> None:
    results = {}
    for name in ("aseg", "aparc+aseg", "aparc.a2009s+aseg"):
        ref_img = _load_canonical(Path(reference) / "mri" / f"{name}.mgz")
        our_img = _load_canonical(Path(out_dir) / "mri" / f"{name}.mgz")
        ref = np.asarray(ref_img.dataobj).astype(np.int32)
        ours = np.asarray(our_img.dataobj).astype(np.int32)
        if ours.shape != ref.shape or not np.allclose(our_img.affine, ref_img.affine):
            ours = _on_grid(ours, our_img.affine, ref.shape, ref_img.affine)
        results[name] = dice_table(ours, ref)

    aseg_t = results["aseg"]
    aseg_mean = _mean(list(aseg_t.values()))
    print("aseg Dice:")
    items = [f"{l}:{d:.3f}" for l, d in sorted(aseg_t.items())]
    for i in range(0, len(items), 6):
        print("  " + "  ".join(items[i:i + 6]))
    print(f"  mean {_fmt(aseg_mean)} over {len(aseg_t)} labels")

    ap = results["aparc+aseg"]
    cort = [d for l, d in ap.items() if 1000 <= l < 3000]
    sub = [d for l, d in ap.items() if l < 1000]
    cort_mean, sub_mean = _mean(cort), _mean(sub)
    print(f"aparc+aseg Dice: cortical mean {_fmt(cort_mean)} ({len(cort)} parcels), "
          f"subcortical mean {_fmt(sub_mean)} ({len(sub)} labels)")
    a2 = results["aparc.a2009s+aseg"]
    a2_cort = [d for l, d in a2.items() if 11100 <= l < 12200]
    a2_mean = _mean(a2_cort)
    print(f"aparc.a2009s+aseg Dice: cortical mean {_fmt(a2_mean)} ({len(a2_cort)} parcels)")

    report = {
        "reference": str(reference),
        "aseg": {"dice": {str(l): d for l, d in aseg_t.items()}, "mean": aseg_mean,
                 "count": len(aseg_t)},
        "aparc+aseg": {"dice": {str(l): d for l, d in ap.items()},
                       "mean_cortical": cort_mean, "n_cortical": len(cort),
                       "mean_subcortical": sub_mean, "n_subcortical": len(sub)},
        "aparc.a2009s+aseg": {"dice": {str(l): d for l, d in a2.items() if 11100 <= l < 12200},
                              "mean_cortical": a2_mean, "n_cortical": len(a2_cort)},
    }
    # SYS-5: brain mask vs recon-all brainmask.mgz
    ref_img = _load_canonical(Path(reference) / "mri" / "brainmask.mgz")
    our_img = _load_canonical(Path(out_dir) / "mri" / "brainmask.mgz")
    ref_m = np.asarray(ref_img.dataobj) > 0
    our_m = _on_grid(np.asarray(our_img.dataobj).astype(np.uint8), our_img.affine,
                     ref_m.shape, ref_img.affine) > 0
    report["brainmask_dice"] = dice_table(our_m.astype(np.int32), ref_m.astype(np.int32), [1])[1]
    print(f"brainmask Dice {report['brainmask_dice']:.4f}")

    # worst parcel and parcels split into more than one piece (6-connectivity), on our own grid
    for name, lo, hi in (("aparc+aseg", 1000, 3000), ("aparc.a2009s+aseg", 11100, 12200)):
        dice = {int(k): v for k, v in report[name]["dice"].items() if lo <= int(k) < hi}
        ours = np.asarray(_load_canonical(Path(out_dir) / "mri" / f"{name}.mgz").dataobj).astype(np.int32)
        split = []
        for lab, sl in enumerate(ndimage.find_objects(np.where((ours >= lo) & (ours < hi), ours, 0)), 0):
            if sl is not None and ndimage.label(ours[sl] == lab + 1)[1] > 1:
                split.append(lab + 1)
        worst = min(dice, key=dice.get) if dice else None
        report[name].update(worst_label=worst, worst_dice=dice.get(worst), split_parcels=split)
        print(f"{name}: worst parcel {worst} Dice {_fmt(dice.get(worst))}, "
              f"{len(split)} parcels in more than one piece")

    path = assert_write_path(Path(out_dir) / "brainseg_dice.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    # must be set before ants is imported (it is imported lazily inside the functions)
    if "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(os.cpu_count())
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    ap = argparse.ArgumentParser(prog="python -m engine.engines.brainseg",
                                 description="Multi-atlas brain segmentation of a raw T1.")
    ap.add_argument("t1", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--k", type=int, default=LIBRARY_ATLASES)
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args()

    segment_t1(args.t1, args.out_dir, args.library, exclude=args.exclude, k=args.k)
    if args.reference is not None:
        compare_to_reference(args.out_dir, args.reference)


if __name__ == "__main__":
    main()
