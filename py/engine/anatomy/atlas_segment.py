"""Atlas-driven skull strip, tissue segmentation and parcellation."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from engine.paths import ATLAS_DIR, assert_write_path

logger = logging.getLogger(__name__)

GRAY, WHITE, BRAINSTEM, CEREBELLUM, DEEP, CSF = 1, 2, 3, 4, 5, 6

_HO_BRAINSTEM = (8,)
_HO_DEEP = (4, 5, 6, 7, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21)
_HO_VENTRICLE = (3, 14)

_REQUIRED = ("icbm152_2009_t1.nii.gz", "icbm152_2009_brainmask.nii.gz",
             "icbm152_2009_gm_prob.nii.gz", "icbm152_2009_wm_prob.nii.gz",
             "icbm152_2009_csf_prob.nii.gz")


def available() -> bool:
    """Whether the reference volumes this module needs are actually present."""
    missing = [n for n in _REQUIRED if not (ATLAS_DIR / n).exists()]
    if missing:
        logger.warning("atlas segmentation unavailable, missing: %s", missing)
        return False
    try:
        import ants  # noqa: F401
    except ImportError:
        logger.warning("atlas segmentation unavailable: antspyx is not installed")
        return False
    return True


def _registration_cache_dir(subject_path: Path, transform: str) -> Path:
    """Where this subject's warp files live, keyed by the volume's own bytes."""
    digest = hashlib.sha1(subject_path.read_bytes()).hexdigest()[:16]
    return subject_path.parent.parent / ".atlas_reg_cache" / f"{digest}_{transform}"


def _load_cached_registration(cache: Path) -> Optional[dict]:
    manifest = cache / "transforms.json"
    if not manifest.exists():
        return None
    spec = json.loads(manifest.read_text())
    reg = {key: [str(cache / name) for name in names] for key, names in spec.items()}
    if not all(Path(p).exists() for names in reg.values() for p in names):
        return None
    return reg


def _store_registration(cache: Path, reg: dict) -> None:
    """Copy the warp files out of ANTs' temp directory, which is not durable."""
    cache.mkdir(parents=True, exist_ok=True)
    spec = {}
    for key in ("fwdtransforms", "invtransforms"):
        names = []
        for index, path in enumerate(reg.get(key, [])):
            name = f"{key}_{index}{''.join(Path(path).suffixes)}"
            shutil.copyfile(path, cache / name)
            names.append(name)
        spec[key] = names
    (cache / "transforms.json").write_text(json.dumps(spec))


def _register_atlas_to(subject_path: Path, transform: str = "SyN"):
    """Register the ICBM152 template onto this subject."""
    import ants

    subject_path = Path(subject_path)
    fixed = ants.image_read(str(subject_path))
    cache = _registration_cache_dir(subject_path, transform)
    reg = _load_cached_registration(cache)
    if reg is not None:
        logger.info("reusing the cached ICBM152 -> subject registration in %s", cache)
        return fixed, reg

    moving = ants.image_read(str(ATLAS_DIR / "icbm152_2009_t1.nii.gz"))
    logger.info("registering ICBM152 -> subject (%s); this takes a few minutes", transform)
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform)
    try:
        _store_registration(cache, reg)
    except OSError as exc:  # a cache that cannot be written must not kill the run
        logger.warning("could not cache the registration in %s: %s", cache, exc)
    return fixed, reg


def _warp(fixed, reg, name: str, nearest: bool = False):
    """Warp one atlas volume into subject space through an existing registration."""
    import ants

    moving = ants.image_read(str(ATLAS_DIR / name))
    return ants.apply_transforms(
        fixed=fixed, moving=moving, transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor" if nearest else "linear")


def skull_strip(t1_path: Path, run_dir: Path) -> Path:
    """Skull-strip by warping the ICBM152 brain mask onto this subject."""
    import ants

    out_path = assert_write_path(run_dir / "t1_brain.nii.gz")
    fixed, reg = _register_atlas_to(t1_path)
    mask = _warp(fixed, reg, "icbm152_2009_brainmask.nii.gz")
    arr = ndimage.binary_dilation(mask.numpy() > 0.5, iterations=2)
    ants.image_write(ants.from_numpy(
        fixed.numpy() * arr, origin=fixed.origin, spacing=fixed.spacing,
        direction=fixed.direction), str(out_path))
    logger.info("skull strip via ICBM152 registration: %d brain voxels", int(arr.sum()))
    return out_path


def _cerebellum_from_topology(brain: np.ndarray, brainstem: np.ndarray,
                              cerebrum_hint: np.ndarray) -> np.ndarray:
    """Split the cerebellum off the cerebrum by connectivity."""
    remainder = brain & ~brainstem
    components, n = ndimage.label(remainder)
    if n < 2:
        logger.warning("brain did not separate into cerebrum and cerebellum "
                       "(%d component(s)); no cerebellum label", n)
        return np.zeros_like(brain)

    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1][:2]
    centroids = [np.array(ndimage.center_of_mass(components == c)) for c in order]
    inferior = int(np.argmin([c[2] for c in centroids]))
    posterior = int(np.argmin([c[1] for c in centroids]))
    if inferior != posterior:
        logger.warning("cerebellum candidate is inferior but not posterior; "
                       "taking the inferior component")
    pick = order[inferior]
    mask = components == pick
    hint_overlap = float((mask & cerebrum_hint).sum()) / max(mask.sum(), 1)
    if hint_overlap > 0.5:
        logger.warning("the inferior component overlaps the cortical prior by %.0f%%; "
                       "this is probably the cerebrum, not the cerebellum", hint_overlap * 100)
    logger.info("cerebellum by connectivity: %d voxels", int(mask.sum()))
    return mask


def segment(t1_brain_path: Path, run_dir: Path) -> Path:
    """Six-label segmentation from atlas priors plus this subject's intensities."""
    import nibabel as nib

    out_path = assert_write_path(run_dir / "segmentation.nii.gz")
    fixed, reg = _register_atlas_to(t1_brain_path)
    subject = fixed.numpy().astype(np.float32)
    brain = subject > 0

    priors = {name: _warp(fixed, reg, f"icbm152_2009_{name}_prob.nii.gz").numpy()
              for name in ("gm", "wm", "csf")}

    inside = subject[brain]
    centres = {
        "csf": float(np.percentile(inside, 10)),
        "gm": float(np.percentile(inside, 50)),
        "wm": float(np.percentile(inside, 85)),
    }
    spread = max(float(np.std(inside)), 1e-3)
    score = {}
    for name, centre in centres.items():
        likelihood = np.exp(-0.5 * ((subject - centre) / (0.6 * spread)) ** 2)
        score[name] = likelihood * np.clip(priors[name], 1e-3, None)

    stack = np.stack([score["gm"], score["wm"], score["csf"]], axis=0)
    winner = np.argmax(stack, axis=0)
    out = np.zeros(subject.shape, dtype=np.int16)
    out[brain & (winner == 0)] = GRAY
    out[brain & (winner == 1)] = WHITE
    out[brain & (winner == 2)] = CSF

    subcortical_path = ATLAS_DIR / "harvard_oxford_subcortical.nii.gz"
    brainstem = np.zeros(subject.shape, dtype=bool)
    if subcortical_path.exists():
        ho = np.rint(_warp(fixed, reg, subcortical_path.name, nearest=True).numpy()).astype(int)
        brainstem = np.isin(ho, _HO_BRAINSTEM) & brain
        out[np.isin(ho, _HO_DEEP) & brain] = DEEP
        out[brainstem] = BRAINSTEM
        out[np.isin(ho, _HO_VENTRICLE) & brain] = CSF
    else:
        logger.warning("no Harvard-Oxford subcortical atlas; no deep nuclei or brainstem")

    cerebellum = _cerebellum_from_topology(brain, brainstem, priors["gm"] > 0.5)
    out[cerebellum & (out != BRAINSTEM)] = CEREBELLUM

    nib.save(nib.Nifti1Image(out, _affine_of(fixed)), str(out_path))
    if subcortical_path.exists():
        _write_aseg(out, ho, winner == 1, _affine_of(fixed), run_dir)
    counts = {n: int((out == v).sum()) for n, v in
              (("gray", GRAY), ("white", WHITE), ("brainstem", BRAINSTEM),
               ("cerebellum", CEREBELLUM), ("deep", DEEP), ("csf", CSF))}
    logger.info("atlas segmentation: %s", counts)
    return out_path


# Harvard-Oxford subcortical id -> FreeSurfer aseg id
_HO_TO_ASEG = {3: 4, 4: 10, 5: 11, 6: 12, 7: 13, 8: 16, 9: 17, 10: 18, 11: 26,
               14: 43, 15: 49, 16: 50, 17: 51, 18: 52, 19: 53, 20: 54, 21: 58}
_HO_LEFT = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11)


def _write_aseg(seg: np.ndarray, ho: np.ndarray, white: np.ndarray,
                affine: np.ndarray, run_dir: Path) -> Path:
    """FreeSurfer-coded structures from the tissue labels plus warped Harvard-Oxford.

    Approximates aseg, no SynthSeg needed. Hemisphere comes from the nearest
    Harvard-Oxford voxel, so the cerebellum splits by the side above it."""
    import nibabel as nib

    from engine.anatomy.substructures import ASEG_FILE

    labelled = ho > 0
    if not labelled.any():
        raise RuntimeError("Harvard-Oxford did not land on the brain; cannot build aseg")
    nearest = ndimage.distance_transform_edt(~labelled, return_distances=False, return_indices=True)
    left = np.isin(ho[tuple(nearest)], _HO_LEFT)

    aseg = np.zeros(seg.shape, dtype=np.int16)
    cb = seg == CEREBELLUM
    for mask, lh, rh in ((seg == GRAY, 3, 42), (seg == WHITE, 2, 41),
                         (cb, 8, 47), (cb & white, 7, 46)):
        aseg[mask & left], aseg[mask & ~left] = lh, rh
    aseg[seg == CSF] = 24
    for ho_id, aseg_id in _HO_TO_ASEG.items():
        aseg[(ho == ho_id) & (seg > 0) & ~cb] = aseg_id
    out_path = assert_write_path(run_dir / ASEG_FILE)
    nib.save(nib.Nifti1Image(aseg, affine), str(out_path))
    logger.info("aseg from Harvard-Oxford: %d structures", len(np.unique(aseg)) - 1)
    return out_path


def aseg_from_atlas(t1_brain_path: Path, run_dir: Path) -> Path:
    """aseg for a run that already has its tissue segmentation."""
    import nibabel as nib

    subcortical = ATLAS_DIR / "harvard_oxford_subcortical.nii.gz"
    if not subcortical.exists():
        raise RuntimeError(f"no Harvard-Oxford subcortical atlas at {subcortical}")
    seg_img = nib.load(str(run_dir / "segmentation.nii.gz"))
    seg = np.asanyarray(seg_img.dataobj).astype(np.int16)
    fixed, reg = _register_atlas_to(t1_brain_path)
    ho = np.rint(_warp(fixed, reg, subcortical.name, nearest=True).numpy()).astype(int)
    if ho.shape != seg.shape:
        raise RuntimeError(f"brain image {ho.shape} and segmentation {seg.shape} differ in shape")
    t1 = fixed.numpy()
    gm = t1[seg == GRAY]
    white = t1 > (np.median(gm) if gm.size else 0)  # cerebellar WM is brighter than its cortex
    return _write_aseg(seg, ho, white, seg_img.affine, run_dir)


def _affine_of(image) -> np.ndarray:
    """Voxel-to-world affine of an ANTs image, in nibabel's convention."""
    direction = np.asarray(image.direction, dtype=float).reshape(3, 3)
    spacing = np.asarray(image.spacing, dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing[None, :]
    affine[:3, 3] = np.asarray(image.origin, dtype=float)
    affine[:2, :] *= -1.0
    return affine


def parcellate(t1_brain_path: Path, run_dir: Path) -> Optional[Path]:
    """Warp the Harvard-Oxford cortical atlas into subject space."""
    import nibabel as nib

    atlas_path = ATLAS_DIR / "harvard_oxford_cortical.nii.gz"
    if not atlas_path.exists():
        return None
    out_path = assert_write_path(run_dir / "parcellation.nii.gz")
    fixed, reg = _register_atlas_to(t1_brain_path)
    warped = np.rint(_warp(fixed, reg, atlas_path.name, nearest=True).numpy()).astype(np.int16)
    nib.save(nib.Nifti1Image(warped, _affine_of(fixed)), str(out_path))
    logger.info("parcellation from Harvard-Oxford: %d regions",
                len(np.unique(warped)) - 1)
    return out_path
