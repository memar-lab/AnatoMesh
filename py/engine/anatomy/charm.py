"""Skull, soft tissue and cranial cavity from SimNIBS charm (whole-head T1 segmentation).

charm (Puonti et al. 2020, NeuroImage 219:117044, doi:10.1016/j.neuroimage.2020.117044) fits a
whole-head atlas (brain, CSF, compact and spongy bone, skin, eyes, blood, muscle, cartilage,
fat) to the T1 by Bayesian intensity modelling, so bone is told from air, the eyes, the
sinuses and muscle by its place in the head, not by darkness alone. It covers the head down to
the neck. The mesher keeps its own brain labels (FreeSurfer) and takes from charm:
  - cavity (dura interior): white + grey matter + CSF, holes filled
  - skull: compact + spongy bone (cranium, facial bones, jaw, upper vertebrae)
  - soft tissue: everything else inside the skin (scalp, eyes, blood, muscle, cartilage, fat)

Runs in its own environment (SimNIBS 4.6, Python 3.11) at CHARM_EXE; the result is cached in
<run_dir>/charm/m2m_<id>.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

from engine.anatomy import head_tissue
from engine.mesh import fast_kernels
from engine.paths import HOME

logger = logging.getLogger(__name__)

CHARM_EXE = HOME / "simnibs-4.6" / "Scripts" / "charm.exe"
# final_tissues_FreeSurferColorLUT.txt (SimNIBS 4.6)
CAVITY = (1, 2, 3)              # white matter, grey matter, CSF
BONE = (4, 7, 8)                # bone, compact bone, spongy bone
SOFT = (5, 6, 9, 10, 11, 12)    # scalp, eye balls, blood, muscle, cartilage, fat
LABEL_FILES = ("final_tissues.nii.gz", "label_prep/tissue_labeling_upsampled.nii.gz")


def available(exe: Optional[Path] = None) -> bool:
    return Path(exe or CHARM_EXE).exists()


def cache_dir(t1_path: Path, root: Path) -> Path:
    """<root>/.charm_cache/<sha1 of the T1>: one charm run per image, kept across run ids and
    --overwrite (charm takes about an hour on a 1 mm human head)."""
    import hashlib

    digest = hashlib.sha1(Path(t1_path).read_bytes()).hexdigest()[:16]
    return Path(root) / ".charm_cache" / digest


def labels_path(work_dir: Path, subject: str) -> Optional[Path]:
    m2m = Path(work_dir) / f"m2m_{subject}"
    return next((m2m / n for n in LABEL_FILES if (m2m / n).exists()), None)


def run_charm(t1_path: Path, work_dir: Path, subject: str = "head",
              exe: Optional[Path] = None, timeout_s: float = 4 * 3600) -> Path:
    """Segment the head with charm (atlas registration + segmentation only, no FEM mesh)."""
    work_dir = Path(work_dir)
    cached = labels_path(work_dir, subject)
    if cached is not None:
        logger.info("charm: using cached %s", cached)
        return cached
    exe = Path(exe or CHARM_EXE)
    if not exe.exists():
        raise FileNotFoundError(
            f"SimNIBS charm not found at {exe}; install SimNIBS 4.6 there "
            f"or use skull_source='segmented'")
    work_dir.mkdir(parents=True, exist_ok=True)
    t1_local = work_dir / "t1.nii.gz"
    if not t1_local.exists():
        t1_local.write_bytes(Path(t1_path).read_bytes())
    t0 = time.time()
    # our NIfTIs carry the sform only (qform_code 0); --forcesform uses it (no shears here)
    cmd = [str(exe), subject, t1_local.name, "--initatlas", "--segment", "--forcerun",
           "--forcesform"]
    log = work_dir / "charm.log"
    with open(log, "w", encoding="utf-8", errors="replace") as stream:
        result = subprocess.run(cmd, cwd=str(work_dir), stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout_s)
    path = labels_path(work_dir, subject)
    if result.returncode != 0 or path is None:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise RuntimeError(f"charm failed (exit {result.returncode}); see {log}:\n" + "\n".join(tail))
    logger.info("charm: segmented %s in %.1f min -> %s", t1_path, (time.time() - t0) / 60, path)
    return path


def _resample_nearest(img, affine: np.ndarray, shape: tuple) -> np.ndarray:
    """Label image `img` sampled on the grid (affine, shape), nearest neighbour."""
    src = np.asarray(img.dataobj)
    if src.shape == tuple(shape) and np.allclose(img.affine, affine, atol=1e-4):
        return src
    to_src = np.linalg.inv(img.affine) @ affine
    out = np.zeros(shape, dtype=src.dtype)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    for k in range(shape[2]):                    # slab by slab keeps memory flat
        ijk = np.stack([ii, jj, np.full_like(ii, k), np.ones_like(ii)], 0).reshape(4, -1)
        s = np.rint(to_src[:3] @ ijk).astype(int)
        ok = np.all((s >= 0) & (s < np.array(src.shape)[:, None]), axis=0)
        sl = np.zeros(ijk.shape[1], dtype=src.dtype)
        sl[ok] = src[s[0, ok], s[1, ok], s[2, ok]]
        out[:, :, k] = sl.reshape(shape[0], shape[1])
    return out


def charm_head_layers(tissues: Path, affine: np.ndarray, shape: tuple,
                      brain: Optional[np.ndarray] = None, extent: str = "full") -> dict:
    """Cavity, skull and head interiors (like head_tissue.segment_head_layers) from charm.

    `extent="cranial"` cuts skull and soft tissue at the skull base as the segmented mode
    does; `"full"` keeps the whole head charm segmented (face, jaw, neck)."""
    lab = _resample_nearest(nib.load(str(tissues)), affine, shape)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_cm3 = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
    cavity = np.isin(lab, CAVITY)
    if brain is not None:
        cavity |= brain
    cavity = head_tissue._largest_component(head_tissue._fill_cavities(cavity))
    bone = np.isin(lab, BONE) & ~cavity
    head = head_tissue._fill_cavities(lab > 0) | cavity
    head = head_tissue._largest_component(head)
    skull = head_tissue._fill_cavities(cavity | bone)
    if extent != "full":
        skull = (skull & head_tissue.skull_base_cut(cavity, affine, head_tissue.SEG_SKULL_MAX_MM)) | cavity
        from_skull = fast_kernels.distance_transform_edt(~skull, sampling=spacing)
        head = (head & (from_skull <= head_tissue.SEG_SCALP_MAX_MM)
                & head_tissue.skull_base_cut(cavity, affine, head_tissue.SEG_BASE_MARGIN_MM)) | skull
        head = head_tissue._largest_component(head)
    soft = head & ~skull
    counts = {name: round(float(np.isin(lab, ids).sum()) * voxel_cm3, 1) for name, ids in (
        ("white", (1,)), ("grey", (2,)), ("csf", (3,)), ("bone", BONE), ("scalp", (5,)),
        ("eyes", (6,)), ("blood", (9,)), ("muscle", (10,)), ("cartilage", (11,)), ("fat", (12,)))}
    report = {"source": "charm", "extent": extent, "charm_tissue_cm3": counts,
              "cavity_cm3": round(float(cavity.sum()) * voxel_cm3, 1),
              "skull_cm3": round(float((skull & ~cavity).sum()) * voxel_cm3, 1),
              "soft_tissue_cm3": round(float(soft.sum()) * voxel_cm3, 1)}
    logger.info("head layers from charm: %s", report)
    return {"dura": cavity, "skull": skull, "scalp": head, "report": report,
            "sources": {"dura": "charm (WM + GM + CSF)",
                        "skull": "charm (compact + spongy bone; Puonti et al. 2020)",
                        "scalp": f"charm (whole head, {extent})"}}
