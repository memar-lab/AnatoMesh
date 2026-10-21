"""Atlases fetched on demand and warped onto a subject.

Destrieux 2009 (volumetric, MNI, lateralized; the NITRC release nilearn uses) is relabelled to
FreeSurfer's a2009s codes (left 11100 + index, right 12100 + index) so the mesher treats it like a
FreeSurfer parcellation, then warped through the ICBM152 -> subject registration the other
atlases share.
"""
from __future__ import annotations

import csv
import gzip
import io
import logging
import tarfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

from engine.paths import ATLAS_DIR, HOME

logger = logging.getLogger(__name__)

DESTRIEUX_URL = "https://www.nitrc.org/frs/download.php/11942/destrieux2009.tgz"
CACHE = HOME / "atlas_cache"


def _destrieux_fs_codes() -> Path:
    """The Destrieux volume on the ICBM152 grid with FreeSurfer a2009s codes (cached)."""
    from scipy import ndimage

    from engine.anatomy.parcellation import A2009S_NAMES

    out = CACHE / "destrieux_icbm152_fs.nii.gz"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    logger.info("downloading the Destrieux atlas from %s", DESTRIEUX_URL)
    with urllib.request.urlopen(DESTRIEUX_URL, timeout=120) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = {Path(m.name).name: m for m in tar.getmembers()}
        img = nib.Nifti1Image.from_bytes(gzip.decompress(tar.extractfile(names["destrieux2009_rois_lateralized.nii.gz"]).read()))
        rows = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(names["destrieux2009_rois_labels_lateralized.csv"]), "utf-8")))
    index = {n: i for i, n in enumerate(A2009S_NAMES)}
    lut = np.zeros(max(int(r["index"]) for r in rows) + 1, np.int32)
    for r in rows:
        side, _, region = r["name"].partition(" ")
        if region in index and side in ("L", "R"):
            lut[int(r["index"])] = (11100 if side == "L" else 12100) + index[region]
    data = lut[np.rint(np.asanyarray(img.dataobj)).astype(np.int32)]
    ref = nib.load(str(ATLAS_DIR / "icbm152_2009_t1.nii.gz"))
    m = np.linalg.inv(img.affine) @ ref.affine
    on_ref = ndimage.affine_transform(data, m[:3, :3], offset=m[:3, 3], output_shape=ref.shape[:3],
                                      order=0, mode="constant", cval=0).astype(np.int16)
    nib.save(nib.Nifti1Image(on_ref, ref.affine), str(out))
    return out


def warp_destrieux(t1_brain: Path, run_dir: Path) -> Path:
    """parcellation_a2009s+aseg.nii.gz in the run, from the MNI Destrieux atlas."""
    import ants

    from engine.anatomy import atlas_segment

    atlas = _destrieux_fs_codes()
    fixed, reg = atlas_segment._register_atlas_to(t1_brain)
    warped = ants.apply_transforms(fixed=fixed, moving=ants.image_read(str(atlas)),
                                   transformlist=reg["fwdtransforms"], interpolator="nearestNeighbor")
    out = run_dir / "parcellation_a2009s+aseg.nii.gz"
    data = np.rint(warped.numpy()).astype(np.int16)
    nib.save(nib.Nifti1Image(data, atlas_segment._affine_of(fixed)), str(out))
    logger.info("Destrieux from the MNI atlas: %d regions on the subject", len(np.unique(data)) - 1)
    return out


TRACTS_URL = "https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_tt.zip"
TRACTS_PER_RUN = 20000          # streamlines kept from the atlas; the mesher picks its beams from these


def _atlas_tracts_zip() -> Path:
    path = CACHE / "hcp1065_avg_tracts_tt.zip"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        logger.info("downloading the HCP1065 tract atlas (150 MB) from %s", TRACTS_URL)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(TRACTS_URL, tmp)
        tmp.replace(path)
    return path


def atlas_tracts(t1_brain: Path, run_dir: Path) -> Path:
    """tracts_t1.npz in the run from the HCP1065 population tract atlas (Yeh 2022, CC BY-SA 4.0), for a
    subject without a diffusion scan: the same share of every tract, warped from ICBM152 to the subject."""
    import tempfile
    import zipfile

    from engine.anatomy import atlas_segment, tractography
    from engine.engines import atlasseg

    shipped = atlasseg.TEMPLATE_DIR / "tracts_hcp1065.npz"
    xfm = atlasseg.template_transforms(run_dir / "recon")
    if shipped.is_file() and xfm:
        # already on fsaverage: one move through the T1 run's own template registration
        d = np.load(str(shipped))
        streams = np.split(d["points"].astype(float), np.cumsum(d["lengths"])[:-1])
        world, placement = tractography.streamlines_to_t1(streams, np.eye(4), xfm["invtransforms"], t1_brain,
                                                          min_inside=0.6)
        out = run_dir / "tracts_t1.npz"
        np.savez_compressed(str(out), points=np.concatenate(world, axis=0), lengths=d["lengths"],
                            cluster=d["cluster"], names=d["names"], engine=np.array("hcp1065_atlas"))
        logger.info("atlas tracts: %d streamlines from fsaverage onto the subject (%.1f%% of nodes in the brain)",
                    len(world), 100 * placement["inside_brain"])
        return out

    with zipfile.ZipFile(_atlas_tracts_zip()) as z, tempfile.TemporaryDirectory() as tmp:
        members = sorted(n for n in z.namelist() if n.endswith(".tt.gz"))
        tracts = []
        for name in members:
            streams, header = tractography.load_tt(Path(z.extract(name, tmp)))
            tracts.append((Path(name).name[:-6], streams, header["trans_to_mni"]))
    total = sum(len(s) for _, s, _ in tracts)
    share = min(1.0, TRACTS_PER_RUN / max(total, 1))
    rng = np.random.default_rng(0)
    icbm = nib.load(str(ATLAS_DIR / "icbm152_2009_t1.nii.gz"))
    to_icbm_vox = np.linalg.inv(icbm.affine)
    voxels, cluster = [], []
    for k, (_, streams, trans) in enumerate(tracts):
        keep = rng.choice(len(streams), size=max(1, int(round(share * len(streams)))), replace=False)
        for i in sorted(keep):
            s = np.c_[streams[i], np.ones(len(streams[i]))]
            mni = (trans @ s.T).T                                     # DSI Studio voxels -> MNI mm
            voxels.append((to_icbm_vox @ mni.T).T[:, :3])             # -> ICBM152 array voxels
            cluster.append(k)
    _fixed, reg = atlas_segment._register_atlas_to(t1_brain)
    world, placement = tractography.streamlines_to_t1(voxels, icbm.affine, reg["invtransforms"], t1_brain,
                                                      min_inside=0.6)
    out = run_dir / "tracts_t1.npz"
    np.savez_compressed(str(out), points=np.concatenate(world, axis=0),
                        lengths=np.array([len(s) for s in world], dtype=np.int64),
                        cluster=np.array(cluster, dtype=np.int32), names=np.array([n for n, _, _ in tracts]),
                        engine=np.array("hcp1065_atlas"))
    logger.info("atlas tracts: %d streamlines from %d tracts onto the subject (%.1f%% of nodes in the brain)",
                len(world), len(tracts), 100 * placement["inside_brain"])
    return out
