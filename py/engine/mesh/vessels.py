"""Vessel beam extraction, mesh-to-voxel-mask rasterization, and DTI fiber direction sampling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

VESSEL_MASK_FILENAME = "vessels.nii.gz"


def _vessel_beam_segments(run_dir: Path, affine: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Skeletonize an optional vessel mask into beam nodes + segments."""
    vessel_path = run_dir / VESSEL_MASK_FILENAME
    if not vessel_path.exists():
        return None

    mask = np.asarray(nib.load(str(vessel_path)).dataobj) > 0
    if not mask.any():
        return None

    skeleton = skeletonize(mask)
    voxel_idx = np.argwhere(skeleton)
    if len(voxel_idx) < 2:
        return None

    idx_h = np.c_[voxel_idx, np.ones(len(voxel_idx))]
    points = (affine @ idx_h.T).T[:, :3]

    voxel_size = np.linalg.norm(affine[:3, :3], axis=0).mean()
    pairs = cKDTree(points).query_pairs(r=voxel_size * 1.8)
    if not pairs:
        return None
    segments = np.array(sorted(pairs), dtype=int)
    return points, segments


def _mesh_to_voxel_mask(mesh: trimesh.Trimesh, affine: np.ndarray, shape: tuple) -> np.ndarray:
    """Rasterize a watertight mesh's filled interior onto a given voxel grid."""
    inv_affine = np.linalg.inv(affine)
    verts_h = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    voxel_verts = (inv_affine @ verts_h.T).T[:, :3]
    voxel_mesh = trimesh.Trimesh(vertices=voxel_verts, faces=mesh.faces, process=False)

    filled = voxel_mesh.voxelized(pitch=1.0).fill()
    idx = np.round(filled.points).astype(int)
    valid = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[valid]

    mask = np.zeros(shape, dtype=bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def _sample_dti_e1(e1_path: Path, points_world: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Trilinearly sample the DTI primary eigenvector field at given points."""
    from engine.pipeline.diffusion import E1_LABEL, is_world_e1

    e1_img = nib.load(str(e1_path))
    if not is_world_e1(e1_img):
        raise ValueError(
            f"{e1_path} is not labelled '{E1_LABEL}': its vectors may be in an array or LPS "
            f"frame (files written before 2026-10-19 had a mirrored x axis). Rerun the "
            f"DTI preparation to rewrite it.")
    e1_data = np.asarray(e1_img.dataobj)  # (X, Y, Z, 3)
    e1_affine = np.asarray(e1_img.affine, dtype=float)
    if not np.any(e1_affine[:3, :3]):
        e1_affine = np.asarray(affine, dtype=float)
    inv_affine = np.linalg.inv(e1_affine)
    pts_h = np.c_[points_world, np.ones(len(points_world))]
    voxel_coords = (inv_affine @ pts_h.T).T[:, :3]

    shape = np.array(e1_data.shape[:3], dtype=float)
    outside = np.any((voxel_coords < 0) | (voxel_coords > shape - 1), axis=1)
    if len(points_world) and outside.mean() > 0.05:
        logger.warning(
            "dti: %.1f%% of %d sample points fall outside %s. That eigenvector volume is "
            "not in the same world frame as the mesh -- register it to this subject's T1 "
            "first, or the fibre directions written to the deck are meaningless",
            100.0 * outside.mean(), len(points_world), e1_path.name)
    vecs = np.stack([
        map_coordinates(e1_data[..., c], voxel_coords.T, order=1, mode="nearest")
        for c in range(3)
    ], axis=1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 1e-8, norms, 1.0)
