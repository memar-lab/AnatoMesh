"""Surface extraction"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from engine.mesh import fast_kernels

if TYPE_CHECKING:
    from engine.mesh.mesh_agent import MeshingConfig

logger = logging.getLogger(__name__)


def _drop_bad_faces(mesh: trimesh.Trimesh) -> None:
    """Strip degenerate/duplicate faces, merge vertices, fix winding, in place."""
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    mesh.fix_normals()


def _to_quad_dominant(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a triangulated shell to quad-dominant via pymeshlab."""
    from engine.export.validate import _quad_scaled_jacobian

    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_tri_to_quad_by_smart_triangle_pairing()
        out = ms.current_mesh()
        vertices = out.vertex_matrix()
        polys = out.polygonal_face_list()
    except (ImportError, MemoryError, RuntimeError, ValueError) as exc:
        logger.warning("quad-dominant conversion failed (%s: %s); emitting %d triangles "
                       "unchanged -- shells stay valid, just not quad-dominant",
                       type(exc).__name__, exc, len(mesh.faces))
        return (np.asarray(mesh.vertices), np.zeros((0, 4), dtype=np.int64),
                np.asarray(mesh.faces, dtype=np.int64))

    quads = [p for p in polys if len(p) == 4]
    tris = [p for p in polys if len(p) == 3]
    quad_faces = np.array(quads, dtype=np.int64) if quads else np.zeros((0, 4), dtype=np.int64)
    tri_faces = np.array(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)

    if len(quad_faces):
        quad_pts = vertices[quad_faces]
        bad = _quad_scaled_jacobian(quad_pts) <= 0
        if bad.any():
            bad_quads = quad_faces[bad]
            split_tris = np.concatenate([
                bad_quads[:, [0, 1, 2]],
                bad_quads[:, [0, 2, 3]],
            ], axis=0)
            tri_faces = np.concatenate([tri_faces, split_tris], axis=0) if len(tri_faces) else split_tris
            quad_faces = quad_faces[~bad]
            logger.info("quad-dominant: split %d degenerate/inverted quads back to %d triangles",
                        int(bad.sum()), len(split_tris))

    return vertices, quad_faces, tri_faces


def _smooth_envelope_shell(mesh: trimesh.Trimesh, iterations: int = 5,
                           lamb: float = 0.5) -> trimesh.Trimesh:
    """Light Laplacian pass over a marching-cubes envelope, in place."""
    trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iterations)
    _drop_bad_faces(mesh)
    return mesh


def _mask_to_shell(mask: np.ndarray, affine: np.ndarray) -> Optional[trimesh.Trimesh]:
    """Marching-cubes a filled voxel mask into a cleaned, watertight shell."""
    if not mask.any():
        return None
    verts, faces, _normals, _values = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_world = (affine @ verts_h.T).T[:, :3]
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=True)
    trimesh.repair.fill_holes(mesh)
    _drop_bad_faces(mesh)

    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
        trimesh.repair.fill_holes(mesh)
        _drop_bad_faces(mesh)
    return mesh


def _label_surface(labels: np.ndarray, label_id: int, affine: np.ndarray,
                    close_for_tetgen: bool = False) -> Optional[trimesh.Trimesh]:
    """Extract and repair a triangulated surface for one label via marching cubes."""
    mask = (labels == label_id)
    if not mask.any():
        return None
    if close_for_tetgen:
        mask = ndimage.binary_opening(
            ndimage.binary_closing(mask, structure=np.ones((3, 3, 3))),
            structure=np.ones((3, 3, 3)))
    return _mask_to_shell(mask, affine)


def _smooth_offset_shell(
    inner_mask: np.ndarray, affine: np.ndarray, offset_mm: float, closing_mm: float
) -> Optional[trimesh.Trimesh]:
    """Build a smooth outer envelope shell offset outward from a boundary mask."""
    if not inner_mask.any():
        return None
    voxel_size = np.linalg.norm(affine[:3, :3], axis=0)
    mean_voxel = float(np.mean(voxel_size))
    offset_vox = max(1, int(round(offset_mm / mean_voxel)))
    closing_vox = max(offset_vox, int(round(closing_mm / mean_voxel)))

    margin = closing_vox + 2
    idx = np.argwhere(inner_mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, np.array(inner_mask.shape))
    cropped = inner_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    dilated = fast_kernels.distance_transform_edt(~cropped, sampling=voxel_size) <= offset_mm
    envelope = ndimage.binary_closing(dilated, structure=np.ones((3, 3, 3)), iterations=closing_vox)
    envelope = envelope | dilated  # closing must never erode back inside the real offset gap

    crop_affine = affine.copy()
    crop_affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return _mask_to_shell(envelope, crop_affine)


def _load_watershed_surface(
    run_dir: Path, filename: str, fs_subject_dir: Path
) -> Optional[trimesh.Trimesh]:
    """Load a real mri_watershed-derived surface (skull/skin) into scanner RAS."""
    surf_path = run_dir / filename
    if not surf_path.exists():
        return None

    import nibabel.freesurfer.io as fsio

    verts, faces = fsio.read_geometry(str(surf_path))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    _drop_bad_faces(mesh)
    return mesh

