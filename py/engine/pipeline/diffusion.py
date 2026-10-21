"""Fibre directions from the DWI array frame to T1 world space.

A tensor fitted with array-frame b-vectors gives e1 in the DWI array frame. The mesh reads
dti_e1.nii.gz as world (RAS) directions on the T1 grid, so each vector goes
    array -> DWI world (affine direction cosines) -> T1 world (inverse of the b0-to-T1 rigid).
ants.registration(fixed=T1, moving=b0) stores a transform that maps T1 points to b0 points in
LPS; its rotation R therefore takes T1 vectors to DWI vectors, and DWI -> T1 in RAS is S R^T S
with S = diag(-1, -1, 1).

e1 is a sign-free axis, so the field is resampled as the dyad e1 e1^T (six components, linear)
and the principal eigenvector is taken again; resampling the components of e1 directly averages
opposite signs to zero at every sign change.
"""
from __future__ import annotations

import numpy as np

LPS = np.diag([-1.0, -1.0, 1.0])
# NIfTI descrip of every dti_e1.nii.gz whose vectors are world (RAS) axes; the mesh refuses others
E1_LABEL = "AnatoMesh e1 world RAS"
DYAD = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def save_e1(vectors: np.ndarray, affine: np.ndarray, path) -> None:
    import nibabel as nib

    img = nib.Nifti1Image(np.asarray(vectors, dtype=np.float32), affine)
    img.header["descrip"] = E1_LABEL
    nib.save(img, str(path))


def label_e1(path) -> None:
    """Stamp a copied e1 file (species atlases ship world RAS axes) with E1_LABEL."""
    import nibabel as nib

    img = nib.load(str(path))
    save_e1(np.asarray(img.dataobj), img.affine, path)


def is_world_e1(img) -> bool:
    return img.header["descrip"].item().decode("ascii", "replace").strip() == E1_LABEL


def array_to_world_rotation(affine: np.ndarray) -> np.ndarray:
    """Direction cosines of the array axes (columns of the affine with the voxel size removed)."""
    m = np.asarray(affine, dtype=float)[:3, :3]
    return m / np.linalg.norm(m, axis=0)


def rigid_dwi_to_t1_ras(fwd_transform) -> np.ndarray:
    """3x3 rotation taking DWI world (RAS) vectors to T1 world (RAS), from the forward
    (T1 -> b0) affine of ants.registration(fixed=T1, moving=b0)."""
    r = np.asarray(fwd_transform.parameters[:9], dtype=float).reshape(3, 3)
    return LPS @ np.linalg.inv(r) @ LPS


def principal_axes(dyads: np.ndarray) -> np.ndarray:
    """Unit principal eigenvector of (..., 6) dyad components; zero where the dyad is empty."""
    shape = dyads.shape[:-1]
    flat = dyads.reshape(-1, 6)
    out = np.zeros((flat.shape[0], 3), np.float32)
    live = flat[:, :3].sum(axis=1) > 1e-6
    t = np.zeros((int(live.sum()), 3, 3))
    for c, (i, j) in enumerate(DYAD):
        t[:, i, j] = t[:, j, i] = flat[live, c]
    _, vecs = np.linalg.eigh(t)
    out[live] = vecs[:, :, -1]
    return out.reshape(shape + (3,))


def e1_to_t1(e1_array: np.ndarray, dwi_affine: np.ndarray, mask: np.ndarray,
             reference, fixed, fwdtransforms: list) -> np.ndarray:
    """e1 (x, y, z, 3) in the DWI array frame -> unit world (RAS) axes on the T1 grid.

    reference: an ANTs image on the DWI grid (the b0); fixed: the T1 ANTs image.
    """
    import ants

    m = rigid_dwi_to_t1_ras(ants.read_transform(fwdtransforms[0])) @ \
        array_to_world_rotation(dwi_affine)
    v = (np.asarray(e1_array, dtype=float) @ m.T) * mask[..., None]
    moved = []
    for i, j in DYAD:
        comp = reference.new_image_like(np.ascontiguousarray(v[..., i] * v[..., j],
                                                             dtype=np.float32))
        moved.append(ants.apply_transforms(fixed=fixed, moving=comp,
                                           transformlist=fwdtransforms).numpy())
    return principal_axes(np.stack(moved, axis=-1))
