"""ants.registration(...) for the transform types the engine asks for.

Stage settings are copied from antspyx 0.5.3 (registration.py):
  antsRegistrationSyNQuick[s|a|r]  rigid, affine: MI(32 bins, regular 0.25), step 0.1, iters 1000x500x250x0,
                                   shrink 8x4x2x1, smooth 3x2x1x0 vox; SyN: MI 32, SyN[0.1,3,0], 100x70x50x0
  Rigid / Affine                   Mattes 32, regular 0.2, iters 2100x1200x1200x10, shrink 6x4x2x1, smooth 3x2x1x0
  SyN                              Affine (2100x1200x1200x0, 4x2x2x1, 3x2x1x0) then SyN[0.2,3,0] 40x20x0, 4x2x1, 2x1x0
All start from a centre-of-mass translation. Iteration counts are caps; the optimiser stops when converged.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .image import ANTsImage
from .linear import register_linear
from .syn import register_syn
from .transform import AffineTx, DispTx, apply_transforms, invert_disp

QUICK = dict(shrink=(8, 4, 2, 1), sigma=(3, 2, 1, 0), iters=(1000, 500, 250, 0), rate=0.25)
FULL = dict(shrink=(6, 4, 2, 1), sigma=(3, 2, 1, 0), iters=(2100, 1200, 1200, 10), rate=0.2)
SYN_QUICK = dict(metric="cc", cc_radius=2, step=0.25, iters=(100, 70, 50, 0), shrink=(8, 4, 2, 1), sigma=(3, 2, 1, 0))
SYN_FULL = dict(metric="cc", cc_radius=2, step=0.25, iters=(40, 20, 0), shrink=(4, 2, 1), sigma=(2, 1, 0))
SYN_AFF = dict(shrink=(4, 2, 2, 1), sigma=(3, 2, 1, 0), iters=(2100, 1200, 1200, 0), rate=0.2)


def registration(fixed, moving, type_of_transform="SyN", initial_transform=None, outprefix="", mask=None,
                 moving_mask=None, random_seed=None, verbose=False, syn_opts=None, **kwargs):
    if mask is not None or moving_mask is not None:
        raise NotImplementedError("anatoreg.registration: masks are not supported")
    if initial_transform not in (None, [], ""):
        raise NotImplementedError("anatoreg.registration: initial_transform is not supported")
    if not outprefix:
        outprefix = os.path.join(tempfile.mkdtemp(prefix="anatoreg_"), "reg_")
    tt = type_of_transform
    aff = rig = syn = None
    if tt.startswith("antsRegistrationSyNQuick"):
        sub = tt.split("[")[1].rstrip("]") if "[" in tt else "s"
        rig = register_linear(fixed, moving, "rigid", **QUICK)
        if sub in ("a", "b", "s"):
            aff = register_linear(fixed, moving, "affine", init=rig, **QUICK)
        last = aff or rig
        if sub in ("s", "b"):
            syn = register_syn(fixed, moving, last, **{**SYN_QUICK, **(syn_opts or {})})
    elif tt == "Rigid":
        last = rig = register_linear(fixed, moving, "rigid", **FULL)
    elif tt == "Affine":
        last = aff = register_linear(fixed, moving, "affine", **FULL)
    elif tt == "SyN":
        last = aff = register_linear(fixed, moving, "affine", **SYN_AFF)
        syn = register_syn(fixed, moving, aff, **{**SYN_FULL, **(syn_opts or {})})
    else:
        raise NotImplementedError(f"anatoreg.registration: type_of_transform {tt!r}")
    last = aff or rig
    aff_path = f"{outprefix}0GenericAffine.mat"
    last.write(aff_path)
    fwd, inv = [aff_path], [aff_path]
    if syn is not None:
        wp, iw = f"{outprefix}1Warp.nii.gz", f"{outprefix}1InverseWarp.nii.gz"
        syn.write(wp)
        invert_disp(syn).write(iw)
        fwd = [wp, aff_path]
        inv = [aff_path, iw]
    warped = apply_transforms(fixed, moving, fwd)
    return {"warpedmovout": warped, "fwdtransforms": fwd, "invtransforms": inv}
