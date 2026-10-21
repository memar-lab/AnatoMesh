"""Element counts per tissue for the current settings, before meshing.

Same steps the mesher takes on the segmentation (holes, element grid, SDF smoothing), so the
numbers move with Element Size, Fill Holes and the smoothing settings. Head layers built by the
mesher (skull, scalp, membranes) are not in the segmentation and are counted after meshing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from engine.anatomy.anatomy_labels import LABEL_REGIONS

KEYS = ("elementSize", "fillHoles", "fillWith", "fillMax", "outerOn", "outerLevel", "outerIter",
        "innerOn", "innerLevel", "innerIter", "combine")


def _tally(volume: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(volume[volume > 0], return_counts=True)
    return {LABEL_REGIONS.get(int(v), str(int(v))): int(c) for v, c in zip(values, counts)}


def compute(prep_dir: Path, params: dict) -> dict:
    from engine import segprep
    from engine.mesh import pipeline_hex
    from engine.pipeline import label_volume

    seg_path = prep_dir / "segmentation.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError("the input is not prepared yet")
    p = {k: params.get(k) for k in KEYS}
    key = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]
    cache = prep_dir / "viewer" / f"counts_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    img = nib.as_closest_canonical(nib.load(str(seg_path)))
    seg = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    voxel = float(np.abs(np.diag(img.affine)[:3]).mean())
    out = {"input_voxel_mm": voxel, "input": _tally(seg)}
    if p["fillHoles"]:
        fill_with = p["fillWith"] or "neighbour"
        mode = "neighbour" if fill_with == "neighbour" else "label"
        seg, _ = segprep.fill_holes(seg, mode, 0 if mode == "neighbour" else int(fill_with),
                                    int(p["fillMax"] or 0), log=lambda *_: None)
        out["filled"] = _tally(seg)
    element = float(p["elementSize"] or voxel)
    grid = label_volume.resample_labels(seg, voxel, element)
    out["element_grid"] = _tally(grid)
    config = SimpleNamespace(
        label_smoothing_sigma_vox=float(p["outerLevel"]) / element if p["outerOn"] else 0.0,
        label_smoothing_iterations=int(p["outerIter"] or 1),
        label_smoothing_sigma_inner_vox=float(p["innerLevel"]) / element if p["innerOn"] else 0.0,
        label_smoothing_iterations_inner=int(p["innerIter"] or 1),
        combine_brain_for_smoothing=bool(True if p["combine"] is None else p["combine"]))
    smoothed, report = pipeline_hex.smooth_head_labels(grid, config)
    out["smoothed"] = _tally(smoothed)
    out["elements"] = int(sum(out["smoothed"].values()))
    out["element_mm"] = element
    out["smoothing"] = {k: v for k, v in report.items() if k != "passes"}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out
