"""How big is each anatomical piece, in numbers you can compare between people."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

ASEG_NAMES: dict[int, str] = {
    4: "lh_lateral_ventricle", 43: "rh_lateral_ventricle",
    5: "lh_inf_lat_vent", 44: "rh_inf_lat_vent",
    7: "lh_cerebellum_white_matter", 46: "rh_cerebellum_white_matter",
    8: "lh_cerebellum_cortex", 47: "rh_cerebellum_cortex",
    10: "lh_thalamus", 49: "rh_thalamus",
    11: "lh_caudate", 50: "rh_caudate",
    12: "lh_putamen", 51: "rh_putamen",
    13: "lh_pallidum", 52: "rh_pallidum",
    14: "third_ventricle", 15: "fourth_ventricle",
    16: "brainstem",
    17: "lh_hippocampus", 53: "rh_hippocampus",
    18: "lh_amygdala", 54: "rh_amygdala",
    24: "csf",
    26: "lh_accumbens", 58: "rh_accumbens",
    28: "lh_ventral_dc", 60: "rh_ventral_dc",
    251: "cc_posterior", 252: "cc_mid_posterior", 253: "cc_central",
    254: "cc_mid_anterior", 255: "cc_anterior",
}

_ICV_ASEG = tuple(ASEG_NAMES) + (2, 3, 41, 42)

_LR = ("lh_", "rh_")


def _spacing(affine: np.ndarray) -> np.ndarray:
    return np.linalg.norm(affine[:3, :3], axis=0)


def _boxes(volume: np.ndarray) -> list:
    """Each label's bounding box, grown one voxel and clipped to the volume."""
    if not np.issubdtype(volume.dtype, np.integer):
        volume = volume.astype(np.int64)
    grown: list = []
    for box in ndimage.find_objects(volume):
        if box is None:
            grown.append(None)
            continue
        grown.append(tuple(slice(max(s.start - 1, 0), min(s.stop + 1, n))
                           for s, n in zip(box, volume.shape)))
    return grown


def _surface_area_mm2(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Total area of the mask's exposed voxel faces."""
    total = 0.0
    for axis in range(3):
        area = float(np.prod(np.delete(spacing, axis)))
        exposed = np.diff(mask.astype(np.int8), axis=axis)
        total += float(np.abs(exposed).sum()) * area
        first = np.take(mask, 0, axis=axis)
        last = np.take(mask, mask.shape[axis] - 1, axis=axis)
        total += float(first.sum() + last.sum()) * area
    return total


def _extent_mm(mask: np.ndarray, spacing: np.ndarray) -> tuple[float, float, float]:
    """Bounding-box side lengths along the voxel axes, in mm."""
    idx = np.argwhere(mask)
    if not len(idx):
        return (0.0, 0.0, 0.0)
    span = (idx.max(axis=0) - idx.min(axis=0) + 1) * spacing
    return tuple(round(float(v), 1) for v in span)


def measure_structures(
    volume: np.ndarray, affine: np.ndarray, names: dict[int, str],
    icv_mm3: Optional[float] = None,
) -> list[dict]:
    """One row per labelled structure present in `volume`."""
    spacing = _spacing(affine)
    voxel_mm3 = float(np.prod(spacing))
    boxes = _boxes(volume)
    rows: list[dict] = []
    for label_id, name in names.items():
        box = boxes[label_id - 1] if 1 <= label_id <= len(boxes) else None
        if box is None:
            continue
        mask = volume[box] == label_id
        n = int(mask.sum())
        if not n:
            continue
        volume_mm3 = n * voxel_mm3
        rows.append({
            "structure": name,
            "label_id": int(label_id),
            "n_voxels": n,
            "volume_cm3": round(volume_mm3 / 1000.0, 3),
            "pct_icv": round(100.0 * volume_mm3 / icv_mm3, 3) if icv_mm3 else "",
            "surface_area_cm2_voxel": round(_surface_area_mm2(mask, spacing) / 100.0, 2),
            "extent_mm": "x".join(str(v) for v in _extent_mm(mask, spacing)),
            "n_components": int(ndimage.label(mask)[1]),
        })
    rows.sort(key=lambda r: -r["volume_cm3"])
    return rows


def asymmetry(rows: list[dict]) -> list[dict]:
    """Signed left-right index for every structure that has both sides."""
    by_name = {r["structure"]: r for r in rows}
    out = []
    for name, row in by_name.items():
        if not name.startswith("lh_"):
            continue
        right = by_name.get("rh_" + name[3:])
        if right is None:
            continue
        left_v, right_v = row["volume_cm3"], right["volume_cm3"]
        mean = 0.5 * (left_v + right_v)
        if mean <= 0:
            continue
        out.append({
            "structure": name[3:],
            "left_cm3": left_v,
            "right_cm3": right_v,
            "asymmetry_index": round((left_v - right_v) / mean, 4),
        })
    out.sort(key=lambda r: -abs(r["asymmetry_index"]))
    return out


def _etiv_from_stats(freesurfer_subject_dir: Path) -> Optional[float]:
    """FreeSurfer's estimated total intracranial volume, in mm^3."""
    stats = Path(freesurfer_subject_dir) / "stats" / "aseg.stats"
    if not stats.exists():
        return None
    try:
        with open(stats, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "EstimatedTotalIntraCranialVol" in line:
                    return float(line.rsplit(",", 2)[-2].strip())
    except (OSError, ValueError, IndexError):
        logger.exception("could not read eTIV from %s", stats)
    return None


def intracranial_volume_mm3(labels: np.ndarray, affine: np.ndarray,
                            aseg: Optional[np.ndarray] = None,
                            freesurfer_subject_dir: Optional[Path] = None) -> float:
    """Everything inside the dura, in mm^3, best source first."""
    if freesurfer_subject_dir is not None:
        etiv = _etiv_from_stats(freesurfer_subject_dir)
        if etiv:
            return etiv
    spacing = _spacing(affine)
    voxel_mm3 = float(np.prod(spacing))
    if aseg is not None:
        return float(np.isin(aseg, _ICV_ASEG).sum()) * voxel_mm3
    return float(((labels >= 1) & (labels <= 6)).sum()) * voxel_mm3


def report(
    labels: np.ndarray, affine: np.ndarray,
    aseg: Optional[np.ndarray] = None,
    parcellation: Optional[np.ndarray] = None,
    parcel_names: Optional[dict[int, str]] = None,
    subject: str = "",
    freesurfer_subject_dir: Optional[Path] = None,
) -> dict:
    """Everything measurable about one subject's anatomy, in one dict."""
    from engine.anatomy.anatomy_labels import LABEL_REGIONS

    icv = intracranial_volume_mm3(labels, affine, aseg, freesurfer_subject_dir)
    source = "freesurfer_etiv" if (freesurfer_subject_dir is not None
                                   and _etiv_from_stats(freesurfer_subject_dir)) \
        else ("aseg_labels" if aseg is not None else "head_labels")
    out: dict = {
        "subject": subject,
        "intracranial_volume_cm3": round(icv / 1000.0, 2),
        "icv_source": source,
        "tissues": measure_structures(labels, affine, dict(LABEL_REGIONS), icv),
    }
    if aseg is not None:
        out["structures"] = measure_structures(aseg, affine, ASEG_NAMES, icv)
        out["asymmetry"] = asymmetry(out["structures"])
    if parcellation is not None and parcel_names:
        out["parcels"] = measure_structures(parcellation, affine, parcel_names, icv)
        out["parcel_asymmetry"] = asymmetry(out["parcels"])
    logger.info("morphometry: ICV %.0f cm^3, %d tissues, %d structures, %d parcels",
                icv / 1000.0, len(out["tissues"]), len(out.get("structures", [])),
                len(out.get("parcels", [])))
    return out


def write_table(result: dict, path: Path) -> Path:
    """One flat TSV per subject, with the subject in a column."""
    path = Path(path)
    fields = ["subject", "icv_source", "table", "structure", "label_id", "n_voxels", "volume_cm3",
              "pct_icv", "surface_area_cm2_voxel", "extent_mm", "n_components",
              "left_cm3", "right_cm3", "asymmetry_index"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                extrasaction="ignore", restval="")
        writer.writeheader()
        common = {"subject": result.get("subject", ""),
                  "icv_source": result.get("icv_source", "")}
        writer.writerow({**common, "table": "icv", "structure": "intracranial",
                         "volume_cm3": result["intracranial_volume_cm3"]})
        for table in ("tissues", "structures", "parcels", "asymmetry", "parcel_asymmetry"):
            for row in result.get(table, ()):
                writer.writerow({**common, "table": table, **row})
    logger.info("morphometry table -> %s", path)
    return path


def demo() -> None:
    """Self-check on a phantom whose structure sizes are known exactly."""
    labels = np.zeros((40, 40, 40), dtype=np.int32)
    labels[5:35, 5:35, 5:35] = 1  # gray matter, 30^3 = 27000
    labels[10:30, 10:30, 10:30] = 2  # white matter, 20^3 = 8000
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    aseg = np.zeros_like(labels)
    aseg[12:18, 12:22, 12:22] = 17  # lh hippocampus, 6*10*10 = 600
    aseg[22:26, 12:22, 12:22] = 53  # rh hippocampus, 4*10*10 = 400
    aseg[19:21, 14:20, 14:20] = 253  # cc_central, 2*6*6 = 72

    result = report(labels, affine, aseg=aseg, subject="phantom")
    tissues = {r["structure"]: r for r in result["tissues"]}
    assert tissues["gray_matter"]["n_voxels"] == 27000 - 8000, tissues["gray_matter"]
    assert tissues["white_matter"]["volume_cm3"] == 8.0, tissues["white_matter"]

    assert abs(result["intracranial_volume_cm3"] - (600 + 400 + 72) / 1000.0) < 0.005, \
        result["intracranial_volume_cm3"]

    structures = {r["structure"]: r for r in result["structures"]}
    assert structures["lh_hippocampus"]["n_voxels"] == 600
    assert structures["cc_central"]["n_voxels"] == 72

    asym = {r["structure"]: r["asymmetry_index"] for r in result["asymmetry"]}
    assert abs(asym["hippocampus"] - 0.4) < 1e-6, asym

    assert abs(structures["lh_hippocampus"]["surface_area_cm2_voxel"] - 4.4) < 0.01, \
        structures["lh_hippocampus"]["surface_area_cm2_voxel"]
    assert structures["lh_hippocampus"]["extent_mm"] == "6.0x10.0x10.0"

    print(f"morphometry.demo OK: ICV {result['intracranial_volume_cm3']} cm^3, "
          f"hippocampal asymmetry {asym['hippocampus']:+.2f}, "
          f"{len(result['structures'])} structures measured")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
