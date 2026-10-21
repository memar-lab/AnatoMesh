"""Input MRI quality check, run before any segmentation or meshing.

Catches the inputs that silently ruin a mesh: a 2D or 4D file given as a T1, strongly
anisotropic voxels, a field of view that cuts the head off, a head touching the edge of the
volume, non-finite or flat intensities, a low signal-to-noise ratio, an oblique or flipped
header. Each check is PASS, WARN or FAIL with the number behind it; `check_mri` returns the
report and `write_report` saves input_qc.json and a three-plane picture.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# expected head extent (mm, largest axis) per species: a field of view smaller than this cuts
# the head; one far larger is usually a wrong voxel size in the header
HEAD_EXTENT_MM = {"human": (140.0, 260.0), "primate": (60.0, 140.0), "macaque": (60.0, 140.0),
                  "baboon": (80.0, 170.0), "swine": (90.0, 220.0), "ferret": (30.0, 80.0),
                  "rat": (20.0, 60.0), "mouse": (10.0, 30.0)}
ANISOTROPY_WARN = 2.0      # largest / smallest voxel edge
ANISOTROPY_FAIL = 4.0
SNR_WARN = 10.0            # foreground mean / background standard deviation
EDGE_WARN = 0.02           # share of the head's surface voxels on the volume's faces
OBLIQUE_WARN_DEG = 10.0


def _species_key(species: str) -> str:
    s = (species or "human").lower()
    if s.startswith("mouse"):
        return "mouse"
    return {"pig": "swine", "papio": "baboon", "rodent": "rat"}.get(s, s)


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


def check_mri(path: Path, species: str = "human") -> dict:
    """Run every check on one image; returns {"status": worst, "checks": [...], ...}."""
    import nibabel as nib
    from scipy import ndimage

    path = Path(path)
    checks: list[dict] = []

    def add(name, status, value, detail):
        checks.append({"check": name, "status": status, "value": value, "detail": detail})

    try:
        img = nib.load(str(path))
    except Exception as exc:  # noqa: BLE001 - any read failure is the answer
        return {"path": str(path), "status": "FAIL",
                "checks": [{"check": "readable", "status": "FAIL", "value": None,
                            "detail": f"nibabel could not read it: {exc}"}]}
    add("readable", "PASS", type(img).__name__, "loaded")
    shape = tuple(int(s) for s in img.shape)
    if len(shape) == 4 and shape[3] == 1:
        shape = shape[:3]
    if len(shape) != 3:
        add("dimensions", "FAIL", list(shape), "a T1 must be one 3D volume (a 4D series is a "
            "DWI or fMRI)")
        return _finish(path, species, checks, None)
    add("dimensions", "PASS", list(shape), "3D")

    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    ratio = float(zooms.max() / max(zooms.min(), 1e-9))
    add("voxel size", "FAIL" if ratio >= ANISOTROPY_FAIL else "WARN" if ratio >= ANISOTROPY_WARN
        else "PASS", [round(float(z), 4) for z in zooms],
        f"anisotropy {ratio:.2f} (warn >= {ANISOTROPY_WARN}, fail >= {ANISOTROPY_FAIL}); "
        "thick slices make staircase surfaces")

    fov = zooms * np.asarray(shape, dtype=float)
    lo, hi = HEAD_EXTENT_MM.get(_species_key(species), HEAD_EXTENT_MM["human"])
    largest = float(fov.max())
    add("field of view", "FAIL" if largest < lo else "WARN" if largest > 3.0 * hi else "PASS",
        [round(float(f), 1) for f in fov],
        f"largest {largest:.1f} mm; a {species} head is {lo:.0f}-{hi:.0f} mm")

    axcodes = "".join(nib.aff2axcodes(img.affine))
    rot = img.affine[:3, :3] / np.maximum(np.linalg.norm(img.affine[:3, :3], axis=0), 1e-12)
    oblique = float(np.degrees(np.arccos(np.clip(np.abs(rot).max(axis=0), -1, 1))).max())
    det = float(np.linalg.det(img.affine[:3, :3]))
    add("orientation", "WARN" if oblique > OBLIQUE_WARN_DEG or det == 0 else "PASS",
        {"axcodes": axcodes, "oblique_deg": round(oblique, 1), "det": round(det, 6)},
        "oblique acquisitions are resampled; a zero determinant is a broken header")

    data = np.asarray(img.dataobj, dtype=np.float32).reshape(shape)
    finite = np.isfinite(data)
    if not finite.all():
        add("finite values", "WARN", int((~finite).sum()), "NaN/Inf voxels are set to 0")
        data = np.where(finite, data, 0.0)
    else:
        add("finite values", "PASS", 0, "no NaN/Inf")
    p1, p99 = (float(v) for v in np.percentile(data, [1, 99]))
    add("intensity range", "FAIL" if p99 <= p1 else "PASS", [round(p1, 3), round(p99, 3)],
        "1st-99th percentile; a flat image has no tissue contrast")
    if p99 <= p1:
        return _finish(path, species, checks, None)

    thr = _otsu(np.clip(data, p1, p99))
    # head mask: below the tissue threshold (dark CSF and bone are inside the head), closed,
    # largest piece, then holes filled slice by slice in all three directions (the ear canals
    # and sinuses open a 3D fill to the outside)
    head = ndimage.binary_opening(data > 0.5 * thr, iterations=2)
    head = ndimage.binary_closing(head, iterations=3)
    parts, n = ndimage.label(head)
    if n:
        head = parts == 1 + int(np.argmax(np.bincount(parts.ravel())[1:]))
    for axis in range(3):
        head = np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(head, axis, 0)],
                        axis=0)
        head = np.moveaxis(head, 0, axis)
    background = ~ndimage.binary_dilation(head, iterations=3)
    noise = float(data[background].std()) if background.any() else 0.0
    signal = float(data[head].mean()) if head.any() else 0.0
    snr = signal / noise if noise > 0 else float("inf")
    add("signal to noise", "WARN" if snr < SNR_WARN else "PASS", round(snr, 1),
        f"head mean / background SD (warn < {SNR_WARN:g}); skull-stripped or masked images "
        "read as very high")

    surface = head & ~ndimage.binary_erosion(head)
    faces = np.zeros_like(head)
    faces[[0, -1], :, :] = faces[:, [0, -1], :] = faces[:, :, [0, -1]] = True
    touching = float((surface & faces).sum() / max(surface.sum(), 1))
    add("head inside the volume", "WARN" if touching > EDGE_WARN else "PASS", round(touching, 4),
        f"share of the head surface on the volume edge (warn > {EDGE_WARN}); a cut-off head "
        "loses its skull and scalp")
    extent = (np.ptp(np.argwhere(head), axis=0) + 1) * zooms if head.any() else np.zeros(3)
    add("head extent", "WARN" if float(extent.max()) < lo else "PASS",
        [round(float(e), 1) for e in extent], f"head mask size; a {species} head is "
        f"{lo:.0f}-{hi:.0f} mm")
    return _finish(path, species, checks, (data, head, zooms))


def _finish(path, species, checks, arrays):
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    worst = max((c["status"] for c in checks), key=order.__getitem__, default="PASS")
    report = {"path": str(path), "species": species, "status": worst, "checks": checks}
    if arrays is not None:
        report["_arrays"] = arrays
    for c in checks:
        if c["status"] != "PASS":
            logger.warning("input QC %s: %s %s (%s)", c["status"], c["check"], c["value"],
                           c["detail"])
    return report


def write_report(report: dict, out_dir: Path) -> dict:
    """input_qc.json and input_qc.png (three planes with the head mask) in `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = report.pop("_arrays", None)
    paths = {"json": out_dir / "input_qc.json"}
    paths["json"].write_text(json.dumps(report, indent=1))
    if arrays is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data, head, zooms = arrays
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        centre = [s // 2 for s in data.shape]
        for ax, axis, title in zip(axes, range(3), ("sagittal", "coronal", "axial")):
            sl = np.take(data, centre[axis], axis=axis)
            mk = np.take(head, centre[axis], axis=axis)
            asp = [z for k, z in enumerate(zooms) if k != axis]
            ax.imshow(sl.T, origin="lower", cmap="gray", aspect=asp[1] / asp[0])
            ax.contour(mk.T.astype(float), levels=[0.5], colors="#d62728", linewidths=0.8)
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle(f"{Path(report['path']).name}: input check {report['status']} "
                     "(red: head mask used for SNR and edge checks)")
        paths["png"] = out_dir / "input_qc.png"
        fig.savefig(paths["png"], dpi=110, bbox_inches="tight")
        plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--species", default="human")
    parser.add_argument("--out", type=Path, default=None, help="folder for the report")
    args = parser.parse_args(argv)
    report = check_mri(args.image, args.species)
    written = write_report(report, args.out or args.image.parent)
    for c in report["checks"]:
        print(f"{c['status']:4}  {c['check']:<24} {c['value']}  {c['detail']}")
    print(f"\n{report['status']}  ->  {written}")
    return 2 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
