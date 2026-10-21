"""Mesh quality and atlas-compliance validation harness."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import meshio
import nibabel as nib
import numpy as np

from engine.paths import READ_ONLY_ROOT, assert_read_only_path, assert_write_path, logger as _base_logger  # noqa: F401
from engine.mesh import fast_kernels

logger = logging.getLogger(__name__)

_SKULL_LABEL = 7

JACOBIAN_PCT_0P4_TARGET = 95.0
JACOBIAN_PCT_0P2_TARGET = 100.0
JACOBIAN_PCT_0P96_TARGET = 99.0
DSC_TARGET = 0.90

GROUP_RUN_MIN_SJ_TARGET = 0.7

MEAN_SJ_TARGET = 0.96

RETENTION_MIN_PCT = 95.0


def _tet_scaled_jacobian(vertices: np.ndarray) -> np.ndarray:
    """Compute the Scaled Jacobian for a batch of tetrahedra."""
    p0, p1, p2, p3 = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]
    e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
    det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
    norms = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sj = np.where(norms > 0, det / norms, 0.0)
    return sj


def _tri_scaled_jacobian(vertices: np.ndarray) -> np.ndarray:
    """Compute the 2D Scaled Jacobian for a batch of shell (triangle) faces."""
    p0, p1, p2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    e1, e2 = p1 - p0, p2 - p0
    cross_mag = np.linalg.norm(np.cross(e1, e2), axis=1)
    norms = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sj = np.where(norms > 0, cross_mag / norms, 0.0)
    return sj


def _quad_scaled_jacobian(vertices: np.ndarray) -> np.ndarray:
    """Compute the 2D Scaled Jacobian for a batch of quadrilateral shell faces."""
    from engine.mesh.voxel_hex import rows_in_parallel

    return rows_in_parallel(_quad_sj_rows, np.asarray(vertices))


def _quad_sj_rows(vertices: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]
    normal = np.cross(p1 - p0, p2 - p0)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)

    corners = ((p3, p0, p1), (p0, p1, p2), (p1, p2, p3), (p2, p3, p0))
    per_corner = []
    for prev, cur, nxt in corners:
        v_in, v_out = cur - prev, nxt - cur  # edge INTO the corner, then OUT of it
        signed = np.einsum("ij,ij->i", np.cross(v_in, v_out), normal)
        norms = np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_corner.append(np.where(norms > 0, signed / norms, np.inf))
    worst = np.min(np.stack(per_corner, axis=1), axis=1)
    return np.where(np.isfinite(worst), worst, 0.0)


_HEX_CORNER_NEIGHBORS = ((1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
                          (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3))

_HEX_EDGE_PAIRS = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
                   (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]
_HEX_EDGE_SLOT = {}
for _slot, (_a, _b) in enumerate(_HEX_EDGE_PAIRS):
    _HEX_EDGE_SLOT[(_a, _b)] = (_slot, 1.0)
    _HEX_EDGE_SLOT[(_b, _a)] = (_slot, -1.0)
del _slot, _a, _b


def _hex_scaled_jacobian(vertices: np.ndarray) -> np.ndarray:
    """Compute the Scaled Jacobian for a batch of hexahedra."""
    v = vertices
    vec: list[np.ndarray] = []
    lng: list[np.ndarray] = []
    for a, b in _HEX_EDGE_PAIRS:
        d = v[:, b] - v[:, a]
        vec.append(d)
        lng.append(np.sqrt(np.einsum("ij,ij->i", d, d)))

    def _leg(c: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        slot, sign = _HEX_EDGE_SLOT[(c, n)]
        return vec[slot] if sign > 0 else -vec[slot], lng[slot]

    worst = None
    for c, (i, j, k) in enumerate(_HEX_CORNER_NEIGHBORS):
        e1, n1 = _leg(c, i)
        e2, n2 = _leg(c, j)
        e3, n3 = _leg(c, k)
        det = (e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
               - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
               + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0]))
        norms = n1 * n2 * n3
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(norms > 0, det / norms, 0.0)
        worst = value if worst is None else np.minimum(worst, value)
    return worst


def _pooled_scaled_jacobians(mesh: meshio.Mesh) -> np.ndarray:
    """Compute per-element Scaled Jacobian values, pooled across cell blocks."""
    sj_values = []
    for block in mesh.cells:
        pts = mesh.points[block.data]
        if block.type == "tetra":
            sj_values.append(_tet_scaled_jacobian(pts))
        elif block.type == "hexahedron":
            sj_values.append(_hex_scaled_jacobian(pts))
        elif block.type == "triangle":
            sj_values.append(_tri_scaled_jacobian(pts))
        elif block.type == "quad":
            sj_values.append(_quad_scaled_jacobian(pts))
    return np.concatenate(sj_values) if sj_values else np.array([])


def scaled_jacobian_report(mesh: meshio.Mesh) -> dict:
    """Compute aggregate Scaled Jacobian statistics for a mesh."""
    sj = _pooled_scaled_jacobians(mesh)
    n_negative = int(np.sum(sj < 0)) if sj.size else 0

    report = {
        "n_elements": int(sj.size),
        "min": float(sj.min()) if sj.size else 0.0,
        "mean": float(sj.mean()) if sj.size else 0.0,
        "median": float(np.median(sj)) if sj.size else 0.0,
        "p01": float(np.percentile(sj, 1)) if sj.size else 0.0,
        "p05": float(np.percentile(sj, 5)) if sj.size else 0.0,
        "pct_above_0p2": float(np.mean(sj > 0.2) * 100) if sj.size else 0.0,
        "pct_above_0p4": float(np.mean(sj > 0.4) * 100) if sj.size else 0.0,
        "pct_above_0p6": float(np.mean(sj > 0.6) * 100) if sj.size else 0.0,
        "pct_above_0p7": float(np.mean(sj > 0.7) * 100) if sj.size else 0.0,
        "pct_above_0p9": float(np.mean(sj > 0.9) * 100) if sj.size else 0.0,
        "pct_above_0p95": float(np.mean(sj > 0.95) * 100) if sj.size else 0.0,
        "pct_above_0p96": float(np.mean(sj > 0.96) * 100) if sj.size else 0.0,
        "n_negative": n_negative,
        "histogram": jacobian_histogram(sj),
    }
    report["status"] = "FAIL" if n_negative > 0 else "PASS"
    logger.info("Scaled Jacobian: n=%d min=%.4f mean=%.4f median=%.4f "
                ">0.9=%.2f%% >0.95=%.2f%% >0.96=%.2f%% negative=%d",
                report["n_elements"], report["min"], report["mean"], report["median"],
                report["pct_above_0p9"], report["pct_above_0p95"],
                report["pct_above_0p96"], n_negative)
    logger.info("Scaled Jacobian histogram:\n%s", format_histogram(report["histogram"]))
    return report


def jacobian_histogram(sj: np.ndarray, bins: int = 20) -> dict:
    """Bin Scaled Jacobian values into a JSON-serializable histogram."""
    if not sj.size:
        return {"edges": [], "counts": [], "bin_width": 0.0}
    edges = np.linspace(min(0.0, float(sj.min())), 1.0, bins + 1)
    counts, edges = np.histogram(sj, bins=edges)
    return {
        "edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "bin_width": round(float(edges[1] - edges[0]), 4),
    }


def format_histogram(histogram: dict, width: int = 48) -> str:
    """Render a histogram as text, for the log and the console."""
    counts = histogram.get("counts") or []
    edges = histogram.get("edges") or []
    if not counts:
        return "  (no elements)"
    peak = max(counts) or 1
    total = sum(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        bar = "#" * int(round(width * count / peak))
        lines.append(f"  [{edges[i]:6.3f},{edges[i + 1]:6.3f})  {count:>9,}  "
                     f"{100.0 * count / total:5.2f}%  {bar}")
    return "\n".join(lines)


def write_quality_report(mesh: meshio.Mesh, out_dir: Path, run_id: str = "") -> dict:
    """Write a Scaled Jacobian histogram (PNG) and metrics (JSON) to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = assert_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sj = _pooled_scaled_jacobians(mesh)
    report = scaled_jacobian_report(mesh)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sj, bins=100, range=(min(-0.1, float(sj.min()) if sj.size else -0.1), 1.0),
            color="#4C72B0", edgecolor="none")
    ax.set_yscale("log")  # the tail is the interesting part and it is tiny
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1, label="0.0 inverted")
    ax.axvline(0.96, color="green", linestyle="--", linewidth=1,
               label=f"0.96 target (99%: {report['pct_above_0p96']:.1f}%)")
    ax.axvline(report["mean"], color="black", linewidth=1.5,
               label=f"mean {report['mean']:.4f}")
    ax.set_xlabel("Scaled Jacobian")
    ax.set_ylabel("Element count (log)")
    title = f"Scaled Jacobian distribution - {run_id}" if run_id else "Scaled Jacobian distribution"
    ax.set_title(f"{title}\nn={sj.size:,}  min={report['min']:.4f}  mean={report['mean']:.4f}  "
                 f">0.96 {report['pct_above_0p96']:.2f}%  negative={report['n_negative']}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    hist_path = out_dir / (f"jacobian_histogram_{run_id}.png" if run_id else "jacobian_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    report["histogram_path"] = str(hist_path)
    metrics_path = out_dir / (f"jacobian_metrics_{run_id}.json" if run_id else "jacobian_metrics.json")
    with open(assert_write_path(metrics_path), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Wrote quality report: %s, %s", hist_path, metrics_path)
    return report


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute the Dice Similarity Coefficient for two binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def hausdorff_distance_95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Compute the 95th-percentile Hausdorff Distance between two masks' surfaces."""
    from scipy.ndimage import distance_transform_edt, binary_erosion

    def surface_points(mask: np.ndarray) -> np.ndarray:
        eroded = binary_erosion(mask)
        border = mask & ~eroded
        return np.argwhere(border) * np.array(spacing)

    pred_pts = surface_points(pred_mask)
    gt_pts = surface_points(gt_mask)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return 0.0

    dt_gt = distance_transform_edt(~gt_mask, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_mask, sampling=spacing)
    d_pred_to_gt = dt_gt[tuple(np.argwhere(pred_mask & ~binary_erosion(pred_mask)).T)]
    d_gt_to_pred = dt_pred[tuple(np.argwhere(gt_mask & ~binary_erosion(gt_mask)).T)]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def atlas_compliance_report(
    generated_labels_path: Path, atlas_reference_path: Optional[Path], label_ids: dict[str, int]
) -> dict:
    """Register the generated segmentation to atlas space and score it."""
    if atlas_reference_path is None or not Path(atlas_reference_path).exists():
        raise FileNotFoundError(
            "Ground-truth atlas reference is unreachable; halting rather than "
            "silently skipping atlas validation. Checked: "
            f"{atlas_reference_path}"
        )
    assert_read_only_path(atlas_reference_path)

    import ants

    gen_img = ants.image_read(str(generated_labels_path))
    gt_img = ants.image_read(str(atlas_reference_path))
    reg = ants.registration(fixed=gt_img, moving=gen_img, type_of_transform="Affine")
    gen_in_gt_space = ants.apply_transforms(
        fixed=gt_img, moving=gen_img, transformlist=reg["fwdtransforms"], interpolator="nearestNeighbor"
    )
    gen_arr = gen_in_gt_space.numpy().astype(int)
    gt_arr = gt_img.numpy().astype(int)
    spacing = gt_img.spacing[:3]

    per_label = {}
    for name, label_id in label_ids.items():
        pred_mask = gen_arr == label_id
        gt_mask = gt_arr == label_id
        dsc = dice_coefficient(pred_mask, gt_mask)
        hd95 = hausdorff_distance_95(pred_mask, gt_mask, spacing)
        per_label[name] = {"DSC": dsc, "HD95_mm": hd95}

    dscs = [v["DSC"] for v in per_label.values()]
    hd95s = [v["HD95_mm"] for v in per_label.values()]
    report = {
        "mean_DSC": float(np.mean(dscs)),
        "min_DSC": float(np.min(dscs)),
        "mean_HD95_mm": float(np.mean(hd95s)),
        "per_label": per_label,
    }
    logger.info("Atlas compliance: mean_DSC=%.4f mean_HD95=%.2fmm", report["mean_DSC"], report["mean_HD95_mm"])
    return report


def structural_report(mesh: meshio.Mesh) -> dict:
    """Is the mesh a valid conforming hex complex at all?"""
    from engine.mesh.hex_mesh import conformity_report

    blocks = [b for b in mesh.cells if b.type == "hexahedron"]
    if not blocks:
        return {"status": "SKIP", "reason": "no hexahedral cell blocks"}

    hexes = np.concatenate([b.data for b in blocks], axis=0)
    report = conformity_report(mesh.points, hexes, on_lattice=False)
    report["status"] = "PASS" if report["conforming"] else "FAIL"
    if report["status"] == "FAIL":
        logger.warning("structural_report: mesh is NOT conforming -- %d unpaired boundary "
                       "edges, %d overshared faces, hanging nodes %s",
                       report["n_boundary_edges_unpaired"], report["n_faces_overshared"],
                       report["n_hanging_nodes"])
    return report


def membrane_containment_report(points: np.ndarray, affine: np.ndarray, labels: np.ndarray,
                                membrane_blocks: dict,
                                measure_dura_band: bool = False, conforming=()) -> dict:
    """SIGNED containment of every membrane node (conforming sheets are hex faces: exact)."""
    from scipy import ndimage

    from engine.mesh import membranes

    if not membrane_blocks:
        return {}
    labels = np.asarray(labels)
    brain = np.isin(labels, membranes._BRAIN)
    csf = np.isin(labels, membranes._INTRACRANIAL)
    csf = ndimage.binary_fill_holes(csf)

    spacing = np.array([float(np.linalg.norm(np.asarray(affine, float)[:3, a]))
                        for a in range(3)])
    d_brain = (fast_kernels.distance_transform_edt(~brain, sampling=spacing)
               - fast_kernels.distance_transform_edt(brain, sampling=spacing))
    d_csf = (fast_kernels.distance_transform_edt(csf, sampling=spacing)
             - fast_kernels.distance_transform_edt(~csf, sampling=spacing))

    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    band_mm = (dura_inner_table_band_mm(csf, labels == _SKULL_LABEL, spacing)
               if measure_dura_band else _DURA_INNER_TABLE_BAND_MM)
    rules = {
        "dura": (("csf", -1, band_mm), ("brain", +1, None)),
        "pia": (("csf", +1, None), ("brain", +1, None)),
        "falx": (("brain", +1, None),),
        "tentorium": (("brain", +1, None),),
    }
    fields = {"brain": d_brain, "csf": d_csf}

    out: dict[str, dict] = {}
    for name, quads in membrane_blocks.items():
        quads = np.asarray(quads)
        if quads.size == 0 or name not in rules:
            continue
        if name in conforming:
            out[name] = {"n_nodes": int(len(np.unique(quads))), "conforming": True,
                         "violations": 0, "passed": True}
            continue
        nodes = np.asarray(points, dtype=float)[np.unique(quads)]
        vox = nodes @ inv[:3, :3].T + inv[:3, 3]
        entry = {"n_nodes": int(len(nodes))}
        violations = 0
        worst = 0.0
        for field, sign, max_mm in rules[name]:
            d = ndimage.map_coordinates(fields[field], vox.T, order=1, mode="nearest")
            bad = (sign * d) <= 0.0
            if max_mm is not None:
                bad |= (sign * d) > max_mm
                entry[f"{field}_max_clearance_mm"] = round(float((sign * d).max()), 3)
            entry[f"{field}_violations"] = int(bad.sum())
            entry[f"{field}_min_clearance_mm"] = round(float((sign * d).min()), 3)
            violations += int(bad.sum())
            if bad.any():
                worst = max(worst, float(np.abs(d[bad]).max()))
        entry["violations"] = violations
        entry["worst_mm"] = round(worst, 3)
        entry["passed"] = violations == 0
        out[name] = entry
        if violations:
            logger.error("membrane_containment_report: %s has %d node(s) on the wrong side of "
                         "its own boundary, worst by %.2fmm", name, violations, worst)
    out["passed"] = all(v.get("passed", True) for v in out.values() if isinstance(v, dict))
    return out


def _tri_aabb_overlap(tri: np.ndarray, box_center: np.ndarray,
                      box_half: np.ndarray) -> np.ndarray:
    """Akenine-Moller triangle/box overlap, vectorised over N pairs."""
    v = np.asarray(tri, dtype=float) - np.asarray(box_center, dtype=float)[:, None, :]
    half = np.asarray(box_half, dtype=float)
    hit = np.ones(len(v), dtype=bool)

    hit &= np.all((v.min(axis=1) <= half) & (v.max(axis=1) >= -half), axis=1)

    edges = (v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2])

    normal = np.cross(edges[0], edges[1])
    dist = np.einsum("ij,ij->i", normal, v[:, 0])
    hit &= np.abs(dist) <= np.einsum("ij,ij->i", np.abs(normal), half)

    for edge in edges:
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            axis = np.zeros_like(edge)
            axis[:, b] = -edge[:, c]
            axis[:, c] = edge[:, b]
            proj = np.einsum("ijk,ik->ij", v, axis)
            reach = np.abs(axis[:, b]) * half[:, b] + np.abs(axis[:, c]) * half[:, c]
            hit &= ~((proj.min(axis=1) > reach) | (proj.max(axis=1) < -reach))
    return hit


_PAIR_MIN_CLEARANCE_MM = 1.0

_DURA_INNER_TABLE_BAND_MM = 2.0

_DURA_BAND_BONE_FRACTION = 1.0 / 3.0


def dura_inner_table_band_mm(intracranial: np.ndarray, skull: np.ndarray,
                             spacing: np.ndarray) -> float:
    """How far outside the CSF envelope a dura node may sit, in mm."""
    from scipy import ndimage

    spacing = np.asarray(spacing, dtype=float)
    one_voxel = float(np.max(spacing))
    skull = np.asarray(skull, dtype=bool)
    if not skull.any():
        return float(_DURA_INNER_TABLE_BAND_MM)
    depth = ndimage.distance_transform_edt(~np.asarray(intracranial, dtype=bool),
                                           sampling=spacing)
    bone_mm = 2.0 * float(np.median(depth[skull]))
    band = float(np.clip(bone_mm * _DURA_BAND_BONE_FRACTION,
                         one_voxel, _DURA_INNER_TABLE_BAND_MM))
    logger.info("dura band %.3fmm (bone measured %.2fmm, %.0f%% of it, "
                "ceiling %.1fmm, one voxel %.2fmm)", band, bone_mm,
                _DURA_BAND_BONE_FRACTION * 100, _DURA_INNER_TABLE_BAND_MM, one_voxel)
    return band


# pairs that touch by design: dura on the bone, the falx and tentorium are folds of the dura
# (TIED to it in the deck), attached along their outer margins to the inner table, and the
# falx rests on the tentorium; the carnivore's bony tentorium is the same sheet, split
_ADJACENT_PAIRS = frozenset({("dura", "skull"), ("scalp", "skull"), ("dura", "falx"),
                               ("dura", "tentorium"), ("falx", "tentorium"),
                               ("falx", "skull"), ("tentorium", "skull"),
                               ("tentorium_osseous", "skull"), ("tentorium_osseous", "tentorium"),
                               ("tentorium_osseous", "falx"), ("tentorium_osseous", "dura")})


def membrane_pair_clearance_report(points: np.ndarray, membrane_blocks: dict,
                                   min_clearance_mm: float = _PAIR_MIN_CLEARANCE_MM,
                                   conforming=()) -> dict:
    """How close any two membrane sheets come to each other.

    Two sheets that both sit on hex faces (`conforming`, or the skull/scalp mid-surfaces) are
    kept apart by the solid elements between them, so their gap is not graded."""
    from itertools import combinations

    from scipy.spatial import cKDTree

    names = [n for n, q in membrane_blocks.items() if np.asarray(q).size]
    if len(names) < 2:
        return {}
    points = np.asarray(points, dtype=float)
    nodes = {n: points[np.unique(np.asarray(membrane_blocks[n]).ravel())] for n in names}

    pairs: dict[str, dict] = {}
    trees: dict = {}                                  # one tree per sheet, queried on all cores
    for a, b in combinations(names, 2):
        if b not in trees:
            trees[b] = cKDTree(nodes[b])
        distance, _ = trees[b].query(nodes[a], workers=-1)
        adiacens = tuple(sorted((a, b))) in {tuple(sorted(p)) for p in _ADJACENT_PAIRS}
        fixed = set(conforming) | {"skull", "scalp"}
        adiacens = adiacens or (bool(conforming) and a in fixed and b in fixed)
        close = 0 if adiacens else int((distance < min_clearance_mm).sum())
        pairs[f"{a}|{b}"] = {
            "apposed": adiacens,
            "min_gap_mm": round(float(distance.min()), 3),
            "nodes_within_min_clearance": close,
            "n_nodes": int(len(distance)),
            "passed": bool(close == 0),
        }
    return {"pairs": pairs,
            "min_clearance_mm": float(min_clearance_mm),
            "passed": bool(all(v["passed"] for v in pairs.values()))}


_INTERSECTION_HOST_PARTS = ("csf", "skull", "scalp", "soft_tissue")
_INTERSECTION_MARGIN_MM = 0.25


def membrane_solid_intersection_report(points: np.ndarray, membrane_blocks: dict,
                                       solid_blocks: dict,
                                       *, chunk: int = 20_000,
                                       host_parts: tuple = _INTERSECTION_HOST_PARTS,
                                       margin_mm: float = _INTERSECTION_MARGIN_MM) -> dict:
    """Does any membrane quad pass THROUGH a BRAIN element? Hard gate at zero."""
    from scipy.spatial import cKDTree

    if not membrane_blocks or not solid_blocks:
        return {}
    points = np.asarray(points, dtype=float)

    hosts = {str(h).lower() for h in host_parts}
    centroids, boxes_half, owner, part_names = [], [], [], []
    skipped = []
    for name, block in solid_blocks.items():
        block = np.asarray(block)
        if block.size == 0:
            continue
        if str(name).lower() in hosts:
            skipped.append(str(name))
            continue
        corners = points[block]  # (n, 8, 3)
        low, high = corners.min(axis=1), corners.max(axis=1)
        centroids.append(0.5 * (low + high))
        boxes_half.append(np.maximum(0.5 * (high - low) - float(margin_mm), 0.0))
        owner.append(np.full(len(block), len(part_names)))
        part_names.append(name)
    if not centroids:
        out = {sheet: {"quads": int(len(np.asarray(q))), "intersecting_quads": 0,
                       "intersections_by_part": {}, "not_graded_parts": sorted(skipped),
                       "margin_mm": float(margin_mm), "passed": True}
               for sheet, q in membrane_blocks.items() if np.asarray(q).size}
        out["passed"] = True
        return out
    centroids = np.concatenate(centroids)
    boxes_half = np.concatenate(boxes_half)
    owner = np.concatenate(owner)
    tree = cKDTree(centroids)
    hex_reach = float(np.linalg.norm(boxes_half, axis=1).max())

    out: dict = {}
    for sheet, quads in membrane_blocks.items():
        quads = np.asarray(quads)
        if quads.size == 0:
            continue
        corners = points[quads]  # (m, 4, 3)
        tris = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]], axis=0)
        tri_center = tris.mean(axis=1)
        radius = np.linalg.norm(tris - tri_center[:, None, :], axis=2).max(axis=1)

        hits: dict[str, int] = {}
        bad_quads: set = set()
        example = None
        for start in range(0, len(tris), chunk):
            block = slice(start, start + chunk)
            near = tree.query_ball_point(tri_center[block], radius[block] + hex_reach,
                                         workers=-1)
            pair_tri, pair_hex = [], []
            for i, candidates in enumerate(near):
                if candidates:
                    pair_tri.append(np.full(len(candidates), start + i))
                    pair_hex.append(np.asarray(candidates, dtype=int))
            if not pair_tri:
                continue
            pair_tri = np.concatenate(pair_tri)
            pair_hex = np.concatenate(pair_hex)
            overlap = _tri_aabb_overlap(tris[pair_tri], centroids[pair_hex],
                                        boxes_half[pair_hex])
            for t, h in zip(pair_tri[overlap], pair_hex[overlap]):
                name = part_names[owner[h]]
                hits[name] = hits.get(name, 0) + 1
                bad_quads.add(int(t % len(quads)))
                if example is None:
                    example = {"quad": int(t % len(quads)), "part": name,
                               "centroid_mm": [round(float(c), 2) for c in centroids[h]]}

        entry = {
            "quads": int(len(quads)),
            "intersecting_quads": len(bad_quads),
            "intersections_by_part": dict(sorted(hits.items(), key=lambda kv: -kv[1])),
            "not_graded_parts": sorted(skipped),
            "margin_mm": float(margin_mm),
            "passed": not bad_quads,
        }
        if example is not None:
            entry["example"] = example
        out[sheet] = entry
        if bad_quads:
            top = list(entry["intersections_by_part"].items())[:3]
            logger.error("membrane_solid_intersection_report: %d of %d %s quads pass through "
                         "a solid element (%s)", len(bad_quads), len(quads), sheet,
                         ", ".join(f"{k} x{v}" for k, v in top))
    out["passed"] = all(v.get("passed", True) for v in out.values() if isinstance(v, dict))
    return out


_SHELL_HOST_PARTS = ("csf", "scalp", "soft_tissue")


def shell_integrity_report(point_set: np.ndarray, squares: dict, solida: dict,
                               *, trianguli: Optional[dict] = None,
                               host_part_names: tuple = _SHELL_HOST_PARTS,
                               conforming=()) -> dict:
    """Every shell part at once"""
    from engine.mesh import shell_repair

    trianguli = trianguli or {}
    topology_info = shell_repair.topology_report(squares, trianguli)

    # a conforming sheet is made of solid faces: it lies on them, it cannot pass through them
    ad_penetrationem = {n: np.asarray(q).reshape(-1, 4) for n, q in squares.items()
                        if np.asarray(q).size and n not in conforming}
    for entry_name, tris in trianguli.items():
        tris = np.asarray(tris).reshape(-1, 3)
        if not len(tris):
            continue
        falsa = np.concatenate([tris, tris[:, [2]]], axis=1)
        ad_penetrationem[entry_name] = (np.concatenate([ad_penetrationem[entry_name], falsa])
                                   if entry_name in ad_penetrationem else falsa)
    penetration_check = membrane_solid_intersection_report(
        point_set, ad_penetrationem, solida, host_parts=host_part_names)

    out: dict = {}
    for entry_name, topos in topology_info.items():
        pene = penetration_check.get(entry_name, {})
        passed = bool(topos["passed"] and pene.get("passed", True))
        out[entry_name] = {"topology": topos, "intersection": pene, "passed": passed}
        if not passed:
            logger.error("shell_integrity_report: %s failed -- %d boundary loop(s) "
                         "(expected %s), %d junction(s), %d non-manifold edge(s), "
                         "%d quad(s) through a solid", entry_name, topos["boundary_loops"],
                         topos["expected_boundary_loops"], topos["boundary_junctions"],
                         topos["non_manifold_edges"], pene.get("intersecting_quads", 0))
    out["not_graded_parts"] = sorted(str(h) for h in host_part_names)
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    out["intersectio_omnium"] = penetration_check
    return out


def falx_centering_report(points: np.ndarray, falx_quads: Optional[np.ndarray],
                          solid_blocks: dict, tolerance_mm: float = 1.0) -> dict:
    """Is the falx on the deck's own midline, with equal brain either side?"""
    if falx_quads is None or len(falx_quads) == 0:
        return {}
    points = np.asarray(points, dtype=float)
    x = points[np.unique(np.asarray(falx_quads).ravel())][:, 0]
    offset = float(np.mean(x))
    spread = float(np.max(np.abs(x - offset))) if len(x) else 0.0

    left = right = 0
    for name, block in (solid_blocks or {}).items():
        block = np.asarray(block)
        if block.size == 0 or name in ("skull", "skull_diploe", "scalp", "soft_tissue"):
            continue
        cx = points[block][..., 0].mean(axis=1)
        left += int((cx < 0).sum())
        right += int((cx > 0).sum())
    total = left + right
    balance = abs(left - right) / total if total else 0.0

    out = {
        "plane_offset_mm": round(offset, 3),
        "plane_spread_mm": round(spread, 3),
        "elements_left": left,
        "elements_right": right,
        "left_right_imbalance": round(balance, 4),
        "centred": bool(abs(offset) <= tolerance_mm and spread <= tolerance_mm),
        "balanced": bool(balance <= 0.05),
    }
    if not out["centred"]:
        logger.warning("falx_centering_report: falx sits %.2fmm off the deck midline with a "
                       "%.2fmm spread, against a %.2fmm tolerance",
                       offset, spread, tolerance_mm)
    if not out["balanced"]:
        logger.warning("falx_centering_report: %d elements left of the falx against %d right "
                       "(%.1f%% imbalance)", left, right, 100 * balance)
    return out


def part_overlap_report(
    points: np.ndarray, affine: np.ndarray, labels: np.ndarray,
    solid_blocks: dict, membrane_blocks: Optional[dict] = None,
    shell_blocks: Optional[dict] = None,
    min_clearance_mm: float = _PAIR_MIN_CLEARANCE_MM,
    host_parts: tuple = _INTERSECTION_HOST_PARTS,
    shell_in_solid: Optional[dict] = None,
    conforming=(),
) -> dict:
    """Do any two parts occupy the same space?"""
    from scipy import ndimage

    from engine.anatomy.anatomy_labels import BRAIN_TISSUE, REGION_LABELS

    shape = tuple(int(s) for s in labels.shape)
    inv = np.linalg.inv(affine)
    owner = np.full(int(np.prod(shape)), -1, dtype=np.int16)
    names: list[str] = []
    pairs: dict[str, int] = {}

    def voxels_of(conn) -> np.ndarray:
        corners = points[np.asarray(conn)] @ inv[:3, :3].T + inv[:3, 3]
        low, high = corners.min(axis=1), corners.max(axis=1)
        medium = 0.5 * (low + high)
        half_count = 0.375 * (high - low)
        lo = np.ceil(medium - half_count).astype(np.int64)
        hi = np.floor(medium + half_count).astype(np.int64)
        vacua = np.any(hi < lo, axis=1)
        if vacua.any():
            own_part = np.floor(medium[vacua] + 0.5).astype(np.int64)
            lo[vacua] = own_part
            hi[vacua] = own_part
        latitudo = np.maximum((hi - lo + 1).max(axis=0), 1)
        label_sets = np.stack(np.meshgrid(*[np.arange(int(s)) for s in latitudo],
                                      indexing="ij"), axis=-1).reshape(-1, 3)
        vox = (lo[:, None, :] + label_sets[None, :, :]).reshape(-1, 3)
        intra = np.repeat(hi, len(label_sets), axis=0)
        vox = vox[np.all(vox <= intra, axis=1)]
        vox = vox[np.all((vox >= 0) & (vox < np.array(shape)), axis=1)]
        return np.unique(np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), shape)) \
            if len(vox) else np.zeros(0, dtype=np.int64)

    for name, conn in (solid_blocks or {}).items():
        if not len(np.asarray(conn)):
            continue
        index = len(names)
        names.append(name)
        flat = voxels_of(conn)
        if not len(flat):
            continue
        held = owner[flat]
        clashed = held[held >= 0]
        for other in np.unique(clashed):
            key = "|".join(sorted((name, names[int(other)])))
            pairs[key] = pairs.get(key, 0) + int((clashed == other).sum())
        owner[flat[held < 0]] = index

    solid_values = sorted(REGION_LABELS[r] for r in BRAIN_TISSUE)
    interior = ndimage.binary_erosion(np.isin(labels, solid_values))
    in_tissue: dict[str, int] = {}
    for name, conn in (membrane_blocks or {}).items():
        conn = np.asarray(conn)
        if not len(conn):
            continue
        vox = np.floor(points[conn].mean(axis=1) @ inv[:3, :3].T + inv[:3, 3] + 0.5).astype(np.int64)
        vox = np.clip(vox, 0, np.array(shape) - 1)
        in_tissue[name] = int(interior[vox[:, 0], vox[:, 1], vox[:, 2]].sum())

    shells = {n: q for n, q in (shell_blocks or {}).items() if np.asarray(q).size}
    if shell_in_solid is None:
        in_solid = membrane_solid_intersection_report(
            points, shells, solid_blocks or {}, host_parts=host_parts) if shells else {}
    else:
        in_solid = shell_in_solid
    clearance = membrane_pair_clearance_report(
        points, shells, min_clearance_mm=min_clearance_mm,
        conforming=conforming) if len(shells) > 1 else {}

    total = int(sum(pairs.values()))
    embedded = int(sum(in_tissue.values()))
    passed = (total == 0 and embedded == 0
              and in_solid.get("passed", True) and clearance.get("passed", True))
    report = {"n_overlapping_voxels": total, "pairs": pairs,
              "membranes_in_tissue": in_tissue,
              "shell_in_solid": in_solid, "shell_clearance": clearance,
              "status": "PASS" if passed else "FAIL"}
    if total:
        logger.warning("part_overlap_report: %d voxels are claimed by two parts at once: %s",
                       total, pairs)
    if embedded:
        logger.warning("part_overlap_report: membrane elements sitting in solid tissue: %s",
                       in_tissue)
    return report


def cortical_fidelity_report(
    labels_path: Path, freesurfer_subject_dir: Optional[Path]
) -> dict:
    """How much cortical folding survived into the meshed volume."""
    if freesurfer_subject_dir is None:
        return {"status": "SKIP", "reason": "no freesurfer_subject_dir for ground truth"}

    import nibabel as nib
    import nibabel.freesurfer.io as fsio

    from engine.anatomy.anatomy_labels import cortical_surface_area

    reference = 0.0
    for hemi in ("lh", "rh"):
        surf = freesurfer_subject_dir / "surf" / f"{hemi}.pial.T1"
        if not surf.exists():
            surf = freesurfer_subject_dir / "surf" / f"{hemi}.pial"
        if not surf.exists():
            continue
        verts, faces = fsio.read_geometry(str(surf))
        tri = verts[faces]
        reference += 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 100.0
    if reference <= 0:
        return {"status": "SKIP", "reason": "no pial surfaces found"}

    img = nib.load(str(labels_path))
    meshed = cortical_surface_area(np.asarray(img.dataobj).astype(int), img.affine)
    retention = meshed / reference * 100.0
    report = {
        "meshed_area_cm2": round(meshed, 1),
        "reference_area_cm2": round(reference, 1),
        "retention_pct": round(retention, 1),
        "status": "PASS" if retention >= RETENTION_MIN_PCT else "FAIL",
    }
    if report["status"] == "FAIL":
        logger.warning("cortical_fidelity_report: only %.0f%% of the reference pial area is "
                       "present in the mesh -- folds are being smoothed shut", retention)
    return report


def sulcal_depth_report(labels_path: Path,
                        freesurfer_subject_dir: Optional[Path] = None) -> dict:  # noqa: ARG001
    """How deep the CSF actually reaches into the folds, in mm."""
    import nibabel as nib
    from scipy import ndimage

    from engine.anatomy.anatomy_labels import BRAIN_TISSUE, REGION_LABELS

    img = nib.load(str(labels_path))
    labels = np.asarray(img.dataobj).astype(int)
    spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
    vox = float(np.mean(spacing))

    brain = np.isin(labels, [REGION_LABELS[r] for r in BRAIN_TISSUE])
    if not brain.any():
        return {"status": "SKIP", "reason": "no brain tissue in the labels"}
    envelope = ndimage.binary_closing(brain, iterations=max(1, int(round(15.0 / vox))))
    sulcal = (labels == REGION_LABELS["csf"]) & envelope
    if not sulcal.any():
        return {"status": "SKIP", "reason": "no CSF inside the gyral envelope"}

    depth = fast_kernels.distance_transform_edt(envelope, sampling=spacing)[sulcal]
    report = {
        "median_mm": round(float(np.median(depth)), 2),
        "p90_mm": round(float(np.percentile(depth, 90)), 2),
        "max_mm": round(float(depth.max()), 2),
        "sulcal_csf_cm3": round(float(sulcal.sum()) * float(np.prod(spacing)) / 1000.0, 1),
        "status": "REPORT",
    }

    logger.info("CSF penetration depth: median %.1fmm p90 %.1fmm max %.1fmm over "
                "%.0f cm^3 (includes the interhemispheric fissure)",
                report["median_mm"], report["p90_mm"], report["max_mm"],
                report["sulcal_csf_cm3"])
    return report


def validate(
    run_id: str, mesh_path: Path, generated_labels_path: Path, atlas_reference_path: Optional[Path],
    freesurfer_subject_dir: Optional[Path] = None,
) -> dict:
    """Run the full validation harness for one mesh."""
    mesh = meshio.read(str(mesh_path))
    jacobian = scaled_jacobian_report(mesh)
    structural = structural_report(mesh)

    if jacobian["n_negative"] > 0 or structural["status"] == "FAIL":
        reason = ("n_negative=%d" % jacobian["n_negative"] if jacobian["n_negative"] > 0
                  else "non-conforming mesh")
        logger.warning("run_id=%s hard-failed: %s", run_id, reason)
        return {
            "run_id": run_id,
            "jacobian": jacobian,
            "structural": structural,
            "cortical": None,
            "atlas": None,
            "overall_status": "FAIL",
        }

    cortical = cortical_fidelity_report(generated_labels_path, freesurfer_subject_dir)
    try:
        sulcal = sulcal_depth_report(generated_labels_path, freesurfer_subject_dir)
    except Exception:  # reported, never a reason to fail a mesh
        logger.exception("sulcal_depth_report failed")
        sulcal = {"status": "SKIP", "reason": "sulcal depth measurement failed"}
    atlas = atlas_compliance_report(generated_labels_path, atlas_reference_path, LABEL_IDS)

    if jacobian["mean"] < MEAN_SJ_TARGET:
        logger.warning("run_id=%s: mean Scaled Jacobian %.4f is below the %.2f target "
                       "(min %.4f, median %.4f, %.2f%% above 0.9). Raise "
                       "MeshingConfig.boundary_snap_quality_floor, or accept a looser "
                       "surface fit -- the two trade directly.",
                       run_id, jacobian["mean"], MEAN_SJ_TARGET, jacobian["min"],
                       jacobian["median"], jacobian["pct_above_0p9"])

    soft_pass = (
        jacobian["pct_above_0p96"] >= JACOBIAN_PCT_0P96_TARGET
        and jacobian["pct_above_0p4"] >= JACOBIAN_PCT_0P4_TARGET
        and jacobian["pct_above_0p2"] >= JACOBIAN_PCT_0P2_TARGET
        and jacobian["mean"] >= MEAN_SJ_TARGET
        and atlas["mean_DSC"] >= DSC_TARGET
        and cortical["status"] != "FAIL"
    )
    result = {
        "run_id": run_id,
        "jacobian": jacobian,
        "structural": structural,
        "cortical": cortical,
        "sulcal_depth": sulcal,
        "atlas": atlas,
        "overall_status": "PASS" if soft_pass else "FAIL",
    }
    return result


LABEL_IDS = {
    "gray_matter": 1, "white_matter": 2, "brainstem": 3, "cerebellum": 4,
    "deep_nuclei": 5, "csf": 6, "skull": 7, "dura": 8, "falx": 9,
    "tentorium": 10, "pia": 11,
}


def main() -> None:
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="AnatoMesh validation harness")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--atlas-reference", type=Path, default=None)
    args = parser.parse_args()

    result = validate(args.run_id, args.mesh, args.labels, args.atlas_reference)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
