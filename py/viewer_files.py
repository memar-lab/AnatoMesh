"""Files the viewer reads: compact part-label volumes with a legend, the image, quality, surfaces.

Every volume is written in RAS voxel order. Surfaces are in the viewer's frame: continuous voxel
index times spacing, with a voxel centre at (i + 0.5) * spacing, the same as the slice planes.
"""
from __future__ import annotations

import colorsys
import json
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

import parts as parts_mod
from engine.anatomy import parcellation, substructures

TISSUE_CODE = {1: "gray_matter", 2: "white_matter", 3: "brainstem", 4: "cerebellum", 5: "deep_nuclei",
               6: "csf", 7: "skull", 8: "scalp"}
BASE_RGB = {
    "gray_matter": (232, 160, 192), "white_matter": (207, 217, 230), "brainstem": (127, 180, 255),
    "cerebellum": (157, 140, 255), "deep_nuclei": (255, 207, 107), "csf": (62, 224, 255),
    "skull": (231, 226, 208), "scalp": (212, 169, 138), "brain": (232, 160, 192), "dura": (140, 207, 154),
    "pia": (240, 166, 166), "falx": (53, 240, 184), "tentorium": (47, 208, 160),
    "venous_sinuses": (70, 90, 200), "arteries": (220, 60, 60), "axons": (255, 122, 217),
}
HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [3, 7, 6, 2], [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5]])


def _canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def _on_grid(img: nib.Nifti1Image, ref: nib.Nifti1Image, order: int) -> np.ndarray:
    """img resampled onto ref's voxels (nearest for labels, linear for images)."""
    data = np.asanyarray(img.dataobj)
    if data.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3):
        return np.asarray(data)
    m = np.linalg.inv(img.affine) @ ref.affine
    return ndimage.affine_transform(np.asarray(data, np.float32), m[:3, :3], offset=m[:3, 3],
                                    output_shape=ref.shape[:3], order=order, mode="constant", cval=0)


def _colour(i: int, base: tuple | None) -> list[int]:
    if base:
        return list(base)
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.62, 0.55)
    return [int(r * 255), int(g * 255), int(b * 255)]


def _parents(tree: list[dict]) -> dict[str, str | None]:
    """Each selectable part's nearest selectable ancestor (where it goes when not selected)."""
    out: dict[str, str | None] = {}

    def walk(nodes, up):
        for n in nodes:
            group = parts_mod._is_group(n)
            if not group:
                out[n["id"]] = up
            walk(n.get("children", []), up if group else n["id"])
    walk(tree, None)
    for side in ("lh", "rh"):           # cortical parcels sit inside their hemisphere when it is a part
        hemi = f"gray_matter_{side}"
        if hemi in out:
            for pid in out:
                if pid.startswith(f"{side}_"):
                    out[pid] = hemi
    for piece in ("cc_anterior", "cc_mid_anterior", "cc_central", "cc_mid_posterior", "cc_posterior"):
        if piece in out and "corpus_callosum" in out:
            out[piece] = "corpus_callosum"
    for tissue, _ in parts_mod.TISSUES:
        if out.get(tissue) is None and tissue in out:
            out[tissue] = "brain"
    return out


def legend(atlas: str, extra: list[str] = ()) -> list[dict]:
    tree = parts_mod.part_tree(atlas)
    names = {}

    def walk(nodes):
        for n in nodes:
            names[n["id"]] = n["name"]
            walk(n.get("children", []))
    walk(tree)
    names.setdefault("brain", "Brain")
    up = _parents(tree)
    ids = parts_mod.leaf_ids(tree) + [i for i in ("brain",) + tuple(extra) if i not in parts_mod.leaf_ids(tree)]
    rows = []
    for code, pid in enumerate(ids, start=1):
        rows.append({"code": code, "id": pid, "name": names.get(pid, pid.replace("_", " ").title()),
                     "parent": up.get(pid), "rgb": _colour(code, BASE_RGB.get(pid))})
    if len(rows) > 255:
        raise ValueError("more than 255 parts")
    return rows


def _paint_atlas(job_dir: Path, atlas: str, seg: np.ndarray, ref: nib.Nifti1Image, code_of: dict) -> np.ndarray:
    """Tissue codes, then structures, then cortical parcels inside gray matter."""
    out = np.zeros(seg.shape, np.uint8)
    for value, pid in TISSUE_CODE.items():
        target = "brain" if atlas == "whole" and value <= 5 else pid
        if target in code_of:
            out[seg == value] = code_of[target]
    if atlas == "whole":
        return out
    vols = {}
    for key, name in (("aseg", substructures.ASEG_FILE), ("brainstem", substructures.BRAINSTEM_FILE)):
        if (job_dir / name).exists():
            vols[key] = np.rint(_on_grid(_canonical(job_dir / name), ref, 0)).astype(np.int32)
    order = sorted(substructures.STRUCTURES.items(), key=lambda kv: (kv[0].startswith("cc_"), not kv[0].endswith(("_lh", "_rh"))))
    for name, s in order:                  # hemispheres first, the corpus callosum pieces last
        vol = vols.get(s.volume)
        if vol is not None and name in code_of:
            out[np.isin(vol, list(s.ids)) & (seg > 0)] = code_of[name]
    if atlas in parts_mod.SCHEME:
        found = parcellation.find_parcellation(job_dir, "region", parts_mod.SCHEME[atlas])
        if found:
            vol = np.rint(_on_grid(_canonical(found[0]), ref, 0)).astype(np.int32)
            for label_id, pname in parcellation.parcel_names(vol, found[1], "region").items():
                if pname in code_of:
                    out[(vol == label_id) & (seg == 1)] = code_of[pname]
    return out


def _image(job_dir: Path, ref: nib.Nifti1Image, fallback: np.ndarray) -> np.ndarray:
    for name in ("t1_head.nii.gz", "t1_n4.nii.gz", "t1_ras.nii.gz", "t1_brain.nii.gz"):
        if (job_dir / name).exists():
            return _on_grid(_canonical(job_dir / name), ref, 1).astype(np.float32)
    return fallback.astype(np.float32)


def write_preview(job_dir: Path, atlas: str) -> dict:
    out_dir = job_dir / "viewer"
    out_dir.mkdir(exist_ok=True)
    ref = _canonical(job_dir / "segmentation.nii.gz")
    seg = np.rint(np.asanyarray(ref.dataobj)).astype(np.int16)
    rows = legend(atlas)
    code_of = {r["id"]: r["code"] for r in rows}
    labels = _paint_atlas(job_dir, atlas, seg, ref, code_of)
    spacing = np.abs(np.diag(ref.affine)[:3])
    nib.save(nib.Nifti1Image(labels, ref.affine), str(out_dir / "labels.nii.gz"))
    nib.save(nib.Nifti1Image(_image(job_dir, ref, seg), ref.affine), str(out_dir / "image.nii.gz"))
    counts = np.bincount(labels.ravel(), minlength=len(rows) + 1)
    for r in rows:
        r["voxels"] = int(counts[r["code"]])
    info = {"legend": rows, "voxel_mm": spacing.tolist(), "shape": list(labels.shape),
            "voxel_volume_mm3": float(np.prod(spacing))}
    (out_dir / "legend.json").write_text(json.dumps(info), encoding="utf-8")
    return info


# ----------------------------------------------------------------------------- result

def _boundary_faces(hexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces of a hex block that belong to one hex only: (quads, owning hex row)."""
    quads = hexes[:, HEX_FACES].reshape(-1, 4)
    owner = np.repeat(np.arange(len(hexes)), 6)
    key = np.sort(quads, axis=1)
    _, inverse, count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = count[inverse.ravel()] == 1
    return quads[once], owner[once]


def _write_surface(path: Path, verts: np.ndarray, quads: np.ndarray, code: np.ndarray, quality: np.ndarray) -> None:
    """AMS1 | nV nT | float32 xyz | uint32 tri | uint8 code | float32 quality (per triangle)."""
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    code2 = np.concatenate([code, code]).astype(np.uint8)
    q2 = np.concatenate([quality, quality]).astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"AMS1" + struct.pack("<II", len(verts), len(tris)))
        f.write(np.ascontiguousarray(verts, np.float32).tobytes())
        f.write(np.ascontiguousarray(tris, np.uint32).tobytes())
        f.write(code2.tobytes())
        f.write(q2.tobytes())


def _histogram(values: np.ndarray, lo: float, hi: float, bins: int = 60) -> dict:
    v = values[np.isfinite(values)]
    counts, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    return {"edges": edges.round(6).tolist(), "counts": counts.tolist(),
            "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None,
            "mean": float(v.mean()) if len(v) else None}


def write_result(job_dir: Path, atlas: str) -> dict:
    from engine.export import scorecard
    from engine.mesh import voxel_hex
    from engine.pipeline import label_volume

    def part_row(name: str, code: int, kind: str, jac: np.ndarray) -> dict:
        finite = jac[np.isfinite(jac)]
        return {"id": name, "code": code, "kind": kind, "elements": int(len(jac)),
                "sj_min": float(finite.min()) if len(finite) else None,
                "sj_mean": float(finite.mean()) if len(finite) else None,
                # 0.01-wide bins over -1..1, for the share above any threshold
                "sj_hist": np.histogram(np.clip(finite, -1, 1), bins=200, range=(-1, 1))[0].tolist(),
                "sj": np.round(finite, 4)}

    out_dir = job_dir / "viewer"
    out_dir.mkdir(exist_ok=True)
    npz = np.load(str(job_dir / "mesh_parts.npz"), allow_pickle=False)
    points = npz["points"].astype(np.float64)
    hex_parts = {k[4:]: npz[k] for k in npz.files if k.startswith("hex:")}
    quad_parts = {k[5:]: npz[k] for k in npz.files if k.startswith("quad:")}
    head = _canonical(job_dir / "head_labels.nii.gz")
    A = head.affine
    spacing = np.abs(np.diag(A)[:3])
    inv = np.linalg.inv(A)

    def to_idx(xyz: np.ndarray) -> np.ndarray:
        return xyz @ inv[:3, :3].T + inv[:3, 3]

    def view(idx: np.ndarray) -> np.ndarray:
        return (idx + 0.5) * spacing

    extra = [n for n in list(hex_parts) + list(quad_parts) if n not in {r["id"] for r in legend(atlas)}]
    rows = legend(atlas, extra)
    code_of = {r["id"]: r["code"] for r in rows}
    shape = head.shape[:3]
    labels = np.zeros(shape, np.uint8)
    quality = np.zeros(shape, np.float32)
    volume_map = np.zeros(shape, np.float32)
    edge_map = np.zeros(shape, np.float32)
    cell_mm3 = float(np.prod(spacing))
    sj_all, vol_all, edge_all, diag_all = [], [], [], []
    per_part, after_q, after_c, after_o, before_quads = [], [], [], [], []
    snapped = points.copy()
    for name, hexes in hex_parts.items():
        coords = points[hexes]
        _, edge, diag = voxel_hex.compute_metrics(coords)
        jac = scorecard._in_chunks(scorecard._solid_metrics, coords)["sj"]    # the scorecard's numbers
        vol =label_volume.hex_volumes(coords)
        sj_all.append(jac); vol_all.append(vol); edge_all.append(edge); diag_all.append(diag)
        centre = np.rint(to_idx(coords.mean(axis=1))).astype(int)
        ok = np.all((centre >= 0) & (centre < shape), axis=1)
        code = code_of[name]
        labels[tuple(centre[ok].T)] = code
        quality[tuple(centre[ok].T)] = np.nan_to_num(jac[ok], nan=0.0)
        # element volume relative to the undeformed cube; edge ratio as stored (>= 1)
        volume_map[tuple(centre[ok].T)] = vol[ok] / cell_mm3
        edge_map[tuple(centre[ok].T)] = edge[ok]
        # voxel (before smoothing) corner of each node: its hex centre +-0.5 voxel per axis
        corner = to_idx(coords)
        snapped_idx = centre[:, None, :] + np.where(corner > centre[:, None, :], 0.5, -0.5)
        snapped[hexes.ravel()] = (snapped_idx.reshape(-1, 3) @ A[:3, :3].T + A[:3, 3])
        quads, owner = _boundary_faces(hexes)
        after_q.append(quads); after_o.append(np.nan_to_num(jac[owner], nan=0.0)); after_c.append(np.full(len(quads), code))
        before_quads.append(quads)
        per_part.append(part_row(name, code, "solid", jac))
    for name, quads in quad_parts.items():
        jac = scorecard._in_chunks(scorecard._shell_metrics, points[quads])["sj"]
        sj_all.append(jac)
        after_q.append(quads); after_o.append(np.nan_to_num(jac, nan=0.0)); after_c.append(np.full(len(quads), code_of[name]))
        per_part.append(part_row(name, code_of[name], "shell", jac))

    used = np.unique(np.concatenate([q.ravel() for q in after_q]))
    remap = np.full(len(points), -1, np.int64)
    remap[used] = np.arange(len(used))
    after_quads = remap[np.concatenate(after_q)]
    _write_surface(out_dir / "surface_after.bin", view(to_idx(points[used])), after_quads,
                   np.concatenate(after_c), np.concatenate(after_o))
    n_solid = sum(len(q) for q in before_quads)
    _write_surface(out_dir / "surface_before.bin", view(to_idx(snapped[used])), after_quads[:n_solid],
                   np.concatenate(after_c)[:n_solid], np.concatenate(after_o)[:n_solid])

    nib.save(nib.Nifti1Image(labels, A), str(out_dir / "parts.nii.gz"))
    nib.save(nib.Nifti1Image(quality, A), str(out_dir / "quality.nii.gz"))
    nib.save(nib.Nifti1Image(volume_map, A), str(out_dir / "element_volume.nii.gz"))
    nib.save(nib.Nifti1Image(edge_map, A), str(out_dir / "edge_ratio.nii.gz"))
    nib.save(nib.Nifti1Image(_image(job_dir, head, labels), A), str(out_dir / "result_image.nii.gz"))
    sj = np.concatenate(sj_all)
    graphs = {"scaled_jacobian": _histogram(sj, -1.0, 1.0), "element_volume": _histogram(np.concatenate(vol_all), 0.0, float(np.prod(spacing)) * 2),
              "edge_ratio": _histogram(np.concatenate(edge_all), 1.0, 4.0), "diagonal_ratio": _histogram(np.concatenate(diag_all), 0.0, 1.0)}
    table = []
    for p in per_part:
        table.append({k: v for k, v in p.items() if k != "sj"})
    (out_dir / "result.json").write_text(json.dumps(
        {"legend": rows, "parts": table, "graphs": graphs, "voxel_mm": spacing.tolist(), "shape": list(shape),
         "shells": sorted(quad_parts)}), encoding="utf-8")
    np.savez_compressed(str(out_dir / "part_sj.npz"), **{p["id"]: p["sj"] for p in per_part})
    return {"parts": table}
