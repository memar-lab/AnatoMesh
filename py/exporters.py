"""Exports of a finished run: the deck, surfaces, volume mesh formats, label volume, tables, report."""
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np

Mesh = tuple[np.ndarray, dict, dict, dict, dict]

_HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
_STL = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_CHUNK = 200000


def _slug(text: str) -> str:
    text = re.sub(r"\.(nii(\.gz)?|mgz|mgh|nrrd|dcm)(?=\s|$)", "", str(text), flags=re.I)
    return re.sub(r"[^\w.\-]+", "_", text).strip("_")


def _result(job_dir: Path) -> dict:
    path = job_dir / "viewer" / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load(job_dir: Path) -> Mesh:
    with np.load(job_dir / "mesh_parts.npz") as z:
        points = z["points"]
        solids = {k.split(":", 1)[1]: z[k] for k in z.files if k.startswith("hex:")}
        shells = {k.split(":", 1)[1]: z[k] for k in z.files if k.startswith("quad:")}
        shells.update({k.split(":", 1)[1]: z[k] for k in z.files if k.startswith("tri:")})
        raw = dict(zip(z["part_names"].tolist(), z["part_ids"].ravel().tolist()))
    legend = {e["id"]: e["name"] for e in _result(job_dir).get("legend", [])}
    nxt = max([v for v in raw.values() if v >= 0], default=0) + 1
    pid: dict[str, int] = {}
    titles: dict[str, str] = {}
    for name in [*solids, *shells]:
        v = raw.get(name, -1)
        titles[name] = legend.get(name) or name.replace("_", " ").title()
        if v < 0:
            v, nxt = nxt, nxt + 1
        pid[name] = int(v)
    return points, solids, shells, pid, titles


def _scaled_jacobian(coords: np.ndarray) -> np.ndarray:
    # the quality scorecard's metric (hexes or quads), so exported values match the report and the viewer
    from engine.export import scorecard

    metrics = scorecard._solid_metrics if coords.shape[1] == 8 else scorecard._shell_metrics
    return scorecard._in_chunks(metrics, coords)["sj"]


def _boundary_quads(hexes: np.ndarray) -> np.ndarray:
    if len(hexes) == 0:
        return np.empty((0, 4), dtype=np.int64)
    faces = hexes[:, _HEX_FACES].reshape(-1, 4)
    _, inv, counts = np.unique(np.sort(faces, axis=1), axis=0,
                               return_inverse=True, return_counts=True)
    return faces[counts[inv.ravel()] == 1]


def _triangles(quads: np.ndarray) -> np.ndarray:
    if quads.shape[1] == 3:
        return quads
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def _surfaces(solids: dict, shells: dict) -> list[tuple[str, np.ndarray]]:
    return [(n, _boundary_quads(h)) for n, h in solids.items()] + list(shells.items())


def _cells(solids: dict, shells: dict) -> list[tuple[str, str, np.ndarray]]:
    cells = [("hexahedron", n, a) for n, a in solids.items()]
    cells += [("quad" if a.shape[1] == 4 else "triangle", n, a) for n, a in shells.items()]
    return cells


def _compact(points: np.ndarray, blocks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    if not blocks:
        return points[:0], []
    used = np.unique(np.concatenate([b.ravel() for b in blocks]))
    return points[used], [np.searchsorted(used, b) for b in blocks]


def _write_lines(path: Path, lines: list[str]) -> None:
    # explicit newline so Windows does not write \r\n
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_k(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    out = folder / f"{stem}.k"
    if parts is None:
        deck = None
        summary = job_dir / "summary.json"
        if summary.exists():
            deck = json.loads(summary.read_text(encoding="utf-8")).get("deck")
        if not deck or not Path(deck).is_file():
            raise FileNotFoundError("the run has no deck")
        shutil.copy2(deck, out)
        return [out]
    points, solids, shells, pid, titles = mesh
    pts, blocks = _compact(points, [*solids.values(), *shells.values()])
    lines = ["*KEYWORD", "$ geometry of the selected parts; sections are in the full deck, material id = part id"]
    for n in [*solids, *shells]:
        lines += ["*PART", titles[n], "%10d%10d%10d" % (pid[n], 1, 1)]
    eid = 1
    solid_rows, shell_rows = [], []
    for n, b in zip(solids, blocks[:len(solids)]):
        for r in (b + 1).tolist():
            solid_rows.append(("%8d" * 10) % (eid, pid[n], *r))
            eid += 1
    for n, b in zip(shells, blocks[len(solids):]):
        if b.shape[1] == 3:
            b = np.column_stack([b, b[:, 2]])
        for r in (b + 1).tolist():
            shell_rows.append(("%8d" * 6) % (eid, pid[n], *r))
            eid += 1
    if solid_rows:
        lines += ["*ELEMENT_SOLID", *solid_rows]
    if shell_rows:
        lines += ["*ELEMENT_SHELL", *shell_rows]
    lines.append("*NODE")
    lines += ["%8d%16.6f%16.6f%16.6f%8d%8d" % (i + 1, x, y, z, 0, 0)
              for i, (x, y, z) in enumerate(pts.tolist())]
    lines.append("*END")
    _write_lines(out, lines)
    return [out]


def _write_stl(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    points, solids, shells, _, _ = mesh
    written = []
    for name, faces in _surfaces(solids, shells):
        tri = _triangles(faces)
        v = points[tri]
        nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
        rec = np.zeros(len(tri), dtype=_STL)
        rec["n"] = nrm
        rec["v"] = v
        path = folder / f"{stem}_{_slug(name)}.stl"
        with open(path, "wb") as fh:
            fh.write(b"AnatoMesh ".ljust(80, b" "))
            fh.write(struct.pack("<I", len(tri)))
            fh.write(rec.tobytes())
        written.append(path)
    return written


def _write_obj(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    points, solids, shells, _, _ = mesh
    surfaces = _surfaces(solids, shells)
    pts, blocks = _compact(points, [f for _, f in surfaces])
    lines = ["# AnatoMesh surfaces, coordinates in mm"]
    lines += ["v %.6f %.6f %.6f" % tuple(p) for p in pts.tolist()]
    for (name, _), f in zip(surfaces, blocks):
        lines.append(f"g {_slug(name)}")
        fmt = "f" + " %d" * f.shape[1]
        lines += [fmt % tuple(r) for r in (f + 1).tolist()]
    path = folder / f"{stem}.obj"
    _write_lines(path, lines)
    return [path]


def _write_vtk(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    import meshio
    points, solids, shells, pid, _ = mesh
    cells = _cells(solids, shells)
    sj = [_scaled_jacobian(points[a]) if t in ("hexahedron", "quad") else np.ones(len(a))
          for t, _, a in cells]
    pts, blocks = _compact(points, [a for _, _, a in cells])
    m = meshio.Mesh(
        pts, [(t, b) for (t, _, _), b in zip(cells, blocks)],
        cell_data={"part_id": [np.full(len(a), pid[n], dtype=int) for _, n, a in cells],
                   "scaled_jacobian": sj})
    path = folder / f"{stem}.vtu"
    meshio.write(str(path), m)
    return [path]


def _write_inp(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    import meshio
    points, solids, shells, _, _ = mesh
    cells = _cells(solids, shells)
    pts, blocks = _compact(points, [a for _, _, a in cells])
    sets = {}
    for i, (_, name, _) in enumerate(cells):
        sets[_slug(name)] = [np.arange(len(blocks[j])) if i == j else np.array([], dtype=int)
                             for j in range(len(cells))]
    m = meshio.Mesh(pts, [(t, b) for (t, _, _), b in zip(cells, blocks)], cell_sets=sets)
    path = folder / f"{stem}.inp"
    meshio.write(str(path), m, file_format="abaqus")
    return [path]


def _write_nifti(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    import nibabel as nib
    _, solids, shells, pid, _ = mesh
    src = job_dir / "viewer" / "parts.nii.gz"
    out = folder / f"{stem}_labels.nii.gz"
    legend = _result(job_dir).get("legend", [])
    if parts is None:
        shutil.copy2(src, out)
        keep = legend
    else:
        chosen = {*solids, *shells}
        keep = [e for e in legend if e["id"] in chosen]
        img = nib.load(str(src))
        data = np.asarray(img.dataobj)
        data = np.where(np.isin(data, [e["code"] for e in keep]), data, 0).astype(data.dtype)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(out))
    table = folder / f"{stem}_labels.csv"
    _write_lines(table, ["code,part,name"] + [f"{e['code']},{e['id']},{e['name']}" for e in keep])
    return [out, table]


def _write_csv(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    points, solids, shells, pid, titles = mesh
    elements = folder / f"{stem}_elements.csv"
    fmt = ["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f"]
    rows = ["part,name,elements,sj_min,sj_mean,share_at_or_above_0.5"]
    eid = 1
    with open(elements, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("element,part,x,y,z,scaled_jacobian\n")
        for name, h in [*solids.items(), *shells.items()]:
            sj = np.empty(len(h))
            for s in range(0, len(h), _CHUNK):
                c = points[h[s:s + _CHUNK]]
                sjc = _scaled_jacobian(c)
                sj[s:s + len(c)] = sjc
                m = len(c)
                arr = np.column_stack([np.arange(eid, eid + m), np.full(m, pid[name]),
                                       c.mean(axis=1), sjc])
                np.savetxt(fh, arr, fmt=fmt, delimiter=",")
                eid += m
            if len(sj):
                rows.append("%d,%s,%d,%.6f,%.6f,%.6f" % (
                    pid[name], titles[name], len(sj), sj.min(), sj.mean(), (sj >= 0.5).mean()))
            else:
                rows.append("%d,%s,0,0.000000,0.000000,0.000000" % (pid[name], titles[name]))
    table = folder / f"{stem}_parts.csv"
    _write_lines(table, rows)
    return [elements, table]


def _write_pdf(job_dir: Path, stem: str, folder: Path, mesh: Mesh, parts: list[str] | None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    text = (job_dir / "report.txt").read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines() or [""]
    graphs = _result(job_dir).get("graphs", {})
    titles = [("scaled_jacobian", "Scaled Jacobian"), ("element_volume", "Element volume (mm^3)"),
              ("edge_ratio", "Edge ratio"), ("diagonal_ratio", "Diagonal ratio")]
    path = folder / f"{stem}_report.pdf"
    with PdfPages(str(path)) as pdf:
        for s in range(0, len(lines), 88):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.97, "\n".join(lines[s:s + 88]), family="monospace",
                     fontsize=7.5, va="top", ha="left")
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        for ax, (key, title) in zip(axes.ravel(), titles):
            edges = np.asarray(graphs[key]["edges"], dtype=float)
            counts = np.asarray(graphs[key]["counts"], dtype=float)
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
            ax.set_title(title)
            ax.set_ylabel("Elements")
            ax.grid(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return [path]


_WRITERS = {
    ".k": _write_k, "STL": _write_stl, "OBJ": _write_obj, ".inp": _write_inp,
    "VTK": _write_vtk, "NIfTI": _write_nifti, "CSV": _write_csv, "PDF": _write_pdf,
}


def export(job_dir: Path, formats: list[str], folder: Path, parts: list[str] | None = None) -> list[Path]:
    if not formats:
        raise ValueError("choose at least one format")
    for name in formats:
        if name not in _WRITERS:
            raise ValueError(f"unknown format {name!r}")
    job_dir, folder = Path(job_dir), Path(folder)
    points, solids, shells, pid, titles = _load(job_dir)
    if parts is not None:
        for name in parts:
            if name not in pid:
                raise ValueError(f"part {name!r} is not in this mesh")
        solids = {n: a for n, a in solids.items() if n in parts}
        shells = {n: a for n, a in shells.items() if n in parts}
    folder.mkdir(parents=True, exist_ok=True)

    job_name = None
    job_file = job_dir / "job.json"
    if job_file.exists():
        job_name = json.loads(job_file.read_text(encoding="utf-8")).get("name")
    stem = _slug(job_name or job_dir.name)
    if parts is not None:
        stem += f"_{_slug(parts[0])}" if len(parts) == 1 else "_parts"

    mesh: Mesh = (points, solids, shells, pid, titles)
    written: list[Path] = []
    for name in formats:
        written += _WRITERS[name](job_dir, stem, folder, mesh, parts)
    return written
