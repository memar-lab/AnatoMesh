"""Grade one or more decks, part by part, against a published head-model quality bar.

Graded (Li, Zhou and Kleiven 2021, ADAPT, doi:10.1007/s10237-020-01391-8: brain minimum
Jacobian 0.45, mesh satisfactory when at least 95% of elements are over 0.5):
  every solid and shell part: no inverted element, minimum scaled Jacobian >= 0.45,
  at least 95% of its elements >= 0.5. Shells are held to the same bar as the solids.
Reported, not graded: edge aspect ratio, face warpage (degrees between the two triangles of
a quad), and the LS-DYNA characteristic length (solid: volume / largest face area; shell:
area / longest edge), whose smallest value sets the explicit time step.
Every run writes scorecard.json (mesh_agent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.export.mesh_audit import read_deck
from engine.export.validate import _hex_scaled_jacobian, _quad_scaled_jacobian

MIN_SJ = 0.45
SHARE_SJ = 0.5
SHARE = 95.0
_HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [3, 7, 6, 2],
                       [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5]])
_HEX_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]])


def _warp_deg(q):
    """Largest angle between the two triangles of a quad, over both diagonals."""
    out = np.zeros(len(q))
    for a, b, c, d in ((0, 1, 2, 3), (1, 2, 3, 0)):
        n1 = np.cross(q[:, b] - q[:, a], q[:, c] - q[:, a])
        n2 = np.cross(q[:, c] - q[:, a], q[:, d] - q[:, a])
        cos = (n1 * n2).sum(1) / np.maximum(np.linalg.norm(n1, axis=1)
                                             * np.linalg.norm(n2, axis=1), 1e-30)
        out = np.maximum(out, np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out


def _quad_area(q):
    return 0.5 * np.linalg.norm(np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]), axis=1)


def _solid_metrics(p):
    edges = np.linalg.norm(p[:, _HEX_EDGES[:, 1]] - p[:, _HEX_EDGES[:, 0]], axis=2)
    faces = p[:, _HEX_FACES]                                   # (n, 6, 4, 3)
    area = np.stack([_quad_area(faces[:, k]) for k in range(6)], axis=1)
    c = p.mean(axis=1)
    vol = np.zeros(len(p))
    for k in range(6):                                         # pyramids on each face
        f = faces[:, k]
        fc = f.mean(axis=1)
        for i in range(4):
            a, b = f[:, i] - c, f[:, (i + 1) % 4] - c
            vol += np.einsum("ij,ij->i", np.cross(a, b), fc - c) / 6.0
    warp = np.max(np.stack([_warp_deg(faces[:, k]) for k in range(6)], axis=1), axis=1)
    from engine.mesh import fast_kernels

    sj = (fast_kernels.hex_scaled_jacobian(p) if fast_kernels.HAVE_NUMBA
          else _hex_scaled_jacobian(p))
    return {"sj": sj, "aspect": edges.max(1) / np.maximum(edges.min(1), 1e-30),
            "warp": warp, "char": np.abs(vol) / np.maximum(area.max(1), 1e-30),
            "edge": np.median(edges, axis=1)}


def _in_chunks(function, rows, chunk: int = 200_000) -> dict:
    """`function(rows)` (a dict of per-row arrays) over row chunks on the shared thread pool;
    the same numbers as one call."""
    from engine.mesh.voxel_hex import THREADS, pool

    if len(rows) <= 2 * chunk or THREADS < 2:
        return function(rows)
    parts = list(pool().map(function, [rows[i:i + chunk] for i in range(0, len(rows), chunk)]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _shell_metrics(q):
    edges = np.linalg.norm(np.roll(q, -1, axis=1) - q, axis=2)
    return {"sj": _quad_scaled_jacobian(q), "aspect": edges.max(1) / np.maximum(edges.min(1), 1e-30),
            "warp": _warp_deg(q), "char": _quad_area(q) / np.maximum(edges.max(1), 1e-30),
            "edge": np.median(edges, axis=1)}


def score(deck: Path) -> dict:
    points, names, solids, shells, beams = read_deck(deck)
    parts = []
    for kind, table in (("solid", solids), ("shell", shells)):
        for pid, conn in table.items():
            if kind == "solid" and conn.shape[1] != 8:
                continue
            m = (_in_chunks(_solid_metrics, points[conn[:, :8]]) if kind == "solid"
                 else _in_chunks(_shell_metrics, points[conn[:, :4]]))
            sj = m["sj"]
            row = {"pid": int(pid), "name": names.get(pid, f"part {pid}"), "kind": kind,
                   "n": int(len(sj)), "sj_min": float(sj.min()), "sj_mean": float(sj.mean()),
                   "pct_sj_ge_0p5": float(np.mean(sj >= SHARE_SJ) * 100),
                   "n_negative": int((sj <= 0).sum()),
                   "aspect_max": float(m["aspect"].max()),
                   "aspect_p99": float(np.percentile(m["aspect"], 99)),
                   "warp_max_deg": float(m["warp"].max()),
                   "warp_p99_deg": float(np.percentile(m["warp"], 99)),
                   "char_min_mm": float(m["char"].min()),
                   "char_min_over_edge": float((m["char"] / np.maximum(m["edge"], 1e-30)).min())}
            row["passed"] = bool(row["n_negative"] == 0 and row["sj_min"] >= MIN_SJ
                                 and row["pct_sj_ge_0p5"] >= SHARE)
            parts.append(row)
    beam_min = min((float(np.linalg.norm(points[c[:, 1]] - points[c[:, 0]], axis=1).min())
                    for c in beams.values() if len(c)), default=None)
    solid = [r for r in parts if r["kind"] == "solid"]
    out = {"deck": str(deck), "parts": sorted(parts, key=lambda r: r["sj_min"]),
           "beam_min_length_mm": beam_min,
           "char_min_mm": min((r["char_min_mm"] for r in parts), default=None),
           "n_negative": sum(r["n_negative"] for r in parts),
           "solid_sj_min": min((r["sj_min"] for r in solid), default=None),
           "passed": all(r["passed"] for r in parts),
           "failing_parts": [r["name"] for r in parts if not r["passed"]]}
    return out
