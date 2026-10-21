"""Procedural vascular tree"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

VESSEL_CLASSES: dict[str, tuple[str, float, str]] = {
    "sinus_superior_sagittal": ("vessel_sinus", 10.0, "Sinus Superior Sagittal"),
    "sinus_inferior_sagittal": ("vessel_sinus", 3.0, "Sinus Inferior Sagittal"),
    "sinus_straight":          ("vessel_sinus", 6.0, "Sinus Straight"),
    "sinus_transverse":        ("vessel_sinus", 8.0, "Sinus Transverse"),
    "bridging_vein":           ("vessel_bridging", 1.8, "Bridging Veins"),
    "artery_large":            ("vessel_artery", 4.0, "Arteries Large"),
    "artery_small":            ("vessel_artery", 1.0, "Arteries Small"),
    "vein_cortical":           ("vessel_vein", 2.5, "Veins Cortical"),
}

_BRIDGING_SPACING_MM = 20.0

_BRIDGING_MAX_MM = 25.0

_MIN_BEAM_FRACTION = 0.5


def _boundary_polylines(verts: np.ndarray, quads: np.ndarray) -> list[np.ndarray]:
    """Ordered vertex-index chains around the open boundary of a quad sheet."""
    if not len(quads):
        return []
    edges = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]],
                            quads[:, [2, 3]], quads[:, [3, 0]]])
    key = np.sort(edges, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    border = uniq[counts == 1]
    if not len(border):
        return []

    neighbours: dict[int, list[int]] = {}
    for a, b in border:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    seen_edge: set[tuple[int, int]] = set()
    chains: list[np.ndarray] = []
    for start in list(neighbours):
        for first in neighbours[start]:
            if (min(start, first), max(start, first)) in seen_edge:
                continue
            chain = [start, first]
            seen_edge.add((min(start, first), max(start, first)))
            while True:
                current, previous = chain[-1], chain[-2]
                nxt = [v for v in neighbours.get(current, [])
                       if v != previous
                       and (min(current, v), max(current, v)) not in seen_edge]
                if not nxt:
                    break
                step = nxt[0]
                seen_edge.add((min(current, step), max(current, step)))
                chain.append(step)
                if step == chain[0]:
                    break
            if len(chain) > 2:
                chains.append(np.array(chain, dtype=np.int64))
    chains.sort(key=len, reverse=True)
    return chains


def _resample(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Even arc-length samples along a polyline, both endpoints kept."""
    if len(points) < 2:
        return points
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = float(arc[-1])
    if total < spacing_mm:
        return points[[0, -1]]
    n = max(2, int(round(total / spacing_mm)) + 1)
    want = np.linspace(0.0, total, n)
    out = np.stack([np.interp(want, arc, points[:, c]) for c in range(3)], axis=1)

    floor = _MIN_BEAM_FRACTION * spacing_mm
    keep = [0]
    for i in range(1, len(out) - 1):
        if np.linalg.norm(out[i] - out[keep[-1]]) >= floor:
            keep.append(i)
    while len(keep) > 1 and np.linalg.norm(out[-1] - out[keep[-1]]) < floor:
        keep.pop()
    keep.append(len(out) - 1)
    return out[keep]


def _chain_world(verts: np.ndarray, chain: np.ndarray) -> np.ndarray:
    return np.asarray(verts, dtype=float)[chain]


def _centreline(cloud: np.ndarray, bin_mm: float) -> np.ndarray:
    """One point per AP station"""
    cloud = np.asarray(cloud, dtype=float)
    if len(cloud) < 2:
        return cloud
    station = np.round(cloud[:, 1] / max(bin_mm, 1e-6)).astype(np.int64)
    order = np.argsort(station, kind="stable")
    cloud, station = cloud[order], station[order]
    cuts = np.flatnonzero(np.diff(station)) + 1
    return np.array([np.median(part, axis=0)
                     for part in np.split(cloud, cuts)], dtype=float)


def _split_by_height(verts: np.ndarray, chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a falx outline into its attached (upper) and free (lower) margins."""
    pts = _chain_world(verts, chain)
    front, back = int(np.argmax(pts[:, 1])), int(np.argmin(pts[:, 1]))
    lo, hi = sorted((front, back))
    part_a = chain[lo:hi + 1]
    part_b = np.concatenate([chain[hi:], chain[:lo + 1]])
    mean_z_a = _chain_world(verts, part_a)[:, 2].mean()
    mean_z_b = _chain_world(verts, part_b)[:, 2].mean()
    return (part_a, part_b) if mean_z_a > mean_z_b else (part_b, part_a)


def _beams(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A polyline's points plus its consecutive 2-node segments."""
    n = len(points)
    return points, np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)


def build_vascular_tree(
    falx: tuple[np.ndarray, np.ndarray] | None,
    tentorium: tuple[np.ndarray, np.ndarray] | None,
    pia_verts: np.ndarray | None = None,
    spacing_mm: float = 4.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sinus and bridging-vein centerlines for this subject, as beam blocks."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if falx is not None and len(falx[1]):
        chains = _boundary_polylines(*falx)
        if chains:
            attached, free = _split_by_height(falx[0], chains[0])
            for name, chain in (("sinus_superior_sagittal", attached),
                                ("sinus_inferior_sagittal", free)):
                pts = _resample(_chain_world(falx[0], chain), spacing_mm)
                if len(pts) >= 2:
                    out[name] = _beams(pts)

    if tentorium is not None and len(tentorium[1]):
        chains = _boundary_polylines(*tentorium)
        if chains:
            pts = _resample(_chain_world(tentorium[0], chains[0]), spacing_mm)
            if len(pts) >= 2:
                out["sinus_transverse"] = _beams(pts)

    if tentorium is not None and len(tentorium[0]):
        tv = np.asarray(tentorium[0], dtype=float)
        midline = np.abs(tv[:, 0] - np.median(tv[:, 0])) < spacing_mm
        if midline.sum() >= 2:
            ridge = _centreline(tv[midline], spacing_mm)
            pts = _resample(ridge, spacing_mm) if len(ridge) >= 2 else ridge
            if len(pts) >= 2:
                out["sinus_straight"] = _beams(pts)

    if pia_verts is not None and len(pia_verts) and "sinus_superior_sagittal" in out:
        veins = _bridging_veins(out["sinus_superior_sagittal"][0],
                                np.asarray(pia_verts, dtype=float))
        if veins is not None:
            out["bridging_vein"] = veins

    for name, (pts, seg) in out.items():
        length = float(np.linalg.norm(pts[seg[:, 1]] - pts[seg[:, 0]], axis=1).sum())
        logger.info("vascular: %s -- %d beams, %.1fmm total", name, len(seg), length)
    return out


def _bridging_veins(sinus_points: np.ndarray, pia_verts: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    """One beam per attachment, from a sinus node down to the nearest cortex."""
    if len(sinus_points) < 2:
        return None
    step = np.linalg.norm(np.diff(sinus_points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] < _BRIDGING_SPACING_MM:
        return None
    want = np.arange(0.0, arc[-1], _BRIDGING_SPACING_MM)
    seeds = np.stack([np.interp(want, arc, sinus_points[:, c]) for c in range(3)], axis=1)

    midline_x = float(np.median(sinus_points[:, 0]))
    points: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    for side in (-1.0, 1.0):
        hemisphere = pia_verts[(pia_verts[:, 0] - midline_x) * side > 0]
        if not len(hemisphere):
            continue
        distance, index = cKDTree(hemisphere).query(seeds)
        keep = distance <= _BRIDGING_MAX_MM
        for seed, target in zip(seeds[keep], hemisphere[index[keep]]):
            segments.append((len(points), len(points) + 1))
            points.extend([seed, target])
    if not segments:
        logger.info("vascular: no bridging vein had a cortical attachment within %.0fmm",
                    _BRIDGING_MAX_MM)
        return None
    return np.array(points, dtype=float), np.array(segments, dtype=np.int64)


def merge_into_mesh(tree: dict[str, tuple[np.ndarray, np.ndarray]],
                    nodes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Append every class's points to the mesh's node array."""
    nodes = np.asarray(nodes, dtype=float)
    blocks: dict[str, np.ndarray] = {}
    extra: list[np.ndarray] = []
    offset = len(nodes)
    for name, (points, segments) in tree.items():
        blocks[name] = segments + offset
        extra.append(points)
        offset += len(points)
    if not extra:
        return nodes, {}
    return np.concatenate([nodes] + extra), blocks


def demo() -> None:
    """Self-check on a flat sheet, where every answer is known by hand."""
    nu, nv = 10, 4
    verts = np.array([[0.0, float(i), float(j)] for i in range(nu) for j in range(nv)])
    quads = np.array([[i * nv + j, i * nv + j + nv, i * nv + j + nv + 1, i * nv + j + 1]
                      for i in range(nu - 1) for j in range(nv - 1)], dtype=np.int64)

    chains = _boundary_polylines(verts, quads)
    assert chains, "a rectangular sheet has a boundary"
    assert len(chains[0]) == 2 * (nu - 1) + 2 * (nv - 1) + 1, len(chains[0])

    attached, free = _split_by_height(verts, chains[0])
    assert verts[attached][:, 2].mean() > verts[free][:, 2].mean()

    pts = _resample(np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]), 2.0)
    assert len(pts) == 6, len(pts)
    assert np.allclose(pts[-1], [0.0, 10.0, 0.0])

    hairpin = np.array([[0.0, y, 0.0] for y in np.linspace(0, 20, 21)]
                       + [[0.1, y, 0.0] for y in np.linspace(20, 0, 21)])
    got = _resample(hairpin, 4.0)
    gaps = np.linalg.norm(np.diff(got, axis=0), axis=1)
    assert gaps.min() >= _MIN_BEAM_FRACTION * 4.0, gaps.min()
    assert np.allclose(got[0], hairpin[0]) and np.allclose(got[-1], hairpin[-1])

    band = np.array([[dx, float(y), 40.0 + dz]
                     for y in range(0, 60)
                     for dx, dz in ((-1.0, 0.5), (0.0, 0.0), (1.0, -0.5))])
    line = _centreline(band, 1.0)
    assert len(line) == 60, len(line)
    assert np.allclose(line[:, 0], 0.0) and np.allclose(line[:, 2], 40.0)
    span = float(line[:, 1].max() - line[:, 1].min())
    walked = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    assert walked <= span * 1.05, (walked, span)

    tree = build_vascular_tree((verts, quads), None, spacing_mm=1.0)
    assert "sinus_superior_sagittal" in tree and "sinus_inferior_sagittal" in tree
    assert "sinus_transverse" not in tree, "no tentorium was given"

    sinus = np.stack([np.zeros(30), np.linspace(0, 120, 30), np.full(30, 50.0)], axis=1)
    near = np.array([[5.0, float(y), 48.0] for y in range(0, 120, 5)]
                    + [[-5.0, float(y), 48.0] for y in range(0, 120, 5)])
    got = _bridging_veins(sinus, near)
    assert got is not None and len(got[1]) > 0
    assert np.linalg.norm(got[0][got[1][:, 1]] - got[0][got[1][:, 0]],
                          axis=1).max() <= _BRIDGING_MAX_MM
    assert _bridging_veins(sinus, near + np.array([0.0, 0.0, -200.0])) is None

    nodes = np.zeros((7, 3))
    merged, blocks = merge_into_mesh(tree, nodes)
    assert len(merged) > len(nodes)
    assert min(b.min() for b in blocks.values()) >= len(nodes), "beams must not reuse mesh nodes"
    print("vascular demo OK:", {k: len(s) for k, (_p, s) in tree.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
