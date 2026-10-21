"""Pick the solid part each beam family actually runs inside."""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

_MAX_HOST_DISTANCE_ELEMENTS = 1.5

_SINGLE_HOST_SHARE = 0.85

_SET_COVERAGE = 0.97


def _centroid_index(solid_blocks: dict, nodes: np.ndarray,
                    candidates: Optional[set] = None):
    """k-d tree over every solid element centroid, tagged with its part name."""
    names: list[str] = []
    centroids: list[np.ndarray] = []
    for name, cells in solid_blocks.items():
        if candidates is not None and name not in candidates:
            continue
        cells = np.asarray(cells)
        if len(cells) == 0:
            continue
        centroids.append(nodes[cells].mean(axis=1))
        names.append(name)
    if not centroids:
        return None, [], np.zeros(0, dtype=int)
    owner = np.concatenate([np.full(len(c), i, dtype=int)
                            for i, c in enumerate(centroids)])
    return cKDTree(np.concatenate(centroids, axis=0)), names, owner


def choose_hosts(beam_points: dict,
                 solid_blocks: dict,
                 nodes: np.ndarray,
                 element_size_mm: float,
                 candidates: Optional[set] = None,
                 ) -> tuple[dict, dict]:
    """Decide, per beam part, which solid part(s) it is embedded in."""
    tree, names, owner = _centroid_index(solid_blocks, nodes, candidates)
    hosts: dict[str, Union[str, list]] = {}
    report: dict = {}
    if tree is None:
        logger.warning("embedding: no solid part has any element; nothing can host a beam")
        return hosts, {name: {"coupled": 0, "reason": "no solid parts in this mesh"}
                       for name in beam_points}

    cutoff = _MAX_HOST_DISTANCE_ELEMENTS * float(element_size_mm)
    for beam_name, pts in beam_points.items():
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            report[beam_name] = {"coupled": 0, "reason": "no beam nodes"}
            continue
        distance, index = tree.query(pts)
        inside = distance <= cutoff
        n_in = int(inside.sum())
        entry = {"nodes": int(len(pts)), "coupled": n_in,
                 "uncoupled": int(len(pts) - n_in),
                 "mean_distance_mm": round(float(distance[inside].mean()), 3) if n_in else None}
        if n_in == 0:
            entry["reason"] = (f"every node is more than {cutoff:.1f}mm from any solid "
                               f"element centroid (nearest {distance.min():.1f}mm)")
            report[beam_name] = entry
            logger.error("embedding: %s has no solid host -- %s", beam_name, entry["reason"])
            continue

        counts = np.bincount(owner[index[inside]], minlength=len(names))
        order = np.argsort(counts)[::-1]
        share = counts[order[0]] / n_in
        entry["vote_share"] = round(float(share), 3)
        if share >= _SINGLE_HOST_SHARE:
            hosts[beam_name] = names[order[0]]
            entry["host"] = names[order[0]]
        else:
            chosen, covered = [], 0
            for i in order:
                if counts[i] == 0:
                    break
                chosen.append(names[i])
                covered += counts[i]
                if covered / n_in >= _SET_COVERAGE:
                    break
            hosts[beam_name] = chosen
            entry["host"] = chosen
            entry["set_coverage"] = round(float(covered / n_in), 3)
        report[beam_name] = entry

    logger.info("embedding: %s", {k: v.get("host", "NONE") for k, v in report.items()})
    return hosts, report


def demo() -> None:
    """Self-check"""
    left = np.array([[0., 0., 0.], [4, 0, 0], [4, 4, 0], [0, 4, 0],
                     [0, 0, 4], [4, 0, 4], [4, 4, 4], [0, 4, 4]])
    right = left + np.array([4.0, 0, 0])
    nodes = np.concatenate([left, right])
    solids = {"white_matter": np.arange(8).reshape(1, 8),
              "gray_matter": (np.arange(8) + 8).reshape(1, 8)}

    inside_left = np.array([[2.0, 2, 1], [2, 2, 2], [2, 2, 3]])
    spanning = np.array([[1.0, 2, 2], [2, 2, 2], [6, 2, 2], [7, 2, 2]])
    far = np.array([[100.0, 100, 100], [101, 100, 100]])
    hosts, report = choose_hosts(
        {"a": inside_left, "b": spanning, "c": far}, solids, nodes, element_size_mm=4.0)

    assert hosts["a"] == "white_matter", hosts["a"]
    assert sorted(hosts["b"]) == ["gray_matter", "white_matter"], hosts["b"]
    assert "c" not in hosts and report["c"]["coupled"] == 0, report["c"]
    assert report["a"]["uncoupled"] == 0
    print("embedding.demo ok:", {k: v.get("host") for k, v in report.items()})


if __name__ == "__main__":
    demo()
