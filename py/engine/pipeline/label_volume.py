"""Shared pieces of the voxel method for label volumes in the tissue scheme."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from engine.mesh import voxel_hex
from engine.paths import assert_write_path
from engine.mesh import fast_kernels

BRAIN_LABELS = (1, 2, 3, 4, 5)
COMBINED_BRAIN = 1


def resample_labels(seg: np.ndarray, voxel_mm: float, element_mm: float) -> np.ndarray:
    """Resample a label volume onto a lattice of `element_mm` voxels (largest signed distance wins)."""
    from engine.pipeline.sdf_smoothing import THREADS

    factor = element_mm / voxel_mm
    if abs(factor - 1) < 1e-9:
        return np.asarray(seg).copy()
    seg = np.asarray(seg)
    shape = tuple(np.maximum(1, np.round(np.array(seg.shape) / factor).astype(int)))
    centres = [(np.arange(n) + 0.5) * factor - 0.5 for n in shape]
    labels, counts = np.unique(seg, return_counts=True)
    order = labels[np.argsort(counts, kind="stable")]
    big = order[-1]

    def field(label):
        # Each label only in its box plus a margin: distances inside the box are exact, and
        # a label cannot win more than a voxel outside it. The largest label (background)
        # takes the box of everything else and wins outright beyond it.
        inside = seg == label
        where = np.argwhere(seg != big if label == big else inside)
        lo = np.maximum(where.min(0) - 3, 0)
        hi = np.minimum(where.max(0) + 4, seg.shape)
        part = inside[tuple(slice(a, b) for a, b in zip(lo, hi))]
        sdf = (fast_kernels.distance_transform_edt(part).astype(np.float32)
               - fast_kernels.distance_transform_edt(~part).astype(np.float32))
        keep = [np.flatnonzero((c >= lo[k] + (lo[k] > 0)) & (c <= hi[k] - 1 - (hi[k] < seg.shape[k])))
                for k, c in enumerate(centres)]
        out = np.full(shape, np.inf if label == big else -np.inf, dtype=np.float32)
        if all(len(k) for k in keep):
            grid = np.meshgrid(*[centres[k][keep[k]] - lo[k] for k in range(3)], indexing="ij")
            out[np.ix_(*keep)] = ndi.map_coordinates(sdf, grid, order=1, mode="nearest")
        return out

    best_value = np.full(shape, -np.inf, dtype=np.float32)
    best = np.zeros(shape, dtype=seg.dtype)
    with ThreadPoolExecutor(max(1, min(THREADS, 4))) as pool:
        for label, value in zip(order, pool.map(field, order)):   # combined in count order
            win = value > best_value
            best_value[win] = value[win]
            best[win] = label
    return best


def combine(seg: np.ndarray, brain_labels=BRAIN_LABELS) -> np.ndarray:
    """combining the brain regions"""
    out = np.asarray(seg).copy()
    out[np.isin(out, brain_labels)] = COMBINED_BRAIN
    return out


def overlay(smoothed: np.ndarray, tissues: np.ndarray, brain_labels=BRAIN_LABELS) -> np.ndarray:
    """restoring the brain regions for the tissue scheme."""
    out = np.asarray(smoothed).copy()
    brain = out == COMBINED_BRAIN
    has_tissue = np.isin(tissues, brain_labels)
    if not has_tissue.any():
        return out
    _, nearest = ndi.distance_transform_edt(~has_tissue, return_indices=True)
    nearest_label = tissues[tuple(nearest)]
    out[brain] = np.where(has_tissue[brain], tissues[brain], nearest_label[brain])
    return out


_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]])
_FACE_PAIRS = (([0, 1, 2, 3], [4, 5, 6, 7]), ([0, 1, 5, 4], [3, 2, 6, 7]),
               ([0, 3, 7, 4], [1, 2, 6, 5]))


def hex_volumes(coords: np.ndarray) -> np.ndarray:
    """Hex volume from a five-tetrahedron split (the reference uses `convhull`)."""
    total = np.zeros(len(coords))
    for a, b, c, d in _TETS:
        total += np.abs(np.einsum("ij,ij->i", coords[:, b] - coords[:, a],
                                  np.cross(coords[:, c] - coords[:, a],
                                           coords[:, d] - coords[:, a]))) / 6
    return total


def write_histograms(coords: np.ndarray, path: Path, title: str = "",
                     volume_unit: str = "lattice units") -> Path:
    """The reference implementation's `FinalHistograms_Iter_<n>.png`"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jac, edge, diag = voxel_hex.compute_metrics(coords)
    volume = hex_volumes(coords)
    path = assert_write_path(Path(path))
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    panels = ((volume, "green", f"Element volume ({volume_unit}$^3$)"),
              (jac, "blue", "Scaled Jacobian"),
              (edge, "magenta", "Edge ratio (L$_{max}$/L$_{min}$)"),
              (diag, "red", "Diagonal ratio (D$_{min}$/D$_{max}$)"))
    for ax, (values, colour, label) in zip(axes, panels):
        ax.hist(values[np.isfinite(values)], bins=10, color=colour, edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"min {np.nanmin(values):.3f}  mean {np.nanmean(values):.3f}", fontsize=9)
    fig.suptitle(f"{title}   {len(coords):,} hexes", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_sections(coords: np.ndarray, parts: np.ndarray, path: Path, title: str = "",
                   names: dict | None = None) -> Path:
    """Three mid-planes"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    path = assert_write_path(Path(path))
    centroids = coords.mean(axis=1)
    size = float(np.median(np.linalg.norm(coords[:, 1] - coords[:, 0], axis=1)))
    palette = plt.get_cmap("tab20")
    ids = {p: i for i, p in enumerate(np.unique(parts))}
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for axis, ax in enumerate(axes):
        plane = float(np.median(centroids[:, axis]))
        cut = np.abs(centroids[:, axis] - plane) < 0.5 * size
        c = coords[cut]
        separation = np.stack([np.abs(c[:, lo].mean(1)[:, axis] - c[:, hi].mean(1)[:, axis])
                               for lo, hi in _FACE_PAIRS], axis=1)
        pair = separation.argmax(axis=1)
        quads = np.empty((len(c), 4, 3))
        for k, (lo, hi) in enumerate(_FACE_PAIRS):
            chosen = pair == k
            quads[chosen] = (c[chosen][:, lo] + c[chosen][:, hi]) / 2
        keep = [a for a in range(3) if a != axis]
        colours = [palette(ids[p] % 20) for p in parts[cut]]
        ax.add_collection(PolyCollection(quads[:, :, keep], facecolors=colours,
                                         edgecolors="black", linewidths=0.15))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"{'xyz'[axis]} = {plane:.1f}  ({cut.sum():,} hexes)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette(i % 20)) for i in ids.values()]
    labels = [(names or {}).get(p, f"part {p}") for p in ids]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(ids), 10), fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_figures(result: dict, step: int, out_dir: Path, title: str = "") -> dict:
    """Both figures for a `voxel_hex.generate_mesh` result at pass `step` (lattice units)."""
    coords = result["nodes_per_iteration"][step][result["elements"] - 1]
    hist = write_histograms(coords, Path(out_dir) / f"FinalHistograms_Iter_{step}.png",
                            f"{title}   iteration {step}")
    sections = write_sections(coords, result["part_ids"],
                              Path(out_dir) / f"Sections_Iter_{step}.png", title)
    return {"histograms": str(hist), "sections": str(sections)}
