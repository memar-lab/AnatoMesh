"""Render a finished LS-DYNA deck to PNG contact sheets you can actually check."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from engine.export import mesh_audit

logger = logging.getLogger(__name__)

PROJECTIONS = (((1, 2), "SAGITTAL (Y-Z), nose right"),
               ((0, 2), "CORONAL (X-Z), from behind"),
               ((0, 1), "AXIAL (X-Y), from above"))

SLICE_AXIS = {(1, 2): 0, (0, 2): 1, (0, 1): 2}

SLAB_HALF_MM = 1.5

MEMBRANE_COLOURS = {"falx": "#d1495b", "tentorium": "#edae49",
                    "pia": "#66a182", "dura": "#2e4057"}

_ENVELOPE_SHEETS = frozenset({"pia", "dura"})

VESSEL_COLOURS = {"sinus superior sagittal": "#1f77b4",
                  "sinus inferior sagittal": "#17becf",
                  "sinus straight": "#9467bd",
                  "sinus transverse": "#2ca02c",
                  "bridging veins": "#d62728"}


def _colours_for(names: dict) -> dict:
    """A stable colour per part id."""
    palette = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    out, i = {}, 0
    for pid in sorted(names, key=lambda p: names[p].lower()):
        fixed = TISSUE_COLOURS.get(names[pid].strip().lower())
        if fixed is not None:
            out[pid] = fixed
            continue
        out[pid] = palette[i % len(palette)]
        i += 1
    return out


def _slab(points: np.ndarray, conn: np.ndarray, axis: int, at: float) -> np.ndarray:
    """Rows of `conn` whose centroid lies within the slab at `at` on `axis`."""
    if not len(conn):
        return conn
    centre = points[conn][:, :, axis].mean(axis=1)
    return conn[np.abs(centre - at) <= SLAB_HALF_MM]


def _hex_outline(points: np.ndarray, hexes: np.ndarray, pair: tuple) -> np.ndarray:
    """A hex's projected footprint, as a quad, for filling."""
    a, b = pair
    corners = points[hexes][:, :, [a, b]]
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    return np.stack([np.stack([lo[:, 0], lo[:, 1]], axis=1),
                     np.stack([hi[:, 0], lo[:, 1]], axis=1),
                     np.stack([hi[:, 0], hi[:, 1]], axis=1),
                     np.stack([lo[:, 0], hi[:, 1]], axis=1)], axis=1)


TISSUE_COLOURS = {
    "csf": "#7fb3d5", "lateral ventricle": "#2874a6", "third ventricle": "#2874a6",
    "fourth ventricle": "#2874a6", "white matter": "#f2e8d5",
    "white matter lh": "#f2e8d5", "white matter rh": "#e8dcc0",
    "gray matter": "#c9a0a8", "skull": "#efe6d0", "scalp": "#d2a679", "soft tissue": "#e8b9a0", "skull diploe": "#e0d2b0",
    "cerebellum": "#7fbf7f", "cerebellar cortex": "#7fbf7f",
    "cerebellar white matter": "#a9d6a9", "brainstem": "#e59866",
    "deep nuclei": "#b39ddb", "thalamus": "#9575cd", "caudate": "#ce93d8",
    "putamen": "#ab47bc", "pallidum": "#8e24aa", "hippocampus": "#f06292",
    "amygdala": "#ec407a", "ventral dc": "#7e57c2", "accumbens": "#ba68c8",
}


def _draw_frame(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#f7f7f7")


def _brain_outline(ax, points: np.ndarray, conn_all: list, pair: tuple,
                   axis: int, at: float) -> None:
    """A faint fill of every solid element in the slab, as a backdrop."""
    patches = [_hex_outline(points, _slab(points, c, axis, at), pair)
               for c in conn_all if len(c) and c.shape[1] >= 4]
    patches = [p for p in patches if len(p)]
    if patches:
        ax.add_collection(PolyCollection(np.concatenate(patches), facecolors="#cccccc",
                                         edgecolors="none", alpha=0.55))
        ax.autoscale_view()


def render_sections(points, names, solids, out_path: Path, title: str) -> Path:
    """Cross sections through the solid mesh, each element in its part's colour."""
    colours = _colours_for({pid: names.get(pid, str(pid)) for pid in solids})
    centre = points.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    drawn: dict = {}
    for ax, (pair, name) in zip(axes, PROJECTIONS):
        axis = SLICE_AXIS[pair]
        for pid, conn in sorted(solids.items()):
            if conn.shape[1] < 4:
                continue
            cut = _slab(points, conn, axis, float(centre[axis]))
            if not len(cut):
                continue
            drawn[pid] = drawn.get(pid, 0) + len(cut)
            ax.add_collection(PolyCollection(
                _hex_outline(points, cut, pair), facecolors=[colours[pid]],
                edgecolors="none"))
        ax.autoscale_view()
        _draw_frame(ax, f"{name}\nslab +/- {SLAB_HALF_MM}mm at the mesh centroid")

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          color=colours[pid], label=f"{names.get(pid, pid)} ({n})")
               for pid, n in sorted(drawn.items(), key=lambda kv: -kv[1])[:24]]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(f"{title} -- solid mesh cross sections, coloured by *PART", fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_membranes(points, names, solids, shells, out_path: Path, title: str) -> Path:
    """The four meningeal sheets in silhouette, over the tissue they bound."""
    by_name = {names.get(pid, "").strip().lower(): (pid, conn)
               for pid, conn in shells.items()}
    order = ("dura", "pia", "tentorium", "falx")
    present = {n: by_name[n] for n in order if n in by_name}
    centre = points.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, (pair, name) in zip(axes, PROJECTIONS):
        axis = SLICE_AXIS[pair]
        _brain_outline(ax, points, list(solids.values()), pair, axis, float(centre[axis]))
        a, b = pair
        for membrane, (_pid, conn) in present.items():
            if not len(conn):
                continue
            if membrane in _ENVELOPE_SHEETS:
                cut = _slab(points, conn, axis, float(centre[axis]))
                if not len(cut):
                    continue
                ax.add_collection(PolyCollection(
                    _hex_outline(points, cut[:, :4], pair),
                    facecolors=MEMBRANE_COLOURS[membrane], edgecolors="none"))
            else:
                ax.add_collection(PolyCollection(
                    points[conn[:, :4]][:, :, [a, b]],
                    facecolors=MEMBRANE_COLOURS[membrane], edgecolors="none",
                    alpha=0.75))
        ax.autoscale_view()
        _draw_frame(ax, name)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          color=MEMBRANE_COLOURS[n], label=f"{n} ({len(c)} quads)")
               for n, (_p, c) in present.items()]
    if not handles:
        axes[0].text(0.5, 0.5, "no membrane shells in this deck", ha="center",
                     va="center", transform=axes[0].transAxes)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{title} -- meningeal membranes over the solid mesh", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _is_axon(names: dict, pid) -> bool:
    """Axon tract parts are titled "Axon Tract ..." by `mesh_agent`."""
    return names.get(pid, "").strip().lower().startswith("axon")


def render_vessels(points, names, solids, beams, out_path: Path, title: str) -> Path:
    """The venous sinuses and bridging veins over the brain outline."""
    centre = points.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    seen: dict = {}
    vessel_beams = {pid: conn for pid, conn in beams.items() if not _is_axon(names, pid)}
    for ax, (pair, name) in zip(axes, PROJECTIONS):
        axis = SLICE_AXIS[pair]
        _brain_outline(ax, points, list(solids.values()), pair, axis, float(centre[axis]))
        a, b = pair
        for pid, conn in sorted(vessel_beams.items()):
            label = names.get(pid, f"part {pid}").strip().lower()
            colour = VESSEL_COLOURS.get(label, "#000000")
            seen[label] = len(conn)
            segments = points[conn][:, :, [a, b]]
            width = 1.0 if label == "bridging veins" else 2.2
            ax.add_collection(LineCollection(segments, colors=colour, linewidths=width))
        ax.autoscale_view()
        _draw_frame(ax, name)
    handles = [plt.Line2D([], [], color=VESSEL_COLOURS.get(n, "#000000"), linewidth=2.5,
                          label=f"{n} ({c} beams)") for n, c in sorted(seen.items())]
    if not handles:
        axes[0].text(0.5, 0.5, "no vessel beams in this deck\n"
                               "(MeshingConfig.vascular_beams)",
                     ha="center", va="center", transform=axes[0].transAxes)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{title} -- vascular beams over the solid mesh", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_axons(points, names, solids, beams, out_path: Path, title: str) -> Path:
    """The axon tracts, coloured by the direction each beam runs."""
    axon_beams = {pid: conn for pid, conn in beams.items() if _is_axon(names, pid)}
    centre = points.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    total = 0
    for ax, (pair, name) in zip(axes, PROJECTIONS):
        axis = SLICE_AXIS[pair]
        _brain_outline(ax, points, list(solids.values()), pair, axis, float(centre[axis]))
        a, b = pair
        for pid, conn in sorted(axon_beams.items()):
            step = points[conn[:, 1]] - points[conn[:, 0]]
            norms = np.linalg.norm(step, axis=1, keepdims=True)
            colours = np.abs(step / np.where(norms > 1e-9, norms, 1.0))
            total += len(conn)
            ax.add_collection(LineCollection(points[conn][:, :, [a, b]],
                                             colors=colours, linewidths=0.6, alpha=0.75))
        ax.autoscale_view()
        _draw_frame(ax, name)
    if not axon_beams:
        axes[0].text(0.5, 0.5, "no axon tracts in this deck\n"
                               "(MeshingConfig.axon_tracts, and --dti on the run)",
                     ha="center", va="center", transform=axes[0].transAxes)
    handles = [plt.Line2D([], [], color=c, linewidth=2.5, label=lab) for c, lab in
               (("#ff0000", "left-right"), ("#00c000", "anterior-posterior"),
                ("#0000ff", "superior-inferior"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{title} -- axon tracts, coloured by direction "
                 f"({total // max(len(PROJECTIONS), 1):,} beams)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_dti(points, names, solids, ortho, fa_path, out_path: Path, title: str) -> Path:
    """The per-element fibre axes the deck carries, and the FA they came from."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 13))
    centre = points.mean(axis=0)
    drawn = 0
    for ax, (pair, name) in zip(axes[0], PROJECTIONS):
        axis = SLICE_AXIS[pair]
        at = float(centre[axis])
        _brain_outline(ax, points, list(solids.values()), pair, axis, at)
        a, b = pair
        for cells, vec in ortho.values():
            mid = points[cells].mean(axis=1)
            keep = np.abs(mid[:, axis] - at) <= SLAB_HALF_MM
            if not keep.any():
                continue
            m, v = mid[keep], vec[keep]
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            unit = v / np.where(norms > 1e-9, norms, 1.0)
            half = SLAB_HALF_MM
            seg = np.stack([m[:, [a, b]] - half * unit[:, [a, b]],
                            m[:, [a, b]] + half * unit[:, [a, b]]], axis=1)
            ax.add_collection(LineCollection(seg, colors=np.abs(unit),
                                             linewidths=0.7, alpha=0.85))
            drawn += int(keep.sum())
        ax.autoscale_view()
        _draw_frame(ax, f"fibre axes -- {name}")
    if not ortho:
        axes[0][0].text(0.5, 0.5, "no *ELEMENT_SOLID_ORTHO in this deck\n"
                                  "(run with --dti to get fibre axes)",
                        ha="center", va="center", transform=axes[0][0].transAxes)

    _render_fa_row(axes[1], fa_path)

    handles = [plt.Line2D([], [], color=c, linewidth=2.5, label=lab) for c, lab in
               (("#ff0000", "left-right"), ("#00c000", "anterior-posterior"),
                ("#0000ff", "superior-inferior"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{title} -- DTI: {drawn:,} element axes on the mid-slabs, "
                 f"{sum(len(c) for c, _ in ortho.values()):,} in the deck", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _render_fa_row(row, fa_path) -> None:
    """Mid-slices of the FA volume on its three axes, or why there are none."""
    if fa_path is None or not Path(fa_path).exists():
        for ax, (_pair, name) in zip(row, PROJECTIONS):
            _draw_frame(ax, f"FA -- {name}")
            ax.text(0.5, 0.5, "no dti_fa.nii.gz beside the deck",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return
    try:
        import nibabel as nib
        fa = np.asanyarray(nib.load(str(fa_path)).dataobj).astype(float)
    except Exception:
        logger.exception("could not read %s", fa_path)
        for ax in row:
            _draw_frame(ax, "FA -- unreadable")
        return
    mean_fa = float(fa[fa > 0].mean()) if (fa > 0).any() else 0.0
    for ax, arr_axis in zip(row, (0, 1, 2)):
        sl = np.take(fa, fa.shape[arr_axis] // 2, axis=arr_axis)
        ax.imshow(np.rot90(sl), cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"FA, mid-slice on array axis {arr_axis} "
                     f"(max {fa.max():.2f}, mean over nonzero {mean_fa:.2f})",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])


def render_quality(rows: list, out_path: Path, title: str, threshold: float) -> Path:
    """Per-part element count and Scaled Jacobian, so the bad part is named."""
    graded = sorted((r for r in rows if r["kind"] != "beam"),
                    key=lambda r: r["pct_above"])[:28]
    fig, (ax_pct, ax_n) = plt.subplots(1, 2, figsize=(17, max(5, 0.34 * len(graded) + 2)))
    y = np.arange(len(graded))
    labels = [f"{r['name']} [{r['kind']}]" for r in graded]

    bars = ax_pct.barh(y, [r["pct_above"] for r in graded],
                       color=["#c0392b" if r["n_negative"] else
                              "#e67e22" if r["pct_above"] < 99 else "#27ae60"
                              for r in graded])
    ax_pct.axvline(99.0, color="#333333", linestyle="--", linewidth=1)
    ax_pct.set_yticks(y, labels, fontsize=8)
    ax_pct.set_xlabel(f"% of elements at Scaled Jacobian >= {threshold}")
    ax_pct.set_xlim(0, 118)
    ax_pct.set_title("worst parts first; dashed line is the 99% target", fontsize=9)
    for bar, r in zip(bars, graded):
        note = f"  min {r['min']:.3f}" + ("  INVERTED" if r["n_negative"] else "")
        ax_pct.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, note,
                    va="center", fontsize=7)

    ax_n.barh(y, [max(r["n"], 1) for r in graded], color="#4477aa")
    ax_n.set_yticks(y, [""] * len(graded))
    ax_n.set_xscale("log")
    ax_n.set_xlabel("elements in the part (log scale)")
    ax_n.set_title("where the element budget is spent", fontsize=9)

    total = sum(r["n"] for r in rows)
    below = sum(r["n_below"] for r in rows)
    fig.suptitle(f"{title} -- {total:,} elements, "
                 f"{(total - below) / max(total, 1) * 100:.2f}% at SJ >= {threshold}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def preview(deck: Path, out_dir=None, threshold: float = 0.96) -> dict:
    """Write every panel for one deck."""
    deck = Path(deck)
    out_dir = Path(out_dir) if out_dir else deck.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = deck.stem
    title = f"{deck.parent.name} / {deck.name}"

    points, names, solids, shells, beams = mesh_audit.read_deck(deck)
    logger.info("%s: %d nodes, %d solid parts, %d shell parts, %d beam parts",
                deck.name, len(points), len(solids), len(shells), len(beams))

    out = {
        "sections": render_sections(points, names, solids,
                                    out_dir / f"{stem}_sections.png", title),
        "membranes": render_membranes(points, names, solids, shells,
                                      out_dir / f"{stem}_membranes.png", title),
    }
    vessel_beams = {pid: conn for pid, conn in beams.items() if not _is_axon(names, pid)}
    if vessel_beams:
        out["vessels"] = render_vessels(points, names, solids, beams,
                                        out_dir / f"{stem}_vessels.png", title)
    axon_beams = {pid: conn for pid, conn in beams.items() if _is_axon(names, pid)}
    if axon_beams:
        out["axons"] = render_axons(points, names, solids, beams,
                                    out_dir / f"{stem}_axons.png", title)
    try:
        ortho = mesh_audit.read_ortho_axes(deck)
        if ortho:
            out["dti"] = render_dti(points, names, solids, ortho,
                                    deck.parent / "dti_fa.nii.gz",
                                    out_dir / f"{stem}_dti.png", title)
    except Exception:
        logger.exception("%s: the DTI panel failed; the other panels stand", deck.name)
    try:
        out["quality"] = render_quality(mesh_audit.audit(deck, threshold),
                                        out_dir / f"{stem}_quality.png", title, threshold)
    except Exception:
        logger.exception("%s: the quality chart failed; the other panels stand", deck.name)
    return {k: str(v) for k, v in out.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path, nargs="+", help="one or more mesh_*.k files")
    parser.add_argument("--out", type=Path, default=None,
                        help="where the PNGs go (default: beside each deck)")
    parser.add_argument("--threshold", type=float, default=0.96)
    args = parser.parse_args()

    for deck in args.deck:
        for name, path in preview(deck, args.out, args.threshold).items():
            print(f"{name:<10} {path}")


if __name__ == "__main__":
    main()
