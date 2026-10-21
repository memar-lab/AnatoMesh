"""Render every atlas segmentation to a PNG, so it can be looked at."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CYCLE = 20


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    """Centroid of the labelled voxels, so the slices cut through the anatomy."""
    idx = np.argwhere(mask)
    if not len(idx):
        return tuple(s // 2 for s in mask.shape)
    return tuple(int(round(v)) for v in idx.mean(axis=0))


def render(
    volume: np.ndarray, path: Path, background: Optional[np.ndarray] = None,
    title: str = "parcellation", names: Optional[dict[int, str]] = None,
) -> Optional[Path]:
    """Write a three-slice contact sheet of `volume`'s labels to `path`."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception:
        logger.info("matplotlib unavailable; skipping the %s preview", title)
        return None

    ids = np.unique(volume)
    ids = ids[ids > 0]
    if not len(ids):
        logger.info("%s is empty; no preview written", title)
        return None

    slot = np.zeros(int(volume.max()) + 1, dtype=np.int32)
    slot[ids] = np.arange(1, len(ids) + 1) % _CYCLE + 1

    base = plt.get_cmap("tab20").colors
    cmap = ListedColormap([(0, 0, 0, 0)] + [(*base[i % _CYCLE], 1.0) for i in range(_CYCLE + 1)])

    centre = _slice_indices(volume > 0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for axis, ax in enumerate(axes):
        painted = np.take(slot[volume], centre[axis], axis=axis).T
        if background is not None and background.shape == volume.shape:
            grey = np.take(background, centre[axis], axis=axis).T.astype(float)
            hi = np.percentile(grey, 99.0) or 1.0
            ax.imshow(np.clip(grey / hi, 0, 1), cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_equal(painted, 0), cmap=cmap, origin="lower",
                  interpolation="nearest", vmin=0, vmax=_CYCLE + 1, alpha=0.75)
        ax.set_title("sagittal coronal axial".split()[axis], fontsize=9)
        ax.axis("off")

    n_named = len({v for v in (names or {}).values()}) if names else len(ids)
    fig.suptitle(f"{title}: {len(ids)} labels, {n_named} named regions", fontsize=11)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("%s preview (%d labels) -> %s", title, len(ids), path)
    return path


def demo() -> None:
    """Self-check"""
    import tempfile

    volume = np.zeros((40, 40, 40), dtype=np.int32)
    for i in range(74):
        volume[i % 37 + 1, :, :] = 11100 + i
    slot_ids = np.unique(volume)
    assert len(slot_ids[slot_ids > 0]) == 37

    with tempfile.TemporaryDirectory() as tmp:
        out = render(volume, Path(tmp) / "p.png", title="destrieux")
        if out is None:
            print("atlas_preview.demo skipped: no matplotlib")
            return
        assert out.exists() and out.stat().st_size > 5000, out.stat().st_size

        assert render(np.zeros((5, 5, 5), dtype=np.int32), Path(tmp) / "e.png") is None
    print(f"atlas_preview.demo OK: {len(slot_ids) - 1} labels rendered")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
