"""Optional finer anatomical parts, carved out of the tissue blocks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ASEG_FILE = "structures_aseg.nii.gz"
BRAINSTEM_FILE = "structures_brainstem.nii.gz"

_ASEG_SOURCES = ("aseg.mgz",)
_BRAINSTEM_SOURCES = (
    "brainstemSsLabels.v13.FSvoxelSpace.mgz",
    "brainstemSsLabels.v12.FSvoxelSpace.mgz",
    "brainstemSsLabels.v10.FSvoxelSpace.mgz",
    "brainstemSsLabels.mgz",
)


class Structure:
    """One optional part"""

    __slots__ = ("name", "source", "ids", "title", "volume")

    def __init__(self, name: str, source: str, ids: tuple[int, ...], title: str,
                 volume: str = "aseg"):
        self.name = name
        self.source = source
        self.ids = ids
        self.title = title
        self.volume = volume

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Structure({self.name!r}, from {self.source!r})"


def _s(*args, **kwargs) -> tuple[str, Structure]:
    structure = Structure(*args, **kwargs)
    return structure.name, structure


STRUCTURE_GROUPS: dict[str, tuple[str, ...]] = {
    "Brainstem": ("midbrain", "pons", "medulla", "superior_cerebellar_peduncle"),
    "Corpus callosum": ("corpus_callosum", "cc_anterior", "cc_mid_anterior",
                        "cc_central", "cc_mid_posterior", "cc_posterior"),
    "Deep nuclei": ("thalamus", "caudate", "putamen", "pallidum", "hippocampus",
                    "amygdala", "accumbens", "ventral_dc"),
    "Cerebellum": ("cerebellar_cortex", "cerebellar_white_matter"),
    "Ventricles": ("lateral_ventricle", "inferior_lateral_ventricle",
                   "third_ventricle", "fourth_ventricle"),
    "Hemispheres": ("gray_matter_lh", "gray_matter_rh",
                    "white_matter_lh", "white_matter_rh"),
}

STRUCTURES: dict[str, Structure] = dict([
    _s("midbrain", "brainstem", (173,), "midbrain", volume="brainstem"),
    _s("pons", "brainstem", (174,), "pons", volume="brainstem"),
    _s("medulla", "brainstem", (175,), "medulla oblongata", volume="brainstem"),
    _s("superior_cerebellar_peduncle", "brainstem", (178,),
       "superior cerebellar peduncle", volume="brainstem"),

    _s("corpus_callosum", "white_matter", (251, 252, 253, 254, 255),
       "corpus callosum, whole"),
    _s("cc_anterior", "white_matter", (255,), "corpus callosum, anterior"),
    _s("cc_mid_anterior", "white_matter", (254,), "corpus callosum, mid-anterior"),
    _s("cc_central", "white_matter", (253,), "corpus callosum, central"),
    _s("cc_mid_posterior", "white_matter", (252,), "corpus callosum, mid-posterior"),
    _s("cc_posterior", "white_matter", (251,), "corpus callosum, posterior"),

    _s("thalamus", "deep_nuclei", (10, 49), "thalamus"),
    _s("caudate", "deep_nuclei", (11, 50), "caudate"),
    _s("putamen", "deep_nuclei", (12, 51), "putamen"),
    _s("pallidum", "deep_nuclei", (13, 52), "globus pallidus"),
    _s("hippocampus", "deep_nuclei", (17, 53), "hippocampus"),
    _s("amygdala", "deep_nuclei", (18, 54), "amygdala"),
    _s("accumbens", "deep_nuclei", (26, 58), "nucleus accumbens"),
    _s("ventral_dc", "deep_nuclei", (28, 60), "ventral diencephalon"),

    _s("cerebellar_cortex", "cerebellum", (8, 47), "cerebellar cortex"),
    _s("cerebellar_white_matter", "cerebellum", (7, 46), "cerebellar white matter"),

    _s("lateral_ventricle", "csf", (4, 43), "lateral ventricles"),
    _s("inferior_lateral_ventricle", "csf", (5, 44), "inferior horns"),
    _s("third_ventricle", "csf", (14,), "third ventricle"),
    _s("fourth_ventricle", "csf", (15,), "fourth ventricle"),

    _s("gray_matter_lh", "gray_matter", (3,), "cortex, left"),
    _s("gray_matter_rh", "gray_matter", (42,), "cortex, right"),
    _s("white_matter_lh", "white_matter", (2,), "cerebral white matter, left"),
    _s("white_matter_rh", "white_matter", (41,), "cerebral white matter, right"),
])

def _resolve_overlaps(wanted: list[str]) -> list[str]:
    return sorted(wanted, key=lambda n: (len(STRUCTURES[n].ids), n))


def available(run_dir: Path) -> dict[str, bool]:
    """Which structures this run actually has source labels for."""
    have = {"aseg": (run_dir / ASEG_FILE).exists(),
            "brainstem": (run_dir / BRAINSTEM_FILE).exists()}
    return {name: have.get(s.volume, False) for name, s in STRUCTURES.items()}


def copy_sources(fs_subject_dir: Path, run_dir: Path) -> dict[str, str]:
    """Copy the label volumes these structures need out of a recon-all subject."""
    import nibabel as nib

    from engine.paths import assert_write_path

    written: dict[str, str] = {}
    for out_name, sources in ((ASEG_FILE, _ASEG_SOURCES),
                              (BRAINSTEM_FILE, _BRAINSTEM_SOURCES)):
        for source_name in sources:
            source = fs_subject_dir / "mri" / source_name
            if not source.exists():
                continue
            nib.save(nib.as_closest_canonical(nib.load(str(source))),
                     str(assert_write_path(run_dir / out_name)))
            written[out_name] = source_name
            break
    if ASEG_FILE not in written:
        logger.info("no aseg in %s; the optional structures will be unavailable",
                    fs_subject_dir)
    if BRAINSTEM_FILE not in written:
        logger.info("no brainstemSsLabels in %s; midbrain/pons/medulla are "
                    "unavailable (run FreeSurfer's segmentBS.sh to get them)",
                    fs_subject_dir)
    return written


def _assign(nodes: np.ndarray, hexes: np.ndarray, volume: np.ndarray,
            affine: np.ndarray, ids_by_name: dict[str, tuple[int, ...]],
            reach_mm: float) -> dict[str, np.ndarray]:
    """Which of `hexes` belong to each named structure, by centroid."""
    from scipy import ndimage

    inv = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    centroids = nodes[hexes].mean(axis=1)
    vox = centroids @ inv[:3, :3].T + inv[:3, 3]
    idx = np.clip(np.floor(vox + 0.5).astype(np.int64), 0, np.array(volume.shape) - 1)

    wanted_ids = sorted({i for ids in ids_by_name.values() for i in ids})
    known = np.isin(volume, wanted_ids)
    if not known.any():
        return {}
    distance, nearest = ndimage.distance_transform_edt(
        ~known, sampling=spacing, return_indices=True)
    filled = volume[nearest[0], nearest[1], nearest[2]]

    at = filled[idx[:, 0], idx[:, 1], idx[:, 2]]
    within = distance[idx[:, 0], idx[:, 1], idx[:, 2]] <= reach_mm

    out: dict[str, np.ndarray] = {}
    claimed = np.zeros(len(hexes), dtype=bool)
    for name in _resolve_overlaps(list(ids_by_name)):
        rows = np.nonzero(within & ~claimed & np.isin(at, list(ids_by_name[name])))[0]
        if len(rows):
            claimed[rows] = True
            out[name] = rows
    if out:
        _report_pieces(out, hexes)
    return out


def _report_pieces(rows_by_name: dict[str, np.ndarray], hexes: np.ndarray) -> None:
    """Log any structure that arrived in more than one connected element set."""
    from scipy import sparse
    from scipy.sparse import csgraph

    from engine.anatomy.parcellation import _hex_adjacency

    left, right = _hex_adjacency(hexes)
    if not len(left):
        return
    owner = np.full(len(hexes), -1, dtype=np.int64)
    for i, rows in enumerate(rows_by_name.values()):
        owner[rows] = i
    sel = (owner[left] == owner[right]) & (owner[left] >= 0)
    graph = sparse.coo_matrix((np.ones(int(sel.sum()), dtype=np.int8),
                               (left[sel], right[sel])), shape=(len(hexes),) * 2)
    _n, piece = csgraph.connected_components(graph, directed=False)
    for name, rows in rows_by_name.items():
        pieces = len(np.unique(piece[rows])) if len(rows) else 0
        if pieces > 1:
            logger.info("substructures: %s arrived in %d pieces, all inside its own atlas "
                        "territory; kept as they are", name, pieces)


_RESIDUE_MIN_ELEMENTS = 100


def _fold_residue(nodes: np.ndarray, block: np.ndarray,
                  parts: dict[str, np.ndarray], residue: np.ndarray) -> None:
    """Give each leftover element to the carved structure nearest it."""
    from scipy.spatial import cKDTree

    centroids = nodes[block].mean(axis=1)
    owner = np.concatenate([np.full(len(rows), i) for i, rows in enumerate(parts.values())])
    claimed_rows = np.concatenate(list(parts.values()))
    tree = cKDTree(centroids[claimed_rows])
    _d, nearest = tree.query(centroids[residue])
    for i, name in enumerate(parts):
        take = residue[owner[nearest] == i]
        if len(take):
            parts[name] = np.concatenate([parts[name], take])


def split_structures(
    solid_blocks: dict, nodes: np.ndarray, run_dir: Path,
    wanted: tuple[str, ...], reach_mm: float = 1.5,
) -> tuple[dict, dict]:
    """Carve the requested structures out of their parent tissue parts."""
    import nibabel as nib

    unknown = [n for n in wanted if n not in STRUCTURES]
    if unknown:
        raise ValueError(f"unknown structures {unknown}; choose from {sorted(STRUCTURES)}")
    if not wanted:
        return solid_blocks, {"status": "skipped", "reason": "none requested"}

    out = dict(solid_blocks)
    report: dict = {"status": "ok", "parts": {}, "unavailable": {}}

    groups: dict[tuple[str, str], list[str]] = {}
    for name in wanted:
        structure = STRUCTURES[name]
        groups.setdefault((structure.volume, structure.source), []).append(name)

    volumes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for volume_key, filename in (("aseg", ASEG_FILE), ("brainstem", BRAINSTEM_FILE)):
        path = run_dir / filename
        if path.exists():
            img = nib.load(str(path))
            volumes[volume_key] = (np.asarray(img.dataobj).astype(np.int32), img.affine)

    for (volume_key, source), names in sorted(groups.items()):
        if volume_key not in volumes:
            for name in names:
                report["unavailable"][name] = f"no {volume_key} label volume in this run"
            continue
        block = out.get(source)
        if block is None or not len(block):
            for name in names:
                report["unavailable"][name] = f"the {source} part is not in this mesh"
            continue
        volume, volume_affine = volumes[volume_key]
        parts = _assign(nodes, block, volume, volume_affine,
                        {n: STRUCTURES[n].ids for n in names}, reach_mm)
        for name in names:
            if name not in parts:
                report["unavailable"][name] = "not present in this subject's labels"
        if not parts:
            continue

        claimed = np.zeros(len(block), dtype=bool)
        for rows in parts.values():
            claimed[rows] = True
        residue = np.flatnonzero(~claimed)
        if len(residue) and len(residue) < _RESIDUE_MIN_ELEMENTS:
            _fold_residue(nodes, block, parts, residue)
            report.setdefault("residue_folded", {})[source] = int(len(residue))
            logger.info("substructures: %s had %d elements left over, under the %d "
                        "floor; folded into the nearest carved structure rather than "
                        "shipped as a part", source, len(residue), _RESIDUE_MIN_ELEMENTS)
            claimed[:] = True
        if claimed.all():
            del out[source]
        else:
            out[source] = block[~claimed]
        for name, rows in parts.items():
            out[name] = block[rows]
            report["parts"][name] = int(len(rows))
        logger.info("substructures: %s -> %s (%d of %d hexes claimed)", source,
                    sorted(parts), int(claimed.sum()), len(block))

    if report["unavailable"]:
        logger.warning("substructures unavailable: %s", report["unavailable"])
    return out, report


def demo() -> None:
    """Self-check on a synthetic block"""
    import tempfile

    import nibabel as nib

    volume = np.zeros((16, 16, 16), dtype=np.int32)
    volume[4:8, 4:12, 4:12] = 10  # left thalamus
    volume[9:12, 4:12, 4:12] = 11  # left caudate
    affine = np.eye(4)

    corners = []
    hexes = []
    index: dict[tuple[int, int, int], int] = {}

    def node(i, j, k):
        key = (i, j, k)
        if key not in index:
            index[key] = len(corners)
            corners.append([i - 0.5, j - 0.5, k - 0.5])
        return index[key]

    for i in range(4, 12):
        for j in range(4, 12):
            for k in range(4, 12):
                hexes.append([node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k),
                              node(i, j + 1, k), node(i, j, k + 1), node(i + 1, j, k + 1),
                              node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)])
    nodes = np.array(corners, dtype=float)
    blocks = {"deep_nuclei": np.array(hexes, dtype=np.int64)}
    total = len(blocks["deep_nuclei"])

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        nib.save(nib.Nifti1Image(volume, affine), str(run_dir / ASEG_FILE))
        out, report = split_structures(blocks, nodes, run_dir, ("thalamus",), reach_mm=0.6)

    assert "thalamus" in out, report
    assert "deep_nuclei" in out, "the unclaimed remainder was lost"
    assert len(out["thalamus"]) + len(out["deep_nuclei"]) == total, "elements went missing"
    assert len(out["thalamus"]) == 4 * 8 * 8, len(out["thalamus"])
    small = {"deep_nuclei": np.array(hexes, dtype=np.int64)[:260]}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        nib.save(nib.Nifti1Image(volume, affine), str(run_dir / ASEG_FILE))
        out2, report2 = split_structures(small, nodes, run_dir, ("thalamus",), reach_mm=0.6)
    assert "deep_nuclei" not in out2, "residue under the floor still shipped as a part"
    assert len(out2["thalamus"]) == 260, "elements went missing in the fold"
    assert report2["residue_folded"]["deep_nuclei"] > 0, report2

    print(f"substructures.demo OK: thalamus {len(out['thalamus'])} hexes, "
          f"deep_nuclei keeps {len(out['deep_nuclei'])}; "
          f"{report2['residue_folded']['deep_nuclei']} residue elements folded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
