"""The arterial tree, from the cerebral artery atlas warped onto this subject."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from engine.paths import ATLAS_DIR

logger = logging.getLogger(__name__)

OCCURRENCE_FILE = "vessel_occurrence_probability.nii.gz"
RADIUS_FILE = "vessel_radius_mean.nii.gz"

ATLAS_TO_ICBM_SHIFT_MM = (0.0, -2.0, 21.0)

_SEED_REFERENCE_THRESHOLD = 10.0

DEFAULT_OCCURRENCE_THRESHOLD = 15.0

_SPUR_MM = 4.0

_SEED_CLAIM_MM = 12.0
_BRANCH_CLAIM_MM = 6.0

_COW_SEEDS: dict[str, tuple[float, float, float]] = {
    "artery_ica_lh":       (-17.0,   1.0, -12.0),
    "artery_ica_rh":       ( 17.0,   1.0, -12.0),
    "artery_mca_m1_lh":    (-26.0,   2.0,  -6.0),
    "artery_mca_m1_rh":    ( 26.0,   2.0,  -6.0),
    "artery_mca_m2_lh":    (-36.0,   4.0,   4.0),
    "artery_mca_m2_rh":    ( 36.0,   4.0,   4.0),
    "artery_aca_a1_lh":    ( -8.0,  12.0,  -6.0),
    "artery_aca_a1_rh":    (  8.0,  12.0,  -6.0),
    "artery_aca_a2_lh":    ( -3.0,  28.0,  -2.0),
    "artery_aca_a2_rh":    (  3.0,  28.0,  -2.0),
    "artery_pericallosal_lh": (-4.0,  22.0,  18.0),
    "artery_pericallosal_rh": ( 4.0,  22.0,  18.0),
    "artery_acomm":        (  0.0,  17.0,  -6.0),
    "artery_pcomm_lh":     (-10.0,  -8.0, -12.0),
    "artery_pcomm_rh":     ( 10.0,  -8.0, -12.0),
    "artery_choroidal_ant_lh": (-26.0, -8.0, -14.0),
    "artery_choroidal_ant_rh": ( 26.0, -8.0, -14.0),
    "artery_basilar":      (  0.0, -26.0, -30.0),
    "artery_pca_p1_lh":    (-10.0, -22.0, -10.0),
    "artery_pca_p1_rh":    ( 10.0, -22.0, -10.0),
    "artery_pca_p2_lh":    (-28.0, -32.0,  -8.0),
    "artery_pca_p2_rh":    ( 28.0, -32.0,  -8.0),
    "artery_sca_lh":       (-12.0, -32.0, -24.0),
    "artery_sca_rh":       ( 12.0, -32.0, -24.0),
    "artery_aica_lh":      (-14.0, -38.0, -38.0),
    "artery_aica_rh":      ( 14.0, -38.0, -38.0),
    "artery_pica_lh":      (-10.0, -46.0, -44.0),
    "artery_pica_rh":      ( 10.0, -46.0, -44.0),
    "artery_vertebral_lh": ( -6.0, -40.0, -46.0),
    "artery_vertebral_rh": (  6.0, -40.0, -46.0),
}

_BRANCH_SEEDS: dict[str, tuple[float, float, float]] = {
    "artery_heubner_lh":          (-14.0,  18.0,  -6.0),
    "artery_heubner_rh":          ( 14.0,  18.0,  -6.0),
    "artery_orbitofrontal_lh":    (-10.0,  40.0, -16.0),
    "artery_orbitofrontal_rh":    ( 10.0,  40.0, -16.0),
    "artery_frontopolar_lh":      ( -8.0,  56.0,   0.0),
    "artery_frontopolar_rh":      (  8.0,  56.0,   0.0),
    "artery_callosomarginal_lh":  ( -8.0,  20.0,  30.0),
    "artery_callosomarginal_rh":  (  8.0,  20.0,  30.0),
    "artery_internal_frontal_lh": (-10.0,  30.0,  34.0),
    "artery_internal_frontal_rh": ( 10.0,  30.0,  34.0),
    "artery_paracentral_lh":      ( -8.0, -22.0,  52.0),
    "artery_paracentral_rh":      (  8.0, -22.0,  52.0),
    "artery_splenial_lh":         ( -6.0, -44.0,  18.0),
    "artery_splenial_rh":         (  6.0, -44.0,  18.0),
    "artery_prefrontal_lh":       (-38.0,  40.0,  16.0),
    "artery_prefrontal_rh":       ( 38.0,  40.0,  16.0),
    "artery_precentral_lh":       (-50.0,   4.0,  30.0),
    "artery_precentral_rh":       ( 50.0,   4.0,  30.0),
    "artery_central_lh":          (-54.0,  -8.0,  32.0),
    "artery_central_rh":          ( 54.0,  -8.0,  32.0),
    "artery_parietal_ant_lh":     (-52.0, -30.0,  40.0),
    "artery_parietal_ant_rh":     ( 52.0, -30.0,  40.0),
    "artery_parietal_post_lh":    (-44.0, -50.0,  40.0),
    "artery_parietal_post_rh":    ( 44.0, -50.0,  40.0),
    "artery_angular_lh":          (-44.0, -62.0,  30.0),
    "artery_angular_rh":          ( 44.0, -62.0,  30.0),
    "artery_temporopolar_lh":     (-36.0,  12.0, -30.0),
    "artery_temporopolar_rh":     ( 36.0,  12.0, -30.0),
    "artery_temporal_ant_lh":     (-52.0,   4.0, -24.0),
    "artery_temporal_ant_rh":     ( 52.0,   4.0, -24.0),
    "artery_temporal_mid_lh":     (-58.0, -20.0, -12.0),
    "artery_temporal_mid_rh":     ( 58.0, -20.0, -12.0),
    "artery_temporal_post_lh":    (-54.0, -46.0,  -4.0),
    "artery_temporal_post_rh":    ( 54.0, -46.0,  -4.0),
    "artery_calcarine_lh":        (-10.0, -80.0,   6.0),
    "artery_calcarine_rh":        ( 10.0, -80.0,   6.0),
    "artery_parieto_occipital_lh": (-10.0, -70.0, 20.0),
    "artery_parieto_occipital_rh": ( 10.0, -70.0, 20.0),
    "artery_temporal_inf_lh":     (-40.0, -50.0, -20.0),
    "artery_temporal_inf_rh":     ( 40.0, -50.0, -20.0),
    "artery_striate_lateral_lh":  (-20.0,   8.0,  -2.0),
    "artery_striate_lateral_rh":  ( 20.0,   8.0,  -2.0),
    "artery_lenticulostriate_lh": (-24.0,   4.0,   2.0),
    "artery_lenticulostriate_rh": ( 24.0,   4.0,   2.0),
    "artery_thalamoperforating_lh": (-4.0, -16.0,  -6.0),
    "artery_thalamoperforating_rh": ( 4.0, -16.0,  -6.0),
    "artery_thalamogeniculate_lh": (-14.0, -22.0, -4.0),
    "artery_thalamogeniculate_rh": ( 14.0, -22.0, -4.0),
    "artery_choroidal_post_med_lh": (-8.0, -28.0,   2.0),
    "artery_choroidal_post_med_rh": ( 8.0, -28.0,   2.0),
    "artery_choroidal_post_lat_lh": (-24.0, -32.0,  4.0),
    "artery_choroidal_post_lat_rh": ( 24.0, -32.0,  4.0),
    "artery_labyrinthine_lh":     (-24.0, -34.0, -34.0),
    "artery_labyrinthine_rh":     ( 24.0, -34.0, -34.0),
    "artery_pontine_lh":          ( -6.0, -28.0, -30.0),
    "artery_pontine_rh":          (  6.0, -28.0, -30.0),
    "artery_pontine_circumflex_lh": (-10.0, -30.0, -30.0),
    "artery_pontine_circumflex_rh": ( 10.0, -30.0, -30.0),
    "artery_circumferential_short_lh": (-6.0, -32.0, -36.0),
    "artery_circumferential_short_rh": ( 6.0, -32.0, -36.0),
    "artery_circumferential_long_lh": (-16.0, -32.0, -32.0),
    "artery_circumferential_long_rh": ( 16.0, -32.0, -32.0),
    "artery_medullary_perforating_lh": (-2.0, -36.0, -54.0),
    "artery_medullary_perforating_rh": ( 2.0, -36.0, -54.0),
    "artery_spinal_anterior":     (  0.0, -38.0, -58.0),
    "artery_spinal_posterior_lh": ( -4.0, -46.0, -56.0),
    "artery_spinal_posterior_rh": (  4.0, -46.0, -56.0),
}

_TRUNK_SEEDS = dict(_COW_SEEDS)
_COW_SEEDS.update(_BRANCH_SEEDS)

_COW_RING = (
    "artery_ica_lh", "artery_ica_rh", "artery_mca_m1_lh", "artery_mca_m1_rh",
    "artery_aca_a1_lh", "artery_aca_a1_rh", "artery_acomm",
    "artery_pcomm_lh", "artery_pcomm_rh", "artery_basilar",
    "artery_pca_p1_lh", "artery_pca_p1_rh",
)
_SEED_SNAP_MM = 10.0

ARTERY_CLASSES: dict[str, tuple[str, float, str]] = {
    name: ("vessel_artery", 3.0 if name in _TRUNK_SEEDS else 1.0,
           "Artery " + name.removeprefix("artery_").replace("_", " ").title())
    for name in _COW_SEEDS
}
ARTERY_CLASSES["artery_large"] = ("vessel_artery", 4.0, "Arteries Large")
ARTERY_CLASSES["artery_small"] = ("vessel_artery", 1.0, "Arteries Small")


def _territory_class(part: str) -> str:
    """The part name for a distal branch ending in the parcel `part`."""
    return "artery_" + part.strip().lower().replace(" ", "_")


def _register_territory(part: str) -> str:
    """Make sure a territory part has a class, and return its name."""
    name = _territory_class(part)
    if name not in ARTERY_CLASSES:
        title = "Artery " + part.replace("_", " ").title()
        ARTERY_CLASSES[name] = ("vessel_artery", 1.0, title)
    return name

_LARGE_RADIUS_MM = 1.0

LENGTH_BAND_MM = (250.0, 2000.0)


def _seed_marker_volume(shape, affine: np.ndarray, seeds: dict[str, np.ndarray],
                        radius_mm: float = 3.0) -> np.ndarray:
    """One small labelled ball per seed, on the atlas grid."""
    inv = np.linalg.inv(affine)
    spacing = float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))
    out = np.zeros(shape, dtype=np.uint8)
    reach = int(np.ceil(radius_mm / max(spacing, 1e-6)))
    for ident, (name, point) in enumerate(seeds.items(), start=1):
        vox = np.rint((inv @ np.append(np.asarray(point, dtype=float), 1.0))[:3]).astype(int)
        lo = np.maximum(vox - reach, 0)
        hi = np.minimum(vox + reach + 1, np.array(shape))
        if (lo >= hi).any():
            logger.warning("arteries: seed %s falls outside the atlas grid", name)
            continue
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = ident
    return out


def corrected_affine(affine: np.ndarray) -> np.ndarray:
    """The vessel atlas's affine, moved into ICBM152's frame."""
    out = np.array(affine, dtype=float, copy=True)
    out[:3, 3] += np.asarray(ATLAS_TO_ICBM_SHIFT_MM, dtype=float)
    return out


def _snap_seeds(occurrence: np.ndarray, affine: np.ndarray,
                threshold: float) -> tuple[dict[str, np.ndarray], dict]:
    """Move each landmark onto the nearest atlas vessel voxel, and say how far."""
    from scipy.spatial import cKDTree

    idx = np.argwhere(occurrence >= threshold)
    if not len(idx):
        return {}, {"seeds_kept": 0, "reason": "no voxel above threshold"}
    world = (affine @ np.c_[idx.astype(float), np.ones(len(idx))].T).T[:, :3]
    tree = cKDTree(world)

    snapped: dict[str, np.ndarray] = {}
    moved: dict[str, float] = {}
    for name, point in _COW_SEEDS.items():
        distance, where = tree.query(np.asarray(point, dtype=float))
        if distance > _SEED_SNAP_MM:
            logger.warning("arteries: seed %s is %.1fmm from any vessel voxel at %.0f%% "
                           "occurrence; dropped", name, distance, threshold)
            continue
        snapped[name] = world[where]
        moved[name] = round(float(distance), 2)
    return snapped, {"seeds_kept": len(snapped), "seed_snap_mm": moved}


def warp_vessel_atlas(t1_brain_path: Path, run_dir: Path,
                      threshold: float = DEFAULT_OCCURRENCE_THRESHOLD,
                      ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]:
    """Occurrence, radius and Circle of Willis markers in this subject's space."""
    import nibabel as nib

    occ_path = ATLAS_DIR / OCCURRENCE_FILE
    rad_path = ATLAS_DIR / RADIUS_FILE
    if not occ_path.exists() or not rad_path.exists():
        logger.warning("arteries: %s / %s not in %s; no arterial tree",
                       OCCURRENCE_FILE, RADIUS_FILE, ATLAS_DIR)
        return None

    run_dir = Path(run_dir)
    cache = {name: run_dir / f"vessels_{name}.nii.gz"
             for name in ("prob", "radius", "seeds")}
    names_file = run_dir / "vessels_seed_names.txt"
    if all(p.exists() for p in cache.values()) and names_file.exists():
        images = {name: nib.load(str(p)) for name, p in cache.items()}
        logger.info("arteries: reusing the warped vessel atlas in %s", run_dir)
        seed_names = names_file.read_text().split()
        return (np.asarray(images["prob"].dataobj, dtype=np.float32),
                np.asarray(images["radius"].dataobj, dtype=np.float32),
                np.rint(np.asarray(images["seeds"].dataobj)).astype(np.int16),
                images["prob"].affine,
                {"warped": "cached", "seed_names": seed_names})

    try:
        import ants
        from engine.anatomy.atlas_segment import _affine_of, _register_atlas_to
    except ImportError:
        logger.warning("arteries: antspyx is not installed; no arterial tree")
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    from engine.engines import atlasseg
    from engine.engines.brainseg import _load_canonical, _nib_to_ants

    shipped = atlasseg.TEMPLATE_DIR / "mri"
    xfm = atlasseg.template_transforms(run_dir / "recon")
    if xfm and all((shipped / f"vessels_{n}.mgz").is_file() for n in ("prob", "radius", "seeds")):
        # already on fsaverage: one warp through the T1 run's own template registration
        fixed = ants.image_read(str(t1_brain_path))
        for name, nearest in (("prob", False), ("radius", False), ("seeds", True)):
            img = _load_canonical(shipped / f"vessels_{name}.mgz")
            moving = _nib_to_ants(img, np.asarray(img.dataobj).astype(np.float32))
            array = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=xfm["fwdtransforms"],
                                          interpolator="nearestNeighbor" if nearest else "linear").numpy()
            nib.save(nib.Nifti1Image(array.astype(np.float32), _affine_of(fixed)), str(cache[name]))
        seed_names = (shipped / "vessels_seed_names.txt").read_text().split()
        names_file.write_text("\n".join(seed_names))
        images = {name: nib.load(str(p)) for name, p in cache.items()}
        return (np.asarray(images["prob"].dataobj, dtype=np.float32),
                np.asarray(images["radius"].dataobj, dtype=np.float32),
                np.rint(np.asarray(images["seeds"].dataobj)).astype(np.int16),
                images["prob"].affine,
                {"warped": "fsaverage -> subject (the T1 run's template registration)",
                 "seed_names": seed_names})

    occ_img = nib.load(str(occ_path))
    atlas_affine = corrected_affine(occ_img.affine)
    occurrence_atlas = np.asarray(occ_img.dataobj, dtype=np.float32)
    seeds, seed_report = _snap_seeds(occurrence_atlas, atlas_affine,
                                     _SEED_REFERENCE_THRESHOLD)
    markers = _seed_marker_volume(occ_img.shape, atlas_affine, seeds)

    marker_path = run_dir / "vessels_seeds_mni.nii.gz"
    nib.save(nib.Nifti1Image(markers, atlas_affine), str(marker_path))
    sources = {"seeds": marker_path}
    for name, path in (("prob", occ_path), ("radius", rad_path)):
        shifted = run_dir / f"vessels_{name}_icbm.nii.gz"
        image = nib.load(str(path))
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj, dtype=np.float32),
                                 corrected_affine(image.affine)), str(shifted))
        sources[name] = shifted

    fixed, reg = _register_atlas_to(Path(t1_brain_path))
    warped = {}
    for name, nearest in (("prob", False), ("radius", False), ("seeds", True)):
        moving = ants.image_read(str(sources[name]))
        image = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor" if nearest else "linear")
        warped[name] = image.numpy()
    affine = _affine_of(fixed)
    for name, array in warped.items():
        nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(cache[name]))

    seed_names = list(seeds)
    names_file.write_text("\n".join(seed_names))
    report = dict(seed_report)
    report["seed_names"] = seed_names
    report["registration"] = "ICBM152 2009c -> subject, ANTs SyN"
    report["atlas_shift_mm"] = list(ATLAS_TO_ICBM_SHIFT_MM)
    snaps = list(seed_report.get("seed_snap_mm", {}).values())
    report["seed_snap_median_mm"] = round(float(np.median(snaps)), 2) if snaps else None
    found = set(seed_names)
    trunks_found = sorted(found & set(_TRUNK_SEEDS))
    report["trunk_seeds_found"] = len(trunks_found)
    report["trunk_seeds_total"] = len(_TRUNK_SEEDS)
    report["branch_seeds_found"] = len(found & set(_BRANCH_SEEDS))
    report["branch_seeds_total"] = len(_BRANCH_SEEDS)
    report["seeds_dropped"] = sorted(set(_COW_SEEDS) - found)
    report["frame_ok"] = bool(snaps and np.median(snaps) <= _SEED_SNAP_MM
                              and len(trunks_found) >= len(_TRUNK_SEEDS) - 2)
    if not report["frame_ok"]:
        logger.error("arteries: the vessel atlas does not sit on this frame "
                     "(%d/%d trunk seeds, median snap %s mm); naming is not trustworthy",
                     len(trunks_found), len(_TRUNK_SEEDS), report["seed_snap_median_mm"])
    logger.info("arteries: %d/%d trunk seeds and %d/%d branch seeds found in the atlas; "
                "%d seed(s) had no vessel within %.0fmm and were dropped",
                len(trunks_found), len(_TRUNK_SEEDS), report["branch_seeds_found"],
                len(_BRANCH_SEEDS), len(report["seeds_dropped"]), _SEED_SNAP_MM)
    logger.info("arteries: warped the vessel atlas onto the subject; %d seed(s) kept",
                len(seed_names))
    return (warped["prob"].astype(np.float32), warped["radius"].astype(np.float32),
            np.rint(warped["seeds"]).astype(np.int16), affine, report)


_NEIGHBOUR_KERNEL = np.ones((3, 3, 3), dtype=np.uint8)
_NEIGHBOUR_KERNEL[1, 1, 1] = 0


def skeleton_polylines(mask: np.ndarray) -> list[np.ndarray]:
    """Centerline voxel chains between the skeleton's own junctions and tips."""
    from skimage.morphology import skeletonize

    skel = skeletonize(np.asarray(mask, dtype=bool))
    if not skel.any():
        return []
    degree = ndimage.convolve(skel.astype(np.uint8), _NEIGHBOUR_KERNEL,
                              mode="constant", cval=0) * skel
    voxels = np.argwhere(skel)
    index = {tuple(int(x) for x in v): i for i, v in enumerate(voxels)}
    deg = degree[skel]

    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]
    shape = skel.shape

    def neighbours(i: int) -> list[int]:
        v = voxels[i]
        out = []
        for d in offsets:
            w = (int(v[0]) + d[0], int(v[1]) + d[1], int(v[2]) + d[2])
            if 0 <= w[0] < shape[0] and 0 <= w[1] < shape[1] and 0 <= w[2] < shape[2]:
                j = index.get(w)
                if j is not None:
                    out.append(j)
        return out

    used: set[tuple[int, int]] = set()
    chains: list[np.ndarray] = []

    def walk(start: int, first: int) -> None:
        chain = [start, first]
        used.add((min(start, first), max(start, first)))
        while deg[chain[-1]] == 2:
            nxt = [j for j in neighbours(chain[-1])
                   if (min(chain[-1], j), max(chain[-1], j)) not in used]
            if not nxt:
                break
            used.add((min(chain[-1], nxt[0]), max(chain[-1], nxt[0])))
            chain.append(nxt[0])
        chains.append(voxels[chain])

    for start in [i for i in range(len(voxels)) if deg[i] != 2]:
        for first in neighbours(start):
            if (min(start, first), max(start, first)) in used:
                continue
            walk(start, first)

    covered = {index[tuple(int(x) for x in v)] for chain in chains for v in chain}
    remaining = {i for i in range(len(voxels)) if deg[i] == 2} - covered
    while remaining:
        start = remaining.pop()
        nb = neighbours(start)
        if not nb:
            continue
        walk(start, nb[0])
        for v in chains[-1]:
            remaining.discard(index[tuple(int(x) for x in v)])
    return [c for c in chains if len(c) >= 2]


def _world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (affine @ np.c_[points, np.ones(len(points))].T).T[:, :3]


def _prune_spurs(chains: list[np.ndarray], affine: np.ndarray,
                 min_mm: float = _SPUR_MM) -> list[np.ndarray]:
    """Drop the short stubs skeletonization throws off a thick trunk."""
    if not chains:
        return chains
    ends: dict[tuple[int, int, int], int] = {}
    for chain in chains:
        for tip in (chain[0], chain[-1]):
            key = tuple(int(x) for x in tip)
            ends[key] = ends.get(key, 0) + 1
    kept = []
    for chain in chains:
        free = sum(1 for tip in (chain[0], chain[-1])
                   if ends[tuple(int(x) for x in tip)] == 1)
        length = float(np.linalg.norm(np.diff(_world(chain, affine), axis=0),
                                      axis=1).sum())
        if free and length < min_mm:
            continue
        kept.append(chain)
    return kept


_TERRITORY_CLAIM_MM = 10.0


def _name_chain(midpoint: np.ndarray, seed_world: dict[str, np.ndarray],
                radius_mm: float, distal: np.ndarray | None = None,
                territories=None) -> str:
    """Which artery a centerline is."""
    best, best_score = None, np.inf
    for name, point in seed_world.items():
        distance = float(np.linalg.norm(np.asarray(midpoint, dtype=float) - point))
        reach = _SEED_CLAIM_MM if name in _TRUNK_SEEDS else _BRANCH_CLAIM_MM
        if distance > reach:
            continue
        score = distance / reach
        if score < best_score:
            best, best_score = name, score
    if best is not None:
        return best
    if territories is not None and distal is not None:
        tree, names = territories
        distance, where = tree.query(np.asarray(distal, dtype=float))
        if float(distance) <= _TERRITORY_CLAIM_MM:
            return _register_territory(names[int(where)])
    return "artery_large" if radius_mm >= _LARGE_RADIUS_MM else "artery_small"


def _territory_index(territories: Optional[dict[str, np.ndarray]]):
    """`(kd-tree, name-per-point)` over the parcels a branch can end in."""
    if not territories:
        return None
    from scipy.spatial import cKDTree

    points, names = [], []
    for part, pts in territories.items():
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if not len(pts):
            continue
        points.append(pts)
        names.extend([part] * len(pts))
    if not points:
        return None
    return cKDTree(np.concatenate(points)), names


def occurrence_sweep(occurrence: np.ndarray, affine: np.ndarray,
                     thresholds=(10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0),
                     seed_world: Optional[dict[str, np.ndarray]] = None) -> dict:
    """Centerline length and Circle of Willis census against threshold."""
    out = {}
    for value in thresholds:
        mask = occurrence >= value
        if not mask.any():
            out[str(value)] = {"segments": 0, "length_mm": 0.0}
            continue
        chains = _prune_spurs(skeleton_polylines(mask), affine)
        length = sum(float(np.linalg.norm(np.diff(_world(c, affine), axis=0), axis=1).sum())
                     for c in chains)
        entry = {"segments": len(chains), "length_mm": round(length, 1)}
        if seed_world:
            named = {_name_chain(_world(c, affine)[len(c) // 2], seed_world, 0.0)
                     for c in chains}
            entry["cow"] = sum(1 for name in _COW_RING if name in named)
            entry["cow_total"] = len(_COW_RING)
        out[str(value)] = entry
    return out


def _seed_world_positions(markers: np.ndarray, affine: np.ndarray,
                          names: list[str]) -> dict[str, np.ndarray]:
    """Centroid of each warped seed ball, in subject world mm."""
    out: dict[str, np.ndarray] = {}
    for ident, name in enumerate(names, start=1):
        where = np.argwhere(markers == ident)
        if len(where):
            out[name] = _world(where, affine).mean(axis=0)
    return out


def _mask_on_grid(mask: np.ndarray, mask_affine: np.ndarray,
                  shape: tuple[int, ...], affine: np.ndarray) -> np.ndarray:
    """Carry a mask from its own grid onto the vessel grid, nearest neighbour."""
    grid = np.indices(shape).reshape(3, -1).astype(float)
    world = (affine @ np.vstack([grid, np.ones(grid.shape[1])]))[:3]
    voxel = np.rint(np.linalg.inv(mask_affine)
                    @ np.vstack([world, np.ones(world.shape[1])]))[:3].astype(np.int64)
    inside = np.ones(voxel.shape[1], dtype=bool)
    for axis in range(3):
        inside &= (voxel[axis] >= 0) & (voxel[axis] < mask.shape[axis])
    out = np.zeros(voxel.shape[1], dtype=bool)
    out[inside] = np.asarray(mask, dtype=bool)[voxel[0, inside], voxel[1, inside],
                                               voxel[2, inside]]
    return out.reshape(shape)


def build_arterial_tree(
    t1_brain_path: Path,
    run_dir: Path,
    occurrence_threshold: float = DEFAULT_OCCURRENCE_THRESHOLD,
    spacing_mm: float = 3.0,
    envelope: Optional[tuple[np.ndarray, np.ndarray]] = None,
    territories: Optional[dict[str, np.ndarray]] = None,
    sweep: bool = True,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Named arterial beams for this subject."""
    from engine.mesh.vascular import _resample

    warped = warp_vessel_atlas(Path(t1_brain_path), Path(run_dir), occurrence_threshold)
    if warped is None:
        return {}, {"status": "no atlas"}
    occurrence, radius, markers, affine, report = warped

    mask = occurrence >= occurrence_threshold
    report["mask_voxels_raw"] = int(mask.sum())
    if envelope is not None:
        mask &= _mask_on_grid(envelope[0], envelope[1], mask.shape, affine)
    report["mask_voxels"] = int(mask.sum())
    report["occurrence_threshold"] = float(occurrence_threshold)
    if not mask.any():
        return {}, dict(report, status=f"nothing above {occurrence_threshold}%")

    chains = _prune_spurs(skeleton_polylines(mask), affine)
    report["segments_found"] = len(chains)

    seed_names = list(report.get("seed_names") or _COW_SEEDS)
    seed_world = _seed_world_positions(markers, affine, seed_names)

    territory_index = _territory_index(territories)
    if sweep:
        report["occurrence_sweep"] = occurrence_sweep(occurrence, affine,
                                                      seed_world=seed_world)

    grouped: dict[str, list[np.ndarray]] = {}
    radii: dict[str, list[float]] = {}
    for chain in chains:
        points = _world(chain, affine)
        chain_radius = float(np.median(radius[tuple(chain.T)]))
        if seed_world:
            hub = np.mean(np.stack(list(seed_world.values())), axis=0)
            distal = points[-1] if (np.linalg.norm(points[-1] - hub)
                                    >= np.linalg.norm(points[0] - hub)) else points[0]
        else:
            distal = points[-1]
        name = _name_chain(points[len(points) // 2], seed_world, chain_radius,
                           distal=distal, territories=territory_index)
        sampled = _resample(points, spacing_mm)
        if len(sampled) < 2:
            continue
        grouped.setdefault(name, []).append(sampled)
        radii.setdefault(name, []).append(chain_radius)

    tree: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, polylines in grouped.items():
        points_list, segments = [], []
        offset = 0
        for line in polylines:
            points_list.append(line)
            segments.append(np.stack([np.arange(len(line) - 1),
                                      np.arange(1, len(line))], axis=1) + offset)
            offset += len(line)
        tree[name] = (np.concatenate(points_list),
                      np.concatenate(segments).astype(np.int64))

    report.update(arterial_report(tree, radii))
    report.update(side_check(tree))
    logger.info("arteries: %d part(s), %d beams, %.0fmm total",
                len(tree), report["beams"], report["length_mm"])
    return tree, report


def arterial_report(tree: dict[str, tuple[np.ndarray, np.ndarray]],
                    radii: Optional[dict[str, list[float]]] = None) -> dict:
    """Per-part lengths, the Circle of Willis census and the symmetry band."""
    parts: dict[str, dict] = {}
    for name, (points, segments) in tree.items():
        length = float(np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]],
                                      axis=1).sum())
        parts[name] = {"beams": int(len(segments)), "length_mm": round(length, 1)}
        if radii and radii.get(name):
            parts[name]["radius_mm"] = round(float(np.median(radii[name])), 2)

    total = sum(p["length_mm"] for p in parts.values())
    cow = {name: name in tree for name in _COW_RING}
    named_trunks = {name: name in tree for name in _TRUNK_SEEDS if name not in _COW_RING}
    named_branches = {name: name in tree for name in _BRANCH_SEEDS}
    unnamed = sum(parts.get(n, {}).get("length_mm", 0.0)
                  for n in ("artery_large", "artery_small"))
    asymmetry = {}
    for name in _COW_SEEDS:
        if not name.endswith("_lh"):
            continue
        right = name[:-3] + "_rh"
        left_mm = parts.get(name, {}).get("length_mm", 0.0)
        right_mm = parts.get(right, {}).get("length_mm", 0.0)
        if left_mm or right_mm:
            asymmetry[name[:-3]] = round(abs(left_mm - right_mm)
                                         / max(left_mm, right_mm, 1e-6), 2)
    return {
        "provenance": "atlas_warped",
        "parts": parts,
        "beams": int(sum(p["beams"] for p in parts.values())),
        "length_mm": round(total, 1),
        "length_in_band": bool(LENGTH_BAND_MM[0] <= total <= LENGTH_BAND_MM[1]),
        "circle_of_willis": cow,
        "circle_of_willis_complete": bool(all(cow.values())),
        "named_trunks": named_trunks,
        "named_branches": named_branches,
        "named_branches_found": int(sum(named_branches.values())),
        "named_branches_total": len(named_branches),
        "unnamed_length_mm": round(unnamed, 1),
        "unnamed_fraction": round(unnamed / total, 3) if total else 0.0,
        "lr_length_asymmetry": asymmetry,
    }


_MIDLINE_TOLERANCE_MM = 2.0


def side_check(tree: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Every `_lh` part left of the midline and every `_rh` part right of it."""
    wrong = {}
    for name, (points, _segments) in tree.items():
        centre = float(np.median(points[:, 0]))
        if abs(centre) <= _MIDLINE_TOLERANCE_MM:
            continue
        if (name.endswith("_lh") and centre > 0) or (name.endswith("_rh") and centre < 0):
            wrong[name] = round(centre, 1)
    return {"misnamed_sides": wrong, "sides_ok": not wrong}


def demo() -> None:
    """Self-check on a synthetic vessel field, where every answer is known."""
    volume = np.zeros((60, 40, 60), dtype=np.float32)
    for z in range(5, 20):
        volume[30, 20, z] = 100.0
    for i in range(15):
        volume[30 + i, 20, 20 + i] = 100.0
        volume[30 - i, 20, 20 + i] = 100.0
    volume[31, 21, 10] = 100.0

    affine = np.eye(4)
    chains = skeleton_polylines(volume >= 25.0)
    assert len(chains) >= 3, f"a Y is three segments, got {len(chains)}"
    pruned = _prune_spurs(chains, affine, min_mm=3.0)
    assert len(pruned) == 3, f"the spur should go and the three limbs stay: {len(pruned)}"

    short = [np.array([[0, 0, 0], [0, 0, 1]]), np.array([[0, 0, 1], [0, 1, 1]]),
             np.array([[0, 0, 0], [0, 1, 0]]), np.array([[0, 1, 0], [0, 1, 1]])]
    assert len(_prune_spurs(short, affine, min_mm=99.0)) == 4

    seeds = {"artery_ica_lh": np.array([-17.0, 1.0, -12.0])}
    assert _name_chain(np.array([-16.0, 1.0, -12.0]), seeds, 2.0) == "artery_ica_lh"
    assert _name_chain(np.array([60.0, 0.0, 0.0]), seeds, 2.0) == "artery_large"
    assert _name_chain(np.array([60.0, 0.0, 0.0]), seeds, 0.2) == "artery_small"

    grid = np.eye(4)
    grid[:3, 3] = [-20.0, -20.0, -20.0]
    markers = _seed_marker_volume((40, 40, 40), grid,
                                  {"artery_basilar": np.array([0.0, -5.0, -10.0])},
                                  radius_mm=2.0)
    where = _seed_world_positions(markers, grid, ["artery_basilar"])
    assert np.allclose(where["artery_basilar"], [0.0, -5.0, -10.0], atol=1.0), where

    tree = {"artery_ica_lh": (np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
                              np.array([[0, 1]]))}
    assert not side_check(tree)["sides_ok"]
    touching = {"artery_vertebral_lh": (np.array([[-3.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                                                  [1.0, 0.0, 0.0]]),
                                        np.array([[0, 1], [1, 2]]))}
    assert side_check(touching)["sides_ok"], side_check(touching)

    rep = arterial_report({"artery_basilar": (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0]]),
                                              np.array([[0, 1]]))})
    assert rep["length_mm"] == 30.0 and rep["circle_of_willis"]["artery_basilar"]
    assert not rep["circle_of_willis_complete"]
    print("arteries demo OK:", len(pruned), "segments,", rep["length_mm"], "mm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
