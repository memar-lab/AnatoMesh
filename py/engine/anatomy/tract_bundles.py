"""Census of the named white-matter bundles"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

logger = logging.getLogger(__name__)

_NEIGHBOURHOOD_MM = 6.0

_SINISTRA, _DEXTRA = 1000, 2000

_TRUNK = (16,)  # brainstem
_CEREBELLUM = (8, 47)
_CORPUS_CALLOSUM = (251, 252, 253, 254, 255)
_THALAMUS_ET_DIENCEPHALON = (10, 49, 28, 60, 250)
_BASAL_NUCLEI = (11, 12, 13, 50, 51, 52)

_FRONTAL = (3, 12, 14, 18, 19, 20, 24, 27, 28, 32)
_OCCIPITAL = (5, 11, 13, 21)
_TEMPORAL = (1, 7, 9, 15, 30, 33, 34)
_PRECENTRAL = (24, 17)  # precentral and paracentral
_SUPERIOR_TEMPORAL = (30, 34, 15)  # superior and transverse temporal, middle
_INFERIOR_FRONTAL = (18, 19, 20)  # pars opercularis / orbitalis / triangularis
_ORBITOFRONTAL = (12, 14)
_CINGULI_ANTERIOR = (2, 26)  # caudal and rostral anterior cingulate
_CINGULI_POSTERIOR = (10, 23)  # isthmus and posterior cingulate


def _side_of(side_name: int, *offsets) -> tuple:
    flat = [n for group in offsets for n in (group if isinstance(group, tuple) else (group,))]
    return tuple(side_name + n for n in flat)


def _lateral_tracts() -> dict:
    """The lateralised bundles, built once per hemisphere from one definition."""
    definitions = {
        "arcuate_fasciculus": (("sinister", "dexter"),
                                (_INFERIOR_FRONTAL, _SUPERIOR_TEMPORAL)),
        "uncinate_fasciculus": (("sinister", "dexter"),
                                 (_ORBITOFRONTAL, (33, 30, 9))),
        "inferior_longitudinal_fasciculus": (("sinister", "dexter"),
                                               (_OCCIPITAL, _TEMPORAL)),
        "inferior_fronto_occipital_fasciculus": (("sinister", "dexter"),
                                                  (_OCCIPITAL, _FRONTAL)),
        "cingulum": (("sinistrum", "dextrum"), (_CINGULI_ANTERIOR, _CINGULI_POSTERIOR)),
        "optic_radiation": (("sinistra", "dextra"), (_OCCIPITAL,)),
    }
    out = {}
    for entry_name, (kinds, label_sets) in definitions.items():
        for suffix, side_name in zip(kinds, (_SINISTRA, _DEXTRA)):
            grades = [_side_of(side_name, *g) for g in label_sets]
            if entry_name == "optic_radiation":
                grades.append((10,) if side_name == _SINISTRA else (49,))
            out[f"{entry_name}_{suffix}"] = tuple(grades)
    return out


EXPECTED_TRACTS: dict = {
    "corpus_callosum": (_CORPUS_CALLOSUM, (2,), (41,)),
    "tractus_corticospinalis_sinister": (_TRUNK, _side_of(_SINISTRA, _PRECENTRAL)),
    "tractus_corticospinalis_dexter": (_TRUNK, _side_of(_DEXTRA, _PRECENTRAL)),
    "fornix_sinister": ((17,), _THALAMUS_ET_DIENCEPHALON),
    "fornix_dexter": ((53,), _THALAMUS_ET_DIENCEPHALON),
    "pedunculus_cerebellaris": (_TRUNK, _CEREBELLUM),
    "capsula_interna_sinistra": (_BASAL_NUCLEI, (10,), _side_of(_SINISTRA, _FRONTAL)),
    "capsula_interna_dextra": (_BASAL_NUCLEI, (49,), _side_of(_DEXTRA, _FRONTAL)),
    **_lateral_tracts(),
}

_MEDULLA_ALBA = (2, 41, 77, 0, 24)


def _neighbour_labels(label_map: np.ndarray, spacing_mm: np.ndarray) -> np.ndarray:
    """For every voxel, the nearest non-white-matter label within `_NEIGHBOURHOOD_MM`."""
    to_mark = ~np.isin(label_map, _MEDULLA_ALBA)
    dist_map, indices = distance_transform_edt(
        ~to_mark, sampling=spacing_mm, return_indices=True)
    neighbour_map = label_map[tuple(indices)]
    neighbour_map[dist_map > _NEIGHBOURHOOD_MM] = 0
    return neighbour_map


def streamline_tracts(streamlines_world: list, label_map: np.ndarray, affine: np.ndarray,
                         *, tract_bundles: dict | None = None) -> list:
    """Which bundles each streamline joins, one set per streamline."""
    tract_bundles = tract_bundles or EXPECTED_TRACTS
    label_map = np.asarray(label_map)
    affine = np.asarray(affine, dtype=float)
    neighbour_map = _neighbour_labels(label_map, np.linalg.norm(affine[:3, :3], axis=0))
    inverse_idx = np.linalg.inv(affine)
    grid_shape = np.array(label_map.shape)

    out = []
    for point_set in streamlines_world:
        point_set = np.asarray(point_set, dtype=float)
        if len(point_set) < 2:
            out.append(set())
            continue
        voxel = np.rint((inverse_idx @ np.c_[point_set, np.ones(len(point_set))].T).T[:, :3]).astype(int)
        in_grid = np.all((voxel >= 0) & (voxel < grid_shape), axis=1)
        if not in_grid.any():
            out.append(set())
            continue
        index = tuple(voxel[in_grid].T)
        touched = set(np.unique(label_map[index]).tolist()) | set(np.unique(neighbour_map[index]).tolist())
        out.append({entry_name for entry_name, label_sets in tract_bundles.items()
                    if all(touched.intersection(g) for g in label_sets)})
    return out


def tract_census(streamlines_world: list, label_map: np.ndarray, affine: np.ndarray,
                        *, tract_bundles: dict | None = None) -> dict:
    """Match every streamline against `EXPECTED_TRACTS` and count the hits."""
    tract_bundles = tract_bundles or EXPECTED_TRACTS
    census = {entry_name: {"streamlines": 0, "length_mm_mean": 0.0} for entry_name in tract_bundles}
    lengths_by_tract = {entry_name: [] for entry_name in tract_bundles}
    sine_nomine = 0
    names_found = streamline_tracts(streamlines_world, label_map, affine, tract_bundles=tract_bundles)
    for point_set, touched in zip(streamlines_world, names_found):
        point_set = np.asarray(point_set, dtype=float)
        if len(point_set) < 2:
            continue
        if not touched:
            sine_nomine += 1
            continue
        length_mm = float(np.linalg.norm(np.diff(point_set, axis=0), axis=1).sum())
        for entry_name in touched:
            census[entry_name]["streamlines"] += 1
            lengths_by_tract[entry_name].append(length_mm)

    for entry_name, vals in lengths_by_tract.items():
        if vals:
            census[entry_name]["length_mm_mean"] = round(float(np.mean(vals)), 1)
    missing_tracts = sorted(n for n, v in census.items() if not v["streamlines"])
    census_report = {
        "tract_bundles": census,
        "missing_tracts": missing_tracts,
        "streamlines_sine_nomine": sine_nomine,
        "expected": len(census),
        "found": len(census) - len(missing_tracts),
        "passed": not missing_tracts,
    }
    if missing_tracts:
        logger.warning("tract_bundles: %d of %d expected bundles have no streamline: %s",
                       len(missing_tracts), len(census), ", ".join(missing_tracts))
    else:
        logger.info("tract_bundles: all %d expected bundles are present", len(census))
    return census_report


_RECOVERY_DILATION = 4  # voxels grown around a bundle's waypoint regions
_RECOVERY_SEEDS = 120_000  # more seeds than seed voxels, so they jitter
_RECOVERY_PERSISTENCE_FA = 0.45

_RECOVERY_STEPS = (
    (0.10, 70.0, 0.95),
    (0.10, 70.0, 0.0),
    (0.15, 60.0, 0.5),
)


def tract_seeds(entry_name: str, label_map: np.ndarray, *,
                     dilation: int = _RECOVERY_DILATION) -> np.ndarray:
    """Where to start tracking to find one named bundle."""
    region_mask = np.zeros(label_map.shape, dtype=bool)
    for label_sets in EXPECTED_TRACTS[entry_name]:
        region_mask |= np.isin(label_map, list(label_sets))
    return binary_dilation(region_mask, iterations=dilation)


def recover_missing(missing_tracts, e1: np.ndarray, fa: np.ndarray, affine: np.ndarray,
                      label_map: np.ndarray, label_map_affine: np.ndarray, *,
                      envelope: np.ndarray, min_length_mm: float = 20.0,
                      seed_count: int = _RECOVERY_SEEDS,
                      label_sets=_RECOVERY_STEPS) -> tuple[list, dict]:
    """Track each missing bundle again, seeded inside itself."""
    from engine.anatomy import tractography

    envelope = np.asarray(envelope, dtype=bool)
    recovered_n: list = []
    census_report: dict = {"seed_count": seed_count, "label_sets": [list(g) for g in label_sets],
                     "tract_bundles": {}}
    for entry_name in missing_tracts:
        seed_voxels = tract_seeds(entry_name, label_map)
        if label_map.shape != fa.shape or not np.allclose(label_map_affine, affine):
            seed_voxels = tractography.mask_on_grid(seed_voxels, label_map_affine, affine, fa.shape)
        seed_voxels = seed_voxels & envelope
        census_report["tract_bundles"][entry_name] = {"streamlines": 0, "label_sets": None}
        if not seed_voxels.any():
            continue
        only_these = {entry_name: EXPECTED_TRACTS[entry_name]}
        for fa_min, angle_deg, persistence in label_sets:
            streamlines = tractography.track_e1_field(
                e1, fa, affine, seed_count=seed_count, fa_min=fa_min,
                angle_max_deg=angle_deg, mask=envelope, seed_voxels=seed_voxels,
                persistence=persistence, persistence_fa=_RECOVERY_PERSISTENCE_FA)
            streamlines, _ = tractography.clip_to_envelope(
                streamlines, envelope, affine, min_length_mm=min_length_mm)
            names_found = streamline_tracts(streamlines, label_map, label_map_affine,
                                          tract_bundles=only_these)
            digna = [s for s, touched in zip(streamlines, names_found) if touched]
            logger.info("tract_bundles: re-seeded %s at fa>=%.2f, %.0f deg, persistence %.2f "
                        "and recovered %d streamlines", entry_name, fa_min, angle_deg,
                        persistence, len(digna))
            if digna:
                census_report["tract_bundles"][entry_name] = {
                    "streamlines": len(digna),
                    "label_sets": [fa_min, angle_deg, persistence]}
                recovered_n.extend(digna)
                break
    census_report["streamlines"] = len(recovered_n)
    return recovered_n, census_report


def demo() -> None:
    """Self-check on a synthetic label volume, with no subject or disk I/O."""
    label_map = np.full((40, 40, 40), 2, dtype=np.int32)  # left white matter
    label_map[20:, :, :] = 41  # right white matter
    label_map[18:22, :, :] = 253  # a callosal slab at the midline
    label_map[:6, :, :] = 1003  # left frontal cortex
    label_map[34:, :, :] = 2003  # right frontal cortex
    affine = np.eye(4)

    commissure = np.stack([np.linspace(4, 36, 33), np.full(33, 20.0), np.full(33, 20.0)], axis=1)
    short_tract = np.stack([np.linspace(8, 14, 7), np.full(7, 20.0), np.full(7, 20.0)], axis=1)

    census_report = tract_census([commissure, short_tract], label_map, affine)
    assert census_report["tract_bundles"]["corpus_callosum"]["streamlines"] == 1, census_report["tract_bundles"]
    assert census_report["streamlines_sine_nomine"] == 1, census_report
    assert "fornix_sinister" in census_report["missing_tracts"]
    assert census_report["found"] == 1 and census_report["expected"] == len(EXPECTED_TRACTS)

    timid = np.stack([np.linspace(8, 32, 25), np.full(25, 20.0), np.full(25, 20.0)], axis=1)
    faint = tract_census([timid], label_map, affine)
    assert faint["tract_bundles"]["corpus_callosum"]["streamlines"] == 1, faint["tract_bundles"]
    print("tract_bundles.demo ok:", census_report["tract_bundles"]["corpus_callosum"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
