"""Split the brain into named atlas regions, as separate parts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMES: dict[str, str] = {
    "aparc": "parcellation_aparc+aseg.nii.gz",
    "a2009s": "parcellation_a2009s+aseg.nii.gz",
    "harvard_oxford": "parcellation.nii.gz",
}

_LEGACY_A2009S_NAME = "parcellation_aparc.nii.gz"

DK_NAMES = [
    "unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
    "corpuscallosum", "cuneus", "entorhinal", "fusiform", "inferiorparietal",
    "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal",
    "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis",
    "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate",
    "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
    "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal",
    "frontalpole", "temporalpole", "transversetemporal", "insula",
]

A2009S_NAMES = [
    "Unknown",
    "G_and_S_frontomargin", "G_and_S_occipital_inf", "G_and_S_paracentral",
    "G_and_S_subcentral", "G_and_S_transv_frontopol", "G_and_S_cingul-Ant",
    "G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post", "G_cingul-Post-dorsal",
    "G_cingul-Post-ventral", "G_cuneus", "G_front_inf-Opercular",
    "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle",
    "G_front_sup", "G_Ins_lg_and_S_cent_ins", "G_insular_short",
    "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor",
    "G_oc-temp_med-Lingual", "G_oc-temp_med-Parahip", "G_orbital",
    "G_pariet_inf-Angular", "G_pariet_inf-Supramar", "G_parietal_sup",
    "G_postcentral", "G_precentral", "G_precuneus", "G_rectus", "G_subcallosal",
    "G_temp_sup-G_T_transv", "G_temp_sup-Lateral", "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo", "G_temporal_inf", "G_temporal_middle",
    "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "Lat_Fis-post",
    # 11142 / 12142 in FreeSurferColorLUT; aparc.a2009s never paints it, but leaving it out
    # named every label from 11143 on after its neighbour (Pole_occipital came out as
    # Pole_temporal, ..., S_temporal_sup as S_temporal_transverse; 11175 got no name)
    "Medial_wall",
    "Pole_occipital", "Pole_temporal", "S_calcarine", "S_central",
    "S_cingul-Marginalis", "S_circular_insula_ant", "S_circular_insula_inf",
    "S_circular_insula_sup", "S_collat_transv_ant", "S_collat_transv_post",
    "S_front_inf", "S_front_middle", "S_front_sup", "S_interm_prim-Jensen",
    "S_intrapariet_and_P_trans", "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal", "S_occipital_ant", "S_oc-temp_lat",
    "S_oc-temp_med_and_Lingual", "S_orbital_lateral", "S_orbital_med-olfact",
    "S_orbital-H_Shaped", "S_parieto_occipital", "S_pericallosal",
    "S_postcentral", "S_precentral-inf-part", "S_precentral-sup-part",
    "S_suborbital", "S_subparietal", "S_temporal_inf", "S_temporal_sup",
    "S_temporal_transverse",
]

LOBES: dict[str, tuple[str, ...]] = {
    "frontal": ("caudalmiddlefrontal", "lateralorbitofrontal", "medialorbitofrontal",
                "parsopercularis", "parsorbitalis", "parstriangularis", "precentral",
                "rostralmiddlefrontal", "superiorfrontal", "frontalpole", "paracentral"),
    "parietal": ("inferiorparietal", "postcentral", "precuneus", "superiorparietal",
                 "supramarginal"),
    "temporal": ("bankssts", "entorhinal", "fusiform", "inferiortemporal",
                 "middletemporal", "parahippocampal", "superiortemporal",
                 "temporalpole", "transversetemporal"),
    "occipital": ("cuneus", "lateraloccipital", "lingual", "pericalcarine"),
    "cingulate": ("caudalanteriorcingulate", "isthmuscingulate", "posteriorcingulate",
                  "rostralanteriorcingulate"),
    "insula": ("insula",),
}
_LOBE_OF = {region: lobe for lobe, regions in LOBES.items() for region in regions}

A2009S_LOBES: dict[str, tuple[str, ...]] = {
    "frontal": (
        "G_and_S_frontomargin", "G_and_S_transv_frontopol", "G_and_S_paracentral",
        "G_and_S_subcentral", "G_front_inf-Opercular", "G_front_inf-Orbital",
        "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "G_orbital",
        "G_precentral", "G_rectus", "G_subcallosal", "Lat_Fis-ant-Horizont",
        "Lat_Fis-ant-Vertical", "S_central", "S_front_inf", "S_front_middle",
        "S_front_sup", "S_orbital_lateral", "S_orbital_med-olfact",
        "S_orbital-H_Shaped", "S_precentral-inf-part", "S_precentral-sup-part",
        "S_suborbital",
    ),
    "parietal": (
        "G_pariet_inf-Angular", "G_pariet_inf-Supramar", "G_parietal_sup",
        "G_postcentral", "G_precuneus", "S_cingul-Marginalis",
        "S_interm_prim-Jensen", "S_intrapariet_and_P_trans", "S_parieto_occipital",
        "S_postcentral", "S_subparietal",
    ),
    "temporal": (
        "G_oc-temp_lat-fusifor", "G_oc-temp_med-Parahip", "G_temp_sup-G_T_transv",
        "G_temp_sup-Lateral", "G_temp_sup-Plan_polar", "G_temp_sup-Plan_tempo",
        "G_temporal_inf", "G_temporal_middle", "Lat_Fis-post", "Pole_temporal",
        "S_collat_transv_ant", "S_collat_transv_post", "S_oc-temp_lat",
        "S_temporal_inf", "S_temporal_sup", "S_temporal_transverse",
    ),
    "occipital": (
        "G_and_S_occipital_inf", "G_cuneus", "G_occipital_middle",
        "G_occipital_sup", "G_oc-temp_med-Lingual", "Pole_occipital",
        "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal",
        "S_occipital_ant", "S_oc-temp_med_and_Lingual",
    ),
    "cingulate": (
        "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post",
        "G_cingul-Post-dorsal", "G_cingul-Post-ventral", "S_pericallosal",
    ),
    "insula": (
        "G_Ins_lg_and_S_cent_ins", "G_insular_short", "S_circular_insula_ant",
        "S_circular_insula_inf", "S_circular_insula_sup",
    ),
}
_A2009S_LOBE_OF = {region: lobe for lobe, regions in A2009S_LOBES.items()
                   for region in regions}

GRANULARITIES = ("lobe", "region", "hemisphere")


# human atlases the operator can choose (desktop "Human atlas"); "auto" keeps the old order
SCHEME_TITLES: dict[str, str] = {
    "auto": "Automatic (best available)",
    "aparc": "Desikan-Killiany (FreeSurfer aparc+aseg, 34 regions per hemisphere)",
    "a2009s": "Destrieux (FreeSurfer a2009s+aseg, 74 regions per hemisphere)",
    "harvard_oxford": "Harvard-Oxford (FSL, 48 cortical regions, no FreeSurfer needed)",
}


def find_parcellation(run_dir: Path, granularity: str = "region",
                      scheme: str = "auto") -> Optional[tuple[Path, str]]:
    """Locate the parcellation volume this run should use, if any. A named `scheme` is used
    only if its volume is in the run (no silent fallback to another atlas)."""
    if scheme and scheme != "auto":
        if scheme not in SCHEMES:
            raise ValueError(f"unknown parcellation scheme {scheme!r}; use {sorted(SCHEMES)}")
        for filename in (SCHEMES[scheme],) + ((_LEGACY_A2009S_NAME,) if scheme == "a2009s" else ()):
            if (run_dir / filename).exists():
                return run_dir / filename, scheme
        logger.warning("parcellation scheme %s requested but %s is not in %s", scheme,
                       SCHEMES[scheme], run_dir)
        return None
    order = (("aparc", SCHEMES["aparc"]),
             ("a2009s", SCHEMES["a2009s"]),
             ("a2009s", _LEGACY_A2009S_NAME),
             ("harvard_oxford", SCHEMES["harvard_oxford"]))
    if granularity != "lobe":
        order = (("a2009s", SCHEMES["a2009s"]),
                 ("a2009s", _LEGACY_A2009S_NAME),
                 ("aparc", SCHEMES["aparc"]),
                 ("harvard_oxford", SCHEMES["harvard_oxford"]))
    for scheme, filename in order:
        path = run_dir / filename
        if path.exists():
            return path, scheme
    return None


_FREESURFER_SCHEMES: dict[str, tuple[list, dict, tuple[int, int]]] = {
    "aparc": (DK_NAMES, _LOBE_OF, (1000, 2000)),
    "a2009s": (A2009S_NAMES, _A2009S_LOBE_OF, (11100, 12100)),
}


def _freesurfer_name(label_id: int, granularity: str,
                     scheme: str = "aparc") -> Optional[str]:
    """Map one FreeSurfer cortical label id to a part name."""
    names, lobe_of, (lh_base, rh_base) = _FREESURFER_SCHEMES[scheme]
    if lh_base <= label_id < lh_base + len(names):
        side, index = "lh", label_id - lh_base
    elif rh_base <= label_id < rh_base + len(names):
        side, index = "rh", label_id - rh_base
    else:
        return None
    if index <= 0 or names[index] == "Medial_wall":   # not cortex
        return None
    region = names[index]
    if granularity == "hemisphere":
        return side
    if granularity == "lobe":
        lobe = lobe_of.get(region)
        if lobe is None:
            logger.warning("%s has no lobe assignment in scheme %s; leaving it "
                           "out of the lobar split", region, scheme)
            return None
        return f"{side}_{lobe}"
    return f"{side}_{region}"


def parcel_names(volume: np.ndarray, scheme: str, granularity: str) -> dict[int, str]:
    """Which part name each label id in the parcellation maps to."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    out: dict[int, str] = {}
    for label_id in np.unique(volume):
        label_id = int(label_id)
        if label_id <= 0:
            continue
        if scheme in _FREESURFER_SCHEMES:
            name = _freesurfer_name(label_id, granularity, scheme)
        else:
            name = f"ho_{label_id:02d}" if granularity == "region" else "cortex"
        if name:
            out[label_id] = name
    return out


_MAX_FILL_MM = 4.0

_PARCEL_REACH_MM = 3.0

_MIN_TRAVEL_RATIO = 1.5


def _fill_parcels(volume: np.ndarray, known: np.ndarray, spacing: np.ndarray,
                  max_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Give every voxel near the ribbon a parcel, without crossing anatomy."""
    from scipy import ndimage

    travel = volume > 0
    n_known, n_travel = int(known.sum()), int(travel.sum())
    report = {"fill": "geodesic", "travel_voxels": n_travel,
              "ribbon_voxels": n_known}
    if not n_known or n_travel < _MIN_TRAVEL_RATIO * n_known:
        dist, nearest = ndimage.distance_transform_edt(
            ~known, sampling=spacing, return_indices=True)
        report["fill"] = "euclidean"
        report["reason"] = ("the atlas carries no non-cortical labels, so there "
                            "is no tissue mask to travel through")
        return volume[nearest[0], nearest[1], nearest[2]], dist, report

    step = float(np.min(spacing))
    filled = np.where(known, volume, 0).astype(volume.dtype)
    reach = np.where(known, 0.0, np.inf)
    struct = ndimage.generate_binary_structure(3, 1)
    for n_step in range(1, int(np.ceil(max_mm / step)) + 1):
        grown = ndimage.grey_dilation(filled, footprint=struct)
        fresh = (filled == 0) & travel & (grown > 0)
        if not fresh.any():
            break
        filled[fresh] = grown[fresh]
        reach[fresh] = n_step * step
    report["unreached_voxels"] = int((travel & ~np.isfinite(reach)).sum())
    return filled, reach, report

_HEX_FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])

_ONE_PIECE_EXCEPT_PREFIXES: tuple[str, ...] = ("S_", "Lat_Fis", "ho_")
_ONE_PIECE_EXCEPT_NAMES: frozenset = frozenset({"Unknown", "unknown"})

_ONE_PIECE_SPECK_FRACTION = 0.02
_ONE_PIECE_MIN_HEXES = 8

_PIECE_SULCAL_FRACTION = 0.8


def one_piece_excepted(part_name: str) -> bool:
    """Whether more than one connected piece is anatomically expected here."""
    bare = part_name[3:] if part_name[:3] in ("lh_", "rh_") else part_name
    return (bare in _ONE_PIECE_EXCEPT_NAMES
            or any(bare.startswith(p) for p in _ONE_PIECE_EXCEPT_PREFIXES))


def one_piece_check(piece_sizes: dict, piece_sulcal: Optional[dict] = None) -> dict:
    """Grade a per-part piece census against the one-piece rule."""
    violations, excused, specks, buried = [], {}, 0, 0
    for name, raw_sizes in sorted(piece_sizes.items()):
        fractions = (piece_sulcal or {}).get(name)
        if fractions is None or len(fractions) != len(raw_sizes):
            fractions = [0.0] * len(raw_sizes)
        pairs = sorted(((int(s), float(f)) for s, f in zip(raw_sizes, fractions)),
                       key=lambda p: -p[0])
        if len(pairs) < 2:
            continue
        largest = pairs[0][0]
        real = []
        for size, sulcal in pairs[1:]:
            if size < _ONE_PIECE_MIN_HEXES and size < _ONE_PIECE_SPECK_FRACTION * largest:
                specks += 1
            elif sulcal >= _PIECE_SULCAL_FRACTION:
                buried += 1
            else:
                real.append(size)
        sizes = [p[0] for p in pairs]
        if not real:
            continue
        row = {"part": name, "pieces": 1 + len(real),
               "largest": largest, "stray_hexes": int(sum(real)),
               "piece_sizes": [largest] + real}
        if one_piece_excepted(name):
            excused[name] = row
        else:
            violations.append(row)
    report = {
        "one_piece_ok": not violations,
        "one_piece_violations": violations,
        "one_piece_excused": sorted(excused),
        "one_piece_specks": int(specks),
        "one_piece_buried_sulci": int(buried),
    }
    if violations:
        logger.warning(
            "parcellation: %d parts are not one connected piece -- %s. Each piece sits "
            "in its own atlas territory so nothing is donated to a neighbour, but a part "
            "in two pieces is two disjoint element sets under one *PART id",
            len(violations),
            ", ".join(f"{v['part']} ({v['pieces']} pieces, {v['stray_hexes']} hexes off "
                      f"the main body)" for v in violations[:8]))
    if excused:
        logger.info("parcellation: %d parts are multi-piece by anatomy and excused: %s",
                    len(excused), ", ".join(sorted(excused)[:8]))
    return report


def exclusivity_check(blocks: dict, tissue_of: Optional[dict] = None) -> dict:
    """If this part claims these elements, no other part may."""
    keys: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    names = sorted(blocks)
    for i, name in enumerate(names):
        cells = np.asarray(blocks[name])
        if not len(cells):
            continue
        keys.append(np.sort(cells, axis=1))
        owner.append(np.full(len(cells), i, dtype=np.int64))
    report = {"exclusive_ok": True, "duplicate_elements": 0,
              "duplicated_parts": [], "tissue_violations": {}}
    if keys:
        key = np.concatenate(keys)
        who = np.concatenate(owner)
        order = np.lexsort(key.T[::-1])
        key, who = key[order], who[order]
        same = np.all(key[1:] == key[:-1], axis=1)
        if same.any():
            report["duplicate_elements"] = int(same.sum())
            pairs = {tuple(sorted((names[a], names[b])))
                     for a, b in zip(who[:-1][same], who[1:][same])}
            report["duplicated_parts"] = sorted(f"{a} + {b}" for a, b in pairs)
            report["exclusive_ok"] = False
            logger.error("parcellation: %d elements are claimed by more than one part "
                         "(%s). Two *ELEMENT_SOLID cards on the same nodes double that "
                         "element's mass and stiffness and LS-DYNA accepts both",
                         report["duplicate_elements"],
                         ", ".join(report["duplicated_parts"][:6]))

    for name, (expected, actual) in (tissue_of or {}).items():
        actual = np.asarray(actual)
        if not len(actual):
            continue
        wrong = int((actual != expected).sum())
        if wrong:
            report["tissue_violations"][name] = wrong
            report["exclusive_ok"] = False
    if report["tissue_violations"]:
        logger.error("parcellation: %d parts contain elements whose own segmentation "
                     "voxel is a different tissue: %s", len(report["tissue_violations"]),
                     dict(sorted(report["tissue_violations"].items())[:6]))
    return report


def _hex_adjacency(hexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of hex rows that share a quad face."""
    if len(hexes) < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    faces = hexes[:, _HEX_FACES].reshape(-1, 4)
    owner = np.repeat(np.arange(len(hexes), dtype=np.int64), len(_HEX_FACES))
    key = np.sort(faces, axis=1)
    order = np.lexsort((key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    return owner[:-1][same], owner[1:][same]


def _enforce_one_piece(parcel: np.ndarray, hexes: np.ndarray, n_parcels: int,
                       own_part: np.ndarray, reach: np.ndarray,
                       parcel_reach_mm: float = _PARCEL_REACH_MM,
                       sulcal: Optional[np.ndarray] = None,
                       max_layers: int = 3) -> tuple[np.ndarray, dict]:
    """Re-home the unclaimed hexes, and never outside a hex's own territory."""
    from scipy import sparse
    from scipy.sparse import csgraph

    out = np.asarray(parcel).copy()
    n = len(out)
    left, right = _hex_adjacency(hexes)
    homeless = out < 0
    report = {"unclaimed": int(homeless.sum()), "extra_components": 0,
              "hexes_rehomed": 0, "hexes_rehomed_by_adjacency": 0, "unresolved": 0}
    if not len(left) or n_parcels <= 0:
        return out, report

    rows = np.concatenate([left, right])
    others = np.concatenate([right, left])
    in_reach = reach <= parcel_reach_mm
    while homeless.any():
        restricted = (homeless[rows] & in_reach[rows]
                      & (out[others] == own_part[rows]) & ~homeless[others])
        if not restricted.any():
            break
        moved = np.unique(rows[restricted])
        out[moved] = own_part[moved]
        homeless[moved] = False
        report["hexes_rehomed"] += len(moved)

    for _layer in range(max(int(max_layers), 0)):
        if not homeless.any():
            break
        take = homeless[rows] & ~homeless[others]
        if not take.any():
            break
        key = rows[take].astype(np.int64) * n_parcels + out[others[take]]
        uniq, votes = np.unique(key, return_counts=True)
        src, dst = np.divmod(uniq, n_parcels)
        order = np.lexsort((-votes, src))
        src, dst = src[order], dst[order]
        first = np.r_[True, src[1:] != src[:-1]]
        out[src[first]] = dst[first]
        homeless[src[first]] = False
        report["hexes_rehomed_by_adjacency"] += int(first.sum())

    report["unresolved"] = int(homeless.sum())
    if report["unresolved"]:
        logger.warning("parcellation: %d hexes have no parcelled neighbour at all; kept in "
                       "the tissue part they started in rather than any named parcel",
                       report["unresolved"])

    same = (out[left] == out[right]) & (out[left] >= 0)
    graph = sparse.coo_matrix((np.ones(int(same.sum()), dtype=np.int8),
                               (left[same], right[same])), shape=(n, n))
    _count, piece = csgraph.connected_components(graph, directed=False)
    settled = np.nonzero(out >= 0)[0]
    if len(settled):
        key = out[settled] * (piece.max() + 1) + piece[settled]
        report["extra_components"] = int(len(np.unique(key))
                                         - len(np.unique(out[settled])))
        if report["extra_components"]:
            logger.info("parcellation: %d parcel pieces beyond one per parcel; each sits in "
                        "its own atlas territory and is kept there",
                        report["extra_components"])
        uniq_key, inverse, counts = np.unique(key, return_inverse=True,
                                              return_counts=True)
        if sulcal is None:
            share = np.zeros(len(uniq_key), dtype=float)
        else:
            share = (np.bincount(inverse, weights=np.asarray(sulcal, float)[settled],
                                 minlength=len(uniq_key)) / counts)
        sizes: dict[int, list] = {}
        shares: dict[int, list] = {}
        stride = int(piece.max()) + 1
        for k, c, s in zip(uniq_key, counts, share):
            sizes.setdefault(int(k) // stride, []).append(int(c))
            shares.setdefault(int(k) // stride, []).append(float(s))
        report["_piece_sizes_by_index"] = sizes
        report["_piece_sulcal_by_index"] = shares
    return out, report


def assign_hexes(
    nodes: np.ndarray, hexes: np.ndarray, volume: np.ndarray, affine: np.ndarray,
    names: dict[int, str], parcel_reach_mm: float = _PARCEL_REACH_MM,
    region_names: Optional[dict[int, str]] = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Split a hex block into parcels by where each element's centroid lands."""
    if not len(hexes):
        return {}, {}
    inv = np.linalg.inv(affine)
    centroids = nodes[hexes].mean(axis=1)
    vox = centroids @ inv[:3, :3].T + inv[:3, 3]
    idx = np.clip(np.floor(vox + 0.5).astype(np.int64), 0, np.array(volume.shape) - 1)

    known = np.isin(volume, list(names))
    if not known.any():
        return {}, {}
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    filled, dist, fill_report = _fill_parcels(volume, known, spacing,
                                              max(_MAX_FILL_MM, parcel_reach_mm))

    at = filled[idx[:, 0], idx[:, 1], idx[:, 2]]
    reach = dist[idx[:, 0], idx[:, 1], idx[:, 2]]

    part_names = sorted(set(names.values()))
    part_index = {name: i for i, name in enumerate(part_names)}
    label_to_part = {label_id: part_index[name] for label_id, name in names.items()}

    parcel = np.full(len(hexes), -1, dtype=np.int64)
    inside = reach <= _MAX_FILL_MM
    for label_id, i in label_to_part.items():
        parcel[inside & (at == label_id)] = i
    out_of_reach = int((~inside).sum())

    label_lut = np.full(max(label_to_part) + 1, -1, dtype=np.int64)
    for label_id, i in label_to_part.items():
        label_lut[label_id] = i
    own_part = label_lut[at]

    sulcal = None
    if region_names:
        excepted = np.zeros(max(region_names) + 1, dtype=bool)
        for label_id, region in region_names.items():
            excepted[label_id] = one_piece_excepted(region)
        safe = np.clip(at, 0, len(excepted) - 1)
        sulcal = excepted[safe] & (at >= 0) & (at < len(excepted))

    element_mm = float(np.median(np.linalg.norm(
        nodes[hexes[:10000, 1]] - nodes[hexes[:10000, 0]], axis=1))) or 1.0
    parcel, report = _enforce_one_piece(parcel, hexes, len(part_names),
                                        own_part, reach, parcel_reach_mm,
                                        sulcal=sulcal,
                                        max_layers=int(np.ceil(parcel_reach_mm / element_mm)))
    report["out_of_territory"] = out_of_reach
    report["max_fill_mm"] = _MAX_FILL_MM
    report["parcel_reach_mm"] = parcel_reach_mm
    report.update(fill_report)

    by_index = report.pop("_piece_sizes_by_index", {})
    sulcal_by_index = report.pop("_piece_sulcal_by_index", {})
    piece_sizes = {name: by_index.get(i, []) for name, i in part_index.items()
                   if by_index.get(i)}
    piece_sulcal = {name: sulcal_by_index.get(i, []) for name, i in part_index.items()
                    if by_index.get(i)}
    report["piece_counts"] = {n: len(s) for n, s in sorted(piece_sizes.items())}
    report.update(one_piece_check(piece_sizes, piece_sulcal))
    logger.info("parcellation: %d hexes were further than %.1fmm from any parcel; %d re-homed "
                "into their own part, %d unresolved, %d extra parcel pieces kept in place",
                out_of_reach, _MAX_FILL_MM, report["hexes_rehomed"], report["unresolved"],
                report["extra_components"])

    out: dict[str, np.ndarray] = {}
    for name, i in part_index.items():
        rows = np.nonzero(parcel == i)[0]
        if len(rows):
            out[name] = hexes[rows]
    report["_unclaimed_rows"] = np.nonzero(parcel < 0)[0]
    return out, report


def split_brain_parts(
    solid_blocks: dict, nodes: np.ndarray, affine: np.ndarray, run_dir: Path,
    granularity: str = "lobe", regions: Optional[tuple[str, ...]] = None,
    source_regions: tuple[str, ...] = ("gray_matter",),
    parcel_reach_mm: float = _PARCEL_REACH_MM, scheme: str = "auto",
) -> tuple[dict, dict]:
    """Replace whole-tissue brain parts with per-parcel parts."""
    import nibabel as nib

    found = find_parcellation(run_dir, granularity, scheme)
    if found is None:
        return solid_blocks, {"status": "skipped", "reason": "no parcellation volume in the run"}
    path, scheme = found

    img = nib.load(str(path))
    volume = np.asarray(img.dataobj).astype(np.int32)
    names = parcel_names(volume, scheme, granularity)
    if not names:
        return solid_blocks, {"status": "skipped", "reason": f"no usable labels in {path.name}"}
    wanted = set(regions) if regions else None
    if wanted is not None and not any(v in wanted for v in names.values()):
        return solid_blocks, {"status": "skipped",
                              "reason": f"none of {sorted(wanted)} are in {path.name}"}

    region_names = parcel_names(volume, scheme, "region") if granularity != "region" else names

    out = dict(solid_blocks)
    report = {"status": "ok", "source": path.name, "scheme": scheme,
              "granularity": granularity, "parts": {}}
    for tissue in source_regions:
        block = out.get(tissue)
        if block is None or not len(block):
            continue
        parcels, continuity = assign_hexes(nodes, block, volume, affine, names,
                                          parcel_reach_mm=parcel_reach_mm,
                                          region_names=region_names)
        assigned = sum(len(v) for v in parcels.values())
        if not parcels or assigned < 0.5 * len(block):
            logger.warning("parcellation covered only %d of %d %s hexes; leaving it whole",
                           assigned, len(block), tissue)
            continue
        leftover = continuity.pop("_unclaimed_rows", None)
        report.setdefault("continuity", {})[tissue] = continuity
        rest = [block[leftover]] if leftover is not None and len(leftover) else []
        if wanted is not None:
            rest += [cells for name, cells in parcels.items() if name not in wanted]
            parcels = {name: cells for name, cells in parcels.items() if name in wanted}
        if rest:
            out[tissue] = np.concatenate(rest, axis=0)
            report["parts"][tissue] = int(len(out[tissue]))
        else:
            del out[tissue]
        for name, cells in parcels.items():
            key = f"{tissue}_{name}" if len(source_regions) > 1 else name
            out[key] = cells
            report["parts"][key] = int(len(cells))
    continuity = report.get("continuity", {})
    report["one_piece_ok"] = all(c.get("one_piece_ok", True) for c in continuity.values())
    report["one_piece_violations"] = [v for c in continuity.values()
                                      for v in c.get("one_piece_violations", ())]

    report["exclusivity"] = exclusivity_check(out)
    logger.info("parcellation: %s -> %d parts from %s, one-piece %s", granularity,
                len(report["parts"]), path.name,
                "OK" if report["one_piece_ok"]
                else f"FAILED on {len(report['one_piece_violations'])} parts")
    return out, report
