"""Parts each atlas offers, and how a part selection becomes meshing settings.

A selected part is its own part in the deck. A sub-part (structure or atlas region) that is not
selected stays inside its tissue; a tissue that is not selected is merged into one "brain" part.
Head layers that are not selected are not built.
"""
from __future__ import annotations

from engine.anatomy import parcellation, substructures

ATLASES = ("aseg", "aparc", "destrieux", "harvard", "whole")
SCHEME = {"aparc": "aparc", "destrieux": "a2009s", "harvard": "harvard_oxford"}

HEAD = [("scalp", "Scalp"), ("skull", "Skull"), ("dura", "Dura"), ("csf", "CSF"), ("pia", "Pia"),
        ("falx", "Falx"), ("tentorium", "Tentorium"), ("venous_sinuses", "Venous Sinuses"),
        ("arteries", "Arteries"), ("axons", "Axon Beams")]
TISSUES = [("gray_matter", "Gray Matter"), ("white_matter", "White Matter"),
           ("deep_nuclei", "Deep Nuclei"), ("brainstem", "Brainstem"), ("cerebellum", "Cerebellum")]
LAYER_BOUNDARIES = ("csf", "skull", "scalp", "pia", "dura", "falx", "tentorium")
DEFAULT_OFF = {"arteries", "axons"}

DK_TITLES = {
    "bankssts": "Banks of the Superior Temporal Sulcus", "caudalanteriorcingulate": "Caudal Anterior Cingulate",
    "caudalmiddlefrontal": "Caudal Middle Frontal", "cuneus": "Cuneus", "entorhinal": "Entorhinal",
    "fusiform": "Fusiform", "inferiorparietal": "Inferior Parietal", "inferiortemporal": "Inferior Temporal",
    "isthmuscingulate": "Isthmus Cingulate", "lateraloccipital": "Lateral Occipital",
    "lateralorbitofrontal": "Lateral Orbitofrontal", "lingual": "Lingual",
    "medialorbitofrontal": "Medial Orbitofrontal", "middletemporal": "Middle Temporal",
    "parahippocampal": "Parahippocampal", "paracentral": "Paracentral", "parsopercularis": "Pars Opercularis",
    "parsorbitalis": "Pars Orbitalis", "parstriangularis": "Pars Triangularis", "pericalcarine": "Pericalcarine",
    "postcentral": "Postcentral", "posteriorcingulate": "Posterior Cingulate", "precentral": "Precentral",
    "precuneus": "Precuneus", "rostralanteriorcingulate": "Rostral Anterior Cingulate",
    "rostralmiddlefrontal": "Rostral Middle Frontal", "superiorfrontal": "Superior Frontal",
    "superiorparietal": "Superior Parietal", "superiortemporal": "Superior Temporal",
    "supramarginal": "Supramarginal", "frontalpole": "Frontal Pole", "temporalpole": "Temporal Pole",
    "transversetemporal": "Transverse Temporal", "insula": "Insula",
}

# FSL HarvardOxford-Cortical maximum-probability labels 1..48 (bilateral)
HO_TITLES = [
    "Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus", "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus, Triangular", "Inferior Frontal Gyrus, Opercular", "Precentral Gyrus",
    "Temporal Pole", "Superior Temporal Gyrus, Anterior", "Superior Temporal Gyrus, Posterior",
    "Middle Temporal Gyrus, Anterior", "Middle Temporal Gyrus, Posterior",
    "Middle Temporal Gyrus, Temporooccipital", "Inferior Temporal Gyrus, Anterior",
    "Inferior Temporal Gyrus, Posterior", "Inferior Temporal Gyrus, Temporooccipital",
    "Postcentral Gyrus", "Superior Parietal Lobule", "Supramarginal Gyrus, Anterior",
    "Supramarginal Gyrus, Posterior", "Angular Gyrus", "Lateral Occipital Cortex, Superior",
    "Lateral Occipital Cortex, Inferior", "Intracalcarine Cortex", "Frontal Medial Cortex",
    "Juxtapositional Lobule", "Subcallosal Cortex", "Paracingulate Gyrus", "Cingulate Gyrus, Anterior",
    "Cingulate Gyrus, Posterior", "Precuneus", "Cuneal Cortex", "Frontal Orbital Cortex",
    "Parahippocampal Gyrus, Anterior", "Parahippocampal Gyrus, Posterior", "Lingual Gyrus",
    "Temporal Fusiform Cortex, Anterior", "Temporal Fusiform Cortex, Posterior",
    "Temporal Occipital Fusiform Cortex", "Occipital Fusiform Gyrus", "Frontal Operculum",
    "Central Opercular Cortex", "Parietal Operculum", "Planum Polare", "Heschl's Gyrus",
    "Planum Temporale", "Supracalcarine Cortex", "Occipital Pole",
]
HO_LOBES = {
    "Frontal": (1, 3, 4, 5, 6, 7, 25, 26, 27, 33, 41), "Temporal": (8, 9, 10, 11, 12, 13, 14, 15, 16, 34, 35, 37, 38, 44, 45, 46),
    "Parietal": (17, 18, 19, 20, 21, 31, 42, 43), "Occipital": (22, 23, 24, 32, 36, 39, 40, 47, 48),
    "Cingulate": (28, 29, 30), "Insula": (2,),
}
LOBE_TITLES = {"frontal": "Frontal", "parietal": "Parietal", "temporal": "Temporal",
               "occipital": "Occipital", "cingulate": "Cingulate", "insula": "Insula"}


def _title(name: str) -> str:
    return name.replace("_", " ").title()


# Destrieux et al. 2010, NeuroImage 53:1-15, Table 1
DESTRIEUX_TITLES = {
    "G_and_S_frontomargin": "Fronto-Marginal Gyrus and Sulcus", "G_and_S_occipital_inf": "Inferior Occipital Gyrus and Sulcus",
    "G_and_S_paracentral": "Paracentral Lobule and Sulcus", "G_and_S_subcentral": "Subcentral Gyrus and Sulci",
    "G_and_S_transv_frontopol": "Transverse Frontopolar Gyri and Sulci", "G_and_S_cingul-Ant": "Anterior Cingulate",
    "G_and_S_cingul-Mid-Ant": "Middle-Anterior Cingulate", "G_and_S_cingul-Mid-Post": "Middle-Posterior Cingulate",
    "G_cingul-Post-dorsal": "Posterior-Dorsal Cingulate Gyrus", "G_cingul-Post-ventral": "Posterior-Ventral Cingulate Gyrus",
    "G_cuneus": "Cuneus", "G_front_inf-Opercular": "Inferior Frontal Gyrus, Opercular",
    "G_front_inf-Orbital": "Inferior Frontal Gyrus, Orbital", "G_front_inf-Triangul": "Inferior Frontal Gyrus, Triangular",
    "G_front_middle": "Middle Frontal Gyrus", "G_front_sup": "Superior Frontal Gyrus",
    "G_Ins_lg_and_S_cent_ins": "Long Insular Gyrus and Central Insular Sulcus", "G_insular_short": "Short Insular Gyri",
    "G_occipital_middle": "Middle Occipital Gyrus", "G_occipital_sup": "Superior Occipital Gyrus",
    "G_oc-temp_lat-fusifor": "Fusiform Gyrus", "G_oc-temp_med-Lingual": "Lingual Gyrus",
    "G_oc-temp_med-Parahip": "Parahippocampal Gyrus", "G_orbital": "Orbital Gyri",
    "G_pariet_inf-Angular": "Angular Gyrus", "G_pariet_inf-Supramar": "Supramarginal Gyrus",
    "G_parietal_sup": "Superior Parietal Lobule", "G_postcentral": "Postcentral Gyrus", "G_precentral": "Precentral Gyrus",
    "G_precuneus": "Precuneus", "G_rectus": "Straight Gyrus", "G_subcallosal": "Subcallosal Gyrus",
    "G_temp_sup-G_T_transv": "Transverse Temporal Gyrus", "G_temp_sup-Lateral": "Superior Temporal Gyrus, Lateral",
    "G_temp_sup-Plan_polar": "Planum Polare", "G_temp_sup-Plan_tempo": "Planum Temporale",
    "G_temporal_inf": "Inferior Temporal Gyrus", "G_temporal_middle": "Middle Temporal Gyrus",
    "Lat_Fis-ant-Horizont": "Lateral Fissure, Anterior Horizontal", "Lat_Fis-ant-Vertical": "Lateral Fissure, Anterior Vertical",
    "Lat_Fis-post": "Lateral Fissure, Posterior", "Pole_occipital": "Occipital Pole", "Pole_temporal": "Temporal Pole",
    "S_calcarine": "Calcarine Sulcus", "S_central": "Central Sulcus", "S_cingul-Marginalis": "Marginal Cingulate Sulcus",
    "S_circular_insula_ant": "Anterior Circular Insular Sulcus", "S_circular_insula_inf": "Inferior Circular Insular Sulcus",
    "S_circular_insula_sup": "Superior Circular Insular Sulcus", "S_collat_transv_ant": "Anterior Collateral Transverse Sulcus",
    "S_collat_transv_post": "Posterior Collateral Transverse Sulcus", "S_front_inf": "Inferior Frontal Sulcus",
    "S_front_middle": "Middle Frontal Sulcus", "S_front_sup": "Superior Frontal Sulcus",
    "S_interm_prim-Jensen": "Intermediate Sulcus of Jensen", "S_intrapariet_and_P_trans": "Intraparietal Sulcus",
    "S_oc_middle_and_Lunatus": "Middle Occipital and Lunate Sulci", "S_oc_sup_and_transversal": "Superior and Transverse Occipital Sulci",
    "S_occipital_ant": "Anterior Occipital Sulcus", "S_oc-temp_lat": "Lateral Occipito-Temporal Sulcus",
    "S_oc-temp_med_and_Lingual": "Medial Occipito-Temporal and Lingual Sulci", "S_orbital_lateral": "Lateral Orbital Sulcus",
    "S_orbital_med-olfact": "Olfactory Sulcus", "S_orbital-H_Shaped": "H-Shaped Orbital Sulci",
    "S_parieto_occipital": "Parieto-Occipital Sulcus", "S_pericallosal": "Pericallosal Sulcus",
    "S_postcentral": "Postcentral Sulcus", "S_precentral-inf-part": "Inferior Precentral Sulcus",
    "S_precentral-sup-part": "Superior Precentral Sulcus", "S_suborbital": "Suborbital Sulcus",
    "S_subparietal": "Subparietal Sulcus", "S_temporal_inf": "Inferior Temporal Sulcus",
    "S_temporal_sup": "Superior Temporal Sulcus", "S_temporal_transverse": "Transverse Temporal Sulcus",
}


def _destrieux_title(name: str) -> str:
    return DESTRIEUX_TITLES.get(name, _title(name))


def _sided(prefix: str, names, titles) -> list[dict]:
    side = {"lh": "Left", "rh": "Right"}[prefix]
    return [{"id": f"{prefix}_{n}", "name": f"{side} {titles(n)}"} for n in names]


def _cortex_groups(atlas: str) -> list[dict]:
    if atlas == "harvard":
        return [{"id": f"ho-{lobe.lower()}", "name": lobe,
                 "children": [{"id": f"ho_{i:02d}", "name": HO_TITLES[i - 1]} for i in ids]}
                for lobe, ids in HO_LOBES.items()]
    if atlas == "aparc":
        lobes, titles = parcellation.LOBES, lambda n: DK_TITLES.get(n, _title(n))
    else:
        lobes, titles = parcellation.A2009S_LOBES, _destrieux_title
    groups = []
    for lobe, names in lobes.items():
        children = _sided("lh", names, titles) + _sided("rh", names, titles)
        groups.append({"id": f"{atlas}-{lobe}", "name": LOBE_TITLES.get(lobe, _title(lobe)), "children": children})
    return groups


def _structures_under(tissue: str) -> list[dict]:
    return [{"id": n, "name": s.title[:1].upper() + s.title[1:]}
            for n, s in substructures.STRUCTURES.items() if s.source == tissue]


def part_tree(atlas: str) -> list[dict]:
    """Head layers, then the brain: tissues with the atlas's own parts under them."""
    if atlas not in ATLASES:
        raise ValueError(f"unknown atlas {atlas!r}")
    head = []
    for pid, name in HEAD:
        node = {"id": pid, "name": name}
        if pid == "csf" and atlas != "whole" and atlas != "harvard":
            node["children"] = _structures_under("csf")
        head.append(node)
    if atlas == "whole":
        return [{"id": "head", "name": "Head", "children": head},
                {"id": "brain", "name": "Brain"}]
    tissues = []
    for tid, name in TISSUES:
        node = {"id": tid, "name": name}
        children = [] if atlas == "harvard" else _structures_under(tid)
        if tid == "gray_matter" and atlas in ("aparc", "destrieux", "harvard"):
            children = children + _cortex_groups(atlas)
        if children:
            node["children"] = children
        tissues.append(node)
    return [{"id": "head", "name": "Head", "children": head},
            {"id": "brain", "name": "Brain", "children": tissues}]


def _is_group(node: dict) -> bool:
    return (node["id"] == "head" or (node["id"] == "brain" and "children" in node)
            or node["id"].startswith(("ho-", "aparc-", "destrieux-")))


def leaf_ids(tree: list[dict]) -> list[str]:
    """Every selectable id; groups (head, brain with tissues, lobes) are not parts themselves."""
    out = []
    for node in tree:
        if not _is_group(node):
            out.append(node["id"])
        out += leaf_ids(node.get("children", []))
    return out


def default_selection(atlas: str) -> list[str]:
    """Head layers and tissues on, finer parts off."""
    top = [pid for pid, _ in HEAD if pid not in DEFAULT_OFF]
    return top + (["brain"] if atlas == "whole" else [tid for tid, _ in TISSUES])


def engine_settings(atlas: str, selected: list[str]) -> dict:
    """MeshingConfig fields for this atlas and selection."""
    chosen = set(selected)
    unknown = chosen - set(leaf_ids(part_tree(atlas)))
    if unknown:
        raise ValueError(f"parts not offered by {atlas}: {sorted(unknown)[:10]}")
    out: dict = {
        "auto_boundaries": tuple(b for b in LAYER_BOUNDARIES if b in chosen),
        "vascular_beams": "venous_sinuses" in chosen,
        "arterial_beams": "arteries" in chosen,
        "axon_tracts": "axons" in chosen,
        "brain_structures": None, "brain_regions": None, "brain_granularity": None,
        "merge_brain_parts": False, "merge_tissues": None,
    }
    if atlas == "whole":
        out["merge_brain_parts"] = True
        return out
    out["merge_tissues"] = tuple(t for t, _ in TISSUES if t not in chosen) or None
    structures = tuple(n for n in substructures.STRUCTURES if n in chosen)
    out["brain_structures"] = structures or None
    if atlas in SCHEME:
        regions = tuple(sorted(i for i in chosen if i.startswith(("lh_", "rh_", "ho_"))))
        if regions:
            out.update(brain_granularity="region", brain_regions=regions, parcellation_scheme=SCHEME[atlas])
    return out
