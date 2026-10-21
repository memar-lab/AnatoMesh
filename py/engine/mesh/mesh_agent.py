"""Head mesh generation and LS-DYNA export."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import meshio
import nibabel as nib
import numpy as np

from engine.export import lsdyna_writer
from engine.mesh import membranes
from engine.paths import READ_WRITE_ROOT, assert_write_path, outputs_root
from engine.mesh.surface_extraction import _load_watershed_surface
from engine.mesh.vessels import _mesh_to_voxel_mask, _sample_dti_e1, _vessel_beam_segments
from engine.mesh import fast_kernels
from engine.pipeline import timing

logger = logging.getLogger(__name__)

SOLID_LABELS = {"gray_matter": 1, "white_matter": 2, "brainstem": 3, "cerebellum": 4,
                 "deep_nuclei": 5}
CSF_LABEL = 6


OPTIONAL_BOUNDARIES = ("csf", "skull", "scalp", "pia", "dura", "falx", "tentorium")

REPRESENTATION_CHOICES = ("solid", "shell", "auto", "conforming")
FLIPPABLE_PARTS = ("skull", "scalp", "falx", "tentorium")
SOLID_LAYER_MIN_ELEMENTS = 3.0   # "auto": hexes when the layer is this many elements thick
SKULL_TABLE_SHARE = 1.0 / 3.0    # "conforming" skull: outer-table shell share of the thickness


def _layer_outer_faces(region: str, solid_blocks: dict) -> np.ndarray:
    """Faces of a layer's hexes that face neither the layer itself nor anything inside it:
    the outside of the mesh, or the scalp. Node indices are the mesh's own."""
    from engine.mesh.pipeline_hex import _HEX_FACES_OUT

    own = np.asarray(solid_blocks[region])
    faces = own[:, _HEX_FACES_OUT].reshape(-1, 4)
    key = np.sort(faces, axis=1)
    inner = [np.asarray(b) for name, b in solid_blocks.items()
             if name not in (region, "scalp") and len(b) and np.asarray(b).shape[1] == 8]
    other = (np.sort(np.concatenate(inner)[:, _HEX_FACES_OUT].reshape(-1, 4), axis=1)
             if inner else np.zeros((0, 4), dtype=key.dtype))
    every = np.concatenate([key, other])
    from engine.mesh.hex_mesh import _unique_rows

    inv, counts = _unique_rows(every)
    inv = inv.ravel()
    return faces[counts[inv[:len(key)]] == 1]
DEFAULT_SHELL_OR_SOLID = {"skull": "shell", "scalp": "shell"}
DEFAULT_REPRESENTATION: dict[str, str] = {
    "skull": "auto", "scalp": "conforming", "csf": "solid",
    "dura": "shell", "pia": "shell", "falx": "shell", "tentorium": "shell",
}

MAX_ELEMENT_GROWTH_RATIO = 1.2


@dataclass
class MeshingConfig:
    """Meshing settings. Presets override these."""

    # Label smoothing (sigmas in element voxels) and mesh smoothing
    label_smoothing_sigma_vox: float = 1.2           # stage 1, outer surface
    label_smoothing_iterations: int = 3
    label_smoothing_sigma_inner_vox: float = 0.6     # stage 2, boundaries between tissues
    label_smoothing_iterations_inner: int = 5
    hierarchical_smoothing_lambda: float = 0.3
    hierarchical_smoothing_iterations: int = 3       # "Iterations"; Iter N = 2x this
    hierarchical_smoothing_repair: bool = True
    repair_jacobian_threshold: float = 0.5           # elements below this are reshaped
    smoothing_target_jacobian: Optional[float] = None  # keep smoothing until min SJ >= this
    smoothing_max_iterations: int = 10

    # parts
    auto_boundaries: tuple[str, ...] = OPTIONAL_BOUNDARIES
    part_representation: Optional[dict] = None       # {"skull": "solid"} etc.; see FLIPPABLE_PARTS
    merge_brain_parts: bool = False                  # one "brain" part instead of five tissues
    merge_tissues: Optional[tuple[str, ...]] = None  # only these tissues merged into "brain"
    combine_brain_for_smoothing: bool = True         # brain tissues smoothed as one, then restored
    membrane_field_placement: bool = False
    brain_granularity: Optional[str] = None          # "lobe", "region" or "hemisphere" parts
    brain_structures: Optional[tuple[str, ...]] = None   # named structures as their own parts
    brain_regions: Optional[tuple[str, ...]] = None      # keep only these parcels as parts
    parcellation_scheme: str = "auto"                # human atlas: "auto", "aparc", "a2009s", "harvard_oxford"
    region_scale: Optional[dict] = None              # {"gray_matter": 1.5} scales a region
    scalp_extent: str = "full_head"                  # "full_head" (down to the bottom of the image:
                                                     # face, jaw, neck) or "cranial" (cut at the
                                                     # skull base)

    # element sizes (mm)
    element_size_mm: Optional[float] = None          # None = input grid spacing
    meninges_element_size_mm: float = 3.0            # pia and dura sheets
    pia_element_size_mm: Optional[float] = None      # None = volume grid spacing
    membrane_element_size_mm: float = 1.0            # falx and tentorium
    membrane_slot_mm: float = 4.0                    # CSF slot carved for each membrane

    # head layers
    subarachnoid_from_physiology: bool = False       # solve meninges_gap_mm from brain/cavity ratio
    vault_bridge_mm: float = 0.0                     # bridge sulci before offsetting the skull
    require_mri_derived_layers: bool = True          # fail instead of using offset skull/scalp
    skull_source: str = "charm"                      # "charm" (SimNIBS whole-head segmentation: bone,
                                                     # soft tissue, cavity; falls back to "segmented"
                                                     # when charm is not installed), "segmented" (skull,
                                                     # scalp and cavity from T1 intensities), "t1" (literature-thickness band
                                                     # off an offset dura), "template_dark_band"
                                                     # (in-skull atlas) or "offset" (thicknesses below)
    scalp_max_mm: Optional[float] = None             # farthest scalp from the skull (None: 30 segmented, 15 t1)
    skull_min_mm: float = 1.5                        # "segmented": thinnest skull kept
    csf_from_image: bool = True                      # "segmented": CSF space segmented from the T1
                                                     # (False: the physiological offset, conventional)
    fix_skull_boundary: bool = True                  # fully constrain the skull nodes
    shell_thickness_dura: float = 0.45
    shell_thickness_pia: float = 0.10224969768668767
    shell_thickness_skull: float = 5.133450929385945
    shell_thickness_scalp: float = 4.799166540426273
    contact_penalty_scale: float = 0.8543639269030632
    sulcal_fold_contact: bool = True
    topology_gate_strict: bool = True                # False: warn instead of fail (mouse)
    falx_cc_gap_mm: float = 10.0                     # falx clearance above the corpus callosum
    membrane_profile: str = "auto"                   # "auto" (by species), "fused", "carnivore", "lateral"
    tentorium_lateral_open_frac: float = 0.35        # "lateral": medial edge of each leaf, share of half-width
    tentorium_osseous_frac: float = 0.33             # "carnivore": bony posteromedial share
    membrane_scale: Optional[float] = 1.0            # set per species by the mesher (None = measure)
    # method choices, new (default) or conventional; see membranes.MEMBRANE_OPTIONS
    tentorium_orientation: str = "auto"              # "auto" (follow the cerebellum) or "level"
    tentorium_surface: str = "smooth"                # "smooth" (envelopes, 3 mm) or "voxel" (clip, 1 mm)
    tentorium_tissue_ease: bool = True               # ease the tentorium out of brain tissue
    membrane_rim_mode: str = "retract"               # "retract" rims out of bone, or "trim" rim quads
    falx_cc_profile: str = "posterior_close"         # or "legacy" (widest over the posterior body)
    falx_cavity: str = "plane"                       # falx outline on its plane's cavity, or "slab"
    membrane_rules: str = "auto"                     # "lab_templates" (human), "2026-10-09" (animals), "anatomical"
    falx_wall_inset_mm: Optional[float] = None       # falx attached margin inside the cavity (None: 0.4)
    tentorium_wall_inset_mm: Optional[float] = None  # tentorium rim inside the cavity (None: 0)
    parcel_reach_mm: float = 3.0

    # vessels
    vascular_beams: bool = True                      # venous sinuses and bridging veins
    arterial_beams: bool = False
    arterial_occurrence_threshold: float = 15.0      # % of atlas subjects with a vessel in a voxel
    arterial_beam_spacing_mm: float = 3.0
    vascular_beam_spacing_mm: float = 4.0

    # axons
    # element quality after the voxel smoothing (same elements, same interfaces)
    hex_quality_pass: bool = True                    # node moves that raise the scaled Jacobian
    hex_quality_target: float = 0.98                 # elements below this are worked on

    axon_tracts: bool = True
    axon_beam_spacing_mm: float = 3.0
    axon_min_length_mm: float = 20.0
    axon_max_tracts: int = 5000                      # most streamlines kept
    axon_volume_per_beam_mm2: Optional[float] = None  # brain volume / (N beams x beam length)
    axon_section_mm2: Optional[float] = None         # axon cross-section for the fill rule (mouse
                                                     # 0.0177 = pi * 0.075^2, a 0.15 mm thick axon)
    axon_fill_fraction: Optional[float] = None       # with axon_section_mm2: total axon length x
                                                     # section = this share of the brain volume
                                                     # (0.5 as Hajiaghamemar 2019); overrides the above
    axon_engine: str = "native"                      # engine.engines.tracking, or the population atlas without a DWI
    axon_clip: str = "mesh"                          # "mesh" (final brain elements), "atlas" or "none"
    axon_beam_study_mm: tuple = ()                   # beam spacings scored unclipped before clipping

    # membranes
    meninges_mode: str = "conforming"                # pia/dura on the hex faces; "free" = offset sheets
    pia_offset_mm: float = 1.0
    csf_ventricles_from_segmentation: bool = True  # False when the atlas CSF is sulcal (ferret)
    membrane_pair_gap_mm: float = 1.0             # minimum gap between two sheets
    meninges_gap_mm: float = 2.6                     # dura offset from pia without a watershed skull

def _drop_degenerate_shells(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Remove zero-area shell elements from a triangle or quad block."""
    if len(faces) == 0:
        return faces
    pts = verts[faces]
    normal = np.zeros((len(faces), 3))
    for i in range(faces.shape[1]):
        a, b = pts[:, i], pts[:, (i + 1) % faces.shape[1]]
        normal += np.cross(a, b)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    keep = area > 1e-9
    if not keep.all():
        logger.info("dropped %d degenerate shell element(s) of %d", int((~keep).sum()), len(faces))
    return faces[keep]


def _layers_from_t1(run_dir: Path, labels: np.ndarray, affine: np.ndarray,
                    config: MeshingConfig) -> dict:
    """Skull and scalp measured off this run's whole-head T1, if there is one."""
    from engine.anatomy import head_tissue

    empty: dict = {"dura": None, "skull": None, "scalp": None}
    source = getattr(config, "skull_source", "t1")
    if source == "offset":
        return empty
    if source == "template_dark_band":
        image_path = run_dir / "t1_brain.nii.gz"  # the whole-head template for atlas species
        if not image_path.exists():
            logger.warning("skull_source=template_dark_band but %s is missing; offset skull",
                           image_path.name)
            return empty
        image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
        if image.shape != labels.shape:
            logger.warning("template %s does not match the labels %s; offset skull",
                           image.shape, labels.shape)
            return empty
        skull, report = head_tissue.thin_skull_from_template(
            image, affine, np.isin(labels, list(SOLID_LABELS.values()) + [CSF_LABEL]))
        return {"dura": None, "skull": skull, "scalp": None,
                "sources": {"skull": "template_dark_band"}, "skull_report": report}
    t1_path = next((run_dir / n for n in ("t1_head.nii.gz", "t1_n4.nii.gz", "t1_ras.nii.gz")
                    if (run_dir / n).exists()), None)
    if t1_path is None:
        logger.info("no whole-head T1 in %s; skull and scalp stay offset envelopes", run_dir)
        return empty

    img = nib.load(str(t1_path))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    if volume.shape != labels.shape:
        logger.warning("T1 %s is %s but the segmentation is %s; skipping the "
                       "T1-derived skull and scalp", t1_path.name, volume.shape, labels.shape)
        return empty

    brain = np.isin(labels, list(SOLID_LABELS.values()))
    scalp_max = getattr(config, "scalp_max_mm", None)
    extent = "full" if getattr(config, "scalp_extent", "full_head") == "full_head" else "cranial"
    if source == "charm":
        from engine.anatomy import charm

        if charm.available():
            try:
                tissues = charm.run_charm(t1_path, charm.cache_dir(t1_path, run_dir.parent),
                                          subject="head")
                layers = charm.charm_head_layers(tissues, affine, labels.shape, brain=brain,
                                                 extent=extent)
                layers["skull_report"] = layers["report"]
                logger.info("head layers from charm: %s", layers["sources"])
                return layers
            except Exception as exc:  # noqa: BLE001 - fall back rather than lose the run
                logger.warning("charm head segmentation failed (%s); using the T1 intensity "
                               "segmentation instead", exc)
        else:
            logger.warning("skull_source=charm but SimNIBS charm is not installed at %s; using "
                           "the T1 intensity segmentation", charm.CHARM_EXE)
        source = "segmented"
    try:
        if source == "segmented":
            # intracranial space, skull and scalp segmented from the T1's own intensities
            layers = head_tissue.segment_head_layers(
                volume, affine, brain, labels == CSF_LABEL,
                scalp_max_mm=float(scalp_max) if scalp_max else head_tissue.SEG_SCALP_MAX_MM,
                skull_min_mm=float(getattr(config, "skull_min_mm", head_tissue.SEG_SKULL_MIN_MM)),
                extent=extent)
        else:
            # "t1": a skull band of the literature thickness off an offset dura envelope,
            # modulated by the T1's darkness; scalp capped over it
            layers = head_tissue.derive_head_layers(
                volume, affine, brain, meninges_gap_mm=config.meninges_gap_mm,
                scalp_extent=getattr(config, "scalp_extent", "cranial"),
                scalp_max_mm=float(scalp_max) if scalp_max else head_tissue.SCALP_MAX_MM)
    except Exception as exc:  # noqa: BLE001 - fall back rather than lose the run
        logger.warning("T1-derived head layers failed (%s); using offset envelopes", exc)
        return empty
    logger.info("head layers from %s: %s", t1_path.name, layers["sources"])
    return layers


_LAYER_ORDER = ("gray_matter", "white_matter", "brainstem", "cerebellum",
                   "deep_nuclei", "csf", "skull", "scalp")


def _probator_exterior(labels: np.ndarray, affine: np.ndarray, region: str,
                       band_thickness_mm: float):
    """Return `f(points) -> bool`, True where a point is outside `region`'s layer."""
    from scipy import ndimage

    from engine.anatomy.anatomy_labels import REGION_LABELS

    if region not in _LAYER_ORDER:
        return None
    inner_mask = np.isin(labels, [REGION_LABELS[n]
                                for n in _LAYER_ORDER[:_LAYER_ORDER.index(region)]])
    if not inner_mask.any():
        return None
    voxel_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    dist_map = np.maximum(
        fast_kernels.distance_transform_edt(~inner_mask, sampling=voxel_spacing) - 0.5 * voxel_spacing.mean(),
        0.0)
    inv = np.linalg.inv(affine)
    half_thickness = 0.5 * float(band_thickness_mm)

    def exterior(point_set: np.ndarray) -> np.ndarray:
        vox = np.asarray(point_set, dtype=float) @ inv[:3, :3].T + inv[:3, 3]
        procul = ndimage.map_coordinates(dist_map, vox.T, order=1, mode="nearest")
        return procul > half_thickness

    return exterior


def _face_junctions(boundary: np.ndarray) -> np.ndarray:
    """Pairs of boundary faces that share a manifold (two-sided) edge."""
    latera = np.concatenate([boundary[:, [i, (i + 1) % 4]] for i in range(4)])
    lookup_key = np.sort(latera, axis=1)
    _uniq, inverse_idx, numeri = np.unique(lookup_key, axis=0, return_inverse=True,
                                       return_counts=True)
    face_pairs = np.tile(np.arange(len(boundary)), 4)
    par = numeri[inverse_idx] == 2
    sort_order = np.argsort(inverse_idx[par], kind="stable")
    return face_pairs[par][sort_order].reshape(-1, 2)


def _componentes(boundary: np.ndarray, joined_mask: np.ndarray) -> np.ndarray:
    """Connected-component label per face, over the given adjacency pairs."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph_ = coo_matrix(
        (np.ones(len(joined_mask)), (joined_mask[:, 0], joined_mask[:, 1])),
        shape=(len(boundary), len(boundary)))
    _n, tituli = connected_components(graph_, directed=False)
    return tituli


def _boundary_sheets(boundary: np.ndarray) -> np.ndarray:
    """Label each boundary face by which SHEET of the layer's surface it is on."""
    return _componentes(boundary, _face_junctions(boundary))


def _outer_shell_quads(nodes: np.ndarray, cells: np.ndarray,
                       exterior=None) -> np.ndarray:
    """The outward-facing boundary faces of a hex block."""
    boundary = _boundary_faces(cells)
    if not len(boundary):
        return boundary

    p = nodes[boundary]
    centre = nodes[np.unique(cells)].mean(axis=0)
    joined_mask = _face_junctions(boundary)
    tituli = _componentes(boundary, joined_mask)
    if tituli.max() == 1:
        medium = p.mean(axis=1)
        longe = [np.linalg.norm(medium[tituli == t] - centre, axis=1).mean()
                 for t in (0, 1)]
        return boundary[tituli == int(np.argmax(longe))]

    logger.info("outer table: the layer's boundary is %d sheet(s), not 2, so it "
                "is not a closed layer; falling back to the per-face vote",
                tituli.max() + 1)
    if exterior is not None:
        classified = exterior(p.mean(axis=1))
    else:
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
        classified = np.einsum("ij,ij->i", normal, p.mean(axis=1) - centre) > 0

    if not classified.any():
        return boundary[:0]

    same_vote = classified[joined_mask[:, 0]] == classified[joined_mask[:, 1]]
    vote_comps = _componentes(boundary, joined_mask[same_vote])
    sizes = np.bincount(vote_comps)
    is_outer = np.zeros(len(sizes), dtype=bool)
    is_outer[vote_comps[classified]] = True

    inner_ids = np.flatnonzero(~is_outer)
    if not len(inner_ids):
        return boundary
    real_inner = inner_ids[np.argmax(sizes[inner_ids])]

    rest = boundary[vote_comps != real_inner]
    rest_comps = _boundary_sheets(rest)
    dominant = int(np.argmax(np.bincount(rest_comps)))
    return rest[rest_comps == dominant]


def _boundary_faces(cells: np.ndarray) -> np.ndarray:
    """Every face of a hex block that no second hex shares, inner and outer."""
    from engine.mesh.hex_mesh import HEX_FACES

    if not len(cells):
        return np.zeros((0, 4), dtype=np.int64)
    faces = np.concatenate([cells[:, f] for f in HEX_FACES], axis=0)
    key = np.sort(faces, axis=1)
    from engine.mesh.hex_mesh import _unique_rows

    inverse, counts = _unique_rows(key)
    return faces[counts[inverse] == 1]


def _layer_thickness_mm(labels: np.ndarray, affine: np.ndarray, region: str) -> float:
    """Mean thickness of a head layer, measured on the label volume."""
    from engine.anatomy import head_tissue
    from engine.anatomy.anatomy_labels import REGION_LABELS

    order = ["gray_matter", "white_matter", "brainstem", "cerebellum", "deep_nuclei",
             "csf", "skull", "scalp"]
    if region not in order:
        return 0.0
    inside = [REGION_LABELS[n] for n in order[:order.index(region)]]
    inner = np.isin(labels, inside)
    outer = inner | (labels == REGION_LABELS[region])
    if not inner.any() or not outer.any():
        return 0.0
    return float(head_tissue.layer_thickness_mm(inner, outer, affine))


def _scalp_thickness_cranial_mm(labels: np.ndarray, affine: np.ndarray) -> float:
    """Median skull-to-skin distance over the cranial vault (skin above the skull base), so
    the face and neck, centimetres deep, do not inflate the scalp shell's section."""
    from scipy import ndimage

    from engine.anatomy import head_tissue
    from engine.anatomy.anatomy_labels import REGION_LABELS

    labels = np.asarray(labels)
    cavity_ids = [REGION_LABELS[n] for n in ("gray_matter", "white_matter", "brainstem",
                                              "cerebellum", "deep_nuclei", "csf")]
    cavity = np.isin(labels, cavity_ids)
    inner = cavity | (labels == REGION_LABELS["skull"])
    head = labels > 0
    if not cavity.any() or not inner.any():
        return 0.0
    skin = head & ~ndimage.binary_erosion(head)
    skin &= head_tissue.skull_base_cut(cavity, affine, 0.0)
    if not skin.any():
        return 0.0
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    depth = fast_kernels.distance_transform_edt(~inner, sampling=spacing)[skin]
    return float(np.median(depth))


def _layer_mid_surface(nodes: np.ndarray, cells: np.ndarray,
                       thickness_mm: float,
                       exterior=None, in_surface_repair: bool = False
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """A layer of hexes as one shell sheet on its own nodes, at mid-thickness."""
    quads = _outer_shell_quads(nodes, cells, exterior)
    if not len(quads):
        return np.zeros((0, 3)), quads, 0.0
    import os

    if os.environ.get("ANATOMESH_DUMP_LAYER"):  # inputs for offline shell experiments
        np.savez(os.environ["ANATOMESH_DUMP_LAYER"], nodes=nodes, cells=cells, quads=quads,
                 thickness=thickness_mm)
    from engine.mesh import membranes

    used, local = np.unique(quads, return_inverse=True)
    local = local.reshape(quads.shape)
    verts = np.array(nodes[used], dtype=float)

    verts = membranes.relax_sheet(verts, local)

    centre = nodes[np.unique(cells)].mean(axis=0)
    p = verts[local]
    face_n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    if np.einsum("ij,ij->i", face_n, p.mean(axis=1) - centre).sum() < 0:
        face_n *= -1.0

    normal = np.zeros_like(verts)
    for col in range(4):
        np.add.at(normal, local[:, col], face_n)
    length = np.linalg.norm(normal, axis=1)
    good = length > 1e-9
    normal[good] /= length[good, None]

    from scipy.spatial import cKDTree

    inner = np.setdiff1d(np.unique(_boundary_faces(cells)), used)
    travel = np.full(len(verts), 0.5 * float(thickness_mm))
    if len(inner):
        reach, _ = cKDTree(nodes[inner]).query(verts)
        travel = np.minimum(travel, 0.5 * reach)
    verts[good] -= travel[good, None] * normal[good]

    verts = membranes.relax_sheet(verts, local)
    verts, _ = membranes.untangle_sheet(verts, local)
    from engine.mesh import shell_repair

    verts, _ = shell_repair.improve_sheet(verts, local, name="layer mid-surface")
    if in_surface_repair:
        from engine.mesh import sheet_quality

        verts, _ = sheet_quality.improve_sheet_in_surface(verts, local, name="layer mid-surface")
    verts, local, _ = membranes.drop_inverted_without_holing(verts, local,
                                                             name="layer mid-surface")
    return verts, local, float(thickness_mm)


_NOT_BRAIN = frozenset({"csf", "skull", "skull_diploe", "scalp", "soft_tissue", "dura", "pia",
                        "falx", "tentorium", "tentorium_osseous"})


def _brain_element_test(nodes: np.ndarray, solid_blocks: dict, affine: np.ndarray):
    """`f(points) -> bool`: True where a point lies inside a brain-tissue hex of the finished
    (smoothed) mesh, by an exact point-in-hex test (mesh.hex_geometry.point_in_hexes)."""
    from engine.mesh.hex_geometry import point_in_hexes

    cells, brain = [], []
    for name, block in solid_blocks.items():
        if len(block) == 0:
            continue
        cells.append(np.asarray(block))
        brain.append(np.full(len(block), name not in _NOT_BRAIN))
    from scipy.spatial import cKDTree

    cells = np.concatenate(cells)
    brain = np.concatenate(brain)
    tree = cKDTree(nodes[cells].mean(axis=1))

    def inside(points: np.ndarray) -> np.ndarray:
        host = point_in_hexes(points, nodes, cells, tree=tree)
        return (host >= 0) & brain[np.maximum(host, 0)]

    return inside


_HEX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [3, 4, 6, 7],
                      [1, 4, 5, 6]])


def _brain_volume_mm3(nodes: np.ndarray, solid_blocks: dict) -> float:
    """Volume of the brain-tissue hexes (CSF, skull and scalp excluded), five tets per hex."""
    total = 0.0
    for name, block in solid_blocks.items():
        if name in _NOT_BRAIN or len(block) == 0:
            continue
        corners = nodes[np.asarray(block)]
        for a, b, c, d in _HEX_TETS:
            p = corners[:, a]
            total += float(np.abs(np.einsum("ij,ij->i", corners[:, b] - p, np.cross(
                corners[:, c] - p, corners[:, d] - p))).sum()) / 6.0
    return total


_DTI_MIN_COVERAGE = 0.90

_DTI_MATERIALS = frozenset({"white_matter"})


def _run_hex_meshing(
    config: MeshingConfig, run_id: str, species: str, use_dti: bool,
    freesurfer_subject_dir: Optional[Path], labels: np.ndarray, affine: np.ndarray,
    run_dir: Path, seg_img, t0: float,
    rotation=None, keywords: Optional[set[str]] = None, progress=None,
) -> dict:
    """Default meshing path"""
    from engine.mesh import pipeline_hex
    from engine.anatomy.anatomy_labels import BRAIN_TISSUE, REGION_LABELS, cortical_surface_area

    pipeline_hex.HIERARCHICAL_SMOOTH_REPORTS.clear()

    real_masks: dict[str, Optional[np.ndarray]] = {"dura": None, "skull": None, "scalp": None}
    skull_measurement = None
    timing.mark("real masks")
    if freesurfer_subject_dir:
        for key, name in (("dura", "skull_inner_surface"), ("skull", "skull_outer_surface"),
                          ("scalp", "skin_outer_surface")):
            surf = _load_watershed_surface(run_dir, name, freesurfer_subject_dir)
            if surf is not None:
                real_masks[key] = _mesh_to_voxel_mask(surf, affine, labels.shape)
                logger.info("run_id=%s: %s from real watershed surface", run_id, key)

    if real_masks["skull"] is None or real_masks["scalp"] is None:
        derived = _layers_from_t1(run_dir, labels, affine, config)
        skull_measurement = derived.get("skull_report")
        keys = ["skull", "scalp"]
        if (getattr(config, "skull_source", "") in ("segmented", "charm")
                and getattr(config, "csf_from_image", True)):
            keys.append("dura")   # the CSF space as imaged, not a physiological offset
        for key in keys:
            if real_masks[key] is None and derived.get(key) is not None:
                real_masks[key] = derived[key]
                logger.info("run_id=%s: %s from the T1 (%s)", run_id, key,
                            derived["sources"].get(key))

    timing.mark("layers and membrane rules")
    wanted_layers = set(getattr(config, "auto_boundaries", None) or ())
    if getattr(config, "require_mri_derived_layers", True):
        missing = [k for k in ("skull", "scalp")
                   if k in wanted_layers and real_masks[k] is None]
        if missing:
            raise RuntimeError(
                f"{' and '.join(missing)} could not be derived from this subject's imaging "
                f"(no mri_watershed surface and no usable whole-head T1 in {run_dir}), and "
                f"MeshingConfig.require_mri_derived_layers is set. The only alternative is "
                f"an offset envelope dilated off the brain, which carries no real bone or "
                f"soft-tissue shape. Supply t1_head.nii.gz (prepare.prepare_from_freesurfer "
                f"copies it) or set require_mri_derived_layers=False to accept the envelope.")

    # falx / tentorium shape follows the species 
    config.membrane_profile = membranes.membrane_profile(
        species, getattr(config, "membrane_profile", "auto"))
    # animal heads: build at human size and scale back (the builders are tuned on the reference adult head)
    config.membrane_scale = 1.0 if (species or "human").lower() == "human" else None
    # the rules fitted to the lab's human models apply to human runs; animals keep theirs
    if getattr(config, "membrane_rules", "auto") == "auto":
        config.membrane_rules = ("lab_templates" if (species or "human").lower() == "human"
                                 else "2026-10-09")
    membranes.RULE_ASEG = None
    if getattr(config, "membrane_rules", "") == "anatomical":
        aseg_path = run_dir / "structures_aseg.nii.gz"
        if aseg_path.exists():
            aseg_img = nib.load(str(aseg_path))
            membranes.RULE_ASEG = (np.asarray(aseg_img.dataobj), aseg_img.affine)
        else:
            logger.warning("run_id=%s: membrane_rules='anatomical' but no %s", run_id, aseg_path.name)
    timing.mark("head mesh")
    with timing.stage("build_head_mesh"):
        result = pipeline_hex.build_head_mesh(
            labels, affine, config, freesurfer_subject_dir,
            dura_mask=real_masks["dura"], skull_mask=real_masks["skull"],
            scalp_mask=real_masks["scalp"], progress=progress)

    nodes = result["nodes"]
    head_labels, head_affine = result["head_labels"], result["affine"]
    solid_blocks = dict(result["solid_blocks"])
    shell_quad_blocks = dict(result["shell_quads"])

    logger.info("run_id=%s: %s", run_id, result["report"]["volume_cm3"])
    logger.info("run_id=%s: conformity %s", run_id, result["conformity"])

    solid_ortho_axes: dict[str, np.ndarray] = {}
    dti_report: dict = {"requested": bool(use_dti)}

    all_points = [nodes]
    timing.mark("parts and shells")
    node_offset = len(nodes)
    skull_fixed_nodes = np.zeros((0,), dtype=int)

    shell_blocks: dict[str, np.ndarray] = {}
    wanted = set(getattr(config, "auto_boundaries", None) or ())
    membrane_sheets: dict = dict(result.get("membrane_sheets") or {})
    conforming_sheets = set(result.get("conforming_sheets") or ())

    representation = dict(DEFAULT_REPRESENTATION)
    representation.update(getattr(config, "part_representation", None) or {})
    for _region, _kind in DEFAULT_REPRESENTATION.items():
        if _region not in FLIPPABLE_PARTS and representation.get(_region) != _kind:
            logger.info("run_id=%s: %s is always %s; ignoring the requested %s",
                        run_id, _region, _kind, representation.get(_region))
            representation[_region] = _kind
    deck_sheets: dict = {}
    for region in ("falx", "tentorium", "pia", "dura"):
        if region not in wanted:
            continue
        sheet = membrane_sheets.get(region)
        if sheet is None and region in shell_quad_blocks:
            continue                                 # conforming: hex faces on shared nodes
        if sheet is None:
            logger.info("run_id=%s: %s absent (source anatomy not present)", run_id, region)
            continue
        quad_verts, quad_faces = sheet
        quad_faces = _drop_degenerate_shells(quad_verts, quad_faces)
        if not len(quad_faces):
            continue
        deck_sheets[region] = (quad_verts, quad_faces)
        if representation.get(region) == "solid":
            quad_verts, cells = membranes.extrude_sheet(
                quad_verts, quad_faces, config.shell_thickness_dura)
            solid_blocks[region] = cells + node_offset
            logger.info("run_id=%s: %s as a solid, %d hexes", run_id, region, len(cells))
        else:
            shell_quad_blocks[region] = quad_faces + node_offset
            if region == "tentorium" and getattr(config, "membrane_profile", "") == "carnivore":
                # carnivores ossify the posteromedial tentorium (Klintworth 1968; ferret
                # tentorium osseum, He et al. 2002): its own part, on the skull card
                bony = membranes.osseous_tentorium_mask(
                    quad_verts, quad_faces, head_affine, np.asarray(head_labels).shape,
                    frac=float(getattr(config, "tentorium_osseous_frac",
                                       membranes.TENTORIUM_OSSEOUS_FRAC)))
                if bony.any() and (~bony).any():
                    shell_quad_blocks["tentorium_osseous"] = quad_faces[bony] + node_offset
                    shell_quad_blocks[region] = quad_faces[~bony] + node_offset
                    logger.info("run_id=%s: %d of %d tentorium quads are the bony "
                                "(posteromedial) tentorium", run_id, int(bony.sum()), len(bony))
        all_points.append(quad_verts)
        node_offset += len(quad_verts)

    welded = set(result.get("welded_regions", solid_blocks))
    grid_mm = float(np.mean(np.linalg.norm(np.asarray(head_affine)[:3, :3], axis=0)))
    if representation.get("skull") == "conforming" and "skull" in solid_blocks:
        # pia on the brain/CSF faces, dura on the CSF/skull faces, and the skull shell on the
        # skull layer's OUTER faces, all on the hex mesh's own nodes, so no two shells share a
        # surface. The skull hexes stay underneath as the diploe (a one-element layer for a
        # thin skull), so nothing is left hollow between dura and skull shell.
        outer = _layer_outer_faces("skull", solid_blocks)
        if len(outer):
            measured_mm = _layer_thickness_mm(head_labels, head_affine, "skull")
            if measured_mm <= 0.0:
                measured_mm = float(config.shell_thickness_skull)
            solid_blocks["skull_diploe"] = solid_blocks.pop("skull")
            if "skull" in welded:
                welded.discard("skull")
                welded.add("skull_diploe")
            shell_quad_blocks["skull"] = outer
            conforming_sheets.add("skull")           # on hex faces: cannot pass through a solid
            # three-layer skull (outer table, diploe, inner table): the shell carries the outer
            # table, taken as a third of the measured thickness
            config.shell_thickness_skull = measured_mm * SKULL_TABLE_SHARE
            logger.info("run_id=%s: skull as a conforming shell, %d quads on the outer skull "
                        "faces (%.2fmm table of a %.2fmm skull), %d diploe hexes", run_id,
                        len(outer), config.shell_thickness_skull, measured_mm,
                        len(solid_blocks["skull_diploe"]))
        representation["skull"] = "shell"
    if representation.get("skull") == "outer_shell":
        # template models:
        # no bone hexes, the skull is one shell on the outer faces of the intracranial block,
        # on the mesh's own nodes
        from engine.mesh.hex_mesh import _unique_rows
        from engine.mesh.pipeline_hex import _HEX_FACES_OUT

        blocks = [np.asarray(b) for b in solid_blocks.values()
                  if len(b) and np.asarray(b).ndim == 2 and np.asarray(b).shape[1] == 8]
        if blocks and "skull" not in solid_blocks:
            faces = np.concatenate(blocks)[:, _HEX_FACES_OUT].reshape(-1, 4)
            _inv, counts = _unique_rows(np.sort(faces, axis=1))
            shell_quad_blocks["skull"] = faces[counts[_inv] == 1]
            conforming_sheets.add("skull")
            logger.info("run_id=%s: skull as a shell on the %d outer faces of the intracranial "
                        "block (no bone hexes)", run_id, len(shell_quad_blocks["skull"]))
        representation["skull"] = "shell"
    if representation.get("scalp") == "conforming" and "scalp" in solid_blocks:
        # the skin as a flat shell, like the falx and tentorium: quads on the outer faces of
        # the head (shared nodes, nothing hollow underneath); every soft tissue under it
        # (scalp, face, eyes, muscle, fat, neck) stays as one solid part, "soft_tissue"
        skin = _layer_outer_faces("scalp", solid_blocks)
        if len(skin):
            measured_mm = _scalp_thickness_cranial_mm(head_labels, head_affine)
            if measured_mm <= 0.0:
                measured_mm = float(config.shell_thickness_scalp)
            solid_blocks["soft_tissue"] = solid_blocks.pop("scalp")
            if "scalp" in welded:
                welded.discard("scalp")
                welded.add("soft_tissue")
            shell_quad_blocks["scalp"] = skin
            conforming_sheets.add("scalp")           # on hex faces: cannot pass through a solid
            config.shell_thickness_scalp = measured_mm
            logger.info("run_id=%s: scalp as a skin shell, %d quads on the head's outer faces "
                        "(%.2fmm, median skull-to-skin over the vault), %d soft-tissue hexes "
                        "in one part", run_id, len(skin), measured_mm,
                        len(solid_blocks["soft_tissue"]))
        representation["scalp"] = "shell"
    for region in ("skull", "scalp"):
        if region not in solid_blocks or representation.get(region) not in ("shell", "auto"):
            continue
        attr = lsdyna_writer.SHELL_THICKNESS_ATTR[region]
        measured_mm = _layer_thickness_mm(head_labels, head_affine, region)
        if measured_mm <= 0.0:
            measured_mm = float(getattr(config, attr))
        if representation[region] == "auto":
            # a layer several elements thick stays as its conforming hexes (ADAPT, Li et al.
            # 2021); a mid-surface offset through it folds and collapses quads
            representation[region] = ("solid" if measured_mm >= SOLID_LAYER_MIN_ELEMENTS * grid_mm
                                      else "shell")
            logger.info("run_id=%s: %s %.2fmm thick on %.3gmm elements -> %s", run_id, region,
                        measured_mm, grid_mm, representation[region])
            if representation[region] == "solid":
                continue
        quad_verts, quads, measured_mm = _layer_mid_surface(
            nodes, solid_blocks[region], measured_mm,
            _probator_exterior(head_labels, head_affine, region, measured_mm),
            in_surface_repair=bool(getattr(config, "hex_quality_pass", False)))
        if not len(quads):
            continue
        del solid_blocks[region]
        welded.discard(region)
        shell_quad_blocks[region] = quads + node_offset
        all_points.append(quad_verts)
        node_offset += len(quad_verts)
        logger.info("run_id=%s: %s as a shell, %d quads at mid-thickness, "
                    "%.2fmm measured off the block (config %.2fmm)",
                    run_id, region, len(quads), measured_mm, getattr(config, attr))
        setattr(config, attr, measured_mm)

    for region, kind in representation.items():
        if kind == "auto":                           # the layer is not in this run
            representation[region] = DEFAULT_SHELL_OR_SOLID.get(region, "shell")

    if "skull" in solid_blocks:
        skull_fixed_nodes = np.unique(solid_blocks["skull"])
    elif "skull" in shell_quad_blocks:
        skull_fixed_nodes = np.unique(shell_quad_blocks["skull"])
    if getattr(config, "fix_skull_boundary", True) and len(skull_fixed_nodes):
        logger.warning(
            "run_id=%s: fully constraining %d skull nodes (*BOUNDARY_SPC_SET). Set "
            "MeshingConfig.fix_skull_boundary=False for a free-boundary model.",
            run_id, len(skull_fixed_nodes))
    lsdyna_writer.reset_dynamic_parts()
    parcel_report: dict = {}
    timing.mark("brain regions and parcellation")
    if getattr(config, "brain_regions", None) or getattr(config, "brain_granularity", None):
        from engine.anatomy import parcellation

        solid_blocks, parcel_report = parcellation.split_brain_parts(
            solid_blocks, nodes, head_affine, run_dir,
            granularity=getattr(config, "brain_granularity", None) or "lobe",
            regions=getattr(config, "brain_regions", None),
            parcel_reach_mm=getattr(config, "parcel_reach_mm", 3.0),
            scheme=getattr(config, "parcellation_scheme", "auto"))
        new_parts = [n for n in solid_blocks if n not in lsdyna_writer.PART_TABLE]
        if new_parts:
            lsdyna_writer.register_parts(new_parts, "gray_matter")
            welded = welded - set(BRAIN_TISSUE) | set(new_parts)

    structure_report: dict = {}
    if getattr(config, "brain_structures", None):
        from engine.anatomy import substructures

        solid_blocks, structure_report = substructures.split_structures(
            solid_blocks, nodes, run_dir, tuple(config.brain_structures))
        new_parts = [n for n in structure_report.get("parts", ()) if n not in lsdyna_writer.PART_TABLE]
        if new_parts:
            for name in new_parts:
                lsdyna_writer.register_parts([name], substructures.STRUCTURES[name].source)
            welded |= set(new_parts)

    if getattr(config, "merge_brain_parts", False) and not parcel_report.get("parts"):
        merge_set = tuple(BRAIN_TISSUE)
    else:
        merge_set = tuple(t for t in (getattr(config, "merge_tissues", None) or ()) if t in BRAIN_TISSUE)
    if merge_set:
        merged = [solid_blocks.pop(r) for r in merge_set if r in solid_blocks]
        if merged:
            solid_blocks["brain"] = np.concatenate(merged, axis=0)
            welded = (welded - set(merge_set)) | {"brain"}
            if "white_matter" in merge_set:
                solid_ortho_axes.pop("white_matter", None)
            logger.info("run_id=%s: merged %d brain tissues into one part, %d hexes",
                        run_id, len(merged), len(solid_blocks["brain"]))

    timing.mark("hex quality pass")
    if getattr(config, "hex_quality_pass", False) and result.get("lattice_colour") is not None:
        # on the final parts (lobes and structures included), so every boundary between two
        # parts is held where it is; before the axons, which are clipped to these hexes
        from engine.mesh import hex_quality

        own = [n for n, b in solid_blocks.items() if len(b) and int(np.max(b)) < len(nodes)]
        cells = np.concatenate([solid_blocks[n] for n in own])
        part = np.repeat(np.arange(len(own)), [len(solid_blocks[n]) for n in own])
        used, local = np.unique(cells, return_inverse=True)
        moved_nodes, quality_report = hex_quality.improve_hex_quality(
            nodes[used], local.reshape(cells.shape), part, result["lattice_colour"][used],
            target=float(getattr(config, "hex_quality_target", 0.98)))
        nodes[used] = moved_nodes                     # in place: all_points[0] is this array
        quality_report["parts"] = len(own)
        if pipeline_hex.HIERARCHICAL_SMOOTH_REPORTS:
            pipeline_hex.HIERARCHICAL_SMOOTH_REPORTS[-1]["hex_quality"] = quality_report

    beam_blocks: dict[str, np.ndarray] = {}
    timing.mark("vessel beams")
    vessel_result = _vessel_beam_segments(run_dir, head_affine)
    if vessel_result is not None:
        vessel_pts, vessel_segments = vessel_result
        all_points.append(vessel_pts)
        beam_blocks["vessels"] = vessel_segments + node_offset
        node_offset += len(vessel_pts)

    vascular_report: dict = {}
    timing.mark("vascular tree")
    if getattr(config, "vascular_beams", True):
        try:
            from engine.mesh import vascular

            pia_sheet = deck_sheets.get("pia")
            pia_verts = pia_sheet[0] if pia_sheet is not None else None
            if pia_verts is None and "pia" in conforming_sheets:
                pia_verts = nodes[np.unique(shell_quad_blocks["pia"])]
            tree = vascular.build_vascular_tree(
                deck_sheets.get("falx"), deck_sheets.get("tentorium"),
                pia_verts=pia_verts,
                spacing_mm=getattr(config, "vascular_beam_spacing_mm", 4.0))
            for name, (pts, segments) in tree.items():
                material, _diameter, title = vascular.VESSEL_CLASSES[name]
                lsdyna_writer.register_parts([name], material, element_type="BEAM")
                lsdyna_writer.PART_TITLES[name] = title
                beam_blocks[name] = segments + node_offset
                all_points.append(pts)
                node_offset += len(pts)
            if tree:
                vascular_report = {
                    name: {"beams": int(len(seg)),
                           "length_mm": round(float(np.linalg.norm(
                               pts[seg[:, 1]] - pts[seg[:, 0]], axis=1).sum()), 1)}
                    for name, (pts, seg) in tree.items()}
                logger.info("run_id=%s: vascular tree -- %s", run_id, vascular_report)
        except Exception:
            logger.exception("run_id=%s: the vascular tree failed; the deck ships "
                             "without vessels", run_id)

    arterial_report: dict = {}
    timing.mark("arteries")
    if getattr(config, "arterial_beams", False):
        try:
            from engine.mesh import arteries as _arteries
            from engine.anatomy.anatomy_labels import REGION_LABELS

            outer = (REGION_LABELS["skull"], REGION_LABELS["scalp"])
            envelope = (~np.isin(head_labels, outer)) & (head_labels != 0)
            territories = {
                part: nodes[block].mean(axis=1)
                for part, block in solid_blocks.items()
                if part.startswith(("lh_", "rh_")) or part in ("cerebellar_cortex",
                                                               "brainstem")}
            tree, arterial_report = _arteries.build_arterial_tree(
                run_dir / "t1_brain.nii.gz", run_dir,
                occurrence_threshold=getattr(config, "arterial_occurrence_threshold",
                                             _arteries.DEFAULT_OCCURRENCE_THRESHOLD),
                spacing_mm=getattr(config, "arterial_beam_spacing_mm", 3.0),
                envelope=(envelope, head_affine),
                territories=territories or None)
            if tree and not arterial_report.get("frame_ok", True):
                logger.error("run_id=%s: the vessel atlas failed its own frame check; "
                             "the deck ships WITHOUT arteries", run_id)
                tree = {}
            for name, (pts, segments) in tree.items():
                material, _diameter, title = _arteries.ARTERY_CLASSES[name]
                lsdyna_writer.register_parts([name], material, element_type="BEAM")
                lsdyna_writer.PART_TITLES[name] = title
                beam_blocks[name] = segments + node_offset
                all_points.append(pts)
                node_offset += len(pts)
            if tree:
                logger.info("run_id=%s: arteries -- %d part(s), %d beams, %.0fmm",
                            run_id, len(tree), arterial_report["beams"],
                            arterial_report["length_mm"])
        except Exception:
            logger.exception("run_id=%s: the arterial tree failed; the deck ships "
                             "without arteries", run_id)

    axon_report: dict = {}
    axon_extras: dict = {}
    timing.mark("axon tracts")
    if (use_dti or (run_dir / "tracts_t1.npz").exists()) and getattr(config, "axon_tracts", True):
        try:
            from engine.anatomy import tractography

            npz_path = run_dir / "tracts_t1.npz"
            fa_path = run_dir / "dti_fa.nii.gz"
            axon_envelope = np.isin(labels, list(SOLID_LABELS.values()))
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"no {npz_path.name}: tractography produced no tractogram")
            blob = np.load(str(npz_path), allow_pickle=False)
            points = blob["points"]          # an npz decompresses on every [] access
            bounds = np.concatenate([[0], np.cumsum(blob["lengths"])])
            streams = [points[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
            tract_ids = tract_names = None
            if "cluster" in blob.files and "names" in blob.files:
                tract_ids = np.asarray(blob["cluster"]).astype(int)
                tract_names = [str(n) for n in blob["names"]]
            engine, parameters = str(blob["engine"]) if "engine" in blob.files else "native", {
                "source": npz_path.name,
                "spacing_mm": getattr(config, "axon_beam_spacing_mm", 3.0),
                "min_length_mm": getattr(config, "axon_min_length_mm", 20.0),
                "clip": getattr(config, "axon_clip", "mesh")}
            from engine.anatomy import parcellation as _parcellation

            parc_data = None
            parc_path = run_dir / _parcellation.SCHEMES["aparc"]
            if parc_path.exists():
                parc_img = nib.load(str(parc_path))
                parc_data = (np.asarray(parc_img.dataobj),
                               np.asarray(parc_img.affine, dtype=float))
            else:
                logger.warning("run_id=%s: no %s, so the axon tracts ship without a bundle "
                               "census -- the report will not say which tracts are missing",
                               run_id, parc_path.name)
            shortest_mm = getattr(config, "axon_min_length_mm", 20.0)
            in_mesh_brain = _brain_element_test(nodes, solid_blocks, head_affine)
            clip = getattr(config, "axon_clip", "mesh")
            # a beam is never longer than one element
            element_mm = float(np.linalg.norm(np.asarray(head_affine)[:3, :3], axis=0).min())
            beam_mm = float(getattr(config, "axon_beam_spacing_mm", 3.0))
            if beam_mm > element_mm + 1e-9:
                logger.info("run_id=%s: axon beam %.4g mm is longer than the %.4g mm element; "
                            "using %.4g mm", run_id, beam_mm, element_mm, element_mm)
                beam_mm = element_mm
            max_streamlines = int(getattr(config, "axon_max_tracts", 5000))
            density: dict = {}
            target = (getattr(config, "axon_volume_per_beam_mm2", None)
                      or getattr(config, "axon_volume_per_tract_mm2", None))
            section = getattr(config, "axon_section_mm2", None)
            fill = getattr(config, "axon_fill_fraction", None)
            if section and fill:
                # length x section = fill x volume  <=>  volume / (beams x beam) = section / fill
                target = float(section) / float(fill)
            brain_mm3 = _brain_volume_mm3(nodes, solid_blocks)
            wanted_beams = None
            if target:
                # N counts axon beam elements: brain volume / (N x beam length) = target
                wanted_beams = tractography.beam_budget(brain_mm3, beam_mm, float(target))
                # a pool a few times what the budget needs, so clipping has spare streamlines
                pool = tractography.streamline_pool(streams, beam_mm, wanted_beams)
                density = {"brain_volume_mm3": round(brain_mm3, 2), "beam_mm": beam_mm,
                           "target_mm2": float(target), "beams_wanted": wanted_beams,
                           "streamlines_available": len(streams),
                           "streamline_pool": min(pool, max_streamlines, len(streams))}
                max_streamlines = min(max_streamlines, pool)
            if tract_ids is not None and len(streams) > max_streamlines:
                pick = tractography.select_per_tract(tract_ids, max_streamlines)
                streams = [streams[i] for i in pick]
                tract_ids = tract_ids[pick]
            study_mm = tuple(getattr(config, "axon_beam_study_mm", ()) or ())
            study = {}
            if study_mm:
                study = tractography.beam_length_study(
                    streams, in_mesh_brain, study_mm, tract_ids, tract_names)
            axon_parts, axon_report = tractography.build_axon_beams(
                streams, fa_path if fa_path.exists() else None,
                spacing_mm=beam_mm,
                min_length_mm=shortest_mm,
                max_streamlines=max_streamlines, max_beams=wanted_beams,
                max_tracts=10 ** 9 if tract_ids is not None else max_streamlines,
                engine=engine, parameters=parameters,
                envelope=(axon_envelope, affine) if clip == "atlas" else None,
                inside=in_mesh_brain if clip == "mesh" else None,
                parc_data=parc_data, tract_ids=tract_ids, tract_names=tract_names)
            axon_report["clip"] = clip
            axon_report["beam_mm"] = beam_mm
            n_beams = sum(v["beams"] for v in axon_report.get("parts", {}).values())
            density.update(brain_volume_mm3=round(brain_mm3, 2), beam_mm=beam_mm,
                           beams_used=n_beams,
                           streamlines_used=axon_report.get(
                               "streamlines_after_budget",
                               axon_report.get("streamlines_surviving_clip", len(streams))))
            density["volume_per_beam_mm2"] = round(
                brain_mm3 / max(n_beams * beam_mm, 1e-12), 4)
            if section:
                density["axon_section_mm2"] = float(section)
                density["fill_fraction"] = round(
                    float(section) * n_beams * beam_mm / max(brain_mm3, 1e-12), 4)
            if wanted_beams and n_beams < 0.98 * wanted_beams:
                logger.warning("run_id=%s: %d beams give volume/(N x beam) = %s but only %d "
                               "fit (raise axon_max_tracts or use a larger tractogram)",
                               run_id, wanted_beams, target, n_beams)
            axon_report["density"] = density
            logger.info("run_id=%s: brain %.1f mm3 / (%d beams x %.4g mm) = %.4f mm2 "
                        "(%s streamlines)", run_id, brain_mm3, n_beams, beam_mm,
                        density["volume_per_beam_mm2"], density["streamlines_used"])
            if study:
                axon_report["beam_length_study"] = study
            beam_tract = axon_report.pop("beam_tract", {})
            if tract_names and beam_tract:
                np.savez_compressed(
                    assert_write_path(run_dir / "axon_beam_tracts.npz"),
                    names=np.array(tract_names),
                    **{name: beam_tract[name] for name in sorted(beam_tract)})
            final_check = {}
            for name, (pts, segments, _fa, _t) in axon_parts.items():
                ok = in_mesh_brain(pts)
                mid = in_mesh_brain(0.5 * (pts[segments[:, 0]] + pts[segments[:, 1]]))
                final_check[name] = {"nodes_outside": int((~ok).sum()),
                                     "midpoints_outside": int((~mid).sum())}
            axon_report["inside_final_mesh"] = final_check
            for name, (pts, segments, beam_fa, tangent) in axon_parts.items():
                lsdyna_writer.register_parts([name], name, element_type="BEAM")
                lsdyna_writer.PART_TITLES[name] = (
                    f"Axon Tract FA {name.removeprefix('axon_fa')}+")
                beam_blocks[name] = segments + node_offset
                all_points.append(pts)
                node_offset += len(pts)
                axon_extras[name] = (beam_fa, tangent)
            logger.info("run_id=%s: axon tracts (%s) -- %s", run_id, engine,
                        axon_report.get("parts", {}))
        except Exception:
            logger.exception("run_id=%s: the axon tracts failed; the deck ships without "
                             "them", run_id)
            axon_report = {"status": "failed, see the traceback in the log"}

    timing.mark("dti fibre axes")
    if use_dti:
        e1_path = run_dir / "dti_e1.nii.gz"
        fibrous = [n for n in solid_blocks
                   if lsdyna_writer.PART_TABLE.get(n, (0, "", ""))[2] in _DTI_MATERIALS]
        if "brain" in solid_blocks and "brain" not in fibrous:
            fibrous.append("brain")
        if not fibrous:
            dti_report["status"] = "no white-matter part in this mesh"
        elif not e1_path.exists():
            dti_report["status"] = f"{e1_path.name} is not in the run directory"
        else:
            sampled, total, covered_n = {}, 0, 0
            for name in fibrous:
                centroids = nodes[solid_blocks[name]].mean(axis=1)
                axes = _sample_dti_e1(e1_path, centroids, head_affine)
                sampled[name] = axes
                total += len(axes)
                covered_n += int((np.linalg.norm(axes, axis=1) > 1e-6).sum())
            covered = covered_n / max(total, 1)
            dti_report.update(n_elements=int(total), coverage=round(covered, 4),
                              parts=sorted(fibrous))
            if covered < _DTI_MIN_COVERAGE:
                dti_report["status"] = (
                    f"only {100 * covered:.1f}% of {total} white-matter elements got a "
                    f"fibre axis, under the {100 * _DTI_MIN_COVERAGE:.0f}% floor")
                logger.error(
                    "run_id=%s: DTI REFUSED. %s. The eigenvector volume is almost certainly "
                    "not registered into this subject's T1 frame -- check dti_e1.nii.gz's "
                    "affine against head_labels.nii.gz. Shipping the isotropic deck.",
                    run_id, dti_report["status"])
            else:
                solid_ortho_axes.update(sampled)
                dti_report["status"] = "ok"
                logger.info("run_id=%s: DTI fibre axes on %.1f%% of %d elements across %d "
                            "part(s): %s", run_id, 100 * covered, total, len(fibrous),
                            ", ".join(sorted(fibrous)))

    timing.mark("orientation")
    points = np.concatenate(all_points, axis=0)

    from engine.anatomy import orientation

    upright = orientation.upright_transform(head_labels, head_affine)
    orientation_report = orientation.report(upright, head_labels, head_affine)
    orientation_report["matrix"] = np.asarray(upright, dtype=float).round(9).tolist()
    if orientation_report["applied"]:
        points = orientation.apply_points(upright, points)
        solid_ortho_axes = {k: orientation.apply_vectors(upright, v)
                            for k, v in solid_ortho_axes.items()}
        head_affine = upright @ head_affine
        logger.info("run_id=%s: head re-seated on its own midsagittal plane (%s)",
                    run_id, orientation_report)

    void_report = {"superseded_by": "*CONSTRAINED_LAGRANGE_IN_SOLID"}

    beam_host_part: object = "csf"
    embedding_report: dict = {}
    timing.mark("beam embedding")
    if beam_blocks:
        try:
            from engine.mesh import embedding

            beam_points = {name: points[np.unique(seg)] for name, seg in beam_blocks.items()}
            hosts, embedding_report = embedding.choose_hosts(
                beam_points, solid_blocks, points,
                element_size_mm=getattr(config, "element_size_mm", None) or 2.0)
            if hosts:
                beam_host_part = hosts
            floating = [n for n in beam_blocks if n not in hosts]
            if floating:
                logger.error("run_id=%s: %d beam part(s) have no solid host and will carry "
                             "mass with no load path: %s", run_id, len(floating), floating)
        except Exception:
            logger.exception("run_id=%s: host selection failed; falling back to coupling "
                             "every beam into csf", run_id)

    mesh_blocks = {
        "points": points,
        "solid": solid_blocks,
        "solid_ortho_axes": solid_ortho_axes,
        "shell": shell_blocks,
        "shell_quad": shell_quad_blocks,
        "sph": {},
        "beam": beam_blocks,
        "beam_host_part": beam_host_part,
        "skull_fixed_nodes": skull_fixed_nodes,
        "ale_volume_fraction": {},
        "representation": representation,
        "welded_parts": welded,
        "offset_tied_pairs": {frozenset(p) for p in result.get("tied_pairs", ())},
    }
    n_elements = (sum(len(v) for v in solid_blocks.values())
                  + sum(len(v) for v in shell_blocks.values())
                  + sum(len(v) for v in shell_quad_blocks.values())
                  + sum(len(v) for v in beam_blocks.values()))

    timing.mark("head labels and morphometry")
    head_labels_path = assert_write_path(run_dir / "head_labels.nii.gz")
    nib.save(nib.Nifti1Image(head_labels.astype(np.int16), head_affine), str(head_labels_path))

    morphometry_path = None
    try:
        from engine.anatomy import morphometry

        aseg = None
        if freesurfer_subject_dir is not None:
            aseg_file = Path(freesurfer_subject_dir) / "mri" / "aseg.mgz"
            if aseg_file.exists():
                aseg_img = nib.load(str(aseg_file))
                aseg = np.asarray(aseg_img.dataobj).astype(np.int32)
                if aseg.shape != head_labels.shape:
                    logger.info("run_id=%s: aseg is %s against head labels %s; "
                                "skipping the subcortical size table",
                                run_id, aseg.shape, head_labels.shape)
                    aseg = None
        parcel_volume = parcel_names = None
        from engine.anatomy import parcellation as _parcellation

        found = _parcellation.find_parcellation(
            run_dir, scheme=getattr(config, "parcellation_scheme", "auto"))
        if found is not None:
            parcel_path, scheme = found
            parcel_img = nib.load(str(parcel_path))
            parcel_volume = np.asarray(parcel_img.dataobj).astype(np.int32)
            parcel_names = _parcellation.parcel_names(parcel_volume, scheme, "region")

        sizes = morphometry.report(head_labels, head_affine, aseg=aseg,
                                   parcellation=parcel_volume, parcel_names=parcel_names,
                                   subject=run_id,
                                   freesurfer_subject_dir=freesurfer_subject_dir)
        morphometry_path = str(morphometry.write_table(
            sizes, assert_write_path(run_dir / f"morphometry_{run_id}.tsv")))
    except Exception:
        logger.exception("run_id=%s: morphometry failed; continuing without it", run_id)
        sizes = {}
        parcel_volume = parcel_names = None

    timing.mark("label previews")
    preview_paths: dict[str, str] = {}
    try:
        from engine.anatomy import atlas_preview

        t1_file = run_dir / "t1_head.nii.gz"
        backdrop = np.asarray(nib.load(str(t1_file)).dataobj) if t1_file.exists() else None
        from engine.anatomy.anatomy_labels import LABEL_REGIONS as _LABEL_REGIONS

        for name, volume, names in (("head_labels", head_labels, dict(_LABEL_REGIONS)),
                                    ("parcellation", parcel_volume, parcel_names)):
            if volume is None:
                continue
            out = atlas_preview.render(
                volume, assert_write_path(run_dir / f"preview_{name}_{run_id}.png"),
                background=backdrop if backdrop is not None
                and backdrop.shape == volume.shape else None,
                title=name, names=names)
            if out is not None:
                preview_paths[name] = str(out)
    except Exception:
        logger.exception("run_id=%s: preview rendering failed; continuing", run_id)

    timing.mark("deck writing")
    keywords = lsdyna_writer.resolve_keyword_selection(keywords)
    if not getattr(config, "fix_skull_boundary", True):
        keywords = frozenset(keywords) - {"boundary_spc"}

    mesh_blocks = lsdyna_writer.prepare_mesh_blocks_for_export(
        mesh_blocks, topology_strict=getattr(config, "topology_gate_strict", True))
    points = mesh_blocks["points"]
    nodti_k_path = run_dir / "mesh_nodti.k"
    lsdyna_writer.write_keyword_file(nodti_k_path, run_id, species, False, mesh_blocks, config,
                                      rotation=rotation, keywords=set(keywords))
    dti_k_path = None
    if use_dti and solid_ortho_axes:
        dti_k_path = run_dir / "mesh_dti.k"
        lsdyna_writer.write_keyword_file(dti_k_path, run_id, species, True, mesh_blocks, config,
                                          rotation=rotation, keywords=set(keywords))
    elif use_dti:
        logger.error("run_id=%s: --dti was requested and NO mesh_dti.k was written: %s. "
                     "The deliverable is %s, which carries no fibre axes.",
                     run_id, dti_report.get("status", "unknown"), nodti_k_path.name)

    if axon_extras:
        try:
            first_eid, eid = {}, 1
            for group in ("solid", "shell_quad", "shell", "sph"):
                for cells in mesh_blocks.get(group, {}).values():
                    eid += len(cells)
            for name, cells in mesh_blocks.get("beam", {}).items():
                first_eid[name] = eid
                eid += len(cells)
            from engine.anatomy import tractography

            tractography.write_fa_sidecar(
                assert_write_path(run_dir / "axon_fa.csv"),
                {n: (None, mesh_blocks["beam"][n], fa, tangent)
                 for n, (fa, tangent) in axon_extras.items()},
                first_eid)
        except Exception:
            logger.exception("run_id=%s: the axon FA sidecar failed to write; the deck is "
                             "unaffected", run_id)

    timing.mark("deck extras")
    try:
        from engine.pipeline import label_volume

        solid_items = [(name, np.asarray(block)) for name, block in solid_blocks.items()
                       if np.asarray(block).ndim == 2 and np.asarray(block).shape[1] == 8
                       and len(block)]
        if solid_items:
            coords = np.concatenate([points[block] for _, block in solid_items])
            part_ids = np.concatenate([np.full(len(block), lsdyna_writer.PART_TABLE[name][0])
                                       for name, block in solid_items])
            names = {lsdyna_writer.PART_TABLE[name][0]: name for name, _ in solid_items}
            preview_paths["quality_histograms"] = str(label_volume.write_histograms(
                coords, run_dir / "FinalHistograms.png", run_id, volume_unit="mm"))
            preview_paths["label_sections"] = str(label_volume.write_sections(
                coords, part_ids, run_dir / "Sections.png", run_id, names))
            from engine.export import surface_preview

            preview_paths["surface"] = str(surface_preview.write(
                points, [(lsdyna_writer.PART_TABLE[name][0], block) for name, block in solid_items],
                names, run_dir / "surface.npz"))
    except Exception:
        logger.exception("run_id=%s: quality figures failed; continuing", run_id)

    timing.mark("vtk and quality report")
    cells = [("hexahedron", block) for block in solid_blocks.values()] + \
            [("triangle", tris) for tris in shell_blocks.values()] + \
            [("quad", quads) for quads in shell_quad_blocks.values()]
    vtk_path = run_dir / "mesh.vtk"
    quality: dict = {}
    if cells:
        mesh = meshio.Mesh(points=points, cells=cells)
        meshio.write(str(assert_write_path(vtk_path)), mesh)
        np.savez_compressed(
            str(assert_write_path(run_dir / "mesh_parts.npz")), points=points,
            **{f"hex:{n}": b for n, b in solid_blocks.items()},
            **{f"quad:{n}": q for n, q in shell_quad_blocks.items()},
            **{f"tri:{n}": t for n, t in shell_blocks.items()},
            part_ids=np.array([[lsdyna_writer.PART_TABLE[n][0] if n in lsdyna_writer.PART_TABLE else -1]
                               for n in list(solid_blocks) + list(shell_quad_blocks) + list(shell_blocks)]),
            part_names=np.array(list(solid_blocks) + list(shell_quad_blocks) + list(shell_blocks)))
        from engine.export import validate as _validate

        quality = _validate.write_quality_report(mesh, run_dir, run_id)
        logger.info("run_id=%s: %s", run_id, _validate.format_histogram(quality["histogram"]))

    timing.mark("overlap and membrane reports")
    from engine.export import validate as _validate_overlap

    try:
        shell_integrity = _validate_overlap.shell_integrity_report(
            points, shell_quad_blocks, solid_blocks, trianguli=shell_blocks,
            host_part_names=("csf", "scalp", "soft_tissue"), conforming=conforming_sheets)
    except Exception:
        logger.exception("run_id=%s: the shell integrity check failed", run_id)
        shell_integrity = {"status": "failed, see the traceback in the log"}

    overlap = _validate_overlap.part_overlap_report(
        points, head_affine, head_labels, solid_blocks,
        membrane_blocks={n: q for n, q in shell_quad_blocks.items()
                         if n in ("falx", "tentorium", "pia", "dura")},
        shell_blocks=shell_quad_blocks,
        min_clearance_mm=getattr(config, "membrane_pair_gap_mm", 1.0),
        shell_in_solid=shell_integrity.get("intersectio_omnium"),
        conforming=conforming_sheets)

    metrics = {
        "run_id": run_id,
        "mesh_k_path": str(dti_k_path or nodti_k_path),
        "mesh_k_path_nodti": str(nodti_k_path),
        "mesh_k_path_dti": str(dti_k_path) if dti_k_path else None,
        "mesh_vtk_path": str(vtk_path) if cells else None,
        "head_labels_path": str(head_labels_path),
        "n_nodes": int(len(points)),
        "n_elements": int(n_elements),
        "conforming": bool(result["conformity"]["conforming"]),
        "n_hanging_nodes": int(result["conformity"]["n_hanging_nodes_on_lattice"]),
        "n_boundary_edges_unpaired": int(result["conformity"]["n_boundary_edges_unpaired"]),
        "conformity_blocks": {k: {kk: vv for kk, vv in v.items() if kk != "n_hanging_nodes"}
                              for k, v in result["conformity"].get("blocks", {}).items()},
        "part_overlap": overlap,
        "shell_integrity": shell_integrity,
        "membrane_void": void_report,
        "orientation": orientation_report,
        "falx_centering": _validate_overlap.falx_centering_report(
            points, shell_quad_blocks.get("falx"), solid_blocks),
        "shell_repair": dict(membranes.SHEET_REPAIR_REPORTS),
        "hierarchical_smoothing": list(pipeline_hex.HIERARCHICAL_SMOOTH_REPORTS),
        "smoothing_parameters": {
            "sigma_outer_vox": getattr(config, "label_smoothing_sigma_vox", None),
            "iterations_outer": getattr(config, "label_smoothing_iterations", None),
            "sigma_inner_vox": getattr(config, "label_smoothing_sigma_inner_vox", None),
            "iterations_inner": getattr(config, "label_smoothing_iterations_inner", None),
            "mesh_iterations": getattr(config, "hierarchical_smoothing_iterations", None),
            "lambda": getattr(config, "hierarchical_smoothing_lambda", None)},
        "membrane_containment": _validate_overlap.membrane_containment_report(
            points, head_affine, head_labels,
            {n: q for n, q in shell_quad_blocks.items()
             if n in ("falx", "tentorium", "pia", "dura")},
            measure_dura_band=bool(getattr(config, "membrane_field_placement", False)),
            conforming=conforming_sheets),
        "membrane_pair_clearance": _validate_overlap.membrane_pair_clearance_report(
            points, {n: q for n, q in shell_quad_blocks.items()
                     if n in ("falx", "tentorium", "pia", "dura")},
            min_clearance_mm=getattr(config, "membrane_pair_gap_mm", 1.0),
            conforming=conforming_sheets),
        "cortical_area_cm2": round(cortical_surface_area(head_labels, head_affine), 1),
        "volume_cm3": result["report"]["volume_cm3"],
        "element_sizes_mm": result.get("element_sizes", {}),
        "tied_pairs": ["+".join(p) for p in result.get("tied_pairs", ())],
        "timings": result["report"].get("timings", {}),
        "layer_sources": result["report"].get("sources", {}),
        "label_smoothing": result["report"].get("label_smoothing", {}),
        "sealed_voxels": result["report"].get("sealed_voxels", {}),
        "skull_measurement": skull_measurement,
        "region_scale": result["report"].get("region_scale", {}),
        "parcellation": parcel_report,
        "structures": structure_report,
        "vascular": vascular_report,
        "arteries": arterial_report,
        "dti": dti_report,
        "tractography": axon_report,
        "embedding": embedding_report,
        "representation": {k: v for k, v in representation.items()
                           if k in solid_blocks or k in shell_quad_blocks},
        "scaled_jacobian": quality,
        "morphometry_path": morphometry_path,
        "morphometry": sizes,
        "preview_paths": preview_paths,
        "wall_time_sec": time.time() - t0,
    }
    timing.mark("scorecard and metrics")
    try:                                      # per-part grade against the ADAPT bar
        from engine.export import scorecard

        card = scorecard.score(Path(dti_k_path or nodti_k_path))
        with open(assert_write_path(run_dir / "scorecard.json"), "w") as f:
            json.dump(card, f, indent=1)
        metrics["scorecard"] = {k: card[k] for k in ("passed", "failing_parts", "n_negative",
                                                     "solid_sj_min", "char_min_mm")}
        logger.info("run_id=%s: scorecard %s", run_id, metrics["scorecard"])
    except Exception:  # noqa: BLE001 - a report must not sink the run
        logger.exception("run_id=%s: scorecard failed", run_id)
    with open(assert_write_path(run_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("run_meshing complete: %s", metrics)
    return metrics


def run_meshing(
    config: MeshingConfig, run_id: str, species: str = "human", use_dti: bool = False,
    freesurfer_subject_dir: Optional[Path] = None,
    rotation=None, keywords: Optional[set[str]] = None, progress=None,
) -> dict:
    """Generate the full multi-material mesh and export it to LS-DYNA .k."""
    t0 = time.time()
    run_dir = assert_write_path(outputs_root() / run_id)
    seg_path = run_dir / "segmentation.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"No segmentation for run_id={run_id!r} at {seg_path}; run prepare.py first")

    seg_img = nib.load(str(seg_path))
    labels = np.asarray(seg_img.dataobj).astype(int)
    affine = seg_img.affine

    return _run_hex_meshing(config, run_id, species, use_dti, freesurfer_subject_dir,
                            labels, affine, run_dir, seg_img, t0,
                            rotation=rotation, keywords=keywords, progress=progress)

def main() -> None:
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="AnatoMesh mesh generation + LS-DYNA export")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--species", default="human")
    parser.add_argument("--use-dti", action="store_true")
    args = parser.parse_args()

    metrics = run_meshing(MeshingConfig(), args.run_id, species=args.species, use_dti=args.use_dti)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
