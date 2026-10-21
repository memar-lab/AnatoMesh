"""LS-DYNA R12 keyword (.k) file serialization."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Optional

import meshio
import numpy as np

from engine.paths import (
    READ_ONLY_ROOT,
    READ_WRITE_ROOT,
    FilesystemViolationError,
    assert_read_only_path,
    safe_open_write,
)

logger = logging.getLogger(__name__)

PART_TABLE: dict[str, tuple[int, str, str]] = {
    "gray_matter":   (1, "SOLID", "gray_matter"),
    "white_matter":  (2, "SOLID", "white_matter"),  
    "brainstem":     (3, "SOLID", "brainstem"),
    "cerebellum":    (4, "SOLID", "cerebellum"),
    "deep_nuclei":   (5, "SOLID", "deep_nuclei"),
    "skull":         (6, "SHELL", "skull"),          
    "dura":          (7, "SHELL", "membrane"),
    "falx":          (8, "SHELL", "membrane"),
    "tentorium":     (9, "SHELL", "membrane"),
    "pia":           (10, "SHELL", "membrane"),
    "csf":           (11, "SOLID", "csf"),   
    "vessels":       (12, "BEAM", "vessel"),
    "scalp":         (13, "SHELL", "scalp"),         
    "brain":         (14, "SOLID", "gray_matter"),   
    "tentorium_osseous": (15, "SHELL", "skull"),     # carnivores: bony posteromedial tentorium
    "skull_diploe":  (16, "SOLID", "skull"),         # "conforming" skull: hexes under the table shell
    "soft_tissue":   (17, "SOLID", "scalp"),         # "conforming" scalp: all soft tissue under the skin shell
}

PART_TITLES: dict[str, str] = {name: name.replace("_", " ").title() for name in PART_TABLE}
PART_TITLES["csf"] = "CSF"

_FIRST_DYNAMIC_PID = 20
_STATIC_PART_TABLE = dict(PART_TABLE)
_STATIC_PART_TITLES = dict(PART_TITLES)

def reset_dynamic_parts() -> None:
    """Drop every run-time-registered part."""
    for name in [n for n in PART_TABLE if n not in _STATIC_PART_TABLE]:
        del PART_TABLE[name]
        PART_TITLES.pop(name, None)

def register_parts(names, material_key: str, element_type: str = "SOLID") -> dict[str, int]:
    """Give a set of run-time regions their own part ids, sections and materials."""
    if material_key not in HUMAN_ADULT_DEFAULTS:
        raise ValueError(f"unknown material key {material_key!r}")
    out: dict[str, int] = {}
    taken = [p for p, _e, _m in PART_TABLE.values() if p >= _FIRST_DYNAMIC_PID]
    pid = max(taken, default=_FIRST_DYNAMIC_PID - 1) + 1
    static_max = max(p for p, _e, _m in _STATIC_PART_TABLE.values())
    if _FIRST_DYNAMIC_PID <= static_max:
        raise ValueError(f"_FIRST_DYNAMIC_PID={_FIRST_DYNAMIC_PID} collides with the static "
                         f"part block, which reaches {static_max}")
    for name in sorted(names):
        if name in PART_TABLE:
            out[name] = PART_TABLE[name][0]
            continue
        PART_TABLE[name] = (pid, element_type, material_key)
        PART_TITLES[name] = name.replace("_", " ").title()
        out[name] = pid
        pid += 1
    logger.info("registered %d run-time part(s) on material %r, ids %d-%d",
                len(out), material_key, min(out.values(), default=0),
                max(out.values(), default=0))
    return out

SHELL_THICKNESS_ATTR: dict[str, str] = {
    "skull": "shell_thickness_skull",
    "dura": "shell_thickness_dura",
    "falx": "shell_thickness_dura",
    "tentorium": "shell_thickness_dura",
    "tentorium_osseous": "shell_thickness_dura",
    "pia": "shell_thickness_pia",
    "scalp": "shell_thickness_scalp",
}

HUMAN_ADULT_DEFAULTS: dict[str, dict] = {
    "gray_matter":  {"density": 1.04e-9, "mu1": 0.026, "alpha1": -4.7, "G_inf": 0.8,
                      "prony_g": [0.4, 0.2], "prony_tau": [0.008, 0.15]},
    "white_matter": {"density": 1.04e-9, "mu1": 0.031, "alpha1": -4.7, "G_inf": 0.9,
                      "prony_g": [0.4, 0.2], "prony_tau": [0.008, 0.15]},
    "brainstem":    {"density": 1.04e-9, "mu1": 0.034, "alpha1": -4.7, "G_inf": 1.0,
                      "prony_g": [0.4, 0.2], "prony_tau": [0.008, 0.15]},
    "cerebellum":   {"density": 1.04e-9, "mu1": 0.028, "alpha1": -4.7, "G_inf": 0.85,
                      "prony_g": [0.4, 0.2], "prony_tau": [0.008, 0.15]},
    "deep_nuclei":  {"density": 1.04e-9, "mu1": 0.030, "alpha1": -4.7, "G_inf": 0.9,
                      "prony_g": [0.4, 0.2], "prony_tau": [0.008, 0.15]},
    "skull":        {"density": 1.9e-9, "E": 6500.0, "PR": 0.22, "yield_stress": 90.0,
                      "damage_exponent": 2.0},
    "membrane":     {"density": 1.13e-9, "E": 31.5, "PR": 0.45},
    "csf":          {"density": 1.0e-9, "C0": 0.0, "C1": 2.25e6, "C2": 0.0, "C3": 0.0,
                      "GAMMA0": 0.0, "E0": 0.0, "V0": 1.0},
    "vessel":       {"density": 1.05e-9, "E": 5.0, "PR": 0.45, "radius_mm": 1.0},
    "scalp":        {"density": 1.13e-9, "E": 16.7, "PR": 0.42},
    "vessel_sinus":    {"density": 1.05e-9, "E": 10.0, "PR": 0.45, "radius_mm": 5.0},
    "vessel_bridging": {"density": 1.05e-9, "E": 6.0, "PR": 0.45, "radius_mm": 0.9},
    "vessel_artery":   {"density": 1.05e-9, "E": 20.0, "PR": 0.45, "radius_mm": 2.0},
    "vessel_vein":     {"density": 1.05e-9, "E": 4.0, "PR": 0.45, "radius_mm": 1.25},
    "axon_fa00": {"density": 1.04e-9, "E": 0.120, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa10": {"density": 1.04e-9, "E": 0.148, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa20": {"density": 1.04e-9, "E": 0.176, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa30": {"density": 1.04e-9, "E": 0.204, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa40": {"density": 1.04e-9, "E": 0.232, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa50": {"density": 1.04e-9, "E": 0.260, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa60": {"density": 1.04e-9, "E": 0.288, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa70": {"density": 1.04e-9, "E": 0.316, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa80": {"density": 1.04e-9, "E": 0.344, "PR": 0.45, "radius_mm": 0.5},
    "axon_fa90": {"density": 1.04e-9, "E": 0.372, "PR": 0.45, "radius_mm": 0.5},
}

KEYWORD_GROUPS: dict[str, dict] = {
    "control_termination": {"title": "Termination time and timestep control",
                            "mandatory": True},
    "nodes_elements": {"title": "*NODE and every *ELEMENT_* block", "mandatory": True},
    "parts_sections": {"title": "*PART, *SECTION_SOLID/SHELL/BEAM, *HOURGLASS",
                       "mandatory": True},
    "control_hourglass": {"title": "*CONTROL_HOURGLASS (global hourglass default)",
                          "default": True},
    "control_energy": {"title": "*CONTROL_ENERGY (hourglass/sliding/rigid energy tracking)",
                       "default": True},
    "control_contact": {"title": "*CONTROL_CONTACT (contact defaults)", "default": False},
    "control_accuracy": {"title": "*CONTROL_ACCURACY (objective stress update, invariant nodes)",
                         "default": False},
    "control_shell": {"title": "*CONTROL_SHELL (shell warpage and thickness update)",
                      "default": False},
    "control_solid": {"title": "*CONTROL_SOLID (degenerate solid handling)", "default": False},
    "database_binary": {"title": "*DATABASE_BINARY_D3PLOT", "default": True},
    "database_ascii": {"title": "*DATABASE_GLSTAT/MATSUM/ELOUT/NODOUT", "default": True},
    "database_extent": {"title": "*DATABASE_EXTENT_BINARY (extra d3plot output)",
                        "default": False},
    "contacts": {"title": "*CONTACT_* between parts that are not welded", "default": True},
    "sets": {"title": "*SET_SOLID / *SET_SHELL, one per region", "default": True},
    "boundary_spc": {"title": "*BOUNDARY_SPC_SET fixing the skull", "default": True},
    "loading_curves": {"title": "*DEFINE_CURVE angular velocity/acceleration (curve.k)",
                       "default": True},
    "prescribed_motion": {"title": "*BOUNDARY_PRESCRIBED_MOTION_SET rotating the skull",
                          "default": True},
    "initial_velocity": {"title": "*INITIAL_VELOCITY_GENERATION (whole-head initial spin)",
                         "default": False},
    "damping": {"title": "*DAMPING_GLOBAL", "default": False},
    "constrained_beams": {"title": "*CONSTRAINED_BEAM_IN_SOLID coupling the vessel beams "
                                   "to the CSF they run in", "default": True},
    "constrained_membranes": {"title": "*CONSTRAINED_LAGRANGE_IN_SOLID immersing the falx, "
                                       "tentorium and dura in the CSF instead of deleting "
                                       "the fluid around them", "default": True},
}

EMBEDDED_MEMBRANES = ("falx", "tentorium", "tentorium_osseous", "dura")

MANDATORY_KEYWORDS = frozenset(k for k, v in KEYWORD_GROUPS.items() if v.get("mandatory"))
DEFAULT_KEYWORDS = frozenset(
    k for k, v in KEYWORD_GROUPS.items() if v.get("mandatory") or v.get("default"))

def resolve_keyword_selection(selected: Optional[set[str]]) -> frozenset[str]:
    """Normalize an operator's keyword selection."""
    if selected is None:
        return DEFAULT_KEYWORDS
    unknown = sorted(set(selected) - set(KEYWORD_GROUPS))
    if unknown:
        raise ValueError(f"unknown keyword group(s) {unknown}; "
                         f"choose from {sorted(KEYWORD_GROUPS)}")
    return frozenset(set(selected) | MANDATORY_KEYWORDS)

def _fmt(*fields, width: int = 10) -> str:
    """Format a comma-separated LS-DYNA free-format field row."""
    return ",".join(
        f"{f:.6g}" if isinstance(f, float) else str(f) for f in fields
    )

def _int_rows(fmt: str, start_id: int, pid: Optional[int], cells: np.ndarray) -> list[str]:
    """`fmt % (eid, [pid,] n1, n2, ...)` per row, via plain Python ints (numpy scalar
    formatting is the slow part of writing a deck)."""
    cells = np.asarray(cells, dtype=np.int64)
    eids = np.arange(start_id, start_id + len(cells), dtype=np.int64)[:, None]
    cols = [eids] if pid is None else [eids, np.full((len(cells), 1), int(pid), dtype=np.int64)]
    rows = np.hstack(cols + [cells.reshape(len(cells), -1)]).tolist()
    return [fmt % tuple(r) for r in rows]


def _format_chunk(job) -> str:
    """Reference Python formatting (ANATOMESH_PY_FORMAT=1); export/fastfmt.py must match it."""
    kind, fmt, start, pid, arr = job
    if kind == "node":
        return "\n".join("%d,%.6g,%.6g,%.6g,0,0" % (nid, x, y, z)
                         for nid, (x, y, z) in enumerate(arr.tolist(), start=start))
    return "\n".join(_int_rows(fmt, start, pid, arr))


def _format_rows(kind: str, fmt: str, start: int, pid, arr: np.ndarray) -> str:
    """A block's lines joined with newlines, by the byte-exact numba formatter. The
    process pool it replaces gave no gain: pickling cost as much as formatting."""
    import os

    if os.environ.get("ANATOMESH_PY_FORMAT"):
        return _format_chunk((kind, fmt, start, pid, arr))
    from engine.export import fastfmt

    if kind == "node":
        return fastfmt.node_rows(np.asarray(arr, dtype=np.float64), start).decode("ascii")
    return fastfmt.int_rows(np.asarray(arr, dtype=np.int64), start,
                            None if pid is None else int(pid)).decode("ascii")


def format_node_block(points: np.ndarray) -> str:
    """Format the *NODE keyword block."""
    pts = np.asarray(points, dtype=float)
    if not len(pts):
        return "*NODE"
    return "*NODE\n" + _format_rows("node", "", 1, None, pts)

def format_element_solid_block(cells: np.ndarray, pid: int, start_id: int, ortho: bool = False) -> str:
    """Format an *ELEMENT_SOLID (or *ELEMENT_SOLID_ORTHO) block for tets."""
    header = "*ELEMENT_SOLID_ORTHO" if ortho else "*ELEMENT_SOLID"
    lines = [header]
    for i, tet in enumerate(cells):
        eid = start_id + i
        n = tet + 1  
        
        lines.append(_fmt(eid, pid, n[0], n[1], n[2], n[3], n[3], n[3], n[3], n[3]))
    return "\n".join(lines)

def format_element_solid_hex_block(cells: np.ndarray, pid: int, start_id: int, ortho: bool = False) -> str:
    """Format an *ELEMENT_SOLID (or *ELEMENT_SOLID_ORTHO) block for true 8-node hexahedra."""
    header = "*ELEMENT_SOLID_ORTHO" if ortho else "*ELEMENT_SOLID"
    cells = np.asarray(cells, dtype=np.int64)
    if not len(cells):
        return header
    return header + "\n" + _format_rows("int", ",".join(["%d"] * 10), start_id, pid, cells + 1)

def format_element_solid_ortho_block(cells: np.ndarray, pid: int, start_id: int,
                                     e1_vectors: np.ndarray) -> str:
    """Format a correct *ELEMENT_SOLID_ORTHO block"""
    cells = np.asarray(cells)
    if cells.shape[1] == 4:
        cells = np.concatenate([cells, np.repeat(cells[:, 3:4], 4, axis=1)], axis=1)
    if len(e1_vectors) != len(cells):
        raise ValueError(f"{len(e1_vectors)} fibre directions for {len(cells)} elements")

    a = np.asarray(e1_vectors, dtype=float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    ref = np.zeros_like(a)
    ref[np.arange(len(a)), np.argmin(np.abs(a), axis=1)] = 1.0
    d = np.cross(a, ref)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    import os

    if not os.environ.get("ANATOMESH_PY_FORMAT"):
        from engine.export import fastfmt

        if not len(cells):
            return "*ELEMENT_SOLID_ORTHO"
        return "*ELEMENT_SOLID_ORTHO\n" + fastfmt.ortho_rows(
            np.asarray(cells, dtype=np.int64) + 1, start_id, int(pid), a, d).decode("ascii")
    one_based = (np.asarray(cells, dtype=np.int64) + 1).tolist()
    av, dv = a.tolist(), d.tolist()
    node_fmt = ",".join(["%d"] * len(one_based[0])) if one_based else ""
    lines = ["*ELEMENT_SOLID_ORTHO"]
    for i, n in enumerate(one_based):
        lines.append("%d,%d" % (start_id + i, pid))
        lines.append(node_fmt % tuple(n))
        lines.append("%.6f,%.6f,%.6f" % tuple(av[i]))
        lines.append("%.6f,%.6f,%.6f" % tuple(dv[i]))
    return "\n".join(lines)

def format_element_shell_block(cells: np.ndarray, pid: int, start_id: int) -> str:
    """Format an *ELEMENT_SHELL block for triangles."""
    lines = ["*ELEMENT_SHELL"]
    for i, tri in enumerate(cells):
        eid = start_id + i
        n = tri + 1
        lines.append(_fmt(eid, pid, n[0], n[1], n[2], n[2]))  
    return "\n".join(lines)

def format_element_shell_quad_block(cells: np.ndarray, pid: int, start_id: int) -> str:
    """Format an *ELEMENT_SHELL block for real 4-node quads."""
    cells = np.asarray(cells, dtype=np.int64)
    if not len(cells):
        return "*ELEMENT_SHELL"
    return "*ELEMENT_SHELL\n" + _format_rows("int", "%d,%d,%d,%d,%d,%d", start_id, pid,
                                             cells[:, :4] + 1)

def format_element_sph_block(node_ids: np.ndarray, pid: int, start_id: int) -> str:
    """Format an *ELEMENT_SPH block."""
    lines = ["*ELEMENT_SPH"]
    for i, nid in enumerate(node_ids):
        eid = start_id + i
        lines.append(_fmt(eid, pid, int(nid) + 1))
    return "\n".join(lines)

def format_element_beam_block(cells: np.ndarray, pid: int, start_id: int) -> str:
    """Format an *ELEMENT_BEAM block for vessel centerline segments."""
    lines = ["*ELEMENT_BEAM"]
    lines.extend(_int_rows("%d,%d,%d,%d", start_id, pid, np.asarray(cells)[:, :2] + 1))
    return "\n".join(lines)

def format_part_and_section_blocks(config, use_dti: bool,
                                    representation: Optional[dict] = None,
                                    present_parts: Optional[set] = None) -> str:
    """Format *PART / *SECTION_* blocks for the regions this deck actually has."""
    flipped = {name: kind.upper() for name, kind in (representation or {}).items()
               if name in PART_TABLE and kind.upper() != PART_TABLE[name][1]}
    if flipped:
        table = {name: (pid, flipped.get(name, etype), mat)
                 for name, (pid, etype, mat) in PART_TABLE.items()}
    else:
        table = dict(PART_TABLE)
    if present_parts is not None:
        absent = [name for name in table if name not in present_parts]
        table = {name: spec for name, spec in table.items() if name in present_parts}
        if absent:
            logger.info("no elements for %s; their *PART/*SECTION cards are omitted",
                        ", ".join(sorted(absent)))
    lines = []
    for name, (pid, etype, mat_key) in table.items():
        part_hgid = 1 if etype == "SOLID" else 0
        lines.append("*PART")
        lines.append(PART_TITLES[name])
        lines.append(_fmt(pid, pid, pid, 0, part_hgid, 0, 0, 0))

    for name, (pid, etype, mat_key) in table.items():
        if etype == "SOLID":
            lines.append("*SECTION_SOLID")
            elform = 1  # every solid is a hex off the voxel lattice
            lines.append(_fmt(pid, elform))
        elif etype == "SHELL":
            thickness = getattr(config, SHELL_THICKNESS_ATTR[name])
            lines.append("*SECTION_SHELL")
            lines.append(_fmt(pid, 16, 5.0, 0, 0, 0))  
            lines.append(_fmt(thickness, thickness, thickness, thickness))
        elif etype == "SPH":
            lines.append("*SECTION_SPH")
            lines.append(_fmt(pid, getattr(config, "sph_particle_spacing", 1.0), 0, 2))
        elif etype == "BEAM":
            card = HUMAN_ADULT_DEFAULTS[mat_key]  # beam radius only; no material values are written
            diameter = 2.0 * float(card.get("radius_mm", 1.0))
            lines.append("*SECTION_BEAM")
            lines.append(_fmt(pid, 1, 0, 2.0, 1, 0, 0))
            lines.append(_fmt(diameter, diameter, 0.0, 0.0))
    lines.append("*HOURGLASS")
    lines.append(_fmt(1, 6, 0.1))

    return "\n".join(lines)

def format_constrained_beam_in_solid(beam_regions, host_part,
                                     present_parts=None) -> str:
    """Couple every vessel beam part to the solid it runs inside."""
    present = [n for n in beam_regions
               if n in PART_TABLE and PART_TABLE[n][1] == "BEAM"]
    have = set(present_parts) if present_parts is not None else set(PART_TABLE)
    if not present:
        return ""
    mapping = host_part if isinstance(host_part, dict) else {n: host_part for n in present}

    def _usable(host):
        """Drop hosts that are missing, empty or not solid."""
        wanted = [host] if isinstance(host, str) else list(host or ())
        return [h for h in wanted
                if h in PART_TABLE and h in have and PART_TABLE[h][1] == "SOLID"]

    lines: list[str] = []
    set_lines: list[str] = []
    cid = 0
    sid = 9001
    coupled: dict = {}
    for name in sorted(present):
        hosts = _usable(mapping.get(name))
        if not hosts:
            continue
        cid += 1
        pid_solid, ssid = 0, 0
        if len(hosts) == 1:
            pid_solid = PART_TABLE[hosts[0]][0]
        else:
            ssid = sid
            sid += 1
            set_lines.append("*SET_SOLID_GENERAL")
            set_lines.append(_fmt(ssid, 0, 0, 0, 0, 0, 0, 0))
            for h in hosts:
                set_lines.append(f"      PART{PART_TABLE[h][0]:>10d}")
        lines.append("*CONSTRAINED_BEAM_IN_SOLID")
        lines.append(_fmt(cid, PART_TABLE[name][0], pid_solid, 0, ssid, 0, 0, 0))
        lines.append(_fmt(0.0, 1.0e20, 0, 0, 0.0))
        coupled[name] = hosts[0] if len(hosts) == 1 else hosts
    if not lines:
        return ""
    logger.info("coupled %d beam part(s) with *CONSTRAINED_BEAM_IN_SOLID: %s",
                len(coupled), coupled)
    return "\n".join(set_lines + lines)

def format_constrained_membranes_in_solid(membrane_parts, host_part_name="csf",
                                          present_parts=None) -> str:
    """Embed the falx, tentorium and dura in the CSF."""
    have = set(present_parts) if present_parts is not None else set(PART_TABLE)
    if (host_part_name not in PART_TABLE or host_part_name not in have
            or PART_TABLE[host_part_name][1] != "SOLID"):
        return ""
    praesentes = [n for n in sorted(membrane_parts)
                  if n in PART_TABLE and n in have and PART_TABLE[n][1] == "SHELL"]
    if not praesentes:
        return ""

    host_pid = PART_TABLE[host_part_name][0]
    lines: list[str] = []
    for entry_name in praesentes:
        lines.append("*CONSTRAINED_LAGRANGE_IN_SOLID")
        lines.append(_fmt(PART_TABLE[entry_name][0], host_pid, 1, 1, 0, 2, 1, 0))
        lines.append(_fmt(0.0, 1.0e20, 0.1, 0.0, 0.5, 0, 0, 0.0))
    logger.info("immersed %d membrane shell part(s) in %s with "
                "*CONSTRAINED_LAGRANGE_IN_SOLID: %s",
                len(praesentes), host_part_name, praesentes)
    return "\n".join(lines)

_MENINGES_CONTACT_RULES: list[tuple[str, str, str]] = [
    ("skull", "dura", "TIED"),
    ("csf", "pia", "SLIDING"),
    ("csf", "dura", "SLIDING"),
    ("dura", "pia", "TIED"),  
    ("skull", "pia", "TIED"),  
    ("falx", "dura", "TIED"),
    ("tentorium", "dura", "TIED"),
    ("tentorium_osseous", "dura", "TIED"),
    ("falx", "pia", "SLIDING"),
    ("tentorium", "pia", "SLIDING"),
    ("tentorium_osseous", "pia", "SLIDING"),
    ("scalp", "skull", "TIED"),
]

def format_contact_blocks(config, present_parts: set[str],
                           welded_parts: Optional[set[str]] = None,
                           offset_tied_pairs: Optional[set] = None) -> str:
    """Format *CONTACT_* cards, auto-derived from which parts this run actually has."""
    welded_parts = welded_parts or set()
    offset_tied_pairs = offset_tied_pairs or set()
    _PART_ID_TYPE = 3
    def contact_cards(ssid: int, msid: int, note: str, keyword: str) -> None:
        lines.append(keyword)
        lines.append(f"$# {note}")
        lines.append(_fmt(ssid, msid, _PART_ID_TYPE, _PART_ID_TYPE, 0, 0, 0, 0))
        lines.append(_fmt(0.0, 0.0, 0.0, 0.0, 20.0, 0, 0.0, 0.0))
        lines.append(_fmt(config.contact_penalty_scale, config.contact_penalty_scale,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    lines = []
    if config.sulcal_fold_contact and "gray_matter" in present_parts:
        contact_cards(1, 1, "gyral fold self-contact (gray matter)",
                      "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE")

    skip_skull_pia_fallback = "dura" in present_parts  
    skip_dura_pia_fallback = "csf" in present_parts    
    for part_a, part_b, kind in _MENINGES_CONTACT_RULES:
        if part_a not in present_parts or part_b not in present_parts:
            continue
        if part_a in welded_parts and part_b in welded_parts:
            logger.info("contact %s-%s (%s) suppressed: both parts are welded into the same "
                        "conforming mesh and already share nodes", part_a, part_b, kind.lower())
            continue
        if (part_a, part_b) == ("skull", "pia") and skip_skull_pia_fallback:
            continue
        if (part_a, part_b) == ("dura", "pia") and skip_dura_pia_fallback:
            continue
        if kind == "TIED":
            keyword = ("*CONTACT_TIED_SURFACE_TO_SURFACE_OFFSET"
                       if frozenset((part_a, part_b)) in offset_tied_pairs
                       else "*CONTACT_TIED_SURFACE_TO_SURFACE")
        else:
            keyword = "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE"
        contact_cards(PART_TABLE[part_a][0], PART_TABLE[part_b][0],
                      f"{part_a} - {part_b} ({kind.lower()})", keyword)

    return "\n".join(lines)

def format_set_blocks(cell_counts: dict[str, int]) -> str:
    """Format *SET_SOLID / *SET_SHELL cards, one set per anatomical region."""
    lines = []
    for name, (pid, etype, _mat_key) in PART_TABLE.items():
        if cell_counts.get(name, 0) == 0:
            continue
        if etype == "SOLID":
            lines.append("*SET_SOLID_GENERAL")
            lines.append(_fmt(pid, 0, 0, 0, 0, 0, 0, 0))
            lines.append(f"      PART{pid:>10d}")
        elif etype == "SHELL":
            lines.append("*SET_SHELL_GENERAL")
            lines.append(_fmt(pid, 0, 0, 0, 0, 0, 0, 0))
            lines.append(f"      PART{pid:>10d}")
    return "\n".join(lines)

def format_boundary_block(skull_fixed_node_ids: np.ndarray) -> str:
    """Format the *BOUNDARY_SPC_SET card fixing skull boundary nodes."""
    set_id = 900
    lines = ["*SET_NODE_LIST", _fmt(set_id, 0, 0, 0, 0, 0, 0, 0)]
    ids = (skull_fixed_node_ids + 1).tolist()
    for i in range(0, len(ids), 8):
        lines.append(_fmt(*ids[i:i + 8]))
    lines.append("*BOUNDARY_SPC_SET")
    lines.append(_fmt(set_id, 0, 1, 1, 1, 1, 1, 1))
    return "\n".join(lines)

def prepare_mesh_blocks_for_export(mesh_blocks, *, topology_strict=True):
    """Refuse a deck with an inverted hex"""
    from engine.mesh.voxel_hex import scaled_jacobian

    points = np.asarray(mesh_blocks["points"], dtype=float)
    for region, cells in mesh_blocks.get("solid", {}).items():
        cells = np.asarray(cells)
        if cells.ndim != 2 or cells.shape[1] != 8 or not len(cells):
            continue
        jac = scaled_jacobian(points[cells])
        invalid = int((~(jac > 0)).sum())  # NaN is a collapsed hex and counts too
        if invalid:
            worst = float(np.nanmin(jac)) if np.isfinite(jac).any() else float("nan")
            raise ValueError(f"{region}: {invalid} inverted or degenerate hexes (scaled "
                             f"Jacobian <= 0, min {worst:.4f}); refusing to write the deck")
    return mesh_blocks


def write_keyword_file(
    out_path: Path,
    run_id: str,
    species: str,
    use_dti: bool,
    mesh_blocks: dict,
    config,
    termination_time: float = 0.05,
    rotation=None,
    keywords: Optional[set[str]] = None,
) -> Path:
    """Assemble and write the complete LS-DYNA .k keyword file."""
    mesh_blocks = prepare_mesh_blocks_for_export(
        mesh_blocks,
        topology_strict=getattr(config, "topology_gate_strict", True) if config else True)
    want = resolve_keyword_selection(keywords)
    if rotation is not None:
        termination_time = float(rotation.termination_time)
    sections: list[str] = []

    sections.append("$# LS-DYNA keyword input, generated by AnatoMesh")
    sections.append(f"$# run id: {run_id}  species: {species}  dti: {use_dti}")
    sections.append("$# No *MAT cards: each *PART uses material id = part id; add your own *MAT_* with those ids.")
    sections.append(f"$# keyword groups: {', '.join(sorted(want))}")
    sections.append("*KEYWORD")
    sections.append("*TITLE")
    sections.append(f"{species}_head_mesh_{run_id}")
    sections.append("*CONTROL_TERMINATION")
    sections.append(_fmt(termination_time, 0, 0, 0.0, 0))
    sections.append("*CONTROL_TIMESTEP")
    sections.append(_fmt(0.0, 0.9, 0, 0.0, 0))
    if "control_hourglass" in want:
        sections.append("*CONTROL_HOURGLASS")
        sections.append(_fmt(6, 0.1))
    if "control_energy" in want:
        sections.append("*CONTROL_ENERGY")
        sections.append(_fmt(2, 2, 1, 2))
    if "control_contact" in want:
        sections.append("*CONTROL_CONTACT")
        sections.append(_fmt(0.0, 0.0, 0, 0, 1, 0, 0, 0))
    if "control_accuracy" in want:
        sections.append("*CONTROL_ACCURACY")
        sections.append(_fmt(1, 4, 0, 0.0))
    if "control_shell" in want:
        sections.append("*CONTROL_SHELL")
        sections.append(_fmt(20.0, 0, -1, -1, 2, 1, 0, 0))
    if "control_solid" in want:
        sections.append("*CONTROL_SOLID")
        sections.append(_fmt(1, 0, 0, 0, 0, 0, 0, 0))
    if "damping" in want:
        sections.append("*DAMPING_GLOBAL")
        sections.append(_fmt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    if "database_binary" in want:
        sections.append("*DATABASE_BINARY_D3PLOT")
        sections.append(_fmt(termination_time / 50.0))
    if "database_extent" in want:
        sections.append("*DATABASE_EXTENT_BINARY")
        sections.append(_fmt(0, 0, 3, 1, 1, 1, 1, 1))
    if "database_ascii" in want:
        for card in ("ELOUT", "GLSTAT", "MATSUM", "NODOUT"):
            sections.append(f"*DATABASE_{card}")
            sections.append(_fmt(termination_time / 500.0))

    sections.append(format_node_block(mesh_blocks["points"]))

    eid = 1
    for region, cells in mesh_blocks.get("solid", {}).items():
        pid = PART_TABLE[region][0]
        ortho = use_dti and region in mesh_blocks.get("solid_ortho_axes", {})
        if ortho:
            sections.append(format_element_solid_ortho_block(
                cells, pid, eid, mesh_blocks["solid_ortho_axes"][region]))
        elif cells.shape[1] == 8:
            sections.append(format_element_solid_hex_block(cells, pid, eid))
        else:
            sections.append(format_element_solid_block(cells, pid, eid))
        eid += len(cells)

    for region, cells in mesh_blocks.get("shell_quad", {}).items():
        pid = PART_TABLE[region][0]
        sections.append(format_element_shell_quad_block(cells, pid, eid))
        eid += len(cells)

    for region, cells in mesh_blocks.get("shell", {}).items():
        pid = PART_TABLE[region][0]
        sections.append(format_element_shell_block(cells, pid, eid))
        eid += len(cells)

    for region, node_ids in mesh_blocks.get("sph", {}).items():
        pid = PART_TABLE[region][0]
        sections.append(format_element_sph_block(node_ids, pid, eid))
        eid += len(node_ids)

    for region, cells in mesh_blocks.get("beam", {}).items():
        pid = PART_TABLE[region][0]
        sections.append(format_element_beam_block(cells, pid, eid))
        eid += len(cells)

    cell_counts: dict[str, int] = {}
    for group in ("solid", "shell", "shell_quad", "sph", "beam"):
        for region, cells in mesh_blocks.get(group, {}).items():
            cell_counts[region] = cell_counts.get(region, 0) + len(cells)
    present_parts = {name for name, count in cell_counts.items() if count > 0}

    sections.append(format_part_and_section_blocks(
        config, use_dti, representation=mesh_blocks.get("representation"),
        present_parts=present_parts))
    if "contacts" in want:
        sections.append(format_contact_blocks(
            config, present_parts, set(mesh_blocks.get("welded_parts", ())),
            offset_tied_pairs=set(mesh_blocks.get("offset_tied_pairs", ()))))
    if "sets" in want:
        sections.append(format_set_blocks(cell_counts))
    if "constrained_beams" in want and mesh_blocks.get("beam"):
        coupling = format_constrained_beam_in_solid(
            mesh_blocks["beam"], mesh_blocks.get("beam_host_part", "csf"),
            present_parts=present_parts)
        if coupling:
            sections.append(coupling)
        else:
            logger.warning(
                "beams are in the deck but nothing couples them to a solid: none of the "
                "hosts %r is a solid part with elements in this run. They will carry mass "
                "and no load. Mesh the CSF (MeshingConfig.auto_boundaries) or couple them "
                "by hand.",
                mesh_blocks.get("beam_host_part", "csf"))
    if "constrained_membranes" in want:
        # a sheet welded into the hex mesh shares its nodes and must not be coupled again
        welded_sheets = set(mesh_blocks.get("welded_parts", ()))
        immersio = format_constrained_membranes_in_solid(
            [m for m in EMBEDDED_MEMBRANES if m not in welded_sheets],
            present_parts=present_parts)
        if immersio:
            sections.append(immersio)

    driven_nodes = np.asarray(mesh_blocks.get("skull_fixed_nodes", np.zeros(0, dtype=int)))
    prescribing = (rotation is not None and "prescribed_motion" in want and len(driven_nodes))
    if "boundary_spc" in want and len(driven_nodes):
        if prescribing:
            logger.warning(
                "*BOUNDARY_SPC_SET suppressed: the same %d skull nodes carry the prescribed "
                "rotation, and an SPC on them would override it. Set "
                "MeshingConfig.fix_skull_boundary=False to stop requesting both.",
                len(driven_nodes))
        else:
            sections.append(format_boundary_block(driven_nodes))

    if rotation is not None:
        from engine.pipeline import loading

        if "loading_curves" in want:
            curve_path = out_path.parent / "curve.k"
            loading.write_curve_file(curve_path, rotation)
            sections.append("*INCLUDE")
            sections.append(curve_path.name)
        if prescribing:
            sections.append(loading.format_prescribed_motion(rotation, driven_nodes))
        if "initial_velocity" in want:
            axis = rotation.axis * rotation.omega
            sections.append("*INITIAL_VELOCITY_GENERATION")
            sections.append(_fmt(0, 2, float(np.linalg.norm(axis)), 0.0, 0.0, 0.0, 0, 0))
            sections.append(_fmt(*[float(v) for v in rotation.axis], 0.0, 0.0, 0.0))

    sections.append("*END")
    n_bytes = 0
    with safe_open_write(out_path, "w") as f:
        for section in sections:
            f.write(section)
            f.write("\n")
            n_bytes += len(section) + 1
    logger.info("Wrote LS-DYNA keyword file: %s (%d bytes)", out_path, n_bytes)
    try:  # what was just written, for the scorecard / previews / audit (no parse back)
        from engine.export import mesh_audit

        solids, shells, beams = {}, {}, {}
        for region, cells in mesh_blocks.get("solid", {}).items():
            cells = np.asarray(cells)
            if cells.shape[1] == 4:   # tets are written as 8-node solids, last node repeated
                cells = np.concatenate([cells, np.repeat(cells[:, 3:4], 4, axis=1)], axis=1)
            solids[PART_TABLE[region][0]] = cells
        for region, cells in mesh_blocks.get("shell_quad", {}).items():
            shells[PART_TABLE[region][0]] = np.asarray(cells)[:, :4]
        for region, cells in mesh_blocks.get("shell", {}).items():
            cells = np.asarray(cells)
            shells[PART_TABLE[region][0]] = np.c_[cells[:, :3], cells[:, 2]]
        for region, cells in mesh_blocks.get("beam", {}).items():
            beams[PART_TABLE[region][0]] = np.asarray(cells)[:, :2]
        names = {PART_TABLE[n][0]: PART_TITLES[n].strip() for n in present_parts
                 if n in PART_TABLE}
        mesh_audit.prime_cache(out_path, mesh_blocks["points"], names, solids, shells, beams)
    except Exception:  # noqa: BLE001 - the cache is an optimisation only
        logger.debug("deck cache not primed", exc_info=True)
    return out_path
