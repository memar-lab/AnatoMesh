"""Per-PART element count and Scaled Jacobian, read straight off an LS-DYNA deck."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from engine.export.validate import _hex_scaled_jacobian, _quad_scaled_jacobian


def _section_of(line: str) -> str | None:
    key = line.strip().upper().lstrip("*")
    if key.startswith("NODE"):
        return "NODE"
    if key.startswith("PART"):
        return "PART"
    if key.startswith("ELEMENT_SOLID_ORTHO"):
        return "SOLID_ORTHO"
    if key.startswith("ELEMENT_SOLID"):
        return "SOLID"
    if key.startswith("ELEMENT_SHELL"):
        return "SHELL"
    if key.startswith("ELEMENT_BEAM"):
        return "BEAM"
    return None


def _flush_part(rows: list[str], names: dict[int, str]) -> None:
    """A *PART block is a title line then a card whose first field is the pid."""
    try:
        pid = int((rows[1].split(",") if "," in rows[1] else rows[1].split())[0])
    except (ValueError, IndexError):
        return
    names[pid] = rows[0].strip() or f"part {pid}"


_DECK_CACHE: dict = {}


def read_deck(path: Path) -> tuple[np.ndarray, dict[int, str], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Parse nodes, part names and per-PID solid/shell/beam connectivity from a .k file."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key not in _DECK_CACHE:
        while len(_DECK_CACHE) >= 2:             # a run's two decks (dti / nodti) at most
            _DECK_CACHE.pop(next(iter(_DECK_CACHE)))
        _DECK_CACHE[key] = _parse_deck(path)
    points, names, solids, shells, beams = _DECK_CACHE[key]
    copy = lambda blocks: {pid: a.copy() for pid, a in blocks.items()}
    return points.copy(), dict(names), copy(solids), copy(shells), copy(beams)


def round_like_deck(points: np.ndarray) -> np.ndarray:
    """Coordinates as the deck stores them (`%.6g`, six significant digits)."""
    x = np.asarray(points, dtype=float)
    ax = np.abs(x)
    exponent = np.floor(np.log10(np.where(ax > 0, ax, 1.0)))
    scale = 10.0 ** (5 - exponent)
    return np.where(ax > 0, np.round(x * scale) / scale, 0.0)


def prime_cache(path: Path, points: np.ndarray, names: dict, solids: dict, shells: dict,
                beams: dict) -> None:
    """Store what the writer just wrote, so read_deck need not parse the file back (the
    scorecard, previews and audit of a run all read its decks). `points` are rounded to the
    deck's six significant digits, so every check grades exactly what is in the file."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    while len(_DECK_CACHE) >= 2:
        _DECK_CACHE.pop(next(iter(_DECK_CACHE)))
    as_int = lambda blocks: {int(p): np.asarray(a, dtype=np.int64) for p, a in blocks.items()}
    _DECK_CACHE[key] = (round_like_deck(points), dict(names), as_int(solids), as_int(shells),
                        as_int(beams))


def _parse_deck(path: Path):
    node_id: dict[int, int] = {}
    xyz: list[tuple[float, float, float]] = []
    names: dict[int, str] = {}
    solids: dict[int, list] = {}
    shells: dict[int, list] = {}
    beams: dict[int, list] = {}

    section = None
    part_pending: list[str] = []
    ortho_card, ortho_pid = 0, -1
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("$"):
                continue
            if line.startswith("*"):
                if section == "PART" and len(part_pending) >= 2:
                    _flush_part(part_pending, names)
                section = _section_of(line)
                part_pending = []
                ortho_card = 0
                continue
            row = line.rstrip("\n")
            if not row.strip():
                continue
            if section == "NODE":
                f = row.split(",") if "," in row else row.split()
                node_id[int(f[0])] = len(xyz)
                xyz.append((float(f[1]), float(f[2]), float(f[3])))
            elif section == "PART":
                part_pending.append(row)
                if len(part_pending) >= 2:
                    _flush_part(part_pending, names)
                    part_pending = []
            elif section == "SOLID_ORTHO":
                f = [x for x in (row.split(",") if "," in row else row.split()) if x]
                if ortho_card == 0 and len(f) >= 2:
                    ortho_pid = int(f[1])
                    ortho_card = 1
                elif ortho_card == 1 and len(f) >= 8:
                    solids.setdefault(ortho_pid, []).append([int(x) for x in f[:8]])
                    ortho_card = 2
                elif ortho_card in (2, 3):
                    ortho_card = 0 if ortho_card == 3 else 3
            elif section in ("SOLID", "SHELL", "BEAM"):
                f = [x for x in (row.split(",") if "," in row else row.split()) if x]
                if len(f) < 3:
                    continue
                if section == "BEAM":
                    if len(f) < 4:
                        continue
                    beams.setdefault(int(f[1]), []).append([int(f[2]), int(f[3])])
                    continue
                target = solids if section == "SOLID" else shells
                target.setdefault(int(f[1]), []).append([int(x) for x in f[2:]])
    if section == "PART" and len(part_pending) >= 2:
        _flush_part(part_pending, names)

    points = np.asarray(xyz, dtype=float)
    lut = np.zeros(max(node_id) + 1, dtype=np.int64)
    for k, v in node_id.items():
        lut[k] = v
    out_solids = {pid: lut[np.asarray(rows, dtype=np.int64)] for pid, rows in solids.items()}
    out_shells = {pid: lut[np.asarray(rows, dtype=np.int64)] for pid, rows in shells.items()}
    out_beams = {pid: lut[np.asarray(rows, dtype=np.int64)] for pid, rows in beams.items()}
    return points, names, out_solids, out_shells, out_beams


def read_ortho_axes(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-element fibre axes from every `*ELEMENT_SOLID_ORTHO` block."""
    node_id: dict[int, int] = {}
    conn: dict[int, list] = {}
    axes: dict[int, list] = {}
    section, card, pid = None, 0, -1
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("$"):
                continue
            if line.startswith("*"):
                section, card = _section_of(line), 0
                continue
            row = line.strip()
            if not row:
                continue
            f = [x for x in (row.split(",") if "," in row else row.split()) if x]
            if section == "NODE":
                node_id[int(f[0])] = len(node_id)
            elif section == "SOLID_ORTHO":
                if card == 0 and len(f) >= 2:
                    pid, card = int(f[1]), 1
                elif card == 1 and len(f) >= 8:
                    conn.setdefault(pid, []).append([int(x) for x in f[:8]])
                    card = 2
                elif card == 2 and len(f) >= 3:
                    axes.setdefault(pid, []).append([float(x) for x in f[:3]])
                    card = 3
                elif card == 3:
                    card = 0
    if not conn:
        return {}
    lut = np.zeros(max(node_id) + 1, dtype=np.int64)
    for k, v in node_id.items():
        lut[k] = v
    out = {}
    for pid, rows in conn.items():
        a = np.asarray(axes.get(pid, []), dtype=float)
        cells = lut[np.asarray(rows, dtype=np.int64)]
        n = min(len(cells), len(a))
        if n:
            out[pid] = (cells[:n], a[:n])
    return out


def audit(path: Path, threshold: float = 0.95) -> list[dict]:
    """Element count and Scaled Jacobian distribution for every part in a deck."""
    points, names, solids, shells, beams = read_deck(path)
    out = []
    for pid, conn in beams.items():
        length = np.linalg.norm(points[conn[:, 1]] - points[conn[:, 0]], axis=1)
        out.append({"pid": pid, "name": names.get(pid, f"part {pid}"), "kind": "beam",
                    "n": int(len(conn)), "min": float(length.min()),
                    "mean": float(length.mean()), "median": float(np.median(length)),
                    "pct_above": 100.0, "n_below": 0, "n_negative": 0})
    for kind, table, sj_fn in (("solid", solids, _hex_scaled_jacobian),
                               ("shell", shells, _quad_scaled_jacobian)):
        for pid, conn in table.items():
            if kind == "solid" and conn.shape[1] != 8:
                continue
            sj = sj_fn(points[conn[:, :8] if kind == "solid" else conn[:, :4]])
            out.append({
                "pid": pid, "name": names.get(pid, f"part {pid}"), "kind": kind,
                "n": int(len(sj)), "min": float(sj.min()), "mean": float(sj.mean()),
                "median": float(np.median(sj)),
                "pct_above": float(np.mean(sj >= threshold) * 100),
                "n_below": int(np.sum(sj < threshold)),
                "n_negative": int(np.sum(sj < 0)),
            })
    return sorted(out, key=lambda r: -r["n_below"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", type=Path)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    rows = audit(args.deck, args.threshold)
    total = sum(r["n"] for r in rows)
    below = sum(r["n_below"] for r in rows)
    print(f"{'part':<28}{'kind':<7}{'n':>10}{'%mesh':>8}"
          f"{'min SJ':>10}{'mean':>8}{f'%>={args.threshold}':>10}{'n below':>10}")
    for r in rows:
        print(f"{r['name']:<28}{r['kind']:<7}{r['n']:>10}{r['n'] / total * 100:>7.1f}%"
              f"{r['min']:>10.4f}{r['mean']:>8.4f}{r['pct_above']:>9.2f}%{r['n_below']:>10}")
    print(f"{'TOTAL':<28}{'':<7}{total:>10}{100.0:>7.1f}%{'':>10}{'':>8}"
          f"{(total - below) / total * 100:>9.2f}%{below:>10}")
    if any(r["kind"] == "beam" for r in rows):
        print("note: for beam parts the min/mean columns are element LENGTH in mm, not "
              "Scaled Jacobian -- a two-node element has no Jacobian. The shortest beam "
              "in the deck sets the explicit timestep for the whole model.")


if __name__ == "__main__":
    main()
