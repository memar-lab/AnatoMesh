"""The plain-text run report: settings, hole filling, smoothing per pass, mesh quality per part."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from engine.anatomy.anatomy_labels import LABEL_REGIONS

RULE = "-" * 78


def _name(code: int, combined: bool) -> str:
    if code == 0:
        return "tissue"
    if combined and code == 1:
        return "brain"
    return LABEL_REGIONS.get(code, str(code)).replace("_", " ")


def _smoothing_table(report: dict) -> list[str]:
    # codes are ints from the engine, strings once read back from metrics.json
    passes = [(label, {int(c): n for c, n in dict(counts).items()}) for label, counts in report.get("passes") or []]
    if not passes:
        return ["  (no smoothing)"]
    combined = bool(report.get("combined", True))
    codes = sorted({c for _, counts in passes for c in counts if c != 0})
    first = dict(passes[0][1])
    head = f"  {'pass':<12}" + "".join(f"{_name(c, combined)[:12]:>13}" for c in codes) + f"{'total':>13}"
    lines = [head]
    for label, counts in passes:
        if 0 in counts:                       # an outer pass only knows the tissue total
            lines.append(f"  {label:<12}" + " " * 13 * len(codes) + f"{counts[0]:>13,}")
            continue
        lines.append(f"  {label:<12}" + "".join(f"{counts.get(c, 0):>13,}" for c in codes)
                     + f"{sum(counts.values()):>13,}")
    last = dict(passes[-1][1]) if 0 not in passes[-1][1] else dict(passes[-2][1])
    lines.append(f"  {'net change':<12}" + "".join(f"{last.get(c, 0) - first.get(c, 0):>+13,}" for c in codes)
                 + f"{sum(last.values()) - sum(first.values()):>+13,}")
    lines.append(f"  voxels gained {report.get('voxels_gained', 0):,}, lost {report.get('voxels_lost', 0):,}, "
                 f"changed between tissues {report.get('internal_voxels_changed', 0):,}")
    return lines


def _mesh_steps(metrics: dict) -> list[str]:
    reports = metrics.get("hierarchical_smoothing") or []
    if not reports:
        return ["  (none)"]
    rows = reports[-1].get("iterations", {})
    lines = [f"  {'step':>4} {'SJ min':>9} {'SJ mean':>9} {'edge max':>9} {'diag min':>9} {'reshaped':>9}"]
    for step, s in sorted(rows.items(), key=lambda kv: int(kv[0])):
        lines.append(f"  {int(step):>4} {s['jacobian_min']:>9.4f} {s['jacobian_mean']:>9.4f} "
                     f"{s['edge_ratio_max']:>9.3f} {s['diagonal_ratio_min']:>9.3f} {s['adjusted_elements']:>9,}")
    return lines


def write(job_dir: Path, job: dict, metrics: dict, hole_report: str) -> Path:
    p = job["params"]
    result = json.loads((job_dir / "viewer" / "result.json").read_text(encoding="utf-8"))
    names = {r["id"]: r["name"] for r in result["legend"]}
    sj_by_part = np.load(str(job_dir / "viewer" / "part_sj.npz"))
    sj = np.concatenate([sj_by_part[k] for k in sj_by_part.files]) if sj_by_part.files else np.zeros(0)
    total = max(1, len(sj))
    bar = float(p["minJac"])
    element = float(p["elementSize"])
    lines = [
        "AnatoMesh run report", RULE,
        f"Run        {job.get('name', job['id'])}  ({job['id']})",
        f"Written    {time.strftime('%Y-%m-%d %H:%M')}",
        f"Input      {job.get('input_name', '')}",
        f"Atlas      {job['atlas']}",
        f"Parts      {len(job['parts'])} selected", "",
        "SETTINGS", RULE,
        f"  Element size                  {element:.3g} mm",
        f"  Smoothing iterations          {p['passes']}  ({2 * int(p['passes'])} steps)",
        f"  Strength (lambda)             {float(p['strength']):.2f}",
        f"  Targeted element adjustment   {'on, below ' + format(float(p.get('adjustThreshold', 0.5)), '.2f') if p['targeted'] else 'off'}",
        f"  Minimum scaled Jacobian       {bar:.2f}{', smoothing continues until reached' if p.get('untilMin') else ''}",
        f"  Outer surface smoothing       {('%.2f mm x %d' % (p['outerLevel'], p['outerIter'])) if p['outerOn'] else 'off'}",
        f"  Inner surface smoothing       {('%.2f mm x %d' % (p['innerLevel'], p['innerIter'])) if p['innerOn'] else 'off'}",
        f"  Combine regions               {'on' if p.get('combine', True) else 'off'}",
        f"  Seal gaps                     voxels added per layer: {', '.join(f'{k} {v:,}' for k, v in (metrics.get('sealed_voxels') or {}).items()) or 'none'}", "",
        "HOLE FILLING", RULE, *("  " + ln for ln in (hole_report or "off").splitlines()), "",
        "SEGMENTATION SMOOTHING  (voxels per label after each pass, element grid)", RULE,
        *_smoothing_table(metrics.get("label_smoothing") or {}), "",
        "MESH SMOOTHING  (per step)", RULE, *_mesh_steps(metrics), "",
        "FINAL MESH", RULE,
        f"  Elements {metrics.get('n_elements', 0):,}   nodes {metrics.get('n_nodes', 0):,}",
    ]
    if len(sj):
        pct = np.percentile(sj, [1, 10, 50, 90, 99])
        lines += [
            f"  Scaled Jacobian  min {sj.min():.4f}  mean {sj.mean():.4f}  max {sj.max():.4f}",
            "  percentiles      " + "  ".join(f"p{q} {v:.3f}" for q, v in zip((1, 10, 50, 90, 99), pct)),
            f"  below {bar:.2f}: {(sj < bar).sum():,} ({100 * (sj < bar).sum() / total:.2f}%)   "
            f"inverted (<= 0): {(sj <= 0).sum():,}",
        ]
    lines += ["", "PARTS", RULE,
              f"  {'part':<34} {'type':<6} {'elements':>10} {'share':>7} {'SJ min':>8} {'SJ mean':>8} {'>= bar':>7}"]
    for row in result["parts"]:
        vals = sj_by_part[row["id"]] if row["id"] in sj_by_part.files else np.zeros(0)
        ok = 100 * (vals >= bar).mean() if len(vals) else 0.0
        # same rule as the Output screen: worst element >= 0.45 and 95% at or above the bar
        passed = row["sj_min"] is not None and row["sj_min"] >= 0.45 and ok >= 95.0
        lines.append(f"  {names.get(row['id'], row['id'])[:34]:<34} {row.get('kind', 'solid'):<6} {row['elements']:>10,} "
                     f"{100 * row['elements'] / total:>6.2f}% {row['sj_min'] if row['sj_min'] is not None else float('nan'):>8.3f} "
                     f"{row['sj_mean'] if row['sj_mean'] is not None else float('nan'):>8.3f} {ok:>6.1f}%  {'PASS' if passed else 'FAIL'}")
    lines += ["", f"Deck  {Path(metrics.get('mesh_k_path', '')).name}", ""]
    path = job_dir / "report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
