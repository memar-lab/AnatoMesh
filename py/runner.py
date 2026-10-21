"""One job in its own process: prepare an input, or mesh a prepared input.

    python runner.py prepare <job.json>     segment the input, build the atlas parts, viewer files
    python runner.py mesh <job.json>        copy a prepared folder, clean labels, mesh, report

The job folder (outputs/<id>) gets status.json (state, progress, stage) and log.txt.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.paths import HOME, INPUTS_DIR  # noqa: E402

OUTPUTS = HOME / "outputs"
STAGES_PREPARE = ["Input check", "Segmentation", "Atlas", "Preview"]
STAGES_MESH = ["Input check", "Segmentation", "Smoothing", "Layers", "Meshing", "Axons", "Quality check", "Export"]

FS_CORTEX = ((1000, 1999, 3), (2000, 2999, 42), (11100, 11199, 3), (12100, 12199, 42))
TISSUE_CODES = set(range(1, 9))
# FreeSurfer aseg and cortical parcel codes; a label map uses almost only these, an MRI does not
FS_CODES = ({2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47,
             49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72, 77, 78, 79, 80, 81, 82, 85, 251, 252, 253, 254, 255}
            | set(range(1000, 1036)) | set(range(2000, 2036)) | set(range(11100, 11176)) | set(range(12100, 12176)))


class Status:
    """status.json of one job, rewritten on every change."""

    def __init__(self, job_dir: Path, stages: list[str]):
        self.path, self.stages = job_dir / "status.json", stages
        self.data = {"state": "running", "progress": 0.0, "stage": 0, "stage_name": stages[0],
                     "message": "", "started": time.time(), "finished": None, "error": None}
        self.write()

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        # Windows refuses the replace while the server has the file open for a read
        for attempt in range(50):
            try:
                tmp.replace(self.path)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.05)

    def stage(self, i: int, message: str = "") -> None:
        self.data.update(stage=i, stage_name=self.stages[i], message=message,
                         progress=max(self.data["progress"], i / len(self.stages)))
        self.write()

    def progress(self, fraction: float, message: str = "") -> None:
        self.data.update(progress=max(self.data["progress"], min(1.0, fraction)), message=message)
        self.write()

    def finish(self, state: str, error: str | None = None) -> None:
        self.data.update(state=state, finished=time.time(), error=error,
                         progress=1.0 if state == "done" else self.data["progress"])
        self.write()


def _log_to(job_dir: Path) -> None:
    handler = logging.FileHandler(job_dir / "log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


# ----------------------------------------------------------------------------- inputs

def detect(path: Path) -> dict:
    """What an input is: a FreeSurfer subject folder, a label map, an MRI, or a diffusion series."""
    if path.is_dir():
        if (path / "mri" / "aseg.mgz").exists():
            return {"kind": "freesurfer"}
        raise ValueError("this folder is not a FreeSurfer subject (no mri/aseg.mgz)")
    img = nib.load(str(path))
    if len(img.shape) == 4 and img.shape[3] > 1:
        return {"kind": "dwi", "shape": [int(n) for n in img.shape]}  # .mgz shapes are numpy ints
    data = np.asanyarray(img.dataobj)
    sample = data.ravel()[:: max(1, data.size // 2_000_000)]
    if np.issubdtype(data.dtype, np.integer) or np.allclose(sample, np.rint(sample)):
        values = np.unique(np.rint(sample).astype(np.int64))
        codes = set(values.tolist()) - {0}
        if codes and values.min() >= 0 and codes <= TISSUE_CODES:
            return {"kind": "labels", "scheme": "tissue", "n_labels": len(codes)}
        if codes and values.min() >= 0 and len(codes & FS_CODES) >= 0.95 * len(codes):
            return {"kind": "labels", "scheme": "freesurfer", "n_labels": len(codes)}
    return {"kind": "mri", "shape": [int(n) for n in img.shape[:3]]}


def _reorient(img: nib.Nifti1Image, axes: list[str] | None) -> nib.Nifti1Image:
    """RAS, or the axes the user stated (letter at the + end of the volume's x, y, z)."""
    from nibabel import orientations as o

    if axes:
        stated = o.axcodes2ornt(tuple(axes))
        target = o.axcodes2ornt(("R", "A", "S"))
        data = o.apply_orientation(np.asanyarray(img.dataobj), o.ornt_transform(stated, target))
        zooms = img.header.get_zooms()[:3]
        perm = [int(stated[i, 0]) for i in range(3)]
        spacing = np.array([zooms[perm.index(k)] for k in range(3)], dtype=float)
        affine = np.diag([*spacing, 1.0])
        return nib.Nifti1Image(data, affine)
    return nib.as_closest_canonical(img)


def stage_label_map(path: Path, run_dir: Path, axes: list[str] | None) -> dict:
    """A label NIfTI as the run's segmentation (tissue scheme), plus the FreeSurfer parts it carries."""
    from engine.anatomy import substructures
    from engine.pipeline import prepare

    img = _reorient(nib.load(str(path)), axes)
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    codes = set(np.unique(data).tolist()) - {0}
    out = {"run_dir": str(run_dir), "use_dti": False, "labels_only": True}
    if codes <= TISSUE_CODES:
        seg = data.astype(np.int16)
    else:
        aseg = data.copy()
        for lo, hi, cortex in FS_CORTEX:
            aseg[(data >= lo) & (data <= hi)] = cortex
        seg = prepare.aseg_to_tissue_labels(aseg)
        if not (seg > 0).any():
            raise ValueError("the label map uses neither FreeSurfer codes nor the 1-8 tissue codes")
        nib.save(nib.Nifti1Image(aseg.astype(np.int16), img.affine), str(run_dir / substructures.ASEG_FILE))
        if any(1000 <= c < 3000 for c in codes):
            nib.save(nib.Nifti1Image(data.astype(np.int16), img.affine), str(run_dir / "parcellation_aparc+aseg.nii.gz"))
        if any(11100 <= c < 12200 for c in codes):
            nib.save(nib.Nifti1Image(data.astype(np.int16), img.affine), str(run_dir / "parcellation_a2009s+aseg.nii.gz"))
    nib.save(nib.Nifti1Image(seg, img.affine), str(run_dir / "segmentation.nii.gz"))
    out["segmentation"] = str(run_dir / "segmentation.nii.gz")
    return out


def pseudo_t1(run_dir: Path) -> Path:
    """A T1-like image painted from the tissue labels, for registering atlases to a label-only input."""
    seg_img = nib.load(str(run_dir / "segmentation.nii.gz"))
    seg = np.asanyarray(seg_img.dataobj)
    look = {1: 80.0, 2: 110.0, 3: 100.0, 4: 85.0, 5: 90.0, 6: 30.0, 7: 15.0, 8: 60.0}
    t1 = np.zeros(seg.shape, np.float32)
    for code, value in look.items():
        t1[seg == code] = value
    out = run_dir / "t1_brain.nii.gz"
    nib.save(nib.Nifti1Image(t1, seg_img.affine), str(out))
    return out


def ensure_atlas(atlas: str, run_dir: Path) -> None:
    """The parcellation the chosen atlas needs, made by registration when the input lacks it."""
    from engine.anatomy import atlas_segment, parcellation, substructures

    if atlas == "whole":
        return
    if atlas == "aseg" and not (run_dir / substructures.ASEG_FILE).exists():
        atlas_segment.aseg_from_atlas(_brain_image(run_dir), run_dir)
    if atlas == "aparc" and not (run_dir / parcellation.SCHEMES["aparc"]).exists():
        raise RuntimeError("Aseg + Aparc needs Desikan labels in the input")
    if atlas == "destrieux" and not (run_dir / parcellation.SCHEMES["a2009s"]).exists():
        import atlas_data

        atlas_data.warp_destrieux(_brain_image(run_dir), run_dir)
    if atlas == "harvard" and not (run_dir / parcellation.SCHEMES["harvard_oxford"]).exists():
        atlas_segment.parcellate(_brain_image(run_dir), run_dir)


def _brain_image(run_dir: Path) -> Path:
    path = run_dir / "t1_brain.nii.gz"
    return path if path.exists() else pseudo_t1(run_dir)


def prepare(job: dict, job_dir: Path) -> None:
    from engine.pipeline import input_qc, prepare as prep

    status = Status(job_dir, STAGES_PREPARE)
    src = Path(job["input"]["path"])
    found = detect(src)
    info: dict = {"input": found, "atlas": job["atlas"]}
    if found["kind"] == "mri":
        qc = input_qc.check_mri(src, "human")
        input_qc.write_report(qc, job_dir)
        info["input_check"] = qc["status"]
    status.stage(1, "segmenting")
    dwi = Path(job["input"]["dwi"]) if job["input"].get("dwi") else None
    axes = job.get("axes")
    if found["kind"] == "freesurfer":
        out = prep.prepare_from_freesurfer(src, job["id"], dti=dwi)
    elif found["kind"] == "labels":
        out = stage_label_map(src, job_dir, axes)
    elif found["kind"] == "mri":
        from engine.engines import atlasseg

        if axes:
            staged = INPUTS_DIR / job["id"] / "t1_oriented.nii.gz"
            staged.parent.mkdir(parents=True, exist_ok=True)
            nib.save(_reorient(nib.load(str(src)), axes), str(staged))
            src = staged
        from engine.pipeline import timing

        # segmenter stages in order, with the share of the Segmentation stage each one starts at
        steps = {"prepare": (0.0, "bias correction"), "template load": (0.2, "loading atlases"),
                 "registration": (0.25, "registering to the template"), "warp": (0.4, "warping the atlases"),
                 "correction": (0.55, "fitting labels to this brain"), "write": (0.9, "writing labels"),
                 "harvard-oxford": (0.93, "Harvard-Oxford")}

        def on_stage(name: str) -> None:
            share, text = steps.get(name, (None, name))
            if share is not None:
                status.progress((1 + share) / len(STAGES_PREPARE), text)

        timing.on_stage = on_stage
        # T1 -> published atlases (fsaverage aseg/Desikan/Destrieux, Harvard-Oxford) in one registration
        try:
            recon = atlasseg.segment_t1(src, job_dir / "recon")
        finally:
            timing.on_stage = None
        out = prep.prepare_from_freesurfer(recon, job["id"], dti=dwi)
        ho = recon / "mri" / "harvard_oxford_cortical.mgz"
        if ho.exists():
            from engine.anatomy import parcellation

            nib.save(nib.as_closest_canonical(nib.load(str(ho))),
                     str(job_dir / parcellation.SCHEMES["harvard_oxford"]))
    else:
        raise ValueError("a diffusion series is a second input, not the main one")
    info.update({k: v for k, v in out.items() if k in ("use_dti", "freesurfer_subject_dir", "labels_only")})
    status.stage(2, "atlas")
    ensure_atlas(job["atlas"], job_dir)
    status.stage(3, "preview")
    import viewer_files

    viewer_files.write_preview(job_dir, job["atlas"])
    (job_dir / "prep.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    status.finish("done")


# ----------------------------------------------------------------------------- mesh

def engine_config(job: dict, prep_info: dict):
    import parts as parts_mod
    from engine.mesh import mesh_agent

    p = job["params"]
    element = float(p["elementSize"])
    config = mesh_agent.MeshingConfig(auto_boundaries=mesh_agent.OPTIONAL_BOUNDARIES, fix_skull_boundary=False)
    for key, value in parts_mod.engine_settings(job["atlas"], job["parts"]).items():
        setattr(config, key, value)
    config.element_size_mm = element
    config.hierarchical_smoothing_iterations = int(p["passes"])
    config.hierarchical_smoothing_lambda = float(p["strength"])
    config.hierarchical_smoothing_repair = bool(p["targeted"])
    config.repair_jacobian_threshold = float(p.get("adjustThreshold", 0.5))
    config.smoothing_target_jacobian = float(p["minJac"]) if p.get("untilMin") else None
    # smoothing level is a Gaussian sigma in mm; the engine smooths on the element grid
    config.label_smoothing_sigma_vox = float(p["outerLevel"]) / element if p["outerOn"] else 0.0
    config.label_smoothing_iterations = int(p["outerIter"])
    config.label_smoothing_sigma_inner_vox = float(p["innerLevel"]) / element if p["innerOn"] else 0.0
    config.label_smoothing_iterations_inner = int(p["innerIter"])
    config.combine_brain_for_smoothing = bool(p.get("combine", True))
    if prep_info.get("labels_only"):
        # no head image: skull and scalp are offset layers of prescribed thickness
        config.skull_source = "offset"
        config.require_mri_derived_layers = False
    return config


def fill_holes(run_dir: Path, p: dict) -> str:
    from engine import segprep

    if not p.get("fillHoles"):
        return "Fill holes: off"
    path = run_dir / "segmentation.nii.gz"
    img = nib.load(str(path))
    seg = np.asanyarray(img.dataobj).astype(np.int16)
    lines: list[str] = []
    mode = "label" if p.get("fillWith", "neighbour") != "neighbour" else "neighbour"
    label = int(p.get("fillWith")) if mode == "label" else 0
    filled, _info = segprep.fill_holes(seg, mode, label, int(p.get("fillMax", 0)), log=lines.append)
    nib.save(nib.Nifti1Image(filled.astype(np.int16), img.affine, img.header), str(path))
    return "\n".join(lines)


def mesh(job: dict, job_dir: Path) -> None:
    from engine.mesh import mesh_agent

    status = Status(job_dir, STAGES_MESH)
    prep_dir = OUTPUTS / job["prep"]
    prep_info = json.loads((prep_dir / "prep.json").read_text(encoding="utf-8"))
    for item in prep_dir.iterdir():
        if item.name in ("status.json", "log.txt", "prep.json", "job.json", "viewer"):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, job_dir / item.name)
    status.stage(1, "cleaning labels")
    hole_report = fill_holes(job_dir, job["params"])
    if "axons" in job["parts"] and not (job_dir / "tracts_t1.npz").exists():
        import atlas_data

        status.progress(0.03, "atlas tracts")
        atlas_data.atlas_tracts(_brain_image(job_dir), job_dir)
    (job_dir / "fill_holes.txt").write_text(hole_report, encoding="utf-8")
    config = engine_config(job, prep_info)

    marks = [(0.05, 2), (0.08, 3), (0.2, 4), (0.8, 5), (0.9, 6)]

    def progress(fraction: float, message: str) -> None:
        stage = 1 + sum(fraction >= f for f, _ in marks)
        status.data.update(stage=stage, stage_name=STAGES_MESH[min(stage, len(STAGES_MESH) - 1)])
        status.progress(0.05 + 0.85 * fraction, message)

    subject = prep_info.get("freesurfer_subject_dir")
    metrics = mesh_agent.run_meshing(
        config, job["id"], species="human", use_dti=bool(prep_info.get("use_dti")),
        freesurfer_subject_dir=Path(subject) if subject else None, progress=progress)
    status.stage(6, "quality")
    import report
    import viewer_files

    viewer_files.write_result(job_dir, job["atlas"])
    report.write(job_dir, job, metrics, hole_report)
    status.stage(7, "export")
    summary = {"elements": metrics.get("n_elements"), "nodes": metrics.get("n_nodes"),
               "sj_min": metrics.get("scaled_jacobian", {}).get("min"),
               "sj_mean": metrics.get("scaled_jacobian", {}).get("mean"),
               "negative": metrics.get("scaled_jacobian", {}).get("n_negative"),
               "deck": metrics.get("mesh_k_path"), "wall_time_sec": metrics.get("wall_time_sec"),
               "passed": (metrics.get("scorecard") or {}).get("passed")}
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    status.finish("done")


def main() -> int:
    # ANTs reads this once, at import
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(os.cpu_count() or 1))
    mode, job_path = sys.argv[1], Path(sys.argv[2])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_dir = OUTPUTS / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    _log_to(job_dir)
    try:
        {"prepare": prepare, "mesh": mesh}[mode](job, job_dir)
        return 0
    except Exception as exc:
        logging.getLogger("runner").exception("job %s failed", job["id"])
        status_path = job_dir / "status.json"
        data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        data.update(state="failed", finished=time.time(), error=str(exc) or type(exc).__name__,
                    trace=traceback.format_exc()[-4000:])
        status_path.write_text(json.dumps(data), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
