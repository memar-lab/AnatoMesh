"""The browser twin of backend/server.py.

The AnatoMesh interface talks to its engine only through a few /api routes. On the desktop, server.py (Flask) answers them and runs each job in
its own process. Here the same routes are answered by this module, inside Pyodide in a Web Worker, on the visitor's own computer:

  * same route names, same JSON, same job folders (inputs/<id>, outputs/<id>) in Pyodide's in-memory disk;
  * the engine (runner.prepare / runner.mesh, engine/...) is the desktop code, unchanged;
  * a job is not a process: the worker runs it after the reply to POST /api/prepare or /api/runs has been sent, and every status write is
    posted to the page, which answers the interface's polling from its own copy (the worker is busy computing and cannot answer).

Nothing here touches the network. Files leave only when the page saves an export into the user's own folder.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import time
import traceback
from collections import deque
from pathlib import Path
from uuid import uuid4

import webstubs

webstubs.install()                               # before the engine is imported: stubs, blocked modules, no threads

import numpy as np                               # noqa: E402

import runner                                    # noqa: E402  (desktop code)
import parts as parts_mod                        # noqa: E402  (desktop code)
from engine.paths import HOME, INPUTS_DIR        # noqa: E402

webstubs.patch_engine()                          # 32-bit integers in WebAssembly (see webstubs.py)

OUTPUTS = HOME / "outputs"
EXPORTS = HOME / "export"
VERSION = "0.1.0-web"
MAX_LOG = 200

try:                                             # inside Pyodide the page listens; in tests the events are collected
    import js

    def _post(obj: dict) -> None:
        js.postMessage(json.dumps(obj, default=str))
except ImportError:
    EVENTS: list[dict] = []

    def _post(obj: dict) -> None:
        EVENTS.append(obj)

QUEUE: list[tuple[str, str]] = []                # (kind, job id) waiting for the worker
RUNNING: set[str] = set()
LOGS: dict[str, deque] = {}
_last_emit: dict[str, float] = {}

# Size guard. A browser tab holds about 2 GB. Measured in Pyodide: 157,845 elements (3 mm) used 464 MB of WebAssembly memory and 155 s.
# Elements ~ tissue volume / h^3 + a layer term / h^2: the head layers (scalp, skull, dura, ...) of a label map are offset layers of at
# least one element each, so they scale with the surface and not the volume. LAYER_AREA was fitted to the two measured meshes of the
# sample (3 mm: 157,845 elements; 4 mm: 87,837; the two fits agree within 5 %); it scales with the tissue volume to the power 2/3.
LAYER_AREA = 1.1e6                    # mm^2 for the sample (1,191,442 mm^3 of tissue)
SAMPLE_VOLUME = 1_191_442.0           # mm^3
MAX_ELEMENTS = 400_000

MIMES = {".json": "application/json", ".txt": "text/plain", ".png": "image/png"}

NOT_IN_BROWSER = ("This input needs the desktop version of AnatoMesh: {why}. The browser edition meshes label maps "
                  "(a NIfTI file with tissue codes 1-8, or FreeSurfer aseg / aparc codes).")


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------------------- small helpers (as in server.py)

def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def _safe_rel(rel: str) -> Path:
    parts = rel.replace("\\", "/").lstrip("/").split("/")
    if any(p in ("", "..") for p in parts) or ":" in parts[0]:
        raise ValueError(f"invalid path {rel!r}")
    return Path(*parts)


def _inside(base: Path, rel: str) -> Path:
    target = (base / _safe_rel(rel)).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes its folder: {rel!r}")
    return target


def _json_read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def _job_dir(job_id: str) -> Path | None:
    try:
        path = _inside(OUTPUTS, job_id)
    except ValueError:
        return None
    return path if path.is_dir() else None


def _input_meta(input_id: str) -> dict | None:
    try:
        return _json_read(_inside(INPUTS_DIR, f"{input_id}/meta.json"), None)
    except ValueError:
        return None


def _stem(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


# ----------------------------------------------------------------------------- jobs

def _job_item(job_dir: Path) -> dict | None:
    job = _json_read(job_dir / "job.json", None)
    if not job:
        return None
    status = _json_read(job_dir / "status.json", {})
    status.pop("trace", None)
    if status.get("state") in ("running", "queued") and job["id"] not in RUNNING and (job["kind"], job["id"]) not in QUEUE:
        status["state"] = "failed"
        status["error"] = status.get("error") or "the job stopped unexpectedly"
    return {"id": job["id"], "kind": job["kind"], "name": job.get("name") or job["id"], "atlas": job.get("atlas"),
            "input_name": job.get("input_name"), "created": job.get("created"), "prep": job.get("prep"), **status,
            "summary": _json_read(job_dir / "summary.json", None)}


def job_payload(job_id: str) -> dict | None:
    """What GET /api/jobs/<id> returns (the page keeps a copy of it)."""
    job_dir = _job_dir(job_id)
    item = _job_item(job_dir) if job_dir else None
    if not item:
        return None
    lines = list(LOGS.get(job_id, ()))
    return {**item, "job": _json_read(job_dir / "job.json", {}), "log": lines,
            "trace": _json_read(job_dir / "status.json", {}).get("trace")}


def emit_job(job_id: str, force: bool = False) -> None:
    now = time.time()
    if not force and now - _last_emit.get(job_id, 0) < 0.15:
        return
    _last_emit[job_id] = now
    payload = job_payload(job_id)
    if payload:
        _post({"type": "job", "id": job_id, "payload": payload})


class _RingHandler(logging.Handler):
    def __init__(self, job_id: str):
        super().__init__(logging.INFO)
        self.job_id = job_id
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        LOGS[job_id] = deque(maxlen=MAX_LOG)

    def emit(self, record):
        LOGS[self.job_id].append(self.format(record))
        emit_job(self.job_id)


_original_write = runner.Status.write


def _write_and_emit(self):
    _original_write(self)
    emit_job(self.path.parent.name, force=self.data.get("state") != "running")


runner.Status.write = _write_and_emit            # every status change is also posted to the page


def _spawn(job: dict, mode: str) -> None:
    job_dir = OUTPUTS / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job_dir / "job.json", job)
    _write_json(job_dir / "status.json", {"state": "queued", "progress": 0, "stage": 0, "stage_name": "", "message": "",
                                          "started": time.time(), "finished": None, "error": None})
    QUEUE.append((mode, job["id"]))
    emit_job(job["id"], force=True)


def run_queued() -> None:
    """Runs the queued jobs, one after the other (the worker calls this after it has replied to the request)."""
    while QUEUE:
        mode, job_id = QUEUE.pop(0)
        job_dir = _job_dir(job_id)
        if not job_dir:
            continue
        job = _json_read(job_dir / "job.json", {})
        RUNNING.add(job_id)
        ring = _RingHandler(job_id)
        file_log = logging.FileHandler(job_dir / "log.txt", encoding="utf-8")
        file_log.setFormatter(ring.formatter)
        root = logging.getLogger()
        root.addHandler(ring)
        root.addHandler(file_log)
        root.setLevel(logging.INFO)
        try:
            {"prepare": runner.prepare, "mesh": runner.mesh}[mode](job, job_dir)
        except Exception as exc:                 # same record as runner.main() leaves behind
            logging.getLogger("runner").exception("job %s failed", job_id)
            status = _json_read(job_dir / "status.json", {})
            status.update(state="failed", finished=time.time(), error=str(exc) or type(exc).__name__,
                          trace=traceback.format_exc()[-4000:])
            (job_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        finally:
            root.removeHandler(ring)
            root.removeHandler(file_log)
            file_log.close()
            RUNNING.discard(job_id)
            emit_job(job_id, force=True)


# ----------------------------------------------------------------------------- inputs

def _ingest(upload: dict[str, bytes], role: str) -> dict:
    base = INPUTS_DIR / _new_id("i")
    try:
        base.mkdir(parents=True, exist_ok=True)
        rels = [_safe_rel(name) for name in upload]
        if len(rels) != 1 or len(rels[0].parts) != 1:
            raise ApiError(NOT_IN_BROWSER.format(why="folders (FreeSurfer subjects) and multi-file inputs are not supported here"))
        if role == "dwi":
            raise ApiError(NOT_IN_BROWSER.format(why="diffusion series (fibre tracking) are not supported here"))
        main = base / rels[0]
        main.write_bytes(next(iter(upload.values())))
        found = runner.detect(main)
        if found["kind"] != "labels":
            why = {"mri": "a raw MRI is segmented by registration (ANTs), which the browser does not have",
                   "dwi": "diffusion series are not supported here",
                   "freesurfer": "FreeSurfer folders are not supported here"}.get(found["kind"], "this kind of input is not supported here")
            raise ApiError(NOT_IN_BROWSER.format(why=why))
        import nibabel as nib

        data = np.asanyarray(nib.load(str(main)).dataobj)
        found["codes"] = sorted(int(c) for c in np.unique(np.rint(data).astype(np.int64)) if c)[:600]
    except ApiError:
        shutil.rmtree(base, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(base, ignore_errors=True)
        raise ApiError(str(exc) or type(exc).__name__)
    meta = {"id": base.name, "name": main.name, "path": str(main), "role": role, "kind": found["kind"], "detected": found,
            "created": time.time()}
    _write_json(base / "meta.json", meta)
    return meta


def _atlas_supported(meta: dict, atlas: str) -> None:
    """The atlases that can be made from a label map without registration."""
    found = meta.get("detected", {})
    codes = set(found.get("codes", []))
    if atlas == "whole":
        return
    if found.get("scheme") == "tissue":
        raise ApiError("This label map has only the tissue codes 1-8, so it has no brain structures: choose Whole Brain, "
                       "or use a label map with FreeSurfer codes.")
    if atlas == "aseg":
        return
    if atlas == "aparc" and not any(1000 <= c < 3000 for c in codes):
        raise ApiError("Aseg + Aparc needs Desikan cortical labels (1000-2999) in the input.")
    if atlas == "destrieux" and not any(11100 <= c < 12200 for c in codes):
        raise ApiError("Destrieux needs a2009s cortical labels (11100-12199) in the input. Without them it is made by "
                       "registration to a template, which the browser edition does not have.")
    if atlas == "harvard":
        raise ApiError("Harvard-Oxford is made by registration to a template, which the browser edition does not have. "
                       "Use Aseg, Aseg + Aparc or Whole Brain.")


def estimate_elements(prep_dir: Path, element_mm: float) -> int:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(prep_dir / "segmentation.nii.gz")))
    voxels = int(np.count_nonzero(np.asanyarray(img.dataobj)))
    voxel_mm = float(np.abs(np.diag(img.affine)[:3]).mean())
    volume = voxels * voxel_mm ** 3
    return int(volume / element_mm ** 3 + LAYER_AREA * (volume / SAMPLE_VOLUME) ** (2 / 3) / element_mm ** 2)


def check_size(prep_dir: Path, element_mm: float) -> None:
    if element_mm <= 0:
        raise ApiError("The element size must be larger than 0 mm.")
    estimate = estimate_elements(prep_dir, element_mm)
    if estimate > MAX_ELEMENTS:
        smallest = element_mm * (estimate / MAX_ELEMENTS) ** (1 / 3)
        smallest = np.ceil(smallest * 2) / 2                          # next 0.5 mm
        raise ApiError(f"An element size of {element_mm:g} mm would give about {estimate:,} elements. A browser tab can hold about "
                       f"{MAX_ELEMENTS:,} (it has about 2 GB of memory). Use an element size of {smallest:g} mm or larger, "
                       f"or run this on the desktop version of AnatoMesh.")


# ----------------------------------------------------------------------------- routes

def route(method: str, path: str, query: dict, body: dict | None, upload: dict[str, bytes] | None):
    """Returns (status, mime, payload). payload is a dict (JSON) or bytes."""
    seg = [s for s in path.split("/") if s]
    if seg[:1] != ["api"]:
        raise ApiError("not found", 404)
    seg = seg[1:]
    body = body or {}

    if method == "GET" and seg == ["health"]:
        return 200, "json", {"ok": True, "version": VERSION, "home": str(HOME), "python": platform.python_version(),
                             "tools": {"ants": False, "synthseg": False}, "web": True}
    if method == "POST" and seg == ["inputs"]:
        if not upload:
            raise ApiError("no files uploaded")
        return 200, "json", _ingest(upload, query.get("role") or "main")
    if method == "POST" and seg == ["inputs", "path"]:
        raise ApiError("Choosing a file by its path is a desktop feature. Use the file chooser: in the browser edition the "
                       "file is read inside the page and never leaves your computer.")
    if method == "GET" and len(seg) == 2 and seg[0] == "inputs":
        meta = _input_meta(seg[1])
        if not meta:
            raise ApiError("input not found", 404)
        return 200, "json", meta
    if method == "POST" and seg == ["prepare"]:
        atlas, dwi_id, axes = body.get("atlas"), body.get("dwi_id"), body.get("axes")
        if atlas not in parts_mod.ATLASES:
            raise ApiError(f"unknown atlas {atlas!r}")
        if axes is not None and not (isinstance(axes, list) and len(axes) == 3):
            raise ApiError("axes must be a list of 3 letters or null")
        if dwi_id:
            raise ApiError(NOT_IN_BROWSER.format(why="diffusion series (fibre tracking) are not supported here"))
        meta = _input_meta(body.get("input_id") or "")
        if not meta:
            raise ApiError("input not found", 404)
        _atlas_supported(meta, atlas)
        for job_file in OUTPUTS.glob("*/job.json"):
            old = _json_read(job_file, {})
            if (old.get("kind") == "prepare" and old.get("input_id") == meta["id"] and old.get("atlas") == atlas
                    and old.get("axes") == axes):
                item = _job_item(job_file.parent)
                if item and item.get("state") in ("done", "running", "queued"):
                    return 200, "json", {"id": old["id"], "reused": True}
        job = {"id": _new_id("p"), "kind": "prepare", "input_id": meta["id"], "dwi_id": None, "atlas": atlas, "axes": axes,
               "input": {"path": meta["path"], "dwi": None}, "input_name": meta["name"], "created": time.time()}
        _spawn(job, "prepare")
        return 200, "json", {"id": job["id"], "reused": False}
    if method == "POST" and seg == ["runs"]:
        prep_dir = _job_dir(body.get("prep") or "")
        if not prep_dir or _json_read(prep_dir / "status.json", {}).get("state") != "done":
            raise ApiError("the input is not prepared yet")
        atlas, selected, params = body.get("atlas"), body.get("parts"), body.get("params")
        if not isinstance(selected, list) or not isinstance(params, dict):
            raise ApiError("parts must be a list and params an object")
        try:
            parts_mod.engine_settings(atlas, selected)
            element = float(params["elementSize"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ApiError(f"invalid run settings: {exc}")
        check_size(prep_dir, element)
        if "arteries" in selected or "axons" in selected:
            raise ApiError(NOT_IN_BROWSER.format(why="arteries and axon fibres use registered atlases and tractography"))
        prep_job = _json_read(prep_dir / "job.json", {})
        job = {"id": _new_id("r"), "kind": "mesh", "prep": prep_dir.name, "atlas": atlas, "parts": selected, "params": params,
               "name": body.get("name") or f"Human {element:.2f} mm", "input_name": body.get("input_name") or prep_job.get("input_name"),
               "created": time.time()}
        _spawn(job, "mesh")
        return 200, "json", {"id": job["id"]}
    if method == "GET" and seg == ["jobs"]:
        kind = query.get("kind")
        items = [_job_item(d) for d in OUTPUTS.iterdir() if d.is_dir()] if OUTPUTS.exists() else []
        items = [i for i in items if i and (not kind or i["kind"] == kind)]
        items.sort(key=lambda i: i.get("created") or 0, reverse=True)
        return 200, "json", {"jobs": items}
    if len(seg) >= 2 and seg[0] == "jobs":
        job_id = seg[1]
        job_dir = _job_dir(job_id)
        if method == "GET" and len(seg) == 2:
            payload = job_payload(job_id)
            if not payload:
                raise ApiError("job not found", 404)
            return 200, "json", payload
        if method == "POST" and seg[2:] == ["cancel"]:
            raise ApiError("cancel is handled by the page", 400)
        if method == "DELETE" and len(seg) == 2:
            if not job_dir:
                raise ApiError("job not found", 404)
            shutil.rmtree(job_dir, ignore_errors=True)
            LOGS.pop(job_id, None)
            return 200, "json", {"ok": True}
        if method == "GET" and seg[2] == "file" and len(seg) >= 4:
            if not job_dir:
                raise ApiError("job not found", 404)
            try:
                target = _inside(job_dir, "/".join(seg[3:]))
            except ValueError as exc:
                raise ApiError(str(exc))
            if not target.is_file():
                raise ApiError("file not found", 404)
            return 200, MIMES.get(target.suffix.lower(), "application/octet-stream"), target.read_bytes()
        if method == "GET" and seg[2:] == ["report"]:
            report = job_dir / "report.txt" if job_dir else None
            if not report or not report.is_file():
                raise ApiError("report not found", 404)
            return 200, "text/plain", report.read_bytes()
        if method == "POST" and seg[2:] == ["export"]:
            import exporters

            if not job_dir:
                raise ApiError("job not found", 404)
            formats = body.get("formats")
            if not isinstance(formats, list) or not formats:
                raise ApiError("formats (a list) is required")
            folder = EXPORTS / job_id
            shutil.rmtree(folder, ignore_errors=True)
            try:
                written = exporters.export(job_dir, formats, folder, body.get("parts"))
            except (ValueError, FileNotFoundError) as exc:
                raise ApiError(str(exc))
            files = {p.relative_to(folder).as_posix(): p.read_bytes() for p in written}
            shutil.rmtree(folder, ignore_errors=True)
            return 200, "files", files
    if method == "GET" and len(seg) == 3 and seg[0] == "atlases" and seg[2] == "parts":
        if seg[1] not in parts_mod.ATLASES:
            raise ApiError(f"unknown atlas {seg[1]!r}", 404)
        tree = parts_mod.part_tree(seg[1])
        return 200, "json", {"tree": tree, "defaults": parts_mod.default_selection(seg[1])}
    if method == "POST" and seg == ["counts"]:
        import counts

        prep_dir = _job_dir(body.get("prep") or "")
        if not prep_dir:
            raise ApiError("prepared input not found", 404)
        try:
            return 200, "json", counts.compute(prep_dir, body.get("params") or {})
        except (ValueError, FileNotFoundError) as exc:
            raise ApiError(str(exc))
    if method == "GET" and seg == ["system"]:
        return 200, "json", {"cpu": 0.0, "ram": 0.0, "gpu": None}
    if method == "GET" and seg == ["folders", "default"]:
        return 200, "json", {"path": "Your computer (you choose the folder when you export)"}
    raise ApiError(f"no such route: {method} {path}", 404)


def call(method: str, path: str, query_json: str = "{}", body_json: str = "null", upload=None):
    """Entry point for the worker. Returns (json string with status/mime, bytes)."""
    try:
        status, mime, payload = route(method, path, json.loads(query_json), json.loads(body_json),
                                      {k: bytes(v) for k, v in upload.items()} if upload else None)
    except ApiError as error:
        return json.dumps({"status": error.status, "mime": "json"}), json.dumps({"error": str(error)}).encode()
    except Exception as error:
        logging.getLogger("webapi").exception("request %s failed", path)
        return json.dumps({"status": 500, "mime": "json"}), json.dumps({"error": str(error) or type(error).__name__,
                                                                         "trace": traceback.format_exc()[-1500:]}).encode()
    if mime == "json":
        return json.dumps({"status": status, "mime": "application/json"}), json.dumps(payload, default=str).encode()
    if mime == "files":                           # export: several files; the page saves them
        names = list(payload)
        return (json.dumps({"status": status, "mime": "files", "names": names, "sizes": [len(payload[n]) for n in names]}),
                b"".join(payload[n] for n in names))
    return json.dumps({"status": status, "mime": mime}), payload
