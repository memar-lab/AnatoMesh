"""Run manifest: everything needed to say exactly how a deck was made, and to check it later.

manifest.json in the run folder holds the command line, the full MeshingConfig, the input
files with SHA-256, a fingerprint of the engine source (so
every .py file is hashed), the Python and library versions, and the outputs with SHA-256.
`verify(run_dir)` re-hashes the outputs and says which changed since the run.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIRS = ("anatomy", "export", "mesh", "pipeline", "tools")
_LIBRARIES = ("numpy", "scipy", "nibabel", "meshio", "skimage", "SimpleITK", "PySide6",
              "matplotlib")
_OUTPUT_PATTERNS = ("*.k", "*.vtk", "*.inp", "head_labels.nii.gz", "segmentation.nii.gz",
                    "scorecard.json", "metrics.json")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: Path = ROOT) -> dict:
    """SHA-256 over every engine .py file (path + content), and the file count."""
    digest = hashlib.sha256()
    files = sorted([p for d in _SOURCE_DIRS for p in (root / d).rglob("*.py")]
                   + sorted(root.glob("*.py")))
    for p in files:
        digest.update(str(p.relative_to(root)).replace("\\", "/").encode())
        digest.update(p.read_bytes())
    return {"sha256": digest.hexdigest(), "files": len(files)}


def library_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in _LIBRARIES:
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absent libraries are recorded as absent
            out[name] = None
    return out


def _config_dict(config) -> dict:
    if config is None:
        return {}
    if dataclasses.is_dataclass(config):
        return json.loads(json.dumps(dataclasses.asdict(config), default=str))
    return json.loads(json.dumps(vars(config), default=str))


def write_manifest(run_dir: Path, config=None, inputs: Optional[dict] = None,
                   argv: Optional[Iterable[str]] = None, extra: Optional[dict] = None) -> Path:
    """Write run_dir/manifest.json and return its path."""
    run_dir = Path(run_dir)
    files_in = {}
    for role, path in (inputs or {}).items():
        if path and Path(path).is_file():
            p = Path(path)
            files_in[role] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
    outputs = {}
    for pattern in _OUTPUT_PATTERNS:
        for p in sorted(run_dir.glob(pattern)):
            outputs[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "run_id": run_dir.name,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": list(argv if argv is not None else sys.argv),
        "inputs": files_in,
        "config": _config_dict(config),
        "source": source_fingerprint(),
        "libraries": library_versions(),
        "outputs": outputs,
    }
    try:
        from engine import params

        manifest.update(params.run_record(config))
    except Exception as exc:  # the manifest must still be written
        manifest["parameters_error"] = repr(exc)
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


def verify(run_dir: Path) -> dict:
    """Re-hash the outputs listed in the manifest: {"changed": [...], "missing": [...]}."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    changed, missing = [], []
    for name, rec in manifest.get("outputs", {}).items():
        p = run_dir / name
        if not p.exists():
            missing.append(name)
        elif sha256(p) != rec["sha256"]:
            changed.append(name)
    same_source = source_fingerprint()["sha256"] == manifest.get("source", {}).get("sha256")
    return {"changed": changed, "missing": missing, "source_unchanged": same_source}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--write", action="store_true", help="(re)write the manifest instead")
    args = parser.parse_args()
    if args.write:
        print(write_manifest(args.run_dir))
    else:
        print(json.dumps(verify(args.run_dir), indent=1))
