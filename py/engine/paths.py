"""Where the engine reads its data and writes its runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# bundled reference data (atlases, templates, reference membranes)
DATA_DIR = Path(os.environ.get("ANATOMESH_DATA") or Path(__file__).resolve().parents[1] / "data").resolve()
# per-user work folder: outputs/<run_id>, inputs/<run_id>, scratch
HOME = Path(os.environ.get("ANATOMESH_HOME")
            or Path(os.environ.get("LOCALAPPDATA") or Path.home(), "AnatoMesh")).resolve()

READ_ONLY_ROOT = DATA_DIR
DTI_READ_ONLY_ROOT = DATA_DIR
READ_WRITE_ROOT = HOME
INPUTS_DIR = HOME / "inputs"
ATLAS_DIR = DATA_DIR / "atlas"
READ_ONLY_ROOTS = (DATA_DIR, INPUTS_DIR)
SCRATCH_ROOT = HOME / "scratch"


def write_roots() -> tuple[Path, Path]:
    return (READ_WRITE_ROOT, SCRATCH_ROOT)


PATHS: dict[str, Path] = {"reference_root": DATA_DIR, "diffusion_root": DATA_DIR}


def root(name: str) -> Path:
    if name not in PATHS:
        raise KeyError(f"no data root {name!r}; known: {sorted(PATHS)}")
    return PATHS[name]


_OUTPUTS_ROOT = None


def outputs_root() -> Path:
    return _OUTPUTS_ROOT or READ_WRITE_ROOT / "outputs"


def set_outputs_root(path) -> None:
    global _OUTPUTS_ROOT
    _OUTPUTS_ROOT = None if path is None else Path(path)


MAX_ITERATIONS = 50


class FilesystemViolationError(Exception):
    """Raised when an operation would read/write outside its allowed root."""


def _is_inside(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def assert_read_only_path(path: Path | str) -> Path:
    p = Path(path)
    if not any(_is_inside(p, root) for root in READ_ONLY_ROOTS):
        raise FilesystemViolationError(f"Refusing access: {p} is not inside {READ_ONLY_ROOTS}")
    return p.resolve()


def assert_write_path(path: Path | str) -> Path:
    p = Path(path)
    if any(_is_inside(p, root) for root in READ_ONLY_ROOTS):
        raise FilesystemViolationError(f"Refusing write: {p} is inside a read-only root")
    if not any(_is_inside(p, root) for root in write_roots()):
        raise FilesystemViolationError(f"Refusing write: {p} is not inside {write_roots()}")
    return p.resolve()


def safe_open_read(path: Path | str, mode: str = "r", **kwargs):
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        raise ValueError(f"safe_open_read() forbids write-capable mode {mode!r}")
    p = Path(path)
    if not any(_is_inside(p, root) for root in (*READ_ONLY_ROOTS, *write_roots())):
        raise FilesystemViolationError(f"Refusing read: {p} is outside all allowed roots")
    return open(p, mode, **kwargs)


def safe_open_write(path: Path | str, mode: str = "w", **kwargs):
    p = assert_write_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode, **kwargs)
