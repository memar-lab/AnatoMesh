"""Built-in engines: the work that used to need outside tools.

Each module is our implementation of a published method (the papers are cited at the top of the
module), with one settings dataclass whose fields are registered in the parameter system
(params.py), so every number can be changed from a parameter file.

    kernels        numba building blocks (sampling, resampling, histograms, random numbers)
    dti            diffusion tensor fit (replaces dipy and DSI Studio "rec")
    tracking       deterministic streamline tracking (replaces DSI Studio "trk")
    registration   rigid, affine and deformable registration (replaces ANTs)

Which engine a stage uses is chosen in MeshingConfig (axon_engine, registration_engine, ...).
EXTERNAL_TOOLS = "never" makes every stage refuse an outside tool; "allowed" keeps them
selectable for comparison.
"""
from __future__ import annotations

import os
from pathlib import Path

# compiled kernels are cached on local disk, not in __pycache__ on the network share (numba
# reads this when it is first imported, so the entry points import engines first)
if "NUMBA_CACHE_DIR" not in os.environ:
    _local = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "AnatoMesh" / "numba_cache"
    try:
        _local.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(_local)
    except OSError:
        pass

EXTERNAL_TOOLS = "allowed"          # "allowed" or "never": whether any stage may call an outside tool


def external_allowed() -> bool:
    """False when the run was told to use our own engines only."""
    return str(EXTERNAL_TOOLS).lower() != "never"
