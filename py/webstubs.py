"""Environment shims that let the unchanged desktop engine run inside a browser (Pyodide, WebAssembly).

What a browser does not have, and what this file does about it:

* ants                            antspyx does not run in Pyodide, so `ants` is routed to anatoreg (py/anatoreg, numpy/scipy) for the subset the engine uses.
* SimpleITK, dipy                compiled libraries that Pyodide does not ship. The engine imports them at the top of some files even when
                                 the job never calls them, so they become STUBS: the import works, and any
                                 use raises UnavailableInBrowser with a plain message (diffusion needs dipy).
* numba, edt, rtree, ...         make `import` fail, which the engine already handles (numba kernels have NumPy twins; `edt` is optional).
* ThreadPoolExecutor             a browser tab's Python cannot start threads, so it becomes a plain loop with the same interface.
* psutil, flask, pymeshlab       not needed by a job; blocked so that nothing half-works.
* the numba row formatter         ANATOMESH_PY_FORMAT=1 selects the engine's own reference Python formatter for the .k deck.
* 32-bit integers                 NumPy's default integer is 64-bit on a desktop and 32-bit in WebAssembly. Engine code that builds an array
                                 with `.astype(int)` and then mixes it with a 64-bit sentinel (hex_quality.classify_nodes) gets the wrong
                                 answer there, silently: it classified a different set of nodes as creases and the mesh came out different.
                                 patch_engine() widens such inputs to int64 around the engine's functions; the engine files are untouched.

Call install() once, before the engine is imported. The same file is used by tests/native_run.py to rehearse the browser on a desktop.
"""
from __future__ import annotations

import concurrent.futures as _cf
import os
import sys
import types

STUBS = ("SimpleITK", "dipy")
BLOCKED = ("numba", "edt", "rtree", "mapbox_earcut", "fast_simplification", "pymeshlab", "psutil", "flask", "werkzeug")

MESSAGE = ("{name} is not available in the browser edition. Raw MRI segmentation, registration-based atlases "
           "(Harvard-Oxford, Destrieux without labels) and diffusion fibres need the desktop version. "
           "Use a label map (NIfTI with tissue codes 1-8 or FreeSurfer aseg / aparc codes).")


class UnavailableInBrowser(ImportError):
    pass


class _Stub(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []                       # lets `import ants.registration` style imports resolve to more stubs
        self.__file__ = f"<{name} unavailable in the browser>"

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise UnavailableInBrowser(MESSAGE.format(name=f"{self.__name__}.{attr}"))


class _StubFinder:
    """Makes `import ants.x.y` and `from dipy.io import z` import-able as stubs."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in STUBS:
            import importlib.machinery as m
            return m.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


class SerialPool:
    """concurrent.futures.ThreadPoolExecutor without threads."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = _cf.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:           # same contract as a pool: the error comes out of .result()
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, **kwargs):
        return [fn(*args) for args in zip(*iterables)]

    def shutdown(self, *args, **kwargs):
        pass


_installed = False


def install(threads_available: bool = False) -> None:
    global _installed
    if _installed:
        return
    _installed = True
    # the deck writer has a reference Python formatter (byte-identical to the numba fast path, see lsdyna_writer.py / fastfmt.py)
    os.environ.setdefault("ANATOMESH_PY_FORMAT", "1")
    sys.meta_path.insert(0, _StubFinder())
    import anatoreg                              # the ants API, in numpy/scipy; the engine's `import ants` gets this module
    sys.modules["ants"] = anatoreg
    for name in BLOCKED:
        sys.modules[name] = None                  # `import name` -> ImportError
    if not threads_available:
        _cf.ThreadPoolExecutor = SerialPool


def patch_engine() -> None:
    """Call after the engine modules are importable (webapi does). Widens integer inputs that the desktop engine receives as int64."""
    import numpy as np
    from engine.mesh import hex_quality

    original = hex_quality.classify_nodes

    def classify_nodes(hexes, part, n_nodes, incidence=None):
        # `part` is int32 in WebAssembly (default int); the function mixes it with np.iinfo(np.int64).max
        return original(hexes, np.asarray(part, dtype=np.int64), n_nodes, incidence=incidence)

    hex_quality.classify_nodes = classify_nodes
