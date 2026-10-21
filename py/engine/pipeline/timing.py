"""Stage timer: wall time, CPU time and peak memory per named pipeline stage.

    with timing.stage("prepare"):
        ...
    timing.write(run_dir)   # -> <run_dir>/timings.json

Stages nest; a nested stage is recorded as "outer/inner". Inside a long function,
timing.mark("step") closes the previous mark of the same stage and opens the next one, so
blocks can be timed without re-indenting them; the stage's exit closes its last mark.
Peak memory is the process peak so far (the OS cannot reset it), so a stage's value is an
upper bound for that stage.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_records: list[dict] = []
_stack: list[str] = []
_laps: dict[int, tuple] = {}  # stack depth -> (name, wall start, cpu start) of the open mark
_t0 = time.perf_counter()
# called with each stage's full name as it opens; a job runner sets it to report progress
on_stage = None


def _peak_rss_mb() -> float | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                               wintypes.DWORD]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 2**20
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux
    except ImportError:
        return None


def _record(full: str, w0: float, c0: float, ok: bool) -> None:
    _records.append({
        "stage": full,
        "start_s": round(w0 - _t0, 3),
        "wall_s": round(time.perf_counter() - w0, 3),
        "cpu_s": round(time.process_time() - c0, 3),
        "peak_rss_mb": None if (m := _peak_rss_mb()) is None else round(m, 1),
        "ok": ok,
    })


def _close_lap(depth: int, ok: bool = True) -> None:
    lap = _laps.pop(depth, None)
    if lap is not None:
        _record(*lap, ok)


def mark(name: str) -> None:
    depth = len(_stack)
    _close_lap(depth)
    _laps[depth] = ("/".join([*_stack, name]), time.perf_counter(), time.process_time())


@contextmanager
def stage(name: str):
    _stack.append(name)
    full = "/".join(_stack)
    w0, c0 = time.perf_counter(), time.process_time()
    ok = False
    if on_stage is not None:
        on_stage(full)
    try:
        yield
        ok = True
    finally:
        _close_lap(len(_stack), ok)
        _stack.pop()
        _record(full, w0, c0, ok)


def records() -> list[dict]:
    return sorted(_records, key=lambda r: r["start_s"])


def reset() -> None:
    global _t0
    _records.clear()
    _stack.clear()
    _laps.clear()
    _t0 = time.perf_counter()


def write(run_dir: Path | str) -> Path:
    _close_lap(0)
    out = Path(run_dir) / "timings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"total_wall_s": round(time.perf_counter() - _t0, 3),
           "cpu_count": os.cpu_count(),
           "peak_rss_mb": _peak_rss_mb(),
           "stages": records()}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return out
