"""Runs the ORIGINAL desktop engine (runner.prepare + runner.mesh, in this process) on a label map and records the result.

    python native_run.py <backend folder> <label nii> <out json> [--element 3] [--atlas aseg] [--browser] [--audit]

--browser  rehearse the browser: stub ants / SimpleITK / dipy, block numba / edt / ..., no threads (py/webstubs.py)
--audit   list the files under backend/data that were opened (what the web build has to ship)
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("backend"); ap.add_argument("label"); ap.add_argument("out")
ap.add_argument("--element", type=float, default=3.0)
ap.add_argument("--atlas", default="aseg")
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--browser", action="store_true")
ap.add_argument("--audit", action="store_true")
ap.add_argument("--home", default="")
args = ap.parse_args()

home = Path(args.home or tempfile.mkdtemp(prefix="anatomesh_native_"))
os.environ["ANATOMESH_HOME"] = str(home)
backend = Path(args.backend).resolve()
sys.path.insert(0, str(backend))

if args.browser:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))
    import webstubs
    webstubs.install()

opened = set()
if args.audit:
    data_root = str(backend / "data")

    def hook(event, a):
        if event == "open" and isinstance(a[0], (str, bytes, os.PathLike)):
            p = os.fspath(a[0])
            if isinstance(p, str) and p.startswith(data_root):
                opened.add(p[len(data_root) + 1:].replace("\\", "/"))
    sys.addaudithook(hook)

import runner                                                    # noqa: E402
import parts as parts_mod                                        # noqa: E402
from engine.paths import INPUTS_DIR                              # noqa: E402

INPUTS_DIR.mkdir(parents=True, exist_ok=True)
(home / "outputs").mkdir(parents=True, exist_ok=True)
inp = INPUTS_DIR / "i-test"
inp.mkdir(parents=True, exist_ok=True)
src = inp / Path(args.label).name
src.write_bytes(Path(args.label).read_bytes())

t0 = time.time()
prep = home / "outputs" / "p-test"; prep.mkdir(parents=True, exist_ok=True)
runner.prepare({"id": "p-test", "kind": "prepare", "atlas": args.atlas, "axes": None, "input": {"path": str(src), "dwi": None},
                "input_name": src.name}, prep)
t_prep = time.time() - t0

params = dict(elementSize=args.element, passes=args.passes, strength=0.3, minJac=0.5, untilMin=False, targeted=True,
              adjustThreshold=0.5, fillHoles=False, fillWith="neighbour", fillMax=1000, combine=True,
              outerOn=True, outerLevel=1.2, outerIter=3, innerOn=True, innerLevel=0.6, innerIter=5)
selected = parts_mod.default_selection(args.atlas)
mesh_dir = home / "outputs" / "r-test"; mesh_dir.mkdir(parents=True, exist_ok=True)
t1 = time.time()
runner.mesh({"id": "r-test", "kind": "mesh", "prep": "p-test", "atlas": args.atlas, "parts": selected, "params": params,
             "name": "test", "input_name": src.name}, mesh_dir)
t_mesh = time.time() - t1

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result = {"home": str(home), "prepare_s": round(t_prep, 1), "mesh_s": round(t_mesh, 1), "atlas": args.atlas, "element": args.element,
          "summary": json.loads((mesh_dir / "summary.json").read_text()),
          "files": {p.relative_to(mesh_dir).as_posix(): [p.stat().st_size, sha(p) if p.stat().st_size < 200e6 else None]
                    for p in sorted(mesh_dir.rglob("*")) if p.is_file()},
          "data_opened": sorted(opened)}
Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("prepare_s", "mesh_s", "summary")}, indent=1))
