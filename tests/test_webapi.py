"""Drives py/webapi.py (the browser edition's /api) in ordinary Python, in the same stripped-down environment as the browser
(no numba / ants / SimpleITK / threads, no bundled atlas data).

    python tests/test_webapi.py [--element 4]          needs numpy, scipy, scikit-image, nibabel, matplotlib, meshio, trimesh

Checks: the engine files in py/ are byte-identical to the desktop ones (when the desktop folder is next to this one), the routes
answer like server.py, a label map goes through prepare -> counts -> mesh -> export, and inputs the browser cannot handle are
refused with a clear message.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--element", type=float, default=4.0)
args = ap.parse_args()

home = Path(tempfile.mkdtemp(prefix="anatomesh_webapi_"))
os.environ["ANATOMESH_HOME"] = str(home)
os.environ["ANATOMESH_DATA"] = str(home / "no_data")            # the browser edition ships no atlas data
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(HERE / "py"))

import numpy as np                                               # noqa: E402
import nibabel as nib                                            # noqa: E402
import webapi                                                    # noqa: E402


def call(method, path, query=None, body=None, upload=None):
    meta, payload = webapi.call(method, path, json.dumps(query or {}), json.dumps(body), upload)
    meta = json.loads(meta)
    return meta, payload


def ok_json(method, path, query=None, body=None, upload=None):
    meta, payload = call(method, path, query, body, upload)
    assert meta["status"] < 400, (path, payload[:400])
    return json.loads(payload)


def refused(method, path, query=None, body=None, upload=None):
    meta, payload = call(method, path, query, body, upload)
    assert meta["status"] >= 400, f"{path} should have been refused"
    return json.loads(payload)["error"]


def main():
    # 1. the engine is the desktop code, unchanged
    expected = json.loads((HERE / "py" / "engine_sha256.json").read_text())
    for name, digest in expected.items():
        assert hashlib.sha256((HERE / "py" / name).read_bytes()).hexdigest() == digest, f"{name} was modified"
    desktop = HERE.parent / "AnatoMesh" / "backend"
    if desktop.is_dir():
        for name in expected:
            src = desktop / name
            assert src.exists() and src.read_bytes() == (HERE / "py" / name).read_bytes(), f"{name} differs from the desktop file"
        print(f"engine files identical to the desktop code: {len(expected)}")

    # 1b. the 32-bit-integer fix: in WebAssembly `part` arrays are int32; classify_nodes must give the same answer as with int64
    from engine.mesh import hex_quality
    grid = np.zeros((6, 6, 6), np.int64); grid[1:5, 1:5, 1:5] = 1; grid[1:5, 1:5, 3:5] = 2
    ids = np.arange(7 ** 3).reshape(7, 7, 7)
    hexes = np.array([[ids[i, j, k], ids[i + 1, j, k], ids[i + 1, j + 1, k], ids[i, j + 1, k], ids[i, j, k + 1],
                       ids[i + 1, j, k + 1], ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]]
                      for i, j, k in np.argwhere(grid > 0)], dtype=np.int64)
    part = grid[grid > 0]
    kind64, _ = hex_quality.classify_nodes(hexes, part.astype(np.int64), 7 ** 3)
    kind32, _ = hex_quality.classify_nodes(hexes, part.astype(np.int32), 7 ** 3)
    assert (kind64 == kind32).all() and len(set(kind64.tolist())) == 3, "int32 part arrays must classify nodes like int64 ones"
    print("int32 part arrays classify nodes like int64 ones: OK")

    # 2. routes that need no job
    health = ok_json("GET", "/api/health")
    assert health["ok"] and health["web"] and health["tools"]["ants"] is False
    tree = ok_json("GET", "/api/atlases/aseg/parts")
    assert tree["tree"] and tree["defaults"]

    # 3. refused inputs
    rng = np.random.default_rng(0)
    mri = nib.Nifti1Image(rng.normal(100, 20, (20, 20, 20)).astype(np.float32), np.eye(4))
    path = home / "t1.nii"; nib.save(mri, str(path))
    msg = refused("POST", "/api/inputs", {"role": "main"}, upload={"t1.nii": path.read_bytes()})
    assert "desktop" in msg, msg
    print("raw MRI refused:", msg[:90], "...")
    tissue = nib.Nifti1Image((rng.integers(1, 4, (20, 20, 20))).astype(np.int16), np.eye(4))
    path = home / "tissue.nii"; nib.save(tissue, str(path))
    meta = ok_json("POST", "/api/inputs", {"role": "main"}, upload={"tissue.nii": path.read_bytes()})
    assert meta["kind"] == "labels" and meta["detected"]["scheme"] == "tissue"
    msg = refused("POST", "/api/prepare", body={"input_id": meta["id"], "atlas": "aseg", "dwi_id": None, "axes": None})
    assert "Whole Brain" in msg, msg
    msg = refused("POST", "/api/inputs/path", body={"path": "C:/x.nii", "role": "main"})
    assert "desktop" in msg.lower(), msg

    # 4. a real label map: upload -> prepare -> counts -> mesh -> files -> export
    sample = (HERE / "samples" / "fsaverage_aparc_aseg.nii.gz").read_bytes()
    meta = ok_json("POST", "/api/inputs", {"role": "main"}, upload={"fsaverage_aparc_aseg.nii.gz": sample})
    assert meta["kind"] == "labels" and meta["detected"]["scheme"] == "freesurfer", meta
    for atlas, needle in (("harvard", "registration"), ("destrieux", "11100")):
        assert needle in refused("POST", "/api/prepare", body={"input_id": meta["id"], "atlas": atlas, "dwi_id": None, "axes": None})
    prep = ok_json("POST", "/api/prepare", body={"input_id": meta["id"], "atlas": "aseg", "dwi_id": None, "axes": None})
    assert not prep["reused"]
    assert ok_json("GET", f"/api/jobs/{prep['id']}")["state"] == "queued"
    webapi.run_queued()
    job = ok_json("GET", f"/api/jobs/{prep['id']}")
    assert job["state"] == "done", job
    again = ok_json("POST", "/api/prepare", body={"input_id": meta["id"], "atlas": "aseg", "dwi_id": None, "axes": None})
    assert again["reused"] and again["id"] == prep["id"]
    legend = json.loads(call("GET", f"/api/jobs/{prep['id']}/file/viewer/legend.json")[1])
    assert legend["legend"]

    params = dict(elementSize=args.element, passes=2, strength=0.3, minJac=0.5, untilMin=False, targeted=True, adjustThreshold=0.5,
                  fillHoles=False, fillWith="neighbour", fillMax=1000, combine=True, outerOn=True, outerLevel=1.2, outerIter=3,
                  innerOn=True, innerLevel=0.6, innerIter=5)
    counts = ok_json("POST", "/api/counts", body={"prep": prep["id"], "params": params})
    assert counts
    assert "desktop" in refused("POST", "/api/runs", body={"prep": prep["id"], "atlas": "aseg", "parts": ["axons"], "params": params})
    big = dict(params, elementSize=1.0)                                  # a 1 mm head would be ~1.7 million elements: refused, with advice
    msg = refused("POST", "/api/runs", body={"prep": prep["id"], "atlas": "aseg", "parts": tree["defaults"], "params": big})
    assert "elements" in msg and "or larger" in msg, msg
    print("too fine refused:", msg[:120], "...")
    run = ok_json("POST", "/api/runs", body={"prep": prep["id"], "atlas": "aseg", "parts": tree["defaults"], "params": params, "name": "web test"})
    webapi.run_queued()
    job = ok_json("GET", f"/api/jobs/{run['id']}")
    assert job["state"] == "done", job.get("error") or job.get("trace")
    summary = job["summary"]
    print(f"mesh job done: {summary['elements']:,} elements, {summary['nodes']:,} nodes, "
          f"min scaled Jacobian {summary['sj_min']:.4f}, {summary['wall_time_sec']:.0f} s")
    assert summary["elements"] > 1000 and summary["negative"] == 0
    updates = [e for e in webapi.EVENTS if e.get("type") == "job" and e["id"] == run["id"]]
    assert updates and updates[-1]["payload"]["state"] == "done" and any(u["payload"]["log"] for u in updates)
    print(f"progress updates posted to the page: {len(updates)}")

    for rel in ("viewer/result.json", "viewer/surface_after.bin", "viewer/parts.nii.gz", "viewer/quality.nii.gz"):
        meta_, data = call("GET", f"/api/jobs/{run['id']}/file/{rel}")
        assert meta_["status"] == 200 and len(data) > 0, rel
    assert call("GET", f"/api/jobs/{run['id']}/report")[0]["status"] == 200

    meta_, data = call("POST", f"/api/jobs/{run['id']}/export", body={"formats": [".k", "VTK", "CSV", "NIfTI", ".inp", "STL", "OBJ"]})
    assert meta_["status"] == 200 and meta_["mime"] == "files", data[:300]
    assert sum(meta_["sizes"]) == len(data)
    print("export files:", dict(zip(meta_["names"], meta_["sizes"])))
    k_name = next(n for n in meta_["names"] if n.endswith(".k"))
    assert data[:200].lstrip().startswith(b"*") or b"*KEYWORD" in data[:2000]

    assert ok_json("DELETE", f"/api/jobs/{run['id']}")["ok"]
    print("ALL OK")


if __name__ == "__main__":
    main()
