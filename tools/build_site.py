"""Builds the website folder from the desktop AnatoMesh folder.

    python tools/build_site.py <AnatoMesh desktop folder> [--interface-only]   e.g. .../Updated_OCT20/AnatoMesh

--interface-only copies and edits the interface and leaves py/ as it is.

* the built interface (app/dist) is copied to the site root; index.html gets one extra <script> (web/shim.js), and the bundle gets TWO
  one-constant edits for the browser edition (BROWSER_EDITS below: the element-size slider reaches 6 mm instead of 2 mm, and the
  default is 3 mm instead of 1 mm, because a browser tab cannot hold a 1 mm head). The build stops if the bundle no longer matches.
* the desktop engine (backend/engine and the job modules) is copied UNCHANGED to py/ (byte for byte; the test checks it)
* py/manifest.json lists the files the browser downloads
* backend/data (130 MB of atlases) is NOT copied: the browser edition needs none of it (label maps need no registration)
Run it again whenever the desktop folder changes.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
JOB_MODULES = ["runner.py", "parts.py", "counts.py", "exporters.py", "viewer_files.py", "report.py", "atlas_data.py"]
# (old, new, why): exact text in the compiled interface -> text for the browser edition
BROWSER_EDITS = [
    ("{label:`Element Size`,value:e.elementSize,min:.05,max:2,step:.05,", "{label:`Element Size`,value:e.elementSize,min:.05,max:6,step:.05,",
     "element-size slider reaches 6 mm"),
    ("elementSize:1,passes:3,strength:.3", "elementSize:3,passes:3,strength:.3", "default element size 3 mm"),
]
OWN_PY = ["webstubs.py", "webapi.py"]                                    # written for the browser edition, kept in py/


def build_interface(dist: Path) -> None:
    """Copy the built interface to the site root and apply the browser edition's edits."""
    for item in dist.iterdir():
        target = SITE / item.name
        if item.name == "index.html":
            continue
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
    for bundle in (SITE / "assets").glob("index-*.js"):
        js = bundle.read_text(encoding="utf-8")
        for old, new, why in BROWSER_EDITS:
            if js.count(old) != 1:
                sys.exit(f"the interface bundle changed: cannot apply the browser edit '{why}' (found {js.count(old)} matches). Update BROWSER_EDITS.")
            js = js.replace(old, new)
        bundle.write_text(js, encoding="utf-8")
    html = (dist / "index.html").read_text(encoding="utf-8")
    if "web/shim.js" not in html:
        html = html.replace("<script type=\"module\"", "<script src=\"./web/shim.js\"></script>\n    <script type=\"module\"", 1)
    (SITE / "index.html").write_text(html, encoding="utf-8")


def main(desktop: Path, interface_only: bool = False) -> None:
    dist, backend = desktop / "app" / "dist", desktop / "backend"
    if not (dist / "index.html").exists() or not (backend / "engine").is_dir():
        sys.exit(f"{desktop} is not an AnatoMesh folder (needs app/dist and backend/engine)")
    build_interface(dist)
    if interface_only:
        print(f"interface copied from {dist}")
        return
    # engine, unchanged
    py = SITE / "py"
    py.mkdir(exist_ok=True)
    if (py / "engine").exists():
        shutil.rmtree(py / "engine")
    shutil.copytree(backend / "engine", py / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in JOB_MODULES:
        shutil.copy2(backend / name, py / name)
    files = sorted(p.relative_to(py).as_posix() for p in py.rglob("*.py") if "__pycache__" not in p.parts)
    (py / "manifest.json").write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    hashes = {f: hashlib.sha256((py / f).read_bytes()).hexdigest() for f in files if f not in OWN_PY}
    (py / "engine_sha256.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    print(f"site built: {len(files)} python files, interface copied from {dist}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), interface_only="--interface-only" in sys.argv)
