"""Packs the pure-Python libraries the engine imports (nibabel, meshio, trimesh; all MIT) into py/vendor.zip.

    python tools/build_vendor.py [site-packages folder]       default: the AnatoMesh runtime on this PC

Pyodide ships numpy, scipy, scikit-image, matplotlib, pandas and rich, but not these three. They contain no compiled code, so the
browser edition carries them itself and never has to reach PyPI. Tests and caches are left out.
"""
import os
import sys
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
default = Path(os.environ.get("LOCALAPPDATA", "")) / "AnatoMesh" / "runtime" / "python" / "Lib" / "site-packages"
sp = Path(sys.argv[1]) if len(sys.argv) > 1 else default
LIBS = ["nibabel", "meshio", "trimesh"]
SKIP_DIRS = {"__pycache__", "tests", "test"}

out = SITE / "py" / "vendor.zip"
count = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for lib in LIBS:
        for p in sorted((sp / lib).rglob("*")):
            rel = p.relative_to(sp)
            if p.is_dir() or SKIP_DIRS & set(rel.parts) or p.suffix in (".pyc", ".pyd", ".so", ".dll"):
                continue
            z.write(p, rel.as_posix())
            count += 1
        for info in sorted(sp.glob(f"{lib}-*.dist-info")):
            for lic in list(info.glob("LICEN*")) + list(info.glob("COPYING*")) + list((info / "licenses").glob("*")):
                if lic.is_file():
                    z.write(lic, f"licenses/{lib}-{lic.name}")
                    count += 1
            meta = (info / "METADATA").read_text(encoding="utf-8", errors="replace").splitlines()
            version = next((l.split(": ", 1)[1] for l in meta if l.startswith("Version: ")), "?")
            z.writestr(f"licenses/{lib}-VERSION.txt", f"{lib} {version}\n")
print(f"{out}: {count} files, {out.stat().st_size / 1e6:.1f} MB")
