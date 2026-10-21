# Third-party material in the browser edition

| Component | Where | Licence |
|---|---|---|
| Pyodide (CPython, NumPy, SciPy, scikit-image, matplotlib, pandas, ... for WebAssembly) | downloaded from the jsDelivr CDN when the page runs; not stored here | Mozilla Public License 2.0 (Pyodide); each bundled package under its own licence |
| nibabel 5.4.2 | `py/vendor.zip` (with its licence file under `licenses/`) | MIT |
| meshio 5.3.5 | `py/vendor.zip` | MIT |
| trimesh 5.1.0 | `py/vendor.zip` | MIT |
| AnatoMesh interface and engine | `index.html`, `assets/`, `models/`, `py/engine/`, `py/*.py` copied from the desktop AnatoMesh | the licence of the AnatoMesh project (to be stated by its owners) |
| `samples/fsaverage_aparc_aseg.nii.gz` | cropped copy of the `aparc+aseg.mgz` of FreeSurfer's `fsaverage` subject (as bundled with the desktop AnatoMesh) | FreeSurfer licence; check that redistribution fits your use before making the repository public |

The browser edition does not include the desktop edition's atlases and templates (ICBM 2009, Harvard-Oxford, HCP1065 tracts, cerebral
artery atlas), so their licences do not apply to this repository.
