# AnatoMesh, browser edition

Hexahedral finite-element meshes of the human head from a **label map**, in your web browser. Open the page, choose a NIfTI label file,
mesh it, and export LS-DYNA (`.k`), Abaqus (`.inp`), VTK, STL, OBJ, NIfTI, CSV or PDF into a folder on **your own computer**.

**Nothing is uploaded and there is no server.** GitHub only hands out static files. The engine (Python with NumPy / SciPy, compiled to
WebAssembly by [Pyodide](https://pyodide.org)) runs inside your browser tab, on your processor and memory.

Live site (after you enable Pages): `https://memar-lab.github.io/AnatoMesh/`

## What it is

The user interface and the meshing engine are the desktop AnatoMesh, **unchanged**. This repository adds only the layer that lets them run
in a browser:

```
index.html + assets/ + models/   the desktop interface (React), as built; two constants changed (see below)
web/shim.js                      answers the interface's /api calls itself (the interface believes it talks to a local server)
web/worker.js                    Web Worker that starts Pyodide and runs the engine
py/webapi.py                     the same routes as the desktop's server.py, in-process (no Flask, no subprocesses)
py/webstubs.py                   what the browser lacks: stubs for ANTs / SimpleITK / dipy, no numba, no threads
py/engine/, py/runner.py, ...    the desktop engine, byte for byte (tests/test_webapi.py checks it)
py/vendor.zip                    nibabel, meshio, trimesh (pure Python, MIT) so the page never needs PyPI
```

```
your browser tab                                         your computer's CPU / RAM
+------------------------------+   messages   +--------------------------------------------+
| AnatoMesh interface (React)  | <==========> | Web Worker: Pyodide (CPython + NumPy +     |
| web/shim.js answers /api/*   |              | SciPy + scikit-image + matplotlib)         |
| jobs: status copied to page  |              | py/webapi.py -> runner.prepare / .mesh     |
+--------------+---------------+              +--------------------------------------------+
               | Export: File System Access API -> a folder you choose (Chrome / Edge), or one .zip download
```

## What works in the browser, and what does not

| | Browser edition | Desktop edition |
|---|---|---|
| Input: label map (tissue codes 1-8, or FreeSurfer aseg / aparc codes), `.nii` / `.nii.gz` | yes | yes |
| Atlases: Aseg, Aseg + Aparc (needs 1000-2999 labels), Destrieux (needs 11100-12199 labels), Whole Brain (tissue codes) | yes | yes |
| Raw T1 MRI (segmented by ANTs registration), Harvard-Oxford, atlases made by registration | **no** | yes |
| FreeSurfer subject folders, diffusion series, axon fibres, arteries | **no** | yes |
| Hex meshing, smoothing, quality, LS-DYNA deck, other exports, viewer, scorecard | yes | yes |
| Element size | 3 mm default, slider to 6 mm; runs predicted above 400,000 elements are refused with advice (about 2.5 mm for the sample head) | 0.05 mm and up |
| Speed | single thread, no compiled kernels (see numbers below) | multi-core, numba |
| Cancel a job | stops the whole engine and clears earlier results of the session | stops that job |

Refused inputs get a plain message in the interface ("this input needs the desktop version ...").

## Same results as the desktop engine

The browser edition runs the desktop engine's files unchanged, and its output matches the desktop's. Checked on this repository's sample
(`samples/fsaverage_aparc_aseg.nii.gz`, a cropped FreeSurfer fsaverage aparc+aseg, 1 mm), desktop engine (numba, 8 threads, full atlas data)
vs the browser (Pyodide, no numba, no threads, no atlas data), default settings otherwise:

| Atlas, element size | Elements / nodes | `.k` deck | Other files |
|---|---|---|---|
| Aseg, 3 mm | 157,845 / 121,848 | **byte-identical** | viewer surfaces (before / after smoothing) byte-identical; quality, element-volume, edge-ratio and part maps identical; histograms, per-part statistics equal to 1e-15; full-precision node coordinates differ by at most 1.6e-13 mm |
| Aseg + Aparc, 4 mm | 87,839 / 67,661 | **byte-identical** | not compared |
| Whole Brain, 5 mm | 59,761 / 47,181 | identical except 22 coordinates that differ by at most 1e-10 mm (values near 1e-6 mm printed with 6 digits) | not compared |

"Byte-identical" means after normalising the two things that cannot match: Windows line endings and the run id in the deck title
(`tests/compare_decks.py` does exactly that).

**What it took.** A first browser run was *not* identical: 18 % of the node coordinates differed, by up to 0.7 mm. The cause was found by
comparing the two engines' logs, then the inputs of the smoothing step (identical), then that step's own classification of nodes: NumPy's
default integer is 64-bit on a desktop and **32-bit in WebAssembly**, and `hex_quality.classify_nodes` mixes an `.astype(int)` array with a
64-bit sentinel, which silently gave a different answer. `webstubs.patch_engine()` now widens that array to int64 around the call; the engine
files stay untouched (`tests/test_webapi.py` has a regression test). Ruled out on the way: CPU vector instructions (disabling AVX2 / FMA
changed nothing), NumPy / SciPy version differences between 2.4.6 / 1.18.0 and 2.5.3 / 1.18.1 (identical decks natively), and OpenBLAS kernel
choice. The probe `tests/numeric_probe.py` shows which NumPy / SciPy primitives differ bit-wise between native and WebAssembly (math library,
BLAS / LAPACK); none of them changes the mesh.

**Limit of the claim.** Identical on the three settings above. Another input or setting could still reach a place where a 32-bit integer
matters; check any new kind of input with `tests/native_run.py` (desktop) and `tests/compare_decks.py` (against the deck exported from the
browser) before trusting it.

## Time and memory

Aseg, 3 mm (157,845 elements): about 155 s in the browser (desktop: 89 s with numba on 8 threads, 146 s single-threaded without numba, native Python),
464 MB of WebAssembly memory. Aseg + Aparc, 4 mm (87,837 elements): 141 s.

## Deploy to GitHub Pages

1. Push this folder to `https://github.com/memar-lab/AnatoMesh` (branch `main`).
2. In the repository: **Settings -> Pages -> Build and deployment -> Source: GitHub Actions**.
3. `.github/workflows/pages.yml` tests the engine, then publishes `index.html`, `assets/`, `models/`, `py/`, `web/`, `samples/`.

Local preview (a Web Worker needs a real web server, not `file://`):

```bash
python -m http.server 8000
```
then open <http://localhost:8000/>.

## Rebuild from the desktop folder

When the desktop AnatoMesh changes, rebuild this site from it:

```bash
python tools/build_site.py <path to the desktop AnatoMesh folder>     # interface + engine, copied; manifest and hashes rewritten
python tools/build_vendor.py                                          # only if nibabel / meshio / trimesh change
python tests/test_webapi.py                                           # the browser-style engine must still pass
```

`build_site.py` stops if the compiled interface no longer contains the two constants it edits (element-size slider limit and default).
The desktop `backend/data` folder (130 MB of atlases and templates) is **not** copied: label maps need none of it.

## Tests

```bash
pip install numpy scipy scikit-image nibabel matplotlib meshio trimesh
python tests/test_webapi.py --element 4        # routes, refused inputs, upload -> prepare -> counts -> mesh -> export
python tests/native_run.py <desktop backend> samples/fsaverage_aparc_aseg.nii.gz out.json --element 3            # desktop engine
python tests/native_run.py <desktop backend> samples/fsaverage_aparc_aseg.nii.gz out_web.json --element 3 --browser   # browser-style
python tests/compare_decks.py out.json <deck exported from the browser>          # byte comparison (line endings and run id normalised)
python tests/numeric_probe.py                                                    # hashes of NumPy / SciPy primitives, to diff native vs Pyodide
```

## AnatoReg (image registration)

AnatoReg is our own image-registration code, written in NumPy and SciPy so that it runs in the browser through Pyodide. In this edition it replaces ANTs. It is an independent implementation, not ANTs code.

What it does now:
- rigid and affine registration, driven by a Mattes mutual-information metric;
- SyN deformable registration;
- reading and writing transforms, and applying them to images (`apply_transforms`).

What it does not do yet: N4 bias-field correction.

This release ships an early version. A newer AnatoReg, with more accurate SyN and N4, is in progress. The package is named `anatoreg` (import name `anatoreg`).

Samples (in `samples/`, see `samples/README.md`):
- `test_t1.nii.gz`: ICBM 2009 nonlinear symmetric T1 template, 1 mm.
- `test_t1_seg.nii.gz`: tissue labels built from the template's probability maps (gray matter, white matter, CSF).
- `fsaverage_aparc_aseg.nii.gz`: cropped FreeSurfer fsaverage aparc+aseg (FreeSurfer licence).

The ICBM 2009 licence text is not yet pasted into this repository. Paste it from the BIC ICBM152 download page before redistributing outside the lab.

## Limits to know

* A browser tab can use about 2 GB. A whole head at 3 mm is about 160,000 elements (tested); the page refuses jobs predicted above 400,000 elements, which is an estimate from two measurements, not a tested limit.
* The first visit downloads Python and its libraries (about 60-90 MB) from the jsDelivr CDN; later visits use the browser cache. Needs internet once.
* Writing straight into a folder needs Chrome or Edge. Other browsers (and phones) download one .zip.
* Needs a browser with module Web Workers (Chrome / Edge, Firefox 114+, Safari 15+).

## Third-party material

See `THIRD_PARTY_NOTICES.md` (Pyodide, nibabel, meshio, trimesh, the sample volume).

## Verified and not verified

Verified (this repository, 2026-10-20):
* Engine files in `py/` are byte-identical to the desktop ones (`tests/test_webapi.py`, 77 files).
* In ordinary Python, without numba / ANTs / threads / atlas data: upload, prepare, counts, mesh, viewer files, report and seven export formats
  through the same routes the interface uses; refused inputs (raw MRI, folders, diffusion, tissue-only with Aseg, Harvard-Oxford, Destrieux
  without labels, arteries / axons, too-fine element size) give clear messages.
* In a Chromium-based browser pane over a local web server: the page starts Pyodide, the unchanged interface loads, a label map uploads,
  prepare and mesh run with live progress, the Output screen shows the quality map and scorecard, and Export produced a valid `.k` deck
  as a .zip download. Results compared with the desktop engine as described above (three settings identical). Cancel stopped a running job, restarted the engine and
  cleared earlier results, as documented. At phone width (390 px) the start screen and the loading overlay fit.

Not verified:
* Writing straight into a chosen folder (File System Access API): the code is in `web/shim.js`, but the test pane gave no user gesture, so only
  the .zip fallback was exercised.
* Firefox, Safari, phones, Windows / macOS / Linux browsers other than the one above.
* Jobs above about 160,000 elements; the 400,000-element guard is an extrapolation.
* The full workflow on a phone-sized screen (only the start screen was looked at), deleting jobs, the Compare Runs and Run History screens, presets save / load.
* The deployed GitHub Pages site (it does not exist until you push and enable Pages).
* Whether the Pyodide download from jsDelivr works behind your organisation's firewall.

