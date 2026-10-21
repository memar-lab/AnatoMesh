# Test pair (samples)

Synthetic-free test data for AnatoMesh: a T1 image and its brain label map on the same grid. No lab subject data.

| File | What |
|---|---|
| `test_t1.nii.gz` | T1-weighted MRI, ICBM 2009 nonlinear symmetric template (1 mm), int16, 197 x 233 x 189 |
| `test_t1_seg.nii.gz` | Tissue label map on the same grid and affine, int16 |
| `make_test_pair.py` | Rebuilds both files from `backend/data/atlas/icbm152_2009_*` |

## Source
- Image and tissue probabilities: ICBM 2009 nonlinear symmetric template, McConnell Brain Imaging Centre (MNI). Copied from `backend/data/atlas/icbm152_2009_t1.nii.gz`.
- Label map: argmax of the GM, WM and CSF probability maps (`icbm152_2009_{gm,wm,csf}_prob.nii.gz`), inside the template brain mask (`icbm152_2009_brainmask.nii.gz`, threshold > 0.5). It is a probability-based tissue map, not a manual or FreeSurfer segmentation.

## Licence and copyright
Copyright notice for the ICBM 2009 nonlinear template (MNI / McConnell Brain Imaging Centre, McGill University):

> Copyright (C) McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University. ICBM 2009 nonlinear symmetric template. Citation: Fonov V, Evans AC, McKinstry RC, Almli CR, Collins DL. Unbiased nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage 2009;47:S102. Fonov V, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL. Unbiased average age-appropriate atlases for pediatric studies. NeuroImage 2011;54(1):313-327.

**Licence text: not yet pasted.** The full McGill licence text was not retrieved (the web search did not return it, and the lab copy has no licence file). Paste it from the BIC ICBM152 2009 download page before any redistribution outside the lab. The code in `make_test_pair.py` is ours.

## Labels (tissue scheme, `backend/viewer_files.py`)
| Code | Tissue | Voxels |
|---|---|---|
| 0 | background | 6,788,750 |
| 1 | gray matter | 1,122,477 |
| 2 | white matter | 636,713 |
| 6 | CSF | 127,349 |

Codes 3 (brainstem), 4 (cerebellum), 5 (deep nuclei), 7 (skull), 8 (scalp) are not present: the template probability maps do not separate them. The labels cover 100% of the brain mask (1,886,539 voxels).

## Grid
- Voxel size: 1.0 x 1.0 x 1.0 mm, RAS, identical shape and affine for both files.

## How to use
- Desktop or web app: choose `samples/test_t1_seg.nii.gz` as a label-map input, atlas Aseg. Prepare runs on it (verified below).
- Or `test_t1.nii.gz` as a raw T1 (runs the full segmentation; slower).
- Rebuild: `python samples/make_test_pair.py` (needs nibabel and numpy).

## Checks
- Both files load in nibabel; same shape (197, 233, 189) and affine.
- Desktop prepare (`runner.py prepare`, runtime Python 3.12, atlas `aseg`) on the seg finished with status `done`, input kind `labels`, scheme `tissue`, 3 labels. Output structures: 22 aseg codes covering all 1,886,539 brain-mask voxels.
