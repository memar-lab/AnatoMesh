"""Hashes the result of common NumPy / SciPy operations on fixed data, to find which primitive gives different bits on another platform.

    python tests/numeric_probe.py            prints {operation: sha256 of the result bytes}

Run it natively and in Pyodide (the browser) and diff the two outputs: an operation whose hash differs is a source of platform-dependent
results (math library, BLAS / LAPACK, summation order). Used to explain why the browser mesh is not bit-identical to the desktop one.
"""
import hashlib
import json

import numpy as np
import scipy
from scipy import ndimage, spatial, sparse


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def probe():
    rng = np.random.default_rng(12345)
    x = rng.uniform(0.1, 10.0, 100_000)
    y = rng.uniform(-1.0, 1.0, 100_000)
    pts = rng.normal(size=(20_000, 8, 3))
    vec = rng.normal(size=(20_000, 3))
    A = rng.normal(size=(200, 200))
    B = rng.normal(size=(200, 200))
    S = A @ A.T + 200 * np.eye(200)
    ties = rng.integers(0, 50, 100_000)
    vol = rng.normal(size=(40, 44, 36))
    mask = vol > 0.3
    out = {"numpy": np.__version__, "scipy": scipy.__version__}
    # elementwise math (the C math library)
    for name in ("exp", "log", "sqrt", "sin", "cos", "tan", "tanh", "arctan"):
        out[name] = h(getattr(np, name)(x))
    out["arccos"] = h(np.arccos(y))
    out["arcsin"] = h(np.arcsin(y))
    out["arctan2"] = h(np.arctan2(y, x))
    out["power_1.5"] = h(np.power(x, 1.5))
    out["power_arr"] = h(np.power(x, y))
    out["hypot"] = h(np.hypot(x, y))
    # reductions (summation order)
    out["sum"] = h(np.sum(x))
    out["sum_axis"] = h(pts.sum(axis=1))
    out["sum_last"] = h(pts.sum(axis=-1))
    out["mean"] = h(np.mean(x))
    out["std"] = h(np.std(x))
    out["cumsum"] = h(np.cumsum(x))
    out["dot"] = h(np.dot(x, y))
    out["norm"] = h(np.linalg.norm(vec, axis=1))
    out["norm_all"] = h(np.linalg.norm(x))
    out["percentile"] = h(np.percentile(x, [10, 50, 90]))
    # BLAS / einsum
    out["matmul"] = h(A @ B)
    out["einsum_mm"] = h(np.einsum("ij,jk->ik", A, B))
    out["einsum_nkj"] = h(np.einsum("nkj,nj->nk", pts, vec))
    out["einsum_cross"] = h(np.einsum("ij,ij->i", vec, vec))
    out["cross"] = h(np.cross(pts[:, 0], pts[:, 1]))
    # LAPACK
    out["solve"] = h(np.linalg.solve(S, B))
    out["inv"] = h(np.linalg.inv(S))
    out["det"] = h(np.linalg.det(S))
    out["eigh"] = h(np.linalg.eigh(S)[0])
    out["svd"] = h(np.linalg.svd(A, compute_uv=False))
    out["lstsq"] = h(np.linalg.lstsq(A, B[:, 0], rcond=None)[0])
    # sorting with ties
    out["argsort_default"] = h(np.argsort(ties))
    out["argsort_stable"] = h(np.argsort(ties, kind="stable"))
    out["unique"] = h(np.unique(ties, return_counts=True)[1])
    out["lexsort"] = h(np.lexsort((ties, ties[::-1])))
    # SciPy
    out["gaussian_filter"] = h(ndimage.gaussian_filter(vol, 1.3))
    out["edt"] = h(ndimage.distance_transform_edt(mask))
    out["edt_sampling"] = h(ndimage.distance_transform_edt(mask, sampling=[0.5, 0.5, 0.75]))
    out["zoom"] = h(ndimage.zoom(vol, 0.7, order=1))
    out["map_coords"] = h(ndimage.map_coordinates(vol, rng.uniform(0, 30, (3, 5000)), order=1))
    out["label"] = h(ndimage.label(mask)[0])
    tree = spatial.cKDTree(pts[:, 0])
    out["kdtree"] = h(tree.query(vec, k=3)[0])
    M = sparse.random(2000, 2000, density=0.01, random_state=1, format="csr")
    out["sparse_matvec"] = h(M @ x[:2000])
    from scipy.interpolate import RegularGridInterpolator
    g = np.arange(10.0)
    rgi = RegularGridInterpolator((g, g, g), rng.normal(size=(10, 10, 10)))
    out["rgi"] = h(rgi(rng.uniform(0, 9, (3000, 3))))
    return out


if __name__ == "__main__":
    print(json.dumps(probe(), indent=1))
