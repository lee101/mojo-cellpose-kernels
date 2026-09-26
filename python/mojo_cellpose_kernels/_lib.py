"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory.  Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.  `_ARITY` pins each kernel's parameter count and
is checked at load time, because a short argtypes list is silently wrong rather
than loudly wrong: ctypes converts the surplus argument with its default
`c_int` rule, which keeps a 32-bit address and only crashes once a buffer
lands above 4 GiB.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-cellpose-kernels.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double

# Parameter count of every exported kernel.
_ARITY = {
    "cpk_diffuse": 10,
    "cpk_flow_grads": 4,
    "cpk_normalize_flows": 2,
    "cpk_flow_error": 4,
    "cpk_bilinear": 7,
    "cpk_follow_flows": 9,
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    def sig(name, restype, argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        if len(fn.argtypes) != _ARITY[name]:
            raise RuntimeError(
                f"{name}: declared {len(fn.argtypes)} argtypes, kernel takes "
                f"{_ARITY[name]}"
            )

    sig("cpk_diffuse", None, [_I64] * 10)
    sig("cpk_flow_grads", None, [_I64] * 4)
    sig("cpk_normalize_flows", None, [_I64] * 2)
    sig("cpk_flow_error", None, [_I64] * 4)
    sig("cpk_bilinear", None, [_I64] * 7)
    sig("cpk_follow_flows", None, [_I64] * 9)
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def diffuse(t, nbr, center, meds, isneighbor, n_iter):
    """In-place diffusion of `t` over the neighbour graph, as `_extend_centers_gpu`."""
    t = _f64(t)
    nbr = _i32(nbr)
    center = _i32(center)
    meds = _i32(meds)
    isneighbor = _f64(isneighbor)
    nneigh, npix = isneighbor.shape
    tmp = np.zeros_like(t)
    lib.cpk_diffuse(
        int(n_iter), meds.size, npix, nneigh, _addr(t), _addr(tmp), _addr(nbr),
        _addr(center), _addr(meds), _addr(isneighbor),
    )
    return t


def flow_grads(t, nbr) -> np.ndarray:
    """`(2, npix)` neighbour differences of `t`, as `_extend_centers_gpu` returns."""
    t = _f64(t)
    nbr = _i32(nbr)
    nneigh, npix = nbr.shape
    out = np.empty((2, npix), dtype=np.float64)
    lib.cpk_flow_grads(npix, _addr(t), _addr(nbr), _addr(out))
    return out


def normalize_flows(mu) -> np.ndarray:
    """In-place unit-length normalisation of a `(2, npix)` flow field."""
    mu = _f64(mu)
    lib.cpk_normalize_flows(mu.shape[1], _addr(mu))
    return mu


def flow_error(dP_masks, dP_net) -> np.ndarray:
    """Per-pixel squared flow discrepancy, as `flow_error` computes it."""
    dpm = _f64(dP_masks)
    dpn = _f64(dP_net)
    shape = dpm.shape[1:]
    dpm = dpm.reshape(2, -1)
    dpn = dpn.reshape(2, -1)
    out = np.empty(dpm.shape[1], dtype=np.float64)
    lib.cpk_flow_error(dpm.shape[1], _addr(dpm), _addr(dpn), _addr(out))
    return out.reshape(shape)


def bilinear(field, gx, gy) -> np.ndarray:
    """Bilinear sample of a `(ly, lx)` field at normalised coordinates."""
    field = _f64(field)
    gx = _f64(gx).reshape(-1)
    gy = _f64(gy).reshape(-1)
    out = np.empty(gx.size, dtype=np.float64)
    ly, lx = field.shape
    lib.cpk_bilinear(gx.size, ly, lx, _addr(field), _addr(gx), _addr(gy), _addr(out))
    return out


def follow_flows(dP, inds, niter=200) -> np.ndarray:
    """Euler integration of a 2D flow field, as `follow_flows` does it.

    `dP` is `(2, ly, lx)`, `inds` is `(2, npts)` of starting pixel coordinates
    (row 0 y, row 1 x).  Returns `(npts, 2)`, row 0 y and row 1 x.
    """
    dp = _f64(dP)
    inds = _f64(inds)
    if dp.ndim != 3 or dp.shape[0] != 2:
        raise ValueError("'dP' must have shape (2, ly, lx)")
    if inds.ndim != 2 or inds.shape[0] != 2:
        raise ValueError("'inds' must have shape (2, npts)")
    ly, lx = dp.shape[1:]
    npts = inds.shape[1]
    scratch = np.empty(2 * npts, dtype=np.float64)
    tmp = np.empty(2 * npts, dtype=np.float64)
    res = np.empty(2 * npts, dtype=np.float64)
    lib.cpk_follow_flows(
        int(niter), npts, ly, lx, _addr(dp), _addr(inds), _addr(scratch),
        _addr(tmp), _addr(res),
    )
    return res.reshape(npts, 2)
