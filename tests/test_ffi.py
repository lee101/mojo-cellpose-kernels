"""FFI contract tests.

Buffers cross the C ABI as 64-bit addresses.  A short `argtypes` list is
silent: ctypes converts the surplus argument with its default `c_int` rule,
truncating the address to 32 bits, and nothing goes wrong until a buffer lands
above 4 GiB.  These tests pin the arity and then run the kernels on mappings
placed at fixed addresses above 4 GiB.
"""

import ctypes

import numpy as np
import pytest

from mojo_cellpose_kernels import _lib

_HIGH = 0x2_0000_0000  # 8 GiB
_STRIDE = 0x0800_0000  # 128 MiB between successive mappings

_PROT_READ, _PROT_WRITE = 1, 2
_MAP_PRIVATE, _MAP_ANONYMOUS, _MAP_FIXED = 2, 0x20, 0x10


def _allocator():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    state = {"n": 0}

    def alloc(n):
        want = _HIGH + state["n"] * _STRIDE
        state["n"] += 1
        got = libc.mmap(
            ctypes.c_void_p(want),
            ctypes.c_size_t(n * 8),
            _PROT_READ | _PROT_WRITE,
            _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_FIXED,
            -1,
            0,
        )
        if got is None or got == ctypes.c_void_p(-1).value:
            pytest.skip("could not map a high address on this system")
        assert got == want, "mmap did not honour the address hint"
        assert got >= 2**32
        return np.ctypeslib.as_array(
            ctypes.cast(ctypes.c_void_p(got), ctypes.POINTER(ctypes.c_double)),
            shape=(n,),
        )

    return alloc


high = pytest.fixture(scope="module")(_allocator)


def test_every_kernel_declares_its_full_arity():
    for name, arity in _lib._ARITY.items():
        assert len(getattr(_lib.lib, name).argtypes) == arity, name


def test_addresses_are_declared_64_bit():
    for name in _lib._ARITY:
        for t in getattr(_lib.lib, name).argtypes:
            assert t in (ctypes.c_int64, ctypes.c_double), name


def test_bilinear_works_above_four_gib(high):
    rng = np.random.default_rng(0)
    n = 1 << 16
    field = rng.standard_normal((256, 256))
    gx = high(n)
    gy = high(n)
    out = high(n)
    gx[:]
    gy[:] = rng.uniform(-1.0, 1.0, n)
    _lib.lib.cpk_bilinear(
        n, 256, 256, field.ctypes.data, gx.ctypes.data, gy.ctypes.data,
            out.ctypes.data,
    )
    # Straight scalar transcription, so any address truncation shows up.
    want = np.array(
        [_lib_ref_bilinear(field, gx[i], gy[i]) for i in range(n)]
    )
    # Mojo emits FMA, so the scalar transcription agrees to a few ULP rather
    # than bit for bit; the point of the test is the address, not the last bit.
    np.testing.assert_allclose(out, want, rtol=1e-14, atol=0.0)


def _lib_ref_bilinear(field, gx, gy):
    ly, lx = field.shape
    ux = ((gx + 1.0) * lx - 1.0) * 0.5
    uy = ((gy + 1.0) * ly - 1.0) * 0.5
    ix0, iy0 = int(np.floor(ux)), int(np.floor(uy))
    fx, fy = ux - ix0, uy - iy0

    def at(iy, ix):
        if 0 <= iy < ly and 0 <= ix < lx:
            return field[iy, ix]
        return 0.0

    return (
        at(iy0, ix0) * (1 - fy) * (1 - fx)
        + at(iy0, ix0 + 1) * (1 - fy) * fx
        + at(iy0 + 1, ix0) * fy * (1 - fx)
        + at(iy0 + 1, ix0 + 1) * fy * fx
    )


def test_flow_error_works_above_four_gib(high):
    rng = np.random.default_rng(1)
    n = 1 << 14
    a = rng.standard_normal(2 * n)
    b = rng.standard_normal(2 * n)
    got = high(n)
    _lib.lib.cpk_flow_error(n, a.ctypes.data, b.ctypes.data, got.ctypes.data)
    ey = a[:n] - b[:n] / 5.0
    ex = a[n:] - b[n:] / 5.0
    np.testing.assert_allclose(got, ey * ey + ex * ex, rtol=1e-14, atol=0.0)


def test_diffuse_works_above_four_gib(high):
    """The diffusion reads and writes every one of its buffers, so a truncated
    address shows up as wrong numbers rather than as a crash."""
    npix = 9 * 64
    size = npix + 64
    nbr = np.zeros((9, npix), dtype=np.int32)
    for p in range(npix):
        for j in range(9):
            nbr[j, p] = (p + j) % size
    center = (np.arange(npix, dtype=np.int32) + 32) % size
    meds = np.array([32, 96, 160], dtype=np.int32)
    isn = np.ones((9, npix))
    isn[3, ::5] = 0.0

    t = high(size)
    tmp = high(size)
    _lib.lib.cpk_diffuse(
        7, 3, npix, 9, t.ctypes.data, tmp.ctypes.data, nbr.ctypes.data,
        center.ctypes.data, meds.ctypes.data, isn.ctypes.data,
    )

    src = np.zeros(size)
    dst = np.zeros(size)
    for _ in range(7):
        for m in range(3):
            src[meds[m]] += 1.0
        for p in range(npix):
            acc = 0.0
            for j in range(9):
                if isn[j, p] != 0.0:
                    acc += src[nbr[j, p]]
            dst[center[p]] = acc / 9.0
        src, dst = dst, src
    np.testing.assert_array_equal(t, src)
