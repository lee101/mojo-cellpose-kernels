"""Correctness-gated benchmark for mojo-cellpose-kernels.

Every case checks agreement with the torch transcription of cellpose's own
routines (or with the closed form) before timing, so a regression in the Mojo
kernels shows up as a correctness failure rather than a suspiciously good
number.

The torch path is the fair baseline: cellpose runs these loops in float32 on
the CPU for a 2D image, and `tests/reference.py` is the same code.  The Mojo
kernels are float64, which is a genuine difference and is stated rather than
hidden.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

import mojo_cellpose_kernels as mcp  # noqa: E402
import reference as ref  # noqa: E402

torch.set_num_threads(1)


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_masks(ly, lx, ncell=40):
    masks = np.zeros((ly, lx), dtype=int)
    rng = np.random.default_rng(0)
    for i in range(ncell):
        y = rng.integers(0, ly - 12)
        x = rng.integers(0, lx - 12)
        masks[y : y + 12, x : x + 12] = i + 1
    return masks


def bench_masks_to_flows(ly=1024, lx=1024, n_iter=100):
    """The diffusion: `n_iter` rounds of a nine-way neighbour gather, averaged."""
    masks = make_masks(ly, lx)
    flat, isn, meds, exts, _ = mcp.build_graph(masks)
    npix = flat.shape[1]
    size = (ly + 2) * (lx + 2)
    field = np.zeros(size)
    t = field.copy()

    def torch_diffuse():
        return ref.extend_centers_gpu(
            torch.from_numpy(np.stack([flat // (lx + 2), flat % (lx + 2)])),
            torch.from_numpy(
                np.stack([meds // (lx + 2), meds % (lx + 2)], axis=1)
            ),
            torch.from_numpy(isn),
            (ly + 2, lx + 2),
            n_iter=n_iter,
        )

    want = torch_diffuse().numpy()
    got = mcp.flow_grads(
        mcp.diffuse(t.copy(), flat, flat[0], meds, isn, n_iter), flat
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)

    return (
        f"diffusion {ly}x{lx} npix={npix} n={n_iter}",
        _time(torch_diffuse),
        _time(
            lambda: mcp.flow_grads(
                mcp.diffuse(t.copy(), flat, flat[0], meds, isn, n_iter), flat
            )
        ),
    )


def bench_flow_grads(ly=1024, lx=1024, n_iter=100):
    masks = make_masks(ly, lx)
    flat, isn, meds, exts, _ = mcp.build_graph(masks)
    npix = flat.shape[1]
    t = np.random.default_rng(1).random((ly + 2) * (lx + 2))
    out = np.empty((2, npix))
    t_t = torch.from_numpy(t)
    flat_t = torch.from_numpy(flat.astype(np.int64))

    def torch_grads():
        g = t_t[flat_t[[2, 1, 4, 3]]]
        return torch.stack((g[0] - g[1], g[2] - g[3]), dim=0)

    np.testing.assert_allclose(
        mcp.flow_grads(t, flat), torch_grads().numpy(), rtol=1e-12, atol=1e-12
    )
    return (
        f"flow_grads npix={npix}",
        _time(torch_grads),
        _time(lambda: mcp.flow_grads(t, flat)),
    )


def bench_follow_flows(ly=512, lx=512, npts=1 << 20, niter=200):
    """Pixel advection: `npts` points integrated `niter` times with a clamped
    bilinear sample of the flow field at each step."""
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:ly, 0:lx]
    dy, dx = ly / 2 - yy, lx / 2 - xx
    norm = np.hypot(dy, dx)
    safe = np.where(norm == 0, 1.0, norm)
    dP = np.stack([0.3 * dy / safe, 0.3 * dx / safe])
    inds = np.stack(
        [rng.integers(0, ly, npts), rng.integers(0, lx, npts)]
    ).astype(np.float64)
    dP32 = dP.astype(np.float32)

    def torch_flows():
        return ref.steps_interp(dP32, inds, niter)

    got = mcp.follow_flows(dP, inds, niter)
    want = np.asarray(torch_flows()).T
    np.testing.assert_allclose(got, want, rtol=0.0, atol=2e-2)

    return (
        f"follow_flows npts={npts} n={niter}",
        _time(torch_flows, 1),
        _time(lambda: mcp.follow_flows(dP, inds, niter), 1),
    )


def bench_flow_error(npix=1 << 22):
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, npix))
    b = rng.standard_normal((2, npix))
    a_t, b_t = torch.from_numpy(a), torch.from_numpy(b)

    def torch_err():
        return ((a_t - b_t / 5.0) ** 2).sum(dim=0)

    np.testing.assert_allclose(
        mcp.flow_error(a, b), torch_err().numpy(), rtol=1e-5, atol=1e-6
    )
    return (
        f"flow_error n={npix}",
        _time(torch_err),
        _time(lambda: mcp.flow_error(a, b)),
    )


def main():
    print(f"{'case':<40}{'cellpose (torch)':>20}{'mojo':>16}{'ratio':>9}")
    print("-" * 85)
    for fn in (
        bench_masks_to_flows,
        bench_flow_grads,
        bench_follow_flows,
        bench_flow_error,
    ):
        label, ref_t, got = fn()
        ratio = ref_t / got if got else float("nan")
        print(
            f"{label:<40}{ref_t * 1e3:>17.1f}ms{got * 1e3:>13.1f}ms{ratio:>8.2f}x"
        )


if __name__ == "__main__":
    main()
