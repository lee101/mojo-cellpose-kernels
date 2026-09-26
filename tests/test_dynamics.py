"""Parity tests for the pixel-advection kernels.

`steps_interp` integrates in float32 upstream and the dynamics are chaotic: a
large flow field over many iterations diverges from a float64 kernel for
reasons that have nothing to do with correctness.  The parity cases therefore
use a gentle field, where the two agree to float32 precision, and the analytic
cases pin the behaviour that chaos would otherwise hide.
"""

import numpy as np
import pytest
import torch

import mojo_cellpose_kernels as mcp

import reference as ref

torch.set_num_threads(2)

# The reference runs in float32, so the floor is float32 epsilon, not float64.
RTOL = 0.0
ATOL = 2e-5


def sample_points(ly, lx, seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    inds = np.stack(np.nonzero(rng.random((ly, lx)) > frac))
    return inds


def test_bilinear_matches_grid_sample():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((37, 53))
    gy, gx = np.meshgrid(
        np.linspace(-1.2, 1.2, 41), np.linspace(-1.1, 1.1, 29), indexing="ij"
    )
    gx, gy = gx.ravel(), gy.ravel()
    want = ref.grid_sample_2d(field, gx, gy)
    got = mcp.bilinear(field, gx, gy)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)

def test_bilinear_is_exact_on_a_constant_field():
    """Where all four corners are on the field, a constant field samples to
    exactly its value; far outside, zeros padding leaves exactly zero."""
    field = np.full((9, 11), 3.25)
    rng = np.random.default_rng(2)
    gx = rng.uniform(-0.8, 0.8, 500)
    gy = rng.uniform(-0.8, 0.8, 500)
    assert np.allclose(mcp.bilinear(field, gx, gy), 3.25, rtol=0.0, atol=1e-12)
    far = np.full(50, 5.0)
    assert np.all(mcp.bilinear(field, far, -far) == 0.0)


@pytest.mark.parametrize("niter", [0, 1, 5, 20])
def test_follow_flows_matches_steps_interp(niter):
    """A smooth radial field.  A random field would be a poor parity probe:
    the dynamics are chaotic, so float32 upstream and float64 here separate
    geometrically however correct both are.  A converging field keeps the
    trajectories in step, and the tolerance stays at the float32 floor."""
    ly, lx = 41, 47
    cy, cx = 20.0, 23.0
    yy, xx = np.mgrid[0:ly, 0:lx]
    dy, dx = cy - yy, cx - xx
    norm = np.hypot(dy, dx)
    amp = 0.3
    dP = np.stack(
        [amp * dy / np.where(norm == 0, 1.0, norm), amp * dx / np.where(norm == 0, 1.0, norm)]
    ).astype(np.float32)
    inds = sample_points(ly, lx, seed=4, frac=0.0)
    want = np.asarray(ref.steps_interp(dP, inds, niter)).T
    got = mcp.follow_flows(dP.astype(np.float64), inds, niter)
    assert got.shape == (inds.shape[1], 2)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def test_follow_flows_on_a_random_field_agrees_over_one_step():
    """Even a chaotic field agrees after a single step, which is where a wrong
    sampling convention would show up immediately."""
    rng = np.random.default_rng(3)
    ly, lx = 41, 47
    dP = (rng.standard_normal((2, ly, lx)) * 0.5).astype(np.float32)
    inds = sample_points(ly, lx, seed=4)
    want = np.asarray(ref.steps_interp(dP, inds, 1)).T
    got = mcp.follow_flows(dP.astype(np.float64), inds, 1)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)




def test_zero_flow_leaves_pixels_where_they_are():
    ly, lx = 30, 35
    dP = np.zeros((2, ly, lx))
    inds = sample_points(ly, lx, seed=5, frac=0.0)
    got = mcp.follow_flows(dP, inds, 200)
    np.testing.assert_allclose(got[:, 0], inds[0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], inds[1], rtol=0.0, atol=1e-9)


def test_constant_flow_translates_by_the_right_amount():
    """A constant field is sampled exactly (all four corners agree), so after
    `niter` steps an interior pixel must have moved by niter * v pixels.  This
    pins the 2 / (size - 1) scale factor `steps_interp` applies, which a sign
    error or a missing factor of two would break.  Pixels near an edge are
    excluded because the [-1, 1] clamp truncates their travel."""
    ly, lx, v, niter = 30, 30, 0.25, 8
    dP = np.stack([np.full((ly, lx), v), np.full((ly, lx), -v)])
    yy, xx = np.mgrid[0:ly, 0:lx]
    inside = (yy >= 8) & (yy <= ly - 9) & (xx >= 8) & (xx <= lx - 9)
    inds = np.stack(np.nonzero(inside)).astype(np.float64)
    got = mcp.follow_flows(dP, inds, niter)
    np.testing.assert_allclose(
        got[:, 0], inds[0] + niter * v, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        got[:, 1], inds[1] - niter * v, rtol=0.0, atol=1e-9
    )


def test_positions_stay_inside_the_image():
    """`steps_interp` clamps to [-1, 1] every step, which is the only thing
    keeping the advected pixels from leaving the field."""
    rng = np.random.default_rng(7)
    ly, lx = 25, 31
    dP = rng.standard_normal((2, ly, lx)) * 0.7
    inds = sample_points(ly, lx, seed=8, frac=0.0)
    got = mcp.follow_flows(dP, inds, 200)
    assert got[:, 0].min() >= -1e-9
    assert got[:, 0].max() <= ly - 1 + 1e-9
    assert got[:, 1].min() >= -1e-9
    assert got[:, 1].max() <= lx - 1 + 1e-9


def test_follow_flows_clamps_at_the_image_edge():
    """A constant rightward flow of one pixel per step must move every pixel
    by exactly four columns, with the one that would leave the field stopping
    at the last column."""
    ly, lx = 21, 21
    dP = np.zeros((2, ly, lx))
    dP[1] = 1.0
    inds = np.array([[10, 10, 10, 10], [2, 4, 14, 16]], dtype=np.float64)
    got = mcp.follow_flows(dP, inds, 4)
    np.testing.assert_allclose(got[:, 0], inds[0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        got[:, 1], [6.0, 8.0, 18.0, 20.0], rtol=0.0, atol=1e-9
    )
    # Push one pixel far enough that the clamp, not the flow, decides.
    got = mcp.follow_flows(dP, np.array([[10.0], [19.0]]), 5)
    assert got[0, 1] == pytest.approx(lx - 1.0, abs=1e-9)


def test_follow_flows_rejects_a_3d_field():
    with pytest.raises(ValueError):
        mcp.follow_flows(np.zeros((3, 8, 8)), np.zeros((2, 4)), 5)
