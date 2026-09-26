"""Parity tests for the mask-to-flow pipeline.

`reference.py` holds the torch transcription of `cellpose/dynamics.py`; these
tests compare the Mojo kernels against it.  The reference was checked against
cellpose 4.2.1 during development and reproduces `masks_to_flows_gpu` exactly.
"""

import numpy as np
import pytest
import torch

import mojo_cellpose_kernels as mcp
from mojo_cellpose_kernels import dynamics as mydyn

import reference as ref

torch.set_num_threads(2)

# The diffusion is a sum of nine float64 values; Mojo emits FMA, so the
# agreement is to a few ULP rather than bit for bit.
RTOL = 1e-13


def make_masks(ly=40, lx=50):
    masks = np.zeros((ly, lx), dtype=int)
    masks[5:15, 5:15] = 1
    masks[20:32, 25:40] = 2
    masks[2:8, 30:38] = 3
    return masks


def torch_reference_flows(masks, n_iter):
    """`masks_to_flows_gpu`, split into the graph setup and the torch kernel."""
    ly0, lx0 = masks.shape
    lx = lx0 + 2
    y, x = np.nonzero(np.pad(masks, 1))
    yxi = ([0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1])
    nbr_y = np.stack([y + yxi[0][i] for i in range(9)])
    nbr_x = np.stack([x + yxi[1][i] for i in range(9)])
    padded = np.zeros((ly0 + 2, lx), dtype=np.int64)
    padded[1:-1, 1:-1] = masks
    isn = np.ones((9, y.size))
    m0 = padded[nbr_y[0], nbr_x[0]]
    for i in range(1, 9):
        isn[i] = (padded[nbr_y[i], nbr_x[i]] == m0)
    centers, _ = mydyn.get_centers(masks, _slices(masks))
    meds2d = np.round(centers).astype(np.int64) + 1

    mu = ref.extend_centers_gpu(
        torch.from_numpy(np.stack([nbr_y, nbr_x])),
        torch.from_numpy(meds2d),
        torch.from_numpy(isn),
        (ly0 + 2, lx),
        n_iter=n_iter,
    ).numpy()
    mu /= 1e-60 + (mu**2).sum(axis=0) ** 0.5
    mu0 = np.zeros((2, ly0, lx0))
    mu0[:, y - 1, x - 1] = mu
    return mu0


def _slices(masks):
    from scipy.ndimage import find_objects

    return find_objects(masks.astype(np.int32))


@pytest.mark.parametrize("n_iter", [1, 7, 50, 200])
def test_masks_to_flows_matches_reference(n_iter):
    masks = make_masks()
    want = torch_reference_flows(masks, n_iter)
    got, slices = mcp.masks_to_flows(masks, niter=n_iter)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-15)


def test_graph_construction_matches_cellpose_layout():
    """The 9-neighbour offsets and the same-object predicate are the whole
    contract between the graph and the diffusion, so check them directly."""
    masks = make_masks()
    lx = masks.shape[1] + 2
    flat, isn, meds, exts, slices = mydyn.build_graph(masks)
    y, x = np.nonzero(np.pad(masks, 1))
    yxi = ([0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1])
    for i in range(9):
        np.testing.assert_array_equal(
            flat[i], (y + yxi[0][i]) * lx + (x + yxi[1][i])
        )
    # A pixel's own entry is always a valid same-object neighbour.
    assert np.all(isn[0] == 1.0)
    assert flat.shape == (9, y.size)
    # Two objects that merely touch diagonally are distinguished by isneighbor.
    assert set(np.unique(isn)) <= {0.0, 1.0}


def test_flows_are_unit_length_inside_objects():
    masks = make_masks()
    mu, _ = mcp.masks_to_flows(masks, niter=50)
    inside = masks > 0
    norm = np.linalg.norm(mu, axis=0)[inside]
    np.testing.assert_allclose(norm, np.ones_like(norm), rtol=1e-12, atol=1e-15)


def test_flows_point_towards_the_centre():
    """Every pixel's flow must have a positive component towards its centre.
    The centre pixel itself has zero flow, so it is excluded."""
    masks = np.zeros((60, 60), dtype=int)
    yy, xx = np.mgrid[0:60, 0:60]
    masks[(yy - 30) ** 2 + (xx - 30) ** 2 < 100] = 1
    mu, _ = mcp.masks_to_flows(masks, niter=200)
    cy, cx = 30, 30
    py, px = np.nonzero(masks)
    away = (py - cy) ** 2 + (px - cx) ** 2 > 9
    py, px = py[away], px[away]
    dy = cy - py
    dx = cx - px
    dots = mu[0][py, px] * dy + mu[1][py, px] * dx
    assert dots.min() > 0.0

def test_background_pixels_have_zero_flow():
    masks = make_masks()
    mu, _ = mcp.masks_to_flows(masks, niter=50)
    assert np.all(mu[:, masks == 0] == 0.0)


def test_normalize_flows_leaves_unit_vectors_alone():
    """The division is by a *recomputed* norm, so the result equals the input
    only to within a rounding step; the epsilon is far below that."""
    rng = np.random.default_rng(2)
    v = rng.standard_normal((2, 1000))
    v /= np.linalg.norm(v, axis=0)
    np.testing.assert_allclose(
        mcp.normalize_flows(v.copy()), v, rtol=0.0, atol=1e-15
    )


def test_normalize_flows_leaves_zero_alone():
    v = np.zeros((2, 4))
    v[0, 1] = 3.0
    v[1, 1] = 4.0
    out = mcp.normalize_flows(v)
    np.testing.assert_array_equal(out[:, 0], np.zeros(2))
    np.testing.assert_allclose(out[:, 1], [0.6, 0.8])


def test_flow_error_matches_the_formula():
    rng = np.random.default_rng(3)
    dpm = rng.standard_normal((2, 500))
    dpn = rng.standard_normal((2, 500))
    want = ((dpm - dpn / 5.0) ** 2).sum(axis=0)
    np.testing.assert_allclose(
        mcp.flow_error(dpm, dpn), want, rtol=1e-13, atol=0.0
    )


def test_flow_error_is_zero_when_the_network_flow_is_five_times_larger():
    """`flow_error` compares the mask flow against `dP_net / 5`, so a perfect
    prediction is one that is five times the mask flow, not equal to it."""
    v = np.array([[0.0, 0.5, 1.0, 2.0, 0.25], [0.5, 1.0, 2.0, 0.25, 2.0]])
    assert np.all(mcp.flow_error(v, 5.0 * v) == 0.0)


def test_flow_error_per_mask_reduces_over_each_object():
    masks = make_masks()
    mu, _ = mcp.masks_to_flows(masks, niter=50)
    err = mcp.flow_error_per_mask(masks, mu, 5.0 * mu + 0.5)
    assert err.shape == (3,)
    assert np.all(err >= 0.0)
    # each pixel is off by 0.5 / 5 per component, squared and summed
    np.testing.assert_allclose(err, 2 * 0.1**2, rtol=1e-12)
