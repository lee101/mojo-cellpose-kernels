"""Cellpose's flow dynamics, with the per-pixel loops in Mojo.

The graph construction here (padding, the 9-neighbour table, the same-object
predicate, the cell centres) is bookkeeping and stays in NumPy.  Every loop over
iterations or over pixels is in `src/kernels.mojo`:

    masks_to_flows   -> cpk_diffuse + cpk_flow_grads + cpk_normalize_flows
    follow_flows     -> cpk_follow_flows
    flow_error       -> cpk_flow_error
    bilinear         -> cpk_bilinear

The graph helpers are transcribed from `cellpose/dynamics.py` 4.2.1
(`center_of_mass`, `get_centers`, and the setup half of
`masks_to_flows_gpu`).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects

from . import _lib

__all__ = [
    "center_of_mass",
    "get_centers",
    "build_graph",
    "masks_to_flows",
    "follow_flows",
    "bilinear",
    "flow_error",
    "flow_error_per_mask",
]

# The 9 offsets of the 3x3 neighbourhood, in the order cellpose indexes them.
# Positions 2/1 give the y difference and 4/3 the x difference, which is why
# `flow_grads` reads neighbours[[2, 1, 4, 3]].
_YXI = ([0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1])


def center_of_mass(mask: np.ndarray):
    """Centre of a labelled mask, as `cellpose.dynamics.center_of_mass`."""
    yi, xi = np.nonzero(mask)
    ymean = int(np.round(yi.sum() / len(yi)))
    xmean = int(np.round(xi.sum() / len(xi)))
    if not ((yi == ymean) * (xi == xmean)).sum():
        # centre is the closest point in the mask to the (ymean, xmean) mean
        imin = ((xi - xmean) ** 2 + (yi - ymean) ** 2).argmin()
        ymean = yi[imin]
        xmean = xi[imin]
    return ymean, xmean


def get_centers(masks: np.ndarray, slices):
    """Cell centres in image coordinates, plus the per-object extent."""
    centers = [
        center_of_mass(masks[slices[i]] == (i + 1)) for i in range(len(slices))
    ]
    centers = np.array(
        [
            [
                centers[i][0] + slices[i][0].start,
                centers[i][1] + slices[i][1].start,
            ]
            for i in range(len(slices))
        ]
    )
    exts = np.array(
        [
            (slc[0].stop - slc[0].start) + (slc[1].stop - slc[1].start) + 2
            for slc in slices
        ]
    )
    return centers, exts


def build_graph(masks: np.ndarray):
    """The 9-neighbour graph over the non-zero pixels of a padded label image.

    Returns `(flat_neighbors, isneighbor, meds, exts, slices)` where
    `flat_neighbors` is `(9, npix)` of flat indices into the padded image,
    `isneighbor` is `(9, npix)` of 0/1, and `meds` holds the flat index of each
    cell centre in the padded image.
    """
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise ValueError("only 2D masks are supported")
    ly0, lx0 = masks.shape
    ly, lx = ly0 + 2, lx0 + 2

    padded = np.zeros((ly, lx), dtype=np.int64)
    padded[1:-1, 1:-1] = masks

    y, x = np.nonzero(padded)
    npix = y.size
    neighbors = np.empty((2, 9, npix), dtype=np.int64)
    for i in range(9):
        neighbors[0, i] = y + _YXI[0][i]
        neighbors[1, i] = x + _YXI[1][i]

    isneighbor = np.ones((9, npix), dtype=np.float64)
    m0 = padded[neighbors[0, 0], neighbors[1, 0]]
    for i in range(1, 9):
        isneighbor[i] = (padded[neighbors[0, i], neighbors[1, i]] == m0).astype(
            np.float64
        )

    flat_neighbors = (neighbors[0] * lx + neighbors[1]).astype(np.int32)

    slices = find_objects(masks.astype(np.int32))
    if len(slices) == 0:
        raise ValueError("the label image has no objects")
    centers, exts = get_centers(masks, slices)
    # +1 on both axes for the padding, then flattened into the padded image.
    meds = (
        (np.round(centers[:, 0]).astype(np.int64) + 1) * lx
        + (np.round(centers[:, 1]).astype(np.int64) + 1)
    ).astype(np.int32)

    return flat_neighbors, isneighbor, meds, exts, slices


def masks_to_flows(masks: np.ndarray, niter: int | None = None):
    """Flows pointing from every pixel to its cell centre.

    Mirrors `cellpose.dynamics.masks_to_flows_gpu` on the 2D CPU path: diffuse
    from the centres over the 9-neighbour graph, take the neighbour differences
    as the flow, normalise it to unit length, and scatter it back onto the
    original image.  Returns `(mu, slices)`.
    """
    masks = np.asarray(masks)
    ly0, lx0 = masks.shape
    if masks.max() <= 0:
        return np.zeros((2, ly0, lx0)), None

    flat_neighbors, isneighbor, meds, exts, slices = build_graph(masks)
    npix = flat_neighbors.shape[1]
    n_iter = int(2 * exts.max()) if niter is None else int(niter)

    # The diffusion runs over the padded image, so the field is (ly0+2)*(lx0+2).
    t = np.zeros((ly0 + 2) * (lx0 + 2), dtype=np.float64)
    _lib.diffuse(
        t, flat_neighbors, flat_neighbors[0], meds, isneighbor, n_iter
    )
    mu = _lib.flow_grads(t, flat_neighbors)
    _lib.normalize_flows(mu)

    mu0 = np.zeros((2, ly0, lx0))
    lx = lx0 + 2
    y, x = np.nonzero(np.pad(masks, 1))
    mu0[:, y - 1, x - 1] = mu
    return mu0, slices


def follow_flows(dP: np.ndarray, inds: np.ndarray, niter: int = 200) -> np.ndarray:
    """Euler integration of a 2D flow field; see `cellpose.dynamics.follow_flows`."""
    return _lib.follow_flows(dP, inds, niter)


def bilinear(field: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2D field at normalised coordinates."""
    return _lib.bilinear(field, gx, gy)


def flow_error(dP_masks: np.ndarray, dP_net: np.ndarray) -> np.ndarray:
    """Per-pixel squared flow discrepancy; see `cellpose.dynamics.flow_error`."""
    return _lib.flow_error(dP_masks, dP_net)


def flow_error_per_mask(
    masks: np.ndarray, dP_masks: np.ndarray, dP_net: np.ndarray
) -> np.ndarray:
    """Mean flow error inside each labelled object.

    This is the second half of `cellpose.dynamics.flow_error`: the per-pixel
    reduction is a Mojo kernel, the per-object mean is a short NumPy gather
    because it needs the connected-component slices.
    """
    masks = np.asarray(masks)
    dP_masks = np.asarray(dP_masks, dtype=np.float64)
    dP_net = np.asarray(dP_net, dtype=np.float64)
    err = _lib.flow_error(dP_masks, dP_net).reshape(masks.shape)
    slices = find_objects(masks.astype(np.int32))
    return np.array(
        [
            err[slc[0], slc[1]][masks[slc[0], slc[1]] == (j + 1)].mean()
            for j, slc in enumerate(slices)
        ]
    )
