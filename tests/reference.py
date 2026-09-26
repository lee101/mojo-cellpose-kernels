"""Torch reference implementations transcribed from `cellpose/dynamics.py`.

Cellpose is not installed in the shared test venv, and the toolchain directory
must not be modified, so the parity tests compare the Mojo kernels against the
routines below, transcribed from cellpose 4.2.1 with the torch calls left as
they are.  They were checked against the real package during development; see
the README.

Each function keeps cellpose's own names and argument order so the transcription
can be diffed against upstream.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_YXI = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]


def extend_centers_gpu(neighbors, meds, isneighbor, shape, n_iter=200):
    """`cellpose.dynamics._extend_centers_gpu`, verbatim apart from `device`."""
    dtype = (
        torch.float32
        if np.prod(shape) > 4e7
        else torch.float64
    )
    T_flat = torch.zeros(np.prod(shape), dtype=dtype)

    ndim = len(shape)
    Ly, Lx = shape[-2:]
    if ndim == 2:
        Ly, Lx = shape
        flat_neighbors = (neighbors[0] * Lx + neighbors[1]).long()
        flat_meds = (meds[:, 0] * Lx + meds[:, 1]).long()
    else:
        flat_neighbors = (
            neighbors[0] * (Ly * Lx) + neighbors[1] * Lx + neighbors[2]
        ).long()
        flat_meds = (
            meds[:, 0] * (Ly * Lx) + meds[:, 1] * Lx + meds[:, 2]
        ).long()

    flat_center = flat_neighbors[0]
    nneigh = flat_neighbors.shape[0]
    for _ in range(n_iter):
        T_flat[flat_meds] += 1
        Tneigh = T_flat[flat_neighbors]
        T_flat[flat_center] = (Tneigh * isneighbor).sum(dim=0) / nneigh

    grads = T_flat[flat_neighbors[[2, 1, 4, 3]]]
    dy = grads[0] - grads[1]
    dx = grads[2] - grads[3]
    return torch.stack((dy, dx), axis=0)


def steps_interp(dP, inds, niter):
    """`cellpose.dynamics.steps_interp`, 2D path only."""
    shape = dP.shape[1:]
    ndim = len(shape)

    pt = torch.zeros((*[1] * ndim, len(inds[0]), ndim), dtype=torch.float32)
    im = torch.zeros((1, ndim, *shape), dtype=torch.float32)
    for n in range(ndim):
        pt[0, 0, :, ndim - n - 1] = torch.from_numpy(np.asarray(inds[n])).to(
            torch.float32
        )
        im[0, ndim - n - 1] = torch.from_numpy(np.asarray(dP[n])).to(
            torch.float32
        )
    shape = np.array(shape)[::-1].astype("float") - 1

    for k in range(ndim):
        im[:, k] *= 2.0 / shape[k]
        pt[..., k] /= shape[k]
    pt *= 2
    pt -= 1

    for _ in range(niter):
        dPt = F.grid_sample(im, pt, align_corners=False)
        for k in range(ndim):
            pt[..., k] += dPt[:, k]
            torch.clamp_(pt[..., k], -1.0, 1.0)

    pt += 1
    pt *= 0.5
    for k in range(ndim):
        pt[..., k] *= shape[k]

    pt = pt[..., [1, 0]].squeeze()
    pt = pt.unsqueeze(0) if pt.ndim == 1 else pt
    return pt.T


def grid_sample_2d(field, gx, gy):
    """One bilinear `grid_sample` of a single 2D field, for the parity test."""
    im = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))[None, None]
    grid = np.stack(
        [np.asarray(gx, dtype=np.float32), np.asarray(gy, dtype=np.float32)],
        axis=-1,
    )[None, None]
    out = F.grid_sample(
        im, torch.from_numpy(grid), align_corners=False
    )
    return out[0, 0, 0].numpy()
