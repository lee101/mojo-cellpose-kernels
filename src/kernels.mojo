"""Per-pixel dynamics kernels from Cellpose's `cellpose/dynamics.py`.

Cellpose turns a predicted flow field into instance masks by integrating the
flow: every pixel is advected along `dP` until it lands on a cell centre, and
pixels that land in the same place belong to the same cell.  Three loops do all
of the work.

1. `_extend_centers_gpu` diffuses a value out from each cell centre over the
   9-neighbour graph, once per iteration, and the *difference* of neighbouring
   values is the flow.  That is `cpk_diffuse` + `cpk_flow_grads`.
2. `steps_interp` advects the pixels with bilinear sampling of the flow field
   and a clamp to the image bounds.  That is `cpk_follow_flows`.
3. `flow_error` scores the result.  That is `cpk_flow_error`.

The flow normalisation in `masks_to_flows_gpu` is `cpk_normalize_flows`, and
`cpk_bilinear` exposes the sampling step on its own.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

from std.math import floor, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

# `flow_error` divides the network flow by 5 before scoring.
comptime FLOW_SCALE: Float64 = 5.0
comptime FLOW_EPS: Float64 = 1e-60


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def _clamp1(v: Float64) -> Float64:
    if v < -1.0:
        return Float64(-1.0)
    if v > 1.0:
        return Float64(1.0)
    return v


@export("cpk_diffuse")
def cpk_diffuse(
    n_iter: Int,
    nmeds: Int,
    npix: Int,
    nneigh: Int,
    t_addr: Int,
    tmp_addr: Int,
    nbr_addr: Int,
    center_addr: Int,
    meds_addr: Int,
    isn_addr: Int,
) abi("C"):
    """Diffuse values from cell centres over the 9-neighbour graph.

    Mirrors `_extend_centers_gpu`.  `t` is a float64 work buffer over the
    padded image that the caller must zero; `tmp` is a second buffer of the
    same size.  `nbr` and `isn` are `(nneigh, npix)` row-major int32/float64:
    `nbr[j * npix + p]` is the flat index of the j-th neighbour of pixel `p`,
    and `isn[j * npix + p]` is nonzero when that neighbour belongs to the same
    object.  `center[p]` is the flat index of pixel `p` itself and `meds[m]` is
    the flat index of the m-th cell centre.

    The update is simultaneous, not in place: torch's
    `T_flat[flat_center] = (Tneigh * isneighbor).sum(dim=0)` reads the whole
    neighbour field before writing any of it, so two pixels that are each
    other's neighbour must not see each other's new value within one step.  A
    single in-place loop is Gauss-Seidel and drifts away from the reference, so
    the two buffers swap roles each iteration and the result is copied back
    into `t` at the end.
    """
    var t = fp(t_addr)
    var tmp = fp(tmp_addr)
    var nbr = ip(nbr_addr)
    var center = ip(center_addr)
    var meds = ip(meds_addr)
    var isn = fp(isn_addr)

    var src = t
    var dst = tmp
    for _ in range(n_iter):
        # T_flat[flat_meds] += 1
        for m in range(nmeds):
            src[unsafe_offset=meds[unsafe_offset=m]] += 1.0
        # T_flat[flat_center] = (Tneigh * isneighbor).sum(dim=0) / nneigh
        for p in range(npix):
            var acc = Float64(0.0)
            for j in range(nneigh):
                if isn[unsafe_offset=j * npix + p] != 0.0:
                    acc += src[unsafe_offset=nbr[unsafe_offset=j * npix + p]]
            dst[unsafe_offset=center[unsafe_offset=p]] = acc / Float64(nneigh)
        var swap = src
        src = dst
        dst = swap

    # `center` enumerates the non-zero pixels, so this restores every written
    # entry regardless of how many swaps happened.
    for p in range(npix):
        var c = center[unsafe_offset=p]
        t[unsafe_offset=c] = src[unsafe_offset=c]


@export("cpk_flow_grads")
def cpk_flow_grads(npix: Int, t_addr: Int, nbr_addr: Int, out_addr: Int) abi("C"):
    """Neighbour differences of the diffused field, i.e. the unnormalised flow.

    Mirrors the tail of `_extend_centers_gpu`:
        grads = T[neighbors[[2, 1, 4, 3]]];  dy = grads[0] - grads[1];
        dx = grads[2] - grads[3]
    `out` is `(2, npix)` row-major: row 0 is dy, row 1 is dx.
    """
    var t = fp(t_addr)
    var nbr = ip(nbr_addr)
    var out = fp(out_addr)
    for p in range(npix):
        out[unsafe_offset=p] = (
            t[unsafe_offset=nbr[unsafe_offset=2 * npix + p]]
            - t[unsafe_offset=nbr[unsafe_offset=1 * npix + p]]
        )
        out[unsafe_offset=npix + p] = (
            t[unsafe_offset=nbr[unsafe_offset=4 * npix + p]]
            - t[unsafe_offset=nbr[unsafe_offset=3 * npix + p]]
        )


@export("cpk_normalize_flows")
def cpk_normalize_flows(npix: Int, mu_addr: Int) abi("C"):
    """In-place `mu /= (1e-60 + (mu**2).sum(axis=0) ** 0.5)` on a `(2, npix)` buffer.

    Zero flow stays zero rather than becoming NaN, which is what the epsilon is
    for.
    """
    var mu = fp(mu_addr)
    for p in range(npix):
        var y = mu[unsafe_offset=p]
        var x = mu[unsafe_offset=npix + p]
        var scale = sqrt(y * y + x * x) + FLOW_EPS
        mu[unsafe_offset=p] = y / scale
        mu[unsafe_offset=npix + p] = x / scale


@export("cpk_flow_error")
def cpk_flow_error(npix: Int, dpm_addr: Int, dpn_addr: Int, out_addr: Int) abi("C"):
    """`err = ((dP_masks - dP_net / 5) ** 2).sum(axis=0)` on `(2, npix)` inputs."""
    var dpm = fp(dpm_addr)
    var dpn = fp(dpn_addr)
    var out = fp(out_addr)
    for p in range(npix):
        var ey = dpm[unsafe_offset=p] - dpn[unsafe_offset=p] / FLOW_SCALE
        var ex = dpm[unsafe_offset=npix + p] - dpn[unsafe_offset=npix + p] / FLOW_SCALE
        out[unsafe_offset=p] = ey * ey + ex * ex


@export("cpk_bilinear")
def cpk_bilinear(
    npts: Int,
    ly: Int,
    lx: Int,
    field_addr: Int,
    gx_addr: Int,
    gy_addr: Int,
    out_addr: Int,
) abi("C"):
    """One bilinear sample of a `(ly, lx)` field at normalised coordinates.

    Reproduces `torch.nn.functional.grid_sample(..., mode="bilinear",
    align_corners=False, padding_mode="zeros")` for a single channel, which is
    what `steps_interp` calls.  With `align_corners=False` the unnormalised
    index is `((g + 1) * size - 1) / 2`, corners outside the field contribute
    nothing, and the four weights are the plain bilinear ones.  Note that this
    is *not* a clamp: a query at g = 1 lands on index `size - 0.5`, whose
    second row is off the field, so it genuinely returns a partial sum.
    """
    var field = fp(field_addr)
    var gx = fp(gx_addr)
    var gy = fp(gy_addr)
    var out = fp(out_addr)
    for n in range(npts):
        out[unsafe_offset=n] = _bilinear(field, 0, ly, lx, gx[unsafe_offset=n], gy[unsafe_offset=n])


def _bilinear(
    field: FPtr, row0: Int, ly: Int, lx: Int, gx: Float64, gy: Float64
) -> Float64:
    """Bilinear fetch from `field[row0 + iy * lx + ix]`, zero-padded at the edges."""
    var ux = ((gx + 1.0) * Float64(lx) - 1.0) * 0.5
    var uy = ((gy + 1.0) * Float64(ly) - 1.0) * 0.5

    var ix0 = Int(floor(ux))
    var iy0 = Int(floor(uy))
    var fx = ux - Float64(ix0)
    var fy = uy - Float64(iy0)
    var ix1 = ix0 + 1
    var iy1 = iy0 + 1

    return (
        _at(field, row0, ly, lx, iy0, ix0) * (1.0 - fy) * (1.0 - fx)
        + _at(field, row0, ly, lx, iy0, ix1) * (1.0 - fy) * fx
        + _at(field, row0, ly, lx, iy1, ix0) * fy * (1.0 - fx)
        + _at(field, row0, ly, lx, iy1, ix1) * fy * fx
    )


def _at(field: FPtr, row0: Int, ly: Int, lx: Int, iy: Int, ix: Int) -> Float64:
    if iy < 0 or iy >= ly or ix < 0 or ix >= lx:
        return Float64(0.0)
    return field[unsafe_offset=row0 + iy * lx + ix]


@export("cpk_follow_flows")
def cpk_follow_flows(
    niter: Int,
    npts: Int,
    ly: Int,
    lx: Int,
    dp_addr: Int,
    inds_addr: Int,
    out_addr: Int,
    tmp_addr: Int,
    res_addr: Int,
) abi("C"):
    """Euler integration of the 2D flow field, mirroring `steps_interp`.

    `dp` is the raw `(2, ly, lx)` flow field, row 0 the y flow and row 1 the x
    flow, exactly as `follow_flows` receives it.  `inds` is `(2, npts)` of
    starting pixel coordinates, row 0 y and row 1 x.  `out` and `tmp` are
    `(2, npts)` scratch owned by the caller.  `res` is `(npts, 2)` of final
    pixel coordinates, row 0 y and row 1 x, which is the layout `follow_flows`
    hands back.

    Positions are normalised to [-1, 1], advected `niter` times, clamped after
    every step, then un-normalised, all inside the kernel.
    """
    var dp = fp(dp_addr)
    var inds = fp(inds_addr)
    var out = fp(out_addr)
    var tmp = fp(tmp_addr)
    var res = fp(res_addr)

    var sy = Float64(ly - 1)
    var sx = Float64(lx - 1)
    var plane = ly * lx

    # Normalise starting positions to [-1, 1]; row 0 holds x, row 1 holds y.
    for n in range(npts):
        out[unsafe_offset=n] = inds[unsafe_offset=npts + n] / sx * 2.0 - 1.0
        out[unsafe_offset=npts + n] = inds[unsafe_offset=n] / sy * 2.0 - 1.0

    for _t in range(niter):
        for n in range(npts):
            var gx = out[unsafe_offset=n]
            var gy = out[unsafe_offset=npts + n]
            # im[0] is the x flow (row 1 of dP) and im[1] the y flow (row 0).
            tmp[unsafe_offset=n] = _bilinear(dp, plane, ly, lx, gx, gy)
            tmp[unsafe_offset=npts + n] = _bilinear(dp, 0, ly, lx, gx, gy)
        for n in range(npts):
            var gx = out[unsafe_offset=n] + tmp[unsafe_offset=n] * (2.0 / sx)
            var gy = out[unsafe_offset=npts + n] + tmp[unsafe_offset=npts + n] * (2.0 / sy)
            out[unsafe_offset=n] = _clamp1(gx)
            out[unsafe_offset=npts + n] = _clamp1(gy)

    for n in range(npts):
        res[unsafe_offset=2 * n] = (out[unsafe_offset=npts + n] + 1.0) * 0.5 * sy
        res[unsafe_offset=2 * n + 1] = (out[unsafe_offset=n] + 1.0) * 0.5 * sx
