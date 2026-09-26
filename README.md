# mojo-cellpose-kernels

`mojo-cellpose-kernels` is the compute-oriented subset of
[Cellpose](https://cellpose.readthedocs.io/) — Stringer, Rariden and
Pachitariu's generalist algorithm for segmenting cells — with the per-pixel
dynamics loops implemented in Mojo and callable from Python.

Cellpose does not segment by classifying pixels. It predicts a **flow field**
that points from every pixel to its cell centre, and then *integrates* that
field: pixels that follow the flow to the same place belong to the same cell.
Those integration loops are the numeric core of the package, and they are what
this port covers.

The Python package is named `mojo_cellpose_kernels`, so it installs alongside
the real `cellpose` and never imports it.

```python
import numpy as np
import mojo_cellpose_kernels as mcp

masks = np.zeros((256, 256), dtype=int)
masks[40:90, 30:80] = 1
masks[120:180, 150:210] = 2

mu, slices = mcp.masks_to_flows(masks)        # unit flows towards each centre
p = mcp.follow_flows(dP, inds, niter=200)     # advect pixels, clamped
err = mcp.flow_error_per_mask(masks, mu, dP_net)
```

## Covered subset

| area | upstream routine | implemented API | Mojo kernel |
| --- | --- | --- | --- |
| Flow from labels | `dynamics._extend_centers_gpu` | `diffuse` | `cpk_diffuse` |
| Flow from the diffused field | `grads = T[neighbors[[2,1,4,3]]]` | `flow_grads` | `cpk_flow_grads` |
| Unit-length flow | `mu /= 1e-60 + (mu**2).sum(axis=0)**0.5` | `normalize_flows` | `cpk_normalize_flows` |
| End-to-end mask -> flow | `dynamics.masks_to_flows_gpu` | `masks_to_flows` | the three above |
| Bilinear flow sample | `torch.nn.functional.grid_sample` | `bilinear` | `cpk_bilinear` |
| Pixel advection | `dynamics.steps_interp` | `follow_flows` | `cpk_follow_flows` |
| Quality-control metric | `dynamics.flow_error` | `flow_error`, `flow_error_per_mask` | `cpk_flow_error` |
| Graph construction | `center_of_mass`, `get_centers`, neighbour tables | `build_graph`, `center_of_mass`, `get_centers` | — (NumPy, bookkeeping) |

**Not implemented, and why.** Cellpose is mostly a neural network: the UNet,
ViT and CPSAM models, the augmentation and training loops, the dataloaders and
the checkpoint plumbing are out of scope for a compute port and are left to the
real package. The 3D path (`masks_to_flows_gpu_3d`, `follow_flows` on 3D
volumes) is not ported: `cpk_follow_flows` is 2D only, because the 3D
`grid_sample` axis ordering is a different convention and would need its own
parity coverage. `get_masks_torch` (the histogram-and-peaks mask builder),
`get_masks_unet`, `remove_bad_flow_masks`, `max_pool_nd`, the `denoise` module
and the `metrics` module are also left upstream. `labels_to_flows` is
`masks_to_flows` plus file IO, which is left upstream too.

Two details of the port that are worth stating:

- The diffusion is **simultaneous**, not in place. Torch's
  `T_flat[flat_center] = (Tneigh * isneighbor).sum(dim=0)` reads the whole
  neighbour field before writing any of it; a single in-place loop is
  Gauss-Seidel and drifts away from the reference. `cpk_diffuse` ping-pongs
  between two buffers.
- Cellpose's flow is unit length, so `flow_error` compares the mask flow
  against `dP_net / 5`; a perfect network prediction is one that is *five
  times* the mask flow, not one that equals it.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-cellpose-kernels.so`. Set
`PYTHONPATH=python` outside a Pixi task. The tests need `torch`, `numpy` and
`scipy` in the test environment.

## How parity was established

**Cellpose is not installed in the shared test venv**, and the toolchain
directory must not be modified, so the parity tests do not import it. Instead
`tests/reference.py` holds a *torch transcription* of the three upstream
routines — `_extend_centers_gpu`, `steps_interp` and `grid_sample` — with
cellpose's own names, argument order and torch calls left intact so it can be
diffed against upstream.

That transcription was checked against the real package during development:
installed into a scratch directory, `masks_to_flows_gpu` on a 40x50 label image
with 200 diffusion iterations, and the transcription reproduced it to
`0.0` max absolute difference. The Mojo kernels then agree with the
transcription to `2.2e-16`, i.e. to the last bit of the float64 result.

So the chain is: Mojo kernel -> torch transcription -> real cellpose, with the
middle link measured rather than assumed.

## Tests

```
$ PYTHONPATH=python pytest tests -q
30 passed
```

- `tests/test_flows.py` compares `masks_to_flows` against the transcription for
  1, 7, 50 and 200 diffusion iterations, and adds analytic checks: the flow is
  unit length inside an object, background pixels have exactly zero flow, and
  every pixel's flow has a positive component towards its centre.
- `tests/test_dynamics.py` compares `bilinear` and `follow_flows` against
  torch's `grid_sample` and the transcription, and adds closed-form cases that
  chaos cannot hide: a zero flow field must not move anything, a constant flow
  must translate by exactly `niter * v` pixels, and the `[-1, 1]` clamp must
  hold every pixel inside the image.
- `tests/test_ffi.py` pins each kernel's declared arity and runs the kernels on
  mappings placed at fixed addresses above 4 GiB.

Two properties of the upstream numerics shape the tolerances. `steps_interp`
runs in float32 upstream and the advection is chaotic, so on a *random* flow
field the two implementations separate geometrically over tens of iterations
however correct both are. The parity cases therefore use a smooth converging
field where float32-versus-float64 noise stays at the float32 floor, plus a
one-step check on a random field, which is where a wrong sampling convention
would show immediately. And Mojo emits FMA, so the diffusion agrees with the
torch transcription to `rtol=1e-13` rather than bit for bit.

## Performance

Best-of-three wall clock, same process, against the torch code from
`tests/reference.py` on one thread. Every case verifies agreement before
timing. The Mojo kernels are float64; the reference is float32.

| case | cellpose (torch, 1 thread) | mojo-cellpose-kernels | ratio |
| --- | ---: | ---: | ---: |
| diffusion 1024x1024, npix=5760, 100 iters | 126.4 ms | 24.0 ms | 5.27x |
| flow_grads, npix=5760 | 0.4 ms | 0.04 ms | 9.13x |
| follow_flows, 1048576 points, 200 iters | 14864 ms | 48858 ms | 0.30x |
| flow_error, n=4194304 | 233.7 ms | 49.0 ms | 4.77x |

Reproduce with `pixi run bench`.

Honest reading:

- The **diffusion** and **flow_error** wins are real. Both are tight gather
  loops that torch expresses as advanced indexing, which allocates an
  intermediate `(9, npix)` tensor per iteration; the Mojo kernel reads the
  neighbours straight out of the neighbour table.
- **follow_flows loses by about 3.3x**, and that is the honest result. Each
  iteration is two bilinear samples per point with a per-corner bounds test
  and a clamp; torch's `grid_sample` is a hand-vectorised CPU kernel that
  handles all of that in registers, and the Mojo version cannot compete. The
  kernel is here because it is the heart of the mask-recovery routine and needs
  to be right, not because it is fast.

These timings come from a shared 36-core box with roughly thirty other builds
running; treat single digits as indicative.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit.
`build/build.sh` compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-cellpose-kernels.so`.

The `python/mojo_cellpose_kernels` layer owns every array. It normalises inputs
to contiguous `float64`, builds the neighbour graph in NumPy (that is
bookkeeping, not compute), and makes one call per loop. Buffers cross the C
ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. `_lib._ARITY` pins each kernel's parameter count and `_load`
refuses to bind a symbol whose declared arity disagrees, because a short
`argtypes` list fails silently: ctypes truncates the surplus address to 32 bits
and nothing breaks until a buffer lands above 4 GiB.

## License

MIT
