"""Compute-oriented subset of Cellpose with the per-pixel dynamics in Mojo.

The package installs alongside the real `cellpose`; it never imports it.
"""

from __future__ import annotations

from . import dynamics
from ._lib import (
    bilinear,
    diffuse,
    flow_error,
    flow_grads,
    follow_flows,
    normalize_flows,
)
from .dynamics import (
    build_graph,
    center_of_mass,
    flow_error_per_mask,
    get_centers,
    masks_to_flows,
)

__version__ = "0.1.0"

__all__ = [
    "bilinear",
    "build_graph",
    "center_of_mass",
    "diffuse",
    "dynamics",
    "flow_error",
    "flow_error_per_mask",
    "flow_grads",
    "follow_flows",
    "get_centers",
    "masks_to_flows",
    "normalize_flows",
]
