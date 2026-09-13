"""One compound eye = 721 hexagonal columns (flyvis extent 15), each looking in a direction.

flyvis axial hex coords (u, v) → planar offsets via its own `hex_to_pixel` convention, scaled so
neighbouring columns are `spacing_deg` apart; the plane is placed on the sphere at the eye's
centre direction. Each column integrates light over a Gaussian acceptance cone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from flyhigh.senses.frame import PanoramicFrame, angular_distance


def _import_flyvis_hex_utils():
    """Import flyvis's hex-lattice helpers without leaking its import-time side effect.

    flyvis's package `__init__` calls `torch.set_default_device(cuda if available else cpu)`
    as a side effect of import. EyeGeometry/EyeSampler only need pure-numpy hex-lattice math
    from it, but importing this module anywhere in a test session (e.g. via pytest collection)
    would otherwise silently switch every other module's default torch device — breaking code
    such as flyhigh.brain.lif that mixes explicit-device and default-device tensors. Restore
    whatever default device was in effect before this import.
    """
    import torch

    prior_device = torch.get_default_device()
    from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

    torch.set_default_device(prior_device)
    return get_hex_coords, hex_to_pixel


get_hex_coords, hex_to_pixel = _import_flyvis_hex_utils()


@dataclass(frozen=True)
class EyeGeometry:
    side: str  # "L" or "R"
    extent: int = 15
    spacing_deg: float = 5.0
    center_az_deg: float = 65.0  # right eye; the left eye is mirrored
    center_el_deg: float = 0.0
    cone_deg: float = 5.0  # acceptance-cone half-width (Gaussian sigma ≈ cone/2)
    u: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    az_deg: np.ndarray = field(init=False, repr=False)
    el_deg: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        u, v = get_hex_coords(self.extent)
        x, y = hex_to_pixel(u, v)  # neighbours are sqrt(3) apart in this convention
        scale = self.spacing_deg / np.sqrt(3.0)
        sign = 1.0 if self.side == "R" else -1.0
        az = sign * (self.center_az_deg + x * scale)
        el = self.center_el_deg - y * scale  # hex_to_pixel's y grows downward
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "az_deg", az.astype(np.float32))
        object.__setattr__(self, "el_deg", el.astype(np.float32))

    @property
    def n_columns(self) -> int:
        return len(self.u)


class EyeSampler:
    """Sparse (n_eyes*721, H*W) matrix of normalised Gaussian cones; sample() is one matmul."""

    def __init__(self, frame_shape: tuple[int, int], eyes: list[EyeGeometry]):
        h, w = frame_shape
        self.eyes = eyes
        frame = PanoramicFrame.grey(h=h, w=w)
        azg, elg = frame.angular_grid()
        rows = []
        for eye in eyes:
            for az0, el0 in zip(eye.az_deg, eye.el_deg):
                d = angular_distance(azg, elg, az0, el0)
                wgt = np.exp(-0.5 * (d / (eye.cone_deg / 2.0)) ** 2)
                # Hard cutoff at the acceptance cone's half-width itself (not a multiple of it):
                # cone_deg is the cone's half-width, so nothing outside it should contribute.
                wgt[d > eye.cone_deg] = 0.0
                # Keep row weights in float64: renormalised float32 weights drift from summing
                # to exactly 1.0 by ~1e-6, which would leak into every sampled luminance.
                rows.append(sp.csr_matrix((wgt / wgt.sum()).ravel()))
        self.M = sp.vstack(rows).tocsr()

    def sample(self, frame: PanoramicFrame) -> np.ndarray:
        out = self.M @ frame.lum.ravel()
        return out.reshape(len(self.eyes), -1)
