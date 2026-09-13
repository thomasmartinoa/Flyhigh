"""A camera-agnostic picture of the world: luminance over azimuth × elevation.

Row 0 is the top of the sky (elevation +90), column 0 is azimuth −180 (behind the fly),
azimuth increases to the right, +90 is the fly's right side, 0 is straight ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TICK_MS = 10.0


@dataclass(frozen=True)
class PanoramicFrame:
    lum: np.ndarray  # (H, W) float32 in [0, 1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.lum.shape

    @property
    def az_deg(self) -> np.ndarray:
        w = self.lum.shape[1]
        return -180.0 + (np.arange(w) + 0.5) * 360.0 / w

    @property
    def el_deg(self) -> np.ndarray:
        h = self.lum.shape[0]
        return 90.0 - (np.arange(h) + 0.5) * 180.0 / h

    @classmethod
    def grey(cls, value: float = 0.5, h: int = 180, w: int = 360) -> PanoramicFrame:
        return cls(np.full((h, w), value, dtype=np.float32))

    def angular_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(az, el) in degrees for every pixel, each (H, W)."""
        return np.meshgrid(self.az_deg, self.el_deg)


def _angular_distance(az, el, az0, el0):
    """Great-circle distance in degrees between every pixel and a direction."""
    a, e, a0, e0 = np.radians(az), np.radians(el), np.radians(az0), np.radians(el0)
    cos_d = np.sin(e) * np.sin(e0) + np.cos(e) * np.cos(e0) * np.cos(a - a0)
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def _disc(base: PanoramicFrame, az, el, radius_deg, lum) -> PanoramicFrame:
    azg, elg = base.angular_grid()
    out = base.lum.copy()
    out[_angular_distance(azg, elg, az, el) <= radius_deg] = lum
    return PanoramicFrame(out)


def looming_disc(
    az, el, start_deg, end_deg, duration_ms, lum=0.0, bg=0.5, h=180, w=360
):
    """A dark disc whose angular *diameter* grows linearly from start_deg to end_deg."""
    n = round(duration_ms / TICK_MS)
    base = PanoramicFrame.grey(bg, h, w)
    return [
        _disc(
            base,
            az + start_deg + 0.5,
            el,
            0.5 * (start_deg + (end_deg - start_deg) * i / max(n - 1, 1)),
            lum,
        )
        for i in range(n)
    ]


def rotating_grating(
    wavelength_deg, deg_per_s, duration_ms, direction=+1, h=180, w=360, contrast=1.0
):
    """Vertical stripes covering the whole panorama, drifting `direction` (+1 = rightwards)."""
    n = round(duration_ms / TICK_MS)
    az = PanoramicFrame.grey(h=h, w=w).az_deg
    frames = []
    for i in range(n):
        phase = direction * deg_per_s * (i * TICK_MS / 1000.0)
        row = 0.5 + 0.5 * contrast * np.sin(2 * np.pi * (az - phase) / wavelength_deg)
        frames.append(PanoramicFrame(np.tile(row.astype(np.float32), (h, 1))))
    return frames


def moving_spot(
    az0, el, deg_per_s, duration_ms, radius_deg=5.0, lum=0.0, bg=0.5, h=180, w=360
):
    n = round(duration_ms / TICK_MS)
    base = PanoramicFrame.grey(bg, h, w)
    return [
        _disc(base, az0 + deg_per_s * i * TICK_MS / 1000.0, el, radius_deg, lum)
        for i in range(n)
    ]
