import numpy as np
import pytest

from flyhigh.senses.frame import PanoramicFrame, looming_disc, moving_spot, rotating_grating


def darkest_azimuth(frame):
    """Centre (mean azimuth) of the columns that are at the minimum column-mean luminance."""
    col = frame.lum.mean(axis=0)
    return frame.az_deg[col <= col.min() + 1e-6].mean()


def test_grey_frame_geometry():
    f = PanoramicFrame.grey()
    assert f.lum.shape == (180, 360) and f.lum.dtype == np.float32
    assert f.lum.min() == f.lum.max() == pytest.approx(0.5)
    assert f.az_deg[0] == pytest.approx(-179.5) and f.az_deg[-1] == pytest.approx(179.5)
    assert f.el_deg[0] == pytest.approx(89.5) and f.el_deg[-1] == pytest.approx(-89.5)  # row 0 = top


def test_looming_disc_grows_from_start_to_end_angle():
    frames = looming_disc(az=60, el=0, start_deg=5, end_deg=60, duration_ms=500)
    assert len(frames) == 50
    dark0 = (frames[0].lum < 0.25).sum()
    dark1 = (frames[-1].lum < 0.25).sum()
    assert 0 < dark0 < dark1
    # the disc is centred at azimuth 60: the darkest column is there
    assert darkest_azimuth(frames[-1]) == pytest.approx(60, abs=1.5)
    # background untouched on the far side
    assert frames[-1].lum[:, 0] == pytest.approx(0.5)


def test_rotating_grating_shifts_by_speed():
    frames = rotating_grating(wavelength_deg=30, deg_per_s=60, duration_ms=100, direction=+1)
    assert len(frames) == 10
    row0 = frames[0].lum[90]
    row5 = frames[5].lum[90]  # 50 ms later the pattern moved 3 degrees = 3 pixels to the right
    np.testing.assert_allclose(np.roll(row0, 3), row5, atol=1e-6)
    assert frames[0].lum.min() >= 0 and frames[0].lum.max() <= 1


def test_moving_spot_moves_right():
    frames = moving_spot(az0=-30, el=0, deg_per_s=100, duration_ms=300)
    assert darkest_azimuth(frames[0]) == pytest.approx(-30, abs=1.5)
    assert darkest_azimuth(frames[-1]) == pytest.approx(-1, abs=2)


def test_looming_disc_centre_does_not_depend_on_start_size():
    for start in (5, 20, 40):
        frames = looming_disc(az=60, el=0, start_deg=start, end_deg=60, duration_ms=200)
        assert darkest_azimuth(frames[-1]) == pytest.approx(60, abs=1.5)
