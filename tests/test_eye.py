import numpy as np
import pytest

from flyhigh.senses.eye import EyeGeometry, EyeSampler
from flyhigh.senses.frame import PanoramicFrame


def test_right_eye_looks_right_and_covers_about_75_degrees():
    eye = EyeGeometry("R")
    assert eye.u.shape == (721,) and eye.az_deg.shape == (721,)
    centre = (eye.u == 0) & (eye.v == 0)
    assert eye.az_deg[centre].item() == pytest.approx(65.0) and eye.el_deg[centre].item() == pytest.approx(0.0)
    assert eye.az_deg.min() > -15 and eye.az_deg.max() < 145      # 15 columns * 5 deg each side
    assert abs(eye.el_deg).max() < 80


def test_left_eye_is_mirror_of_right():
    l, r = EyeGeometry("L"), EyeGeometry("R")
    np.testing.assert_allclose(np.sort(l.az_deg), np.sort(-r.az_deg), atol=1e-6)


def test_bright_spot_lights_the_column_looking_at_it():
    eye = EyeGeometry("R")
    sampler = EyeSampler((180, 360), [eye])
    frame = PanoramicFrame.grey(0.0)
    target = 100  # some column
    az, el = eye.az_deg[target], eye.el_deg[target]
    azg, elg = frame.angular_grid()
    lum = frame.lum.copy(); lum[(abs(azg - az) < 3) & (abs(elg - el) < 3)] = 1.0
    out = sampler.sample(PanoramicFrame(lum))
    assert out.shape == (1, 721)
    assert out[0].argmax() == target and out[0, target] > 0.5


def test_grey_frame_samples_to_grey_everywhere():
    sampler = EyeSampler((180, 360), [EyeGeometry("L"), EyeGeometry("R")])
    out = sampler.sample(PanoramicFrame.grey(0.5))
    assert out.shape == (2, 721)
    np.testing.assert_allclose(out, 0.5, atol=1e-6)
