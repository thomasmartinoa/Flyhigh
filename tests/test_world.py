import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from flyhigh.senses.eye import EyeGeometry, EyeSampler
from flyhigh.world.eyes import CubemapPanorama, bearing_to_direction
from flyhigh.world.scene import RoomParams, build_mjcf


def test_scene_builds_with_bodies_hand_and_cameras():
    m = mujoco.MjModel.from_xml_string(build_mjcf(2))
    assert m.nbody >= 4 and m.nmocap == 1
    for i in range(2):
        for f in ("front", "left", "right", "up", "down"):
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"agent{i}_{f}") >= 0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert abs(d.body("agent0").xpos[2] - 1.0) < 1e-6


def test_bearing_directions():
    np.testing.assert_allclose(bearing_to_direction(0, 0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(bearing_to_direction(90, 0), [0, -1, 0], atol=1e-12)  # + az = right
    np.testing.assert_allclose(bearing_to_direction(0, 90), [0, 0, 1], atol=1e-12)


def test_cubemap_covers_everything_but_the_rear():
    m = mujoco.MjModel.from_xml_string(build_mjcf(1))
    pano = CubemapPanorama(m, agent=0, face_px=32)
    cov = pano.covered.reshape(180, 360)
    az = -180 + np.arange(360) + 0.5
    assert cov[:, np.abs(az) < 130].all()  # front, sides, up, down
    assert not cov[90, np.abs(az) > 170].any()  # straight behind, on the horizon


@pytest.mark.parametrize("az,el", [(0, 0), (60, 0), (-60, 10), (0, 60), (120, -30), (90, 0)])
def test_bright_sphere_at_a_bearing_lights_that_panorama_pixel_and_eye_column(az, el):
    """A white sphere 1 m from agent 0 at bearing (az, el), in a dark room: the brightest
    panorama pixel and the brightest eye column must look at it."""
    p = RoomParams()
    xml = build_mjcf(1, p, start=[(0.0, 0.0, 1.5)]).replace(
        'rgb2="0.85 0.85 0.85"', 'rgb2="0.15 0.15 0.15"'
    ).replace(
        "</worldbody>",
        '<body name="probe" mocap="true" pos="0 0 0"><geom type="sphere" size="0.08" rgba="1 1 1 1" '
        'contype="0" conaffinity="0"/></body></worldbody>',
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.mocap_pos[1] = np.array([0.0, 0.0, 1.5]) + bearing_to_direction(az, el)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 96, 96)
    frame = CubemapPanorama(m, agent=0, face_px=96).render(r, d)
    r.close()
    bright = frame.lum > 0.5 * frame.lum.max()  # the sphere's disc (its lit top is brightest)
    rows, cols = np.nonzero(bright)
    assert abs(frame.az_deg[cols].mean() - az) < 3 and abs(frame.el_deg[rows].mean() - el) < 3
    if 0 < az < 130 and abs(el) < 60:  # inside the right eye's field: the eye sees it too
        eye = EyeGeometry("R")
        out = EyeSampler(frame.shape, [eye]).sample(frame)[0]
        j = out.argmax()
        assert abs(eye.az_deg[j] - az) < 6 and abs(eye.el_deg[j] - el) < 6
