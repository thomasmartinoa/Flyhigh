"""Cubemap eyes: five 90° pinhole renders per agent, resampled into a PanoramicFrame.

The lookup (which face, which pixel, for every panorama pixel) is derived from the cameras'
orientation quaternions in the model, not from the axis strings in scene.py, so a convention
slip there shows up in the bearing test instead of as a silently rotated eye.
"""

from __future__ import annotations

import numpy as np

from flyhigh.senses.frame import PanoramicFrame
from flyhigh.world.scene import FACE_FOVY, FACES, agent_group


def bearing_to_direction(az_deg, el_deg):
    """Body-frame unit vector for a bearing: az 0 = +x (forward), + = right (−y); el + = up."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


class CubemapPanorama:
    def __init__(self, model, agent: int = 0, face_px: int = 96, h: int = 180, w: int = 360,
                 cam_names: dict[str, str] | None = None, hide_groups=None, eye_frame=None):
        """`cam_names`: face -> camera name (default the M3 scene's `agent{i}_{face}`);
        `hide_groups`: geom groups the eyes skip (default the agent's own group);
        `eye_frame`: 3x3 matrix taking eye-frame directions (x forward, y left, z up) to the
        cameras' parent-body frame, when that body's own frame is something else (flybody's head)."""
        import mujoco

        self.face_px, self.h, self.w = face_px, h, w
        self.opt = mujoco.MjvOption()  # everything but the agent's own body
        self.opt.geomgroup[:] = 1
        for g in ([agent_group(agent)] if hide_groups is None else hide_groups):
            self.opt.geomgroup[g] = 0
        cam_names = cam_names or {f: f"agent{agent}_{f}" for f in FACES}
        self.cam_ids = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_names[f]) for f in FACES}
        assert min(self.cam_ids.values()) >= 0, f"missing eye camera among {cam_names}"
        frame = PanoramicFrame.grey(h=h, w=w)
        azg, elg = frame.angular_grid()
        d = bearing_to_direction(azg.ravel(), elg.ravel())  # (h*w, 3) in the eye frame
        if eye_frame is not None:
            d = d @ np.asarray(eye_frame, dtype=float).T  # -> the parent body's frame
        half = np.tan(np.radians(FACE_FOVY / 2))
        self.face_of = np.full(h * w, -1, dtype=np.int64)
        self.row = np.zeros(h * w, dtype=np.int64)
        self.col = np.zeros(h * w, dtype=np.int64)
        best = np.zeros(h * w)
        for k, (face, cid) in enumerate(self.cam_ids.items()):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.cam_quat[cid])
            c = d @ R.reshape(3, 3)  # camera-frame coords: columns of R are the camera axes in the body frame
            depth = -c[:, 2]  # the camera looks along its -z
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = c[:, 0] / depth, c[:, 1] / depth
            inside = (depth > 0) & (np.abs(x) <= half) & (np.abs(y) <= half) & (depth > best)
            self.face_of[inside] = k
            best[inside] = depth[inside]
            self.col[inside] = np.clip(((x[inside] / half + 1) / 2 * face_px).astype(int), 0, face_px - 1)
            self.row[inside] = np.clip(((1 - y[inside] / half) / 2 * face_px).astype(int), 0, face_px - 1)
        self.covered = self.face_of >= 0

    def render(self, renderer, data) -> PanoramicFrame:
        """`renderer`: a mujoco.Renderer of size (face_px, face_px), shared across agents."""
        import mujoco

        lum = np.full(self.h * self.w, 0.5, dtype=np.float32)
        for k, cid in enumerate(self.cam_ids.values()):
            renderer.update_scene(data, camera=cid, scene_option=self.opt)
            # no shadows or reflections: a 5°-column eye cannot see them, and a model with
            # cinematic shadow maps (flybody: 8192², four lights) spends 30 ms per face on them
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            rgb = renderer.render()
            grey = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255.0
            m = self.face_of == k
            lum[m] = grey[self.row[m], self.col[m]]
        return PanoramicFrame(lum.reshape(self.h, self.w))
