"""The room as an MJCF string: checkered walls, a mocap hand, and n agent bodies with eye cameras.

Sizes are metres. What matters to a 5°-column eye is angular size: a 10 cm body at 1 m is 6°, the
30 cm hand at 50 cm is 33° (M2's looming disc at its escape size).
"""

from __future__ import annotations

from dataclasses import dataclass

# Camera frames are given as `xyaxes` (image-right, image-up) in the body frame, body +x forward,
# +y left, +z up. MuJoCo cameras look along their -z; these five cover everything but the rear.
FACES = {
    "front": "0 -1 0 0 0 1",
    "left": "1 0 0 0 0 1",
    "right": "-1 0 0 0 0 1",
    "up": "0 -1 0 -1 0 0",
    "down": "0 -1 0 1 0 0",
}
FACE_FOVY = 90.0


@dataclass(frozen=True)
class RoomParams:
    size: tuple[float, float, float] = (4.0, 4.0, 3.0)  # x, y, z extent (m)
    checker_m: float = 0.5  # texture period on the walls (14° at 2 m; 0.25 m aliases at 96 px faces)
    hand_radius: float = 0.15
    body_size: float = 0.05  # half-width of the box body (10 cm across)
    body_mass: float = 0.03
    timestep: float = 0.002


def agent_body(i: int, p: RoomParams, pos: tuple[float, float, float]) -> str:
    s = p.body_size
    cams = "".join(
        f'<camera name="agent{i}_{face}" pos="0 0 0" xyaxes="{axes}" fovy="{FACE_FOVY}"/>'
        for face, axes in FACES.items()
    )
    rotors = "".join(
        f'<geom type="cylinder" size="{0.6 * s} 0.003" pos="{x * s * 1.4} {y * s * 1.4} {s}" '
        f'rgba="0.1 0.1 0.1 1" mass="0"/>'
        for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    )
    return (
        f'<body name="agent{i}" pos="{pos[0]} {pos[1]} {pos[2]}">'
        f"<freejoint/>"
        f'<geom type="box" size="{s} {s} {0.4 * s}" rgba="0.05 0.05 0.05 1" mass="{p.body_mass}"/>'
        f'<geom type="box" size="{0.4 * s} {0.2 * s} {0.1 * s}" pos="{s} 0 0" rgba="0.8 0.1 0.1 1" mass="0"/>'
        f"{rotors}{cams}</body>"
    )


def build_mjcf(n_agents: int = 2, p: RoomParams | None = None, start=None) -> str:
    p = p or RoomParams()
    x, y, z = (v / 2 for v in p.size)
    rep = p.size[0] / p.checker_m
    start = start or [(-0.75 + 1.5 * i, (-1) ** i * 0.2, 1.0) for i in range(n_agents)]
    walls = "".join(
        f'<geom type="box" size="{sx} {sy} {sz}" pos="{px} {py} {pz}" material="grid"/>'
        for sx, sy, sz, px, py, pz in (
            (x, y, 0.05, 0, 0, -0.05), (x, y, 0.05, 0, 0, p.size[2] + 0.05),
            (x, 0.05, z, 0, y + 0.05, z), (x, 0.05, z, 0, -y - 0.05, z),
            (0.05, y, z, x + 0.05, 0, z), (0.05, y, z, -x - 0.05, 0, z),
        )
    )
    return f"""<mujoco model="the_box">
  <option timestep="{p.timestep}" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.85 0.85 0.85" width="128" height="128"/>
    <material name="grid" texture="grid" texrepeat="{rep} {rep}" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 0 {p.size[2] - 0.2}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    {walls}
    <body name="hand" mocap="true" pos="{x - 0.5} {y - 0.5} 1.5">
      <geom type="sphere" size="{p.hand_radius}" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="overview" pos="0 -{y + 2.5} {p.size[2] + 1.5}" xyaxes="1 0 0 0 0.55 0.83"/>
    {"".join(agent_body(i, p, s) for i, s in enumerate(start[:n_agents]))}
  </worldbody>
</mujoco>"""
