"""The room as an MJCF string: a striped drum, a mocap hand, and n agent bodies with eye cameras.

The room is the fly physiologist's optomotor drum: vertical stripes on the side walls give
rotation its optic flow (HS), while floor, ceiling and end walls are plain grey so that flying
forward does not paint an expanding pattern on the eye (a checkered floor made the looming
detectors fire at the room itself). The only dark objects are the hand, the other flies and any
`pillars` -- dark floor-to-ceiling cylinders, the static obstacle a cruising body has to notice.

Sizes are metres. What matters to a 5°-column eye is angular size: the 30 cm bodies and hand
subtend 40° -- M2's escape size -- at 0.4 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    size: tuple[float, float, float] = (6.0, 6.0, 3.0)  # x, y, z extent (m); at 4 m the flies' own
    # cruise past the stripes tripped the (marginal) giant fiber in most runs
    stripe_m: float = 0.75  # stripe period on the side walls (14° at 3 m; 0.125 m aliases at 96 px faces)
    hand_radius: float = 0.15
    hand_home: tuple[float, float, float] = (1.0, -1.2, 1.2)  # ahead-right of agent 0, over the plain end wall
    pillars: tuple[tuple[float, float], ...] = ()  # (x, y) of dark floor-to-ceiling obstacles
    balls: tuple[tuple[float, float, float], ...] = ()  # (x, y, z) of dark compact obstacles
    pillar_radius: float = 0.15  # 30 cm across, like a body: ~40° at 0.4 m
    body_size: float = 0.15  # half-width of the box body: 30 cm across, so another fly looms to the
    # escape size (~40°, M2) at 0.4 m instead of at contact
    body_mass: float = 0.03
    timestep: float = 0.002


def agent_group(i: int) -> int:
    """Geom group of agent i's own geoms: its eyes render every group but this one (max 5 agents)."""
    return 1 + i


def agent_body(i: int, p: RoomParams, pos) -> str:
    """`pos` = (x, y, z) or (x, y, z, yaw_deg); yaw 0 faces +x."""
    s = p.body_size
    g = agent_group(i)
    yaw = float(pos[3]) if len(pos) > 3 else 0.0
    half = np.radians(yaw) / 2
    cams = "".join(
        f'<camera name="agent{i}_{face}" pos="0 0 0" xyaxes="{axes}" fovy="{FACE_FOVY}"/>'
        for face, axes in FACES.items()
    )
    rotors = "".join(
        f'<geom type="cylinder" size="{0.6 * s} 0.003" pos="{x * s * 1.4} {y * s * 1.4} {s}" '
        f'rgba="0.1 0.1 0.1 1" mass="0" group="{g}"/>'
        for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    )
    return (
        f'<body name="agent{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{np.cos(half)} 0 0 {np.sin(half)}">'
        f"<freejoint/>"
        f'<geom type="box" size="{s} {s} {0.4 * s}" rgba="0.05 0.05 0.05 1" mass="{p.body_mass}" group="{g}"/>'
        f'<geom type="box" size="{0.4 * s} {0.2 * s} {0.1 * s}" pos="{s} 0 0" rgba="0.8 0.1 0.1 1" mass="0" '
        f'group="{g}"/>'
        f"{rotors}{cams}</body>"
    )


def build_mjcf(n_agents: int = 2, p: RoomParams | None = None, start=None) -> str:
    p = p or RoomParams()
    x, y, z = (v / 2 for v in p.size)
    rep = 1.0 / p.stripe_m  # texuniform: repeats per metre; one repeat = a light + a dark stripe
    # default: a line along x, alternating sides, facing each other's general direction
    start = start or [(-0.75 + 1.5 * i, (-1) ** i * 0.2, 1.0, 180.0 * (i % 2)) for i in range(n_agents)]
    pillars = "".join(
        f'<geom name="pillar{i}" type="cylinder" size="{p.pillar_radius} {p.size[2] / 2}" '
        f'pos="{px} {py} {p.size[2] / 2}" rgba="0.05 0.05 0.05 1"/>'
        for i, (px, py) in enumerate(p.pillars)
    ) + "".join(
        f'<geom name="ball{i}" type="sphere" size="{p.pillar_radius}" pos="{bx} {by} {bz}" '
        f'rgba="0.05 0.05 0.05 1"/>'
        for i, (bx, by, bz) in enumerate(p.balls)
    )
    plain = 'rgba="0.5 0.5 0.5 1"'
    walls = "".join(
        f'<geom type="box" size="{sx} {sy} {sz}" pos="{px} {py} {pz}" {look}/>'
        for sx, sy, sz, px, py, pz, look in (
            (x, y, 0.05, 0, 0, -0.05, plain), (x, y, 0.05, 0, 0, p.size[2] + 0.05, plain),  # floor, ceiling
            (x, 0.05, z, 0, y + 0.05, z, 'material="stripes"'), (x, 0.05, z, 0, -y - 0.05, z, 'material="stripes"'),
            (0.05, y, z, x + 0.05, 0, z, plain), (0.05, y, z, -x - 0.05, 0, z, plain),  # end walls
        )
    )
    return f"""<mujoco model="the_box">
  <option timestep="{p.timestep}" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/><map znear="0.002"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="stripes" type="2d" builtin="checker" rgb1="0.15 0.15 0.15" rgb2="0.85 0.85 0.85" width="2" height="1"/>
    <material name="stripes" texture="stripes" texrepeat="{rep} 0.0001" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 0 {p.size[2] - 0.2}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    {walls}{pillars}
    <body name="hand" mocap="true" pos="{p.hand_home[0]} {p.hand_home[1]} {p.hand_home[2]}">
      <geom type="sphere" size="{p.hand_radius}" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="overview" pos="0 {-y + 0.2} {p.size[2] - 0.2}" xyaxes="1 0 0 0 0.7 0.7" fovy="80"/>
    {"".join(agent_body(i, p, s) for i, s in enumerate(start[:n_agents]))}
  </worldbody>
</mujoco>"""
