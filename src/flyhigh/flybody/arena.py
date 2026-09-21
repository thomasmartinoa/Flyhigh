"""The M3 drum at fly scale (flybody works in centimetres): a striped drum, a fingertip 'hand'.

Stripe period 28° (M3 used 14°): flyvis's sustained T4 response to side-wall stripes is a
third of the full-field one, and coarser stripes help. Taller walls or striped end walls give
HS more, but then the rotating drum -- and even the hovering fly's own wing-beat jitter --
fire the marginal giant fiber (docs/05).
"""

from __future__ import annotations

from dataclasses import dataclass

from dm_control import composer, mjcf


@dataclass(frozen=True)
class DrumParams:
    size: tuple[float, float, float] = (30.0, 30.0, 15.0)  # cm; ~100 body lengths like the M3 room
    stripe_cm: float = 7.5  # 28° at 15 cm
    hand_radius: float = 4.0  # a fingertip
    hand_home: tuple[float, float, float] = (8.0, -10.0, 8.0)  # ahead-right of the fly


class FlyDrum(composer.Arena):
    """Floor, ceiling and end walls plain grey; side walls striped; one light; a mocap hand."""

    def _build(self, params: DrumParams | None = None, name="drum"):
        super()._build(name=name)
        self.params = p = params or DrumParams()
        x, y, z = (v / 2 for v in p.size)
        root = self._mjcf_root
        # visual settings must match the walker's on attach: the headlight is set as dm_control's
        # Floor arena (which flybody's tasks use) sets it
        root.visual.headlight.set_attributes(ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1])
        tex = root.asset.add("texture", name="stripes", type="2d", builtin="checker",
                             rgb1=(0.15, 0.15, 0.15), rgb2=(0.85, 0.85, 0.85), width=2, height=1)
        root.asset.add("material", name="stripes", texture=tex, texrepeat=(1.0 / p.stripe_cm, 0.0001),
                       texuniform=True)
        root.worldbody.add("light", pos=(0, 0, p.size[2] - 1), dir=(0, 0, -1), diffuse=(0.6, 0.6, 0.6))
        self._ground = []
        plain = {"rgba": (0.5, 0.5, 0.5, 1)}
        for sx, sy, sz, px, py, pz in ((x, y, 0.1, 0, 0, -0.1), (x, y, 0.1, 0, 0, p.size[2] + 0.1)):  # floor, ceiling
            self._ground.append(root.worldbody.add("geom", type="box", size=(sx, sy, sz), pos=(px, py, pz), **plain))
        # the four walls are one mocap body, so the drum can be spun around the fly (the
        # optomotor experiment: a disturbance the fly's own flight controller cannot cancel)
        drum = root.worldbody.add("body", name="drum", mocap=True, pos=(0, 0, 0))
        for size, pos, look in (
            ((x, 0.1, z), (0, y + 0.1, z), {"material": "stripes"}), ((x, 0.1, z), (0, -y - 0.1, z), {"material": "stripes"}),
            ((0.1, y, z), (x + 0.1, 0, z), plain), ((0.1, y, z), (-x - 0.1, 0, z), plain),
        ):
            drum.add("geom", type="box", size=size, pos=pos, **look)
        hand = root.worldbody.add("body", name="hand", mocap=True, pos=p.hand_home)
        hand.add("geom", type="sphere", size=(p.hand_radius,), rgba=(0.05, 0.05, 0.05, 1), contype=0, conaffinity=0)
        root.worldbody.add("camera", name="overview", pos=(0, -y + 1, p.size[2] - 1), xyaxes=(1, 0, 0, 0, 0.7, 0.7), fovy=80)

    @property
    def ground_geoms(self):
        return tuple(self._ground)  # floor and ceiling: contact-free in flight tasks

    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root
