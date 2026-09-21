"""The M3 drum at fly scale (flybody works in centimetres): a striped drum, a fingertip 'hand'."""

from __future__ import annotations

from dataclasses import dataclass

from dm_control import composer, mjcf


@dataclass(frozen=True)
class DrumParams:
    size: tuple[float, float, float] = (30.0, 30.0, 15.0)  # cm; ~100 body lengths like the M3 room
    stripe_cm: float = 3.75  # 14° at 15 cm
    hand_radius: float = 4.0  # a fingertip
    hand_home: tuple[float, float, float] = (8.0, -10.0, 8.0)  # ahead-right of the fly, over a plain wall


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
        for sx, sy, sz, px, py, pz, look, ground in (
            (x, y, 0.1, 0, 0, -0.1, plain, True), (x, y, 0.1, 0, 0, p.size[2] + 0.1, plain, False),
            (x, 0.1, z, 0, y + 0.1, z, {"material": "stripes"}, False),
            (x, 0.1, z, 0, -y - 0.1, z, {"material": "stripes"}, False),
            (0.1, y, z, x + 0.1, 0, z, plain, False), (0.1, y, z, -x - 0.1, 0, z, plain, False),
        ):
            g = root.worldbody.add("geom", type="box", size=(sx, sy, sz), pos=(px, py, pz), **look)
            if ground:
                self._ground.append(g)
        hand = root.worldbody.add("body", name="hand", mocap=True, pos=p.hand_home)
        hand.add("geom", type="sphere", size=(p.hand_radius,), rgba=(0.05, 0.05, 0.05, 1), contype=0, conaffinity=0)
        root.worldbody.add("camera", name="overview", pos=(0, -y + 1, p.size[2] - 1), xyaxes=(1, 0, 0, 0, 0.7, 0.7), fovy=80)

    @property
    def ground_geoms(self):
        return tuple(self._ground)

    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root
