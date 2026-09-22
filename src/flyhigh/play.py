"""Interactive box: the `Play` controller that turns mouse and keyboard into the simulation.

`scripts/play_box.py` is the command line around this. The controller is here so it can be
tested without opening a window: every key handler is a plain method, and the mouse arrives
through `Simulation.on_substep`, which runs after the bodies have written their own forces so a
drag adds to them instead of being overwritten.
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

from flyhigh.motor.readout import ReadoutParams
from flyhigh.sim import Simulation

KEY_SPACE, KEY_H, KEY_J, KEY_K, KEY_R, KEY_F = 32, ord("H"), ord("J"), ord("K"), ord("R"), ord("F")


class Play:
    def __init__(self, sim: Simulation, refractory_ms: float = 300.0):
        self.sim = sim
        self.paused = False
        self.refractory_ms = refractory_ms
        self.viewer = None
        sim.on_substep = self._perturb

    @staticmethod
    def all_keys() -> tuple[int, ...]:
        """Every key the controller answers to, for a keyboard-free smoke test."""
        return (KEY_SPACE, KEY_SPACE, KEY_H, KEY_J, KEY_K, KEY_R, KEY_F, KEY_F, ord("Z"))

    # ----------------------------------------------------------------- input
    def selected_body(self) -> int:
        """Which agent the user has selected in the viewer, or 0."""
        if self.viewer is None:
            return 0
        sel = int(self.viewer.perturb.select)
        for i, body in enumerate(self.sim.bodies):
            if sel and self.sim.model.body_rootid[sel] == body.id:
                return i
        return 0

    def _perturb(self, sim: Simulation) -> None:
        """Let the mouse move mocap bodies and push dynamic ones, on top of the controller."""
        if self.viewer is None:
            return
        pert = self.viewer.perturb
        if not int(pert.active):
            return
        mujoco.mjv_applyPerturbPose(sim.model, sim.data, pert, 0)  # mocap bodies: set the pose
        mujoco.mjv_applyPerturbForce(sim.model, sim.data, pert)    # dynamic bodies: add a force

    def key(self, keycode: int) -> None:
        sim = self.sim
        if keycode == KEY_SPACE:
            self.paused = not self.paused
            print(f"\n[{'paused' if self.paused else 'running'}]")
        elif keycode == KEY_H:
            i = self.selected_body()
            sim.hand.approach(lambda b=sim.bodies[i]: b.pos)
            print(f"\n[hand → body {i}]")
        elif keycode == KEY_J:
            sim.hand.park(); print("\n[hand parked]")
        elif keycode == KEY_K:
            sim.hand.idle(); sim.hand.d.mocap_pos[sim.hand.mocap_id] = sim.hand.home
            print("\n[hand home]")
        elif keycode == KEY_R:
            sim.hand.idle(); sim.hand.d.mocap_pos[sim.hand.mocap_id] = sim.hand.home
            for i, body in enumerate(sim.bodies):
                adr = sim.model.jnt_qposadr[sim.model.body_jntadr[body.id]]
                sim.data.qpos[adr:adr + 3] = [-0.75 + 1.5 * i, (-1) ** i * 0.2, 1.0]
                sim.data.qvel[sim.model.jnt_dofadr[sim.model.body_jntadr[body.id]]:][:6] = 0
                body.z_hold = 1.0
            mujoco.mj_forward(sim.model, sim.data)
            print("\n[reset]")
        elif keycode == KEY_F:
            readout = getattr(sim.agent, "readout", None)
            if readout is None:
                return
            p = readout.p
            new = 0.0 if p.escape_refractory_ms else self.refractory_ms
            readout.p = ReadoutParams(**{**p.__dict__, "escape_refractory_ms": new})
            print(f"\n[escape refractory {new:.0f} ms]")

    # ---------------------------------------------------------------- markers
    def mark_escapes(self) -> None:
        """A red ball over any body whose escape reflex is firing."""
        scn = self.viewer.user_scn if self.viewer is not None else None
        if scn is None:
            return
        scn.ngeom = 0
        for body, cmd in zip(self.sim.bodies, self.sim.last_commands):
            if not (cmd.escape or body.escape_left_ms > 0):
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.12, 0.12, 0.12]),
                                body.pos + np.array([0.0, 0.0, 0.45]), np.eye(3).ravel(),
                                np.array([0.86, 0.3, 0.25, 0.9], np.float32))
            scn.ngeom += 1

    # ------------------------------------------------------------------- loop
    def status(self, wall_dt: float) -> None:
        sim = self.sim
        row = sim.rows[-1] if sim.rows else {}
        gf = " ".join(f"{r['gf_hz']:3.0f}" for r in sim.rows[-sim.n_agents:]) if sim.rows else ""
        esc = "ESCAPE" if any(c.escape for c in sim.last_commands) else "      "
        sys.stdout.write(f"\rt {sim.tick / 100:6.2f} s ({1 / max(wall_dt, 1e-6) / 100:4.2f}x real) "
                         f"| GF {gf} Hz | yaw {row.get('yaw', 0):+.2f} lift {row.get('lift', 0):+.2f} "
                         f"| hand {row.get('hand_dist', 0):4.2f} m | {esc}")
        sys.stdout.flush()

    def run(self, seconds: float | None) -> None:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data, key_callback=self.key,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            self.viewer = viewer
            viewer.cam.lookat[:] = [0.0, 0.0, 1.2]
            viewer.cam.distance = 7.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                t0 = time.perf_counter()
                if not self.paused:
                    self.sim.step()
                    self.mark_escapes()
                viewer.sync()
                self.status(time.perf_counter() - t0)
                if seconds is not None and self.sim.tick >= seconds * 100:
                    break
        print()
