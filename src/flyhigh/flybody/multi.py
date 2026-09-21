"""Several flybody flies in one world: a composer task built from flybody's parts.

flybody's own tasks hold one walker and keep the flight configuration (wing actuator gains,
the fluid model, retracted legs), the wing-beat pattern generator and the action mapping in
the task. Here each fly gets its own copy of that configuration (`configure_for_flight`), its
own pattern generator and its own carrot (`Carrot`, the steering of `task.py`), and the task
maps a concatenated action vector onto each fly's actuators itself (flybody's
`apply_action` writes the whole control vector with walker-local indices, which two walkers
cannot share). The first fly is named `walker` because a few flybody lookups hard-code it.
"""

from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from flybody.fruitfly import fruitfly
from flybody.quaternions import get_dquat_local, mult_quat
from flybody.tasks.constants import _FLY_CONTROL_TIMESTEP, _FLY_PHYSICS_TIMESTEP, _WING_PARAMS
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import com2root
from flybody.utils import any_substr_in_str

from flyhigh.flybody.task import BODY_PITCH_DEG, SteerParams, yaw_quat
from flyhigh.motor.command import MotorCommand
from flyhigh.world.scene import FACE_FOVY, FACES

FUTURE_STEPS = 5


def fly_name(i: int) -> str:
    return "walker" if i == 0 else f"walker{i}"


def configure_for_flight(walker, arena) -> tuple[list, list, list]:
    """What flybody's `Flying.__init__` does to one walker. Returns (wing joints, leg joints, leg springrefs)."""
    for i, dclass in enumerate(["yaw", "roll", "pitch"]):
        walker.mjcf_model.find("default", dclass).general.gainprm[0] = _WING_PARAMS["gainprm"][i]
    for geom in walker.mjcf_model.find_all("geom"):
        if "fluid" in geom.name:
            geom.fluidshape = "ellipsoid"
            geom.fluidcoef = _WING_PARAMS["fluidcoef"]
    wing_joints = [walker.mjcf_model.find("joint", f"wing_{axis}_{side}") for side in ("left", "right")
                   for axis in ("yaw", "roll", "pitch")]
    wing_default = walker.mjcf_model.find("default", "wing").joint
    wing_default.stiffness = _WING_PARAMS["stiffness"]
    wing_default.damping = _WING_PARAMS["damping"]
    contact = walker.mjcf_model.contact
    for body in walker.mjcf_model.find_all("body"):
        if any_substr_in_str(["coxa", "femur", "tibia", "tarsus", "claw"], body.name):
            for wing in ("wing_left", "wing_right"):
                contact.add("exclude", name=f"{body.name}_{wing}", body1=body.name, body2=wing)
    leg_joints, springrefs = [], []
    for joint in walker.mjcf_model.find_all("joint"):
        if any_substr_in_str(["coxa", "femur", "tibia", "tarsus"], joint.name):
            leg_joints.append(joint)
            springrefs.append(joint.springref or joint.dclass.joint.springref or 0.0)
    for sensor in walker.observables.vestibular + walker.observables.proprioception:
        sensor.enabled = True
    walker.observables.thorax_height.enabled = False
    return wing_joints, leg_joints, springrefs


class Carrot:
    """The steering of `task.SteeredFlight`, as an object per fly: a reference point that moves
    at the commanded velocity and a heading that turns at the commanded rate, both kept within
    reach of the fly, so that the flight policy always sees a small gap to close."""

    def __init__(self, params: SteerParams):
        self.p = params
        self.dt = _FLY_CONTROL_TIMESTEP
        q0, _ = constant_speed_trajectory(n_steps=2, speed=0.0, init_pos=(0, 0, 0), init_heading=0.0,
                                          body_rot_angle_y=-BODY_PITCH_DEG, control_timestep=self.dt)
        self.q_flight = q0[0, 3:7]
        self.v_local = np.zeros(3)
        self.omega = 0.0
        self.escape_left_ms = 0.0
        self.ref_pos = np.zeros(3)
        self.ref_yaw = 0.0

    def start_pose(self, start, yaw_deg):
        """(root position, quaternion) for a hover at `start` facing `yaw_deg`; also resets the carrot."""
        q = mult_quat(yaw_quat(np.radians(yaw_deg)), self.q_flight)
        root = com2root(np.array(start, dtype=float)[None], q[None])[0]
        self.ref_pos = root.copy()
        self.ref_yaw = np.radians(yaw_deg)
        self.escape_left_ms = 0.0
        return root, q

    def set_command(self, cmd: MotorCommand, dt_ms: float) -> None:
        p = self.p
        if cmd.escape and self.escape_left_ms <= 0:
            self.escape_left_ms = p.escape_ms
        escaping = self.escape_left_ms > 0
        self.escape_left_ms = max(0.0, self.escape_left_ms - dt_ms)
        if escaping:
            self.v_local = np.array([-p.escape_back, 0.0, p.escape_up])
            self.omega = 0.0
        else:
            self.v_local = np.array([p.v_max * cmd.forward, 0.0, p.vz_max * cmd.lift])
            self.omega = -np.radians(p.w_max_deg) * cmd.yaw

    @staticmethod
    def heading(physics, walker) -> float:
        w, x, y, z = np.array(walker.get_pose(physics)[1])
        return float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))

    def v_world(self):
        return np.array([np.cos(self.ref_yaw) * self.v_local[0], np.sin(self.ref_yaw) * self.v_local[0], self.v_local[2]])

    def advance(self, physics, walker) -> None:
        p = self.p
        fly_pos = np.array(walker.get_pose(physics)[0])
        self.ref_pos = self.ref_pos + self.v_world() * self.dt
        lead = self.ref_pos - fly_pos
        dist = np.linalg.norm(lead)
        if dist > p.lead_max:
            self.ref_pos = fly_pos + lead * (p.lead_max / dist)
        self.ref_yaw += self.omega * self.dt
        psi = self.heading(physics, walker)
        gap = (self.ref_yaw - psi + np.pi) % (2 * np.pi) - np.pi
        limit = np.radians(p.lead_max_deg)
        if abs(gap) > limit:
            self.ref_yaw = psi + np.sign(gap) * limit

    def steering(self, physics, walker):
        k = np.arange(FUTURE_STEPS + 1)[:, None] * self.dt
        fly_pos = np.array(walker.get_pose(physics)[0])
        disp = walker.transform_vec_to_egocentric_frame(physics, self.ref_pos + k * self.v_world() - fly_pos)
        q_fly = np.array(walker.get_pose(physics)[1])
        quats = np.stack([get_dquat_local(q_fly, mult_quat(yaw_quat(self.ref_yaw + self.omega * kk), self.q_flight))
                          for kk in k[:, 0]])
        return disp.astype(np.float32), quats.astype(np.float32)


class Fly:
    """One fly of the task: walker, wing-beat generator, carrot, and its actuator bookkeeping."""

    def __init__(self, i: int, arena, wpg_path, params: SteerParams, start, start_yaw_deg: float):
        self.i, self.name = i, fly_name(i)
        self.walker = fruitfly.FruitFly(name=self.name, use_legs=False, use_wings=True, use_mouth=False,
                                        use_antennae=False, force_actuators=False, joint_filter=0.0,
                                        body_pitch_angle=BODY_PITCH_DEG, physics_timestep=_FLY_PHYSICS_TIMESTEP,
                                        control_timestep=_FLY_CONTROL_TIMESTEP, num_user_actions=1)
        vis = self.walker.mjcf_model.visual  # before attaching: attached trees must agree on these
        getattr(vis, "global").offwidth, getattr(vis, "global").offheight = 640, 480
        vis.quality.shadowsize = 1024
        arena.add_free_entity(self.walker)
        self.wing_joints, self.leg_joints, self.leg_springrefs = configure_for_flight(self.walker, arena)
        self.wbpg = WingBeatPatternGenerator(base_pattern_path=str(wpg_path))
        self.carrot = Carrot(params)
        self.start, self.start_yaw_deg = start, start_yaw_deg
        self.actuators = self.walker.mjcf_model.find_all("actuator")
        w = self.walker
        self.wing_idx = w._action_indices["wings"]
        self.user_idx = w._action_indices["user"][0]
        head = w.mjcf_model.find("body", "head")
        for face, axes in FACES.items():
            head.add("camera", name=f"cube_{face}", pos=(0, 0, 0), xyaxes=[float(v) for v in axes.split()], fovy=FACE_FOVY)
        w.observables.add_observable("ref_displacement", observable.Generic(lambda ph: self.carrot.steering(ph, w)[0]))
        w.observables.add_observable("ref_root_quat", observable.Generic(lambda ph: self.carrot.steering(ph, w)[1]))

    def apply(self, physics, action) -> None:
        """flybody's action mapping, written onto this walker's actuators only."""
        w = self.walker
        ctrl = np.zeros(len(self.actuators))
        for key, indices in w._action_indices.items():
            if w._ctrl_indices[key] and indices:
                ctrl[w._ctrl_indices[key]] = action[indices]
        physics.bind(self.actuators).ctrl = ctrl


class MultiFlyTask(composer.Task):
    def __init__(self, arena, wpg_path, n: int = 2, params: SteerParams | None = None, starts=None):
        self._arena = arena
        p = params or SteerParams()
        starts = starts or [(-4.0 + 8.0 * i, (-1) ** i * 1.5, 7.0, 180.0 * (i % 2)) for i in range(n)]
        self.flies = [Fly(i, arena, wpg_path, p, s[:3], s[3] if len(s) > 3 else 0.0) for i, s in enumerate(starts[:n])]
        for geom in arena.ground_geoms:
            geom.contype = 0
            geom.conaffinity = 0
        self.set_timesteps(physics_timestep=_FLY_PHYSICS_TIMESTEP, control_timestep=_FLY_CONTROL_TIMESTEP)

    @property
    def root_entity(self):
        return self._arena

    def action_spec(self, physics):
        specs = [f.walker.get_action_spec(physics) for f in self.flies]
        from dm_env import specs as dm_specs
        return dm_specs.BoundedArray(shape=(sum(s.shape[0] for s in specs),), dtype=np.float32,
                                     minimum=np.concatenate([s.minimum for s in specs]),
                                     maximum=np.concatenate([s.maximum for s in specs]), name="flies")

    def slices(self, physics):
        out, k = [], 0
        for f in self.flies:
            n = f.walker.get_action_spec(physics).shape[0]
            out.append(slice(k, k + n)); k += n
        return out

    def initialize_episode(self, physics, random_state):
        for f in self.flies:
            root, q = f.carrot.start_pose(f.start, f.start_yaw_deg)
            f.walker.set_pose(physics, root, q)
            wing_qpos, wing_qvel = f.wbpg.reset(initial_phase=random_state.uniform(), return_qvel=True)
            physics.bind(f.wing_joints).qpos = wing_qpos
            physics.bind(f.wing_joints).qvel = wing_qvel
            if f.leg_joints:
                physics.bind(f.leg_joints).qpos = f.leg_springrefs
        self._slices = self.slices(physics)

    def before_step(self, physics, action, random_state):
        for f, sl in zip(self.flies, self._slices):
            a = np.array(action[sl], dtype=float)
            f.carrot.advance(physics, f.walker)
            base, rel = f.wbpg.base_beat_freq, f.wbpg.rel_freq_range
            ctrl = f.wbpg.step(ctrl_freq=base * (1 + rel * a[f.user_idx]))
            a[f.wing_idx] += ctrl - physics.bind(f.wing_joints).qpos
            f.apply(physics, a)

    def get_reward(self, physics):
        return 0.0

    def should_terminate_episode(self, physics):
        return False

