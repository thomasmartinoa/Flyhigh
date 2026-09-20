"""The box: a MuJoCo room, flying bodies, a moving hand, and cubemap eyes that fill a PanoramicFrame."""

import os

# MuJoCo needs a GL backend chosen before its first import. GLFW works on this desktop; a
# headless machine sets MUJOCO_GL=egl (or osmesa) itself -- setdefault never overrides.
os.environ.setdefault("MUJOCO_GL", "glfw")
