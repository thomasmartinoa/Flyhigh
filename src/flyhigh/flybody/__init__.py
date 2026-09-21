"""M4: Janelia's flybody fruit fly, flying on its own wings under the brain's command.

TensorFlow (CPU) runs flybody's trained flight policy; torch runs the brain on the GPU. They
coexist in one process as long as TensorFlow never sees the GPU (`tensorflow-cpu`).
"""

import os

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
