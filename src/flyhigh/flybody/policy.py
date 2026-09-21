"""flybody's trained low-level flight controller, loaded from its TensorFlow SavedModel."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_PATH = Path("data/flybody/trained-fly-policies/flight")
STEERING_KEYS = ("walker/ref_displacement", "walker/ref_root_quat")


def _register_legacy_tfp_specs():
    """The policy was saved with a tfp that registered its distribution type specs under their
    module paths; tfp >= 0.24 registers them as `tfp.distributions.X`. Alias the old names."""
    import tensorflow_probability as tfp
    from tensorflow.python.framework import type_spec_registry as registry

    tfd = tfp.distributions
    _ = tfd.Independent, tfd.Normal, tfd.MultivariateNormalDiag  # triggers registration
    for spec, name in list(registry._TYPE_SPEC_TO_NAME.items()):
        if name.startswith("tfp.distributions.") and name.endswith("_ACTTypeSpec"):
            cls_name = name.rsplit(".", 1)[1][: -len("_ACTTypeSpec")]
            cls = getattr(tfd, cls_name, None)
            if cls is not None:
                registry._NAME_TO_TYPE_SPEC.setdefault(f"{cls.__module__}.{cls_name}_ACTTypeSpec", spec)


class FlightPolicy:
    def __init__(self, path=DEFAULT_PATH):
        import tensorflow as tf

        _register_legacy_tfp_specs()
        self.tf = tf
        self.model = tf.saved_model.load(str(path))
        sig = self.model.__call__.concrete_functions[0].structured_input_signature[0][0]
        self.keys = sorted(sig)
        self.shapes = {k: tuple(v.shape.as_list()[1:]) for k, v in sig.items()}
        self._fn = self.model.__call__

    def __call__(self, obs: dict) -> np.ndarray:
        """obs: the flight task's observation dict (numpy). Returns the action mean (12,)."""
        return self.batch([obs])[0]

    def batch(self, observations: list[dict]) -> np.ndarray:
        """Several flies' observation dicts in one call. Returns (n, 12)."""
        tf = self.tf
        batch = {k: tf.convert_to_tensor(np.stack([np.asarray(o[k], dtype=np.float32) for o in observations]))
                 for k in self.keys}
        return self._fn(batch).mean().numpy()
