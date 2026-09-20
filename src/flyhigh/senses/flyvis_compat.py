"""Import flyvis without its import-time side effects.

flyvis's package __init__ calls `torch.set_default_device(cuda if available else cpu)`
on import. That silently changes the default device for every other module in the
process (flyhigh.brain.lif mixes explicit-device and default-device tensors and breaks).
Every flyvis import in this project goes through `import_flyvis()`.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def preserve_torch_default_device():
    import torch

    prior = torch.get_default_device()
    try:
        yield
    finally:
        torch.set_default_device(prior)


def import_flyvis():
    """Return the `flyvis` module, with torch's default device left as it was."""
    with preserve_torch_default_device():
        import flyvis
    return flyvis
