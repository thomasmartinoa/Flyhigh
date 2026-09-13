import torch

from flyhigh.senses.flyvis_compat import import_flyvis, preserve_torch_default_device


def test_import_flyvis_leaves_default_device_unchanged():
    before = torch.get_default_device()
    flyvis = import_flyvis()
    assert flyvis is not None
    assert torch.get_default_device() == before
    assert torch.zeros(1).device == before


def test_preserve_restores_after_explicit_change():
    before = torch.get_default_device()
    with preserve_torch_default_device():
        torch.set_default_device("cpu")
    assert torch.get_default_device() == before
