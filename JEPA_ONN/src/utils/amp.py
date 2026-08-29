import torch


def autocast_context(device, use_bfloat16):
    """Return an autocast context matching the use_bfloat16 flag."""
    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=bool(use_bfloat16),
    )
