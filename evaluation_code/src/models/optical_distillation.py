"""Independent attention-projection distillation for optical QKV blocks."""

from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn


def _predictor_core(predictor: nn.Module) -> nn.Module:
    return predictor.backbone if hasattr(predictor, "backbone") else predictor


def attention_proj_nmse(
    teacher_block: nn.Module,
    student_block: nn.Module,
    attention_input: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """Compare the tensor immediately after attn.proj and before proj_drop."""
    teacher_block.eval()
    with torch.no_grad():
        _, _, teacher_proj = teacher_block.attn(
            attention_input,
            mask=mask,
            return_proj_output=True,
        )
        teacher_proj = teacher_proj.detach()

    _, _, student_proj = student_block.attn(
        attention_input,
        mask=mask,
        return_proj_output=True,
    )
    denominator = teacher_proj.square().sum().clamp_min(eps)
    loss = (student_proj - teacher_proj).square().sum() / denominator
    return loss, teacher_proj, student_proj


def freeze_stage_one(student_predictor: nn.Module) -> None:
    """Freeze all electronic student parameters and leave optical parameters trainable."""
    core = _predictor_core(student_predictor)
    for name, parameter in core.named_parameters():
        parameter.requires_grad = "optical_qkv" in name


def freeze_teacher(teacher_predictor: nn.Module) -> None:
    core = _predictor_core(teacher_predictor)
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad = False


def optical_parameters(student_predictor: nn.Module) -> Iterable[nn.Parameter]:
    core = _predictor_core(student_predictor)
    return (
        parameter
        for name, parameter in core.named_parameters()
        if "optical_qkv" in name and parameter.requires_grad
    )


def independent_block_distillation_loss(
    teacher_predictor: nn.Module,
    student_predictor: nn.Module,
    ctxt,
    tgt,
    masks_ctxt,
    masks_tgt,
    block_indices: Sequence[int],
):
    """Compute all block losses from the same frozen-teacher block inputs."""
    teacher_core = _predictor_core(teacher_predictor)
    student_core = _predictor_core(student_predictor)
    freeze_teacher(teacher_core)
    freeze_stage_one(student_core)
    teacher_inputs, attention_mask = teacher_core.collect_attention_inputs(
        ctxt, tgt, masks_ctxt, masks_tgt
    )

    losses = {}
    total = None
    for block_index in block_indices:
        loss, _, _ = attention_proj_nmse(
            teacher_core.predictor_blocks[block_index],
            student_core.predictor_blocks[block_index],
            teacher_inputs[block_index],
            mask=attention_mask,
        )
        losses[int(block_index)] = loss
        total = loss if total is None else total + loss
    if total is None:
        raise ValueError("at least one block index is required")
    return total, losses


def independent_block_distillation_step(
    teacher_predictor: nn.Module,
    student_predictor: nn.Module,
    ctxt,
    tgt,
    masks_ctxt,
    masks_tgt,
    block_indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
):
    """Distill every selected block from teacher-generated, non-drifting inputs."""
    optimizer.zero_grad(set_to_none=True)
    total, losses = independent_block_distillation_loss(
        teacher_predictor,
        student_predictor,
        ctxt,
        tgt,
        masks_ctxt,
        masks_tgt,
        block_indices,
    )
    total.backward()
    optimizer.step()
    return total.detach(), {index: value.detach() for index, value in losses.items()}


def optical_state_dict(student_predictor: nn.Module) -> Dict[str, torch.Tensor]:
    core = _predictor_core(student_predictor)
    return {
        name: parameter.detach().cpu()
        for name, parameter in core.state_dict().items()
        if "optical_qkv" in name
    }


def build_optical_checkpoint(
    student_predictor: nn.Module,
    optical_config: dict,
    replace_layers: Sequence[int],
    teacher_checkpoint: str,
    distill_target: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    best_nmse: Optional[dict] = None,
    epoch: int = 0,
    metadata: Optional[dict] = None,
):
    return {
        "optical_state_dict": optical_state_dict(student_predictor),
        "optical_config": optical_config,
        "replace_layers": [int(index) for index in replace_layers],
        "teacher_checkpoint": teacher_checkpoint,
        "distill_target": distill_target,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "best_nmse": dict(best_nmse or {}),
        "metadata": dict(metadata or {}),
    }


def load_optical_checkpoint(student_predictor: nn.Module, checkpoint: dict) -> None:
    core = _predictor_core(student_predictor)
    expected = set(optical_state_dict(core))
    incoming = checkpoint.get("optical_state_dict", {})
    missing = sorted(expected.difference(incoming))
    unexpected = sorted(set(incoming).difference(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"optical checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    core.load_state_dict(incoming, strict=False)
