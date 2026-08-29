"""Realtime optical Predictor distillation helpers."""

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
        parameter.requires_grad_(
            "optical_qkv" in name
        )


def freeze_teacher(teacher_predictor: nn.Module) -> None:
    core = _predictor_core(teacher_predictor)
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)


def optical_parameters(student_predictor: nn.Module) -> Iterable[nn.Parameter]:
    core = _predictor_core(student_predictor)
    return (
        parameter
        for name, parameter in core.named_parameters()
        if "optical_qkv" in name and parameter.requires_grad
    )



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
    target_node: Optional[str] = None,
    optimization_scope: str = "last_layer",
    cosine_loss_weight: float = 0.1,
):
    metadata = dict(metadata or {})
    if target_node is not None:
        metadata.setdefault("target_node", target_node)
    metadata.setdefault("optimization_scope", optimization_scope)
    metadata.setdefault("cosine_loss_weight", float(cosine_loss_weight))
    return {
        "format_version": 2,
        "mode": "realtime_last_node_distillation",
        "optical_state_dict": optical_state_dict(student_predictor),
        "optical_config": optical_config,
        "replace_layers": [int(index) for index in replace_layers],
        "teacher_checkpoint": teacher_checkpoint,
        "distill_target": distill_target,
        "target_node": target_node or distill_target,
        "optimization_scope": optimization_scope,
        "cosine_loss_weight": float(cosine_loss_weight),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "best_nmse": dict(best_nmse or {}),
        "metadata": metadata,
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
