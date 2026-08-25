"""Epoch-based optical QKV distillation with a fixed Train split."""

import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import yaml

from src.models import predictor as vit_pred
from src.masks.utils import apply_masks
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.models.fsonn import OpticalQKVConfig
from src.models.optical_distillation import (
    build_optical_checkpoint,
    independent_block_distillation_loss,
    independent_block_distillation_step,
    optical_parameters,
)
from evals.intuitive_physics.data_manager import init_data
from evals.intuitive_physics.eval import init_model
from evals.intuitive_physics.optical_split import load_or_create_video_split
from evals.intuitive_physics.utils import get_dataset_paths, get_time_masks
from src.utils.transforms import make_transforms


def _configure_logging(log_path):
    log_path = os.path.abspath(log_path)
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    logger = logging.getLogger("fsonn.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _sync_for_timing(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _last_checkpoint_path(output_path):
    output_path = Path(output_path)
    suffix = output_path.suffix or ".pt"
    return output_path.with_name(f"{output_path.stem}.last{suffix}")


def _save_checkpoint(checkpoint, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def _resolve_run_outputs(output_arg, project_root=None, timestamp=None):
    output_arg = Path(output_arg)
    stem = output_arg.stem if output_arg.suffix else output_arg.name
    if not stem or stem in {".", ".."}:
        raise ValueError("output must provide a non-empty base name")
    suffix = output_arg.suffix or ".pt"
    code_root = Path(PROJECT_ROOT if project_root is None else project_root)
    output_root = code_root.parent / "output"
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{stem}_{run_timestamp}"
    collision_index = 1
    while run_dir.exists():
        run_dir = output_root / f"{stem}_{run_timestamp}_{collision_index:02d}"
        collision_index += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / f"{stem}{suffix}"
    return {
        "run_dir": run_dir,
        "output": output_path,
        "last_output": _last_checkpoint_path(output_path),
        "final_output": output_path.with_name(
            f"{output_path.stem}.final{output_path.suffix}"
        ),
        "log": Path(f"{output_path}.log"),
        "split_manifest": Path(f"{output_path}.split.json"),
    }


def _path_in_run_dir(run_dir, requested_path, default_path):
    if requested_path is None:
        return default_path
    name = Path(requested_path).name
    if not name or name in {".", ".."}:
        raise ValueError("output file name must not be empty")
    return Path(run_dir) / name


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _apply_cli_overrides(config, batch_size=None):
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        config.setdefault("data", {})["batch_size"] = int(batch_size)
    return config


def _compute_jepa_loss(predictions, targets, loss_exp=1.0):
    if len(predictions) != len(targets) or not predictions:
        raise ValueError('predictions and targets must be non-empty and have equal length')
    losses = []
    exponent = float(loss_exp)
    if exponent <= 0:
        raise ValueError('loss_exp must be positive')
    for prediction, target in zip(predictions, targets):
        losses.append(torch.mean(torch.abs(prediction - target) ** exponent) / exponent)
    return torch.stack(losses).mean()


def _prepare_models(args_eval, device):
    optical_cfg = args_eval.get("optical_qkv", {})
    if optical_cfg.get("qkv_backend") != "fsonn_tdm":
        raise ValueError("train_optical.py requires optical_qkv.qkv_backend=fsonn_tdm")
    optical_config = OpticalQKVConfig.from_mapping(optical_cfg)
    replace_layers = optical_cfg.get("replace_layers", "all")
    if replace_layers == "all":
        replace_layers = list(range(args_eval["pretrain"].get("pred_depth", 12)))

    encoder, target_encoder, electronic_predictor = init_model(
        crop_size=args_eval["data"].get("resolution", 224),
        device=device,
        pretrained=os.path.join(
            args_eval["pretrain"]["folder"],
            args_eval["pretrain"]["checkpoint"],
        ),
        model_name=args_eval["pretrain"]["model_name"],
        patch_size=args_eval["pretrain"].get("patch_size", 16),
        tubelet_size=args_eval["pretrain"].get("tubelet_size", 2),
        frames_per_clip=args_eval["data"].get("frames_per_clip", 16),
        is_causal=args_eval["pretrain"].get("is_causal", False),
        pred_is_causal=args_eval["pretrain"].get("pred_is_causal", False),
        pred_depth=args_eval["pretrain"].get("pred_depth", 12),
        uniform_power=args_eval["pretrain"].get("uniform_power", False),
        enc_checkpoint_key=args_eval["pretrain"].get("enc_checkpoint_key", "encoder"),
        pred_checkpoint_key=args_eval["pretrain"].get("pred_checkpoint_key", "predictor"),
        use_SiLU=args_eval["pretrain"].get("use_silu", False),
        wide_SiLU=args_eval["pretrain"].get("wide_silu", True),
        use_sdpa=args_eval["pretrain"].get("use_sdpa", True),
        is_mae=False,
        optical_qkv={},
    )
    teacher_predictor = copy.deepcopy(electronic_predictor)
    student_predictor = copy.deepcopy(electronic_predictor)
    vit_pred.install_optical_qkv(
        student_predictor,
        optical_config=optical_config,
        replace_layers=replace_layers,
    )

    for module in (encoder, target_encoder, teacher_predictor):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return (
        encoder,
        target_encoder,
        teacher_predictor,
        student_predictor,
        optical_config,
        replace_layers,
    )


def _get_training_sampling_rate(args_eval):
    data_cfg = args_eval["data"]
    sampling_rate = data_cfg.get("sampling_rate")
    if sampling_rate is None:
        sampling_rate = args_eval.get("pretrain", {}).get("sampling_rate", 4)
    if isinstance(sampling_rate, list):
        sampling_rate = sampling_rate[0]
    sampling_rate = int(sampling_rate)
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    return sampling_rate


def _make_loader(args_eval, video_ids, deterministic=False, world_size=1, rank=0):
    data_cfg = args_eval["data"]
    frames_per_clip = int(data_cfg.get("frames_per_clip", 16))
    if frames_per_clip != 16:
        raise ValueError("optical distillation training requires frames_per_clip=16")
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg.get("resolution", 224),
    )
    data_name = "IntPhys-train"
    return init_data(
        batch_size=data_cfg.get("batch_size", 1),
        transform=transform,
        data=data_name,
        collator=None,
        pin_mem=True,
        num_workers=0,
        world_size=world_size,
        rank=rank,
        root_path=get_dataset_paths([data_name])[0],
        clip_len=frames_per_clip,
        frame_sample_rate=_get_training_sampling_rate(args_eval),
        deterministic=deterministic,
        log_dir=None,
        video_ids=video_ids,
        train_format=True,
    )[0]


def _extract_train_clips(batch, device):
    clips = batch[0]
    if clips.ndim == 6:
        if clips.shape[1] != 1:
            raise ValueError(
                "Train dataset must provide exactly one scene sequence per video"
            )
        clips = clips[:, 0]
    if clips.ndim != 5:
        raise ValueError(f"unexpected Train clip shape: {tuple(clips.shape)}")
    return clips.to(device)


def _prepare_features(batch, args_eval, encoder, target_encoder, device):
    clips = _extract_train_clips(batch, device)
    frames_per_clip = args_eval["data"].get("frames_per_clip", 16)
    if clips.ndim != 5 or clips.shape[1] != 3 or clips.shape[2] != frames_per_clip:
        raise ValueError(
            "optical distillation expects [B,3,16,H,W], "
            f"got {tuple(clips.shape)}"
        )
    pieces = clips.contiguous()
    batch_size = pieces.shape[0]

    context_length = args_eval["data"].get("context_lengths", [2])[0]
    masks_ctxt, masks_tgt, full_mask = get_time_masks(
        context_length,
        spatial_size=(
            args_eval["pretrain"].get("patch_size", 16),
            args_eval["pretrain"].get("patch_size", 16),
        ),
        temporal_dim=frames_per_clip,
        as_bool=False,
    )
    masks_ctxt = [masks_ctxt.unsqueeze(0).to(device).repeat(batch_size, 1)]
    masks_tgt = [masks_tgt.unsqueeze(0).to(device).repeat(batch_size, 1)]
    full_mask = [full_mask.unsqueeze(0).to(device).repeat(batch_size, 1)]

    with torch.no_grad():
        targets = target_encoder(pieces, full_mask)[0]
        targets = apply_masks(targets, masks_tgt, concat=False)[0]
        context = encoder(pieces, masks_ctxt)[0]
    return context, targets, masks_ctxt, masks_tgt, pieces.shape[0]


def _run_epoch(
    loader,
    args_eval,
    encoder,
    target_encoder,
    teacher_predictor,
    student_predictor,
    replace_layers,
    optimizer,
    device,
    logger,
    epoch,
    training,
    max_steps=None,
):
    stage = "train" if training else "val"
    if training:
        student_predictor.train()
    else:
        student_predictor.eval()
        teacher_predictor.eval()

    total_sum = 0.0
    block_sums = {int(index): 0.0 for index in replace_layers}
    batches = 0
    stage_started = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if max_steps is not None and batch_index >= max_steps:
            break
        step_started = time.perf_counter()
        feature_started = time.perf_counter()
        context, targets, masks_ctxt, masks_tgt, batch_size = _prepare_features(
            batch, args_eval, encoder, target_encoder, device
        )
        _sync_for_timing(device)
        feature_time = time.perf_counter() - feature_started

        distill_started = time.perf_counter()
        if training:
            total_loss, block_losses = independent_block_distillation_step(
                teacher_predictor,
                student_predictor,
                context,
                targets,
                masks_ctxt,
                masks_tgt,
                replace_layers,
                optimizer,
            )
        else:
            with torch.no_grad():
                total_loss, block_losses = independent_block_distillation_loss(
                    teacher_predictor,
                    student_predictor,
                    context,
                    targets,
                    masks_ctxt,
                    masks_tgt,
                    replace_layers,
                )
            total_loss = total_loss.detach()
            block_losses = {
                index: value.detach() for index, value in block_losses.items()
            }
        _sync_for_timing(device)
        distill_time = time.perf_counter() - distill_started

        total_value = float(total_loss)
        total_sum += total_value
        for index, value in block_losses.items():
            block_sums[int(index)] += float(value)
        batches += 1
        logger.info(
            "epoch=%d stage=%s step=%d batch_size=%d total_nmse=%.6f "
            "block_nmse={%s} feature_time_s=%.3f distill_time_s=%.3f "
            "step_time_s=%.3f",
            epoch,
            stage,
            batches,
            batch_size,
            total_value,
            ",".join(
                f"{index}:{float(value):.6f}"
                for index, value in block_losses.items()
            ),
            feature_time,
            distill_time,
            time.perf_counter() - step_started,
        )

    if batches == 0:
        raise RuntimeError(f"{stage} loader produced no batches")
    metrics = {
        "total_nmse": total_sum / batches,
        "block_nmse": {
            index: value / batches for index, value in block_sums.items()
        },
        "batches": batches,
        "elapsed_s": time.perf_counter() - stage_started,
    }
    logger.info(
        "epoch=%d stage=%s_done batches=%d total_nmse=%.6f block_nmse={%s} "
        "stage_time_s=%.3f",
        epoch,
        stage,
        batches,
        metrics["total_nmse"],
        ",".join(
            f"{index}:{value:.6f}"
            for index, value in metrics["block_nmse"].items()
        ),
        metrics["elapsed_s"],
    )
    return metrics


def _default_jepa_mask_config():
    return [
        {
            "aspect_ratio": [0.75, 1.5],
            "num_blocks": 8,
            "spatial_scale": [0.15, 0.15],
            "temporal_scale": [1.0, 1.0],
            "max_temporal_keep": 1.0,
            "max_keep": None,
        },
        {
            "aspect_ratio": [0.75, 1.5],
            "num_blocks": 2,
            "spatial_scale": [0.7, 0.7],
            "temporal_scale": [1.0, 1.0],
            "max_temporal_keep": 1.0,
            "max_keep": None,
        },
    ]


def _make_jepa_mask_collator(args_eval):
    data_cfg = args_eval["data"]
    pretrain_cfg = args_eval["pretrain"]
    return MB3DMaskCollator(
        cfgs_mask=args_eval.get("mask") or _default_jepa_mask_config(),
        crop_size=(
            int(data_cfg.get("resolution", 224)),
            int(data_cfg.get("resolution", 224)),
        ),
        num_frames=int(data_cfg.get("frames_per_clip", 16)),
        patch_size=(
            int(pretrain_cfg.get("patch_size", 16)),
            int(pretrain_cfg.get("patch_size", 16)),
        ),
        tubelet_size=int(pretrain_cfg.get("tubelet_size", 2)),
    )


def _make_jepa_loader(
    args_eval,
    video_ids,
    deterministic=False,
    collator=None,
):
    data_cfg = args_eval["data"]
    frames_per_clip = int(data_cfg.get("frames_per_clip", 16))
    if frames_per_clip != 16:
        raise ValueError("end_to_end_jepa requires frames_per_clip=16")
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg.get("resolution", 224),
    )
    return init_data(
        batch_size=data_cfg.get("batch_size", 1),
        transform=transform,
        data="IntPhys-train",
        collator=collator,
        pin_mem=True,
        num_workers=0,
        world_size=1,
        rank=0,
        root_path=get_dataset_paths(["IntPhys-train"])[0],
        clip_len=frames_per_clip,
        frame_sample_rate=_get_training_sampling_rate(args_eval),
        deterministic=deterministic,
        log_dir=None,
        video_ids=video_ids,
        train_format=True,
    )[0]


def _extract_jepa_clips(batch, device):
    payload = batch[0]
    if isinstance(payload, (tuple, list)):
        clips = payload[0]
    else:
        clips = payload
    if clips.ndim == 6:
        if clips.shape[1] != 1:
            raise ValueError(
                "Train dataset must provide exactly one clip per video"
            )
        clips = clips[:, 0]
    if clips.ndim != 5 or clips.shape[1] != 3 or clips.shape[2] != 16:
        raise ValueError(
            "end_to_end_jepa expects [B,3,16,H,W], "
            f"got {tuple(clips.shape)}"
        )
    return clips.to(device)


def _move_masks(masks, device):
    return [mask.to(device=device, dtype=torch.long) for mask in masks]


def _prepare_jepa_batch(batch, args_eval, encoder, target_encoder, device):
    clips = _extract_jepa_clips(batch, device)
    if len(batch) == 3:
        masks_ctxt = _move_masks(batch[1], device)
        masks_tgt = _move_masks(batch[2], device)
    else:
        context_length = args_eval["data"].get("context_lengths", [2])[0]
        masks_ctxt, masks_tgt, full_mask = get_time_masks(
            context_length,
            spatial_size=(
                args_eval["pretrain"].get("patch_size", 16),
                args_eval["pretrain"].get("patch_size", 16),
            ),
            temporal_dim=16,
            as_bool=False,
        )
        batch_size = clips.shape[0]
        masks_ctxt = [masks_ctxt.unsqueeze(0).repeat(batch_size, 1).to(device)]
        masks_tgt = [masks_tgt.unsqueeze(0).repeat(batch_size, 1).to(device)]
    num_tokens = (16 // int(args_eval["pretrain"].get("tubelet_size", 2))) * (
        int(args_eval["data"].get("resolution", 224))
        // int(args_eval["pretrain"].get("patch_size", 16))
    ) ** 2
    full_mask = [
        torch.arange(num_tokens, device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(clips.shape[0], 1)
    ]
    with torch.no_grad():
        target = target_encoder(clips, full_mask)[0]
        target = F.layer_norm(target, (target.shape[-1],))
        targets = apply_masks(target, masks_tgt, concat=False)
        context = encoder(clips, masks_ctxt)
    return clips, context, targets, masks_ctxt, masks_tgt


def _prepare_end_to_end_models(args_eval, device):
    optical_cfg = args_eval.get("optical_qkv", {})
    if optical_cfg.get("qkv_backend") != "fsonn_tdm":
        raise ValueError("end_to_end_jepa requires optical_qkv.qkv_backend=fsonn_tdm")
    optical_config = OpticalQKVConfig.from_mapping(optical_cfg)
    replace_layers = optical_cfg.get("replace_layers", "all")
    if replace_layers == "all":
        replace_layers = list(range(args_eval["pretrain"].get("pred_depth", 12)))
    encoder, target_encoder, predictor = init_model(
        crop_size=args_eval["data"].get("resolution", 224),
        device=device,
        pretrained=os.path.join(
            args_eval["pretrain"]["folder"],
            args_eval["pretrain"]["checkpoint"],
        ),
        model_name=args_eval["pretrain"]["model_name"],
        patch_size=args_eval["pretrain"].get("patch_size", 16),
        tubelet_size=args_eval["pretrain"].get("tubelet_size", 2),
        frames_per_clip=args_eval["data"].get("frames_per_clip", 16),
        is_causal=args_eval["pretrain"].get("is_causal", False),
        pred_is_causal=args_eval["pretrain"].get("pred_is_causal", False),
        pred_depth=args_eval["pretrain"].get("pred_depth", 12),
        uniform_power=args_eval["pretrain"].get("uniform_power", False),
        enc_checkpoint_key=args_eval["pretrain"].get("enc_checkpoint_key", "encoder"),
        pred_checkpoint_key=args_eval["pretrain"].get("pred_checkpoint_key", "predictor"),
        use_SiLU=args_eval["pretrain"].get("use_silu", False),
        wide_SiLU=args_eval["pretrain"].get("wide_silu", True),
        use_sdpa=args_eval["pretrain"].get("use_sdpa", True),
        is_mae=False,
        optical_qkv={},
    )
    vit_pred.install_optical_qkv(
        predictor,
        optical_config=optical_config,
        replace_layers=replace_layers,
    )
    for module in (encoder, target_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    predictor.train()
    for parameter in predictor.parameters():
        parameter.requires_grad = True
    return encoder, target_encoder, predictor, optical_config, replace_layers


def _run_jepa_epoch(
    loader,
    args_eval,
    encoder,
    target_encoder,
    predictor,
    optimizer,
    device,
    logger,
    epoch,
    training,
    max_steps=None,
    clip_grad=10.0,
):
    stage = "train" if training else "val"
    predictor.train(training)
    encoder.eval()
    target_encoder.eval()
    total_loss = 0.0
    batches = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_steps is not None and batch_index >= max_steps:
            break
        step_started = time.perf_counter()
        feature_started = time.perf_counter()
        clips, context, targets, masks_ctxt, masks_tgt = _prepare_jepa_batch(
            batch, args_eval, encoder, target_encoder, device
        )
        _sync_for_timing(device)
        feature_time = time.perf_counter() - feature_started
        predictor_started = time.perf_counter()
        if training:
            optimizer.zero_grad(set_to_none=True)
            predictions = predictor(context, targets, masks_ctxt, masks_tgt)
            loss = _compute_jepa_loss(
                predictions,
                targets,
                loss_exp=args_eval.get("loss", {}).get("loss_exp", 1.0),
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), float(clip_grad)
            )
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = predictor(context, targets, masks_ctxt, masks_tgt)
                loss = _compute_jepa_loss(
                    predictions,
                    targets,
                    loss_exp=args_eval.get("loss", {}).get("loss_exp", 1.0),
                )
            grad_norm = torch.tensor(0.0, device=device)
        _sync_for_timing(device)
        predictor_time = time.perf_counter() - predictor_started
        loss_value = float(loss.detach())
        total_loss += loss_value
        batches += 1
        gpu_memory = (
            torch.cuda.memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        logger.info(
            "epoch=%d stage=%s step=%d batch_size=%d jepa_loss=%.6f "
            "feature_time_s=%.3f predictor_time_s=%.3f step_time_s=%.3f "
            "grad_norm=%.3f gpu_allocated_mib=%.0f",
            epoch,
            stage,
            batches,
            clips.shape[0],
            loss_value,
            feature_time,
            predictor_time,
            time.perf_counter() - step_started,
            float(grad_norm),
            gpu_memory,
        )
    if batches == 0:
        raise RuntimeError(f"{stage} loader produced no batches")
    mean_loss = total_loss / batches
    metrics = {
        "jepa_loss": mean_loss,
        "batches": batches,
        "elapsed_s": time.perf_counter() - started,
    }
    logger.info(
        "epoch=%d stage=%s_done batches=%d jepa_loss=%.6f stage_time_s=%.3f",
        epoch,
        stage,
        batches,
        mean_loss,
        metrics["elapsed_s"],
    )
    return metrics


def _end_to_end_checkpoint(
    predictor,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val_loss,
    split,
    split_manifest,
    args_eval,
    kind,
    best_epoch=None,
):
    predictor_state = {
        key: value.detach().cpu().clone()
        for key, value in predictor.state_dict().items()
    }
    return {
        "format_version": 1,
        "mode": "end_to_end_jepa",
        "checkpoint_kind": kind,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_epoch": int(epoch if kind == "best" else (best_epoch or epoch)),
        "best_val_jepa_loss": float(best_val_loss),
        "predictor": predictor_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "optical_qkv": copy.deepcopy(args_eval.get("optical_qkv", {})),
        "replace_layers": copy.deepcopy(args_eval.get("optical_qkv", {}).get("replace_layers", "all")),
        "pretrain_checkpoint": os.path.join(
            args_eval["pretrain"]["folder"], args_eval["pretrain"]["checkpoint"]
        ),
        "training_config": copy.deepcopy(args_eval.get("training", {})),
        "data_split": copy.deepcopy(split),
        "split_manifest": os.path.abspath(split_manifest),
    }


def _load_end_to_end_checkpoint(
    checkpoint_path,
    predictor,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "end_to_end_jepa":
        raise ValueError(
            "end_to_end_jepa can resume only from an end_to_end_jepa checkpoint"
        )
    if "predictor" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError("end_to_end_jepa checkpoint is missing full Predictor state")
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


def _run_final_intphys_evaluation(args_eval, best_path, run_dir, logger):
    from evals.intuitive_physics import eval as dev_eval
    from evals.intphys_test import eval as test_eval
    for label, module, folder_name in (
        ("dev", dev_eval, "intphys_dev"),
        ("test", test_eval, "intphys_test"),
    ):
        evaluation_cfg = copy.deepcopy(args_eval)
        evaluation_cfg["predictor_checkpoint"] = os.path.abspath(best_path)
        evaluation_cfg["output_dir"] = os.path.join(run_dir, folder_name)
        logger.info("final_evaluation_start split=%s checkpoint=%s", label, best_path)
        module.main(evaluation_cfg)
        logger.info("final_evaluation_done split=%s", label)


def run(
    args_eval,
    output_path,
    block=None,
    max_steps=None,
    learning_rate=1e-4,
    log_path=None,
    epochs=None,
    split_manifest=None,
    last_output=None,
):
    if log_path is None:
        log_path = f"{output_path}.log"
    logger = _configure_logging(log_path)
    run_started = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_cfg = args_eval.get("training", {})
    split_cfg = args_eval.get("data_split", {})
    epochs = int(training_cfg.get("epochs", 1) if epochs is None else epochs)
    if max_steps is None:
        max_steps = training_cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if block is not None:
        logger.info("block_arg=%s ignored_for_training_source=all_train_videos", block)

    train_root = get_dataset_paths(["IntPhys-train"])[0]
    if split_manifest is None:
        split_manifest = training_cfg.get("split_manifest")
    if split_manifest is None:
        split_manifest = f"{output_path}.split.json"
    if last_output is None:
        last_output = training_cfg.get("last_output")
    if last_output is None:
        last_output = _last_checkpoint_path(output_path)
    split = load_or_create_video_split(
        train_root,
        split_manifest,
        num_train_videos=int(split_cfg.get("num_train_videos", 1500)),
        num_val_videos=int(split_cfg.get("num_val_videos", 300)),
        split_seed=int(split_cfg.get("split_seed", 42)),
    )
    logger.info(
        "run_start epochs=%d max_steps=%s learning_rate=%g output=%s log=%s "
        "device=%s train_videos=%d val_videos=%d split_manifest=%s last_output=%s",
        epochs,
        max_steps,
        learning_rate,
        os.path.abspath(output_path),
        os.path.abspath(log_path),
        device,
        len(split["train_video_ids"]),
        len(split["val_video_ids"]),
        os.path.abspath(split_manifest),
        os.path.abspath(last_output),
    )

    model_started = time.perf_counter()
    logger.info("model_prepare_start")
    (
        encoder,
        target_encoder,
        teacher_predictor,
        student_predictor,
        optical_config,
        replace_layers,
    ) = _prepare_models(args_eval, device)
    _sync_for_timing(device)
    logger.info("model_prepare_done elapsed_s=%.3f", time.perf_counter() - model_started)

    optimizer = torch.optim.AdamW(
        optical_parameters(student_predictor),
        lr=learning_rate,
    )
    loader_started = time.perf_counter()
    train_loader = _make_loader(
        args_eval, split["train_video_ids"], deterministic=False
    )
    val_loader = _make_loader(
        args_eval, split["val_video_ids"], deterministic=True
    )
    logger.info(
        "data_loaders_ready elapsed_s=%.3f batch_size=%s train_batches=%d "
        "val_batches=%d",
        time.perf_counter() - loader_started,
        args_eval["data"].get("batch_size", 1),
        len(train_loader),
        len(val_loader),
    )

    best_val_nmse = float("inf")
    best_val_blocks = {}
    best_epoch = 0
    global_step = 0
    checkpoint = None

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            train_loader,
            args_eval,
            encoder,
            target_encoder,
            teacher_predictor,
            student_predictor,
            replace_layers,
            optimizer,
            device,
            logger,
            epoch,
            training=True,
            max_steps=max_steps,
        )
        global_step += train_metrics["batches"]
        val_metrics = _run_epoch(
            val_loader,
            args_eval,
            encoder,
            target_encoder,
            teacher_predictor,
            student_predictor,
            replace_layers,
            optimizer,
            device,
            logger,
            epoch,
            training=False,
            max_steps=max_steps,
        )
        improved = val_metrics["total_nmse"] < best_val_nmse
        if improved:
            best_val_nmse = val_metrics["total_nmse"]
            best_val_blocks = dict(val_metrics["block_nmse"])
            best_epoch = epoch
            checkpoint = build_optical_checkpoint(
                student_predictor,
                optical_config=vars(optical_config),
                replace_layers=replace_layers,
                teacher_checkpoint=os.path.join(
                    args_eval["pretrain"]["folder"],
                    args_eval["pretrain"]["checkpoint"],
                ),
                distill_target=args_eval["optical_qkv"].get(
                    "distill_target",
                    "attention_proj_output_pre_residual",
                ),
                optimizer=optimizer,
                step=global_step,
                best_nmse={
                    "total": best_val_nmse,
                    "block_nmse": best_val_blocks,
                },
                epoch=epoch,
                metadata={
                    "best_epoch": best_epoch,
                    "split_manifest": os.path.abspath(split_manifest),
                    "train_video_ids": split["train_video_ids"],
                    "val_video_ids": split["val_video_ids"],
                },
            )
            save_started = time.perf_counter()
            _save_checkpoint(checkpoint, output_path)
            logger.info(
                "best_checkpoint_saved epoch=%d best_val_total_nmse=%.6f "
                "path=%s save_time_s=%.3f",
                best_epoch,
                best_val_nmse,
                os.path.abspath(output_path),
                time.perf_counter() - save_started,
            )
        logger.info(
            "epoch_done epoch=%d train_total_nmse=%.6f val_total_nmse=%.6f "
            "best_val_total_nmse=%.6f improved=%s total_elapsed_s=%.3f",
            epoch,
            train_metrics["total_nmse"],
            val_metrics["total_nmse"],
            best_val_nmse,
            improved,
            time.perf_counter() - run_started,
        )

    if checkpoint is None:
        raise RuntimeError("no best checkpoint was produced")

    last_checkpoint = build_optical_checkpoint(
        student_predictor,
        optical_config=vars(optical_config),
        replace_layers=replace_layers,
        teacher_checkpoint=os.path.join(
            args_eval["pretrain"]["folder"],
            args_eval["pretrain"]["checkpoint"],
        ),
        distill_target=args_eval["optical_qkv"].get(
            "distill_target",
            "attention_proj_output_pre_residual",
        ),
        optimizer=optimizer,
        step=global_step,
        best_nmse={
            "total": best_val_nmse,
            "block_nmse": best_val_blocks,
        },
        epoch=epochs,
        metadata={
            "checkpoint_kind": "last",
            "last_epoch": epochs,
            "best_epoch": best_epoch,
            "split_manifest": os.path.abspath(split_manifest),
            "train_video_ids": split["train_video_ids"],
            "val_video_ids": split["val_video_ids"],
        },
    )
    last_save_started = time.perf_counter()
    _save_checkpoint(last_checkpoint, last_output)
    logger.info(
        "last_checkpoint_saved epoch=%d path=%s save_time_s=%.3f",
        epochs,
        os.path.abspath(last_output),
        time.perf_counter() - last_save_started,
    )
    logger.info(
        "run_done best_epoch=%d best_val_total_nmse=%.6f total_elapsed_s=%.3f",
        best_epoch,
        best_val_nmse,
        time.perf_counter() - run_started,
    )
    return checkpoint


def run_end_to_end_jepa(
    args_eval,
    output_path,
    max_steps=None,
    learning_rate=1e-4,
    log_path=None,
    epochs=None,
    split_manifest=None,
    last_output=None,
    final_output=None,
    resume_checkpoint=None,
    skip_final_eval=False,
):
    if log_path is None:
        log_path = f"{output_path}.log"
    logger = _configure_logging(log_path)
    run_started = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_cfg = args_eval.get("training", {})
    split_cfg = args_eval.get("data_split", {})
    epochs = int(training_cfg.get("epochs", 1) if epochs is None else epochs)
    if max_steps is None:
        max_steps = training_cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if split_manifest is None:
        split_manifest = training_cfg.get("split_manifest")
    if split_manifest is None:
        split_manifest = f"{output_path}.split.json"
    if last_output is None:
        last_output = training_cfg.get("last_output")
    if last_output is None:
        last_output = _last_checkpoint_path(output_path)
    if final_output is None:
        final_output = Path(output_path).with_name(
            f"{Path(output_path).stem}.final{Path(output_path).suffix or '.pt'}"
        )
    train_root = get_dataset_paths(["IntPhys-train"])[0]
    split = load_or_create_video_split(
        train_root,
        split_manifest,
        num_train_videos=int(split_cfg.get("num_train_videos", 1500)),
        num_val_videos=int(split_cfg.get("num_val_videos", 300)),
        split_seed=int(split_cfg.get("split_seed", 42)),
    )
    logger.info(
        "run_start mode=end_to_end_jepa epochs=%d max_steps=%s learning_rate=%g "
        "output=%s log=%s device=%s train_videos=%d val_videos=%d "
        "split_manifest=%s last_output=%s",
        epochs,
        max_steps,
        learning_rate,
        os.path.abspath(output_path),
        os.path.abspath(log_path),
        device,
        len(split["train_video_ids"]),
        len(split["val_video_ids"]),
        os.path.abspath(split_manifest),
        os.path.abspath(last_output),
    )
    model_started = time.perf_counter()
    logger.info("model_prepare_start mode=end_to_end_jepa")
    encoder, target_encoder, predictor, optical_config, replace_layers = (
        _prepare_end_to_end_models(args_eval, device)
    )
    _sync_for_timing(device)
    logger.info("model_prepare_done elapsed_s=%.3f", time.perf_counter() - model_started)
    trainable = [parameter for parameter in predictor.parameters() if parameter.requires_grad]
    if len(trainable) != len(list(predictor.parameters())):
        raise RuntimeError("end_to_end_jepa requires every Predictor parameter to be trainable")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_loader = _make_jepa_loader(
        args_eval,
        split["train_video_ids"],
        deterministic=False,
        collator=_make_jepa_mask_collator(args_eval),
    )
    val_loader = _make_jepa_loader(
        args_eval,
        split["val_video_ids"],
        deterministic=True,
        collator=None,
    )
    logger.info(
        "data_loaders_ready mode=end_to_end_jepa batch_size=%s train_batches=%d "
        "val_batches=%d clip_shape=[B,3,16,H,W]",
        args_eval["data"].get("batch_size", 1),
        len(train_loader),
        len(val_loader),
    )
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        logger.info("resume_start checkpoint=%s", os.path.abspath(resume_checkpoint))
        resumed = _load_end_to_end_checkpoint(
            resume_checkpoint, predictor, optimizer, scheduler
        )
        previous_split = resumed.get("data_split")
        if previous_split is not None and previous_split != split:
            raise ValueError("resume checkpoint split does not match current split manifest")
        best_val_loss = float(resumed.get("best_val_jepa_loss", float("inf")))
        best_epoch = int(resumed.get("best_epoch", resumed.get("epoch", 0)))
        global_step = int(resumed.get("global_step", 0))
        start_epoch = int(resumed.get("epoch", 0)) + 1
    best_checkpoint = None
    clip_grad = float(training_cfg.get("clip_grad", 10.0))
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_jepa_epoch(
            train_loader,
            args_eval,
            encoder,
            target_encoder,
            predictor,
            optimizer,
            device,
            logger,
            epoch,
            training=True,
            max_steps=max_steps,
            clip_grad=clip_grad,
        )
        global_step += train_metrics["batches"]
        val_metrics = _run_jepa_epoch(
            val_loader,
            args_eval,
            encoder,
            target_encoder,
            predictor,
            optimizer,
            device,
            logger,
            epoch,
            training=False,
            max_steps=max_steps,
            clip_grad=clip_grad,
        )
        scheduler.step()
        improved = val_metrics["jepa_loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["jepa_loss"]
            best_epoch = epoch
            best_checkpoint = _end_to_end_checkpoint(
                predictor,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_val_loss,
                split,
                split_manifest,
                args_eval,
                "best",
            )
            _save_checkpoint(best_checkpoint, output_path)
            logger.info(
                "best_checkpoint_saved mode=end_to_end_jepa epoch=%d "
                "val_jepa_loss=%.6f path=%s",
                epoch,
                best_val_loss,
                os.path.abspath(output_path),
            )
        last_checkpoint = _end_to_end_checkpoint(
            predictor,
            optimizer,
            scheduler,
            epoch,
            global_step,
            best_val_loss,
            split,
            split_manifest,
            args_eval,
            "last",
            best_epoch=best_epoch,
        )
        _save_checkpoint(last_checkpoint, last_output)
        logger.info(
            "epoch_done mode=end_to_end_jepa epoch=%d train_jepa_loss=%.6f "
            "val_jepa_loss=%.6f best_val_jepa_loss=%.6f improved=%s "
            "total_elapsed_s=%.3f",
            epoch,
            train_metrics["jepa_loss"],
            val_metrics["jepa_loss"],
            best_val_loss,
            improved,
            time.perf_counter() - run_started,
        )
    if best_checkpoint is None:
        if resume_checkpoint is not None and Path(output_path).exists():
            best_checkpoint = torch.load(
                output_path, map_location="cpu", weights_only=False
            )
        else:
            raise RuntimeError("no best end_to_end_jepa checkpoint was produced")
    _save_checkpoint(
        _end_to_end_checkpoint(
            predictor,
            optimizer,
            scheduler,
            max(epochs, start_epoch - 1),
            global_step,
            best_val_loss,
            split,
            split_manifest,
            args_eval,
            "final",
            best_epoch=best_epoch,
        ),
        final_output,
    )
    logger.info(
        "run_done mode=end_to_end_jepa best_epoch=%d "
        "best_val_jepa_loss=%.6f best=%s last=%s final=%s "
        "total_elapsed_s=%.3f",
        best_epoch,
        best_val_loss,
        os.path.abspath(output_path),
        os.path.abspath(last_output),
        os.path.abspath(final_output),
        time.perf_counter() - run_started,
    )
    if (
        not skip_final_eval
        and bool(args_eval.get("evaluation", {}).get("run_after_training", True))
    ):
        _run_final_intphys_evaluation(
            args_eval, output_path, Path(output_path).parent, logger
        )
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="base output name; files are written under jepa_onn/output/<name>_<timestamp>/",
    )
    parser.add_argument(
        "--mode",
        choices=("end_to_end_jepa", "qkv_distill"),
        default=None,
        help="training mode; config training.mode is the fallback",
    )
    parser.add_argument(
        "--block",
        default=None,
        help="deprecated compatibility argument; qkv_distill keeps the legacy argument",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional per-epoch step cap; use 1 for smoke tests",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--last-output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    args = parser.parse_args()
    config = _apply_cli_overrides(
        _load_config(args.config),
        batch_size=args.batch_size,
    )
    outputs = _resolve_run_outputs(args.output)
    mode = args.mode or config.get("training", {}).get(
        "mode", "end_to_end_jepa"
    )
    if mode == "qkv_distill":
        if args.resume is not None:
            raise ValueError("qkv_distill does not use --resume")
        run(
            config,
            output_path=outputs["output"],
            block=args.block,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            log_path=_path_in_run_dir(
                outputs["run_dir"], args.log_file, outputs["log"]
            ),
            epochs=args.epochs,
            split_manifest=_path_in_run_dir(
                outputs["run_dir"], args.split_manifest, outputs["split_manifest"]
            ),
            last_output=_path_in_run_dir(
                outputs["run_dir"], args.last_output, outputs["last_output"]
            ),
        )
    elif mode == "end_to_end_jepa":
        run_end_to_end_jepa(
            config,
            output_path=outputs["output"],
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            log_path=_path_in_run_dir(
                outputs["run_dir"], args.log_file, outputs["log"]
            ),
            epochs=args.epochs,
            split_manifest=_path_in_run_dir(
                outputs["run_dir"], args.split_manifest, outputs["split_manifest"]
            ),
            last_output=_path_in_run_dir(
                outputs["run_dir"], args.last_output, outputs["last_output"]
            ),
            final_output=outputs["final_output"],
            resume_checkpoint=args.resume,
            skip_final_eval=args.skip_final_eval,
        )
    else:
        raise ValueError(f"unsupported training mode: {mode}")


if __name__ == "__main__":
    main()
