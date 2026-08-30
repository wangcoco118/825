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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import yaml

from src.models import predictor as vit_pred
from src.masks.utils import apply_masks
from src.masks.multiblock3d import (
    MaskCollator as MB3DMaskCollator,
    UnifiedMaskCollator,
    make_mask_collator,
    normalize_mask_mode,
)
from src.models.fsonn import OpticalQKVConfig
from src.models.optical_distillation import (
    build_optical_checkpoint,
    freeze_stage_one,
    optical_parameters,
)
from evals.intuitive_physics.data_manager import init_data
from evals.intuitive_physics.eval import init_model
from evals.intuitive_physics.optical_split import (
    load_or_create_video_split,
    require_existing_video_split,
)
from evals.intuitive_physics.utils import get_dataset_paths, get_time_masks
from src.utils.transforms import make_transforms


_require_existing_video_split = require_existing_video_split


def _resolve_gpu_ids(gpu=None, gpus=None):
    """Normalize single- and multi-GPU CLI selections."""
    if gpu is not None and gpus is not None:
        raise ValueError("--gpu and --gpus are mutually exclusive")
    values = [gpu] if gpu is not None else (list(gpus) if gpus else [0])
    ids = [int(value) for value in values]
    if any(value < 0 for value in ids):
        raise ValueError("GPU ids must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError("GPU ids must not contain duplicates")
    return ids


def _device_for_gpu(gpu_id, world_size=1):
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError("multi-GPU training requires CUDA")
        return torch.device("cpu")
    gpu_id = int(gpu_id)
    count = torch.cuda.device_count()
    if gpu_id >= count:
        raise ValueError(
            f"GPU id {gpu_id} is unavailable; visible GPU count is {count}"
        )
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def _unwrap_module(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _feedback_runtime_metadata(predictor):
    predictor_model = _unwrap_module(predictor)
    if hasattr(predictor_model, "backbone"):
        predictor_model = predictor_model.backbone
    onn_core = getattr(predictor_model, "onn_core", None)
    config = getattr(onn_core, "config", None)
    gain_parameter = getattr(onn_core, "feedback_gain_raw", None)
    gain_reader = getattr(onn_core, "_effective_feedback_gains", None)
    if config is None or gain_parameter is None or gain_reader is None:
        return {}
    layer_indices = list(config.active_feedback_layer_indices)
    gains = gain_reader().detach().cpu().reshape(-1).tolist()
    metadata = {
        "feedback_enabled": bool(config.feedback_enabled),
        "feedback_mode": config.feedback_mode,
        "feedback_layer_mode": config.feedback_layer_mode,
        "feedback_phase_max_rad": float(config.feedback_phase_max_rad),
        "feedback_gain_epsilon": float(config.feedback_gain_epsilon),
        "effective_feedback_gains": [float(value) for value in gains],
        "feedback_gain_parameter_count": int(gain_parameter.numel()),
        "physical_feedback_layers": [index + 1 for index in layer_indices],
    }
    if config.feedback_layer_mode == "single":
        metadata["feedback_layer_index"] = int(config.feedback_layer_index)
    else:
        metadata["feedback_layer_indices"] = layer_indices
        metadata["feedback_gain_mode"] = config.feedback_gain_mode
    return metadata


def _format_feedback_metadata(metadata):
    if not metadata:
        return ""
    parts = [
        f"feedback_enabled={metadata['feedback_enabled']}",
        f"feedback_mode={metadata['feedback_mode']}",
        f"feedback_layer_mode={metadata['feedback_layer_mode']}",
    ]
    if metadata["feedback_layer_mode"] == "single":
        parts.append(f"feedback_layer_index={metadata['feedback_layer_index']}")
    else:
        parts.extend(
            [
                f"feedback_layer_indices={metadata['feedback_layer_indices']}",
                f"feedback_gain_mode={metadata['feedback_gain_mode']}",
            ]
        )
    parts.extend(
        [
            f"physical_feedback_layers={metadata['physical_feedback_layers']}",
            f"feedback_phase_max_rad={metadata['feedback_phase_max_rad']:.12g}",
            f"feedback_gain_parameter_count={metadata['feedback_gain_parameter_count']}",
        ]
    )
    gains = metadata["effective_feedback_gains"]
    if len(gains) == 1:
        gains = gains * len(metadata["physical_feedback_layers"])
    parts.extend(
        f"SLM{layer}_K={gain:.6f}"
        for layer, gain in zip(metadata["physical_feedback_layers"], gains)
    )
    return " ".join(parts)


def _distributed_worker(rank, config, args, outputs, gpu_ids):
    world_size = len(gpu_ids)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.dist_port)
    device = _device_for_gpu(gpu_ids[rank], world_size=world_size)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    try:
        _run_selected_mode(
            config,
            args,
            outputs,
            rank=rank,
            world_size=world_size,
            gpu_ids=gpu_ids,
            device=device,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

def _configure_logging(log_path, rank=0):
    log_path = os.path.abspath(log_path)
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    logger = logging.getLogger(
        "fsonn.train" if int(rank) == 0 else f"fsonn.train.rank{int(rank)}"
    )
    logger.setLevel(logging.INFO if int(rank) == 0 else logging.ERROR)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    if int(rank) != 0:
        logger.addHandler(logging.NullHandler())
        return logger

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


def _format_progress(step, total_steps, width=20):
    total_steps = max(int(total_steps), 1)
    step = min(max(int(step), 0), total_steps)
    ratio = step / total_steps
    filled = int(round(ratio * int(width)))
    bar = "#" * filled + "-" * (int(width) - filled)
    return f"[{bar}] {ratio * 100.0:.1f}% ({step}/{total_steps})"


def _format_jepa_batch_log(
    epoch,
    stage,
    mask_mode,
    step,
    total_steps,
    batch_size,
    n_ctxt,
    n_tgt,
    covered_count,
    missing_count,
    loss,
    grad_norm,
    time_s,
):
    return (
        f"epoch={int(epoch)} stage={stage} mask_mode={mask_mode} "
        f"{_format_progress(step, total_steps)} "
        f"batch={int(batch_size)} n_ctxt={n_ctxt} n_tgt={n_tgt} "
        f"covered_count={covered_count} missing_count={missing_count} "
        f"loss={float(loss):.6f} grad_norm={float(grad_norm):.3f} "
        f"time={float(time_s):.3f}s"
    )


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


def _apply_cli_overrides(config, batch_size=None, target_node=None, mask_mode=None):
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        config.setdefault("data", {})["batch_size"] = int(batch_size)
    if target_node is not None:
        config.setdefault("distillation", {})["target_node"] = target_node
    if mask_mode is not None:
        config["mask_mode"] = normalize_mask_mode(mask_mode)
    return config


def _resolve_mask_mode(args_eval):
    training_cfg = args_eval.get("training", {})
    configured = args_eval.get("mask_mode", training_cfg.get("mask_mode"))
    return normalize_mask_mode(configured)


def _resolve_experiment_mode(args_eval):
    training_cfg = args_eval.get("training", {})
    configured = training_cfg.get("experiment_mode")
    if configured is None:
        configured = training_cfg.get("mode", "end_to_end_jepa")
    legacy_mapping = {
        "end_to_end_jepa": "optical_qkv",
        "qkv_distill": "realtime_last_node_distillation",
    }
    mode = legacy_mapping.get(configured, configured)
    if mode not in {
        "electronic_control",
        "optical_qkv",
        "realtime_last_node_distillation",
        "onn_feedback",
    }:
        raise ValueError(f"unsupported experiment mode: {mode}")
    if (
        bool(args_eval.get("distillation", {}).get("enabled", False))
        and mode != "realtime_last_node_distillation"
    ):
        raise ValueError(
            "distillation.enabled=true requires "
            "training.experiment_mode=realtime_last_node_distillation"
        )
    return mode


_DISTILLATION_NODES = (
    "qkv",
    "attention_output",
    "post_output",
    "block_output",
    "predictor_output",
)


def _resolve_distillation_config(args_eval):
    distill_cfg = copy.deepcopy(args_eval.get("distillation", {}))
    target_node = distill_cfg.get("target_node", "qkv")
    if target_node not in _DISTILLATION_NODES:
        raise ValueError(
            f"distillation.target_node must be one of {_DISTILLATION_NODES}, "
            f"got {target_node}"
        )
    optimization_scope = distill_cfg.get("optimization_scope", "last_layer")
    if optimization_scope != "last_layer":
        raise ValueError(
            "realtime_last_node_distillation only supports "
            "distillation.optimization_scope=last_layer"
        )
    cosine_loss_weight = float(distill_cfg.get("cosine_loss_weight", 0.1))
    if cosine_loss_weight < 0:
        raise ValueError("distillation.cosine_loss_weight must be non-negative")
    return {
        "enabled": True,
        "target_node": target_node,
        "optimization_scope": optimization_scope,
        "log_all_layers": bool(distill_cfg.get("log_all_layers", True)),
        "cosine_loss_weight": cosine_loss_weight,
    }


def _should_run_internal_validation(args_eval):
    return _resolve_experiment_mode(args_eval) != (
        "realtime_last_node_distillation"
    )


def _compute_distillation_loss(
    student,
    teacher,
    cosine_loss_weight=0.1,
    eps=1e-6,
):
    if student.shape != teacher.shape:
        raise ValueError(
            f"student and teacher shapes must match, got "
            f"{tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if student.ndim < 2:
        raise ValueError("distillation tensors must include a feature dimension")
    feature_dim = student.shape[-1]
    student_flat = student.reshape(-1, feature_dim)
    teacher_flat = teacher.reshape(-1, feature_dim)
    diff = student_flat - teacher_flat
    nmse = (
        diff.square().sum(dim=-1)
        / teacher_flat.square().sum(dim=-1).clamp_min(float(eps))
    ).mean()
    cosine = (
        1.0
        - F.cosine_similarity(
            student_flat, teacher_flat, dim=-1, eps=float(eps)
        )
    ).mean()
    total = nmse + float(cosine_loss_weight) * cosine
    return total, nmse, cosine


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

    for module in (encoder, target_encoder, teacher_predictor, student_predictor):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    freeze_stage_one(student_predictor)
    if not list(optical_parameters(student_predictor)):
        raise RuntimeError("realtime distillation requires trainable optical parameters")
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


def _predictor_core(predictor):
    return predictor.backbone if hasattr(predictor, "backbone") else predictor


def _forward_predictor_with_nodes(
    predictor,
    context,
    targets,
    masks_ctxt,
    masks_tgt,
    target_node,
    qkv_include_bias=True,
):
    core = _predictor_core(predictor)
    if isinstance(context, (list, tuple)):
        context = context[0]
    if isinstance(targets, (list, tuple)):
        targets = targets[0]
    return core.forward_with_nodes(
        context,
        targets,
        masks_ctxt,
        masks_tgt,
        target_node=target_node,
        qkv_include_bias=qkv_include_bias,
    )


def _optical_gradient_norms(student_predictor):
    norms = {}
    core = _predictor_core(student_predictor)
    for name, parameter in core.named_parameters():
        if "optical_qkv" not in name or parameter.grad is None:
            continue
        parts = name.split(".")
        try:
            block_index = int(parts[1])
        except (IndexError, ValueError):
            block_index = -1
        value = parameter.grad.detach().float().square().sum()
        norms[block_index] = norms.get(block_index, 0.0) + float(value)
    return {
        index: value ** 0.5
        for index, value in sorted(norms.items())
    }


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
    clip_grad=10.0,
    stage_name=None,
):
    stage = stage_name or ("train" if training else "val")
    distill_cfg = _resolve_distillation_config(args_eval)
    target_node = distill_cfg["target_node"]
    student_predictor.eval()
    teacher_predictor.eval()
    encoder.eval()
    target_encoder.eval()
    depth = len(_predictor_core(student_predictor).predictor_blocks)
    last_block = depth - 1
    if not replace_layers:
        raise ValueError("realtime distillation requires at least one optical block")

    total_loss = 0.0
    total_nmse = 0.0
    total_cosine = 0.0
    layer_loss_sum = {}
    layer_nmse_sum = {}
    layer_cosine_sum = {}
    batches = 0
    started = time.perf_counter()
    stage_total = len(loader)
    if max_steps is not None:
        stage_total = min(stage_total, max_steps)

    for batch_index, batch in enumerate(loader):
        if max_steps is not None and batch_index >= max_steps:
            break
        step_started = time.perf_counter()
        context, targets, masks_ctxt, masks_tgt, batch_size = _prepare_features(
            batch, args_eval, encoder, target_encoder, device
        )

        teacher_started = time.perf_counter()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        )
        with torch.no_grad():
            _, teacher_nodes = _forward_predictor_with_nodes(
                teacher_predictor,
                context,
                targets,
                masks_ctxt,
                masks_tgt,
                target_node,
                qkv_include_bias=(target_node != "qkv"),
            )
        _sync_for_timing(device)
        teacher_time = time.perf_counter() - teacher_started
        torch.set_rng_state(cpu_rng_state)
        if device.type == "cuda" and cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device=device)

        student_started = time.perf_counter()
        if training:
            _, student_nodes = _forward_predictor_with_nodes(
                student_predictor,
                context,
                targets,
                masks_ctxt,
                masks_tgt,
                target_node,
                qkv_include_bias=True,
            )
        else:
            with torch.no_grad():
                _, student_nodes = _forward_predictor_with_nodes(
                    student_predictor,
                    context,
                    targets,
                    masks_ctxt,
                    masks_tgt,
                    target_node,
                    qkv_include_bias=True,
                )
        _sync_for_timing(device)
        student_time = time.perf_counter() - student_started

        if target_node == "predictor_output":
            train_loss, train_nmse, train_cosine = _compute_distillation_loss(
                student_nodes[target_node],
                teacher_nodes[target_node],
                cosine_loss_weight=distill_cfg["cosine_loss_weight"],
            )
            current_layer_losses = {}
            current_layer_nmse = {}
            current_layer_cosine = {}
        else:
            student_layer_nodes = student_nodes[target_node]
            teacher_layer_nodes = teacher_nodes[target_node]
            if len(student_layer_nodes) != depth or len(teacher_layer_nodes) != depth:
                raise RuntimeError(
                    f"{target_node} must produce one node per Block; "
                    f"got student={len(student_layer_nodes)} teacher={len(teacher_layer_nodes)} "
                    f"depth={depth}"
                )
            current_layer_losses = {}
            current_layer_nmse = {}
            current_layer_cosine = {}
            last_loss = None
            last_nmse = None
            last_cosine = None
            for block_index, (student_node, teacher_node) in enumerate(
                zip(student_layer_nodes, teacher_layer_nodes)
            ):
                loss, nmse, cosine = _compute_distillation_loss(
                    student_node,
                    teacher_node,
                    cosine_loss_weight=distill_cfg["cosine_loss_weight"],
                )
                current_layer_losses[block_index] = float(loss.detach())
                current_layer_nmse[block_index] = float(nmse.detach())
                current_layer_cosine[block_index] = float(cosine.detach())
                if block_index == last_block:
                    last_loss = loss
                    last_nmse = nmse
                    last_cosine = cosine
            train_loss = last_loss
            train_nmse = last_nmse
            train_cosine = last_cosine

        if training:
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(optical_parameters(student_predictor)),
                float(clip_grad),
            )
            gradient_norms = _optical_gradient_norms(student_predictor)
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0, device=device)
            gradient_norms = {}

        loss_value = float(train_loss.detach())
        nmse_value = float(train_nmse.detach())
        cosine_value = float(train_cosine.detach())
        total_loss += loss_value
        total_nmse += nmse_value
        total_cosine += cosine_value
        for index, value in current_layer_losses.items():
            layer_loss_sum[index] = layer_loss_sum.get(index, 0.0) + value
            layer_nmse_sum[index] = layer_nmse_sum.get(index, 0.0) + current_layer_nmse[index]
            layer_cosine_sum[index] = layer_cosine_sum.get(index, 0.0) + current_layer_cosine[index]
        batches += 1
        logger.info(
            "epoch=%d stage=%s %s target_node=%s last_block=%d "
            "batch_size=%d loss=%.6f nmse=%.6f cosine=%.6f "
            "teacher_time_s=%.3f student_time_s=%.3f elapsed_s=%.3f "
            "grad_norm=%.3f optical_grad_norms={%s} layer_nmse={%s}",
            epoch,
            stage,
            _format_progress(batches, stage_total),
            target_node,
            last_block,
            batch_size,
            loss_value,
            nmse_value,
            cosine_value,
            teacher_time,
            student_time,
            time.perf_counter() - step_started,
            float(grad_norm),
            ",".join(f"{i}:{v:.6f}" for i, v in gradient_norms.items()),
            ",".join(f"{i}:{v:.6f}" for i, v in current_layer_nmse.items()),
        )

    if batches == 0:
        raise RuntimeError(f"{stage} loader produced no batches")
    metrics = {
        "train_loss": total_loss / batches,
        "train_nmse": total_nmse / batches,
        "train_cosine": total_cosine / batches,
        "layer_loss": {i: v / batches for i, v in layer_loss_sum.items()},
        "layer_nmse": {i: v / batches for i, v in layer_nmse_sum.items()},
        "layer_cosine": {i: v / batches for i, v in layer_cosine_sum.items()},
        "batches": batches,
        "elapsed_s": time.perf_counter() - started,
        "last_block": last_block,
        "target_node": target_node,
    }
    logger.info(
        "epoch=%d stage=%s_done target_node=%s last_block=%d batches=%d "
        "loss=%.6f nmse=%.6f cosine=%.6f layer_nmse={%s} "
        "elapsed_s=%.3f",
        epoch,
        stage,
        target_node,
        last_block,
        batches,
        metrics["train_loss"],
        metrics["train_nmse"],
        metrics["train_cosine"],
        ",".join(f"{i}:{v:.6f}" for i, v in metrics["layer_nmse"].items()),
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
    return make_mask_collator(
        mask_mode=_resolve_mask_mode(args_eval),
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
    world_size=1,
    rank=0,
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
        world_size=world_size,
        rank=rank,
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


def _prepare_end_to_end_models(args_eval, device, experiment_mode="optical_qkv"):
    predictor_type = args_eval.setdefault("predictor_type", "onn_feedback")
    optical_cfg = args_eval.get("optical_qkv", {})
    onn_cfg = args_eval.get(
        "onn", args_eval.get("onn_feedback", optical_cfg)
    )

    if predictor_type == "onn_feedback":
        optical_config = copy.deepcopy(onn_cfg)
        replace_layers = []
    elif experiment_mode == "optical_qkv":
        if optical_cfg.get("qkv_backend") != "fsonn_tdm":
            raise ValueError(
                "optical_qkv requires optical_qkv.qkv_backend=fsonn_tdm"
            )
        optical_config = OpticalQKVConfig.from_mapping(optical_cfg)
        replace_layers = optical_cfg.get("replace_layers", "all")
        if replace_layers == "all":
            replace_layers = list(range(args_eval["pretrain"].get("pred_depth", 12)))
    elif experiment_mode == "electronic_control":
        optical_config = None
        replace_layers = []
    else:
        raise ValueError(
            f"_prepare_end_to_end_models does not support {experiment_mode}"
        )

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
        optical_qkv={} if predictor_type == "onn_feedback" else optical_cfg,
        predictor_type=predictor_type,
        onn_feedback_config=onn_cfg,
    )
    if predictor_type != "onn_feedback" and experiment_mode == "optical_qkv":
        vit_pred.install_optical_qkv(
            predictor,
            optical_config=optical_config,
            replace_layers=replace_layers,
        )

    for module in (encoder, target_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    predictor.train()
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    if predictor_type == "onn_feedback":
        assert all(not p.requires_grad for p in encoder.parameters())
        assert all(not p.requires_grad for p in target_encoder.parameters())
        predictor_parameters = dict(predictor.named_parameters())
        predictor_buffers = dict(predictor.named_buffers())
        pos_names = [
            name for name in predictor_buffers
            if name.endswith("predictor_pos_embed")
        ]
        assert pos_names
        assert all(
            "predictor_pos_embed" not in name
            for name in predictor_parameters
        )
        assert all(
            not predictor_buffers[name].requires_grad for name in pos_names
        )
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
    rank=0,
    world_size=1,
):
    stage = "train" if training else "val"
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))
    predictor.train(training)
    encoder.eval()
    target_encoder.eval()
    total_loss = 0.0
    batches = 0
    started = time.perf_counter()
    stage_total = len(loader)
    if max_steps is not None:
        stage_total = min(stage_total, max_steps)
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
        predictor_core = _unwrap_module(predictor)
        predictor_core = (
            predictor_core.backbone
            if hasattr(predictor_core, "backbone")
            else predictor_core
        )
        trace = getattr(predictor_core, "last_trace", {})
        if int(rank) == 0:
            logger.info(
                _format_jepa_batch_log(
                    epoch=epoch,
                    stage=stage,
                    mask_mode=_resolve_mask_mode(args_eval),
                    step=batches,
                    total_steps=stage_total,
                    batch_size=clips.shape[0],
                    n_ctxt=trace.get("n_ctxt"),
                    n_tgt=trace.get("n_tgt"),
                    covered_count=trace.get("covered_count"),
                    missing_count=trace.get("missing_count"),
                    loss=loss_value,
                    grad_norm=grad_norm,
                    time_s=time.perf_counter() - step_started,
                )
            )
    if world_size > 1:
        stats = torch.tensor(
            [total_loss, float(batches)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        batches = int(round(float(stats[1].item())))
    if batches == 0:
        raise RuntimeError(f"{stage} loader produced no batches")
    mean_loss = total_loss / batches
    metrics = {
        "jepa_loss": mean_loss,
        "batches": batches,
        "elapsed_s": time.perf_counter() - started,
    }
    if int(rank) == 0:
        logger.info(
            "epoch=%d stage=%s_done %s batches=%d jepa_loss=%.6f time=%.3fs",
            epoch,
            stage,
            _format_progress(batches, batches),
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
    experiment_mode="optical_qkv",
    world_size=1,
    gpu_ids=None,
):
    predictor_model = _unwrap_module(predictor)
    predictor_state = {
        key: value.detach().cpu().clone()
        for key, value in predictor_model.state_dict().items()
    }
    checkpoint_mode = {
        "optical_qkv": "end_to_end_jepa",
        "electronic_control": "electronic_control",
        "onn_feedback": "onn_feedback",
    }.get(experiment_mode)
    if checkpoint_mode is None:
        raise ValueError(f"unsupported checkpoint experiment mode: {experiment_mode}")
    checkpoint = {
        "format_version": 1,
        "mode": checkpoint_mode,
        "experiment_mode": experiment_mode,
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
        "predictor_type": args_eval.get("predictor_type", "onn_feedback"),
        "mask_mode": _resolve_mask_mode(args_eval),
        "num_tokens": 1568,
        "num_chunks": 8,
        "chunk_tokens": 196,
        "predictor_dim": 384,
        "output_dim": 1024,
        "readout_mode": "intensity_minus_learnable_offset",
        "differential_detector": False,
        "world_size": int(world_size),
        "gpu_ids": list(gpu_ids or []),
        "onn": copy.deepcopy(
            args_eval.get("onn", args_eval.get("onn_feedback", {}))
        ),
        "onn_feedback": copy.deepcopy(args_eval.get("onn_feedback", {})),
        "optical_qkv": (
            copy.deepcopy(args_eval.get("optical_qkv", {}))
            if experiment_mode == "optical_qkv"
            else {}
        ),
        "replace_layers": (
            copy.deepcopy(args_eval.get("optical_qkv", {}).get("replace_layers", "all"))
            if experiment_mode == "optical_qkv"
            else []
        ),
        "pretrain_checkpoint": os.path.join(
            args_eval["pretrain"]["folder"], args_eval["pretrain"]["checkpoint"]
        ),
        "config_format_version": 1,
        "config": copy.deepcopy(args_eval),
        "training_config": copy.deepcopy(args_eval.get("training", {})),
        "data_split": copy.deepcopy(split),
        "split_manifest": os.path.abspath(split_manifest),
    }
    checkpoint.update(_feedback_runtime_metadata(predictor))
    return checkpoint


def _load_end_to_end_checkpoint(
    checkpoint_path,
    predictor,
    optimizer,
    scheduler,
    expected_config=None,
    expected_mode="optical_qkv",
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_mode = checkpoint.get("experiment_mode")
    if saved_mode is None and checkpoint.get("mode") == "end_to_end_jepa":
        saved_mode = "optical_qkv"
    if saved_mode is None:
        raise ValueError(
            "end_to_end_jepa/electronic_control can resume only from a "
            "full end-to-end checkpoint"
        )
    if saved_mode != expected_mode:
        raise ValueError(
            f"{expected_mode} cannot resume from {saved_mode} checkpoint"
        )
    if "predictor" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError(
            f"{expected_mode} checkpoint is missing full Predictor state"
        )
    saved_config = checkpoint.get("config")
    if saved_config is None:
        raise ValueError(
            f"{expected_mode} checkpoint has no complete config and cannot be "
            "resumed safely"
        )
    if expected_config is not None and saved_config != expected_config:
        raise ValueError(
            f"{expected_mode} checkpoint config does not match the current config"
        )
    _unwrap_module(predictor).load_state_dict(
        checkpoint["predictor"], strict=True
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


def _run_final_intphys_evaluation(
    args_eval, best_path, run_dir, logger, dev_only=False
):
    from evals.intuitive_physics import eval as dev_eval
    evaluation_jobs = [("dev", dev_eval, "intphys_dev")]
    if not dev_only:
        from evals.intphys_test import eval as test_eval
        evaluation_jobs.append(("test", test_eval, "intphys_test"))
    for label, module, folder_name in evaluation_jobs:
        evaluation_cfg = copy.deepcopy(args_eval)
        evaluation_cfg["predictor_checkpoint"] = os.path.abspath(best_path)
        evaluation_cfg.setdefault(
            "predictor_type", args_eval.get("predictor_type", "onn_feedback")
        )
        evaluation_cfg.setdefault(
            "onn_feedback",
            copy.deepcopy(
                args_eval.get("onn_feedback", args_eval.get("optical_qkv", {}))
            ),
        )
        evaluation_cfg["output_dir"] = os.path.join(run_dir, folder_name)
        if label == "dev":
            evaluation_cfg["dataset"] = "intphys"
            evaluation_cfg["eval_name"] = "intuitive_physics"
        else:
            evaluation_cfg["dataset"] = "intphys-test"
            evaluation_cfg["eval_name"] = "intphys_test"
        if _resolve_experiment_mode(args_eval) == "electronic_control":
            evaluation_cfg["optical_qkv"] = {}
        logger.info(
            "final_evaluation_start split=%s dataset=%s checkpoint=%s",
            label,
            evaluation_cfg["dataset"],
            best_path,
        )
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
    skip_final_eval=False,
    device=None,
    gpu_id=0,
):
    if _resolve_experiment_mode(args_eval) != "realtime_last_node_distillation":
        raise ValueError(
            "run() is reserved for realtime_last_node_distillation"
        )
    distill_cfg = _resolve_distillation_config(args_eval)
    if log_path is None:
        log_path = f"{output_path}.log"
    logger = _configure_logging(log_path)
    run_started = time.perf_counter()
    if device is None:
        device = _device_for_gpu(gpu_id, world_size=1)
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
        num_val_videos=0,
        split_seed=int(split_cfg.get("split_seed", 42)),
    )
    split["val_video_ids"] = []
    split["num_val_videos"] = 0
    logger.info(
        "run_start mode=realtime_last_node_distillation target_node=%s "
        "optimization_scope=%s epochs=%d max_steps=%s learning_rate=%g "
        "output=%s log=%s device=%s train_videos=%d validation=disabled "
        "split_manifest=%s",
        distill_cfg["target_node"],
        distill_cfg["optimization_scope"],
        epochs,
        max_steps,
        learning_rate,
        os.path.abspath(output_path),
        os.path.abspath(log_path),
        device,
        len(split["train_video_ids"]),
        os.path.abspath(split_manifest),
    )

    model_started = time.perf_counter()
    logger.info("model_prepare_start mode=realtime_last_node_distillation")
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

    trainable = list(optical_parameters(student_predictor))
    if not trainable:
        raise RuntimeError("no optical parameters are trainable")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    loader_started = time.perf_counter()
    train_loader = _make_loader(
        args_eval, split["train_video_ids"], deterministic=False
    )
    logger.info(
        "data_loader_ready mode=realtime_last_node_distillation "
        "target_node=%s batch_size=%s train_batches=%d validation=disabled "
        "clip_shape=[B,3,16,H,W] trainable_optical_params=%d elapsed_s=%.3f",
        distill_cfg["target_node"],
        args_eval["data"].get("batch_size", 1),
        len(train_loader),
        sum(parameter.numel() for parameter in trainable),
        time.perf_counter() - loader_started,
    )

    clip_grad = float(training_cfg.get("clip_grad", 10.0))
    global_step = 0
    last_train_metrics = None
    teacher_checkpoint = os.path.join(
        args_eval["pretrain"]["folder"],
        args_eval["pretrain"]["checkpoint"],
    )

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
            clip_grad=clip_grad,
        )
        global_step += train_metrics["batches"]
        last_train_metrics = train_metrics
        logger.info(
            "epoch_done mode=realtime_last_node_distillation target_node=%s "
            "epoch=%d train_loss=%.6f train_nmse=%.6f train_cosine=%.6f "
            "checkpoint=deferred_until_final elapsed_s=%.3f",
            distill_cfg["target_node"],
            epoch,
            train_metrics["train_loss"],
            train_metrics["train_nmse"],
            train_metrics["train_cosine"],
            time.perf_counter() - run_started,
        )

    if last_train_metrics is None:
        raise RuntimeError("no training checkpoint was produced")

    final_checkpoint = build_optical_checkpoint(
        student_predictor,
        optical_config=vars(optical_config),
        replace_layers=replace_layers,
        teacher_checkpoint=teacher_checkpoint,
        distill_target=distill_cfg["target_node"],
        optimizer=optimizer,
        step=global_step,
        best_nmse={
            "final_train_loss": last_train_metrics["train_loss"],
            "final_train_nmse": last_train_metrics["train_nmse"],
            "final_train_cosine": last_train_metrics["train_cosine"],
        },
        epoch=epochs,
        target_node=distill_cfg["target_node"],
        optimization_scope=distill_cfg["optimization_scope"],
        cosine_loss_weight=distill_cfg["cosine_loss_weight"],
        metadata={
            "checkpoint_kind": "final_last",
            "last_epoch": epochs,
            "last_block": last_train_metrics["last_block"],
            "validation_disabled": True,
            "split_manifest": os.path.abspath(split_manifest),
            "train_video_ids": split["train_video_ids"],
            "val_video_ids": split["val_video_ids"],
            "heldout_test_video_ids": [],
            "final_evaluation_dataset": "intphys-dev",
        },
    )
    save_started = time.perf_counter()
    _save_checkpoint(final_checkpoint, output_path)
    logger.info(
        "final_checkpoint_saved mode=realtime_last_node_distillation "
        "target_node=%s epoch=%d path=%s save_time_s=%.3f",
        distill_cfg["target_node"],
        epochs,
        os.path.abspath(output_path),
        time.perf_counter() - save_started,
    )
    logger.info(
        "run_done mode=realtime_last_node_distillation target_node=%s "
        "final_epoch=%d final_train_loss=%.6f final=%s "
        "validation=disabled final_evaluation_dataset=intphys-dev elapsed_s=%.3f",
        distill_cfg["target_node"],
        epochs,
        last_train_metrics["train_loss"],
        os.path.abspath(output_path),
        time.perf_counter() - run_started,
    )
    if (
        not skip_final_eval
        and bool(args_eval.get("evaluation", {}).get("run_after_training", True))
    ):
        _run_final_intphys_evaluation(
            args_eval,
            output_path,
            Path(output_path).parent,
            logger,
            dev_only=True,
        )
    return final_checkpoint



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
    experiment_mode="optical_qkv",
    device=None,
    gpu_id=0,
    rank=0,
    world_size=1,
    gpu_ids=None,
):
    if experiment_mode not in {"optical_qkv", "electronic_control", "onn_feedback"}:
        raise ValueError(
            f"run_end_to_end_jepa does not support {experiment_mode}"
        )
    if log_path is None:
        log_path = f"{output_path}.log"
    logger = _configure_logging(log_path, rank=rank)
    run_started = time.perf_counter()
    if device is None:
        device = _device_for_gpu(gpu_id, world_size=world_size)
    if world_size > 1 and not dist.is_initialized():
        raise RuntimeError("multi-GPU training requires an initialized process group")
    gpu_ids = list(gpu_ids or [gpu_id])
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
    split = require_existing_video_split(
        train_root,
        split_manifest,
        num_train_videos=int(split_cfg.get("num_train_videos", 1500)),
        num_val_videos=int(split_cfg.get("num_val_videos", 300)),
        split_seed=int(split_cfg.get("split_seed", 42)),
    )
    logger.info(
        "run_start experiment_mode=%s mask_mode=%s epochs=%d max_steps=%s learning_rate=%g "
        "output=%s log=%s device=%s train_videos=%d val_videos=%d "
        "split_manifest=%s last_output=%s",
        experiment_mode,
        _resolve_mask_mode(args_eval),
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
    logger.info("model_prepare_start experiment_mode=%s", experiment_mode)
    encoder, target_encoder, predictor, optical_config, replace_layers = (
        _prepare_end_to_end_models(
            args_eval, device, experiment_mode=experiment_mode
        )
    )
    if world_size > 1:
        predictor = DistributedDataParallel(
            predictor,
            device_ids=[int(gpu_id)],
            output_device=int(gpu_id),
            broadcast_buffers=True,
        )
    _sync_for_timing(device)
    logger.info("model_prepare_done elapsed_s=%.3f", time.perf_counter() - model_started)
    if int(rank) == 0:
        feedback_metadata = _feedback_runtime_metadata(predictor)
        if feedback_metadata:
            logger.info(
                "feedback_init %s",
                _format_feedback_metadata(feedback_metadata),
            )
    trainable = [
        parameter for parameter in predictor.parameters()
        if parameter.requires_grad
    ]
    encoder_trainable = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )
    target_encoder_trainable = sum(
        parameter.numel()
        for parameter in target_encoder.parameters()
        if parameter.requires_grad
    )
    if encoder_trainable or target_encoder_trainable:
        raise RuntimeError(
            "context encoder and target encoder must be fully frozen"
        )
    if not trainable:
        raise RuntimeError("no Predictor parameters are trainable")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    logger.info(
        "optimizer_ready predictor_trainable_params=%d "
        "encoder_trainable_params=%d target_encoder_trainable_params=%d",
        sum(parameter.numel() for parameter in trainable),
        encoder_trainable,
        target_encoder_trainable,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_loader = _make_jepa_loader(
        args_eval,
        split["train_video_ids"],
        deterministic=False,
        collator=_make_jepa_mask_collator(args_eval),
        world_size=world_size,
        rank=rank,
    )
    val_loader = _make_jepa_loader(
        args_eval,
        split["val_video_ids"],
        deterministic=True,
        collator=_make_jepa_mask_collator(args_eval),
        world_size=world_size,
        rank=rank,
    )
    logger.info(
        "data_loaders_ready experiment_mode=%s batch_size=%s train_batches=%d "
        "val_batches=%d clip_shape=[B,3,16,H,W]",
        experiment_mode,
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
            resume_checkpoint,
            predictor,
            optimizer,
            scheduler,
            expected_config=args_eval,
            expected_mode=experiment_mode,
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
    if resume_checkpoint is None:
        initial_val_metrics = _run_jepa_epoch(
            val_loader,
            args_eval,
            encoder,
            target_encoder,
            predictor,
            optimizer,
            device,
            logger,
            0,
            training=False,
            max_steps=max_steps,
            clip_grad=clip_grad,
            rank=rank,
            world_size=world_size,
        )
        best_val_loss = initial_val_metrics["jepa_loss"]
        best_epoch = 0
        best_checkpoint = _end_to_end_checkpoint(
            predictor,
            optimizer,
            scheduler,
            0,
            global_step,
            best_val_loss,
            split,
            split_manifest,
            args_eval,
            "best",
            best_epoch=0,
            experiment_mode=experiment_mode,
            world_size=world_size,
            gpu_ids=gpu_ids,
        )
        if rank == 0:
            _save_checkpoint(best_checkpoint, output_path)
        if world_size > 1:
            dist.barrier()
        logger.info(
            "best_checkpoint_saved experiment_mode=%s epoch=0 "
            "val_jepa_loss=%.6f path=%s",
            experiment_mode,
            best_val_loss,
            os.path.abspath(output_path),
        )
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
            rank=rank,
            world_size=world_size,
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
            rank=rank,
            world_size=world_size,
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
                experiment_mode=experiment_mode,
                world_size=world_size,
                gpu_ids=gpu_ids,
            )
            if rank == 0:
                _save_checkpoint(best_checkpoint, output_path)
            if world_size > 1:
                dist.barrier()
            logger.info(
                "best_checkpoint_saved experiment_mode=%s epoch=%d "
                "val_jepa_loss=%.6f path=%s",
                experiment_mode,
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
            experiment_mode=experiment_mode,
            world_size=world_size,
            gpu_ids=gpu_ids,
        )
        if rank == 0:
            _save_checkpoint(last_checkpoint, last_output)
        if world_size > 1:
            dist.barrier()
        if int(rank) == 0:
            feedback_metadata = _feedback_runtime_metadata(predictor)
            if feedback_metadata:
                logger.info(
                    "feedback_epoch epoch=%d %s",
                    epoch,
                    _format_feedback_metadata(feedback_metadata),
                )
        logger.info(
            "epoch_done experiment_mode=%s epoch=%d train_jepa_loss=%.6f "
            "val_jepa_loss=%.6f best_val_jepa_loss=%.6f improved=%s "
            "elapsed_s=%.3f",
            experiment_mode,
            epoch,
            train_metrics["jepa_loss"],
            val_metrics["jepa_loss"],
            best_val_loss,
            improved,
            time.perf_counter() - run_started,
        )
    if rank == 0:
        if best_checkpoint is None:
            if resume_checkpoint is not None and Path(output_path).exists():
                best_checkpoint = torch.load(
                    output_path, map_location="cpu", weights_only=False
                )
            else:
                raise RuntimeError("no best end_to_end_jepa checkpoint was produced")
        logger.info(
            "run_done experiment_mode=%s world_size=%d gpu_ids=%s best_epoch=%d "
            "best_val_jepa_loss=%.6f best=%s last=%s "
            "final_checkpoint=disabled elapsed_s=%.3f",
            experiment_mode,
            world_size,
            gpu_ids,
            best_epoch,
            best_val_loss,
            os.path.abspath(output_path),
            os.path.abspath(last_output),
            time.perf_counter() - run_started,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return None

    if (
        not skip_final_eval
        and bool(args_eval.get("evaluation", {}).get("run_after_training", True))
    ):
        _run_final_intphys_evaluation(
            args_eval, output_path, Path(output_path).parent, logger
        )
    return best_checkpoint



def _run_selected_mode(
    config,
    args,
    outputs,
    rank=0,
    world_size=1,
    gpu_ids=None,
    device=None,
):
    gpu_ids = list(gpu_ids or [0])
    mode = _resolve_experiment_mode(config)
    if mode == "realtime_last_node_distillation":
        if world_size > 1:
            raise ValueError(
                "multi-GPU mode is supported for end-to-end training only; "
                "realtime_last_node_distillation remains single-GPU"
            )
        if args.resume is not None:
            raise ValueError(
                "realtime_last_node_distillation does not use --resume"
            )
        return run(
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
            skip_final_eval=args.skip_final_eval,
            device=device,
            gpu_id=gpu_ids[rank],
        )
    if mode in {"onn_feedback", "optical_qkv", "electronic_control"}:
        return run_end_to_end_jepa(
            config,
            output_path=outputs["output"],
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            log_path=_path_in_run_dir(
                outputs["run_dir"], args.log_file, outputs["log"]
            ),
            epochs=args.epochs,
            split_manifest=args.split_manifest,
            last_output=_path_in_run_dir(
                outputs["run_dir"], args.last_output, outputs["last_output"]
            ),
            final_output=outputs["final_output"],
            resume_checkpoint=args.resume,
            skip_final_eval=args.skip_final_eval,
            experiment_mode=mode,
            device=device,
            gpu_id=gpu_ids[rank],
            rank=rank,
            world_size=world_size,
            gpu_ids=gpu_ids,
        )
    raise ValueError(f"unsupported training mode: {mode}")


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
        choices=("end_to_end_jepa", "qkv_distill", "realtime_last_node_distillation"),
        default=None,
        help="legacy mode alias; use --experiment-mode for the new switch",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=(
            "onn_feedback",
            "electronic_control",
            "optical_qkv",
            "realtime_last_node_distillation",
        ),
        default=None,
        help="end-to-end experiment mode",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="single visible CUDA GPU id; defaults to 0",
    )
    gpu_group.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="visible CUDA GPU ids for one DDP process per GPU",
    )
    parser.add_argument(
        "--dist-port",
        type=int,
        default=29517,
        help="local TCP port used by the internally launched DDP processes",
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
    parser.add_argument(
        "--mask-mode",
        choices=("unified_random", "classic_random"),
        default=None,
        help="override the configured mask generation mode",
    )
    parser.add_argument(
        "--target-node",
        choices=("qkv", "attention_output", "post_output", "block_output", "predictor_output"),
        default=None,
        help="override distillation.target_node for realtime distillation",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--last-output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "do not combine this launcher with torchrun; pass --gpu or --gpus "
            "to train_optical.py directly"
        )

    config = _apply_cli_overrides(
        _load_config(args.config),
        batch_size=args.batch_size,
        target_node=args.target_node,
        mask_mode=args.mask_mode,
    )
    outputs = _resolve_run_outputs(args.output)
    training_cfg = config.setdefault("training", {})
    if args.mode is not None:
        training_cfg["experiment_mode"] = {
            "end_to_end_jepa": "optical_qkv",
            "qkv_distill": "realtime_last_node_distillation",
            "realtime_last_node_distillation": "realtime_last_node_distillation",
        }[args.mode]
    if args.experiment_mode is not None:
        training_cfg["experiment_mode"] = args.experiment_mode

    gpu_ids = _resolve_gpu_ids(args.gpu, args.gpus)
    if len(gpu_ids) > 1:
        mp.spawn(
            _distributed_worker,
            args=(config, args, outputs, gpu_ids),
            nprocs=len(gpu_ids),
            join=True,
        )
    else:
        _run_selected_mode(
            config,
            args,
            outputs,
            rank=0,
            world_size=1,
            gpu_ids=gpu_ids,
            device=None,
        )


if __name__ == "__main__":
    main()
