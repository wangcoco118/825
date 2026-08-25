"""Epoch-based optical QKV distillation with a fixed Train split."""

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml

from src.models import predictor as vit_pred
from src.masks.utils import apply_masks
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


def _load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _apply_cli_overrides(config, batch_size=None):
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        config.setdefault("data", {})["batch_size"] = int(batch_size)
    return config


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--block",
        default=None,
        help="deprecated compatibility argument; training always uses all Train videos",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="number of train/validation epochs; config training.epochs is the fallback",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional per-epoch train and validation step cap; use 1 for smoke tests",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override data.batch_size for this run",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--last-output", default=None)
    args = parser.parse_args()
    config = _apply_cli_overrides(
        _load_config(args.config),
        batch_size=args.batch_size,
    )
    run(
        config,
        output_path=args.output,
        block=args.block,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        log_path=args.log_file,
        epochs=args.epochs,
        split_manifest=args.split_manifest,
        last_output=args.last_output,
    )


if __name__ == "__main__":
    main()

