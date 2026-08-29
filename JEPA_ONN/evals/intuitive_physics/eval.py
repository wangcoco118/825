# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import time

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np
from einops import rearrange

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import all_gather
from sklearn.metrics import precision_recall_curve,roc_curve,auc,roc_auc_score
from scipy.stats import mannwhitneyu,wilcoxon,ttest_rel,ttest_ind
import torch.distributed as dist


from src.utils.tensors import repeat_interleave_batch
from src.utils.amp import autocast_context
from src.masks.utils import apply_masks
import src.models.vision_transformer as vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.models.fsonn import OpticalQKVConfig
from src.models.optical_distillation import load_optical_checkpoint
from evals.intuitive_physics.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import (
    MaskCollator as MB3DMaskCollator,
    make_mask_collator,
    normalize_mask_mode,
)
from src.masks.causal import MaskCollator as CausalMaskCollator
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from src.utils.transforms import make_transforms

from evals.intuitive_physics.utils import get_matches, get_breaking_points, get_time_masks,get_dataset_paths,batch_all_gather,PROPERTIES_BY_DATASET,pad_tensors
import evals.intuitive_physics.videomae as videomae

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


class SynchronizedProgressLog:
    def __init__(self, path, print_enabled=True):
        self.path = path
        self.print_enabled = print_enabled
        self.stream = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {message}"
        if self.print_enabled:
            print(line, flush=True)
        self.stream.write(line + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def _format_metric_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return float(value)


def _format_key_metrics(metrics):
    return (
        f"rel_acc_max={_format_metric_value(metrics['Relative Accuracy (max)']):.2f}% "
        f"abs_acc_max={_format_metric_value(metrics['Absolute Accuracy (max)']):.2f}% "
        f"auprc_max={_format_metric_value(metrics['AUPRC (max)']):.4f} "
        f"auroc_max={_format_metric_value(metrics['AUROC (max)']):.4f}"
    )


def _format_official_metrics(metrics):
    return (
        f"official_mean[LR={metrics['official_mean_LR']:.4f} "
        f"RA={metrics['official_mean_relative_accuracy']:.4f} "
        f"AUC={metrics['official_mean_AUC']:.4f} "
        f"LA={metrics['official_mean_LA']:.4f}]; "
        f"official_max[LR={metrics['official_max_LR']:.4f} "
        f"RA={metrics['official_max_relative_accuracy']:.4f} "
        f"AUC={metrics['official_max_AUC']:.4f} "
        f"LA={metrics['official_max_LA']:.4f}]"
    )


def _format_running_metrics(
    all_losses,
    all_labels,
    context_lengths,
    official_group_losses=None,
    official_group_labels=None,
):
    if not all_losses:
        return "metrics=warming_up"

    losses = torch.cat([loss.detach().cpu() for loss in all_losses], dim=0)
    labels = torch.cat([label.detach().cpu() for label in all_labels], dim=0)
    if torch.unique(labels).numel() < 2:
        return f"metrics=waiting_for_both_labels samples={labels.numel()}"

    parts = []
    for index, context in enumerate(context_lengths):
        try:
            parts.append(
                f"ctx{context}[{_format_key_metrics(compute_metrics(losses[:, index], labels))}]"
            )
        except (IndexError, ValueError, RuntimeError):
            parts.append(f"ctx{context}[metrics=warming_up samples={labels.numel()}]")

    try:
        filtered = losses.min(1)[0]
        parts.append(f"filtered[{_format_key_metrics(compute_metrics(filtered, labels))}]")
    except (IndexError, ValueError, RuntimeError):
        parts.append(f"filtered[metrics=warming_up samples={labels.numel()}]")

    if official_group_losses and official_group_labels:
        group_losses = torch.stack(
            [loss.detach().cpu() for loss in official_group_losses], dim=0
        )
        group_labels = torch.stack(
            [label.detach().cpu() for label in official_group_labels], dim=0
        )
        try:
            for index, context in enumerate(context_lengths):
                official = compute_official_metrics(
                    group_losses[:, :, index, :], group_labels
                )
                parts.append(
                    f"official_ctx{context}[{_format_official_metrics(official)}]"
                )
            filtered_group_losses = group_losses.min(2)[0]
            official = compute_official_metrics(filtered_group_losses, group_labels)
            parts.append(f"official_filtered[{_format_official_metrics(official)}]")
        except (IndexError, ValueError, RuntimeError):
            parts.append("official_metrics=warming_up")
    return "metrics=" + "; ".join(parts)


def _gpu_peak_memory_mib(device):
    if device.type != "cuda":
        return "n/a"
    return f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.0f}MiB"


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    enc_checkpoint_key = args_pretrain.get('enc_checkpoint_key', 'encoder')
    pred_checkpoint_key = args_pretrain.get('pred_checkpoint_key', 'predictor')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    wide_SiLU = args_pretrain.get('wide_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    is_causal = args_pretrain.get('is_causal', False)
    pred_is_causal = args_pretrain.get('pred_is_causal', False)
    pred_depth = args_pretrain.get('pred_depth', 12)
    optical_qkv = args_eval.get('optical_qkv', {})
    predictor_checkpoint = args_eval.get("predictor_checkpoint")
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- MASK

    # -- DATA
    args_data = args_eval.get('data')
    resolution = args_data.get('resolution', 224)
    batch_size = args_data.get('batch_size', 1)
    stride_sliding_window = args_data.get('stride_sliding_window',2)
    use_bfloat16 = args_data.get('use_bfloat16')
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)

    all_context_lengths = args_data.get('context_lengths', 4)
    eval_frame_steps = args_data.get('frame_steps', 4)

    normalize_enc =  args_data.get('normalize_context', False)


    # -- EXPERIMENT
    eval_tag = args_eval.get('tag', None)
    mode = args_eval.get('mode', 'all')
    assert mode in ['all','losses','metrics']
    dataset = args_eval.get('dataset', 'intphys')
    assert dataset in ['intphys','grasp','grasp_v2','inflevel_lab','inflevel_lab_priming']
    is_mae = args_eval.get('is_mae', False)
    mae_decoder_blocks = args_eval.get('mae_decoder_blocks', -1)
    normalize_targets =args_eval.get('normalize_targets',True)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = args_eval.get("output_dir") or os.path.join(pretrain_folder, 'intuitive_physics/')
    if eval_tag is not None:
        folder = os.path.join(folder, f"{dataset}-{eval_tag}")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    progress_log_file = os.path.join(folder, f'{tag}_r{rank}.log')
    progress_log = SynchronizedProgressLog(progress_log_file, print_enabled=(rank == 0))
    progress_log.log(
        f"START rank={rank}/{world_size} device={device} dataset={dataset} "
        f"model={model_name} checkpoint={pretrained_path} batch_size={batch_size} "
        f"resolution={resolution} frames_per_clip={eval_frames_per_clip} "
        f"log_file={progress_log_file}"
    )
    # Initialize model

    # -- pretrained encoder (frozen)
    encoder,target_encoder, predictor = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=eval_frames_per_clip,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        pred_depth=pred_depth,
        uniform_power=uniform_power,
        enc_checkpoint_key=enc_checkpoint_key,
        pred_checkpoint_key=pred_checkpoint_key,
        use_SiLU=use_SiLU,
        wide_SiLU=wide_SiLU,
        use_sdpa=use_sdpa,
        is_mae=is_mae,
        optical_qkv=optical_qkv,
        predictor_checkpoint=predictor_checkpoint,
        predictor_type=args_eval.get("predictor_type", "onn_feedback"),
        onn_feedback_config=args_eval.get(
            "onn", args_eval.get("onn_feedback", optical_qkv)
        ),
    )

    if not is_mae:
        target_encoder.eval()
        predictor.eval()
        for p in target_encoder.parameters():
            p.requires_grad = False

        for p in predictor.parameters():
            p.requires_grad = False

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Initialize data loaders
    #TODO: Might be issues here, we want a resize more than a center crop
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1/2,2/1] if  dataset in ["inflevel_lab","inflevel_lab_priming"] else [1/1, 1/1],
        random_resize_scale=[1.0, 1.0],
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=resolution)

    if not isinstance(eval_frame_steps, list):
        eval_frame_steps = [eval_frame_steps]

    init_logger = True
    for block in PROPERTIES_BY_DATASET[dataset]:
        for frame_step in eval_frame_steps:
            progress_log.log(f"STAGE_START block={block} frame_step={frame_step}")
            logger.info(f"Doing property {block} ...")
            if mode in ['losses','all']:
                logger.info(f"Extracting loss ...")
                all_losses,all_labels,official_group_losses,official_group_labels  = extract_losses(
                    device=device,
                    encoder=encoder,
                    target_encoder=target_encoder,
                    predictor=predictor,
                    transform=transform,
                    use_bfloat16=use_bfloat16,
                    block=block,
                    frame_step=frame_step,
                    context_lengths=all_context_lengths,
                    batch_size=batch_size,
                    frames_per_clip=eval_frames_per_clip,
                    stride=stride_sliding_window,
                    world_size=world_size,
                    rank=rank,
                    normalize_enc=normalize_enc,
                    dataset=dataset,
                    is_mae=is_mae,
                    mae_decoder_blocks=mae_decoder_blocks,
                    patch_size=patch_size,
                    resolution=resolution,
                    normalize_targets=normalize_targets,
                    progress_log=progress_log,
                    mask_mode=args_eval.get("mask_mode"),
                    mask_config=args_eval.get("mask"))

                all_losses = batch_all_gather(all_losses).cpu()
                all_labels = batch_all_gather(all_labels).cpu()
                if dataset == "intphys":
                    official_group_losses = batch_all_gather(
                        official_group_losses
                    ).cpu()
                    official_group_labels = batch_all_gather(
                        official_group_labels
                    ).cpu()
                    official_group_losses, official_group_labels = (
                        normalize_official_group_tensors(
                            official_group_losses, official_group_labels
                        )
                    )

                if rank == 0 :
                    torch.save({"block":block,
                                "frame_step":frame_step,
                                "context_lengths":all_context_lengths,
                                "losses":all_losses,
                                "labels":all_labels,
                                "official_group_losses":official_group_losses,
                                "official_group_labels":official_group_labels,
                                },
                                os.path.join(folder, f'losses_{block}_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))

            if mode in ['metrics','all']:
                logger.info(f"Computing metrics ...")
                if mode == "metrics":
                    data = torch.load(os.path.join(folder, f'losses_{block}_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
                    all_losses = data["losses"]
                    all_labels = data["labels"]
                    official_group_losses = data.get("official_group_losses")
                    official_group_labels = data.get("official_group_labels")

                if dataset == "intphys":
                    if official_group_losses is None or official_group_labels is None:
                        raise RuntimeError(
                            "This loss cache does not contain complete IntPhys "
                            "quadruplets. Re-run with mode=all or mode=losses."
                        )
                    official_group_losses, official_group_labels = (
                        normalize_official_group_tensors(
                            official_group_losses, official_group_labels
                        )
                    )

                for i,context in enumerate(all_context_lengths):
                    losses = all_losses[:,i]
                    metrics = compute_metrics(losses,all_labels)
                    if dataset == "intphys":
                        metrics.update(
                            compute_official_metrics(
                                official_group_losses[:, :, i, :],
                                official_group_labels,
                            )
                        )

                    if init_logger and rank == 0:
                        keys = metrics.keys()
                        csv_logger = CSVLogger(log_file,
                                ('%s', 'Block'),
                                ('%s', 'Context length(s)'),
                                ('%d', 'Frame skip'),
                                *[
                                    ('%s', key)
                                    if isinstance(metrics[key], str)
                                    else ('%.5f', key)
                                    for key in keys
                                ],
                                    delim=';',)
                        init_logger = False
                    if rank == 0:
                        # Weguarantee that the order is the same as before, rather than using .values()
                        csv_logger.log(block,context,frame_step,*[metrics[key] for key in keys])
                        progress_message = (
                            f"METRIC block={block} frame_step={frame_step} "
                            f"context={context} {_format_key_metrics(metrics)}"
                        )
                        if dataset == "intphys":
                            progress_message += (
                                f" { _format_official_metrics(metrics) }"
                            )
                        progress_log.log(progress_message)


                filtered = all_losses.min(1)[0]
                metrics = compute_metrics(filtered,all_labels)
                if dataset == "intphys":
                    filtered_group_losses = official_group_losses.min(2)[0]
                    metrics.update(
                        compute_official_metrics(
                            filtered_group_losses,
                            official_group_labels,
                        )
                    )
                if rank == 0:
                    csv_logger.log(block,"Filtered",frame_step,*[metrics[key] for key in keys])
                    progress_message = (
                        f"METRIC block={block} frame_step={frame_step} "
                        f"context=Filtered {_format_key_metrics(metrics)}"
                    )
                    if dataset == "intphys":
                        progress_message += (
                            f" {_format_official_metrics(metrics)}"
                        )
                    progress_log.log(progress_message)





            progress_log.log(f"STAGE_DONE block={block} frame_step={frame_step}")

    progress_log.log("RUN_DONE")
    progress_log.close()


def normalize_official_group_tensors(group_losses, group_labels):
    """Return official tensors with shapes [groups, 4, ...] and [groups, 4].

    Older loss caches created by the first implementation flattened the group
    axis with torch.concat. They are losslessly restored here.
    """
    group_losses = torch.as_tensor(group_losses)
    group_labels = torch.as_tensor(group_labels)

    if group_losses.ndim == 3 and group_labels.ndim == 1:
        if group_losses.shape[0] != group_labels.shape[0]:
            raise ValueError(
                "official group loss/label counts do not match"
            )
        if group_losses.shape[0] % 4 != 0:
            raise ValueError(
                "flattened official IntPhys cache is not divisible into quadruplets"
            )
        group_losses = group_losses.reshape(
            -1, 4, group_losses.shape[1], group_losses.shape[2]
        )
        group_labels = group_labels.reshape(-1, 4)

    if group_losses.ndim != 4 or group_labels.ndim != 2:
        raise ValueError(
            "official IntPhys tensors must have shapes "
            "[num_groups, 4, context, window] and [num_groups, 4]"
        )
    return group_losses, group_labels

def compute_official_metrics(group_losses, group_labels):
    """Compute official IntPhys metrics from complete 4-video groups.

    group_losses has shape [num_groups, 4, num_windows] and group_labels
    uses True for possible and False for impossible. The native V-JEPA
    loss is converted only by reversing its direction: plausibility=-loss.
    """
    group_losses = torch.as_tensor(group_losses).detach().cpu()
    group_labels = torch.as_tensor(group_labels).detach().cpu().bool()

    if group_losses.ndim != 3 or group_losses.shape[1] != 4:
        raise ValueError(
            "official IntPhys group_losses must have shape [num_groups, 4, num_windows]"
        )
    if group_labels.shape != group_losses.shape[:2]:
        raise ValueError(
            "official IntPhys group labels must have shape [num_groups, 4]"
        )
    if group_losses.shape[0] == 0:
        raise ValueError("official IntPhys metrics require at least one group")

    possible_counts = group_labels.sum(dim=1)
    impossible_counts = (~group_labels).sum(dim=1)
    if not torch.all(possible_counts == 2) or not torch.all(impossible_counts == 2):
        raise ValueError(
            "official IntPhys groups must contain exactly 2 possible and 2 impossible videos"
        )

    labels_np = group_labels.numpy().astype(np.int64)
    results = {
        "official_metric_definition": "official_intphys",
        "official_relative_metric": "quadruplet_sum",
        "official_absolute_metric": "1_minus_auc",
        "official_score_definition": "plausibility=-surprise",
        "official_aggregation": "mean_and_max",
        "official_num_groups": int(group_losses.shape[0]),
        "official_num_videos": int(group_losses.shape[0] * group_losses.shape[1]),
        "official_num_possible": int(group_labels.sum().item()),
        "official_num_impossible": int((~group_labels).sum().item()),
    }

    for aggregation in ("mean", "max"):
        if aggregation == "mean":
            video_plausibility = (-group_losses.mean(dim=-1)).numpy()
        else:
            video_plausibility = (-group_losses.max(dim=-1).values).numpy()

        possible_sum = np.where(
            labels_np, video_plausibility, 0.0
        ).sum(axis=1)
        impossible_sum = np.where(
            ~labels_np.astype(bool), video_plausibility, 0.0
        ).sum(axis=1)
        relative_error = float(np.mean(possible_sum < impossible_sum))
        auc_value = float(
            roc_auc_score(labels_np.reshape(-1), video_plausibility.reshape(-1))
        )

        results[f"official_{aggregation}_LR"] = relative_error
        results[f"official_{aggregation}_relative_accuracy"] = 1.0 - relative_error
        results[f"official_{aggregation}_AUC"] = auc_value
        results[f"official_{aggregation}_LA"] = 1.0 - auc_value

    return results

def compute_metrics(losses,labels):
    metrics = {}
    loss_real = losses[torch.where(labels == 1)]
    loss_fake = losses[torch.where(labels == 0)]

    # Average loss metrics

    acc_pairwise_mean = (loss_real.mean(1) < loss_fake.mean(1)).sum()/loss_real.shape[0] * 100
    acc_pairwise_max = (loss_real.max(1)[0] < loss_fake.max(1)[0]).sum()/loss_real.shape[0] * 100


    metrics["Relative Accuracy (avg)"] = acc_pairwise_mean.item()
    metrics["Relative Accuracy (max)"] = acc_pairwise_max.item()

    #Compute single video classif

    #Calibrate on real videos
    data1= loss_real.max(1)[0]
    data2 = loss_fake.max(1)[0]
    #Get 90% loss
    thresh = data1.sort()[0][int(np.ceil(0.90*len(data1)))]
    #logger.info(f"Threshold: {thresh}")
    accuracy_abs = ((data1 < thresh).sum() + (data2 > thresh).sum())/ (data1.shape[0] + data2.shape[0]) * 100

    metrics["Absolute Accuracy (max)"] = accuracy_abs.item()
    metrics["Classifier threhshold"] = thresh

    threshs = np.linspace(data1.min().item(),data2.max().item(),100)
    accs = []
    for thresh in threshs:
        accs.append(((data1 < thresh).sum() + (data2 > thresh).sum())/ (data1.shape[0] + data1.shape[0]))
    best_accuracy_abs = torch.max(torch.Tensor(accs))*100
    oracle_thresh = threshs[torch.argmax(torch.Tensor(accs))]

    metrics["Best Absolute Accuracy (max)"] = best_accuracy_abs
    metrics["Best Classifier threhshold"] = oracle_thresh
    # AUPRC

    precision_max, recall_max, thresholds = precision_recall_curve(labels, -losses.max(1)[0])
    precision_mean, recall_mean, thresholds = precision_recall_curve(labels, -losses.mean(1))
    auprc_max = auc(recall_max,precision_max)
    auprc_mean = auc(recall_mean,precision_mean)

    metrics["AUPRC (avg)"] = auprc_mean
    metrics["AUPRC (max)"] = auprc_max

    # AUROC
    fpr_max, tpr_max, thresholds = roc_curve(labels, -losses.max(1)[0])
    fpr_mean, tpr_mean, thresholds = roc_curve(labels, -losses.mean(1))
    auroc_max = auc(fpr_max,tpr_max)
    auroc_mean = auc(fpr_mean,tpr_mean)

    metrics["AUROC (avg)"] = auroc_mean
    metrics["AUROC (max)"] = auroc_max
    return metrics



@torch.no_grad()
def extract_losses(
    device,
    encoder,
    target_encoder,
    predictor,
    transform,
    block="O1",
    use_bfloat16=False,
    frame_step=1,
    context_lengths=[2],
    batch_size=1,
    frames_per_clip=16,
    stride=2,
    world_size=1,
    rank=0,
    normalize_enc=False,
    dataset="intphys",
    is_mae=False,
    mae_decoder_blocks=-1,
    patch_size=16,
    resolution=224,
    normalize_targets=True,
    progress_log=None,
    mask_mode=None,
    mask_config=None,
):
    if progress_log is None:
        progress_log = SynchronizedProgressLog(
            os.path.join(os.getcwd(), "evaluation_progress.log"),
            print_enabled=True,
        )
        close_progress_log = True
    else:
        close_progress_log = False

    progress_log.log(f"CONTEXT_LENGTHS values={context_lengths}")
    if mask_mode is not None:
        mask_mode = normalize_mask_mode(mask_mode)
        progress_log.log(f"MASK_MODE mode={mask_mode}")

    sampling_rate, num_frames = frame_step, 99 // frame_step

    progress_log.log(f"SAMPLING rate=1/{sampling_rate} frames")

    if dataset == "intphys":
        data_name = f"IntPhys-dev-{block}"
    elif dataset == "grasp":
        data_name = f'GRASP-level-2'
    elif dataset == 'inflevel_lab':
        data_name = f"InfLevel-lab"
    elif dataset == 'inflevel_lab_priming':
        data_name = f"InfLevel-lab-priming"

    (data,unsupervised_sampler) = init_data(
        batch_size = batch_size,
        transform=transform,
        data=data_name,
        property=block, # Only used for GRASP
        collator=None,
        pin_mem=True,
        num_workers=8,
        world_size=world_size,
        rank=rank,
        root_path=get_dataset_paths([data_name])[0],
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        deterministic=True,
        log_dir=None)


    loader = iter(data)
    total_batches = len(data)
    mask_collator = None
    if mask_mode is not None and not is_mae:
        mask_collator = make_mask_collator(
            mask_mode=mask_mode,
            cfgs_mask=mask_config,
            crop_size=(resolution, resolution),
            num_frames=frames_per_clip,
            patch_size=(patch_size, patch_size),
            tubelet_size=2,
        )
    extract_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress_log.log(
        f"BATCH_START block={block} frame_step={frame_step} "
        f"batch_size={batch_size} total_batches={total_batches}"
    )

    all_labels = []
    all_losses = []
    official_group_losses = []
    official_group_labels = []

    for i in range(total_batches):
        udata_labels = next(loader)

        labels = udata_labels[1][0]

        clip = udata_labels[0]
        clip = clip[0].to(device)

        #if we have quadruplets or pairs
        num_videos = clip.shape[0]

        if dataset == "intphys":
            bps = get_breaking_points(clip)
            matches = get_matches(bps)
        elif "grasp" in dataset:
            matches = [[0,1]]
        elif "inflevel" in dataset :
            matches = [[0,1]]

        pieces = clip.unfold(2, frames_per_clip,stride).permute(0,2,-1,1,3,4).contiguous()

        pieces = pieces.flatten(0,1)#.view(-1,3,16,224,224)
        pieces = rearrange(pieces,"b t c h w ->  b c t h w")

        pieces = pieces.contiguous()

        B, C, T, H, W = pieces.shape


        all_losses_ctxt = []
        for CTXT_LEN in context_lengths:

            if mask_collator is None:
                m,m_,full_m = get_time_masks(CTXT_LEN,spatial_size=(patch_size,patch_size),temporal_dim=frames_per_clip,as_bool=is_mae)
                full_m = full_m.unsqueeze(0).to(device)
                m = m.unsqueeze(0).to(device)
                m_ = m_.unsqueeze(0).to(device)
            else:
                random_masks_ctxt, random_masks_tgt = mask_collator.generate_masks(B)
                m = random_masks_ctxt[0].to(device=device, dtype=torch.long)
                m_ = random_masks_tgt[0].to(device=device, dtype=torch.long)
                full_m = torch.arange(1568, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)

            if is_mae:
                masks_enc = m.repeat(B, 1)
                masks_pred = m_.repeat(B, 1)
                full_mask = full_m.repeat(B, 1)
            else:
                masks_enc = [m.repeat(B, 1)]
                masks_pred = [m_.repeat(B, 1)]
                full_mask = [full_m.repeat(B, 1)]

            with autocast_context(device, use_bfloat16):
                if is_mae:
                    if mae_decoder_blocks == -1:
                        mean = torch.as_tensor((0.485, 0.456, 0.406)).to(device)[None, :, None, None, None]
                        std = torch.as_tensor((0.229, 0.224, 0.225)).to(device)[None, :, None, None, None]
                        unnorm_videos = pieces * std + mean  # in [0, 1]

                        videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                        var = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                        mean = videos_squeeze.mean(dim=-2, keepdim=True)
                        videos_norm = (videos_squeeze - mean) / (var)
                        # we find that the mean is about 0.48 and standard deviation is about 0.08.
                        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                        B, _, C = videos_patch.shape
                        targets = videos_patch[masks_pred].reshape(B, -1, C)
                    else:
                        targets = encoder(pieces,~full_m.repeat(B,1),decoder_blocks=mae_decoder_blocks)
                        B, _, C = targets.shape
                        targets = targets[masks_pred].reshape(B, -1, C)

                    preds = encoder(pieces,masks_pred,decoder_blocks=mae_decoder_blocks)

                    preds = preds.view(num_videos,-1,*preds.shape[1:])
                    targets = targets.view(num_videos,-1,*targets.shape[1:])


                else:
                    h = target_encoder(pieces,full_mask)[0]
                    if normalize_targets:
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                    # -- create targets (masked regions of h)
                    targets = apply_masks(h, masks_pred, concat=False)


                    context = encoder(pieces, masks_enc)
                    if normalize_enc:
                        z_ = []
                        for zi in context:
                            z_ += [F.layer_norm(zi,(zi.size(-1),))]
                        context = z_

                    preds = predictor(context, targets, masks_enc, masks_pred)


                    preds = preds[0].view(num_videos,-1,*preds[0].shape[1:])
                    targets = targets[0].view(num_videos,-1,*targets[0].shape[1:])
            all_losses_ctxt.append(F.l1_loss(preds,targets,reduction="none").mean((2,3)).detach())
        losses = torch.stack(all_losses_ctxt)
        losses = losses.permute(1,0,2)

        if dataset == "intphys":
            official_group_losses.append(losses)
            official_group_labels.append(labels)

        # Always append by matches for easy filtering later
        # i.e. all_losses[all_labels == 0] and 1 are matched pairwise
        for match in matches:
            all_losses.append(losses[match])
            all_labels.append(labels[match])

        current_batch = i + 1
        if current_batch % 10 == 0 or current_batch == total_batches:
            elapsed = time.perf_counter() - extract_start
            running_metrics = _format_running_metrics(
                all_losses,
                all_labels,
                context_lengths,
                official_group_losses if dataset == "intphys" else None,
                official_group_labels if dataset == "intphys" else None,
            )
            progress_log.log(
                f"BATCH_PROGRESS block={block} frame_step={frame_step} "
                f"batch={current_batch}/{total_batches} batch_size={batch_size} "
                f"elapsed={elapsed:.1f}s videos={num_videos} "
                f"max_gpu_mem={_gpu_peak_memory_mib(device)} "
                f"{running_metrics}"
            )
    extract_elapsed = time.perf_counter() - extract_start
    # This padding is only used for InfLevel but ensures easy processing
    # The padding can be removed by filtering end zeros since the loss is never zero
    # This can lead to slighlty innacurate metrics computed from this script
    lengths = []
    for l in all_losses:
        lengths.append(l.size(-1))
    max_length = torch.tensor([max(lengths)]).to(device)
    #We need to sync the max lengths otherwise we can't gather the losses afterwards
    dist.all_reduce(max_length, op=dist.ReduceOp.MAX)

    all_losses = torch.concat(pad_tensors(all_losses,max_length.item()))
    all_labels = torch.concat(all_labels)
    if dataset == "intphys":
        official_group_losses = torch.stack(
            pad_tensors(official_group_losses, max_length.item())
        )
        official_group_labels = torch.stack(official_group_labels)
    else:
        official_group_losses = None
        official_group_labels = None

    progress_log.log(
        f"LOSS_DONE block={block} frame_step={frame_step} "
        f"elapsed={extract_elapsed:.1f}s"
    )
    if close_progress_log:
        progress_log.close()

    return all_losses,all_labels.to(device),official_group_losses,official_group_labels.to(device) if official_group_labels is not None else None



def _load_state_dict_checked(module, state_dict, label):
    normalized = {key.replace('module.', ''): value for key, value in state_dict.items()}
    expected = module.state_dict()
    missing = sorted(set(expected).difference(normalized))
    unexpected = sorted(set(normalized).difference(expected))
    mismatched = sorted(
        key for key in set(expected).intersection(normalized)
        if expected[key].shape != normalized[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{label} checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    return module.load_state_dict(normalized, strict=True)


def load_pretrained(
    encoder,
    target_encoder,
    predictor,
    pretrained,
    enc_checkpoint_key='encoder',
    target_enc_checkpoint_key='target_encoder',
    pred_checkpoint_key='predictor',
    is_mae=False,
    load_predictor=True,
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')

    try:
        enc_pretrained_dict = checkpoint[enc_checkpoint_key]
    except Exception:
        enc_pretrained_dict = checkpoint['encoder']
    msg = _load_state_dict_checked(encoder, enc_pretrained_dict, "encoder")
    logger.info(f'loaded pretrained model with msg: {msg}')
    print(encoder)

    if not is_mae:
        try:
            target_enc_pretrained_dict = checkpoint[target_enc_checkpoint_key]
        except Exception:
            target_enc_pretrained_dict = checkpoint["target_encoder"]
        msg = _load_state_dict_checked(
            target_encoder,
            target_enc_pretrained_dict,
            "target_encoder",
        )
        logger.info(f'loaded pretrained model with msg: {msg}')
        print(target_encoder)

        if load_predictor:
            try:
                pred_pretrained_dict = checkpoint[pred_checkpoint_key]
            except Exception:
                pred_pretrained_dict = checkpoint['predictor']
            msg = _load_state_dict_checked(predictor, pred_pretrained_dict, "predictor")
            logger.info(f'loaded pretrained model with msg: {msg}')
            logger.info(
                f'loaded pretrained predictor from epoch: {checkpoint["epoch"]}\n'
                f' path: {pretrained}'
            )
            print(predictor)
        else:
            logger.info("skipping legacy Predictor checkpoint for ONN feedback mode")

    del checkpoint
    return encoder, target_encoder, predictor


def _is_full_predictor_checkpoint_mode(checkpoint_mode):
    return checkpoint_mode in {"end_to_end_jepa", "electronic_control", "onn_feedback"}


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    wide_SiLU=False,
    is_causal=False,
    pred_is_causal=False,
    uniform_power=False,
    enc_checkpoint_key='encoder',
    pred_checkpoint_key='predictor',
    use_mask_tokens=True,
    pred_embed_dim=384,
    pred_depth=12,
    num_mask_tokens=2,
    is_mae=False,
    optical_qkv=None,
    predictor_checkpoint=None,
    predictor_type="vit_transformer",
    onn_feedback_config=None,
):
    optical_qkv = optical_qkv or {}
    if is_mae:
        encoder = videomae.__dict__[model_name]()
        target_encoder = None
        predictor = None
    else:
        optical_enabled = bool(optical_qkv.get("enabled", False))
        qkv_backend = (
            optical_qkv.get("qkv_backend", "electronic")
            if optical_enabled else "electronic"
        )
        replace_layers = optical_qkv.get("replace_layers", "all")
        optical_config = (
            OpticalQKVConfig.from_mapping(optical_qkv)
            if qkv_backend == "fsonn_tdm" else None
        )
        encoder = vit.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=frames_per_clip,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_SiLU=use_SiLU,
            wide_SiLU=wide_SiLU,
            is_causal=is_causal,
        )
        target_encoder = copy.deepcopy(encoder)
        encoder = MultiMaskWrapper(encoder)
        target_encoder = MultiMaskWrapper(target_encoder)

        if predictor_type == "onn_feedback":
            onn_config = dict(onn_feedback_config or optical_qkv)
            predictor = vit_pred.ONNFeedbackPredictor(
                img_size=crop_size,
                patch_size=patch_size,
                num_frames=frames_per_clip,
                tubelet_size=tubelet_size,
                embed_dim=encoder.backbone.embed_dim,
                predictor_embed_dim=pred_embed_dim,
                num_tokens=1568,
                num_chunks=8,
                chunk_tokens=196,
                output_mlp_hidden_dim=int(
                    onn_config.get("output_mlp_hidden_dim", pred_embed_dim)
                ),
                feedback_mode=onn_config.get("feedback_mode", "fixed_middle"),
                feedback_layer_index=onn_config.get("feedback_layer_index"),
                uniform_power=uniform_power,
                optical_config=onn_config,
            )
        elif predictor_type == "vit_transformer":
            use_rope = 'rope' in model_name
            rope_is_1D = 'rope1D' in model_name
            predictor = vit_pred.vit_predictor(
                img_size=crop_size,
                use_mask_tokens=use_mask_tokens,
                is_causal=pred_is_causal,
                patch_size=patch_size,
                num_frames=frames_per_clip,
                tubelet_size=tubelet_size,
                embed_dim=encoder.backbone.embed_dim,
                predictor_embed_dim=pred_embed_dim,
                depth=pred_depth,
                num_heads=encoder.backbone.num_heads,
                uniform_power=uniform_power,
                num_mask_tokens=num_mask_tokens,
                zero_init_mask_tokens=True,
                use_sdpa=use_sdpa,
                use_SiLU=use_SiLU,
                use_rope=use_rope,
                rope_is_1D=rope_is_1D,
                wide_SiLU=wide_SiLU,
                qkv_backend="electronic",
                optical_config=None,
                replace_layers="all",
            )
        else:
            raise ValueError(f"unsupported predictor_type: {predictor_type}")
        predictor = PredictorMultiMaskWrapper(predictor)
        predictor.to(device)
        target_encoder.to(device)

    encoder.to(device)
    encoder, target_encoder, predictor = load_pretrained(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        pretrained=pretrained,
        enc_checkpoint_key=enc_checkpoint_key,
        pred_checkpoint_key=pred_checkpoint_key,
        is_mae=is_mae,
        load_predictor=(predictor_type != "onn_feedback"),
    )

    if predictor_type != "onn_feedback" and qkv_backend == "fsonn_tdm":
        vit_pred.install_optical_qkv(
            predictor,
            optical_config=optical_config,
            replace_layers=replace_layers,
        )
    if predictor_checkpoint is not None:
        checkpoint = torch.load(
            predictor_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_mode = checkpoint.get("mode")
        if checkpoint_mode == "realtime_last_node_distillation":
            if qkv_backend != "fsonn_tdm":
                raise ValueError(
                    "realtime optical checkpoint requires qkv_backend=fsonn_tdm"
                )
            load_optical_checkpoint(predictor, checkpoint)
        else:
            if not _is_full_predictor_checkpoint_mode(checkpoint_mode):
                raise ValueError(
                    "trained Predictor checkpoint must have mode=end_to_end_jepa "
                    "or mode=electronic_control "
                    "or mode=realtime_last_node_distillation"
                )
            if "predictor" not in checkpoint:
                raise ValueError(
                    "trained Predictor checkpoint has no full Predictor state"
                )
            _load_state_dict_checked(
                predictor, checkpoint["predictor"], "trained predictor"
            )
    return encoder, target_encoder, predictor
