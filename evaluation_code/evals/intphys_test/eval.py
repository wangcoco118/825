# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy

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
import pandas as pd

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import all_gather
from sklearn.metrics import precision_recall_curve,roc_curve,auc
from scipy.stats import mannwhitneyu,wilcoxon,ttest_rel,ttest_ind
import torch.distributed as dist


from src.utils.tensors import repeat_interleave_batch
from src.masks.utils import apply_masks
import src.models.vision_transformer as vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from evals.intphys_test.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
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

from evals.intphys_test.utils import get_time_masks,get_dataset_paths,batch_all_gather,PROPERTIES_BY_DATASET,pad_tensors
import evals.intphys_test.videomae as videomae

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


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
    assert dataset in ['intphys-test']
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
    folder = os.path.join(pretrain_folder, 'intphys_test/')
    if eval_tag is not None:
        folder = os.path.join(folder, f"{dataset}-{eval_tag}")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
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
        is_mae=is_mae)
    
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
        random_resize_aspect_ratio=[1/1, 1/1],
        random_resize_scale=[1.0, 1.0],
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=resolution)

    if not isinstance(eval_frame_steps, list):
        eval_frame_steps = [eval_frame_steps]

    init_logger = True
    for frame_step in eval_frame_steps:
        if mode in ['losses','all']:
            logger.info(f"Extracting loss ...")
            all_losses,all_labels,tasks  = extract_losses(
                device=device,
                encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                transform=transform,
                use_bfloat16=use_bfloat16,
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
                normalize_targets=normalize_targets)
            
            all_losses = batch_all_gather(all_losses).cpu()
            all_labels = batch_all_gather(all_labels).cpu().numpy().astype(int)
            all_tasks = list(np.array(tasks)[all_labels])
            
            if rank == 0 :
                logger.info('saving')
                logger.info(os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
                torch.save({"frame_step":frame_step,
                            "context_lengths":all_context_lengths,
                            "losses":all_losses,
                            "tasks":all_tasks,
                            },
                            os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
        
        if mode in ['metrics','all']:
            logger.info(f"Computing metrics ...")
            if mode == "metrics":
                data = torch.load(os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
                all_losses = data["losses"]
                all_tasks = data["tasks"]

            filtered = all_losses.min(1)[0]
            metrics = compute_metrics(filtered)

            if rank == 0:
                for metric,values in metrics.items():
                    data = {'tasks':all_tasks, 'surprises': values}
                    df = pd.DataFrame(data)
                    # Save to CSV without headers
                    log_file = os.path.join(folder, f'{metric}_answer.txt')
                    df.to_csv(log_file, index=False, header=False,sep=" ")
            
        
        



@torch.no_grad()
def extract_losses(
    device,
    encoder,
    target_encoder,
    predictor,
    transform,
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
    normalize_targets=True
):
    print(context_lengths)

    sampling_rate,num_frames = frame_step ,99//frame_step
    
    print(f"Sampling rate 1/{sampling_rate} frames")

    if dataset == "intphys-test":
        data_name = f"IntPhys-test" 

    (data,unsupervised_sampler) = init_data(
        batch_size = batch_size,
        transform=transform,
        data=data_name,
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

    all_tasks = []
    all_losses = []

    for i in range(len(loader)):
        udata = next(loader)

        tasks = udata[1]

        clip = udata[0]
        clip = clip.to(device)

        #Batch size
        num_videos = clip.shape[0]

        pieces = clip.unfold(2, frames_per_clip,stride).permute(0,2,-1,1,3,4).contiguous()

        pieces = pieces.flatten(0,1)#.view(-1,3,16,224,224)
        pieces = rearrange(pieces,"b t c h w ->  b c t h w")

        pieces = pieces.contiguous()

        B, C, T, H, W = pieces.shape

        
        all_losses_ctxt = []
        for CTXT_LEN in context_lengths:

            m,m_,full_m = get_time_masks(CTXT_LEN,spatial_size=(patch_size,patch_size),temporal_dim=frames_per_clip,as_bool=is_mae)
            full_m = full_m.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            m_ = m_.unsqueeze(0).to(device)
            
            if is_mae:
                masks_enc = m.repeat(B, 1)
                masks_pred = m_.repeat(B, 1)
                full_mask = full_m.repeat(B, 1)
            else:
                masks_enc = [m.repeat(B, 1)]
                masks_pred = [m_.repeat(B, 1)]
                full_mask = [full_m.repeat(B, 1)]

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                if is_mae: 
                    if mae_decoder_blocks == -1:
                        mean = torch.as_tensor((0.485, 0.456, 0.406)).to(device)[None, :, None, None, None]
                        std = torch.as_tensor((0.229, 0.224, 0.225)).to(device)[None, :, None, None, None]
                        unnorm_videos = pieces * std + mean  # in [0, 1]

                        videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                        var = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                        mean = videos_squeeze.mean(dim=-2, keepdim=True)
                        videos_norm = (videos_squeeze - mean) / (var)
                        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                        B, _, C = videos_patch.shape
                        targets = videos_patch[masks_pred].reshape(B, -1, C)
                    else:
                        targets = encoder(pieces,~full_m.repeat(B,1),decoder_blocks=mae_decoder_blocks)
                        B, _, C = targets.shape
                        targets = targets[masks_pred].reshape(B, -1, C)

                    preds = encoder(pieces,masks_pred,decoder_blocks=mae_decoder_blocks)
                  
                    preds = preds.view(num_videos,-1,*preds.shape[1:])
                    preds = torch.zeros_like(preds,device=preds.device)
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

        # Always append by matches for easy filtering later
        # i.e. all_losses[all_labels == 0] and 1 are matched pairwise
        all_losses.append(losses)
        all_tasks.append(tasks)
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
    all_tasks = torch.concat(all_tasks).flatten()
    logger.info(all_tasks.shape)

    return all_losses,all_tasks.to(device),data.dataset.tasks


def compute_metrics(losses):
    metrics = {}

    average_losses = 1-losses.mean(1)
    max_losses = 1-losses.max(1)[0]

    metrics["maximum_surprise"] = max_losses
    metrics["average_surprise"] = average_losses

    return metrics





def load_pretrained(
    encoder,
    target_encoder,
    predictor,
    pretrained,
    enc_checkpoint_key='encoder',
    target_enc_checkpoint_key='target_encoder',
    pred_checkpoint_key='predictor',
    is_mae=False
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')

    # Load encoder
    try:
        enc_pretrained_dict = checkpoint[enc_checkpoint_key]
    except Exception:
        enc_pretrained_dict = checkpoint['encoder']
    # --
    enc_pretrained_dict = {k.replace('module.', ''): v for k, v in enc_pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in enc_pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif enc_pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            enc_pretrained_dict[k] = v
    msg = encoder.load_state_dict(enc_pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    print(encoder)

    if not is_mae:
        # Load target encoder
        try:
            target_enc_pretrained_dict = checkpoint[target_enc_checkpoint_key]
        except Exception:
            target_enc_pretrained_dict = checkpoint["target_encoder"]
        # --
        target_enc_pretrained_dict = {k.replace('module.', ''): v for k, v in target_enc_pretrained_dict.items()}
        for k, v in target_encoder.state_dict().items():
            if k not in target_enc_pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif target_enc_pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                target_enc_pretrained_dict[k] = v
        msg = target_encoder.load_state_dict(target_enc_pretrained_dict, strict=False)
        logger.info(f'loaded pretrained model with msg: {msg}')
        print(target_encoder)

        # Load predictor
        try:
            pred_pretrained_dict = checkpoint[pred_checkpoint_key]
        except Exception:
            pred_pretrained_dict = checkpoint['predictor']
        # --
        pred_pretrained_dict = {k.replace('module.', ''): v for k, v in pred_pretrained_dict.items()}
        for k, v in predictor.state_dict().items():
            if k not in pred_pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pred_pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pred_pretrained_dict[k] = v
        msg = predictor.load_state_dict(pred_pretrained_dict, strict=False)
        logger.info(f'loaded pretrained model with msg: {msg}')
        logger.info(f'loaded pretrained predictor from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
        print(predictor)

    del checkpoint
    return encoder,target_encoder, predictor


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
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
):
    if is_mae:
        
        encoder = videomae.__dict__[model_name]()
            #decoder_depth=pred_depth
        target_encoder = None
        predictor = None

    else:
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

        use_rope = 'rope' in model_name
        rope_is_1D = 'rope1D' in model_name
        encoder = MultiMaskWrapper(encoder)
        target_encoder = MultiMaskWrapper(target_encoder)
        predictor = vit_pred.__dict__['vit_predictor'](
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
        )
        predictor = PredictorMultiMaskWrapper(predictor)

        
        predictor.to(device)
        target_encoder.to(device)

    encoder.to(device)

    encoder,target_encoder, predictor = load_pretrained(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        pretrained=pretrained,
        enc_checkpoint_key=enc_checkpoint_key,
        pred_checkpoint_key=pred_checkpoint_key,
        is_mae=is_mae,
        )
    return encoder,target_encoder, predictor
