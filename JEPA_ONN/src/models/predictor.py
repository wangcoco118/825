# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block
from src.models.fsonn import FeedbackFSONN, ONNConfig, OpticalQKVConfig, TimeDivisionFSONN
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_rope=False,  # RoPE currently only supported for video 
        use_SiLU=False,
        wide_SiLU=True,
        is_causal=False,
        qkv_backend="electronic",
        optical_config=None,
        replace_layers="all",
        **kwargs
    ):
        super().__init__()
        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for i in range(num_mask_tokens)
            ])

        # Determine positional embedding
        self.input_size = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        if isinstance(optical_config, dict):
            optical_config = OpticalQKVConfig.from_mapping(optical_config)
        if replace_layers == "all" or replace_layers is None:
            optical_layers = set(range(depth))
        else:
            optical_layers = {int(index) for index in replace_layers}
        invalid_layers = optical_layers.difference(range(depth))
        if invalid_layers:
            raise ValueError(f"replace_layers contains invalid block indices: {sorted(invalid_layers)}")
        self.qkv_backend = qkv_backend
        self.optical_layers = tuple(sorted(optical_layers))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.num_patches = num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )
        # Position embedding
        self.uniform_power = uniform_power
        self.predictor_pos_embed = None
        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim),
                requires_grad=False)

        # Attention Blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.SiLU if use_SiLU else nn.GELU,
                wide_SiLU=wide_SiLU,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                grid_size=grid_size,
                grid_depth=grid_depth,
                is_causal=is_causal,
                use_rope=use_rope,
                norm_layer=norm_layer,
                qkv_backend=qkv_backend if i in optical_layers else "electronic",
                optical_config=optical_config if i in optical_layers else None)
            for i in range(depth)])

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):

        # Prepare diffusion noise schedule
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i*(b2-b1)/steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.-_beta
            alpha_scheduler += [_alpha]

        # Sample diffusion time step
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)

        # Normalize features and apply noise
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1.-alpha)**0.5 * torch.randn(x.shape, device=x.device)
        return x

    @torch.no_grad()
    def collect_attention_inputs(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        B = len(ctxt) // len(masks_ctxt)
        x = self.predictor_embed(ctxt)
        _, N_ctxt, _ = x.shape

        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)

        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            pred_tokens = self.mask_tokens[1 % self.num_mask_tokens]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)

        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        masks = torch.cat([torch.cat(masks_ctxt, dim=0), torch.cat(masks_tgt, dim=0)], dim=1)

        attention_inputs = []
        for block in self.predictor_blocks:
            attention_inputs.append(block.norm1(x))
            x = block(x, False, masks)
        return attention_inputs, masks

    def forward_with_nodes(
        self,
        ctxt,
        tgt,
        masks_ctxt,
        masks_tgt,
        target_node,
        mask_index=1,
        qkv_include_bias=True,
    ):
        """Run the serial Predictor path and return one selected node per Block."""
        valid_nodes = {
            "qkv",
            "attention_output",
            "post_output",
            "block_output",
            "predictor_output",
        }
        if target_node not in valid_nodes:
            raise ValueError(f"unsupported target_node: {target_node}")
        assert (masks_ctxt is not None) and (masks_tgt is not None), (
            "Cannot run predictor without mask indices"
        )
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        B = len(ctxt) // len(masks_ctxt)
        x = self.predictor_embed(ctxt)
        _, N_ctxt, _ = x.shape

        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)

        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)

        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)

        nodes = {} if target_node == "predictor_output" else {target_node: []}
        for blk in self.predictor_blocks:
            if target_node == "predictor_output":
                x = blk(x, False, masks)
            else:
                x, captured = blk(
                    x,
                    False,
                    masks,
                    capture_nodes={target_node},
                    qkv_include_bias=qkv_include_bias,
                )
                nodes[target_node].append(captured[target_node])

        x = self.predictor_norm(x)
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        if target_node == "predictor_output":
            nodes[target_node] = x
        return x, nodes


    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1, num_blocks=None):
        """
        :param ctxt: context tokens
        :param tgt: target tokens
        :param masks_ctxt: indices of context tokens in input
        :params masks_tgt: indices of target tokens in input
        """

        assert (masks_ctxt is not None) and (masks_tgt is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]

        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        # Batch Size
        B = len(ctxt) // len(masks_ctxt)

        # Map context tokens to pedictor dimensions
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape

        # Add positional embedding to ctxt tokens
        if self.predictor_pos_embed is not None:
            ctxt_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(ctxt_pos_embed, masks_ctxt)

        # Map target tokens to predictor dimensions & add noise (fwd diffusion)
        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)

        # Add positional embedding to target tokens
        if self.predictor_pos_embed is not None:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_tgt)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
            pred_tokens += pos_embs

        # Concatenate context & target tokens
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # FIXME: this implementation currently assumes masks_ctxt and masks_tgt
        # are alligned 1:1 (ok with MultiMask wrapper on predictor but
        # otherwise will break)
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            x = torch.utils.checkpoint.checkpoint(blk,x,False,masks,use_reentrant=False)
            #x = blk(x, mask=masks)
            if (num_blocks is not None and i >= num_blocks - 1):
                break
        x = self.predictor_norm(x)

        # Return output corresponding to target tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class ONNFeedbackPredictor(nn.Module):
    """Mask-token Predictor with a shared serial ONN feedback core."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1024,
        predictor_embed_dim=384,
        num_tokens=1568,
        num_chunks=8,
        chunk_tokens=196,
        output_mlp_hidden_dim=384,
        feedback_mode="fixed_middle",
        feedback_layer_index=None,
        uniform_power=False,
        optical_config=None,
        onn_core=None,
    ):
        super().__init__()
        if num_tokens != 1568 or num_chunks != 8 or chunk_tokens != 196:
            raise ValueError("ONN feedback Predictor requires 1568 = 8 * 196 tokens")
        if num_chunks * chunk_tokens != num_tokens:
            raise ValueError("num_chunks * chunk_tokens must equal num_tokens")
        if feedback_mode != "fixed_middle":
            raise ValueError("only feedback_mode='fixed_middle' is supported")

        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.num_tokens = int(num_tokens)
        self.num_chunks = int(num_chunks)
        self.chunk_tokens = int(chunk_tokens)
        self.feedback_mode = feedback_mode

        self.predictor_embed = nn.Linear(
            self.embed_dim, self.predictor_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.predictor_embed_dim)
        )
        predictor_pos_embed = torch.zeros(
            1, self.num_tokens, self.predictor_embed_dim
        )
        self.register_buffer(
            "predictor_pos_embed", predictor_pos_embed, persistent=True
        )
        self._init_pos_embed(
            self.predictor_pos_embed,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
        )
        self.predictor_norm = nn.LayerNorm(self.predictor_embed_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.predictor_embed_dim, int(output_mlp_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(output_mlp_hidden_dim), self.embed_dim),
        )

        if onn_core is None:
            config_values = dict(optical_config or {})
            config_values.update(
                {
                    "chunk_tokens": self.chunk_tokens,
                    "input_dim": self.predictor_embed_dim,
                    "output_dim": self.predictor_embed_dim,
                    "grid_height": self.chunk_tokens,
                    "grid_width": self.predictor_embed_dim,
                    "learnable_intensity_offset": True,
                }
            )
            onn_config = ONNConfig.from_mapping(config_values)
            onn_core = FeedbackFSONN(onn_config)
        self.onn_core = onn_core

        if feedback_layer_index is None:
            feedback_layer_index = 2
        self.feedback_layer_index = int(feedback_layer_index)
        slm_layers = getattr(self.onn_core, "slm_layers", ())
        if slm_layers and not 0 <= self.feedback_layer_index < len(slm_layers):
            raise ValueError("feedback_layer_index must identify an existing SLM layer")
        self.feedback_adapter = nn.Identity()
        self.last_trace = {}

    def _init_pos_embed(
        self,
        pos_embed,
        img_size,
        patch_size,
        num_frames,
        tubelet_size,
        uniform_power,
    ):
        grid_size = int(img_size) // int(patch_size)
        grid_depth = int(num_frames) // int(tubelet_size)
        if grid_size * grid_size * grid_depth != self.num_tokens:
            raise ValueError(
                "img_size, patch_size, num_frames and tubelet_size must produce "
                "1568 tokens"
            )
        sincos = get_3d_sincos_pos_embed(
            self.predictor_embed_dim,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=uniform_power,
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _validate_masks(self, masks_ctxt, masks_tgt, batch_size):
        if masks_ctxt.ndim != 2 or masks_tgt.ndim != 2:
            raise ValueError("masks_ctxt and masks_tgt must have shape [B, N]")
        if masks_ctxt.shape[0] != batch_size or masks_tgt.shape[0] != batch_size:
            raise ValueError("mask batch dimensions must match context")
        if masks_ctxt.dtype != torch.long or masks_tgt.dtype != torch.long:
            raise TypeError("mask indices must use torch.long")
        if torch.any(masks_ctxt < 0) or torch.any(masks_ctxt >= self.num_tokens):
            raise ValueError("masks_ctxt indices must be in [0,1567]")
        if torch.any(masks_tgt < 0) or torch.any(masks_tgt >= self.num_tokens):
            raise ValueError("masks_tgt indices must be in [0,1567]")

        missing_counts = []
        for batch_index in range(batch_size):
            merged = torch.cat(
                (masks_ctxt[batch_index], masks_tgt[batch_index]), dim=0
            )
            unique_count = torch.unique(merged).numel()
            if unique_count != merged.numel():
                raise ValueError("masks_ctxt and masks_tgt overlap or duplicate a token")
            missing_counts.append(self.num_tokens - unique_count)
        return missing_counts

    def forward(
        self,
        ctxt,
        tgt=None,
        masks_ctxt=None,
        masks_tgt=None,
        mask_index=0,
        num_blocks=None,
    ):
        del tgt, mask_index, num_blocks
        if ctxt.ndim != 3 or ctxt.shape[-1] != self.embed_dim:
            raise ValueError(
                f"ctxt must have shape [B,N,{self.embed_dim}]"
            )
        if masks_ctxt is None or masks_tgt is None:
            raise ValueError("ONN feedback Predictor requires both mask indices")
        missing_counts = self._validate_masks(
            masks_ctxt, masks_tgt, ctxt.shape[0]
        )

        batch_size = ctxt.shape[0]
        context_384 = self.predictor_embed(ctxt)
        dense_input = torch.zeros(
            batch_size,
            self.num_tokens,
            self.predictor_embed_dim,
            device=ctxt.device,
            dtype=context_384.dtype,
        )
        dense_input.scatter_(
            1,
            masks_ctxt.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim),
            context_384,
        )
        target_placeholder = self.mask_token.expand(
            batch_size, masks_tgt.shape[1], self.predictor_embed_dim
        )
        dense_input.scatter_(
            1,
            masks_tgt.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim),
            target_placeholder,
        )
        dense_input = dense_input + self.predictor_pos_embed.to(
            device=ctxt.device, dtype=dense_input.dtype
        )
        chunks = dense_input.reshape(
            batch_size, self.num_chunks, self.chunk_tokens, self.predictor_embed_dim
        )

        feedback = None
        outputs = []
        for chunk_index in range(self.num_chunks):
            x_chunk = chunks[:, chunk_index]
            result = self.onn_core(
                x_chunk,
                feedback=feedback,
                feedback_layer_index=self.feedback_layer_index,
            )
            if isinstance(result, (tuple, list)):
                y_chunk, feedback_source = result
            else:
                y_chunk, feedback_source = result, result
            if y_chunk.shape != x_chunk.shape:
                raise ValueError(
                    "ONN output must have shape "
                    f"{tuple(x_chunk.shape)}, got {tuple(y_chunk.shape)}"
                )
            if not torch.isfinite(y_chunk).all():
                raise FloatingPointError("ONN output contains NaN or Inf")
            outputs.append(y_chunk)
            feedback = self.feedback_adapter(feedback_source)

        dense_output = torch.cat(outputs, dim=1)
        dense_output = self.predictor_norm(dense_output)
        pred_tgt_384 = torch.gather(
            dense_output,
            1,
            masks_tgt.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim),
        )
        pred_tgt_1024 = self.output_mlp(pred_tgt_384)
        self.last_trace = {
            "ctxt_shape": tuple(ctxt.shape),
            "context_384_shape": tuple(context_384.shape),
            "onn_input_shape": tuple(dense_input.shape),
            "chunk_shape": tuple(chunks.shape),
            "dense_output_shape": tuple(dense_output.shape),
            "pred_tgt_384_shape": tuple(pred_tgt_384.shape),
            "pred_tgt_1024_shape": tuple(pred_tgt_1024.shape),
            "n_ctxt": int(masks_ctxt.shape[1]),
            "n_tgt": int(masks_tgt.shape[1]),
            "missing_count": (
                missing_counts[0]
                if len(set(missing_counts)) <= 1
                else missing_counts
            ),
            "current_chunk": self.num_chunks,
            "num_chunks": self.num_chunks,
            "feedback_layer_index": self.feedback_layer_index,
        }
        return pred_tgt_1024


def install_optical_qkv(
    predictor: nn.Module,
    optical_config: OpticalQKVConfig,
    replace_layers="all",
) -> nn.Module:
    """Replace selected loaded electronic QKV modules after checkpoint loading."""
    core = predictor.backbone if hasattr(predictor, "backbone") else predictor
    depth = len(core.predictor_blocks)
    if replace_layers == "all" or replace_layers is None:
        selected = set(range(depth))
    else:
        selected = {int(index) for index in replace_layers}
    invalid = selected.difference(range(depth))
    if invalid:
        raise ValueError(f"replace_layers contains invalid block indices: {sorted(invalid)}")

    for index in sorted(selected):
        attention = core.predictor_blocks[index].attn
        if not hasattr(attention, "qkv") or attention.qkv is None:
            raise ValueError(f"Predictor block {index} does not have an electronic QKV module")
        if not isinstance(attention.qkv, nn.Linear):
            raise TypeError(f"Predictor block {index} QKV is not an nn.Linear")
        if attention.qkv.in_features != optical_config.input_dim:
            raise ValueError(f"optical input_dim does not match Predictor block {index}")
        device = attention.qkv.weight.device
        attention.qkv = None
        attention.qkv_backend = "fsonn_tdm"
        attention.optical_qkv = TimeDivisionFSONN(optical_config).to(device=device)
    core.optical_layers = tuple(sorted(selected))
    core.qkv_backend = "fsonn_tdm"
    return predictor


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
