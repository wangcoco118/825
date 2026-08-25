# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import drop_path

from einops import rearrange, repeat

from src.models.utils.rope import RotaryEmbedding
from src.models.fsonn import OpticalQKVConfig, TimeDivisionFSONN
import sys

#disable if using python 3.12 or higher
#@torch.compile(disable=sys.version_info[1] >= 12)
def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, 'Embedding dimension must be a multiple of 2 for block matrix rotation'

    # -- compute angle for each position
    omega = torch.arange(D // 2).to(x)
    omega /= D / 2.
    omega = 1. / 10000**omega   # (D/2,)
    freq = torch.einsum('..., f -> ... f', pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    emb_sin = repeat(emb_sin, '... d -> ... (d r)', r=2)  # (..., N, D)
    emb_cos = repeat(emb_cos, '... d -> ... (d r)', r=2)  # (..., N, D)
    # --
    y = rearrange(x, '... (d r) -> ... d r', r=2)
    y1, y2 = y.unbind(dim=-1,)
    y = torch.stack((-y2, y1), dim=-1)
    y = rearrange(y, '... d r -> ... (d r)')
    return (x * emb_cos) + (y * emb_sin)

#@torch.compile(disable=sys.version_info[1] >= 12)
def rotate_queries_and_keys(q,k, pos):
    B, num_heads, N, D = q.size()
    assert D % 2 == 0, 'Embedding dimension must be a multiple of 2 for block matrix rotation'

    # -- compute angle for each position
    omega = torch.arange(D // 2).to(q)
    omega /= D / 2.
    omega = 1. / 10000**omega   # (D/2,)
    freq = torch.einsum('..., f -> ... f', pos, omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    emb_sin = repeat(emb_sin, '... d -> ... (d r)', r=2)  # (..., N, D)
    emb_cos = repeat(emb_cos, '... d -> ... (d r)', r=2)  # (..., N, D)
    # --
    q1, q2 = q.chunk(2, dim=-1)
    qy= torch.cat((-q2, q1), dim=-1)
    q = (q * emb_cos) + (qy * emb_sin)
    
    k1, k2 = k.chunk(2, dim=-1)
    ky= torch.cat((-k2, k1), dim=-1)
    k = (k * emb_cos) + (ky * emb_sin)
    
    return q,k

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.,
        wide_SiLU=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_SiLU:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)

class RoPEAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True,
        grid_size=14,
        grid_depth=8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = d_dim = head_dim // 3
        self.h_dim = h_dim = head_dim // 3
        self.w_dim = w_dim = head_dim // 3
        self.d_rotary_emb = RotaryEmbedding(dim=d_dim)
        self.h_rotary_emb = RotaryEmbedding(dim=h_dim)
        self.w_rotary_emb = RotaryEmbedding(dim=w_dim)
        self.grid_size = grid_size
        self.grid_depth = grid_depth

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_size*self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_size*self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        # --
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def separate_positions(self, ids):
        tokens_per_frame = int(self.grid_size*self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        # --
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask)
        else:
            mask = torch.arange(int(self.grid_depth*self.grid_size*self.grid_size)).to(x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask)
        """
        s = 0
        # Rotate depth
        qd = self.d_rotary_emb.rotate_queries_or_keys(
            q[:, :, :, s:s+self.d_dim], seq=d_mask)
        kd = self.d_rotary_emb.rotate_queries_or_keys(
            k[:, :, :, s:s+self.d_dim], seq=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = self.h_rotary_emb.rotate_queries_or_keys(
            q[:, :, :, s:s+self.h_dim], seq=h_mask)
        kh = self.h_rotary_emb.rotate_queries_or_keys(
            k[:, :, :, s:s+self.h_dim], seq=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = self.w_rotary_emb.rotate_queries_or_keys(
            q[:, :, :, s:s+self.w_dim], seq=w_mask)
        kw = self.w_rotary_emb.rotate_queries_or_keys(
            k[:, :, :, s:s+self.w_dim], seq=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[:, :, :, s:]
            kr = k[:, :, :, s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)
        """
        s = 0
        # Rotate depth
        q[:, :, :, s:s+self.d_dim] = self.d_rotary_emb.rotate_queries_or_keys(
            q[:, :, :, s:s+self.d_dim], seq=d_mask)
        k[:, :, :, s:s+self.d_dim] = self.d_rotary_emb.rotate_queries_or_keys(
            k[:, :, :, s:s+self.d_dim], seq=d_mask)
        s += self.d_dim
        # Rotate height dim
        q[:, :, :, s:s+self.h_dim] = self.h_rotary_emb.rotate_queries_or_keys(
            q[:, :, :, s:s+self.h_dim], seq=h_mask)
        k[:, :, :, s:s+self.h_dim] = self.h_rotary_emb.rotate_queries_or_keys(
            k[:, :, :, s:s+self.h_dim], seq=h_mask)
        s += self.h_dim
        # Rotate width dim
        q[:, :, :, s:s+self.w_dim] = self.w_rotary_emb.rotate_queries_or_keys(
            q[:, :, :, s:s+self.w_dim], seq=w_mask)
        k[:, :, :, s:s+self.w_dim] = self.w_rotary_emb.rotate_queries_or_keys(
            k[:, :, :, s:s+self.w_dim], seq=w_mask)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True,
        is_causal=False,
        qkv_backend="electronic",
        optical_config=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_backend = qkv_backend
        if qkv_backend == "electronic":
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.optical_qkv = None
        elif qkv_backend == "fsonn_tdm":
            if optical_config is None:
                raise ValueError("optical_config is required for qkv_backend=fsonn_tdm")
            if optical_config.input_dim != dim:
                raise ValueError("optical input_dim must match attention dim")
            self.qkv = None
            self.optical_qkv = TimeDivisionFSONN(optical_config)
        else:
            raise ValueError(f"unsupported qkv_backend: {qkv_backend}")
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, return_proj_output=False):
        B, N, C = x.shape
        qkv_source = self.qkv(x) if self.qkv_backend == "electronic" else self.optical_qkv(x)
        qkv = qkv_source.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        proj_output = x
        x = self.proj_drop(x)
        if return_proj_output:
            return x, attn, proj_output
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        wide_SiLU=True,
        norm_layer=nn.LayerNorm,
        use_rope=False,
        grid_size=None,
        grid_depth=None,
        is_causal=False,
        use_sdpa=True,
        qkv_backend="electronic",
        optical_config=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not use_rope:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                is_causal=is_causal,
                use_sdpa=use_sdpa,
                proj_drop=drop,
                qkv_backend=qkv_backend,
                optical_config=optical_config)
        else:
            self.attn = RoPEAttention3D(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    grid_size=grid_size,
                    grid_depth=grid_depth,
                    use_sdpa=use_sdpa,
                    proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_SiLU=wide_SiLU,
                drop=drop)
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop)

    def forward(self, x, return_attention=False, mask=None, return_proj_output=False):
        y_result = self.attn(
            self.norm1(x),
            mask=mask,
            return_proj_output=return_proj_output,
        )
        if return_proj_output:
            y, attn, proj_output = y_result
        else:
            y, attn = y_result
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_proj_output:
            return x, proj_output
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
