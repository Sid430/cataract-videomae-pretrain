"""
V-JEPA 2 — Video Joint-Embedding Predictive Architecture (self-supervised).

Faithful re-implementation of the core V-JEPA 2 design (Assran et al., 2025,
arXiv:2506.09985), including the feature that distinguishes V-JEPA 2 from
V-JEPA 1: 3D Rotary Position Embeddings (3D-RoPE) applied inside attention,
instead of absolute sin-cos position embeddings.

Components:
  - Tubelet patch embedding (2 x 16 x 16).
  - 3D-RoPE: head_dim is split into 3 segments (temporal / height / width);
    a 1D rotary embedding is applied to each using that token's grid coordinate.
    Applied to q and k inside every attention block (encoder + predictor).
  - CONTEXT ENCODER E_theta : ViT that processes only the *visible* tokens.
  - TARGET ENCODER  E_bar   : EMA (momentum) copy of the encoder; processes the
                              *full* video (no grad) to produce targets.
  - PREDICTOR       P_phi   : narrow ViT taking context + learnable mask tokens
                              (positions supplied via RoPE) -> predicts targets.
  - Multi-block spatiotemporal masking (~90%).
  - Loss = L1 between predictor outputs and LayerNorm'd target embeddings at the
    masked positions. Latent space only: no pixel decoder, no negatives.

This is a faithful research implementation, not a bit-exact clone of Meta's
code (which adds a multi-mask collator, progressive resolution, ViT-g scale,
etc.). For publication-grade results, initialize from the official
facebookresearch/vjepa2 weights.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.videomae_v2 import Mlp, PatchEmbed3D


# ---------------------------------------------------------------------------
# 3D Rotary Position Embedding
# ---------------------------------------------------------------------------
def build_3d_rope_tables(grid_t, grid_h, grid_w, head_dim, base=10000.0):
    """Precompute cos/sin tables of shape (N, head_dim) for 3D-RoPE, where the
    head dimension is split into ~equal temporal/height/width segments and each
    gets a 1D rotary embedding driven by that axis's grid coordinate.
    Tokens are ordered (t, h, w) to match PatchEmbed3D's flatten order."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    N = grid_t * grid_h * grid_w
    idx = torch.arange(N)
    tt = (idx // (grid_h * grid_w)).float()
    hh = ((idx // grid_w) % grid_h).float()
    ww = (idx % grid_w).float()

    half = head_dim // 2
    kt = half // 3
    kh = half // 3
    kw = half - kt - kh  # remainder to width

    def axis_angles(coord, k):
        if k == 0:
            return torch.zeros(N, 0)
        inv_freq = 1.0 / (base ** (torch.arange(k).float() / k))   # (k,)
        return coord[:, None] * inv_freq[None, :]                  # (N, k)

    ang = torch.cat([axis_angles(tt, kt), axis_angles(hh, kh), axis_angles(ww, kw)],
                    dim=-1)                                        # (N, half)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    cos = torch.cat([cos, cos], dim=-1)                           # (N, head_dim)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: (B, heads, T, head_dim). cos/sin: (T, head_dim) [shared] or
    (B, T, head_dim) [per-sample, when tokens were gathered by position]."""
    if cos.dim() == 2:
        cos = cos[None, None]      # (1,1,T,hd)
        sin = sin[None, None]
    else:
        cos = cos[:, None]         # (B,1,T,hd)
        sin = sin[:, None]
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# RoPE attention + block
# ---------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # (B, heads, N, hd)
        q = apply_rope(q, cos, sin)                # RoPE on q and k only
        k = apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Multi-block spatiotemporal mask generator
# ---------------------------------------------------------------------------
class MultiBlockMask:
    """Samples contiguous 3D blocks to mask until ~mask_ratio of tokens are
    masked, then fixes the count so tensors are batchable. Returns per-sample
    visible / masked index tensors ordered over (t, h, w)."""

    def __init__(self, grid_t, grid_h, grid_w, mask_ratio=0.9,
                 block_t=(1, 2), block_scale=(0.15, 0.25)):
        self.gt, self.gh, self.gw = grid_t, grid_h, grid_w
        self.N = grid_t * grid_h * grid_w
        self.n_mask = int(round(self.N * mask_ratio))
        self.block_t = block_t
        self.block_scale = block_scale

    def _one(self, device):
        mask = torch.zeros(self.gt, self.gh, self.gw, dtype=torch.bool, device=device)
        guard = 0
        while mask.sum().item() < self.n_mask and guard < 200:
            guard += 1
            bt = int(torch.randint(self.block_t[0], self.block_t[1] + 1, (1,), device=device).item())
            s = float(torch.empty(1, device=device).uniform_(self.block_scale[0], self.block_scale[1]).item())
            bh = max(1, int(round(self.gh * s)))
            bw = max(1, int(round(self.gw * s)))
            t0 = int(torch.randint(0, max(1, self.gt - bt + 1), (1,), device=device).item())
            h0 = int(torch.randint(0, max(1, self.gh - bh + 1), (1,), device=device).item())
            w0 = int(torch.randint(0, max(1, self.gw - bw + 1), (1,), device=device).item())
            mask[t0:t0 + bt, h0:h0 + bh, w0:w0 + bw] = True

        flat = mask.flatten()
        masked = torch.nonzero(flat, as_tuple=False).squeeze(1)
        visible = torch.nonzero(~flat, as_tuple=False).squeeze(1)
        if masked.numel() > self.n_mask:
            perm = torch.randperm(masked.numel(), device=device)
            move = masked[perm[self.n_mask:]]
            masked = masked[perm[:self.n_mask]]
            visible = torch.cat([visible, move])
        elif masked.numel() < self.n_mask:
            need = self.n_mask - masked.numel()
            perm = torch.randperm(visible.numel(), device=device)
            masked = torch.cat([masked, visible[perm[:need]]])
            visible = visible[perm[need:]]
        return visible.sort().values, masked.sort().values

    def __call__(self, batch_size, device):
        vis, msk = [], []
        for _ in range(batch_size):
            v, m = self._one(device)
            vis.append(v); msk.append(m)
        return torch.stack(vis, 0), torch.stack(msk, 0)


# ---------------------------------------------------------------------------
# Encoder (shared class for context + target); uses 3D-RoPE
# ---------------------------------------------------------------------------
class ViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, tubelet_size, num_frames, in_chans,
                 embed_dim, depth, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, tubelet_size,
                                        num_frames, in_chans, embed_dim)
        gt, gh, gw = self.patch_embed.grid_t, self.patch_embed.grid_h, self.patch_embed.grid_w
        head_dim = embed_dim // num_heads
        cos, sin = build_3d_rope_tables(gt, gh, gw, head_dim)
        self.register_buffer("rope_cos", cos)      # (N, head_dim)
        self.register_buffer("rope_sin", sin)
        self.blocks = nn.ModuleList(
            [RoPEBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _gather(seq, ids):
        return torch.gather(seq, 1, ids.unsqueeze(-1).expand(-1, -1, seq.size(-1)))

    def forward(self, x, keep_ids=None):
        tokens = self.patch_embed(x)                       # (B,N,D) — NO abs pos embed
        if keep_ids is not None:
            tokens = self._gather(tokens, keep_ids)        # (B,N_vis,D)
            cos = self.rope_cos[keep_ids]                  # (B,N_vis,hd)
            sin = self.rope_sin[keep_ids]
        else:
            cos, sin = self.rope_cos, self.rope_sin        # (N,hd)
        for blk in self.blocks:
            tokens = blk(tokens, cos, sin)
        return self.norm(tokens)


# ---------------------------------------------------------------------------
# V-JEPA 2 model
# ---------------------------------------------------------------------------
class VJEPA2(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16, tubelet_size=2, num_frames=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        predictor_embed_dim=384, predictor_depth=12, predictor_num_heads=12,
        mlp_ratio=4.0, mask_ratio=0.9,
    ):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, tubelet_size, num_frames,
                                  in_chans, embed_dim, depth, num_heads, mlp_ratio)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        gt, gh, gw = self.encoder.patch_embed.grid_t, self.encoder.patch_embed.grid_h, self.encoder.patch_embed.grid_w
        N = self.encoder.patch_embed.num_patches
        self.num_patches = N
        self.grid_t, self.grid_h, self.grid_w = gt, gh, gw
        self.embed_dim = embed_dim

        # predictor (RoPE-based, its own head_dim -> its own rope table)
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        pred_head_dim = predictor_embed_dim // predictor_num_heads
        pcos, psin = build_3d_rope_tables(gt, gh, gw, pred_head_dim)
        self.register_buffer("pred_rope_cos", pcos)
        self.register_buffer("pred_rope_sin", psin)
        self.predictor_blocks = nn.ModuleList(
            [RoPEBlock(predictor_embed_dim, predictor_num_heads, mlp_ratio)
             for _ in range(predictor_depth)])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self.mask_gen = MultiBlockMask(gt, gh, gw, mask_ratio)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self._sync_target()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.no_grad()
    def _sync_target(self):
        for pe, te in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            te.data.copy_(pe.data)
        for be, bt in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bt.data.copy_(be.data)

    @torch.no_grad()
    def update_target(self, momentum: float):
        for pe, te in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            te.data.mul_(momentum).add_(pe.data, alpha=1.0 - momentum)

    @staticmethod
    def _gather(seq, ids):
        return torch.gather(seq, 1, ids.unsqueeze(-1).expand(-1, -1, seq.size(-1)))

    def forward(self, x):
        B = x.size(0)
        vis_ids, mask_ids = self.mask_gen(B, x.device)

        # targets from EMA target encoder (full video, no grad)
        with torch.no_grad():
            full = self.target_encoder(x)                     # (B,N,D)
            targets = self._gather(full, mask_ids)            # (B,N_mask,D)
            targets = F.layer_norm(targets, (targets.size(-1),))

        # context representation from online encoder (visible tokens only)
        ctx = self.encoder(x, keep_ids=vis_ids)               # (B,N_vis,D)

        # predictor: context + mask tokens, positions supplied via RoPE
        ctx = self.predictor_embed(ctx)                       # (B,N_vis,Dp)
        n_mask = mask_ids.size(1)
        mtok = self.mask_token.expand(B, n_mask, -1)          # (B,N_mask,Dp)
        z = torch.cat([ctx, mtok], dim=1)                     # (B, N_vis+N_mask, Dp)
        pos_ids = torch.cat([vis_ids, mask_ids], dim=1)       # positions of z's tokens
        pcos = self.pred_rope_cos[pos_ids]                    # (B,L,hd_p)
        psin = self.pred_rope_sin[pos_ids]
        for blk in self.predictor_blocks:
            z = blk(z, pcos, psin)
        z = self.predictor_norm(z)
        pred = self.predictor_proj(z[:, -n_mask:])            # (B,N_mask,D)

        loss = F.l1_loss(pred, targets)
        return loss, pred


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def vjepa2_base(**kw):
    """ViT-B/16 context encoder + narrow predictor (width 384, depth 12)."""
    cfg = dict(embed_dim=768, depth=12, num_heads=12,
               predictor_embed_dim=384, predictor_depth=12, predictor_num_heads=12)
    cfg.update(kw)
    return VJEPA2(**cfg)


def vjepa2_tiny(**kw):
    """Tiny preset for fast CPU/dummy sanity runs."""
    cfg = dict(embed_dim=192, depth=4, num_heads=3,
               predictor_embed_dim=96, predictor_depth=2, predictor_num_heads=3)
    cfg.update(kw)
    return VJEPA2(**cfg)