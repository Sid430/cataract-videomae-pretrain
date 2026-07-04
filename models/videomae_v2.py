import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# get_sinusoid_encoding_table, so it is safe to warm-start an encoder)
# ---------------------------------------------------------------------------
def sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed sine-cosine positional embedding, shape (1, n_position, d_hid)."""
    pos = torch.arange(n_position).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    pe = torch.zeros(n_position, d_hid)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, N, D)


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath)
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)          # per-sample
    rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rand.floor_()                                        # binarize
    return x.div(keep_prob) * rand


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, dim, hidden, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """VideoMAE-style attention: a single qkv projection WITHOUT bias, plus
    separate learnable q_bias / v_bias and a fixed (zero, non-learnable) k_bias.
    Uses the explicit Attention-Is-All-You-Need formulation (not SDPA), as in
    the reference implementation."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            # k gets a fixed zero bias (detached), q and v get learnable bias.
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # (B, heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale,
                              attn_drop, drop, attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer, drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# 3D (tubelet) patch embedding
# ---------------------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 num_frames=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.grid_t = num_frames // tubelet_size
        self.num_spatial = self.grid_h * self.grid_w
        self.num_patches = self.grid_t * self.num_spatial
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: (B, C, T, H, W) -> (B, N, D) with N ordered as (t, h, w)
        x = self.proj(x)                  # (B, D, t, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# Mask generators
#   Both return a unified, batch-stackable 3-tuple:
#     vis_ids : (B, N_vis)  encoder-visible token indices
#     dec_ids : (B, N_dec)  decoder reconstruction positions
#     loss_w  : (B, N_dec)  1.0 where dec_id was encoder-masked, else 0.0
# ---------------------------------------------------------------------------
class RunningCellMaskGenerator:
    """Tube encoder mask + RUNNING-CELL decoder mask (faithful VideoMAE v2).

    A 2x2 cell pattern is tiled over the spatial grid; the kept/hidden split
    shifts by one each frame (it "runs"). Replicates the reference Cell logic
    (increment pointer, then read; one phase-map per starting offset)."""

    def __init__(self, grid_t, grid_h, grid_w,
                 enc_mask_ratio=0.9, dec_mask_ratio=0.5):
        assert grid_h % 2 == 0 and grid_w % 2 == 0, \
            "running-cell needs an even spatial grid (2x2 cells)"
        self.grid_t, self.grid_h, self.grid_w = grid_t, grid_h, grid_w
        self.num_spatial = grid_h * grid_w
        self.keep_spatial = max(1, round(self.num_spatial * (1 - enc_mask_ratio)))

        n_hidden = int(4 * dec_mask_ratio)              # hidden cells per 2x2 unit
        assert 0 < n_hidden < 4, \
            "dec_mask_ratio must give 0 < int(4*r) < 4 (e.g. .25 / .5 / .75)"
        queue = np.hstack([np.ones(n_hidden), np.zeros(4 - n_hidden)])  # e.g. [1,1,0,0]
        size = queue.size
        maps = []
        for start in range(size):                       # one map per starting phase
            ptr, frames = start, []
            for _ in range(grid_t):
                ptr += 1                                # "run" one step per frame
                unit = queue[(np.arange(size) + ptr) % size].reshape(2, 2)
                tiled = np.tile(unit, (grid_h // 2, grid_w // 2)).flatten()
                frames.append(tiled)                    # spatial idx s = h*W + w
            maps.append(np.stack(frames, 0))            # (grid_t, num_spatial)
        self.decode_maps = torch.from_numpy(np.stack(maps, 0))  # (size, grid_t, S)
        self.n_phases = size

    def __call__(self, batch_size, device):
        S, T = self.num_spatial, self.grid_t
        t_off = (torch.arange(T, device=device) * S).unsqueeze(1)   # (T, 1)
        vis_ids, dec_ids, loss_w = [], [], []
        for _ in range(batch_size):
            perm = torch.randperm(S, device=device)
            keep_s = perm[:self.keep_spatial]
            vis_ids.append((t_off + keep_s.unsqueeze(0)).reshape(-1))

            visible_spatial = torch.zeros(S, dtype=torch.bool, device=device)
            visible_spatial[keep_s] = True

            dmap = self.decode_maps[torch.randint(self.n_phases, (1,)).item()].to(device)
            dec_t, w_t = [], []
            for t in range(T):
                dec_s = (dmap[t] == 0).nonzero(as_tuple=False).squeeze(1)
                dec_t.append(t * S + dec_s)
                w_t.append((~visible_spatial[dec_s]).float())   # 0 if encoder-visible
            dec_ids.append(torch.cat(dec_t))
            loss_w.append(torch.cat(w_t))
        return (torch.stack(vis_ids, 0),
                torch.stack(dec_ids, 0),
                torch.stack(loss_w, 0))


class RandomMaskGenerator:


    def __init__(self, grid_t, grid_h, grid_w,
                 enc_mask_ratio=0.9, dec_mask_ratio=0.5):
        self.grid_t = grid_t
        self.num_spatial = grid_h * grid_w
        self.num_patches = grid_t * self.num_spatial
        self.keep_spatial = max(1, round(self.num_spatial * (1 - enc_mask_ratio)))
        self.n_decode = max(1, round(self.num_patches * (1 - dec_mask_ratio)))

    def __call__(self, batch_size, device):
        S, T, N = self.num_spatial, self.grid_t, self.num_patches
        t_off = (torch.arange(T, device=device) * S).unsqueeze(1)
        vis_ids, dec_ids, loss_w = [], [], []
        for _ in range(batch_size):
            perm = torch.randperm(S, device=device)
            keep_s = perm[:self.keep_spatial]
            vis = (t_off + keep_s.unsqueeze(0)).reshape(-1)         # tube-visible
            visible = torch.zeros(N, dtype=torch.bool, device=device)
            visible[vis] = True
            dec = torch.randperm(N, device=device)[:self.n_decode]  # random subset of ALL
            vis_ids.append(vis)
            dec_ids.append(dec)
            loss_w.append((~visible[dec]).float())                 # 0 if encoder-visible
        return (torch.stack(vis_ids, 0),
                torch.stack(dec_ids, 0),
                torch.stack(loss_w, 0))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class VideoMAEv2(nn.Module):
    def __init__(
        self,
        img_size=224, patch_size=16, tubelet_size=2, num_frames=16, in_chans=3,
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, decoder_drop_path_rate=0.0,
        init_values=None,                          # LayerScale; None/0 disables it
        enc_mask_ratio=0.9, dec_mask_ratio=0.5,
        decoder_masking="running_cell",            # "running_cell" | "random"
        norm_layer=nn.LayerNorm, norm_pix_loss=True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, tubelet_size,
                                        num_frames, in_chans, embed_dim)
        N = self.patch_embed.num_patches
        self.num_patches = N
        self.in_chans = in_chans
        self.patch_dim = in_chans * tubelet_size * patch_size * patch_size
        self.norm_pix_loss = norm_pix_loss
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.grid_t = self.patch_embed.grid_t
        self.grid_h = self.patch_embed.grid_h
        self.grid_w = self.patch_embed.grid_w

        # encoder ---------------------------------------------------------
        self.register_buffer("pos_embed", sinusoid_encoding(N, embed_dim))
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, enc_dpr[i], init_values, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # decoder ---------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer("decoder_pos_embed",
                             sinusoid_encoding(N, decoder_embed_dim))
        dec_dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, dec_dpr[i], init_values, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, self.patch_dim)

        # mask generator --------------------------------------------------
        if decoder_masking == "running_cell":
            self.mask_gen = RunningCellMaskGenerator(
                self.grid_t, self.grid_h, self.grid_w,
                enc_mask_ratio, dec_mask_ratio)
        elif decoder_masking == "random":
            self.mask_gen = RandomMaskGenerator(
                self.grid_t, self.grid_h, self.grid_w,
                enc_mask_ratio, dec_mask_ratio)
        else:
            raise ValueError(f"unknown decoder_masking: {decoder_masking!r}")

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init_weights(m):
        # VideoMAE reference init: truncated normal (std 0.02) for Linear weights.
        # q_bias / v_bias / gamma_* are initialized in their modules and left alone.
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ----- reconstruction target ----------------------------------------
    def patchify(self, x):
        """(B,C,T,H,W) -> (B, N, patch_dim), N ordered as (t,h,w).

        Within each patch the layout is (ts, ps, ps, C) flattened, i.e. the
        channel axis is innermost / fastest-varying."""
        B, C, T, H, W = x.shape
        ts, ps = self.tubelet_size, self.patch_size
        t, h, w = T // ts, H // ps, W // ps
        x = x.reshape(B, C, t, ts, h, ps, w, ps)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)         # (B,t,h,w,ts,ps,ps,C)
        x = x.reshape(B, t * h * w, ts * ps * ps * C)
        return x

    @staticmethod
    def _gather(seq, ids):
        """seq: (B,N,D), ids: (B,K) -> (B,K,D)."""
        D = seq.size(-1)
        return torch.gather(seq, 1, ids.unsqueeze(-1).expand(-1, -1, D))

    def forward(self, x):
        device = x.device
        B = x.size(0)
        vis_ids, dec_ids, loss_w = self.mask_gen(B, device)

        # --- encoder on visible tokens ---
        tokens = self.patch_embed(x) + self.pos_embed         # (B,N,D)
        x_vis = self._gather(tokens, vis_ids)                 # (B,N_vis,D)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)

        # --- decoder: visible features + mask tokens at reconstruction pos ---
        x_vis = self.decoder_embed(x_vis)                     # (B,N_vis,Dd)
        dpos = self.decoder_pos_embed.expand(B, -1, -1)
        x_vis = x_vis + self._gather(dpos, vis_ids)
        n_dec = dec_ids.size(1)
        mask_tok = self.mask_token.expand(B, n_dec, -1)
        mask_tok = mask_tok + self._gather(dpos, dec_ids)
        dec_in = torch.cat([x_vis, mask_tok], dim=1)
        for blk in self.decoder_blocks:
            dec_in = blk(dec_in)
        dec_in = self.decoder_norm(dec_in)
        pred = self.decoder_head(dec_in[:, -n_dec:])          # (B,n_dec,patch_dim)

        # --- target + weighted loss on encoder-masked decode positions ---
        target = self._gather(self.patchify(x), dec_ids)      # (B,n_dec,patch_dim)
        if self.norm_pix_loss:
            # Per-channel normalization, matching the official VideoMAE v2 engine:
            # normalize over the within-patch pixels for each channel independently.
            C = self.in_chans
            B_, n_, D_ = target.shape
            t = target.reshape(B_, n_, D_ // C, C)            # (B,n_dec,ts*ps*ps,C)
            mean = t.mean(dim=-2, keepdim=True)
            var = t.var(dim=-2, unbiased=True, keepdim=True)
            t = (t - mean) / (var.sqrt() + 1e-6)
            target = t.reshape(B_, n_, D_)
        se = ((pred - target) ** 2).mean(dim=-1)              # (B,n_dec)
        loss = (se * loss_w).sum() / loss_w.sum().clamp(min=1.0)
        return loss, pred

    # ----- warm-start the encoder from official weights -----------------
    @torch.no_grad()
    def load_pretrained_encoder(self, state_dict, verbose=True):
        """Copy encoder tensors (patch_embed / blocks / norm) from an official
        VideoMAE / VideoMAE v2 checkpoint. Decoder, mask_token and the fixed
        sinusoidal pos-embeds are intentionally left as-is. Loads defensively:
        only keys that exist here with a matching shape are copied. The printed
        report should be spot-checked against your specific checkpoint, since
        key naming varies between pretrain and fine-tune checkpoints."""
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]
        own = self.state_dict()
        skip_roots = ("decoder_embed", "decoder_blocks", "decoder_norm",
                      "decoder_head", "mask_token", "decoder_pos_embed", "pos_embed")
        loaded, skipped = [], []
        for k, v in state_dict.items():
            key = k
            for pre in ("encoder.", "backbone.", "module."):
                if key.startswith(pre):
                    key = key[len(pre):]
            if key.split(".")[0] in skip_roots:
                skipped.append(k); continue
            if key in own and own[key].shape == v.shape:
                own[key].copy_(v); loaded.append(key)
            else:
                skipped.append(k)
        if verbose:
            print(f"[warm-start] copied {len(loaded)} encoder tensors, "
                  f"skipped {len(skipped)}")
        return loaded, skipped


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------
def videomae_v2_base(**kw):
    """ViT-B/16 preset — the standard VideoMAE v2 backbone (running-cell mask)."""
    cfg = dict(embed_dim=768, depth=12, num_heads=12,
               decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6)
    cfg.update(kw)
    return VideoMAEv2(**cfg)


def videomae_v2_giant(**kw):
    """ViT-g/14 preset — billion-parameter backbone; uses LayerScale by default."""
    cfg = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16,
               mlp_ratio=48 / 11, init_values=1e-5,
               decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8)
    cfg.update(kw)
    return VideoMAEv2(**cfg)


def videomae_v2_tiny(**kw):
    """Tiny preset — for fast dummy/CPU sanity runs only."""
    cfg = dict(embed_dim=192, depth=4, num_heads=3,
               decoder_embed_dim=96, decoder_depth=2, decoder_num_heads=3)
    cfg.update(kw)
    return VideoMAEv2(**cfg)