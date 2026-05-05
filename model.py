"""Scaled-down LeWM for 64x64 Snake — FIRST-RUN ARCHITECTURE.

This is the architecture from the user's first (best-so-far) run, restored
verbatim. The patch-token visual experiments were rolled back.

- ViT encoder: patch 8 -> 8x8 = 64 patches, dim 128, depth 6, returns [CLS] only.
- AR predictor: 4 layers, AdaLN-zero action conditioning, dim 128.
- SIGReg per-timestep on the [CLS] latent (T, B, D).
- Decoder: ConvTranspose pyramid, post-hoc renderer (stop-grad on input latent).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ----- SIGReg (Epps-Pulley + Cramer-Wold) ---------------------------------------
class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: (S, B, D) — S = sample axis (e.g. T), B = batch, D = embedding dim
        D = proj.size(-1)
        A = torch.randn(D, self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)
        x_t = (proj @ A).unsqueeze(-1) * self.t          # (S, B, M, K)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ----- Transformer pieces ------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.dropout_p = dropout

    def forward(self, x, causal=True):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        drop = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Standard pre-norm transformer block (no causal mask — for encoder)."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(x, causal=False)
        x = x + self.mlp(x)
        return x


class ConditionalBlock(nn.Module):
    """AdaLN-zero block — for predictor (causal across time)."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, c):
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=-1)
        x = x + g_a * self.attn(modulate(self.norm1(x), s_a, sc_a), causal=True)
        x = x + g_m * self.mlp(modulate(self.norm2(x), s_m, sc_m))
        return x


# ----- Encoder: tiny ViT, returns [CLS] only -----------------------------------
class TinyViT(nn.Module):
    def __init__(self, img_size=64, patch=8, dim=128, depth=6, heads=4, dim_head=32, mlp_dim=512, dropout=0.0):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                     # (B, dim) — [CLS] only


# ----- Action embedder ----------------------------------------------------------
class ActionEmbedder(nn.Module):
    def __init__(self, n_actions=4, emb_dim=128):
        super().__init__()
        self.table = nn.Embedding(n_actions, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim), nn.SiLU(), nn.Linear(4 * emb_dim, emb_dim))

    def forward(self, a):
        return self.mlp(self.table(a))


# ----- Predictor: causal AdaLN transformer, single token per timestep ----------
class ARPredictor(nn.Module):
    def __init__(self, max_frames=8, dim=128, depth=4, heads=4, dim_head=32, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_frames, dim) * 0.02)
        self.blocks = nn.ModuleList([
            ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x, c)
        return self.norm(x)


# ----- BN-MLP projector (paper detail: BN here, not LN) -------------------------
class ProjMLP(nn.Module):
    def __init__(self, dim=128, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


# ----- Decoder: ConvT pyramid [CLS] -> 64x64 raw output (channels configurable) -
class TinyDecoder(nn.Module):
    """Returns raw logits of shape (B, out_channels, 64, 64). The interpretation
    of the channels depends on the loss_kind used during training:
      mse / focal-mse  -> 3 channels: raw RGB (apply sigmoid to render)
      gauss-nll        -> 6 channels: 3 mean + 3 log_var
      categorical      -> 4 channels: per-pixel logits over 4 cell types
      disc-logistic    -> 6 channels: 3 mean + 3 log_scale (per-channel)
    """
    def __init__(self, dim=128, ch=128, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(dim, ch * 4 * 4)
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1),
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
            nn.GroupNorm(8, ch // 2), nn.SiLU(),
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1),
            nn.GroupNorm(8, ch // 4), nn.SiLU(),
            nn.ConvTranspose2d(ch // 4, out_channels, 4, 2, 1),
        )

    def forward(self, z):
        z = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.net(z)


# ----- Oracle encoder for the decoder-isolation harness ------------------------
class OracleEncoder(nn.Module):
    """MLP encoder: ground-truth state features (variable-d) -> 128-d latent."""
    def __init__(self, in_dim: int = 99, out_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, state):
        return self.net(state)


class OracleEncoderCNN(nn.Module):
    """CNN encoder: spatial state mask (4, 64, 64) -> 128-d latent.
    Translation-equivariant — preserves cell precision exactly through the
    convolutional pipeline."""
    def __init__(self, in_channels: int = 4, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.GELU(),    # 64 -> 32
            nn.Conv2d(32, 64, 4, 2, 1),          nn.GELU(),    # 32 -> 16
            nn.Conv2d(64, 128, 4, 2, 1),         nn.GELU(),    # 16 -> 8
            nn.Conv2d(128, 128, 4, 2, 1),        nn.GELU(),    # 8 -> 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
        )

    def forward(self, state):
        return self.net(state)


def kmeans_palette_unique(pixels, K=8, n_iter=20, quant=255, seed=0):
    """K-means on UNIQUE pixel colors so each color contributes equally
    to clustering, regardless of class imbalance. Quantises RGB to `quant`
    levels per channel before unique-extraction."""
    if pixels.dim() > 2:
        pixels = pixels.reshape(-1, pixels.size(-1))
    q = (pixels * quant).round().long()
    keys = q[:, 0] * (quant + 1) ** 2 + q[:, 1] * (quant + 1) + q[:, 2]
    _, idx = torch.unique(keys, return_inverse=False, sorted=False, return_counts=False), None
    # extract unique rows (small set for discrete games)
    uniq_keys, first_idx = torch.unique(keys, return_inverse=False, return_counts=False, sorted=True), None
    # Map each unique key back to one representative pixel
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_pix = pixels[sort_idx]
    diff = torch.cat([torch.tensor([1], device=keys.device), (sorted_keys[1:] != sorted_keys[:-1]).long()])
    starts = diff.nonzero(as_tuple=True)[0]
    uniq_pix = sorted_pix[starts]                                              # (U, 3)
    if uniq_pix.size(0) <= K:
        return uniq_pix
    # K-means on the U unique colors (typically small)
    g = torch.Generator(device=pixels.device).manual_seed(seed)
    centers = uniq_pix[torch.randperm(uniq_pix.size(0), generator=g, device=pixels.device)[:K]].clone()
    for _ in range(n_iter):
        dist = (uniq_pix.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
        labels = dist.argmin(dim=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = uniq_pix[mask].mean(dim=0)
    return centers


def kmeans_palette(pixels, K=8, n_samples=100_000, n_iter=20, seed=0):
    """pixels: (N, 3) tensor in [0,1] (or higher-rank, will flatten). Returns (K, 3) palette."""
    if pixels.dim() > 2:
        pixels = pixels.reshape(-1, pixels.size(-1))
    n = pixels.size(0)
    g = torch.Generator(device=pixels.device).manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=pixels.device)[:n_samples]
    pix = pixels[idx]
    centers = pix[torch.randperm(pix.size(0), generator=g, device=pixels.device)[:K]].clone()
    for _ in range(n_iter):
        dist = (pix.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
        labels = dist.argmin(dim=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = pix[mask].mean(dim=0)
    return centers


def _mol_loss(raw, target, K):
    """Mixture-of-K-Logistics NLL (PixelCNN++-style) per pixel.
    raw: (B, K + 2*3*K, H, W). First K = mixture logits.
    Next 3*K = per-channel means; last 3*K = per-channel log-scales.
    """
    B, _, H, W = raw.shape
    mw_logits = raw[:, :K]                                                       # (B, K, H, W)
    means = raw[:, K:K + 3 * K].reshape(B, K, 3, H, W).sigmoid()                 # (B, K, 3, H, W)
    log_scales = raw[:, K + 3 * K:K + 6 * K].reshape(B, K, 3, H, W).clamp(-7.0, 2.0)
    scales = log_scales.exp()
    x = target.unsqueeze(1)                                                       # (B, 1, 3, H, W)
    eps = 0.5 / 255.0
    plus = torch.sigmoid((x + eps - means) / scales)
    minus = torch.sigmoid((x - eps - means) / scales)
    prob = (plus - minus).clamp(min=1e-12)                                        # (B, K, 3, H, W)
    log_prob = prob.log().sum(dim=2)                                              # (B, K, H, W)
    log_w = F.log_softmax(mw_logits, dim=1)
    log_mix = torch.logsumexp(log_w + log_prob, dim=1)                            # (B, H, W)
    return -log_mix.mean()


def _chan256_loss(raw, target):
    """Per-channel 256-way softmax CE (PixelRNN-style)."""
    B, _, H, W = raw.shape
    logits = raw.reshape(B, 3, 256, H, W)
    labels = (target * 255).long().clamp(0, 255)
    return sum(F.cross_entropy(logits[:, c], labels[:, c]) for c in range(3)) / 3


def _cat_kmeans_loss(raw, target, palette):
    """Per-pixel CE over a learned (K-means) palette.
    raw: (B, K, H, W). palette: (K, 3).
    """
    p = palette.to(target.device, dtype=target.dtype).view(1, palette.size(0), 3, 1, 1)
    dist = (target.unsqueeze(1) - p).pow(2).sum(dim=2)                            # (B, K, H, W)
    labels = dist.argmin(dim=1)                                                    # (B, H, W)
    return F.cross_entropy(raw, labels)


def _cat_kmeans_focal_loss(raw, target, palette, gamma=2.0):
    """Per-pixel focal CE: (1 - p_correct)^gamma * CE."""
    p = palette.to(target.device, dtype=target.dtype).view(1, palette.size(0), 3, 1, 1)
    labels = (target.unsqueeze(1) - p).pow(2).sum(dim=2).argmin(dim=1)              # (B, H, W)
    log_probs = F.log_softmax(raw, dim=1)
    log_p_correct = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)             # (B, H, W)
    p_correct = log_p_correct.exp()
    focal = (1.0 - p_correct).pow(gamma)
    return -(focal * log_p_correct).mean()


def _cat_kmeans_weighted_loss(raw, target, palette, class_weights):
    """Per-pixel CE with per-class inverse-frequency weights."""
    p = palette.to(target.device, dtype=target.dtype).view(1, palette.size(0), 3, 1, 1)
    labels = (target.unsqueeze(1) - p).pow(2).sum(dim=2).argmin(dim=1)
    w = class_weights.to(raw.device, dtype=raw.dtype)
    return F.cross_entropy(raw, labels, weight=w)


def oracle_decoder_loss(raw, target, loss_kind, K=None, palette=None, class_weights=None):
    """Pixel-level loss for the decoder-isolation experiment.
    raw: (B, C, H, W) — raw decoder output, channels depend on loss_kind.
    target: (B, 3, H, W) ground-truth pixels in [0,1].
    K: int, mixture / palette size (for mol / cat-kmeans / chan-256).
    palette: (K, 3) tensor, used by cat-kmeans only.
    """
    if loss_kind == "mse":
        return F.mse_loss(raw.sigmoid(), target)
    if loss_kind == "focal":
        recon = raw.sigmoid()
        err = (recon - target).pow(2)
        weight = (1.0 - (-err).exp()).pow(2)
        return (err * weight).mean()
    if loss_kind == "weight":
        recon = raw.sigmoid()
        bg = SNAKE_COLORS[0].to(target.device, dtype=target.dtype).view(1, 3, 1, 1)
        fg_mask = (target - bg).pow(2).sum(dim=1, keepdim=True).gt(0.005).to(target.dtype)
        w = 1.0 + 50.0 * fg_mask
        return ((recon - target).pow(2) * w).mean()
    if loss_kind == "gauss":
        mean = raw[:, :3].sigmoid()
        log_var = raw[:, 3:6].clamp(-7.0, 2.0)
        var = log_var.exp()
        return (0.5 * ((target - mean).pow(2) / var + log_var)).mean()
    if loss_kind == "cat":
        labels = quantize_to_class(target)
        return F.cross_entropy(raw, labels)
    if loss_kind == "cat-kmeans":
        return _cat_kmeans_loss(raw, target, palette)
    if loss_kind == "cat-kmeans-focal":
        return _cat_kmeans_focal_loss(raw, target, palette, gamma=2.0)
    if loss_kind == "cat-kmeans-weighted":
        assert class_weights is not None, "class_weights required for cat-kmeans-weighted"
        return _cat_kmeans_weighted_loss(raw, target, palette, class_weights=class_weights)
    if loss_kind == "cat-kmeans-unique":
        return _cat_kmeans_loss(raw, target, palette)  # palette is built differently upstream
    if loss_kind == "mol":
        return _mol_loss(raw, target, K)
    if loss_kind == "chan-256":
        return _chan256_loss(raw, target)
    if loss_kind == "disc-logistic":
        mean = raw[:, :3].sigmoid()
        log_scale = raw[:, 3:6].clamp(-7.0, 2.0)
        scale = log_scale.exp()
        eps = 0.5 / 255.0
        plus = torch.sigmoid((target + eps - mean) / scale)
        minus = torch.sigmoid((target - eps - mean) / scale)
        prob = (plus - minus).clamp(min=1e-12)
        return -prob.log().mean()
    raise ValueError(f"unknown loss_kind: {loss_kind}")


def oracle_decoder_out_channels(loss_kind, K=None):
    if loss_kind in ("mse", "focal", "weight"):
        return 3
    if loss_kind in ("gauss", "disc-logistic"):
        return 6
    if loss_kind == "cat":
        return 4
    if loss_kind in ("cat-kmeans", "cat-kmeans-unique", "cat-kmeans-focal", "cat-kmeans-weighted"):
        assert K is not None
        return K
    if loss_kind == "mol":
        assert K is not None
        return K + 6 * K   # K mixture weights + K means*3 channels + K log_scales*3 channels
    if loss_kind == "chan-256":
        return 3 * 256
    raise ValueError(loss_kind)


def render_oracle_output(raw, loss_kind, K=None, palette=None):
    """Convert raw decoder output to RGB in [0,1] for visualisation."""
    if loss_kind in ("mse", "focal", "weight"):
        return raw.sigmoid()
    if loss_kind in ("gauss", "disc-logistic"):
        return raw[:, :3].sigmoid()
    if loss_kind == "cat":
        labels = raw.argmax(dim=1)
        colors = SNAKE_COLORS.to(raw.device, dtype=raw.dtype)
        return colors[labels].permute(0, 3, 1, 2)
    if loss_kind in ("cat-kmeans", "cat-kmeans-unique", "cat-kmeans-focal", "cat-kmeans-weighted"):
        labels = raw.argmax(dim=1)
        colors = palette.to(raw.device, dtype=raw.dtype)
        return colors[labels].permute(0, 3, 1, 2)
    if loss_kind == "mol":
        B = raw.size(0); H, W = raw.shape[-2:]
        mw = F.softmax(raw[:, :K], dim=1)
        means = raw[:, K:K + 3 * K].reshape(B, K, 3, H, W).sigmoid()
        return (means * mw.unsqueeze(2)).sum(dim=1).clamp(0, 1)
    if loss_kind == "chan-256":
        B = raw.size(0); H, W = raw.shape[-2:]
        logits = raw.reshape(B, 3, 256, H, W)
        return logits.argmax(dim=2).float() / 255
    raise ValueError(loss_kind)


# Snake's 4 known cell colors, normalised to [0,1]; used by categorical loss/render
SNAKE_COLORS = torch.tensor([
    [15.0 / 255, 15.0 / 255, 25.0 / 255],     # background
    [90.0 / 255, 200.0 / 255, 110.0 / 255],   # body
    [240.0 / 255, 240.0 / 255, 80.0 / 255],   # head
    [230.0 / 255, 80.0 / 255, 80.0 / 255],    # food
])


def quantize_to_class(pixels):
    """pixels: (..., 3, H, W) in [0,1]. Returns (..., H, W) long labels in [0..3]."""
    colors = SNAKE_COLORS.to(pixels.device, dtype=pixels.dtype)
    # broadcast over last 2 dims
    p = pixels.unsqueeze(-4)              # (..., 1, 3, H, W)
    c = colors.view(*([1] * (p.dim() - 4)), 4, 3, 1, 1)
    dist = (p - c).pow(2).sum(dim=-3)     # (..., 4, H, W)
    return dist.argmin(dim=-3)            # (..., H, W)


def render_decoder_output(raw, loss_kind):
    """Convert raw decoder output to RGB in [0,1] for visualisation."""
    if loss_kind in ("mse", "focal-mse"):
        return raw.sigmoid()
    if loss_kind == "gauss-nll":
        return raw[:, :3].sigmoid()
    if loss_kind == "disc-logistic":
        return raw[:, :3].sigmoid()
    if loss_kind == "categorical":
        labels = raw.argmax(dim=1)        # (B, H, W)
        colors = SNAKE_COLORS.to(raw.device, dtype=raw.dtype)
        return colors[labels].permute(0, 3, 1, 2)
    raise ValueError(loss_kind)


# ----- Sharp decoder: [CLS] -> 16x16 latent grid -> pixel-shuffle to 64x64 ------
class SharpDecoder(nn.Module):
    """Each 4x4 output cell is a distinct learned function of [CLS]; pixel-shuffle
    keeps cell boundaries crisp (no upsample-blur between adjacent cells).

    Linear: [CLS] -> 16*16 spatial positions x hidden channels.
    Refinement conv at 16x16 lets neighbouring cell features interact lightly.
    PixelShuffle 4x maps each (1,1) position to a (4,4) RGB block.
    """
    def __init__(self, dim=128, hidden=128, grid=16, scale=4):
        super().__init__()
        self.grid = grid
        self.scale = scale
        self.fc = nn.Linear(dim, grid * grid * hidden)
        self.refine = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GroupNorm(8, hidden), nn.SiLU(),
        )
        self.to_pixels = nn.Conv2d(hidden, 3 * scale * scale, 1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, -1, self.grid, self.grid)
        x = x + self.refine(x)
        x = self.to_pixels(x)
        x = self.shuffle(x)
        return x.sigmoid()


# ----- Per-pixel coordinate-MLP decoder: each pixel computed independently ------
class PerPixelDecoder(nn.Module):
    """Each output pixel is computed by a small MLP taking [CLS] + a learned
    per-pixel position embedding. Output per pixel: (mean_rgb, log_var_rgb) for
    Gaussian NLL. No upsampling layers, no convolution, so adjacent pixels do
    not share the smearing that ConvTranspose introduces."""
    def __init__(self, dim=128, img_size=64, hidden=256):
        super().__init__()
        self.img_size = img_size
        n_px = img_size * img_size
        self.pos = nn.Parameter(torch.randn(1, n_px, dim) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )

    def forward_with_var(self, z):
        B = z.size(0)
        n_px = self.img_size * self.img_size
        z_e = z.unsqueeze(1).expand(-1, n_px, -1)
        pos = self.pos.expand(B, -1, -1)
        x = torch.cat([z_e, pos], dim=-1)             # (B, P, 2*dim)
        out = self.net(x)                              # (B, P, 6)
        mean = out[..., :3].sigmoid().transpose(1, 2).reshape(B, 3, self.img_size, self.img_size)
        log_var = out[..., 3:].clamp(-7.0, 2.0).transpose(1, 2).reshape(B, 3, self.img_size, self.img_size)
        return mean, log_var

    def forward(self, z):
        mean, _ = self.forward_with_var(z)
        return mean


# ----- Cross-attention decoder: [CLS] -> 64x64 RGB via 256 spatial queries -----
class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, q, kv):
        a, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + a
        q = q + self.mlp(self.norm_m(q))
        return q


class CrossAttnDecoder(nn.Module):
    """[CLS] (B, dim) -> img RGB via spatial queries + cross-attention + Gaussian NLL.

    Output: per-pixel (mean, log_var) in 6 channels via pixel-shuffle. Same
    rare-pixel-aware loss as TinyDecoder (Gaussian NLL).
    """
    def __init__(self, dim=512, n_queries=256, n_blocks=3, heads=8, mlp_dim=2048,
                 grid=16, scale=4):
        super().__init__()
        assert n_queries == grid * grid, "queries must lay out as a square"
        self.grid = grid
        self.scale = scale
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.in_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([CrossAttnBlock(dim, heads, mlp_dim) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(dim)
        # 6 channels per pixel-shuffle slot: 3 RGB means + 3 RGB log-variances.
        self.head = nn.Linear(dim, 6 * scale * scale)

    def forward_with_var(self, z):
        B = z.size(0)
        memory = self.in_proj(z).unsqueeze(1)
        q = self.queries.expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, memory)
        q = self.out_norm(q)
        q = self.head(q)
        q = q.transpose(1, 2).reshape(B, 6 * self.scale * self.scale, self.grid, self.grid)
        out = F.pixel_shuffle(q, self.scale)             # (B, 6, H, W)
        mean = out[:, :3].sigmoid()
        log_var = out[:, 3:].clamp(-7.0, 2.0)
        return mean, log_var

    def forward(self, z):
        mean, _ = self.forward_with_var(z)
        return mean


# ----- Full LeWM-Snake bundle ---------------------------------------------------
class LeWMSnake(nn.Module):
    def __init__(self, dim=128, history=4, n_actions=4, img_size=64, patch=8,
                 decoder_kind="convtranspose", out_channels=3, latent_nll=False):
        super().__init__()
        self.dim = dim
        self.history = history
        self.img_size = img_size
        self.decoder_kind = decoder_kind
        # Scale heads with dim: 4 heads at dim=128, 8 heads at dim>=256.
        heads = 8 if dim >= 256 else 4
        dim_head = dim // heads
        mlp_dim = 4 * dim
        self.encoder = TinyViT(img_size=img_size, patch=patch, dim=dim,
                               depth=6, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        self.projector = ProjMLP(dim=dim, hidden=mlp_dim)
        self.action_embed = ActionEmbedder(n_actions=n_actions, emb_dim=dim)
        self.predictor = ARPredictor(max_frames=history + 4, dim=dim,
                                     depth=4, heads=heads, dim_head=dim_head,
                                     mlp_dim=mlp_dim, dropout=0.1)
        self.pred_proj = ProjMLP(dim=dim, hidden=mlp_dim)
        self.latent_nll = latent_nll
        if latent_nll:
            # Per-dim log-variance head on the predictor — turns the JEPA pred
            # loss into a Gaussian NLL whose gradient is intrinsically rarity-
            # aware. Init to zero so var=1 initially (equivalent to MSE).
            self.pred_logvar_head = nn.Linear(dim, dim)
            nn.init.constant_(self.pred_logvar_head.weight, 0)
            nn.init.constant_(self.pred_logvar_head.bias, 0)
        if decoder_kind == "convtranspose":
            self.decoder = TinyDecoder(dim=dim, ch=128, out_channels=out_channels)
        elif decoder_kind == "sharp":
            self.decoder = SharpDecoder(dim=dim, hidden=128, grid=16,
                                         scale=img_size // 16)
        elif decoder_kind == "perpixel":
            self.decoder = PerPixelDecoder(dim=dim, img_size=img_size, hidden=2 * dim)
        elif decoder_kind == "crossattn":
            # 32x32 = 1024 queries, each renders a 2x2 RGB block (scale=2) -> 64x64.
            grid = 32
            scale = img_size // grid
            self.decoder = CrossAttnDecoder(dim=dim, n_queries=grid * grid,
                                            n_blocks=3, heads=heads, mlp_dim=mlp_dim,
                                            grid=grid, scale=scale)
        else:
            raise ValueError(f"unknown decoder_kind: {decoder_kind}")

    def encode(self, pixels):
        # pixels: (B, T, 3, H, W) in [0,1]   ->   (B, T, D)
        B, T = pixels.shape[:2]
        x = rearrange(pixels, "b t c h w -> (b t) c h w")
        z = self.encoder(x)                         # (B*T, D)
        z = self.projector(z)
        return rearrange(z, "(b t) d -> b t d", b=B)

    def predict(self, emb, act_emb):
        # Returns mean only (used at inference / play.py).
        if self.latent_nll:
            mean, _ = self.predict_with_var(emb, act_emb)
            return mean
        B, T, D = emb.shape
        out = self.predictor(emb, act_emb)
        out = self.pred_proj(rearrange(out, "b t d -> (b t) d"))
        return rearrange(out, "(b t) d -> b t d", b=B)

    def predict_with_var(self, emb, act_emb):
        assert self.latent_nll, "predict_with_var requires latent_nll=True"
        B, T, D = emb.shape
        out = self.predictor(emb, act_emb)
        out_flat = rearrange(out, "b t d -> (b t) d")
        mean = self.pred_proj(out_flat)
        log_var = self.pred_logvar_head(out_flat).clamp(-7.0, 2.0)
        mean = rearrange(mean, "(b t) d -> b t d", b=B)
        log_var = rearrange(log_var, "(b t) d -> b t d", b=B)
        return mean, log_var


def lewm_loss(model, pixels, actions, sigreg: SIGReg,
              lam: float = 0.1, loss_kind: str = "mse"):
    """JEPA pred MSE + SIGReg + post-hoc decoder loss (loss_kind selects form).

    Decoder reads encoder's latent (stop-grad) and reconstructs the current
    frame. Encoder is shaped purely by JEPA pred_loss + sigreg.
    """
    emb = model.encode(pixels)
    act_emb = model.action_embed(actions)

    ctx_emb = emb[:, :-1]
    target = emb[:, 1:]
    if getattr(model, "latent_nll", False):
        pred_mean, pred_log_var = model.predict_with_var(ctx_emb, act_emb)
        pred_var = pred_log_var.exp()
        pred_loss = (0.5 * ((target - pred_mean).pow(2) / pred_var + pred_log_var)).mean()
    else:
        pred_emb = model.predict(ctx_emb, act_emb)
        pred_loss = (pred_emb - target).pow(2).mean()

    sigreg_loss = sigreg(emb.transpose(0, 1))

    B, T = pixels.shape[:2]
    H = W = model.img_size
    z_flat = emb.detach().reshape(B * T, -1)
    pix_target = pixels.reshape(B * T, 3, H, W)
    raw = model.decoder(z_flat)                                 # (B*T, C, H, W)

    if loss_kind == "mse":
        decoder_loss = F.mse_loss(raw.sigmoid(), pix_target)
    elif loss_kind == "focal-mse":
        recon = raw.sigmoid()
        err = (recon - pix_target).pow(2)
        weight = (1.0 - (-err).exp()).pow(2)
        decoder_loss = (err * weight).mean()
    elif loss_kind == "gauss-nll":
        mean = raw[:, :3].sigmoid()
        log_var = raw[:, 3:6].clamp(-7.0, 2.0)
        var = log_var.exp()
        decoder_loss = (0.5 * ((pix_target - mean).pow(2) / var + log_var)).mean()
    elif loss_kind == "categorical":
        labels = quantize_to_class(pix_target)                   # (B*T, H, W) long
        decoder_loss = F.cross_entropy(raw, labels)
    elif loss_kind == "disc-logistic":
        mean = raw[:, :3].sigmoid()
        log_scale = raw[:, 3:6].clamp(-7.0, 2.0)
        scale = log_scale.exp()
        eps = 0.5 / 255.0
        plus = torch.sigmoid((pix_target + eps - mean) / scale)
        minus = torch.sigmoid((pix_target - eps - mean) / scale)
        prob = (plus - minus).clamp(min=1e-12)
        decoder_loss = -prob.log().mean()
    else:
        raise ValueError(f"unknown loss_kind: {loss_kind}")

    total = pred_loss + lam * sigreg_loss + decoder_loss
    return total, {
        "pred": pred_loss.detach(),
        "sigreg": sigreg_loss.detach(),
        "decoder": decoder_loss.detach(),
    }
