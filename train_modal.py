"""Decoder-isolation 20-way ablation: oracle encoder + (4 decoders × 5 losses).

Replaces the JEPA encoder + predictor with a deterministic 'oracle encoder'
that takes hand-crafted ground-truth state features (head, food, body, dir)
and produces a 128-d vector. Trained jointly with the decoder under one of
five pixel losses. There is NO JEPA, NO predictor, NO SIGReg — pure supervised
pixel reconstruction with perfect input.

Each (decoder_kind, loss_kind) combo writes to /<dec>__<loss>/ in the shared
lewm-snake-ckpts volume.
"""

import time
from pathlib import Path

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.3",
        "einops==0.8.0",
        "pillow==11.0.0",
    )
    .add_local_python_source("snake", "model")
)

vol = modal.Volume.from_name("lewm-snake-ckpts", create_if_missing=True)
CKPT_DIR = "/ckpts"

app = modal.App("lewm-snake")


DECODERS = ["convt", "pixshuf", "crossattn", "perpixel"]
LOSSES = ["mse", "gauss", "cat", "focal", "weight"]


# Generalizable-loss ablation: same convt decoder, vary loss
GENERALIZABLE_RUNS = [
    # (run_name, loss_kind, K)
    ("cat-control",   "cat",        None),
    ("cat-kmeans-K8", "cat-kmeans", 8),
    ("mol-K5",        "mol",        5),
    ("mol-K10",       "mol",        10),
    ("chan-256",      "chan-256",   256),
]

# Class-imbalance ablation: 5 ways to handle "head/food rendered as green"
CE_BALANCE_RUNS = [
    # (run_name, loss_kind, K)
    ("ce-baseline", "cat-kmeans",          8),    # control: K-means raw + plain CE
    ("ce-unique",   "cat-kmeans-unique",   8),    # K-means on UNIQUE colors + plain CE
    ("ce-focal",    "cat-kmeans-focal",    8),    # K-means raw + focal CE
    ("ce-weighted", "cat-kmeans-weighted", 8),    # K-means raw + class-weighted CE
    ("ce-K32",      "cat-kmeans",          32),   # bigger palette so rare colors get own cluster
]

# Precision ablation: same loss (cat-kmeans-unique-K8), vary state encoding / decoder
PRECISION_RUNS = [
    # (run_name, state_encoding, decoder_kind)
    ("pos-baseline",   "baseline",   "convt"),
    ("pos-onehot",     "onehot",     "convt"),
    ("pos-sinusoidal", "sinusoidal", "convt"),
    ("pos-spatial",    "spatial",    "convt"),
    ("pos-pixshuf",    "baseline",   "pixshuf"),
]

# Pinpoint ablation: same loss + dim=16 latent, vary one axis
PINPOINT_RUNS = [
    # (run_name, state_encoding, decoder_kind, freq_K, deep_cnn)
    ("pin-baseline",      "sinusoidal", "convt",   8,  False),
    ("pin-K32freq",       "sinusoidal", "convt",   32, False),
    ("pin-pixshuf",       "sinusoidal", "pixshuf", 8,  False),
    ("pin-spatial",       "spatial",    "convt",   8,  False),
    ("pin-spatial-deep",  "spatial",    "convt",   8,  True),
]
PINPOINT_DIM = 16


@app.function(
    image=image,
    gpu="H100",
    volumes={CKPT_DIR: vol},
    timeout=20 * 60,
)
def train_oracle(
    decoder_kind: str,
    loss_kind: str,
    K: int = None,
    run_name_override: str = None,
    state_encoding: str = "baseline",
    deep_cnn: bool = False,
    epochs: int = 20,
    batch: int = 256,
    lr: float = 1e-3,
    num_episodes: int = 1500,
    dim: int = 128,
    seed: int = 0,
):
    import json
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    from snake import generate_oracle_dataset
    from model import (
        OracleEncoder, OracleEncoderCNN, oracle_decoder_loss, oracle_decoder_out_channels,
        TinyDecoder, SharpDecoder, CrossAttnDecoder, PerPixelDecoder,
        kmeans_palette, kmeans_palette_unique,
    )

    run_name = run_name_override or f"{decoder_kind}__{loss_kind}"
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    out_channels = oracle_decoder_out_channels(loss_kind, K=K)

    print(f"[{run_name}] device={device} bf16={use_bf16} dec={decoder_kind} "
          f"loss={loss_kind} out_channels={out_channels} epochs={epochs} batch={batch}",
          flush=True)

    t0_total = time.time()
    print(f"[{run_name}] generating {num_episodes} episodes (state_encoding={state_encoding}) ...", flush=True)
    t0 = time.time()
    frames_list, states_list = generate_oracle_dataset(num_episodes, seed=seed, encoding=state_encoding)
    frames = np.concatenate(frames_list, axis=0).astype(np.float32) / 255.0   # (N, 64, 64, 3)
    frames = frames.transpose(0, 3, 1, 2)                                      # (N, 3, 64, 64)
    states = np.concatenate(states_list, axis=0)                               # (N, *)
    print(f"[{run_name}] dataset built in {time.time()-t0:.1f}s  N={frames.shape[0]}  state_shape={states.shape}", flush=True)

    frames_t = torch.from_numpy(frames)
    states_t = torch.from_numpy(states)

    # Build K-means palette + class weights if needed.
    palette = None
    class_weights = None
    if loss_kind in ("cat-kmeans", "cat-kmeans-focal", "cat-kmeans-weighted"):
        print(f"[{run_name}] K-means raw with K={K} ...", flush=True)
        sample_pix = frames_t[:400].permute(0, 2, 3, 1).reshape(-1, 3)
        palette = kmeans_palette(sample_pix.to(device), K=K, n_samples=400_000, n_iter=20).cpu()
        print(f"[{run_name}] palette: {palette.tolist()}", flush=True)
    elif loss_kind == "cat-kmeans-unique":
        print(f"[{run_name}] K-means on UNIQUE colors with K={K} ...", flush=True)
        sample_pix = frames_t[:400].permute(0, 2, 3, 1).reshape(-1, 3)
        palette = kmeans_palette_unique(sample_pix.to(device), K=K).cpu()
        print(f"[{run_name}] palette (unique): {palette.tolist()}", flush=True)

    if loss_kind == "cat-kmeans-weighted":
        # Compute per-class freq over the dataset → inverse-sqrt weights
        print(f"[{run_name}] computing class frequencies ...", flush=True)
        N_sample = min(200, frames_t.size(0))
        sf = frames_t[:N_sample].to(device)
        with torch.no_grad():
            p_t = palette.to(device, dtype=sf.dtype).view(1, K, 3, 1, 1)
            dist = (sf.unsqueeze(1) - p_t).pow(2).sum(dim=2)
            labels = dist.argmin(dim=1).reshape(-1)
            counts = torch.bincount(labels, minlength=K).float()
        freq = counts / counts.sum()
        class_weights = (1.0 / (freq.sqrt() + 1e-3))
        class_weights = class_weights * (K / class_weights.sum())  # normalise mean to 1
        class_weights = class_weights.cpu()
        print(f"[{run_name}] class freq: {freq.tolist()}", flush=True)
        print(f"[{run_name}] class weights: {class_weights.tolist()}", flush=True)

    class OracleSet(Dataset):
        def __len__(self): return frames_t.size(0)
        def __getitem__(self, idx): return states_t[idx], frames_t[idx]

    loader = DataLoader(
        OracleSet(), batch_size=batch, shuffle=True, num_workers=4,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )

    if state_encoding == "spatial":
        enc = OracleEncoderCNN(in_channels=states.shape[1], out_dim=dim, deep=deep_cnn).to(device)
    else:
        enc = OracleEncoder(in_dim=states.shape[1], out_dim=dim).to(device)

    if decoder_kind == "convt":
        dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels)
    elif decoder_kind == "pixshuf":
        dec = _make_pixshuf(dim=dim, out_channels=out_channels)
    elif decoder_kind == "crossattn":
        dec = _make_crossattn(dim=dim, out_channels=out_channels)
    elif decoder_kind == "perpixel":
        dec = _make_perpixel(dim=dim, out_channels=out_channels)
    else:
        raise ValueError(decoder_kind)
    dec = dec.to(device)

    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"[{run_name}] params: {n_params/1e6:.2f}M  steps/epoch: {len(loader)}", flush=True)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    run_dir = Path(CKPT_DIR, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        run_name=run_name, decoder_kind=decoder_kind, loss_kind=loss_kind,
        K=K, out_channels=out_channels, dim=dim, epochs=epochs, batch=batch, lr=lr,
        num_episodes=num_episodes, mode="oracle",
        state_dim=states.shape[1] if states.ndim == 2 else None,
        state_encoding=state_encoding,
        state_shape=list(states.shape[1:]),
        deep_cnn=deep_cnn,
    )
    if palette is not None:
        cfg["palette_K"] = palette.shape[0]
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    autocast_kw = dict(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else dict(device_type="cpu", enabled=False)
    step = 0
    log_every = 100
    train_t0 = time.time()

    for epoch in range(epochs):
        enc.train(); dec.train()
        agg_loss = 0.0
        n = 0
        ep_t0 = time.time()
        for state, frame in loader:
            state = state.to(device, non_blocking=True)
            frame = frame.to(device, non_blocking=True)
            with torch.amp.autocast(**autocast_kw):
                z = enc(state)                                    # (B, dim)
                recon = dec(z)
                pal_t = palette.to(device) if palette is not None else None
                cw_t = class_weights.to(device) if class_weights is not None else None
                loss = oracle_decoder_loss(recon, frame, loss_kind, K=K, palette=pal_t, class_weights=cw_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
            sched.step()
            agg_loss += loss.item(); n += 1; step += 1
            if step % log_every == 0:
                msg = (f"[{run_name} step {step:>6}] ep {epoch+1:02d}/{epochs} "
                       f"loss={agg_loss/n:.4f} ({time.time()-train_t0:.0f}s)")
                print(msg, flush=True)

        # held-out FG-MSE for diagnostics
        enc.eval(); dec.eval()
        with torch.no_grad():
            n_eval = min(2048, frames_t.size(0))
            idx = torch.randperm(frames_t.size(0))[:n_eval]
            sf = states_t[idx].to(device)
            ft = frames_t[idx].to(device)
            with torch.amp.autocast(**autocast_kw):
                z = enc(sf)
                recon = dec(z)
            from model import SNAKE_COLORS
            bg = SNAKE_COLORS[0].to(device, dtype=ft.dtype).view(1, 3, 1, 1)
            mean = recon[:, :3].sigmoid() if recon.size(1) >= 3 else recon
            fg = (ft - bg).pow(2).sum(dim=1, keepdim=True).gt(0.005).to(ft.dtype)
            fg_mse = ((mean - ft).pow(2) * fg).sum() / (fg.sum() * 3 + 1e-9)

        ckpt_path = run_dir / f"epoch_{epoch+1:03d}.pt"
        payload = {
            "epoch": epoch + 1, "step": step,
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "loss": agg_loss / max(n, 1),
            "fg_mse": float(fg_mse),
            "cfg": cfg,
        }
        if palette is not None:
            payload["palette"] = palette.cpu()
        torch.save(payload, ckpt_path)
        torch.save(payload, run_dir / "latest.pt")
        vol.commit()
        print(f"[{run_name}] saved {ckpt_path.name}  ep_loss={agg_loss/max(n,1):.4f}  "
              f"fg_mse={float(fg_mse):.5f}  ({time.time()-ep_t0:.1f}s)", flush=True)

    print(f"[{run_name}] DONE in {time.time()-train_t0:.1f}s training, {time.time()-t0_total:.1f}s total", flush=True)


# Auxiliary decoder factories with variable out_channels and raw (no-activation) output

def _make_pixshuf(dim, out_channels, hidden=128, grid=16, scale=4):
    import torch.nn as nn
    class PixShufDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = grid
            self.scale = scale
            self.hidden = hidden
            self.fc = nn.Linear(dim, grid * grid * hidden)
            self.refine = nn.Sequential(
                nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv2d(hidden, hidden, 3, 1, 1),
                nn.GroupNorm(8, hidden), nn.SiLU(),
            )
            self.head = nn.Conv2d(hidden, out_channels * scale * scale, 1)
            self.shuffle = nn.PixelShuffle(scale)

        def forward(self, z):
            B = z.size(0)
            x = self.fc(z).view(B, self.hidden, self.grid, self.grid)
            x = x + self.refine(x)
            x = self.head(x)
            return self.shuffle(x)
    return PixShufDecoder()


def _make_crossattn(dim, out_channels, n_queries=256, n_blocks=3, heads=4,
                     mlp_dim=512, grid=16, scale=4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from model import CrossAttnBlock

    class XAttnDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = grid
            self.scale = scale
            self.out_c = out_channels
            self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
            self.in_proj = nn.Linear(dim, dim)
            self.blocks = nn.ModuleList([CrossAttnBlock(dim, heads, mlp_dim) for _ in range(n_blocks)])
            self.out_norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, out_channels * scale * scale)

        def forward(self, z):
            B = z.size(0)
            memory = self.in_proj(z).unsqueeze(1)
            q = self.queries.expand(B, -1, -1).contiguous()
            for blk in self.blocks:
                q = blk(q, memory)
            q = self.out_norm(q)
            q = self.head(q)                                              # (B, P, oc*s*s)
            q = q.transpose(1, 2).reshape(B, self.out_c * self.scale * self.scale,
                                           self.grid, self.grid)
            return F.pixel_shuffle(q, self.scale)                          # (B, out_c, H, W)
    return XAttnDecoder()


def _make_perpixel(dim, out_channels, img_size=64, hidden=256):
    import torch
    import torch.nn as nn

    class PerPixelDec(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_size = img_size
            n_px = img_size * img_size
            self.pos = nn.Parameter(torch.randn(1, n_px, dim) * 0.02)
            self.net = nn.Sequential(
                nn.Linear(2 * dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, out_channels),
            )

        def forward(self, z):
            B = z.size(0)
            n_px = self.img_size * self.img_size
            z_e = z.unsqueeze(1).expand(-1, n_px, -1)
            pos = self.pos.expand(B, -1, -1)
            x = torch.cat([z_e, pos], dim=-1)
            out = self.net(x)                                              # (B, P, oc)
            return out.transpose(1, 2).reshape(B, -1, self.img_size, self.img_size)
    return PerPixelDec()


@app.local_entrypoint()
def main():
    print(f"Spawning {len(PINPOINT_RUNS)} parallel H100 jobs (pinpoint ablation, dim={PINPOINT_DIM}) ...")
    for run_name, state_enc, dec_kind, freq_K, deep in PINPOINT_RUNS:
        # `freq_K` selects which sinusoidal encoding to use
        actual_state_enc = state_enc
        if state_enc == "sinusoidal" and freq_K == 32:
            actual_state_enc = "sinusoidal-K32"
        h = train_oracle.spawn(
            decoder_kind=dec_kind,
            loss_kind="cat-kmeans-unique",
            K=8,
            run_name_override=run_name,
            state_encoding=actual_state_enc,
            deep_cnn=deep,
            dim=PINPOINT_DIM,
        )
        print(f"  spawned {run_name} (state={actual_state_enc}, dec={dec_kind}, deep={deep}, dim={PINPOINT_DIM}): {h.object_id}")
    print("All jobs spawned and detached. Local entrypoint exiting; jobs continue on Modal.")
