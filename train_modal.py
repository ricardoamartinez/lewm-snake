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

# Frozen-oracle predictor ablation: 5 ways to learn dynamics in 16-d latent
PREDICTOR_RUNS = [
    # (run_name, predictor_kind, multi_step_horizon, history)
    ("pred-mlp",         "mlp",        1, 1),
    ("pred-residual",    "residual",   1, 1),
    ("pred-transformer", "transformer", 1, 4),
    ("pred-multistep",   "multistep",  4, 1),
    ("pred-rnn",         "rnn",        1, 1),
]


# Manifold-drift fix ablation: 20 ways to make predictor outputs renderable
MANIFOLD_FIX_RUNS = [
    # name,                  predictor,     quantizer,   joint, oracle_ep, pred_ep, pred_loss,    multi_step, decoder_noise
    ("c-mlp-baseline",       "mlp",         "none",      False, 10, 15, "mse",        1, 0.0),
    ("c-residual",           "residual",    "none",      False, 10, 15, "mse",        1, 0.0),
    ("c-transformer",        "transformer", "none",      False, 10, 15, "mse",        1, 0.0),
    ("c-multistep",          "mlp",         "none",      False, 10, 15, "mse",        4, 0.0),
    ("c-rnn",                "rnn",         "none",      False, 10, 15, "mse",        1, 0.0),
    ("fsq-8d5l-mlp",         "mlp",         "fsq8x5",    False, 10, 15, "mse",        1, 0.0),
    ("fsq-16d4l-mlp",        "mlp",         "fsq16x4",   False, 10, 15, "mse",        1, 0.0),
    ("fsq-4d8l-mlp",         "mlp",         "fsq4x8",    False, 10, 15, "mse",        1, 0.0),
    ("vq-K512-mlp",          "mlp",         "vq-K512",   False, 10, 15, "mse",        1, 0.0),
    ("vq-K1024-mlp",         "mlp",         "vq-K1024",  False, 10, 15, "mse",        1, 0.0),
    ("joint-mlp",            "mlp",         "none",      True,  10, 15, "mse",        1, 0.0),
    ("joint-residual",       "residual",    "none",      True,  10, 15, "mse",        1, 0.0),
    ("joint-multistep",      "mlp",         "none",      True,  10, 15, "mse",        4, 0.0),
    ("joint-fsq",            "mlp",         "fsq8x5",    True,  10, 15, "mse",        1, 0.0),
    ("joint-vq",             "mlp",         "vq-K512",   True,  10, 15, "mse",        1, 0.0),
    ("dist-noise-0.1",       "mlp",         "none",      False, 10, 15, "mse",        1, 0.1),
    ("dist-noise-0.3",       "mlp",         "none",      False, 10, 15, "mse",        1, 0.3),
    ("long-train",           "mlp",         "none",      False, 30, 30, "mse",        1, 0.0),
    ("rollout-train",        "mlp",         "none",      False, 10, 15, "rollout",    8, 0.0),
    ("rollout-fsq",          "mlp",         "fsq8x5",    False, 10, 15, "rollout",    8, 0.0),
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={CKPT_DIR: vol},
    timeout=30 * 60,
)
def train_predictor(
    run_name: str,
    predictor_kind: str,
    multi_step: int = 1,
    history: int = 1,
    oracle_epochs: int = 10,
    pred_epochs: int = 15,
    batch: int = 256,
    lr: float = 1e-3,
    num_episodes: int = 1500,
    dim: int = 16,
    seed: int = 0,
):
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    from snake import generate_oracle_dataset
    from model import (
        OracleEncoderCNN, TinyDecoder, kmeans_palette_unique,
        oracle_decoder_loss, oracle_decoder_out_channels, make_predictor,
    )

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    autocast_kw = dict(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else dict(device_type="cpu", enabled=False)
    K = 8
    out_channels = oracle_decoder_out_channels("cat-kmeans-unique", K=K)

    print(f"[{run_name}] predictor={predictor_kind} multi_step={multi_step} history={history} dim={dim}", flush=True)

    print(f"[{run_name}] generating {num_episodes} episodes (spatial state) ...", flush=True)
    t0 = time.time()
    frames_list, states_list, actions_list = generate_oracle_dataset(
        num_episodes, seed=seed, encoding="spatial", return_actions=True,
    )

    print(f"[{run_name}] dataset built in {time.time()-t0:.1f}s", flush=True)

    # K-means palette on UNIQUE colors (keep channel-last before flatten — each row = one (R,G,B) triplet)
    sample_pix = torch.from_numpy(np.stack([f for fr in frames_list[:200] for f in fr])).float() / 255.0  # (N, 64, 64, 3)
    palette = kmeans_palette_unique(sample_pix.to(device).reshape(-1, 3), K=K).cpu()
    print(f"[{run_name}] palette: {palette.tolist()}", flush=True)

    # Stage 1: train oracle encoder + decoder on (state, frame) pairs
    enc = OracleEncoderCNN(in_channels=4, out_dim=dim).to(device)
    dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels).to(device)

    all_frames = []
    all_states = []
    all_actions_aligned = []  # action that goes from frame i to frame i+1 (last frame has no action)
    for f, s, a in zip(frames_list, states_list, actions_list):
        n = min(len(f), len(s), len(a) + 1)
        all_frames.append(f[:n])
        all_states.append(s[:n])
        all_actions_aligned.append(np.concatenate([a[:n - 1], np.array([0], dtype=np.int64)]))  # placeholder for last
    frames_t = torch.from_numpy(np.concatenate(all_frames, axis=0)).float().permute(0, 3, 1, 2) / 255.0
    states_t = torch.from_numpy(np.concatenate(all_states, axis=0))
    actions_t = torch.from_numpy(np.concatenate(all_actions_aligned, axis=0))
    # Build episode boundaries (which transitions are valid, i.e., not the last frame of an episode)
    ep_lengths = np.array([len(s) for s in all_states])
    ep_ends = np.cumsum(ep_lengths) - 1   # the index of the last frame of each episode
    valid_transition = np.ones(len(states_t), dtype=bool)
    valid_transition[ep_ends] = False     # no transition out of the last frame
    valid_idx_t = torch.from_numpy(np.where(valid_transition)[0]).long()

    print(f"[{run_name}] N frames {frames_t.size(0)}, N transitions {valid_idx_t.size(0)}", flush=True)

    class FrameSet(Dataset):
        def __len__(self): return frames_t.size(0)
        def __getitem__(self, i): return states_t[i], frames_t[i]

    loader = DataLoader(FrameSet(), batch_size=batch, shuffle=True, num_workers=2, drop_last=True, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=oracle_epochs * len(loader))
    palette_t = palette.to(device)

    print(f"[{run_name}] stage 1: training oracle encoder+decoder for {oracle_epochs} epochs", flush=True)
    for epoch in range(oracle_epochs):
        enc.train(); dec.train()
        ep_t0 = time.time()
        total = 0.0; n = 0
        for state, frame in loader:
            state = state.to(device, non_blocking=True)
            frame = frame.to(device, non_blocking=True)
            with torch.amp.autocast(**autocast_kw):
                z = enc(state)
                raw = dec(z)
                loss = oracle_decoder_loss(raw, frame, "cat-kmeans-unique", K=K, palette=palette_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            total += loss.item(); n += 1
        print(f"[{run_name}] oracle ep {epoch+1}/{oracle_epochs} loss={total/n:.4f} ({time.time()-ep_t0:.1f}s)", flush=True)

    enc.eval(); dec.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)

    # Stage 2: train predictor on (z_t, a_t) -> z_{t+1} MSE in latent space
    print(f"[{run_name}] stage 2: training predictor for {pred_epochs} epochs", flush=True)
    predictor = make_predictor(predictor_kind, dim=dim).to(device)
    action_embed = nn.Embedding(4, dim).to(device)
    opt2 = torch.optim.AdamW(list(predictor.parameters()) + list(action_embed.parameters()), lr=lr)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=pred_epochs * (valid_idx_t.numel() // batch + 1))

    states_t = states_t.to(device)
    actions_t = actions_t.to(device)
    valid_idx_t = valid_idx_t.to(device)

    # Pre-encode all states once (frozen oracle)
    print(f"[{run_name}] pre-encoding all states ...", flush=True)
    with torch.no_grad():
        Z = []
        bs = 1024
        for i in range(0, states_t.size(0), bs):
            with torch.amp.autocast(**autocast_kw):
                z = enc(states_t[i:i + bs])
            Z.append(z.float())
        Z = torch.cat(Z, dim=0)
    print(f"[{run_name}] Z shape: {tuple(Z.shape)}", flush=True)

    for epoch in range(pred_epochs):
        predictor.train()
        action_embed.train()
        ep_t0 = time.time()
        # shuffle valid transitions
        perm = torch.randperm(valid_idx_t.numel(), device=device)
        total = 0.0; n = 0
        for s in range(0, valid_idx_t.numel() - batch, batch):
            idx = valid_idx_t[perm[s:s + batch]]
            z_t = Z[idx]                                        # (B, dim)
            a_t = action_embed(actions_t[idx])                   # (B, dim)

            if predictor_kind == "transformer":
                hist_idx = torch.stack([idx - h for h in range(history - 1, -1, -1)], dim=1).clamp(min=0)  # (B, H)
                z_hist = Z[hist_idx]                              # (B, H, dim)
                a_hist = action_embed(actions_t[hist_idx])
                with torch.amp.autocast(**autocast_kw):
                    pred = predictor.step(z_hist, a_hist)         # (B, dim)
                target = Z[idx + 1].detach()
                loss = F.mse_loss(pred, target)
            elif predictor_kind == "rnn":
                # one-step training: feed (z_t, a_t), reset hidden each batch element
                with torch.amp.autocast(**autocast_kw):
                    pred, _ = predictor(z_t, a_t)
                target = Z[idx + 1].detach()
                loss = F.mse_loss(pred, target)
            elif predictor_kind == "multistep":
                # roll predictor forward `multi_step` steps, sum losses
                with torch.amp.autocast(**autocast_kw):
                    z_hat = z_t
                    loss = z_t.new_zeros(())
                    for k in range(multi_step):
                        next_idx = (idx + k + 1).clamp(max=Z.size(0) - 1)
                        # mask out invalid (crossing episode boundary)
                        in_ep = (idx + k + 1 < Z.size(0)) & (~torch.isin(idx + k, valid_idx_t.new_tensor(ep_ends)))
                        a_k = action_embed(actions_t[(idx + k).clamp(max=actions_t.numel() - 1)])
                        z_hat = predictor(z_hat, a_k)
                        target_k = Z[next_idx].detach()
                        loss = loss + F.mse_loss(z_hat[in_ep], target_k[in_ep]) if in_ep.any() else loss
                    loss = loss / multi_step
            else:  # mlp / residual
                with torch.amp.autocast(**autocast_kw):
                    pred = predictor(z_t, a_t)
                target = Z[idx + 1].detach()
                loss = F.mse_loss(pred, target)

            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step(); sched2.step()
            total += loss.item(); n += 1
        print(f"[{run_name}] predictor ep {epoch+1}/{pred_epochs} loss={total/n:.5f} ({time.time()-ep_t0:.1f}s)", flush=True)

        # save checkpoint each epoch
        run_dir = Path(CKPT_DIR, run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(
            run_name=run_name, mode="predictor",
            predictor_kind=predictor_kind, multi_step=multi_step, history=history,
            dim=dim, K=K, out_channels=out_channels,
            decoder_kind="convt", state_encoding="spatial",
            state_shape=[4, 64, 64], deep_cnn=False,
            num_episodes=num_episodes, batch=batch,
        )
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        torch.save({
            "epoch": epoch + 1, "step": (epoch + 1) * n,
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "predictor_state": predictor.state_dict(),
            "action_embed_state": action_embed.state_dict(),
            "palette": palette,
            "cfg": cfg,
            "predictor_loss": total / max(n, 1),
        }, run_dir / "latest.pt")
        torch.save({
            "epoch": epoch + 1, "step": (epoch + 1) * n,
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "predictor_state": predictor.state_dict(),
            "action_embed_state": action_embed.state_dict(),
            "palette": palette,
            "cfg": cfg,
        }, run_dir / f"epoch_{epoch+1:03d}.pt")
        vol.commit()
    print(f"[{run_name}] DONE", flush=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={CKPT_DIR: vol},
    timeout=60 * 60,
)
def train_arch_jepa(
    run_name: str,
    arch_kind: str,             # "flat", "spatial-16", "spatial-8", "spatial-4"
    loss_kind: str = "cat-kmeans-perclass",
    epochs: int = 30,
    batch: int = 128,
    lr: float = 5e-4,
    num_episodes: int = 1500,
    history: int = 4,
    pred_horizon: int = 4,
    dim: int = 16,
    seed: int = 0,
    pred_lambda: float = 1.0,
    dec_lambda: float = 1.0,
    rollout_dec: bool = True,   # if True, decoder loss is on PREDICTED latents (option B);
                                 # if False, on encoded latents (option A — manifold-locked)
):
    """Full JEPA system with spatial latent: encoder + ConvPredictor + decoder.
    Trains on T-step windows with both pred MSE in latent space and dec CE in pixel space.
    """
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    from snake import generate_dataset
    from model import (
        OracleEncoderCNN, TinyDecoder, SpatialEncoder, SpatialDecoder,
        MLPPredictor, ConvPredictor, make_predictor,
        kmeans_palette_unique, oracle_decoder_loss, oracle_decoder_out_channels,
    )

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    autocast_kw = dict(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else dict(device_type="cpu", enabled=False)

    K_PALETTE = 8
    out_channels = oracle_decoder_out_channels("cat-kmeans-unique", K=K_PALETTE)
    print(f"[{run_name}] JEPA arch={arch_kind} loss={loss_kind} dim={dim} epochs={epochs} rollout_dec={rollout_dec}", flush=True)

    print(f"[{run_name}] generating {num_episodes} episodes ...", flush=True)
    t0 = time.time()
    frames_list, actions_list = generate_dataset(num_episodes, seed=seed)
    print(f"[{run_name}] dataset built in {time.time()-t0:.1f}s", flush=True)

    # T-step windows
    T = history + pred_horizon
    windows = []
    for f, a in zip(frames_list, actions_list):
        if len(f) < T:
            continue
        for i in range(len(f) - T + 1):
            windows.append((f[i:i + T], a[i:i + T - 1]))
    print(f"[{run_name}] training windows: {len(windows)}  T={T}", flush=True)

    sample_pix = torch.from_numpy(np.stack([f for fr in frames_list[:200] for f in fr])).float() / 255.0
    palette = kmeans_palette_unique(sample_pix.to(device).reshape(-1, 3), K=K_PALETTE).cpu()
    palette_t = palette.to(device)
    print(f"[{run_name}] palette: {palette.tolist()}", flush=True)

    class WMSet(Dataset):
        def __len__(self): return len(windows)
        def __getitem__(self, i):
            f, a = windows[i]
            return torch.from_numpy(f).float().permute(0, 3, 1, 2) / 255.0, torch.from_numpy(a).long()

    loader = DataLoader(
        WMSet(), batch_size=batch, shuffle=True, num_workers=4,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )

    # Build architecture. Parse "spatial-LAT[-deep][-bigCh]" patterns.
    is_spatial = arch_kind.startswith("spatial")
    pred_kind = "conv" if is_spatial else "mlp"
    if arch_kind == "flat":
        enc = OracleEncoderCNN(in_channels=3, out_dim=dim).to(device)
        dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels).to(device)
    elif is_spatial:
        parts = arch_kind.split("-")          # ["spatial", "16"] or ["spatial", "32", "deep"] etc.
        lat_size = int(parts[1])
        deep = "deep" in parts
        bigch = "bigCh" in parts
        enc_refine = 3 if deep else 1
        dec_refine = 3 if deep else 1
        enc_base = 64 if bigch else 32
        dec_base = 256 if bigch else 128
        enc = SpatialEncoder(in_channels=3, dim=dim, lat_size=lat_size,
                             base_ch=enc_base, refine_blocks=enc_refine).to(device)
        dec = SpatialDecoder(dim=dim, out_channels=out_channels, lat_size=lat_size,
                             base_ch=dec_base, refine_blocks=dec_refine).to(device)
    else:
        raise ValueError(arch_kind)

    pred = make_predictor(pred_kind, dim=dim).to(device)
    action_embed = nn.Embedding(4, dim).to(device)

    params = (list(enc.parameters()) + list(pred.parameters())
              + list(dec.parameters()) + list(action_embed.parameters()))
    n_params = sum(p.numel() for p in params)
    print(f"[{run_name}] params: {n_params/1e6:.2f}M  pred={pred_kind}  steps/epoch: {len(loader)}", flush=True)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    run_dir = Path(CKPT_DIR, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        run_name=run_name, mode="arch_jepa", arch_kind=arch_kind,
        loss_kind=loss_kind, predictor_kind=pred_kind, dim=dim,
        K=K_PALETTE, out_channels=out_channels,
        epochs=epochs, batch=batch, num_episodes=num_episodes,
        history=history, pred_horizon=pred_horizon,
        rollout_dec=rollout_dec, is_spatial=is_spatial,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train_t0 = time.time()
    for epoch in range(epochs):
        enc.train(); pred.train(); dec.train(); action_embed.train()
        total = {"loss": 0.0, "pred": 0.0, "dec": 0.0}; n = 0
        ep_t0 = time.time()
        for f, a in loader:
            f = f.to(device, non_blocking=True)             # (B, T, 3, 64, 64)
            a = a.to(device, non_blocking=True)             # (B, T-1)
            B, T_, C, H, W = f.shape
            with torch.amp.autocast(**autocast_kw):
                # Encode all frames
                emb = enc(f.reshape(B * T_, C, H, W))
                if is_spatial:
                    emb = emb.view(B, T_, dim, emb.size(-2), emb.size(-1))
                else:
                    emb = emb.view(B, T_, dim)
                act_emb = action_embed(a)                    # (B, T-1, dim)

                # Rollout: z_hat[0] = emb[0], then predictor
                z_hat = [emb[:, 0]]
                for t in range(T_ - 1):
                    if is_spatial:
                        z_next = pred(z_hat[-1], act_emb[:, t])
                    else:
                        z_next = pred(z_hat[-1], act_emb[:, t])
                    z_hat.append(z_next)
                z_hat_stack = torch.stack(z_hat, dim=1)      # (B, T, dim, [H_lat, W_lat])

                # Pred loss: rollout matches encoded targets (teacher signal)
                pred_loss = (z_hat_stack[:, 1:] - emb[:, 1:].detach()).pow(2).mean()

                # Dec loss: render and compare to GT pixels
                z_for_dec = z_hat_stack if rollout_dec else emb
                if is_spatial:
                    raw = dec(z_for_dec.reshape(B * T_, dim, z_for_dec.size(-2), z_for_dec.size(-1)))
                else:
                    raw = dec(z_for_dec.reshape(B * T_, dim))
                pix_target = f.reshape(B * T_, C, H, W)
                dec_loss = oracle_decoder_loss(
                    raw, pix_target, loss_kind, K=K_PALETTE, palette=palette_t,
                )

                loss = pred_lambda * pred_loss + dec_lambda * dec_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            total["loss"] += loss.item()
            total["pred"] += pred_loss.item()
            total["dec"] += dec_loss.item()
            n += 1

        payload = {
            "epoch": epoch + 1, "stage": "arch_jepa",
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "predictor_state": pred.state_dict(),
            "action_embed_state": action_embed.state_dict(),
            "palette": palette,
            "cfg": cfg,
            **{k: total[k] / max(n, 1) for k in total},
        }
        torch.save(payload, run_dir / f"epoch_{epoch+1:03d}.pt")
        torch.save(payload, run_dir / "latest.pt")
        vol.commit()
        print(f"[{run_name}] ep {epoch+1}/{epochs} "
              f"loss={total['loss']/n:.4f} pred={total['pred']/n:.4f} "
              f"dec={total['dec']/n:.4f} ({time.time()-ep_t0:.1f}s)", flush=True)

    print(f"[{run_name}] DONE in {time.time()-train_t0:.0f}s", flush=True)


# Precision ablation v3 — H100 (80 GB) so we can do spatial-64 (pixel-perfect)
# and bigCh variants. All pred_lambda=0, rollout dec_loss only.
ARCH_RUNS = [
    # (run_name,                       arch_kind)
    ("prec3-spatial-32",               "spatial-32"),
    ("prec3-spatial-64",               "spatial-64"),               # 1 px per cell
    ("prec3-spatial-32-deep",          "spatial-32-deep"),
    ("prec3-spatial-64-deep",          "spatial-64-deep"),
    ("prec3-spatial-32-deep-bigCh",    "spatial-32-deep-bigCh"),
]


# Class-balance ablation: TRUE root-cause is per-pixel loss averaging.
# GT distribution: BG=4090, body=4, head=1, food=1 (per 4096-pixel frame).
# Plain CE means head/food contribute 1/4096 of the gradient -> never learned.
# Test 5 strategies, all pure AE, dim=16, plain training.
BG_DIAG_RUNS = [
    # (run_name,            dim, pred_lambda, sigreg_lambda, dec_lambda, loss_kind,                       focal_gamma, bg_weight)
    ("bal-control",         16,  0.0,         0.0,           1.0,        "cat-kmeans-unique",            0.0,         0.0),
    ("bal-perclass",        16,  0.0,         0.0,           1.0,        "cat-kmeans-perclass",          0.0,         0.0),
    ("bal-perclass-focal2", 16,  0.0,         0.0,           1.0,        "cat-kmeans-perclass-focal",    2.0,         0.0),
    ("bal-weighted",        16,  0.0,         0.0,           1.0,        "cat-kmeans-weighted",          0.0,         0.0),
    ("bal-focal-g10",       16,  0.0,         0.0,           1.0,        "cat-kmeans-focal",             10.0,        0.0),
]


# Legacy ablation (kept for reference — superseded by BG_DIAG_RUNS)
FULL_SYSTEM_RUNS = [
    ("full-mlp-dg",            "mlp",       0.0, 1, True),
    ("full-residual-dg",       "residual",  0.0, 1, True),
    ("full-multistep-dg",      "mlp",       0.0, 4, True),
    ("full-decnoise-dg",       "mlp",       0.3, 1, True),
    ("full-kitchensink-dg",    "residual",  0.3, 4, True),
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={CKPT_DIR: vol},
    timeout=60 * 60,
)
def train_full(
    run_name: str,
    predictor_kind: str,
    dec_noise: float,
    multi_step: int,
    dec_grad: bool = True,
    epochs: int = 50,
    batch: int = 256,
    lr: float = 5e-4,
    sigreg_lambda: float = 0.1,
    num_episodes: int = 1500,
    history: int = 4,
    pred_horizon: int = 4,
    dim: int = 16,
    seed: int = 0,
    loss_kind: str = "cat-kmeans-focal",
    focal_gamma: float = 2.0,
    bg_weight: float = 0.0,
    pred_lambda: float = 1.0,
    dec_lambda: float = 1.0,
):
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    from snake import generate_dataset
    from model import (
        OracleEncoderCNN, TinyDecoder, SIGReg, kmeans_palette_unique,
        oracle_decoder_loss, oracle_decoder_out_channels, make_predictor,
    )

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    autocast_kw = dict(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else dict(device_type="cpu", enabled=False)

    K_PALETTE = 8
    out_channels = oracle_decoder_out_channels("cat-kmeans-unique", K=K_PALETTE)
    print(f"[{run_name}] FULL: predictor={predictor_kind} dec_noise={dec_noise} "
          f"multi_step={multi_step} dim={dim} epochs={epochs}", flush=True)

    print(f"[{run_name}] generating {num_episodes} episodes ...", flush=True)
    t0 = time.time()
    frames_list, actions_list = generate_dataset(num_episodes, seed=seed)
    print(f"[{run_name}] dataset built in {time.time()-t0:.1f}s", flush=True)

    # Build sliding windows (T = history + pred_horizon frames)
    T = history + pred_horizon
    windows = []
    for f, a in zip(frames_list, actions_list):
        if len(f) < T:
            continue
        for i in range(len(f) - T + 1):
            windows.append((f[i:i + T], a[i:i + T - 1]))
    print(f"[{run_name}] training windows: {len(windows)}  T={T}", flush=True)

    class WMSet(Dataset):
        def __len__(self): return len(windows)
        def __getitem__(self, idx):
            f, a = windows[idx]
            return torch.from_numpy(f).float().permute(0, 3, 1, 2) / 255.0, torch.from_numpy(a).long()

    loader = DataLoader(
        WMSet(), batch_size=batch, shuffle=True, num_workers=4,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )

    # K-means palette on UNIQUE colors (channel-last reshape — correct triplets)
    sample_pix = torch.from_numpy(np.stack([f for fr in frames_list[:200] for f in fr])).float() / 255.0  # (N, 64, 64, 3)
    palette = kmeans_palette_unique(sample_pix.to(device).reshape(-1, 3), K=K_PALETTE).cpu()
    palette_t = palette.to(device)
    print(f"[{run_name}] palette: {palette.tolist()}", flush=True)

    # Compute per-class inverse-frequency weights (used by cat-kmeans-weighted)
    class_weights_t = None
    if loss_kind == "cat-kmeans-weighted":
        sf = sample_pix[:200].permute(0, 3, 1, 2).to(device)  # (N, 3, 64, 64)
        with torch.no_grad():
            p_t = palette.to(device, dtype=sf.dtype).view(1, palette.size(0), 3, 1, 1)
            dist = (sf.unsqueeze(1) - p_t).pow(2).sum(dim=2)
            labels = dist.argmin(dim=1).reshape(-1)
            counts = torch.bincount(labels, minlength=palette.size(0)).float()
        freq = counts / counts.sum().clamp(min=1.0)
        # Pad to K_PALETTE channels in case palette has fewer than K real entries
        K_actual = palette.size(0)
        cw = torch.zeros(K_PALETTE, device=device)
        cw_real = (1.0 / (freq.sqrt() + 1e-3))
        cw_real = cw_real * (K_actual / cw_real.sum())
        cw[:K_actual] = cw_real
        class_weights_t = cw
        print(f"[{run_name}] class freq: {freq.tolist()}  weights: {cw.tolist()}", flush=True)

    # Build full system: pixel CNN encoder + predictor + decoder
    enc = OracleEncoderCNN(in_channels=3, out_dim=dim).to(device)
    pred = make_predictor(predictor_kind, dim=dim).to(device)
    dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels).to(device)
    action_embed = nn.Embedding(4, dim).to(device)
    sigreg = SIGReg(knots=17, num_proj=512).to(device)

    n_params = (sum(p.numel() for p in enc.parameters())
                + sum(p.numel() for p in pred.parameters())
                + sum(p.numel() for p in dec.parameters())
                + sum(p.numel() for p in action_embed.parameters()))
    print(f"[{run_name}] params: {n_params/1e6:.2f}M  steps/epoch: {len(loader)}", flush=True)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()) + list(dec.parameters())
        + list(action_embed.parameters()), lr=lr, weight_decay=1e-3,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    run_dir = Path(CKPT_DIR, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        run_name=run_name, mode="full", encoder_kind="pixel-cnn",
        predictor_kind=predictor_kind, dec_noise=dec_noise, multi_step=multi_step,
        dec_grad=dec_grad,
        history=history, pred_horizon=pred_horizon, dim=dim,
        K=K_PALETTE, out_channels=out_channels, decoder_kind="convt",
        num_episodes=num_episodes, batch=batch, epochs=epochs,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train_t0 = time.time()
    for epoch in range(epochs):
        enc.train(); pred.train(); dec.train(); action_embed.train()
        agg = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0, "decoder": 0.0}
        n = 0
        ep_t0 = time.time()
        for f, a in loader:
            f = f.to(device, non_blocking=True)                                   # (B, T, 3, 64, 64)
            a = a.to(device, non_blocking=True)                                   # (B, T-1)
            with torch.amp.autocast(**autocast_kw):
                B, T_, C, H, W = f.shape
                emb = enc(f.reshape(B * T_, C, H, W)).reshape(B, T_, dim)         # (B, T, dim)
                act_emb = action_embed(a)                                          # (B, T-1, dim)

                # JEPA prediction loss (multi-step teacher-forced)
                pred_loss = emb.new_zeros(())
                steps_used = 0
                for k in range(min(multi_step, T_ - 1)):
                    if predictor_kind == "transformer":
                        z_hat = pred.step(emb[:, :T_ - 1 - k], act_emb[:, :T_ - 1 - k])
                    elif predictor_kind == "rnn":
                        z_hat, _ = pred(emb[:, k], act_emb[:, k])
                    else:
                        z_hat = pred(emb[:, k], act_emb[:, k])
                    target = emb[:, k + 1].detach()
                    pred_loss = pred_loss + (z_hat - target).pow(2).mean()
                    steps_used += 1
                pred_loss = pred_loss / max(steps_used, 1)

                # SIGReg on the encoder output (T, B, D)
                sigreg_loss = sigreg(emb.transpose(0, 1))

                # Decoder loss — dec_grad controls whether decoder loss reaches encoder
                z_for_dec = emb if dec_grad else emb.detach()
                z_flat = z_for_dec.reshape(B * T_, dim)
                if dec_noise > 0:
                    z_flat = z_flat + dec_noise * torch.randn_like(z_flat)
                raw = dec(z_flat)
                pix_target = f.reshape(B * T_, 3, H, W)
                dec_loss = oracle_decoder_loss(
                    raw, pix_target, loss_kind, K=K_PALETTE, palette=palette_t,
                    focal_gamma=focal_gamma, bg_weight=bg_weight,
                    class_weights=class_weights_t,
                )

                loss = pred_lambda * pred_loss + sigreg_lambda * sigreg_loss + dec_lambda * dec_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()) + list(dec.parameters())
                + list(action_embed.parameters()), 1.0,
            )
            opt.step(); sched.step()
            agg["loss"] += loss.item()
            agg["pred"] += pred_loss.item(); agg["sigreg"] += sigreg_loss.item()
            agg["decoder"] += dec_loss.item()
            n += 1

        payload = {
            "epoch": epoch + 1, "stage": "full",
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "predictor_state": pred.state_dict(),
            "action_embed_state": action_embed.state_dict(),
            "palette": palette,
            "cfg": cfg,
            **{k: agg[k] / max(n, 1) for k in agg},
        }
        torch.save(payload, run_dir / f"epoch_{epoch+1:03d}.pt")
        torch.save(payload, run_dir / "latest.pt")
        vol.commit()
        print(f"[{run_name}] ep {epoch+1}/{epochs} "
              f"loss={agg['loss']/n:.4f} pred={agg['pred']/n:.4f} "
              f"sigreg={agg['sigreg']/n:.4f} dec={agg['decoder']/n:.4f} "
              f"({time.time()-ep_t0:.1f}s)", flush=True)

    print(f"[{run_name}] DONE in {time.time()-train_t0:.0f}s", flush=True)


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
    gpu="A10G",
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


@app.function(
    image=image,
    gpu="A10G",
    volumes={CKPT_DIR: vol},
    timeout=60 * 60,
)
def train_manifold(
    run_name: str,
    predictor_kind: str,
    quantizer_kind: str,
    joint: bool,
    oracle_epochs: int,
    pred_epochs: int,
    pred_loss_kind: str,
    multi_step: int,
    decoder_noise: float,
    history: int = 1,
    batch: int = 256,
    lr: float = 1e-3,
    num_episodes: int = 1500,
    dim: int = 16,
    seed: int = 0,
):
    import json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    from snake import generate_oracle_dataset
    from model import (
        OracleEncoderCNN, TinyDecoder, kmeans_palette_unique,
        oracle_decoder_loss, oracle_decoder_out_channels, make_predictor, make_quantizer,
    )

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda"
    autocast_kw = dict(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else dict(device_type="cpu", enabled=False)
    K_PALETTE = 8
    out_channels = oracle_decoder_out_channels("cat-kmeans-unique", K=K_PALETTE)

    # Adjust dim for FSQ flavors that imply different latent shapes
    if quantizer_kind == "fsq16x4":
        dim = 16
    elif quantizer_kind == "fsq4x8":
        dim = 4
    elif quantizer_kind == "fsq8x5":
        dim = 8

    print(f"[{run_name}] pred={predictor_kind} quant={quantizer_kind} joint={joint} "
          f"oracle_ep={oracle_epochs} pred_ep={pred_epochs} pred_loss={pred_loss_kind} "
          f"multi_step={multi_step} dec_noise={decoder_noise} dim={dim}", flush=True)

    print(f"[{run_name}] generating {num_episodes} episodes (spatial state) ...", flush=True)
    t0 = time.time()
    frames_list, states_list, actions_list = generate_oracle_dataset(
        num_episodes, seed=seed, encoding="spatial", return_actions=True,
    )
    print(f"[{run_name}] dataset built in {time.time()-t0:.1f}s", flush=True)

    # K-means palette: keep channel-last before flatten so each row is a real (R,G,B) triplet
    sample_pix = torch.from_numpy(np.stack([f for fr in frames_list[:200] for f in fr])).float() / 255.0  # (N, 64, 64, 3)
    palette = kmeans_palette_unique(sample_pix.to(device).reshape(-1, 3), K=K_PALETTE).cpu()
    palette_t = palette.to(device)

    # Build models (predictor + action_embed early so we can save full checkpoints
    # during the oracle stage too — play windows can pop up immediately)
    enc = OracleEncoderCNN(in_channels=4, out_dim=dim).to(device)
    dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels).to(device)
    quantizer = make_quantizer(quantizer_kind, dim=dim)
    if quantizer is not None:
        quantizer = quantizer.to(device)
    predictor = make_predictor(predictor_kind, dim=dim).to(device)
    action_embed = nn.Embedding(4, dim).to(device)

    run_dir = Path(CKPT_DIR, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        run_name=run_name, mode="predictor",
        predictor_kind=predictor_kind, quantizer_kind=quantizer_kind,
        joint=joint, multi_step=multi_step, history=history,
        dim=dim, K=K_PALETTE, out_channels=out_channels,
        decoder_kind="convt", state_encoding="spatial",
        state_shape=[4, 64, 64], deep_cnn=False,
        num_episodes=num_episodes, batch=batch,
        pred_loss_kind=pred_loss_kind, decoder_noise=decoder_noise,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def save_full(stage_name, epoch_num, extra=None):
        payload = {
            "epoch": epoch_num, "stage": stage_name,
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "predictor_state": predictor.state_dict(),
            "action_embed_state": action_embed.state_dict(),
            "palette": palette,
            "cfg": cfg,
        }
        if quantizer is not None:
            payload["quantizer_state"] = quantizer.state_dict()
        if extra:
            payload.update(extra)
        ckpt_name = f"{stage_name}_epoch_{epoch_num:03d}.pt"
        torch.save(payload, run_dir / ckpt_name)
        torch.save(payload, run_dir / "latest.pt")
        vol.commit()

    # Stack everything into tensors
    all_frames = []
    all_states = []
    all_actions = []
    ep_starts = [0]
    for f, s, a in zip(frames_list, states_list, actions_list):
        n = min(len(f), len(s), len(a) + 1)
        all_frames.append(f[:n])
        all_states.append(s[:n])
        # last frame has no outgoing action; use placeholder
        all_actions.append(np.concatenate([a[:n - 1], np.array([0], dtype=np.int64)]))
        ep_starts.append(ep_starts[-1] + n)
    frames_t = torch.from_numpy(np.concatenate(all_frames, axis=0)).float().permute(0, 3, 1, 2) / 255.0
    states_t = torch.from_numpy(np.concatenate(all_states, axis=0))
    actions_t = torch.from_numpy(np.concatenate(all_actions, axis=0))
    ep_ends = np.array(ep_starts[1:]) - 1
    valid_transition = np.ones(len(states_t), dtype=bool)
    valid_transition[ep_ends] = False
    valid_idx_t = torch.from_numpy(np.where(valid_transition)[0]).long().to(device)

    print(f"[{run_name}] N frames {frames_t.size(0)}, N transitions {valid_idx_t.size(0)}", flush=True)

    # Stage 1 — train oracle encoder + decoder (with optional noise injection in latent)
    class FrameSet(Dataset):
        def __len__(self): return frames_t.size(0)
        def __getitem__(self, i): return states_t[i], frames_t[i]
    loader = DataLoader(FrameSet(), batch_size=batch, shuffle=True, num_workers=2, drop_last=True, pin_memory=True, persistent_workers=True)
    opt1 = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters())
                             + (list(quantizer.parameters()) if quantizer is not None else []), lr=lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=oracle_epochs * len(loader))
    print(f"[{run_name}] stage 1: oracle for {oracle_epochs} epochs ...", flush=True)
    for epoch in range(oracle_epochs):
        enc.train(); dec.train()
        if quantizer is not None: quantizer.train()
        ep_t0 = time.time()
        total = 0.0; n = 0
        for state, frame in loader:
            state = state.to(device, non_blocking=True)
            frame = frame.to(device, non_blocking=True)
            with torch.amp.autocast(**autocast_kw):
                z = enc(state)
                if quantizer is not None:
                    out = quantizer(z)
                    if isinstance(out, tuple):
                        z, commit_loss, _ = out
                    else:
                        z, commit_loss = out, 0.0
                else:
                    commit_loss = 0.0
                if decoder_noise > 0:
                    z = z + decoder_noise * torch.randn_like(z)
                raw = dec(z)
                pixel_loss = oracle_decoder_loss(raw, frame, "cat-kmeans-unique", K=K_PALETTE, palette=palette_t)
                loss = pixel_loss + (commit_loss if isinstance(commit_loss, torch.Tensor) else 0.0)
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            opt1.step(); sched1.step()
            total += loss.item(); n += 1
        save_full("oracle", epoch + 1, extra={"oracle_loss": total / max(n, 1)})
        print(f"[{run_name}] oracle ep {epoch+1}/{oracle_epochs} loss={total/n:.4f} ({time.time()-ep_t0:.1f}s) [saved]", flush=True)

    if not joint:
        for p in enc.parameters(): p.requires_grad_(False)
        for p in dec.parameters(): p.requires_grad_(False)
        if quantizer is not None:
            for p in quantizer.parameters(): p.requires_grad_(False)
        enc.eval(); dec.eval()
        if quantizer is not None: quantizer.eval()

    # Move to device now that DataLoader workers won't access them
    states_t = states_t.to(device)
    actions_t = actions_t.to(device)
    valid_idx_t = valid_idx_t.to(device)

    # Pre-encode all states under current oracle (stale during joint training but OK)
    print(f"[{run_name}] encoding all states ...", flush=True)
    Z_list = []
    with torch.no_grad():
        bs = 1024
        for i in range(0, states_t.size(0), bs):
            with torch.amp.autocast(**autocast_kw):
                z = enc(states_t[i:i + bs])
                if quantizer is not None:
                    out = quantizer(z)
                    z = out[0] if isinstance(out, tuple) else out
            Z_list.append(z.float())
        Z = torch.cat(Z_list, dim=0)

    pred_params = list(predictor.parameters()) + list(action_embed.parameters())
    valid_idx_t = valid_idx_t.to(device)
    if joint:
        pred_params = pred_params + list(enc.parameters()) + list(dec.parameters())
        if quantizer is not None:
            pred_params = pred_params + list(quantizer.parameters())
    opt2 = torch.optim.AdamW(pred_params, lr=lr)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=pred_epochs * (valid_idx_t.numel() // batch + 1))

    print(f"[{run_name}] stage 2: predictor for {pred_epochs} epochs ...", flush=True)
    for epoch in range(pred_epochs):
        predictor.train(); action_embed.train()
        if joint:
            enc.train(); dec.train()
            if quantizer is not None: quantizer.train()
        ep_t0 = time.time()
        perm = torch.randperm(valid_idx_t.numel(), device=device)
        total = 0.0; n = 0

        # If joint, refresh Z each epoch
        if joint and epoch > 0:
            Z_list = []
            with torch.no_grad():
                for i in range(0, states_t.size(0), 1024):
                    with torch.amp.autocast(**autocast_kw):
                        z = enc(states_t[i:i + 1024])
                        if quantizer is not None:
                            out = quantizer(z)
                            z = out[0] if isinstance(out, tuple) else out
                    Z_list.append(z.float())
                Z = torch.cat(Z_list, dim=0)

        for s in range(0, valid_idx_t.numel() - batch, batch):
            idx = valid_idx_t[perm[s:s + batch]]
            z_t = Z[idx]
            a_t = action_embed(actions_t[idx])
            target = Z[idx + 1].detach()

            with torch.amp.autocast(**autocast_kw):
                if pred_loss_kind == "rollout":
                    # Multi-step free-rollout: feed predictor's own outputs back
                    z_hat = z_t
                    loss = z_t.new_zeros(())
                    valid_mask = torch.ones(idx.size(0), dtype=torch.bool, device=device)
                    for k in range(multi_step):
                        a_k = action_embed(actions_t[(idx + k).clamp(max=actions_t.numel() - 1)])
                        if predictor_kind == "transformer":
                            z_hat = predictor.step(z_hat.unsqueeze(1), a_k.unsqueeze(1))
                        elif predictor_kind == "rnn":
                            z_hat, _ = predictor(z_hat, a_k)
                        else:
                            z_hat = predictor(z_hat, a_k)
                        # mask out indices that have crossed an episode boundary
                        next_idx = (idx + k + 1).clamp(max=Z.size(0) - 1)
                        in_ep = (idx + k + 1 < Z.size(0)) & (~torch.isin(idx + k, torch.from_numpy(ep_ends).long().to(device)))
                        valid_mask = valid_mask & in_ep
                        target_k = Z[next_idx].detach()
                        if valid_mask.any():
                            loss = loss + F.mse_loss(z_hat[valid_mask], target_k[valid_mask])
                    loss = loss / multi_step
                elif multi_step > 1:
                    # multi-step teacher-forcing: predict z_{t+1}, z_{t+2}, ..., z_{t+multi_step}
                    z_hat = z_t
                    loss = z_t.new_zeros(())
                    for k in range(multi_step):
                        a_k = action_embed(actions_t[(idx + k).clamp(max=actions_t.numel() - 1)])
                        if predictor_kind == "transformer":
                            z_hat = predictor.step(z_hat.unsqueeze(1), a_k.unsqueeze(1))
                        elif predictor_kind == "rnn":
                            z_hat, _ = predictor(z_hat, a_k)
                        else:
                            z_hat = predictor(z_hat, a_k)
                        next_idx = (idx + k + 1).clamp(max=Z.size(0) - 1)
                        in_ep = (idx + k + 1 < Z.size(0))
                        target_k = Z[next_idx].detach()
                        if in_ep.any():
                            loss = loss + F.mse_loss(z_hat[in_ep], target_k[in_ep])
                    loss = loss / multi_step
                else:
                    if predictor_kind == "transformer":
                        pred = predictor.step(z_t.unsqueeze(1), a_t.unsqueeze(1))
                    elif predictor_kind == "rnn":
                        pred, _ = predictor(z_t, a_t)
                    else:
                        pred = predictor(z_t, a_t)
                    loss = F.mse_loss(pred, target)

                if joint:
                    # ALSO add an oracle pixel loss on the current batch's frames
                    with torch.amp.autocast(**autocast_kw):
                        z_recon = enc(states_t[idx])
                        if quantizer is not None:
                            out = quantizer(z_recon)
                            z_recon = out[0] if isinstance(out, tuple) else out
                        raw_recon = dec(z_recon)
                        pixel_loss = oracle_decoder_loss(raw_recon, frames_t[idx].to(device),
                                                          "cat-kmeans-unique", K=K_PALETTE, palette=palette_t)
                    loss = loss + pixel_loss

            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step(); sched2.step()
            total += loss.item(); n += 1

        save_full("pred", epoch + 1, extra={"predictor_loss": total / max(n, 1)})
        print(f"[{run_name}] predictor ep {epoch+1}/{pred_epochs} loss={total/n:.5f} ({time.time()-ep_t0:.1f}s) [saved]", flush=True)

    print(f"[{run_name}] DONE", flush=True)


@app.local_entrypoint()
def main():
    print(f"Spawning {len(ARCH_RUNS)} parallel A10G jobs (precision: finer latents + deeper) ...")
    for run_name, arch_kind in ARCH_RUNS:
        h = train_arch_jepa.spawn(
            run_name=run_name, arch_kind=arch_kind,
            rollout_dec=True, pred_lambda=0.0,
        )
        print(f"  spawned {run_name} ({arch_kind}): {h.object_id}")
    print("All jobs spawned.")
    return
    # legacy entrypoint below
    print(f"Spawning {len(PREDICTOR_RUNS)} parallel H100 jobs (frozen-oracle predictor ablation) ...")
    for run_name, kind, multi_step, history in PREDICTOR_RUNS:
        h = train_predictor.spawn(
            run_name=run_name, predictor_kind=kind,
            multi_step=multi_step, history=history,
        )
        print(f"  spawned {run_name} (kind={kind}): {h.object_id}")
    print("All jobs spawned.")
    return
    # ---- legacy: pinpoint ablation entrypoint ----
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
