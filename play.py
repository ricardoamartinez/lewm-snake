"""Live imagined Snake — play through the trained LeWM patch-token world model.

Pipeline at every keystroke:
  1) The user's chosen action is embedded.
  2) A sliding window of the last `history` patch grids and actions is fed to
     the predictor. The predictor outputs a fresh (P, D) patch grid per timestep;
     we take the last one.
  3) That predicted patch grid is decoded into a 64x64 RGB frame.
  4) The grid is appended to the history buffer for the next step.

This keeps the snake's "world" entirely inside the model's imagined latent
space — no real Snake env in the loop after seeding.

Controls:
  arrows / WASD : move
  R             : pull latest checkpoint from the Modal volume + reset
  N             : reset imagined game with the current model
  Q / Esc       : quit
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from snake import Snake
from model import (
    LeWMSnake, render_decoder_output,
    OracleEncoder, TinyDecoder, SharpDecoder, CrossAttnDecoder, PerPixelDecoder,
    SNAKE_COLORS, render_oracle_output as model_render_oracle,
)


CKPT_LOCAL_ROOT = Path("./_ckpts")
CKPT_LOCAL_ROOT.mkdir(exist_ok=True)
VOLUME = "lewm-snake-ckpts"
CKPT_LOCAL = CKPT_LOCAL_ROOT  # default; overridden by --run


def modal_bin():
    candidates = [
        os.path.expanduser("~/Library/Python/3.9/bin/modal"),
        os.path.expanduser("~/Library/Python/3.11/bin/modal"),
        "modal",
    ]
    for c in candidates:
        try:
            r = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def pull_checkpoint(name="latest.pt", run=None):
    mb = modal_bin()
    local_dir = CKPT_LOCAL_ROOT if run is None else CKPT_LOCAL_ROOT / run
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / name
    remote_name = name if run is None else f"{run}/{name}"
    if mb is None:
        print("[play] modal CLI not found; using cached checkpoint")
        return target
    if target.exists():
        target.unlink()
    print(f"[play] pulling {remote_name} from volume {VOLUME} ...")
    r = subprocess.run(
        [mb, "volume", "get", VOLUME, remote_name, str(target), "--force"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[play] modal volume get failed: {r.stderr.strip()}")
    return target


def load_model(ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob.get("cfg", {})
    model = LeWMSnake(
        dim=cfg.get("dim", 128),
        history=cfg.get("history", 4),
        n_actions=4,
        decoder_kind=cfg.get("decoder_kind", "convtranspose"),
        out_channels=cfg.get("out_channels", 3),
        latent_nll=cfg.get("latent_nll", False),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    model.loss_kind = cfg.get("loss_kind", "mse")
    return model, blob.get("epoch", "?"), blob.get("step", "?")


def _build_oracle_decoder(decoder_kind, dim, out_channels):
    from train_modal import _make_pixshuf, _make_crossattn, _make_perpixel
    if decoder_kind == "convt":
        return TinyDecoder(dim=dim, ch=128, out_channels=out_channels)
    if decoder_kind == "pixshuf":
        return _make_pixshuf(dim=dim, out_channels=out_channels)
    if decoder_kind == "crossattn":
        return _make_crossattn(dim=dim, out_channels=out_channels)
    if decoder_kind == "perpixel":
        return _make_perpixel(dim=dim, out_channels=out_channels)
    raise ValueError(decoder_kind)


def load_oracle(ckpt_path):
    """Load oracle-mode checkpoint. Returns (encoder, decoder, cfg, epoch, palette)."""
    from model import OracleEncoderCNN
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    if cfg.get("state_encoding") == "spatial":
        in_ch = cfg.get("state_shape", [4, 64, 64])[0]
        enc = OracleEncoderCNN(in_channels=in_ch, out_dim=cfg["dim"],
                                deep=cfg.get("deep_cnn", False))
    else:
        enc = OracleEncoder(in_dim=cfg.get("state_dim", 99), out_dim=cfg["dim"])
    enc.load_state_dict(blob["encoder_state"])
    enc.eval()
    dec = _build_oracle_decoder(cfg["decoder_kind"], cfg["dim"], cfg["out_channels"])
    dec.load_state_dict(blob["decoder_state"])
    dec.eval()
    palette = None
    if "palette" in blob:
        palette = blob["palette"]
        if not isinstance(palette, torch.Tensor):
            palette = torch.tensor(palette)
    return enc, dec, cfg, blob.get("epoch", "?"), palette


def render_oracle_output(raw, loss_kind, K=None, palette=None):
    return model_render_oracle(raw, loss_kind, K=K, palette=palette)


def run_arch_jepa_play(args, ckpt_path):
    """JEPA rollout: encode SEED frame once, then predictor evolves latent under
    user keypresses. Decoder renders each rolled latent. Real frames never touch
    the encoder after t=0.
    Left = real Snake env, right = imagined rollout."""
    import pygame
    import torch.nn as nn
    from snake import Snake
    from model import (
        OracleEncoderCNN, TinyDecoder, SpatialEncoder, SpatialDecoder, SpatialPixShufDecoder,
        make_predictor, render_oracle_output,
    )

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    arch = cfg["arch_kind"]; dim = cfg["dim"]; K = cfg["K"]
    out_channels = cfg["out_channels"]
    is_spatial = cfg["is_spatial"]
    pred_kind = cfg["predictor_kind"]
    palette = blob["palette"]
    if not isinstance(palette, torch.Tensor): palette = torch.tensor(palette)

    if arch == "flat":
        enc = OracleEncoderCNN(in_channels=3, out_dim=dim)
        dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels)
    else:
        parts = arch.split("-")
        lat_size = int(parts[1])
        deep = "deep" in parts
        bigch = "bigCh" in parts
        enc_refine = 3 if deep else 1
        dec_refine = 3 if deep else 1
        enc_base = 64 if bigch else 32
        dec_base = 256 if bigch else 128
        dec_kind = cfg.get("dec_kind", "convt")
        enc = SpatialEncoder(in_channels=3, dim=dim, lat_size=lat_size,
                             base_ch=enc_base, refine_blocks=enc_refine)
        if dec_kind == "pixshuf":
            dec = SpatialPixShufDecoder(dim=dim, out_channels=out_channels, lat_size=lat_size,
                                         base_ch=dec_base, refine_blocks=dec_refine)
        else:
            dec = SpatialDecoder(dim=dim, out_channels=out_channels, lat_size=lat_size,
                                 base_ch=dec_base, refine_blocks=dec_refine)
    enc.load_state_dict(blob["encoder_state"]); enc.eval()
    dec.load_state_dict(blob["decoder_state"]); dec.eval()
    pred = make_predictor(pred_kind, dim=dim)
    pred.load_state_dict(blob["predictor_state"]); pred.eval()
    act_embed = nn.Embedding(4, dim)
    act_embed.load_state_dict(blob["action_embed_state"]); act_embed.eval()

    enc = enc.to(args.device); dec = dec.to(args.device)
    pred = pred.to(args.device); act_embed = act_embed.to(args.device)
    ep = blob.get("epoch", "?")

    pygame.init()
    cell = 64 * args.scale
    screen = pygame.display.set_mode((cell * 2 + 8, cell + 40))
    title = f"LeWM-Snake JEPA [{args.run}] ep={ep} arch={arch}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    env = Snake(seed=int(rng.integers(1 << 30)))

    def reset_imag():
        with torch.no_grad():
            seed_frame = env.render()
            f = torch.from_numpy(seed_frame).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
            z0 = enc(f)
        return z0

    z = reset_imag()
    current_action = 3
    step_idx = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    new_ckpt = pull_checkpoint("latest.pt", run=args.run)
                    if new_ckpt.exists():
                        blob2 = torch.load(new_ckpt, map_location="cpu", weights_only=False)
                        enc.load_state_dict(blob2["encoder_state"])
                        dec.load_state_dict(blob2["decoder_state"])
                        pred.load_state_dict(blob2["predictor_state"])
                        act_embed.load_state_dict(blob2["action_embed_state"])
                        ep = blob2.get("epoch", "?")
                        pygame.display.set_caption(f"LeWM-Snake JEPA [{args.run}] ep={ep} arch={arch}")
                        env = Snake(seed=int(rng.integers(1 << 30)))
                        z = reset_imag(); step_idx = 0
                elif event.key == pygame.K_n:
                    env = Snake(seed=int(rng.integers(1 << 30)))
                    z = reset_imag(); step_idx = 0
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        _, done = env.step(current_action)
        if done:
            env = Snake(seed=int(rng.integers(1 << 30)))
            z = reset_imag(); step_idx = 0
            continue
        real_frame = env.render()

        with torch.no_grad():
            a_t = act_embed(torch.tensor([current_action], device=args.device))  # (1, dim)
            z = pred(z, a_t)                                                       # roll latent
            raw = dec(z)
            recon = render_oracle_output(raw, "cat-kmeans-unique", K=K, palette=palette).clamp(0, 1)[0]

        screen.fill((0, 0, 0))
        real_surf = pygame.surfarray.make_surface(real_frame.swapaxes(0, 1))
        real_surf = pygame.transform.scale(real_surf, (cell, cell))
        screen.blit(real_surf, (0, 0))
        rec_arr = (recon.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        rec_surf = pygame.surfarray.make_surface(rec_arr.swapaxes(0, 1))
        rec_surf = pygame.transform.scale(rec_surf, (cell, cell))
        screen.blit(rec_surf, (cell + 8, 0))
        info = font.render(
            f"ep={ep} arch={arch} step={step_idx} act={NAMES[current_action]} L=REAL R=IMAGINED(JEPA)",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, cell + 12))
        pygame.display.flip()
        clock.tick(args.fps)
        step_idx += 1
    pygame.quit()


def run_arch_ae_play(args, ckpt_path):
    """Pure-AE replay: real frame -> encoder -> decoder -> reconstruction.
    Left = real Snake, right = AE reconstruction. No predictor.
    Tests whether the architecture can render snake/food correctly per-frame.
    """
    import pygame
    from snake import Snake
    from model import (
        OracleEncoderCNN, TinyDecoder, UNetAE, SpatialEncoder, SpatialDecoder,
        render_oracle_output,
    )

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    arch = cfg["arch_kind"]
    dim = cfg["dim"]; K = cfg["K"]
    out_channels = cfg["out_channels"]
    palette = blob["palette"]
    if not isinstance(palette, torch.Tensor): palette = torch.tensor(palette)

    is_unet = arch.startswith("unet")
    if arch == "flat":
        enc = OracleEncoderCNN(in_channels=3, out_dim=dim)
        dec = TinyDecoder(dim=dim, ch=128, out_channels=out_channels)
    elif arch == "spatial-16":
        enc = SpatialEncoder(in_channels=3, dim=dim, lat_size=16)
        dec = SpatialDecoder(dim=dim, out_channels=out_channels, lat_size=16)
    elif arch == "spatial-8":
        enc = SpatialEncoder(in_channels=3, dim=dim, lat_size=8)
        dec = SpatialDecoder(dim=dim, out_channels=out_channels, lat_size=8)
    elif arch == "unet-tiny":
        unet = UNetAE(in_channels=3, out_channels=out_channels, base_ch=16)
        enc = unet; dec = unet
    elif arch == "unet-base":
        unet = UNetAE(in_channels=3, out_channels=out_channels, base_ch=32)
        enc = unet; dec = unet
    else:
        raise ValueError(arch)

    if is_unet:
        enc.load_state_dict(blob["unet_state"])
    else:
        enc.load_state_dict(blob["encoder_state"])
        dec.load_state_dict(blob["decoder_state"])
    enc.eval()
    if not is_unet:
        dec.eval()
    enc = enc.to(args.device)
    if not is_unet:
        dec = dec.to(args.device)
    ep = blob.get("epoch", "?")

    pygame.init()
    cell = 64 * args.scale
    screen = pygame.display.set_mode((cell * 2 + 8, cell + 40))
    title = f"LeWM-Snake AE [{args.run}] ep={ep} arch={arch}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    env = Snake(seed=int(rng.integers(1 << 30)))
    current_action = 3
    step_idx = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    new_ckpt = pull_checkpoint("latest.pt", run=args.run)
                    if new_ckpt.exists():
                        blob2 = torch.load(new_ckpt, map_location="cpu", weights_only=False)
                        if is_unet:
                            enc.load_state_dict(blob2["unet_state"])
                        else:
                            enc.load_state_dict(blob2["encoder_state"])
                            dec.load_state_dict(blob2["decoder_state"])
                        ep = blob2.get("epoch", "?")
                        pygame.display.set_caption(f"LeWM-Snake AE [{args.run}] ep={ep} arch={arch}")
                elif event.key == pygame.K_n:
                    env = Snake(seed=int(rng.integers(1 << 30)))
                    step_idx = 0
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        _, done = env.step(current_action)
        if done:
            env = Snake(seed=int(rng.integers(1 << 30)))
            step_idx = 0
            continue
        real_frame = env.render()
        f = torch.from_numpy(real_frame).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
        with torch.no_grad():
            if is_unet:
                raw = enc(f)
            else:
                z = enc(f)
                raw = dec(z)
            recon = render_oracle_output(raw, "cat-kmeans-unique", K=K, palette=palette).clamp(0, 1)[0]

        screen.fill((0, 0, 0))
        real_surf = pygame.surfarray.make_surface(real_frame.swapaxes(0, 1))
        real_surf = pygame.transform.scale(real_surf, (cell, cell))
        screen.blit(real_surf, (0, 0))
        rec_arr = (recon.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        rec_surf = pygame.surfarray.make_surface(rec_arr.swapaxes(0, 1))
        rec_surf = pygame.transform.scale(rec_surf, (cell, cell))
        screen.blit(rec_surf, (cell + 8, 0))
        info = font.render(
            f"ep={ep}  arch={arch}  step={step_idx}  L=REAL  R=AE-RECON",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, cell + 12))
        pygame.display.flip()
        clock.tick(args.fps)
        step_idx += 1
    pygame.quit()


def run_full_play(args, ckpt_path):
    """Full-system replay: pixel encoder reads real frames once at reset to seed,
    then predictor + decoder roll out forever under user keypresses.
    Left = real Snake env, right = imagined."""
    import pygame
    import torch.nn as nn
    from snake import Snake
    from model import OracleEncoderCNN, TinyDecoder, make_predictor, render_oracle_output

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    dim = cfg["dim"]; K = cfg["K"]
    enc = OracleEncoderCNN(in_channels=3, out_dim=dim)
    enc.load_state_dict(blob["encoder_state"]); enc.eval()
    dec = TinyDecoder(dim=dim, ch=128, out_channels=cfg["out_channels"])
    dec.load_state_dict(blob["decoder_state"]); dec.eval()
    pred = make_predictor(cfg["predictor_kind"], dim=dim)
    pred.load_state_dict(blob["predictor_state"]); pred.eval()
    act_embed = nn.Embedding(4, dim)
    act_embed.load_state_dict(blob["action_embed_state"]); act_embed.eval()
    palette = blob["palette"]
    if not isinstance(palette, torch.Tensor): palette = torch.tensor(palette)

    enc = enc.to(args.device); dec = dec.to(args.device)
    pred = pred.to(args.device); act_embed = act_embed.to(args.device)
    ep = blob.get("epoch", "?")

    pygame.init()
    cell = 64 * args.scale
    screen = pygame.display.set_mode((cell * 2 + 8, cell + 40))
    title = f"LeWM-Snake FULL [{args.run}] ep={ep} pred={cfg['predictor_kind']} dn={cfg.get('dec_noise', 0)}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    env = Snake(seed=int(rng.integers(1 << 30)))

    def reset_imag():
        with torch.no_grad():
            seed_frame = env.render()                                       # (64,64,3) uint8
            f = torch.from_numpy(seed_frame).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
            z0 = enc(f)                                                     # (1, dim)
        return z0

    z = reset_imag()
    h_state = None
    current_action = 3
    step_idx = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    new_ckpt = pull_checkpoint("latest.pt", run=args.run)
                    if new_ckpt.exists():
                        blob2 = torch.load(new_ckpt, map_location="cpu", weights_only=False)
                        enc.load_state_dict(blob2["encoder_state"])
                        dec.load_state_dict(blob2["decoder_state"])
                        pred.load_state_dict(blob2["predictor_state"])
                        act_embed.load_state_dict(blob2["action_embed_state"])
                        ep = blob2.get("epoch", "?")
                        pygame.display.set_caption(
                            f"LeWM-Snake FULL [{args.run}] ep={ep} pred={cfg['predictor_kind']} dn={cfg.get('dec_noise',0)}"
                        )
                        env = Snake(seed=int(rng.integers(1 << 30)))
                        z = reset_imag(); h_state = None; step_idx = 0
                        print(f"[play] {args.run}: reloaded epoch={ep}")
                elif event.key == pygame.K_n:
                    env = Snake(seed=int(rng.integers(1 << 30)))
                    z = reset_imag(); h_state = None; step_idx = 0
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        _, done = env.step(current_action)
        if done:
            env = Snake(seed=int(rng.integers(1 << 30)))
            z = reset_imag(); h_state = None; step_idx = 0
            continue
        real_frame = env.render()

        with torch.no_grad():
            a_t = act_embed(torch.tensor([current_action], device=args.device))
            if cfg["predictor_kind"] == "transformer":
                z_next = pred.step(z.unsqueeze(1), a_t.unsqueeze(1))
            elif cfg["predictor_kind"] == "rnn":
                z_next, h_state = pred(z, a_t, h_state)
            else:
                z_next = pred(z, a_t)
            z = z_next
            raw = dec(z)
            recon = render_oracle_output(raw, "cat-kmeans-unique", K=K, palette=palette).clamp(0, 1)[0]

        screen.fill((0, 0, 0))
        real_surf = pygame.surfarray.make_surface(real_frame.swapaxes(0, 1))
        real_surf = pygame.transform.scale(real_surf, (cell, cell))
        screen.blit(real_surf, (0, 0))
        rec_arr = (recon.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        rec_surf = pygame.surfarray.make_surface(rec_arr.swapaxes(0, 1))
        rec_surf = pygame.transform.scale(rec_surf, (cell, cell))
        screen.blit(rec_surf, (cell + 8, 0))
        info = font.render(
            f"ep={ep}  step={step_idx}  action={NAMES[current_action]}  L=REAL  R=IMAGINED",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, cell + 12))
        pygame.display.flip()
        clock.tick(args.fps)
        step_idx += 1
    pygame.quit()


def run_predictor_play(args, ckpt_path):
    """Side-by-side interactive: left = real Snake env, right = imagined rollout
    via frozen oracle decoder + trainable predictor. Both driven by the same
    keypress."""
    import pygame
    import torch.nn as nn
    from snake import Snake, state_features_v2
    from model import OracleEncoderCNN, TinyDecoder, make_predictor, render_oracle_output

    from model import make_quantizer
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    dim = cfg["dim"]
    K = cfg["K"]
    enc = OracleEncoderCNN(in_channels=4, out_dim=dim)
    enc.load_state_dict(blob["encoder_state"]); enc.eval()
    dec = TinyDecoder(dim=dim, ch=128, out_channels=cfg["out_channels"])
    dec.load_state_dict(blob["decoder_state"]); dec.eval()
    pred = make_predictor(cfg["predictor_kind"], dim=dim)
    pred.load_state_dict(blob["predictor_state"]); pred.eval()
    act_embed = nn.Embedding(4, dim)
    act_embed.load_state_dict(blob["action_embed_state"]); act_embed.eval()
    quantizer = make_quantizer(cfg.get("quantizer_kind", "none"), dim=dim)
    if quantizer is not None and "quantizer_state" in blob:
        quantizer.load_state_dict(blob["quantizer_state"]); quantizer.eval()
        quantizer = quantizer.to(args.device)
    palette = blob["palette"]
    if not isinstance(palette, torch.Tensor):
        palette = torch.tensor(palette)

    enc = enc.to(args.device); dec = dec.to(args.device); pred = pred.to(args.device); act_embed = act_embed.to(args.device)
    history_size = cfg.get("history", 1)
    ep = blob.get("epoch", "?")

    pygame.init()
    cell = 64 * args.scale
    w = cell * 2 + 8
    h = cell + 40
    screen = pygame.display.set_mode((w, h))
    title = f"LeWM-Snake [{args.run}] ep={ep} pred={cfg['predictor_kind']}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    env = Snake(seed=int(rng.integers(1 << 30)))

    def reset_imag():
        s = state_features_v2(env, encoding="spatial")
        with torch.no_grad():
            z0 = enc(torch.from_numpy(s).unsqueeze(0).to(args.device))         # (1, dim)
            if quantizer is not None:
                out = quantizer(z0)
                z0 = out[0] if isinstance(out, tuple) else out
        z_hist = z0.unsqueeze(1)                                                # (1, 1, dim)
        a_hist = torch.zeros(1, 1, dim, device=args.device)
        h_state = None
        return z_hist, a_hist, h_state

    z_hist, a_hist, h_state = reset_imag()
    current_action = 3
    step_idx = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    new_ckpt = pull_checkpoint("latest.pt", run=args.run)
                    if new_ckpt.exists():
                        # reload
                        blob2 = torch.load(new_ckpt, map_location="cpu", weights_only=False)
                        enc.load_state_dict(blob2["encoder_state"])
                        dec.load_state_dict(blob2["decoder_state"])
                        pred.load_state_dict(blob2["predictor_state"])
                        act_embed.load_state_dict(blob2["action_embed_state"])
                        ep = blob2.get("epoch", "?")
                        pygame.display.set_caption(f"LeWM-Snake [{args.run}] ep={ep} pred={cfg['predictor_kind']}")
                        env = Snake(seed=int(rng.integers(1 << 30)))
                        z_hist, a_hist, h_state = reset_imag()
                        step_idx = 0
                        print(f"[play] {args.run}: reloaded epoch={ep}")
                elif event.key == pygame.K_n:
                    env = Snake(seed=int(rng.integers(1 << 30)))
                    z_hist, a_hist, h_state = reset_imag()
                    step_idx = 0
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        # advance both sides under current_action
        _, done = env.step(current_action)
        if done:
            env = Snake(seed=int(rng.integers(1 << 30)))
            z_hist, a_hist, h_state = reset_imag()
            step_idx = 0
            continue
        real_frame = env.render()

        with torch.no_grad():
            a_t = act_embed(torch.tensor([current_action], device=args.device))   # (1, dim)
            if cfg["predictor_kind"] == "transformer":
                # build sliding window of length history_size
                a_hist = torch.cat([a_hist, a_t.unsqueeze(1)], dim=1)
                if a_hist.size(1) > history_size:
                    a_hist = a_hist[:, -history_size:]
                z_in = z_hist[:, -history_size:] if z_hist.size(1) >= history_size else z_hist
                z_next = pred.step(z_in, a_hist[:, -z_in.size(1):])              # (1, dim)
            elif cfg["predictor_kind"] == "rnn":
                z_next, h_state = pred(z_hist[:, -1], a_t, h_state)
            else:
                z_next = pred(z_hist[:, -1], a_t)
            if quantizer is not None:
                out = quantizer(z_next)
                z_next = out[0] if isinstance(out, tuple) else out
            z_hist = torch.cat([z_hist, z_next.unsqueeze(1)], dim=1)
            raw = dec(z_next)
            recon = render_oracle_output(raw, "cat-kmeans-unique", K=K, palette=palette).clamp(0, 1)[0]

        # blit
        screen.fill((0, 0, 0))
        real_surf = pygame.surfarray.make_surface(real_frame.swapaxes(0, 1))
        real_surf = pygame.transform.scale(real_surf, (cell, cell))
        screen.blit(real_surf, (0, 0))
        rec_arr = (recon.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        rec_surf = pygame.surfarray.make_surface(rec_arr.swapaxes(0, 1))
        rec_surf = pygame.transform.scale(rec_surf, (cell, cell))
        screen.blit(rec_surf, (cell + 8, 0))
        info = font.render(
            f"ep={ep}  step={step_idx}  action={NAMES[current_action]}  L=REAL  R=IMAGINED",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, cell + 12))
        pygame.display.flip()
        clock.tick(args.fps)
        step_idx += 1

    pygame.quit()


def run_oracle_replay(args, ckpt_path):
    import pygame
    enc, dec, cfg, ep, palette = load_oracle(ckpt_path)
    enc = enc.to(args.device); dec = dec.to(args.device)
    K = cfg.get("K")

    from snake import Snake, heuristic_action, state_features, state_features_v2
    pygame.init()
    cell = 64 * args.scale
    w = cell * 2 + 8
    h = cell + 40
    screen = pygame.display.set_mode((w, h))
    title = f"LeWM-Snake [{args.run}] ep={ep} dec={cfg['decoder_kind']} loss={cfg['loss_kind']}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()

    rng = np.random.default_rng(int(time.time()) & 0xFFFF)

    def fresh_episode():
        env = Snake(seed=int(rng.integers(1 << 30)))
        return env

    env = fresh_episode()
    running = True
    step_idx = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    new_ckpt = pull_checkpoint("latest.pt", run=args.run)
                    if new_ckpt.exists():
                        enc, dec, cfg, ep, palette = load_oracle(new_ckpt)
                        enc = enc.to(args.device); dec = dec.to(args.device)
                        K = cfg.get("K")
                        title = f"LeWM-Snake [{args.run}] ep={ep} dec={cfg['decoder_kind']} loss={cfg['loss_kind']}"
                        pygame.display.set_caption(title)
                        print(f"[play] {args.run}: reloaded epoch={ep}")
                elif event.key == pygame.K_n:
                    env = fresh_episode(); step_idx = 0

        # advance the env one step under heuristic + occasional random
        if rng.random() < 0.2:
            a = int(rng.integers(4))
        else:
            a = heuristic_action(env)
        _, done = env.step(a)
        if done:
            env = fresh_episode(); step_idx = 0; continue

        real_frame = env.render()
        encoding = cfg.get("state_encoding", "baseline")
        feats = state_features_v2(env, encoding=encoding)
        with torch.no_grad():
            z = enc(torch.from_numpy(feats).unsqueeze(0).to(args.device))
            raw = dec(z)
            recon = render_oracle_output(raw, cfg["loss_kind"], K=K, palette=palette).clamp(0, 1)[0]

        # blit real (left) and rendered (right)
        screen.fill((0, 0, 0))
        real_arr = real_frame  # already uint8 64x64x3
        real_surf = pygame.surfarray.make_surface(real_arr.swapaxes(0, 1))
        real_surf = pygame.transform.scale(real_surf, (cell, cell))
        screen.blit(real_surf, (0, 0))
        rec_arr = (recon.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        rec_surf = pygame.surfarray.make_surface(rec_arr.swapaxes(0, 1))
        rec_surf = pygame.transform.scale(rec_surf, (cell, cell))
        screen.blit(rec_surf, (cell + 8, 0))
        info = font.render(
            f"ep={ep}  step={step_idx}  left=REAL right=RENDERED  R=reload  N=new  Q=quit",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, cell + 12))
        pygame.display.flip()
        clock.tick(args.fps)
        step_idx += 1
    pygame.quit()


def seed_from_real(history):
    """Seed the imagined game with `history` real Snake frames produced by
    repeatedly going right. Returns (pixels, actions) where actions[i] is the
    action that produced the transition from frames[i] to frames[i+1]; the last
    entry is a placeholder (overwritten on the user's first input)."""
    env = Snake(seed=int(time.time()) & 0xFFFF)
    frames = [env.render()]
    actions = []
    for _ in range(history - 1):
        a = env.dir
        f, _ = env.step(a)
        frames.append(f)
        actions.append(a)
    actions.append(env.dir)  # placeholder for last slot
    seed = np.stack(frames)
    return (
        torch.from_numpy(seed).float().permute(0, 3, 1, 2) / 255.0,
        torch.tensor(actions, dtype=torch.long),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--scale", type=int, default=8, help="display upscaling factor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--run", default=None, help="experiment subdir under volume root")
    args = ap.parse_args()

    name = f"epoch_{args.epoch:03d}.pt" if args.epoch else "latest.pt"
    if args.no_pull:
        ckpt = (CKPT_LOCAL_ROOT / args.run / name) if args.run else (CKPT_LOCAL_ROOT / name)
    else:
        ckpt = pull_checkpoint(name, run=args.run)
    if not ckpt.exists():
        print(f"[play] checkpoint not found at {ckpt}")
        sys.exit(1)

    # Detect mode
    blob_peek = torch.load(ckpt, map_location="cpu", weights_only=False)
    mode = blob_peek.get("cfg", {}).get("mode")
    if mode == "oracle":
        print(f"[play] {args.run or '?'}: oracle replay mode")
        run_oracle_replay(args, ckpt)
        return
    if mode == "predictor":
        print(f"[play] {args.run or '?'}: predictor (frozen-oracle world model) mode")
        run_predictor_play(args, ckpt)
        return
    if mode == "full":
        print(f"[play] {args.run or '?'}: full-system (pixel encoder + predictor + decoder) mode")
        run_full_play(args, ckpt)
        return
    if mode == "arch_jepa":
        print(f"[play] {args.run or '?'}: arch-JEPA mode (seed once, predictor rolls latent)")
        run_arch_jepa_play(args, ckpt)
        return
    if mode == "arch_ae":
        print(f"[play] {args.run or '?'}: arch-AE mode (no predictor — encode/decode every real frame)")
        run_arch_ae_play(args, ckpt)
        return

    model, ep, step = load_model(ckpt)
    model = model.to(args.device)
    H_hist = model.history
    label = f"[{args.run}] " if args.run else ""
    print(f"[play] {label}loaded {name}  (epoch={ep}  step={step})  history={H_hist}  loss={model.loss_kind}")

    try:
        import pygame
    except ImportError:
        print("[play] pygame not installed. Run: python3 -m pip install --user pygame")
        sys.exit(1)

    pygame.init()
    W = HH = 64 * args.scale
    screen = pygame.display.set_mode((W, HH + 40))
    title = f"LeWM-Snake [{args.run}] {name} ({model.loss_kind})" if args.run else f"LeWM-Snake (imagined) — {name}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0,
        pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2,
        pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def reset_imagined():
        seed_pixels, seed_actions = seed_from_real(H_hist)
        with torch.no_grad():
            seed_d = seed_pixels.to(args.device).unsqueeze(0)
            emb = model.encode(seed_d)                                  # (1, H, D)
            act_buf = model.action_embed(seed_actions.to(args.device).unsqueeze(0))  # (1, H, D)
        return seed_pixels, emb, act_buf, seed_pixels[-1]

    seed_pixels, emb, act_buf, last_frame = reset_imagined()
    current_action = 3

    def render_frame(frame_tensor):
        arr = (frame_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (W, HH))
        screen.blit(surf, (0, 0))

    render_frame(last_frame)
    pygame.display.flip()

    clock = pygame.time.Clock()
    step_count = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    ckpt = pull_checkpoint("latest.pt", run=args.run)
                    if ckpt.exists():
                        model_, ep, step = load_model(ckpt)
                        model = model_.to(args.device)
                        H_hist = model.history
                        seed_pixels, emb, act_buf, last_frame = reset_imagined()
                        step_count = 0
                        cap = f"LeWM-Snake [{args.run}] epoch={ep} ({model.loss_kind})" if args.run \
                              else f"LeWM-Snake (imagined) — latest.pt epoch={ep}"
                        pygame.display.set_caption(cap)
                        print(f"[play] {label}reloaded latest.pt (epoch={ep})")
                elif event.key == pygame.K_n:
                    seed_pixels, emb, act_buf, last_frame = reset_imagined()
                    step_count = 0
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        # Roll forward one step in imagination.
        # Replace the last action slot with the user's chosen action so the
        # predictor knows what action produced the next frame.
        with torch.no_grad():
            a = torch.tensor([[current_action]], device=args.device, dtype=torch.long)
            new_act = model.action_embed(a)                             # (1, 1, D)
            act_buf = torch.cat([act_buf[:, 1:], new_act], dim=1)
            pred = model.predict(emb[:, -H_hist:], act_buf)             # (1, H, D)
            next_z = pred[:, -1:]                                       # (1, 1, D)
            emb = torch.cat([emb, next_z], dim=1)
            raw = model.decoder(next_z[:, 0])                           # (1, C, 64, 64)
            frame = render_decoder_output(raw, model.loss_kind).clamp(0, 1)[0]

        last_frame = frame
        step_count += 1
        screen.fill((0, 0, 0))
        render_frame(last_frame)
        info = font.render(
            f"ep={ep}  step={step_count}  action={NAMES[current_action]}  R=reload  N=new",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, HH + 12))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
