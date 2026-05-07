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
        ConvPredictor, StochasticConvPredictor, GlobalConvPredictor, AttnPredictor,
        VariationalConvPredictor,
        SpatialVQ, SpatialFSQ, make_predictor, render_oracle_output,
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
    pb = cfg.get("pred_blocks", 2)
    phid = cfg.get("pred_hidden", 64)
    # Fallback for old checkpoints that don't have pred_blocks/pred_hidden saved
    rn = cfg.get("run_name", "")
    if "deep-stoch" in rn:
        pb, phid = 6, 128
    elif "attn-deep" in rn:
        pb, phid = 4, 128
    if pred_kind == "stoch-conv":
        pred = StochasticConvPredictor(dim=dim, hidden=phid, n_blocks=pb)
    elif pred_kind == "global-conv":
        pred = GlobalConvPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=False)
    elif pred_kind == "global-stoch-conv":
        pred = GlobalConvPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=True)
    elif pred_kind == "attn":
        pred = AttnPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=False)
    elif pred_kind == "attn-stoch":
        pred = AttnPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=True)
    elif pred_kind == "variational":
        gnd = cfg.get("global_noise_dim", 0)
        pred = VariationalConvPredictor(dim=dim, hidden=phid, n_blocks=pb, global_noise_dim=gnd)
    else:
        pred = make_predictor(pred_kind, dim=dim)
    pred.load_state_dict(blob["predictor_state"]); pred.eval()
    act_embed = nn.Embedding(4, dim)
    act_embed.load_state_dict(blob["action_embed_state"]); act_embed.eval()

    vq_K = cfg.get("vq_K", 0)
    fsq_levels = cfg.get("fsq_levels", 0)
    vq = None
    if fsq_levels > 0:
        vq = SpatialFSQ(dim=dim, levels=fsq_levels)
        if "vq_state" in blob:
            vq.load_state_dict(blob["vq_state"])
        vq.eval(); vq = vq.to(args.device)
    elif vq_K > 0 and "vq_state" in blob:
        vq = SpatialVQ(dim=dim, K=vq_K)
        vq.load_state_dict(blob["vq_state"]); vq.eval()
        vq = vq.to(args.device)

    enc = enc.to(args.device); dec = dec.to(args.device)
    pred = pred.to(args.device); act_embed = act_embed.to(args.device)
    ep = blob.get("epoch", "?")

    pygame.init()
    cell = 64 * args.scale
    # 3-pane layout: GT | AE | JEPA
    screen = pygame.display.set_mode((cell * 3 + 16, cell + 60))
    title = f"LeWM-Snake [{args.run}] ep={ep} arch={arch}"
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("monospace", 14)
    small_font = pygame.font.SysFont("monospace", 11)
    clock = pygame.time.Clock()

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    grid_cells = cfg.get("grid_cells", 64)
    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    env = Snake(seed=int(rng.integers(1 << 30)), grid_cells=grid_cells)

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
                    env = Snake(seed=int(rng.integers(1 << 30)), grid_cells=grid_cells)
                    z = reset_imag(); step_idx = 0
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        _, done = env.step(current_action)
        if done:
            env = Snake(seed=int(rng.integers(1 << 30)), grid_cells=grid_cells)
            z = reset_imag(); step_idx = 0
            continue
        real_frame = env.render()
        real_t = torch.from_numpy(real_frame).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0

        with torch.no_grad():
            # AE: encode + decode current real frame (no rollout) — tests encoder/decoder fidelity
            z_ae = enc(real_t)
            if vq is not None:
                z_ae_q, _ = vq(z_ae)
            else:
                z_ae_q = z_ae
            raw_ae = dec(z_ae_q)
            recon_ae = render_oracle_output(raw_ae, "cat-kmeans-unique", K=K, palette=palette).clamp(0, 1)[0]

            # JEPA: roll predictor forward — tests dynamics
            a_t = act_embed(torch.tensor([current_action], device=args.device))
            pred_out = pred(z, a_t)
            z = pred_out[0] if isinstance(pred_out, tuple) else pred_out
            if cfg.get("bound_latent", False):
                z = z.tanh()
            if vq is not None:
                z, _ = vq(z)
            raw_jepa = dec(z)
            recon_jepa = render_oracle_output(raw_jepa, "cat-kmeans-unique", K=K, palette=palette).clamp(0, 1)[0]

        screen.fill((0, 0, 0))
        # GT
        gt_surf = pygame.surfarray.make_surface(real_frame.swapaxes(0, 1))
        gt_surf = pygame.transform.scale(gt_surf, (cell, cell))
        screen.blit(gt_surf, (0, 0))
        # AE
        ae_arr = (recon_ae.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        ae_surf = pygame.surfarray.make_surface(ae_arr.swapaxes(0, 1))
        ae_surf = pygame.transform.scale(ae_surf, (cell, cell))
        screen.blit(ae_surf, (cell + 8, 0))
        # JEPA
        je_arr = (recon_jepa.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
        je_surf = pygame.surfarray.make_surface(je_arr.swapaxes(0, 1))
        je_surf = pygame.transform.scale(je_surf, (cell, cell))
        screen.blit(je_surf, (cell * 2 + 16, 0))
        # Labels
        screen.blit(small_font.render("GT", True, (180, 180, 180)), (cell // 2 - 8, cell + 4))
        screen.blit(small_font.render("AE (enc+dec on GT)", True, (180, 180, 180)), (cell + 8 + cell // 2 - 50, cell + 4))
        screen.blit(small_font.render("JEPA (predictor rollout)", True, (180, 180, 180)), (cell * 2 + 16 + cell // 2 - 60, cell + 4))
        info = font.render(
            f"ep={ep} arch={arch} grid={grid_cells} step={step_idx} act={NAMES[current_action]}",
            True, (200, 200, 200),
        )
        screen.blit(info, (8, cell + 28))
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


def _build_arch_jepa_model(blob, device):
    """Helper that builds enc/pred/dec/act_embed/vq from a checkpoint blob."""
    import torch.nn as nn
    from model import (
        OracleEncoderCNN, TinyDecoder, SpatialEncoder, SpatialDecoder, SpatialPixShufDecoder,
        ConvPredictor, StochasticConvPredictor, GlobalConvPredictor, AttnPredictor,
        VariationalConvPredictor,
        SpatialVQ, SpatialFSQ, make_predictor,
    )
    cfg = blob["cfg"]
    arch = cfg["arch_kind"]; dim = cfg["dim"]; K = cfg["K"]
    out_channels = cfg["out_channels"]
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

    pb = cfg.get("pred_blocks", 2)
    phid = cfg.get("pred_hidden", 64)
    rn = cfg.get("run_name", "")
    if "deep-stoch" in rn: pb, phid = 6, 128
    elif "attn-deep" in rn: pb, phid = 4, 128

    if pred_kind == "stoch-conv":
        pred = StochasticConvPredictor(dim=dim, hidden=phid, n_blocks=pb)
    elif pred_kind == "global-conv":
        pred = GlobalConvPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=False)
    elif pred_kind == "global-stoch-conv":
        pred = GlobalConvPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=True)
    elif pred_kind == "attn":
        pred = AttnPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=False)
    elif pred_kind == "attn-stoch":
        pred = AttnPredictor(dim=dim, hidden=phid, n_blocks=pb, stochastic=True)
    elif pred_kind == "variational":
        gnd = cfg.get("global_noise_dim", 0)
        pred = VariationalConvPredictor(dim=dim, hidden=phid, n_blocks=pb, global_noise_dim=gnd)
    else:
        pred = make_predictor(pred_kind, dim=dim)
    pred.load_state_dict(blob["predictor_state"]); pred.eval()
    act_embed = nn.Embedding(4, dim)
    act_embed.load_state_dict(blob["action_embed_state"]); act_embed.eval()

    vq_K = cfg.get("vq_K", 0)
    fsq_levels = cfg.get("fsq_levels", 0)
    vq = None
    if fsq_levels > 0:
        vq = SpatialFSQ(dim=dim, levels=fsq_levels)
        if "vq_state" in blob:
            vq.load_state_dict(blob["vq_state"])
        vq.eval(); vq = vq.to(device)
    elif vq_K > 0 and "vq_state" in blob:
        vq = SpatialVQ(dim=dim, K=vq_K)
        vq.load_state_dict(blob["vq_state"]); vq.eval()
        vq = vq.to(device)

    enc = enc.to(device); dec = dec.to(device); pred = pred.to(device); act_embed = act_embed.to(device)
    return dict(enc=enc, dec=dec, pred=pred, act_embed=act_embed, vq=vq, palette=palette,
                K=K, dim=dim, cfg=cfg, ep=blob.get("epoch", "?"))


def _viz_latent_rgb(z, out_size=64):
    """Visualize a spatial latent z (B, C, H, W) or (B, dim) as an RGB image.
    Takes first 3 channels, normalizes, upsamples to out_size."""
    import torch.nn.functional as F
    if z.dim() == 2:
        # flat latent: just show as horizontal stripe
        v = z[0]
        v = (v - v.min()) / (v.max() - v.min() + 1e-6)
        img = v.unsqueeze(0).expand(3, -1).unsqueeze(-1).expand(-1, -1, 4)
        img = F.interpolate(img.unsqueeze(0), size=(out_size, out_size), mode="nearest")[0]
        return img.clamp(0, 1)
    # spatial: take first 3 channels
    v = z[0, :3]                                                        # (3, H, W)
    v_min = v.amin(dim=(-2, -1), keepdim=True)
    v_max = v.amax(dim=(-2, -1), keepdim=True)
    v = (v - v_min) / (v_max - v_min + 1e-6)
    if v.size(-1) != out_size or v.size(-2) != out_size:
        v = F.interpolate(v.unsqueeze(0), size=(out_size, out_size), mode="nearest")[0]
    return v.clamp(0, 1)


def _compute_class_counts(palette, grid_cells, n_frames=20):
    """Sample fresh game frames, count avg pixels per palette class.
    Used to cap per-class rendering at play time (no hardcoding)."""
    from snake import Snake
    env = Snake(seed=0, grid_cells=grid_cells)
    frames = [env.render()]
    for _ in range(n_frames - 1):
        f, d = env.step(env.dir)
        frames.append(f)
        if d:
            env = Snake(seed=int(time.time()) & 0xFFFF, grid_cells=grid_cells)
            frames.append(env.render())
    pix = torch.from_numpy(np.stack(frames)).float() / 255.0                 # (N, H, W, 3)
    pal = palette.view(1, palette.size(0), 1, 1, 3)
    dist = (pix.unsqueeze(1) - pal).pow(2).sum(dim=-1)                       # (N, K, H, W)
    labels = dist.argmin(dim=1)                                               # (N, H, W)
    counts = torch.bincount(labels.flatten(), minlength=palette.size(0)).float()
    avg = counts / len(frames)
    return avg.round().long()                                                  # (K,) avg per-frame


def run_combo_play(args):
    """Open ONE window with GT | ENC | DEC | JEPA per run, stacked vertically.
    Each row is a separate run with its own grid_cells / model.
    Single keyboard input drives all envs simultaneously.
    """
    import pygame
    from snake import Snake
    from model import render_oracle_output, render_topk_per_class

    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    name = f"epoch_{args.epoch:03d}.pt" if args.epoch else "latest.pt"

    rows = []
    for rn in run_names:
        ckpt = pull_checkpoint(name, run=rn)
        if not ckpt.exists():
            print(f"[combo] {rn}: checkpoint not found, skipping")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if blob.get("cfg", {}).get("mode") != "arch_jepa":
            print(f"[combo] {rn}: not an arch_jepa run, skipping")
            continue
        m = _build_arch_jepa_model(blob, args.device)
        gc = m["cfg"].get("grid_cells", 64)
        env = Snake(seed=int(time.time()) & 0xFFFF, grid_cells=gc)
        # Per-class pixel count prior (auto-derived from this game's data distribution)
        class_counts = _compute_class_counts(m["palette"], gc).to(args.device)
        print(f"  {rn}: class_counts={class_counts.tolist()}")
        # Seed JEPA latent
        seed_frame = env.render()
        f0 = torch.from_numpy(seed_frame).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
        with torch.no_grad():
            z0 = m["enc"](f0)
            if m["cfg"].get("bound_latent", False):
                z0 = z0.tanh()
        rows.append(dict(run=rn, model=m, env=env, z=z0, grid_cells=gc, class_counts=class_counts))

    if not rows:
        print("[combo] no valid runs")
        return

    pygame.init()
    cell = 64 * args.scale
    n_panes = 4  # GT, ENC, DEC, JEPA
    n_rows = len(rows)
    pad_x, pad_y = 8, 28
    W = cell * n_panes + pad_x * (n_panes - 1)
    H = (cell + pad_y) * n_rows + 24
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"LeWM-Snake combo: {len(rows)} runs × 4 panes")
    font = pygame.font.SysFont("monospace", 14)
    small_font = pygame.font.SysFont("monospace", 11)
    clock = pygame.time.Clock()

    KEY_TO_ACTION = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
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
                elif event.key == pygame.K_n:
                    s = int(rng.integers(1 << 30))
                    for r in rows:
                        r["env"] = Snake(seed=s, grid_cells=r["grid_cells"])
                        f0 = torch.from_numpy(r["env"].render()).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
                        with torch.no_grad():
                            r["z"] = r["model"]["enc"](f0)
                            if r["model"]["cfg"].get("bound_latent", False):
                                r["z"] = r["z"].tanh()
                    step_idx = 0
                elif event.key == pygame.K_r:
                    # Reload each row's latest checkpoint from the volume
                    print(f"[combo] reloading {len(rows)} runs ...")
                    for r in rows:
                        new_ckpt = pull_checkpoint("latest.pt", run=r["run"])
                        if not new_ckpt.exists(): continue
                        blob2 = torch.load(new_ckpt, map_location="cpu", weights_only=False)
                        r["model"]["enc"].load_state_dict(blob2["encoder_state"])
                        r["model"]["dec"].load_state_dict(blob2["decoder_state"])
                        r["model"]["pred"].load_state_dict(blob2["predictor_state"])
                        r["model"]["act_embed"].load_state_dict(blob2["action_embed_state"])
                        if r["model"]["vq"] is not None and "vq_state" in blob2:
                            r["model"]["vq"].load_state_dict(blob2["vq_state"])
                        r["model"]["ep"] = blob2.get("epoch", "?")
                        # Re-seed JEPA latent from current frame
                        f0 = torch.from_numpy(r["env"].render()).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
                        with torch.no_grad():
                            r["z"] = r["model"]["enc"](f0)
                            if r["model"]["cfg"].get("bound_latent", False):
                                r["z"] = r["z"].tanh()
                        print(f"  {r['run']}: reloaded ep={r['model']['ep']}")
                    pygame.display.set_caption(
                        "LeWM-Snake combo  " + "  ".join(f"{r['run']}:ep{r['model']['ep']}" for r in rows)
                    )
                elif event.key in KEY_TO_ACTION:
                    current_action = KEY_TO_ACTION[event.key]

        screen.fill((0, 0, 0))
        for ri, r in enumerate(rows):
            y0 = ri * (cell + pad_y)
            env = r["env"]
            m = r["model"]
            _, done = env.step(current_action)
            if done:
                env = Snake(seed=int(rng.integers(1 << 30)), grid_cells=r["grid_cells"])
                r["env"] = env
                f0 = torch.from_numpy(env.render()).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
                with torch.no_grad():
                    r["z"] = m["enc"](f0)
                    if m["cfg"].get("bound_latent", False):
                        r["z"] = r["z"].tanh()
                continue
            real_frame = env.render()
            real_t = torch.from_numpy(real_frame).float().permute(2, 0, 1).unsqueeze(0).to(args.device) / 255.0
            with torch.no_grad():
                z_enc = m["enc"](real_t)
                if m["cfg"].get("bound_latent", False):
                    z_enc = z_enc.tanh()
                if m["vq"] is not None:
                    z_enc_q, _ = m["vq"](z_enc)
                else:
                    z_enc_q = z_enc
                raw_dec = m["dec"](z_enc_q)
                if getattr(args, "cap_classes", False):
                    recon_dec = render_topk_per_class(raw_dec, m["palette"], r["class_counts"]).clamp(0, 1)[0]
                else:
                    recon_dec = render_oracle_output(raw_dec, "cat-kmeans-unique",
                                                      K=m["K"], palette=m["palette"]).clamp(0, 1)[0]
                a_t = m["act_embed"](torch.tensor([current_action], device=args.device))
                pred_out = m["pred"](r["z"], a_t)
                z_new = pred_out[0] if isinstance(pred_out, tuple) else pred_out
                if m["cfg"].get("bound_latent", False):
                    z_new = z_new.tanh()
                if m["vq"] is not None:
                    z_new, _ = m["vq"](z_new)
                r["z"] = z_new
                raw_jepa = m["dec"](z_new)
                if getattr(args, "cap_classes", False):
                    recon_jepa = render_topk_per_class(raw_jepa, m["palette"], r["class_counts"]).clamp(0, 1)[0]
                else:
                    recon_jepa = render_oracle_output(raw_jepa, "cat-kmeans-unique",
                                                      K=m["K"], palette=m["palette"]).clamp(0, 1)[0]
                enc_viz = _viz_latent_rgb(z_enc, out_size=64)

            def blit_pane(frame_chw_or_hwc, col):
                if isinstance(frame_chw_or_hwc, np.ndarray):
                    arr = frame_chw_or_hwc
                else:
                    arr = (frame_chw_or_hwc.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).round().astype(np.uint8)
                surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                surf = pygame.transform.scale(surf, (cell, cell))
                screen.blit(surf, (col * (cell + pad_x), y0))

            blit_pane(real_frame, 0)
            blit_pane(enc_viz, 1)
            blit_pane(recon_dec, 2)
            blit_pane(recon_jepa, 3)
            # row label
            label = f"{r['run']}  grid={r['grid_cells']}  ep={m['ep']}"
            screen.blit(small_font.render(label, True, (200, 200, 200)), (4, y0 + cell + 2))
            for col, name_ in enumerate(["GT", "ENC", "DEC(AE)", "JEPA"]):
                screen.blit(small_font.render(name_, True, (140, 140, 140)),
                            (col * (cell + pad_x) + cell // 2 - 18, y0 + cell + 14))
        # bottom: action
        info = font.render(f"step={step_idx}  action={NAMES[current_action]}  (n=new game, q=quit)", True, (200, 200, 200))
        screen.blit(info, (8, H - 20))
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
    ap.add_argument("--scale", type=int, default=4, help="display upscaling factor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--run", default=None, help="experiment subdir under volume root")
    ap.add_argument("--runs", default=None,
                    help="comma-separated run names — opens combo window with GT|ENC|DEC|JEPA per run")
    ap.add_argument("--cap-classes", action="store_true",
                    help="cap each rendered class to its data prior (top-K-per-class). Off by default — "
                         "use argmax to see raw model output.")
    args = ap.parse_args()

    if args.runs:
        run_combo_play(args)
        return

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
