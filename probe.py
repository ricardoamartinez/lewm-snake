"""Probe diagnostic: frozen-encoder linear/MLP probes for head & food position.

For each trained checkpoint, freeze the encoder and ask:
  - Can a linear probe (or MLP probe) recover snake-head position from [CLS]?
  - Can it recover food position from [CLS]?

If `head` is recoverable but `food` isn't, the encoder is selectively dropping
food → fix is at the encoder / JEPA level. If both are unrecoverable, more
training / bigger encoder. If both are recoverable, the decoder is the issue.

Generates fresh episodes locally, encodes through each model, trains probes.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from snake import GRID, CELL, generate_dataset
from model import LeWMSnake


RUNS = ["dim8", "dim16", "dim32", "dim64", "dim128"]
VOLUME = "lewm-snake-ckpts"


def modal_bin():
    import os, subprocess
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
        except Exception:
            continue
    return None


def pull_run(run, name="latest.pt"):
    import subprocess
    mb = modal_bin()
    local = Path("./_ckpts") / run
    local.mkdir(parents=True, exist_ok=True)
    target = local / name
    if target.exists():
        target.unlink()
    r = subprocess.run([mb, "volume", "get", VOLUME, f"{run}/{name}", str(target), "--force"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[{run}] pull failed: {r.stderr.strip()}")
        return None
    return target


def gen_probing_dataset(num_eps=400, seed=0):
    """Returns (frames (N, 3, 64, 64), targets (N, 4)).
    targets columns: head_row, head_col, food_row, food_col, normalised to [0,1]."""
    from snake import Snake, heuristic_action
    rng = np.random.default_rng(seed)
    frames, tgts = [], []
    for ep in range(num_eps):
        env = Snake(seed=seed + ep)
        for _ in range(rng.integers(60, 180)):
            f = env.render()
            head = env.body[0]
            food = env.food if env.food is not None else (-1, -1)
            frames.append(f)
            tgts.append([head[0] / GRID, head[1] / GRID, food[0] / GRID, food[1] / GRID])
            a = heuristic_action(env) if rng.random() > 0.2 else int(rng.integers(4))
            _, done = env.step(a)
            if done:
                break
    frames = np.stack(frames).astype(np.float32) / 255.0       # (N, 64, 64, 3)
    frames = frames.transpose(0, 3, 1, 2)                       # (N, 3, 64, 64)
    tgts = np.asarray(tgts, dtype=np.float32)                   # (N, 4)
    return torch.from_numpy(frames), torch.from_numpy(tgts)


@torch.no_grad()
def encode_all(model, frames, batch=128, device="cpu"):
    """Run frames through model.encode and concatenate the [CLS] vectors.
    encode expects (B, T, 3, H, W). We pretend T=1 to reuse the path."""
    out = []
    model.eval()
    for i in range(0, frames.size(0), batch):
        chunk = frames[i:i + batch].to(device).unsqueeze(1)     # (B, 1, 3, 64, 64)
        emb = model.encode(chunk)                               # (B, 1, D)
        out.append(emb.squeeze(1).cpu())
    return torch.cat(out, dim=0)


def train_probe(z_train, y_train, z_test, y_test, kind="linear", epochs=200, lr=1e-2, device="cpu"):
    D = z_train.size(1)
    if kind == "linear":
        probe = nn.Linear(D, y_train.size(1))
    else:
        probe = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, y_train.size(1)))
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    bs = 1024
    z_train, y_train = z_train.to(device), y_train.to(device)
    for _ in range(epochs):
        idx = torch.randperm(z_train.size(0), device=device)
        for s in range(0, z_train.size(0), bs):
            b = idx[s:s + bs]
            pred = probe(z_train[b])
            loss = (pred - y_train[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(z_test.to(device))
        # Per-target MSE in normalized units, RMSE in cells
        per = (pred - y_test.to(device)).pow(2).mean(dim=0).sqrt() * GRID
    return per.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_eps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args()

    print(f"[probe] generating {args.num_eps} episodes for probing ...")
    frames, tgts = gen_probing_dataset(num_eps=args.num_eps, seed=0)
    print(f"[probe] dataset: frames {tuple(frames.shape)}  tgts {tuple(tgts.shape)}")
    n = frames.size(0)
    n_train = int(n * 0.9)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    f_tr, t_tr = frames[train_idx], tgts[train_idx]
    f_te, t_te = frames[test_idx], tgts[test_idx]

    results = []
    for run in RUNS:
        if not args.no_pull:
            ckpt = pull_run(run, "latest.pt")
        else:
            ckpt = Path("./_ckpts") / run / "latest.pt"
        if ckpt is None or not ckpt.exists():
            print(f"[{run}] checkpoint missing, skipping")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = blob.get("cfg", {})
        model = LeWMSnake(
            dim=cfg.get("dim", 128), history=cfg.get("history", 4),
            n_actions=4, decoder_kind=cfg.get("decoder_kind", "convtranspose"),
            out_channels=cfg.get("out_channels", 3),
            latent_nll=cfg.get("latent_nll", False),
        )
        model.load_state_dict(blob["state_dict"])
        model.to(args.device)
        ep = blob.get("epoch", "?")

        print(f"\n[{run}] (epoch {ep}, loss={cfg.get('loss_kind')}) encoding {n} frames ...")
        z_all = encode_all(model, frames, batch=128, device=args.device)
        z_tr = z_all[train_idx]
        z_te = z_all[test_idx]
        # Standardise
        mu = z_tr.mean(0, keepdim=True)
        sd = z_tr.std(0, keepdim=True) + 1e-6
        z_tr = (z_tr - mu) / sd
        z_te = (z_te - mu) / sd

        rmse_lin = train_probe(z_tr, t_tr, z_te, t_te, kind="linear", device=args.device)
        rmse_mlp = train_probe(z_tr, t_tr, z_te, t_te, kind="mlp", device=args.device)
        # Chance baseline: predict the mean of training targets
        baseline_pred = t_tr.mean(0, keepdim=True).expand_as(t_te)
        rmse_chance = (baseline_pred - t_te).pow(2).mean(dim=0).sqrt() * GRID
        results.append((run, ep, rmse_lin.tolist(), rmse_mlp.tolist(), rmse_chance.tolist()))
        labels = ["head_y", "head_x", "food_y", "food_x"]
        print(f"  RMSE in cells (lower=better; chance ≈ random):")
        print(f"    {'target':<10} {'linear':>8} {'mlp':>8} {'chance':>8}")
        for lab, rl, rm, rc in zip(labels, rmse_lin, rmse_mlp, rmse_chance):
            print(f"    {lab:<10} {rl.item():>8.3f} {rm.item():>8.3f} {rc.item():>8.3f}")

    print("\n========== Summary (linear-probe RMSE, lower = better; chance ≈ uniform-on-grid) ==========")
    print(f"{'run':<14} {'epoch':>5} | {'head_y':>8} {'head_x':>8} {'food_y':>8} {'food_x':>8}")
    for run, ep, rl, rm, rc in results:
        print(f"{run:<14} {ep!s:>5} | {rl[0]:>8.3f} {rl[1]:>8.3f} {rl[2]:>8.3f} {rl[3]:>8.3f}")
    if results:
        chance = results[0][4]
        print(f"{'chance-baseline':<14} {'':>5} | {chance[0]:>8.3f} {chance[1]:>8.3f} {chance[2]:>8.3f} {chance[3]:>8.3f}")


if __name__ == "__main__":
    main()
