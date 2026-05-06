"""Side-by-side: real Snake vs LeWM imagined Snake under the same action sequence.

Useful for sanity-checking how faithful the world model is at a given checkpoint.

Usage:
    python compare.py                # latest.pt, 30-step rollout, random actions
    python compare.py --epoch 12 --steps 60
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from snake import Snake
from model import LeWMSnake
from play import pull_checkpoint, load_model, CKPT_LOCAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--out", type=str, default="compare.png")
    args = ap.parse_args()

    name = f"epoch_{args.epoch:03d}.pt" if args.epoch else "latest.pt"
    ckpt = (CKPT_LOCAL / name) if args.no_pull else pull_checkpoint(name)
    model, ep, step = load_model(ckpt)
    model.eval()
    H = model.history

    rng = np.random.default_rng(args.seed)
    actions = rng.integers(0, 4, size=args.steps).tolist()

    env = Snake(seed=args.seed)
    real_frames = [env.render()]
    real_actions = []
    for _ in range(H - 1):
        f, _ = env.step(env.dir)
        real_frames.append(f)
        real_actions.append(env.dir)
    seed_pixels = np.stack(real_frames)
    seed_actions = real_actions + [env.dir]
    for a in actions:
        f, _ = env.step(int(a))
        real_frames.append(f)

    seed_pixels_t = torch.from_numpy(seed_pixels).float().permute(0, 3, 1, 2) / 255.0

    with torch.no_grad():
        seed_d = seed_pixels_t.unsqueeze(0)
        emb = model.encode(seed_d)                                # (1, H, P, D)
        act_buf = model.action_embed(torch.tensor(seed_actions).unsqueeze(0))
        imagined = [seed_pixels_t[i] for i in range(H)]
        for a in actions:
            new_act = model.action_embed(torch.tensor([[a]]))
            act_buf = torch.cat([act_buf[:, 1:], new_act], dim=1)
            pred = model.predict(emb[:, -H:], act_buf)             # (1, H, P, D)
            next_z = pred[:, -1:]                                   # (1, 1, P, D)
            emb = torch.cat([emb, next_z], dim=1)
            frame = model.decoder(next_z[:, 0]).clamp(0, 1)[0]
            imagined.append(frame)

    real_arr = np.stack(real_frames).astype(np.float32) / 255.0
    imagined_arr = torch.stack(imagined).numpy().transpose(0, 2, 3, 1)
    n = min(len(real_arr), len(imagined_arr))

    try:
        import PIL.Image as Image
    except ImportError:
        print("install pillow first: python3 -m pip install pillow")
        return
    s = args.scale
    cell = 64 * s
    canvas = np.zeros((cell * 2 + 20, cell * n + (n - 1) * 2, 3), dtype=np.uint8)
    canvas[:] = 30
    for i in range(n):
        x0 = i * (cell + 2)
        a = (real_arr[i] * 255).round().astype(np.uint8)
        b = (imagined_arr[i] * 255).clip(0, 255).round().astype(np.uint8)
        a_im = Image.fromarray(a).resize((cell, cell), Image.NEAREST)
        b_im = Image.fromarray(b).resize((cell, cell), Image.NEAREST)
        canvas[:cell, x0:x0 + cell] = np.array(a_im)
        canvas[cell + 20:cell * 2 + 20, x0:x0 + cell] = np.array(b_im)
    Image.fromarray(canvas).save(args.out)
    mse = ((real_arr[:n] - imagined_arr[:n]) ** 2).mean()
    print(f"[compare] saved {args.out}  ep={ep}  steps={n}  pixel-MSE(real-vs-imagined)={mse:.4f}")
    print("[compare] top row = real Snake, bottom row = LeWM imagined")


if __name__ == "__main__":
    main()
