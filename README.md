# LeWM-Snake — Imagined Snake via a JEPA World Model

A scaled-down [LeWorldModel](https://arxiv.org/abs/2603.19312) trained to model the
dynamics of a 64×64 Snake game. After training, you play "the game" entirely
inside the model's latent imagination — no Snake env in the loop.

## What's here

```
snake.py         16×16-cell Snake env + heuristic dataset generator -> 64x64 RGB frames, 4 actions
model.py         Tiny ViT encoder + AdaLN-zero AR predictor + decoder + SIGReg
train_modal.py   Modal app: trains on an A10G GPU, dumps checkpoints to a Modal volume each epoch
play.py          Pulls latest checkpoint and lets you drive the imagined game with arrow keys
```

Loss exactly mirrors the LeWM paper:
- `pred_loss` — MSE between predicted next-step embedding and encoder embedding of the next frame
- `sigreg_loss` — Sketched Isotropic Gaussian Regularizer, applied per-timestep (Cramer-Wold + Epps-Pulley)
- `decoder_loss` — auxiliary pixel reconstruction with **stop-grad on the latent** so the JEPA loss is unaffected; only used so we can render frames during playback

No EMA, no stop-gradient on the prediction target, no pretrained encoder.

## Train (already kicked off)

```bash
~/Library/Python/3.9/bin/modal run --detach train_modal.py
```

Checkpoints are written to the Modal volume `lewm-snake-ckpts`:
- `latest.pt` — overwritten each epoch
- `epoch_001.pt`, `epoch_002.pt`, ... — per-epoch snapshots

## Play the imagined game

```bash
cd ~/study/lewm-snake
python3 play.py            # pulls latest.pt, opens a 512×512 window
python3 play.py --epoch 5  # play with a specific checkpoint
python3 play.py --no-pull  # use whatever's already in ./_ckpts/
```

Controls: arrow keys / WASD to move, `R` to pull the latest checkpoint and reset,
`N` to reset with the current model, `Q` to quit.

The game window shows the model's *imagined* next frame on every keypress.
Early on it'll be soup — that's expected; SIGReg keeps the latent from collapsing,
but the predictor needs many epochs to learn move semantics. As training
progresses you'll see snake-shaped blobs, then proper movement, then food
interactions. Reload (`R`) between checkpoints to feel the curriculum.
