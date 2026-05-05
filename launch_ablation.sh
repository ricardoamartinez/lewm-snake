#!/usr/bin/env bash
# Spawn 5 parallel H100 training jobs and start 5 local watchers, one per
# experiment. Each watcher polls the volume's run subdir for latest.pt and
# launches its own play.py window the moment that experiment's first epoch
# lands. Auto-exits once all 5 windows are open.

set -e
cd "$(dirname "$0")"

MODAL=~/Library/Python/3.9/bin/modal
RUNS=(dim8 dim16 dim32 dim64 dim128)

echo "[launch_ablation] wiping volume contents ..."
$MODAL volume ls lewm-snake-ckpts 2>/dev/null | tail -n +4 | head -n -1 | awk '{print $2}' | grep -v '^$' | while read entry; do
  $MODAL volume rm lewm-snake-ckpts "$entry" --recursive 2>/dev/null \
    || $MODAL volume rm lewm-snake-ckpts "$entry" 2>/dev/null \
    || true
done

echo "[launch_ablation] cleaning local _ckpts/"
rm -rf ./_ckpts && mkdir -p ./_ckpts

echo "[launch_ablation] spawning 5 H100 jobs ..."
$MODAL run --detach train_modal.py 2>&1 | tee /tmp/lewm_ablation_launch.log

echo "[launch_ablation] arming 5 watchers ..."
PIDS=()
for run in "${RUNS[@]}"; do
  (
    # Poll the run's subdir until latest.pt appears. modal CLI prefixes
    # entries with the subdir, so we grep for "<run>/latest.pt".
    until $MODAL volume ls lewm-snake-ckpts "$run" 2>/dev/null \
          | grep -qE "(^|/)latest\\.pt$"; do
      sleep 5
    done
    echo "[$run watcher] first epoch detected, launching play window"
    nohup python3 -u play.py --run "$run" > "/tmp/play_${run}.log" 2>&1 &
    disown
  ) &
  PIDS+=($!)
done

# Wait for every watcher subshell to finish (each exits right after firing play.py)
for p in "${PIDS[@]}"; do
  wait "$p" || true
done

echo "[launch_ablation] all 5 watchers fired. Pygame windows are open and survive shell exit."
