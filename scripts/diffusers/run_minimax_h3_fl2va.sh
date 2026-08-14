#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/edge-dit.cpp
MODEL=/models/MiniMax-H3-Diffusers
OUTPUT=${1:-$ROOT/outputs/minimax-h3/diffusers-fl2va-768p/final.mp4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=/models/MiniMax-H3-Diffusers/code/diffusers-main/src${PYTHONPATH:+:$PYTHONPATH}

python3 "$ROOT/scripts/diffusers/run_minimax_h3_fl2va.py" \
  --model "$MODEL" \
  --image "$ROOT/outputs/minimax-h3/fl2va/edge/frames/first.png" \
  --last-image "$ROOT/outputs/minimax-h3/fl2va/edge/frames/last.png" \
  --output "$OUTPUT" \
  --height 768 --width 1376 --num-frames 124 --steps 20 --seed 42
