#!/bin/bash

ENV_NAME="Reacher-v4"
ALGO="PPO"
TRAIN_MIN=-1
TRAIN_MAX=1
TIMESTEPS=200000
LOG_DIR="info/debugs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${ENV_NAME}_run_${ALGO}_free_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

(
# === Training ===
python3 train_model_free.py \
  --env "$ENV_NAME" \
  --algo "$ALGO" \
  --train_min $TRAIN_MIN \
  --train_max $TRAIN_MAX \
  --timesteps $TIMESTEPS \
  --mode train

# === Evaluation 1: Same constraint ===
python3 train_model_free.py \
  --env "$ENV_NAME" \
  --algo "$ALGO" \
  --train_min $TRAIN_MIN \
  --train_max $TRAIN_MAX \
  --test_min $TRAIN_MIN \
  --test_max $TRAIN_MAX \
  --mode test

# === Evaluation 2: Tighter constraint ===
python3 train_model_free.py \
  --env "$ENV_NAME" \
  --algo "$ALGO" \
  --train_min $TRAIN_MIN \
  --train_max $TRAIN_MAX \
  --test_min -0.8 \
  --test_max 0.8 \
  --mode test
) > "$LOG_FILE" 2>&1 &

echo "🚀 $ALGO on $ENV_NAME launched in background."
echo "📄 Logs: $LOG_FILE"
