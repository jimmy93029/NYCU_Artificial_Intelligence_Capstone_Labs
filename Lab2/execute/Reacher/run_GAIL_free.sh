#!/bin/bash

ENV_NAME="Reacher-v4"
ALGO="GAIL"
TRAIN_MIN=-1
TRAIN_MAX=1
TIMESTEPS=200000
LOG_DIR="info/debugs/${ENV_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${ENV_NAME}_run_${ALGO}_free_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

(
# === Training Expert ===
python train_imitation.py \
    --env_id "$ENV_NAME" \
    --mode train_expert \
    --train_min $TRAIN_MIN \
    --train_max $TRAIN_MAX \
    --timesteps 300000

# === Training Imitation ===
python train_imitation.py \
    --env_id "$ENV_NAME" \
    --algo "$ALGO" \
    --mode train_imitation \
    --train_min $TRAIN_MIN \
    --train_max $TRAIN_MAX

# === Evaluation 1: Same constraint ===
python train_imitation.py \
    --env_id "$ENV_NAME" \
    --algo "$ALGO" \
    --mode evaluate \
    --train_min $TRAIN_MIN \
    --train_max $TRAIN_MAX \
    --test_min $TRAIN_MIN \
    --test_max $TRAIN_MAX

# === Evaluation 2: Tighter constraint ===
python train_imitation.py \
    --env_id "$ENV_NAME" \
    --algo "$ALGO" \
    --mode evaluate \
    --train_min $TRAIN_MIN \
    --train_max $TRAIN_MAX \
    --test_min -0.8 \
    --test_max 0.8

) > "$LOG_FILE" 2>&1 &

echo "🚀 $ALGO on $ENV_NAME launched in background."
echo "📄 Logs: $LOG_FILE"
