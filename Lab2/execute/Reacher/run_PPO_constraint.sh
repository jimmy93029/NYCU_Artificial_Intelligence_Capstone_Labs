#!/bin/bash

# ==== Configurable Parameters ====
ENV_NAME="Reacher-v4"
ALGO="PPO"
TRAIN_MIN=-0.8
TRAIN_MAX=0.8
TEST_MIN=-0.8
TEST_MAX=0.8
TIMESTEPS=200000

# ==== Logging Setup ====
mkdir -p info/debugs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="info/debugs/${ENV_NAME}_run_${ALGO}_constraint_${TIMESTAMP}.log"

(
# === Train ===
python3 train_model_free.py \
  --env "$ENV_NAME" \
  --algo "$ALGO" \
  --train_min $TRAIN_MIN \
  --train_max $TRAIN_MAX \
  --timesteps $TIMESTEPS \
  --mode train

# === Evaluate ===
python3 train_model_free.py \
  --env "$ENV_NAME" \
  --algo "$ALGO" \
  --train_min $TRAIN_MIN \
  --train_max $TRAIN_MAX \
  --test_min $TEST_MIN \
  --test_max $TEST_MAX \
  --mode test
) > "$LOG_FILE" 2>&1 &

echo "🚀 $ALGO training + evaluation with constraint [$TRAIN_MIN, $TRAIN_MAX] launched."
echo "📄 Logs: $LOG_FILE"
