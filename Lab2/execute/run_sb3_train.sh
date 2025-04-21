#!/bin/bash

# ===== CLI args with defaults =====
ENV_NAME=${1:-"Reacher-v2"}
ALGO=${2:-"PPO"}
TRAIN_MIN=${3:--0.8}
TRAIN_MAX=${4:-0.8}
TEST_MIN=${5:-$TRAIN_MIN}
TEST_MAX=${6:-$TRAIN_MAX}
TIMESTEPS=${7:-200000}

# ===== Logging setup =====
mkdir -p info/debugs/${ENV_NAME}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="info/debugs/${ENV_NAME}/${ENV_NAME}_run_${ALGO}_${TIMESTAMP}.log"

(
python3 <<EOF
import numpy as np
import importlib
from train_model_free import train_sb3, evaluate_sb3

algo_name = "${ALGO}"
SB3 = importlib.import_module("stable_baselines3")
AlgoClass = getattr(SB3, algo_name)

# === Optional training ===
# train_sb3(AlgoClass, "${ENV_NAME}", total_timesteps=${TIMESTEPS}, mini=${TRAIN_MIN}, maxi=${TRAIN_MAX})

# === Evaluation ===
model_path = f"info/models/${ENV_NAME}_${algo_name}_min${TRAIN_MIN}_max${TRAIN_MAX}.zip"
rewards = evaluate_sb3(
    AlgoClass,
    model_path=model_path,
    env_id="${ENV_NAME}",
    train_min=${TRAIN_MIN},
    train_max=${TRAIN_MAX},
    test_min=${TEST_MIN},
    test_max=${TEST_MAX}
)

print(f"📊 ${ALGO} on ${ENV_NAME} | Mean reward over {{len(rewards)}} episodes: {{np.mean(rewards):.2f}}")
EOF
) > "$LOG_FILE" 2>&1 &

echo "🚀 ${ALGO} launched on ${ENV_NAME}"
echo "📄 Logs: $LOG_FILE"
