#!/bin/bash

# ==== Common Settings ====
ENV_NAME="CarRacing-v2"
ALGO="planet"
DEVICE="cuda:0"
SEED=42
NOW=$(date +%Y%m%d_%H%M%S)

mkdir -p info/debugs/${ENV_NAME}
LOG_FILE="info/debugs/${ENV_NAME}/${ENV_NAME}_run_PLANET_constraint_${NOW}.log"
echo "📄 Logging to: $LOG_FILE"

COMMON_FLAGS="\
algorithm=${ALGO} \
experiment=constraint \
overrides=planet_Car_Racing \
dynamics_model=planet \
device=${DEVICE} \
seed=${SEED} \
save_video=true"

(
echo "🚀 Training with constraint [-0.8, 0.8]..."
python train_model_based.py ${COMMON_FLAGS} +train_min=-0.8 +train_max=0.8 +mode=train

echo
echo "🔍 Testing with both train/test constraint [-0.8, 0.8]..."
python train_model_based.py ${COMMON_FLAGS} +train_min=-0.8 +train_max=0.8 +test_min=-0.8 +test_max=0.8 +mode=test

echo
echo "✅ All PlaNet jobs finished."
) > "$LOG_FILE" 2>&1 &

echo "🧵 Job started in background. Logs: $LOG_FILE"
