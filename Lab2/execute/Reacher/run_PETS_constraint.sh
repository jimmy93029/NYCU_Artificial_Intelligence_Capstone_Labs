#!/bin/bash

# ==== Common Settings ====
ENV_NAME="Reacher-v2"
ALGO="pets"
DEVICE="cuda:0"
SEED=42
NOW=$(date +%Y%m%d_%H%M%S)
mkdir -p info/debugs/${ENV_NAME}

LOG_FILE="info/debugs/${ENV_NAME}/${ENV_NAME}_run_PETS_constraint_${NOW}.log"
echo "📄 Logging to: $LOG_FILE"

COMMON_FLAGS="\
algorithm=pets \
experiment=constraint \
overrides=pets_reacher \
dynamics_model=gaussian_mlp_ensemble \
action_optimizer=cem \
device=${DEVICE} \
seed=${SEED} \
save_video=true"

(
# echo "🚀 Training with constraint [-0.8, 0.8]..."
# python train_model_based.py ${COMMON_FLAGS} +train_min=-0.8 +train_max=0.8 +mode=train

echo
echo "🔍 Testing with both train/test constraint [-0.8, 0.8]..."
python train_model_based.py ${COMMON_FLAGS} +train_min=-0.8 +train_max=0.8 +test_min=-0.8 +test_max=0.8 +mode=test

echo
echo "✅ All PETS jobs finished."
) > "$LOG_FILE" 2>&1 &

echo "🧵 Job started in background. Logs: $LOG_FILE"
