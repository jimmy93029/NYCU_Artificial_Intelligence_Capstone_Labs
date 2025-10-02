#!/bin/bash

ENV_NAME="Bipedal_Walker-v3"
ALGO="pets"
DEVICE="cuda:0"
SEED=42
NOW=$(date +%Y%m%d_%H%M%S)

mkdir -p info/debugs/${ENV_NAME}
LOG_FILE="info/debugs/${ENV_NAME}/${ENV_NAME}_run_PETS_free_${NOW}.log"
echo "📄 Logging to $LOG_FILE"

COMMON_FLAGS="\
algorithm=pets \
experiment=free \
overrides=pets_Bipedal_Walker \
dynamics_model=gaussian_mlp_ensemble \
action_optimizer=cem \
device=${DEVICE} \
seed=${SEED} \
save_video=true"


(
# echo "🚀 Training with constraint [-1, 1]..."
# python3 train_model_based.py ${COMMON_FLAGS} +train_min=-1 +train_max=1 +mode=train

echo
echo "🔍 Testing with test constraint [-1, 1]..."
python3 train_model_based.py ${COMMON_FLAGS} +train_min=-1 +train_max=1 +test_min=-1 +test_max=1 +mode=test

echo
echo "🔍 Testing with test constraint [-0.8, 0.8]..."
python3 train_model_based.py ${COMMON_FLAGS} +train_min=-1 +train_max=1 +test_min=-0.8 +test_max=0.8 +mode=test

echo
echo "🔍 Testing with both train/test constraint [-0.8, 0.8]..."
python3 train_model_based.py ${COMMON_FLAGS} +train_min=-0.8 +train_max=0.8 +test_min=-0.8 +test_max=0.8 +mode=test

echo
echo "✅ All tasks finished!"
) > "$LOG_FILE" 2>&1 &

echo "🧵 Background job started. Log: $LOG_FILE"
