#!/bin/bash

# ==== 共通參數設定 ====
ENV_NAME="BipedalWalker-v3"
ALGO="pets"
DEVICE="cuda:0"
SEED=42
NOW=$(date +%Y%m%d_%H%M%S)
mkdir -p info/debugs

LOG_FILE="info/debugs/${ENV_NAME}_run_PETS_constraint_${NOW}.log"
echo "📄 Logging to: $LOG_FILE"

COMMON_FLAGS="\
overrides.env=${ENV_NAME} \
algorithm.name=${ALGO} \
algorithm.agent._target_=mbrl.planning.TrajectoryOptimizerAgent \
algorithm.agent.action_lb=-1 \
algorithm.agent.action_ub=1 \
algorithm.agent.planning_horizon=20 \
algorithm.agent.replan_freq=1 \
algorithm.agent.verbose=false \
dynamics_model.model._target_=mbrl.models.BasicEnsemble \
dynamics_model.model.num_layers=4 \
dynamics_model.model.in_size=24 \
dynamics_model.model.out_size=24 \
dynamics_model.model.layer_size=200 \
dynamics_model.ensemble_size=5 \
device=${DEVICE} \
seed=${SEED} \
save_video=true"

(
echo "🚀 Training with constraint [-0.8, 0.8]..."
python train_model_based.py train_entry ${COMMON_FLAGS} train_min=-0.8 train_max=0.8

echo
echo "🔍 Testing with both train/test constraint [-0.8, 0.8]..."
python train_model_based.py train_entry ${COMMON_FLAGS} train_min=-0.8 train_max=0.8 test_min=-0.8 test_max=0.8

echo
echo "✅ All PETS jobs finished."
) > "$LOG_FILE" 2>&1 &

echo "🧵 Job started in background. Logs: $LOG_FILE"
