#!/bin/bash

ENV_NAME="Bipedal_Walker-v3"
mkdir -p info/debugs/${ENV_NAME}
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="info/debugs/${ENV_NAME}/${ENV_NAME}_run_PPO_constraint_${timestamp}.log"

(
python3 <<EOF
import numpy as np
from train_model_free import train_sb3, evaluate_sb3
from stable_baselines3 import PPO

# Train PPO on BipedalWalker-v3 with action constraint [-0.8, 0.8]
# train_sb3(PPO, "BipedalWalker-v3", total_timesteps=200_000, mini=-0.8, maxi=0.8)

# Evaluate with same constraint
ppo_rewards_8 = evaluate_sb3(
    PPO,
    model_path="info/models/BipedalWalker-v3_PPO_min-0.8_max0.8.zip",
    env_id="BipedalWalker-v3",
    train_min=-0.8,
    train_max=0.8,
    test_min=-0.8,
    test_max=0.8
)

# Print mean reward
print(f"📊 Mean reward over {len(ppo_rewards_8)} episodes: {np.mean(ppo_rewards_8):.2f}")
EOF
) > "$log_file" 2>&1 &

echo "🚀 PPO training launched in background."
echo "📄 Logs: $log_file"
