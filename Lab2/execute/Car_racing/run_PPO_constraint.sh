#!/bin/bash

mkdir -p info/debugs

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="info/debugs/CarRacing_run_PPO_constraint_${timestamp}.log"

(
python3 <<EOF
import numpy as np
from train_model_free import train_sb3, evaluate_sb3
from stable_baselines3 import PPO

# Train PPO on CarRacing-v2 with steering constraint [-0.8, 0.8]
train_sb3(PPO, "CarRacing-v2", total_timesteps=200_000, mini=-0.8, maxi=0.8)

# Evaluate with same constraint
ppo_rewards_8 = evaluate_sb3(
    PPO,
    model_path="info/models/CarRacing-v2_PPO_min-0.8_max0.8.zip",
    env_id="CarRacing-v2",
    train_min=-0.8,
    train_max=0.8,
    test_min=-0.8,
    test_max=0.8
)

# Print mean reward
print(f"📊 Mean reward over {len(ppo_rewards_8)} episodes: {np.mean(ppo_rewards_8):.2f}")
EOF
) > "$log_file" 2>&1 &

echo "🚀 PPO CarRacing training + evaluation launched in background."
echo "📄 Logs: $log_file"
