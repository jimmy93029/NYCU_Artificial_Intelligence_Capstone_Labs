#!/bin/bash

mkdir -p info/debugs

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="info/debugs/Bipedal_Walker_run_A2C_constraint_${timestamp}.log"

(
python3 <<EOF
import numpy as np
from train_model_free import train_sb3, evaluate_sb3
from stable_baselines3 import A2C

# Train A2C on BipedalWalker-v3 with action constraint [-0.8, 0.8]
# train_sb3(A2C, "BipedalWalker-v3", total_timesteps=200_000, mini=-0.8, maxi=0.8)

# Evaluate with same constraint
a2c_rewards_8 = evaluate_sb3(
    A2C,
    model_path="info/models/BipedalWalker-v3_A2C_min-0.8_max0.8.zip",
    env_id="BipedalWalker-v3",
    train_min=-0.8,
    train_max=0.8,
    test_min=-0.8,
    test_max=0.8
)
EOF
) > "$log_file" 2>&1 &

echo "🚀 A2C training launched in background."
echo "📄 Logs: $log_file"
