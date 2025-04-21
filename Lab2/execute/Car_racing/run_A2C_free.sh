#!/bin/bash

mkdir -p info/debugs

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="info/debugs/CarRacing_run_A2C_free_${timestamp}.log"

(
python3 <<EOF
import numpy as np
from train_model_free import train_sb3, evaluate_sb3
from stable_baselines3 import A2C

# Train A2C on CarRacing-v2 with full steering range [-1, 1]
train_sb3(A2C, 'CarRacing-v2', total_timesteps=200_000, mini=-1, maxi=1)

# Evaluation 1: Same constraint
a2c_rewards_1 = evaluate_sb3(
    A2C,
    model_path='info/models/CarRacing-v2_A2C_min-1_max1.zip',
    env_id='CarRacing-v2',
    train_min=-1,
    train_max=1,
    test_min=-1,
    test_max=1
)
print(f"📊 Mean reward with test constraint [-1, 1]: {np.mean(a2c_rewards_1):.2f}")

# Evaluation 2: Tighter test-time constraint
a2c_rewards_2 = evaluate_sb3(
    A2C,
    model_path='info/models/CarRacing-v2_A2C_min-1_max1.zip',
    env_id='CarRacing-v2',
    train_min=-1,
    train_max=1,
    test_min=-0.8,
    test_max=0.8
)
print(f"📊 Mean reward with test constraint [-0.8, 0.8]: {np.mean(a2c_rewards_2):.2f}")
EOF
) > "$log_file" 2>&1 &

echo "🚀 A2C CarRacing training + evaluation launched in background."
echo "📄 Logs: $log_file"
