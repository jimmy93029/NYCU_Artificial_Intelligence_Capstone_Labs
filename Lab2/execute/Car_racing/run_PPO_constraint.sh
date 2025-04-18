#!/bin/bash

echo "🚗 Starting PPO training and evaluation on CarRacing-v3 with constraint [-0.8, 0.8]..."

python3 <<EOF
from train_model_free import train_sb3, evaluate_sb3
from stable_baselines3 import PPO

# Training with steering constraint [-0.8, 0.8]
train_sb3(PPO, "CarRacing-v3", total_timesteps=200_000, mini=-0.8, maxi=0.8)

# Evaluation with same constraint
print("🎯 Evaluating PPO CarRacing | Test-time = Train-time constraint [-0.8, 0.8]")
ppo_rewards_8 = evaluate_sb3(
    PPO,
    model_path="/content/drive/MyDrive/AI_capstone/models/CarRacing-v3_PPO_min-0.8_max0.8.zip",
    env_id="CarRacing-v3",
    train_min=-0.8,
    train_max=0.8,
    test_min=-0.8,
    test_max=0.8
)
EOF

echo "✅ All done!"
