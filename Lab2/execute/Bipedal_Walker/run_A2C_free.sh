#!/bin/bash

echo "Starting A2C training and evaluation on BipedalWalker-v3..."

python3 <<EOF
from train_model_free import train_sb3, evaluate_sb3
from stable_baselines3 import A2C

# Training phase
train_sb3(A2C, 'BipedalWalker-v3', total_timesteps=200_000, mini=-1, maxi=1)

# Evaluation (Test-time constraint = Train-time constraint)
print("A2C BipedalWalker | Test-time = Train-time constraint [-1, 1]")
evaluate_sb3(
    A2C,
    model_path='info/models/BipedalWalker-v3_A2C_min-1_max1.zip',
    env_id='BipedalWalker-v3',
    train_min=-1,
    train_max=1,
    test_min=-1,
    test_max=1
)

# Evaluation (Tighter test-time constraint)
print("A2C BipedalWalker | Test-time constraint [-0.8, 0.8]")
evaluate_sb3(
    A2C,
    model_path='info/models/BipedalWalker-v3_A2C_min-1_max1.zip',
    env_id='BipedalWalker-v3',
    train_min=-1,
    train_max=1,
    test_min=-0.8,
    test_max=0.8
)
EOF

echo "✅ All done!"
