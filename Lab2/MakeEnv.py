# env_creator.py

import gymnasium as gym  # ✅ switched from gym to gymnasium
import numpy as np
import cv2
import torch
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

# === Constraint Wrappers ===
class CarRacingSteeringConstraint(gym.ActionWrapper):
    def __init__(self, env, mini=-0.8, maxi=0.8):
        super().__init__(env)
        self.min_steering = mini
        self.max_steering = maxi

    def action(self, action):
        action[0] = np.clip(action[0], self.min_steering, self.max_steering)
        return action


class BipedalWalkerActionConstraint(gym.ActionWrapper):
    def __init__(self, env, mini=-0.8, maxi=0.8):
        super().__init__(env)
        self.min_action = mini
        self.max_action = maxi

    def action(self, action):
        return np.clip(action, self.min_action, self.max_action)


class ReacherActionConstraint(gym.ActionWrapper):
    def __init__(self, env, mini=-1.0, maxi=1.0):
        super().__init__(env)
        self.min_action = mini
        self.max_action = maxi

    def action(self, action):
        return np.clip(action, self.min_action, self.max_action)


# === Preprocessing for CarRacing ===
def preprocess_car_racing(env):
    def process(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (64, 64))
        obs = np.expand_dims(obs, axis=0)
        return obs.astype(np.uint8)

    env = TransformObservation(env, process)
    return env


# === Shared constraint wrapper ===
def wrap_with_constraints(env, env_id, mini=-1, maxi=1):
    if "CarRacing" in env_id:
        env = CarRacingSteeringConstraint(env, mini, maxi)
    elif "BipedalWalker" in env_id:
        env = BipedalWalkerActionConstraint(env, mini, maxi)
    elif "Reacher" in env_id:
        env = ReacherActionConstraint(env, mini, maxi)
    return env


# === For SB3 / Imitation Training ===
def make_sb3_env(env_id, mini=-1, maxi=1):
    def _make():
        env = gym.make(env_id, render_mode="rgb_array")
        if "CarRacing" in env_id:
            env = preprocess_car_racing(env)
        env = wrap_with_constraints(env, env_id, mini, maxi)
        env = Monitor(env)
        return env

    return DummyVecEnv([_make])


# === For MBRL Training (same gymnasium) ===
def make_mbrl_env(env_id, mini, maxi):
    env = gym.make(env_id)
    env = wrap_with_constraints(env, env_id, mini, maxi)
    if "CarRacing" in env_id:
        env = preprocess_car_racing(env)
        env.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 64, 64), dtype=np.uint8
        )
    return env

def make_imitation_env(env_id, mini=-1, maxi=1, n_envs=8, seed=42):
    def post_wrappers(env, _):
        env = wrap_with_constraints(env, env_id, mini, maxi)
        return RolloutInfoWrapper(env)

    env = make_vec_env(
        env_id,
        rng=np.random.default_rng(seed),
        n_envs=n_envs,
        post_wrappers=[post_wrappers],
    )
    return env



# === Constraint Function for Evaluation ===
def make_action_constraint_fn(min_val, max_val):
    def constrain(action):
        return np.clip(action, min_val, max_val)
    return constrain


# === Reward/Termination Functions for BipedalWalker (used in MBRL)
def bipedalwalker_reward_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    vel_x = next_obs[:, 2]
    angle = next_obs[:, 0]
    shaping = 130 * vel_x - 5.0 * torch.abs(angle)
    torque_cost = 0.00035 * 80.0 * torch.abs(act).clamp(0, 1).sum(dim=1)
    reward = shaping - torque_cost
    return reward.view(-1, 1)


def bipedalwalker_termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    vel_y = next_obs[:, 3]
    done = (vel_y < -1.0) | (next_obs[:, 2] > 1.5)
    return done.view(-1, 1)
