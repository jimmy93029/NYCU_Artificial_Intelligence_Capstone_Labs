import gymnasium as gym
import numpy as np
import cv2
import torch
from gymnasium.wrappers import TransformObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


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


# === Preprocessing for CarRacing ===
def preprocess_car_racing(env):
    env = TransformObservation(env, lambda obs: cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[..., None])
    env = ResizeObservation(env, shape=(84, 84))
    return env


# === Shared constraint wrapper ===
def wrap_with_constraints(env, env_id, mini=-1, maxi=1):
    if "CarRacing" in env_id:
        env = CarRacingSteeringConstraint(env, mini, maxi)
    elif "BipedalWalker" in env_id:
        env = BipedalWalkerActionConstraint(env, mini, maxi)
    return env


# === For SB3 Training ===
def make_sb3_env(env_id, mini=-1, maxi=1):
    env = gym.make(env_id)
    env = Monitor(env)
    if "CarRacing" in env_id:
        env = preprocess_car_racing(env)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4)
    env = wrap_with_constraints(env, env_id, mini, maxi)
    return env


# === For MBRL Training ===
def make_mbrl_env(env_id, mini, maxi):
    env = gym.make(env_id)
    env = wrap_with_constraints(env, env_id, mini, maxi)
    return env  


# === Constraint Function for Evaluation ===
def make_action_constraint_fn(min_val, max_val):
    def constrain(action):
        return np.clip(action, min_val, max_val)
    return constrain


def bipedalwalker_reward_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    Approximate BipedalWalker-v3 reward shaping, as used in Gym implementation.

    next_obs: tensor of shape (B, 24)
      - [0]: hull angle
      - [1]: hull angular velocity
      - [2]: horizontal velocity (proxy for pos[0] derivative)
      - [3]: vertical velocity
    act: tensor of shape (B, 4)

    Returns: reward (B, 1)
    """
    vel_x = next_obs[:, 2]  # proxy of pos[0] increase
    angle = next_obs[:, 0]

    shaping = 130 * vel_x - 5.0 * torch.abs(angle)
    torque_cost = 0.00035 * 80.0 * torch.abs(act).clamp(0, 1).sum(dim=1)
    reward = shaping - torque_cost
    return reward.view(-1, 1)


def bipedalwalker_termination_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    Done if the agent falls (vertical velocity too negative)
    or moves beyond the terrain.
    """
    vel_y = next_obs[:, 3]
    done = (vel_y < -1.0) | (next_obs[:, 2] > 1.5)  # falls too fast or runs too far
    return done.view(-1, 1)
