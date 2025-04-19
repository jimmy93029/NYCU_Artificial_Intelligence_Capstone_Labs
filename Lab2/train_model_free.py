import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, RecordVideo, TransformObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO, A2C, SAC
import os

Base_dir = "info"

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
    # Convert to grayscale
    env = TransformObservation(env, lambda obs: cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[..., None])
    # Resize to (84, 84)
    env = ResizeObservation(env, shape=(84, 84))
    # Wrap in DummyVecEnv and apply VecFrameStack
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


# === Wrap with Constraints for Training and Evaluation ===
def wrap_with_constraints(env, env_id, mini=-1, maxi=1):
    if "CarRacing" in env_id:
        env = preprocess_car_racing(env)
        env = CarRacingSteeringConstraint(env, mini, maxi)
    if "BipedalWalker" in env_id:
        env = BipedalWalkerActionConstraint(env, mini, maxi)
    return env


# === Vectorized Environment Creation for Training ===
def make_custom_env(env_id, mini=-1, maxi=1):
    env = gym.make(env_id)
    env = wrap_with_constraints(env, env_id, mini, maxi)
    return Monitor(env)


# === Training Function ===
def train_sb3(algo, env_id, total_timesteps=100_000, mini=-1, maxi=1):
    policy = "CnnPolicy" if "ALE" in env_id or "CarRacing" in env_id else "MlpPolicy"
    algo_name = algo.__name__

    log_dir = f"{Base_dir}/logs/{env_id.replace('/', '_')}_{algo_name}_min{mini}_max{maxi}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_custom_env(env_id=env_id, mini=mini, maxi=maxi)

    model = algo(policy, env, verbose=1)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    model.learn(total_timesteps=total_timesteps)

    model_path = f"{Base_dir}/models/{env_id.replace('/', '_')}_{algo_name}_min{mini}_max{maxi}.zip"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    print(f"✅ Model saved to: {model_path}")
    print(f"📁 Logs stored in: {log_dir}")


# === Evaluation Function ===
def evaluate_sb3(model_class, model_path, env_id, episodes=10, train_min=-1, train_max=1, test_min=-1, test_max=1):
    model = model_class.load(model_path)
    algo = model_class.__name__

    video_folder = (
        f"{Base_dir}/videos/"
        f"{env_id.replace('/', '_')}_{algo}_"
        f"train_min{train_min}_max{train_max}_"
        f"test_min{test_min}_max{test_max}"
    )
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = wrap_with_constraints(env, env_id, test_min, test_max)

    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep == 0,
        name_prefix=f"{env_id.replace('/', '_')}_{algo}_train_min{train_min}_max{train_max}_test_min{test_min}_max{test_max}"
    )

    print(f"🎥 Recording to {video_folder}")

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1} | Reward: {total_reward:.2f}")

    print(f"📊 Mean reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")

    env.close()
    print(f"✅ Finished. Video: {video_folder}")
    return rewards