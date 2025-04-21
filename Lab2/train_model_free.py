import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation, RecordVideo, TransformObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO, A2C, SAC
from MakeEnv import make_sb3_env, wrap_with_constraints, bipedalwalker_termination_fn
import os

Base_dir = "info"

# === Training Function ===
def train_sb3(algo, model_path, env_id, total_timesteps=100_000, mini=-1, maxi=1):
    policy = "CnnPolicy" if "ALE" in env_id or "CarRacing" in env_id else "MlpPolicy"
    algo_name = algo.__name__

    log_dir = f"{Base_dir}/logs/{env_id.replace('/', '_')}_{algo_name}_min{mini}_max{maxi}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_sb3_env(env_id=env_id, mini=mini, maxi=maxi)

    model = algo(policy, env, verbose=1)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    model.learn(total_timesteps=total_timesteps)

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


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Reacher-v2")
    parser.add_argument("--algo", type=str, default="PPO")  # PPO, A2C, SAC, etc.
    parser.add_argument("--train_min", type=float, default=-1)
    parser.add_argument("--train_max", type=float, default=1)
    parser.add_argument("--test_min", type=float, default=None)
    parser.add_argument("--test_max", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    # Default test range = training range if not set
    test_min = args.test_min if args.test_min is not None else args.train_min
    test_max = args.test_max if args.test_max is not None else args.train_max

    # Dynamically load SB3 algorithm
    SB3 = importlib.import_module("stable_baselines3")
    algo_class = getattr(SB3, args.algo)

    model_path = f"{Base_dir}/models/{args.env.replace('/', '_')}_{args.algo}_min{args.train_min}_max{args.train_max}.zip"

    # === Train ===
    if args.mode in ["train", "both"]:
        train_sb3(
            algo_class,
            model_path=model_path,
            env_id=args.env,
            total_timesteps=args.timesteps,
            mini=args.train_min,
            maxi=args.train_max
        )

    # === Test ===
    if args.mode in ["test", "both"]:
        evaluate_sb3(
            algo_class,
            model_path=model_path,
            env_id=args.env,
            train_min=args.train_min,
            train_max=args.train_max,
            test_min=test_min,
            test_max=test_max,
        )
