import os
import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from mbrl.algorithms import pets, planet
from mbrl.util.env import EnvHandler
from mbrl.util.video import save_video
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TransformObservation, ResizeObservation, RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


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


def preprocess_car_racing(env):
    import cv2
    env = TransformObservation(env, lambda obs: cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[..., None])
    env = ResizeObservation(env, shape=(84, 84))
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


def wrap_with_constraints(env, env_id, mini=-1, maxi=1):
    if "CarRacing" in env_id:
        env = preprocess_car_racing(env)
        env = CarRacingSteeringConstraint(env, mini, maxi)
    if "BipedalWalker" in env_id:
        env = BipedalWalkerActionConstraint(env, mini, maxi)
    return env


def make_custom_env(env_id, mini=-1, maxi=1):
    env = gym.make(env_id)
    env = wrap_with_constraints(env, env_id, mini, maxi)
    return Monitor(env)


def make_action_constraint_fn(min_val, max_val):
    def constrain(action):
        return np.clip(action, min_val, max_val)
    return constrain


def evaluate_agent(agent, env, num_episodes=10, video_dir=None, constraint_fn=None):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset()
        done = False
        total_reward = 0.0
        frames = []

        while not done:
            action = agent.act(obs)
            if constraint_fn:
                action = constraint_fn(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if video_dir and ep == 0:
                frames.append(env.render())

        if video_dir and ep == 0:
            os.makedirs(video_dir, exist_ok=True)
            save_video(frames, os.path.join(video_dir, "eval_ep0.mp4"))

        rewards.append(total_reward)
    return float(np.mean(rewards))


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    # Uncomment below to train + eval in one go:
    # agent, model_path = train_mbrl(cfg)
    # evaluate_mbrl(cfg, agent, model_path)
    evaluate_mbrl(cfg)  # Assume we're only testing now


def train_mbrl(cfg: DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    base_dir = "info"
    log_dir = os.path.join(base_dir, "logs")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    env, term_fn, reward_fn = EnvHandler.make_env(cfg)
    agent = None

    if cfg.algorithm.name == "pets":
        agent = pets.train(env, term_fn, reward_fn, cfg)
    elif cfg.algorithm.name == "planet":
        agent = planet.train(env, cfg)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")

    model = getattr(agent, 'dynamics_model', None)
    model_path = None
    if model is not None:
        model_filename = f"{cfg.overrides.env}_{cfg.algorithm.name}_min{cfg.algorithm.agent.action_lb[0]}_max{cfg.algorithm.agent.action_ub[0]}.pth"
        model_path = os.path.join(model_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to: {model_path}")

    final_reward = agent.logger.data[mbrl.constants.RESULTS_LOG_NAME]["episode_reward"][-1] if agent.logger else None
    if final_reward is not None:
        writer.add_scalar("train/final_episode_reward", final_reward, 0)

    print("✅ Training finished.")
    writer.close()
    return agent, model_path


def evaluate_mbrl(cfg: DictConfig, agent=None, model_path=None):
    base_dir = "info"
    model_dir = os.path.join(base_dir, "models")
    video_dir = os.path.join(base_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    env_id = cfg.overrides.env
    min_a = cfg.algorithm.agent.action_lb[0]
    max_a = cfg.algorithm.agent.action_ub[0]

    # Auto-detect model path if not provided
    if model_path is None:
        model_filename = f"{env_id}_{cfg.algorithm.name}_min{min_a}_max{max_a}.pth"
        model_path = os.path.join(model_dir, model_filename)

    # Reload agent and load model weights
    env, term_fn, reward_fn = EnvHandler.make_env(cfg)
    if cfg.algorithm.name == "pets":
        agent = pets.train(env, term_fn, reward_fn, cfg)
    elif cfg.algorithm.name == "planet":
        agent = planet.train(env, cfg)
    else:
        raise ValueError("Only PETS and PlaNet supported for eval.")

    model = getattr(agent, 'dynamics_model', None)
    if model is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded model from {model_path}")

    constraint_fn = make_action_constraint_fn(min_a, max_a)
    test_env = make_custom_env(env_id, min_a, max_a)

    test_env = RecordVideo(
        test_env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep == 0,
        name_prefix=f"{env_id}_eval"
    )

    avg_reward = evaluate_agent(
        agent,
        test_env,
        num_episodes=10,
        video_dir=video_dir if cfg.save_video else None,
        constraint_fn=constraint_fn,
    )
    print(f"✅ Evaluation finished. Avg reward = {avg_reward:.2f}")
