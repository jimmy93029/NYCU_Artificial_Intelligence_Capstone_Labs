from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.util.logger import configure as imitation_configure
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from MakeEnv import make_sb3_env
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
import os


Base_dir = "info"

def train_expert(env_id, mini=-1, maxi=1, total_timesteps=100_000):
    env_fn = lambda: gym.make(env_id)
    vec_env = make_sb3_env(env_id, mini, maxi)

    # === Train PPO expert ===
    expert = PPO("MlpPolicy", vec_env, verbose=1)
    expert.learn(total_timesteps=total_timesteps)  # longer for stability

    # === Generate Expert Trajectories ===
    sample_until = rollout.make_sample_until(min_timesteps=10000, min_episodes=10)
    trajectories = rollout.rollout(expert, vec_env, sample_until=sample_until)

    # === Convert to Transitions ===
    transitions = rollout.flatten_trajectories(trajectories)

    # === Save Expert Transitions ===
    os.makedirs("expert_data", exist_ok=True)
    with open("expert_data/bipedalwalker_transitions.pkl", "wb") as f:
        pickle.dump(transitions, f)

    print("✅ Expert transitions saved.")


def train_imitation(algo_name, env_id, expert_transitions: Transitions, total_timesteps=100_000, mini=-1, maxi=1):
    assert algo_name in ["BC", "GAIL"], "Only BC or GAIL supported"
    env = make_sb3_env(env_id, mini, maxi)

    log_dir = f"{Base_dir}/logs/imitation/{algo_name}_min{mini}_max{maxi}"
    os.makedirs(log_dir, exist_ok=True)
    logger = imitation_configure(log_dir, ["stdout", "tensorboard", "csv"])

    if algo_name == "BC":
        trainer = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=np.random.default_rng(),
            logger=logger,
        )
        trainer.train(n_epochs=10)

    elif algo_name == "GAIL":
        policy = PPO("MlpPolicy", env, verbose=1)
        trainer = GAIL(
            venv=env,
            demonstrations=expert_transitions,
            gen_algo=policy,
            demo_batch_size=128,
            n_disc_updates_per_round=4,
            disc_batch_size=256,
            init_tensorboard=True,
            log_dir=log_dir,
            allow_variable_horizon=True,
        )
        trainer.train(total_timesteps)

    model_path = f"{Base_dir}/models/imitation_{algo_name}_min{mini}_max{maxi}.zip"
    trainer.policy.save(model_path)
    print(f"✅ Imitation model saved to: {model_path}")
    print(f"📁 Logs stored in: {log_dir}")


def evaluate_imitation(env_id, algo_name, train_min=-1, train_max=1, test_min=-1, test_max=1, episodes=10):
    # Auto-resolve model path
    model_path = (
        f"{Base_dir}/models/imitation_{algo_name}_{env_id}_"
        f"min{train_min}_max{train_max}.zip"
    )
    model = PPO.load(model_path)

    # Define video folder: based only on train_min and test_max
    video_folder = (
        f"{Base_dir}/videos/{env_id}/train{train_min}_test{test_max}"
    )
    os.makedirs(video_folder, exist_ok=True)

    # Define video filename: include full config
    video_name_prefix = (
        f"{algo_name}_{env_id}_"
        f"train_min{train_min}_max{train_max}_"
        f"test_min{test_min}_max{test_max}"
    )

    env = gym.make(env_id, render_mode="rgb_array")
    env = wrap_with_constraints(env, env_id, test_min, test_max)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep == 0,
        name_prefix=video_name_prefix
    )

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

    print(f"📊 Mean reward: {np.mean(rewards):.2f}")
    print(f"🎥 Video saved to folder: {video_folder}")
    env.close()
    return rewards

