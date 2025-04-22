from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from imitation.data.rollout import generate_trajectories
from imitation.util.logger import configure as imitation_configure
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.data import rollout
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from MakeEnv import make_imitation_env, make_sb3_env, wrap_with_constraints
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
import pickle
import os


Base_dir = "info"


def train_expert(env_id, mini=-1, maxi=1, total_timesteps=500_000):
    # === Paths ===
    model_path = f"{Base_dir}/models/expert_{env_id}_PPO_min{mini}_max{maxi}.zip"
    transition_path = f"expert_data/{env_id}_transitions.pkl"

    # === Environment Setup ===
    vec_env = make_sb3_env(env_id, mini, maxi)
    rng = np.random.default_rng()

    # === Train PPO Expert ===
    if not os.path.exists(model_path):
        print("Make one PPO")
        print(f"model path = {model_path}")
        expert = PPO("MlpPolicy", vec_env, verbose=1)
        expert.learn(total_timesteps=total_timesteps)
        expert.save(model_path)
        print(f"✅ Expert model saved to: {model_path}")
    else:
        print(f"⏩ Skipping training: model already exists for {env_id}")
        expert = PPO.load(model_path, env=vec_env)

    # === Generate Expert Trajectories ===
    if not os.path.exists(transition_path):
        sample_until = rollout.make_sample_until(min_timesteps=10000, min_episodes=10)
        trajectories = generate_trajectories(
            policy=expert,
            venv=vec_env,
            sample_until=sample_until,
            rng=rng,
        )
        transitions = rollout.flatten_trajectories(trajectories)

        # === Save Transitions ===
        os.makedirs("expert_data", exist_ok=True)
        with open(transition_path, "wb") as f:
            pickle.dump(transitions, f)
        print(f"✅ Expert transitions saved to: {transition_path}")
    else:
        print(f"⏩ Skipping training: Transitions already exist for {env_id}")


def train_imitation(algo_name, env_id, expert_transitions: Transitions, total_timesteps=100_000, mini=-1, maxi=1):
    assert algo_name in ["BC", "GAIL"], "Only BC or GAIL supported"
    env = make_imitation_env(env_id, mini, maxi)

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
        trainer.train(n_epochs=total_timesteps)
        model_to_save = trainer.policy

    elif algo_name == "GAIL":
        policy = PPO("MlpPolicy", env, verbose=1)

        # === Create reward network ===
        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
            normalize_input_layer=None,  # You can enable this later if needed
        )

        trainer = GAIL(
            venv=env,
            demonstrations=expert_transitions,
            gen_algo=policy,
            reward_net=reward_net,
            demo_batch_size=128,
            n_disc_updates_per_round=4,
            init_tensorboard=True,
            log_dir=log_dir,
            allow_variable_horizon=True,
        )
        trainer.train(total_timesteps)
        model_to_save = trainer.gen_algo 

    model_path = f"{Base_dir}/models/imitation_{algo_name}_{env_id}_min{mini}_max{maxi}.zip"
    model_to_save.save(model_path)
    print(f"✅ Imitation model saved to: {model_path}")
    print(f"📁 Logs stored in: {log_dir}")


def evaluate_imitation(env_id, algo_name, train_min=-1, train_max=1, test_min=-1, test_max=1, episodes=10):
    from stable_baselines3.common.policies import ActorCriticPolicy

    # Auto-resolve model path
    model_path = (
        f"{Base_dir}/models/imitation_{algo_name}_{env_id}_"
        f"min{train_min}_max{train_max}.zip"
    )

    # === Load model ===
    if algo_name == "GAIL":
        model = PPO.load(model_path)
    elif algo_name == "BC":
        from stable_baselines3.common import policies
        # Rebuild policy with same architecture used during training
        env_tmp = make_imitation_env(env_id, test_min, test_max)
        policy = ActorCriticPolicy(env_tmp.observation_space, env_tmp.action_space)
        policy.load(model_path)
        model = policy  # Policy-only model
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    # === Setup evaluation env + video ===
    video_folder = f"{Base_dir}/videos/{env_id}/train{train_min}_test{test_max}"
    os.makedirs(video_folder, exist_ok=True)
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
            if algo_name == "GAIL":
                action, _ = model.predict(obs)
            else:  # BC policy: use policy.forward()
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1} | Reward: {total_reward:.2f}")

    print(f"📊 Mean reward: {np.mean(rewards):.2f}")
    print(f"🎥 Video saved to folder: {video_folder}")
    env.close()
    return rewards
1


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True, help="Gym environment ID")
    parser.add_argument("--algo", type=str, choices=["BC", "GAIL"], help="Imitation algorithm")
    parser.add_argument("--mode", type=str, choices=["train_expert", "train_imitation", "evaluate"], required=True)
    parser.add_argument("--train_min", type=float, default=-1.0)
    parser.add_argument("--train_max", type=float, default=1.0)
    parser.add_argument("--test_min", type=float, default=-1.0)
    parser.add_argument("--test_max", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "train_expert":
        train_expert(args.env_id, mini=args.train_min, maxi=args.train_max, total_timesteps=args.timesteps)

    elif args.mode == "train_imitation":
        expert_data_path = f"expert_data/{args.env_id}_transitions.pkl"
        with open(expert_data_path, "rb") as f:
            expert_transitions = pickle.load(f)
        train_imitation(args.algo, args.env_id, expert_transitions, total_timesteps=args.timesteps,
                        mini=args.train_min, maxi=args.train_max)

    elif args.mode == "evaluate":
        evaluate_imitation(args.env_id, args.algo,
                           train_min=args.train_min, train_max=args.train_max,
                           test_min=args.test_min, test_max=args.test_max,
                           episodes=args.episodes)
