import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from mbrl.algorithms import pets, planet
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
from MakeEnv import make_mbrl_env, make_action_constraint_fn, bipedalwalker_reward_fn, bipedalwalker_termination_fn
import pandas as pd
import mbrl


def evaluate_agent(agent, env, num_episodes=10, constraint_fn=None):
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(obs)
            if constraint_fn:
                action = constraint_fn(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
    return float(np.mean(rewards))

def replay_csv_to_tensorboard(csv_path, writer, prefix="train"):
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        for col in df.columns:
            if col != "step":
                writer.add_scalar(f"{prefix}/{col}", row[col], row["step"])

def train_mbrl(cfg: DictConfig):
    print("inside----------------")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log_dir = os.getcwd()  # Hydra sets working dir automatically
    print(f"log dir = {log_dir}")
    model_dir = os.path.join(log_dir, "info")
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    env = make_mbrl_env(cfg.env_id, cfg.train_min, cfg.train_max)
    term_fn = bipedalwalker_termination_fn
    reward_fn = bipedalwalker_reward_fn

    if cfg.algorithm.name == "pets":
        agent = pets.train(env, term_fn, reward_fn, cfg, work_dir=log_dir)
    elif cfg.algorithm.name == "planet":
        agent = planet.train(env, cfg)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")

    model = getattr(agent, 'dynamics_model', None)
    if model is not None:
        model.save(model_dir)
        print(f"✅ Model saved to: {model_dir}/model.pth")

    csv_path = os.path.join(log_dir, "train.csv")
    if os.path.exists(csv_path):
        replay_csv_to_tensorboard(csv_path, writer)

    print("✅ Training finished.")
    writer.close()
    return agent, model_dir

def evaluate_mbrl(cfg: DictConfig):
    log_dir = os.getcwd()
    model_dir = os.path.join(log_dir, "info")
    model_path = os.path.join(model_dir, "model.pth")

    env = make_mbrl_env(cfg.env_id, cfg.test_min, cfg.test_max)
    term_fn = bipedalwalker_termination_fn
    reward_fn = bipedalwalker_reward_fn

    # === Recreate agent (w/ dynamics model loaded) ===
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    dynamics_model.load(model_dir)

    model_env = mbrl.models.ModelEnv(env, dynamics_model, term_fn, reward_fn)
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    constraint_fn = make_action_constraint_fn(cfg.test_min, cfg.test_max)

    video_name = f"{cfg.env_id}_eval_trainmin{cfg.train_min}_trainmax{cfg.train_max}_testmin{cfg.test_min}_testmax{cfg.test_max}"
    video_dir = os.path.join(log_dir, "videos", video_name)
    os.makedirs(video_dir, exist_ok=True)

    test_env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep == 0,
        name_prefix=video_name
    )

    avg_reward = evaluate_agent(
        agent,
        test_env,
        num_episodes=10,
        constraint_fn=constraint_fn,
    )
    print(f"✅ Evaluation finished. Avg reward = {avg_reward:.2f}")

@hydra.main(config_path="conf", config_name="main")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train_mbrl(cfg)
    elif cfg.mode == "test":
        evaluate_mbrl(cfg)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

if __name__ == "__main__":
    main()
