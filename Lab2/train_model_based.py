import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from mbrl.algorithms import pets, planet
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
from MakeEnv import make_mbrl_env, make_action_constraint_fn, bipedalwalker_reward_fn, bipedalwalker_termination_fn
from mbrl.env.termination_fns import no_termination
import pandas as pd
import mbrl


def replay_csv_to_tensorboard(csv_path, writer, prefix="train"):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        for col in df.columns:
            if col != "step":
                writer.add_scalar(f"{prefix}/{col}", row[col], row["step"])

def train_mbrl(cfg: DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_mbrl_env(cfg.env_id, cfg.train_min, cfg.train_max)
    term_fn = bipedalwalker_termination_fn
    reward_fn = bipedalwalker_reward_fn

    if cfg.algorithm.name == "pets":
        agent = pets.train(env, term_fn, reward_fn, cfg)
    elif cfg.algorithm.name == "planet":
        agent = planet.train(env, cfg)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")

    csv_path = os.path.join(os.getcwd(), "train.csv")
    if os.path.exists(csv_path):
        writer = SummaryWriter(os.getcwd())
        replay_csv_to_tensorboard(csv_path, writer)
        writer.close()

    print("\u2705 Training finished.")
    return agent

def evaluate_mbrl(cfg: DictConfig):
    log_dir = os.getcwd()
    model_dir = os.path.join(log_dir)

    env = make_mbrl_env(cfg.env_id, cfg.test_min, cfg.test_max)
    term_fn = bipedalwalker_termination_fn
    reward_fn = bipedalwalker_reward_fn
    constraint_fn = make_action_constraint_fn(cfg.test_min, cfg.test_max)

    agent, model = create_agent_and_model_env(cfg, env, term_fn, reward_fn, model_dir)

    video_name = f"{cfg.env_id}_eval_trainmin{cfg.train_min}_trainmax{cfg.train_max}_testmin{cfg.test_min}_testmax{cfg.test_max}"
    video_dir = os.path.join(log_dir, "videos", video_name)
    os.makedirs(video_dir, exist_ok=True)

    test_env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep == 0,
        name_prefix=video_name
    )

    evaluate_agent(agent, test_env, model=model, constraint_fn=constraint_fn)



def evaluate_agent(agent, env, model = None, num_episodes=10, constraint_fn=None):
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        action = None  # initialize to None for update_posterior

        # Reset PlaNet state
        agent.reset()
        if model:
            model.reset_posterior()

        done = False
        total_reward = 0.0

        while not done:
            if model:
                model.update_posterior(obs, action)

            action = agent.act(obs)
            if constraint_fn:
                action = constraint_fn(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"🎬 Episode {ep + 1}: total reward = {total_reward:.2f}")
        rewards.append(total_reward)

    avg_reward = float(np.mean(rewards))
    print(f"✅ Average reward over {num_episodes} episodes: {avg_reward:.2f}")


def create_agent_and_model_env(cfg, env, term_fn, reward_fn, model_dir):
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    if cfg.algorithm.name == "planet":
        cfg.dynamics_model.action_size = act_shape[0]
        model = hydra.utils.instantiate(cfg.dynamics_model)
        assert isinstance(model, mbrl.models.PlaNetModel)
        model_env = mbrl.models.ModelEnv(env, model, no_termination)
        agent = mbrl.planning.create_trajectory_optim_agent_for_model(
            model_env, cfg.algorithm.agent
        )
        return agent, model
    elif cfg.algorithm.name == "pets":
        model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
        model.load(model_dir)
        model_env = mbrl.models.ModelEnv(env, model, term_fn, reward_fn)
        agent = mbrl.planning.create_trajectory_optim_agent_for_model(
            model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
        )
        return agent, None
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")
    

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
