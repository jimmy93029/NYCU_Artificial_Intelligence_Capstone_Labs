import ale_py
# if using gymnasium
import shimmy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import os


class CNNREINFORCEPolicy(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 16, 512), nn.ReLU(),
            nn.Linear(512, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute(2, 0, 1).unsqueeze(0) / 255.0
        return self.fc(self.conv(obs)).squeeze(0)


class CNNValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 16, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute(2, 0, 1).unsqueeze(0) / 255.0
        return self.fc(self.conv(obs)).squeeze()


def train_reinforce_variant(env_id, variant="original", num_episodes=500, lr=1e-4, gamma=0.99):
    env = gym.make(env_id)
    policy = CNNREINFORCEPolicy(env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    if variant == "advantage":
        value_net = CNNValueNet()
        value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        log_probs, rewards, states = [], [], []
        done = False

        while not done:
            probs = policy(obs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            states.append(obs)
            obs, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)

        # Compute returns G_t
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Compute loss
        if variant == "baseline":
            baseline = returns.mean()
            advantages = returns - baseline
        elif variant == "advantage":
            values = torch.stack([value_net(s) for s in states])
            advantages = returns - values.detach()

            value_loss = nn.functional.mse_loss(values, returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        else:
            advantages = returns

        loss = -torch.sum(torch.stack(log_probs) * advantages)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[{variant.upper()}] Ep {ep+1}/{num_episodes} | Reward: {sum(rewards):.1f}")

    env.close()
    model_path = f"/content/drive/MyDrive/AI_capstone/{env_id.split('/')[-1]}_{variant}_REINFORCE.pth"
    torch.save(policy.state_dict(), model_path)
    print(f"✅ Saved model: {model_path}")


def evaluate_reinforce(env_id, model_path, episodes=10, variant="original"):
    # Setup video directory in Google Drive
    video_dir = f"/content/drive/MyDrive/AI_capstone/videos/{env_id.replace('/', '_')}_{variant}"
    os.makedirs(video_dir, exist_ok=True)

    # Load policy
    act_dim = gym.make(env_id).action_space.n
    policy = CNNREINFORCEPolicy(act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    # Wrap environment for video
    env = RecordVideo(
        gym.make(env_id, render_mode="rgb_array"),
        video_dir,
        episode_trigger=lambda ep: ep == 0,
        disable_logger=True
    )

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, total_reward, total_len = False, 0, 0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                probs = policy(obs_tensor)
                action = torch.argmax(probs).item()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            total_len += 1

        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1} | Reward: {total_reward:.2f} | Total len = {total_len}")

    env.close()
    print(f"🎥 Video saved to: {video_dir}")
    return rewards

# train_reinforce_variant("ALE/Assault-v5", variant="original", num_episodes=100)
# train_reinforce_variant("ALE/Assault-v5", variant="baseline", num_episodes=100)
# train_reinforce_variant("ALE/Assault-v5", variant="advantage", num_episodes=100)

# rewards_original = evaluate_reinforce(
#     "ALE/Assault-v5",
#     f"{Base_dir}/Assault-v5_original_REINFORCE.pth",
#     variant="original"
# )

# rewards_baseline = evaluate_reinforce(
#     "ALE/Assault-v5",
#     f"{Base_dir}/Assault-v5_baseline_REINFORCE.pth",
#     variant="baseline"
# )

# rewards_advantage = evaluate_reinforce(
#     "ALE/Assault-v5",
#     f"{Base_dir}/Assault-v5_advantage_REINFORCE.pth",
#     variant="advantage"
# )
