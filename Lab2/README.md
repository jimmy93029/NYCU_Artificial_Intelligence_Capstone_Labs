# NYCU AI Capstone Lab2: BipedalWalker-v3 Baseline Comparison

This repository provides baseline scripts and configuration for **Reinforcement Learning (RL)** and **Imitation Learning (IL)** on the BipedalWalker-v3 environment. Both **action-constrained** and **unconstrained (free)** settings are supported.

---

## Algorithms Overview

### Model-Free RL
- **A2C (Advantage Actor-Critic)**: A synchronous actor-critic RL algorithm that improves training stability and efficiency.
- **PPO (Proximal Policy Optimization)**: A popular RL method using clipped objectives for robust and stable training.
- **SAC (Soft Actor-Critic)**: An off-policy RL method for continuous control, leveraging entropy regularization for automatic exploration-exploitation balance.

### Model-Based RL
- **PETS (Probabilistic Ensembles with Trajectory Sampling)**: Combines an ensemble of dynamics models with trajectory sampling for data-efficient learning and robust planning.

### Imitation Learning
- **GAIL (Generative Adversarial Imitation Learning)**: Uses adversarial training to imitate expert trajectories without requiring an explicit reward function.

---

## How to Run

Go to the `Lab2/` directory. Each baseline has ready-to-use shell scripts. Choose **free** (full action range) or **constraint** (restricted action range) scripts as needed.

### Example Usage

- **Model-Free Baselines**
    - **Unconstrained (Free) Training & Evaluation**
      ```bash
      bash execute/Bipedal_Walker/run_A2C_free.sh
      bash execute/Bipedal_Walker/run_PPO_free.sh
      bash execute/Bipedal_Walker/run_SAC_free.sh
      ```
    - **Constrained Training & Evaluation**
      ```bash
      bash execute/Bipedal_Walker/run_A2C_constraint.sh
      bash execute/Bipedal_Walker/run_PPO_constraint.sh
      bash execute/Bipedal_Walker/run_SAC_constraint.sh
      ```

- **Model-Based Baseline**
    - **PETS**
      ```bash
      bash execute/Bipedal_Walker/run_PETS_free.sh
      bash execute/Bipedal_Walker/run_PETS_constraint.sh
      ```

- **Imitation Learning Baseline**
    - **GAIL**
      ```bash
      bash execute/Bipedal_Walker/run_GAIL_free.sh
      bash execute/Bipedal_Walker/run_GAIL_constraint.sh
      ```

---

## What’s the Difference? Constraint vs. Free

- **Free**: Training and testing are both performed with the full action range allowed by the environment (`[-1, 1]`).
- **Constraint**: Training and/or testing are performed with a narrower action range (e.g., `[-0.8, 0.8]`). This setup is used to test generalization and robustness when the agent faces restricted actions at test time.

**Example comparison:**  
In the *free* setting, agents can utilize the entire range of available actions, typically achieving higher rewards and better performance if well trained. In the *constraint* setting, agents are either trained and/or evaluated with restricted action bounds, making the control problem more challenging and often resulting in lower scores. Comparing these two setups allows us to evaluate how well an agent adapts to limited action flexibility.


---

## Output and Logging

- All logs, models, and results are saved under `info/` or `info/model_based/` directories automatically.
- Evaluation metrics (mean rewards) are printed to the terminal and stored in log files at the end of each script.

---

## Notes

- Ensure dependencies are installed, especially `moviepy` for video recording.
- You can modify the shell scripts to change training steps, action bounds, or any other parameters as needed.
- For more details, refer to each `.sh` script and the corresponding YAML configuration files.

---
