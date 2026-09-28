"""
First RL agent: PPO on CartPole-v1 with Stable-Baselines3 + Gymnasium.

Install:
    pip install "stable-baselines3[extra]>=2.3.0" "gymnasium[classic-control]" matplotlib

Run:
    python train_cartpole.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy

ENV_ID = "CartPole-v1"
LOG_DIR = "logs/cartpole"
MODEL_PATH = "models/ppo_cartpole"
TOTAL_TIMESTEPS = 100_000
N_ENVS = 4
SEED = 0


def moving_average(values, window):
    """Simple smoothing so the learning curve is readable."""
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_learning_curve(log_dir, window=20, out_path="cartpole_learning_curve.png"):
    """Read Monitor csv logs and plot episode return vs environment steps."""
    results = load_results(log_dir)
    x, y = ts2xy(results, "timesteps")  # x = cumulative steps, y = episode return

    if len(y) < window:
        print(f"Only {len(y)} episodes logged, skipping the plot")
        return

    y_smooth = moving_average(y, window)
    x_smooth = x[window - 1:]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, alpha=0.25, label="episode return")
    plt.plot(x_smooth, y_smooth, linewidth=2, label=f"moving average ({window})")
    plt.xlabel("environment steps")
    plt.ylabel("episode return")
    plt.title(f"PPO on {ENV_ID}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Learning curve saved to {out_path}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Training environment.
    # make_vec_env runs N_ENVS copies of the env in parallel and wraps each one
    # in Monitor, which writes episode returns and lengths to LOG_DIR/*.monitor.csv.
    # Parallel envs give PPO more diverse data per update, which speeds up training.
    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED, monitor_dir=LOG_DIR)

    # 2. Separate evaluation environment.
    # Never evaluate on the training env: its episode counters and normalization
    # state belong to the training loop.
    eval_env = Monitor(gym.make(ENV_ID))

    # 3. The agent.
    # MlpPolicy = small fully connected net, correct choice for vector observations.
    # CartPole observation is 4 floats, action space is Discrete(2).
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,        # steps collected per env before each update
        batch_size=64,
        n_epochs=10,         # passes over the collected batch
        gamma=0.99,          # discount factor
        gae_lambda=0.95,     # bias/variance tradeoff in advantage estimation
        clip_range=0.2,      # PPO trust region size
        ent_coef=0.0,        # entropy bonus, raise it if the policy collapses early
        seed=SEED,
        verbose=1,
        tensorboard_log="logs/tb",
    )

    # 4. Score before training, as a baseline.
    mean_before, std_before = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Before training: {mean_before:.1f} +/- {std_before:.1f}")

    # 5. Training. One "timestep" is one env step summed over all parallel envs.
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")

    # 6. Score after training.
    # deterministic=True takes the argmax action instead of sampling, which is
    # the standard way to report final performance.
    mean_after, std_after = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"After training:  {mean_after:.1f} +/- {std_after:.1f}  (max possible: 500)")

    # 7. Learning curve from the Monitor logs.
    plot_learning_curve(LOG_DIR)

    train_env.close()
    eval_env.close()


def watch(n_episodes=3):
    """Load a saved model and render it. Call this manually after training."""
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(MODEL_PATH)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: return = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
    # watch()
