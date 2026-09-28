"""
Continuous control in MuJoCo with SAC.

Start with balancing (InvertedPendulum-v5, ~50k steps, minutes on CPU),
then move to locomotion (Hopper-v5 or Walker2d-v5, ~1M steps, hours).

Install:
    pip install "stable-baselines3[extra]>=2.3.0" "gymnasium[mujoco]" matplotlib

Run:
    python train_mujoco.py --env InvertedPendulum-v5 --steps 50000
    python train_mujoco.py --env Hopper-v5 --steps 1000000
    python train_mujoco.py --env Hopper-v5 --watch
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy


def moving_average(values, window):
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_learning_curve(log_dir, env_id, window=20):
    x, y = ts2xy(load_results(log_dir), "timesteps")
    if len(y) < window:
        print(f"Only {len(y)} episodes logged, skipping the plot")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, alpha=0.25, label="episode return")
    plt.plot(x[window - 1:], moving_average(y, window), linewidth=2,
             label=f"moving average ({window})")
    plt.xlabel("environment steps")
    plt.ylabel("episode return")
    plt.title(f"SAC on {env_id}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = f"{env_id.lower()}_learning_curve.png"
    plt.savefig(out_path, dpi=130)
    print(f"Learning curve saved to {out_path}")


def train(env_id, total_timesteps, seed):
    log_dir = f"logs/{env_id}"
    model_dir = f"models/{env_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Single env is fine for SAC: it is off-policy and learns from a replay
    # buffer, so extra parallel envs help much less than they do for PPO.
    train_env = Monitor(gym.make(env_id), filename=os.path.join(log_dir, "train"))
    eval_env = Monitor(gym.make(env_id))

    # Periodic evaluation with the deterministic policy. The best checkpoint by
    # eval return is kept, which matters because RL performance is not monotone.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # SAC: off-policy actor-critic for continuous actions, maximizes reward plus
    # policy entropy. It is the default baseline for MuJoCo style locomotion
    # because it needs far fewer environment steps than PPO.
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,   # collect random data before the first update
        batch_size=256,
        tau=0.005,                # target network soft update rate
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",          # entropy temperature is tuned automatically
        seed=seed,
        verbose=1,
        tensorboard_log="logs/tb",
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(model_dir, "final")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"Final policy: {mean_r:.1f} +/- {std_r:.1f}")

    plot_learning_curve(log_dir, env_id)

    train_env.close()
    eval_env.close()


def watch(env_id, n_episodes=3):
    """Render the best saved checkpoint."""
    model_path = os.path.join(f"models/{env_id}", "best_model")
    env = gym.make(env_id, render_mode="human")
    model = SAC.load(model_path)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: return = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="InvertedPendulum-v5")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.watch:
        watch(args.env)
    else:
        train(args.env, args.steps, args.seed)
