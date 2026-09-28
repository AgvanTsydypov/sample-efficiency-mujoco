"""
Experiment runner: PPO vs SAC on a MuJoCo task, one (algo, seed) run per call.

Install:
    pip install "stable-baselines3[extra]>=2.3.0" "gymnasium[mujoco]" matplotlib

Single run:
    python run_experiment.py --algo sac --seed 0 --env Hopper-v5 --steps 1000000

All six runs in parallel (adjust to your core count):
    for algo in ppo sac; do
      for seed in 0 1 2; do
        python run_experiment.py --algo $algo --seed $seed &
      done
    done
    wait

Results land in results/<env>/<algo>/seed_<n>/evaluations.npz
"""

import argparse
import json
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Hyperparameters taken from rl-baselines3-zoo tuned configs for MuJoCo
# locomotion. Using published values for both algorithms keeps the comparison
# honest: neither side is tuned by hand for this specific study.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=9.8e-5,
    n_steps=512,
    batch_size=32,
    n_epochs=5,
    gamma=0.999,
    gae_lambda=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.19,
    max_grad_norm=0.7,
)

SAC_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
)

# PPO needs running normalization of observations and rewards on MuJoCo.
# SAC does not, and normalizing rewards would interfere with its entropy
# temperature tuning.
NEEDS_NORMALIZATION = {"ppo": True, "sac": False}

EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10


def build_envs(env_id, seed, algo, run_dir):
    """Create training and evaluation envs, normalized if the algo needs it."""
    train_env = make_vec_env(
        env_id, n_envs=1, seed=seed, monitor_dir=os.path.join(run_dir, "monitor")
    )
    # Different seed offset so evaluation episodes are not the same rollouts
    # the agent was trained on.
    eval_env = make_vec_env(env_id, n_envs=1, seed=seed + 10_000)

    if NEEDS_NORMALIZATION[algo]:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        # training=False freezes the statistics, norm_reward=False keeps the
        # reported return on the raw scale so it is comparable with SAC.
        # EvalCallback copies the training statistics over before each eval.
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
        )

    return train_env, eval_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    run_dir = os.path.join(args.out, args.env, args.algo, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    train_env, eval_env = build_envs(args.env, args.seed, args.algo, run_dir)

    # Periodic evaluation with the deterministic policy. Writes evaluations.npz
    # with timesteps and per-episode returns. This file, not the training
    # rewards, is what the aggregation script reads: training returns come from
    # a stochastic exploring policy and are not a clean performance measure.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    algo_cls = {"ppo": PPO, "sac": SAC}[args.algo]
    algo_kwargs = {"ppo": PPO_KWARGS, "sac": SAC_KWARGS}[args.algo]

    model = algo_cls(env=train_env, seed=args.seed, verbose=1, **algo_kwargs)
    model.learn(total_timesteps=args.steps, callback=eval_callback, progress_bar=False)

    model.save(os.path.join(run_dir, "final_model"))
    if NEEDS_NORMALIZATION[args.algo]:
        # Normalization statistics are part of the policy. Without them the
        # saved model cannot be reloaded and reproduced.
        train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    # Record everything needed to rerun this exact experiment.
    config = {
        "algo": args.algo,
        "env": args.env,
        "seed": args.seed,
        "total_timesteps": args.steps,
        "eval_freq": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
        "hyperparameters": {k: str(v) for k, v in algo_kwargs.items()},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_env.close()
    eval_env.close()
    print(f"Done: {args.algo} seed {args.seed} -> {run_dir}")


if __name__ == "__main__":
    main()
