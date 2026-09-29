"""
Record a trained policy to video, for the README.

Writes an mp4 at the environment's own frame rate and a smaller looping GIF,
since GitHub renders a GIF inline but only links an mp4.

Install:
    pip install imageio imageio-ffmpeg pillow

Examples:
    python record_policy.py --env Hopper-v5 --arm sac
    python record_policy.py --env Walker2d-v5 --arm sac --seed 2 --episodes 2
    python record_policy.py --env Hopper-v5 --arm ppo --gif-width 480
"""

import argparse
import glob
import os

import numpy as np
import imageio.v2 as imageio
from PIL import Image

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ALGO_CLASSES = {"ppo": PPO, "sac": SAC}


def algo_of(arm):
    """Arm names may carry an ablation suffix, as in sac_ls1000."""
    base = arm.split("_")[0]
    if base not in ALGO_CLASSES:
        raise SystemExit(f"Cannot tell which algorithm '{arm}' is. "
                         f"Expected a name starting with one of {sorted(ALGO_CLASSES)}.")
    return base


def best_seed(results_dir, env_id, arm, window=10):
    """Seed whose final evaluation score is highest.

    Picking the strongest seed is fine for an illustration and misleading as a
    performance claim, so this is only ever used for video. Numbers reported
    anywhere else aggregate across all seeds.
    """
    scores = {}
    for path in sorted(glob.glob(os.path.join(
            results_dir, env_id, arm, "seed_*", "evaluations.npz"))):
        seed = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        curve = np.load(path)["results"].mean(axis=1)
        scores[seed] = curve[-window:].mean()
    if not scores:
        raise SystemExit(f"No runs found under {results_dir}/{env_id}/{arm}/")
    chosen = max(scores, key=scores.get)
    print(f"Seeds and final scores: "
          f"{', '.join(f'{s}={v:.0f}' for s, v in sorted(scores.items()))}")
    print(f"Recording seed {chosen}")
    return chosen


def load_policy(results_dir, env_id, arm, seed):
    """Load the best checkpoint, plus PPO's normalization statistics."""
    run_dir = os.path.join(results_dir, env_id, arm, f"seed_{seed}")

    checkpoint = None
    for name in ("best_model", "final_model"):
        if os.path.exists(os.path.join(run_dir, f"{name}.zip")):
            checkpoint = os.path.join(run_dir, name)
            break
    if checkpoint is None:
        raise SystemExit(f"No best_model.zip or final_model.zip in {run_dir}")
    print(f"Loading {checkpoint}.zip")

    model = ALGO_CLASSES[algo_of(arm)].load(checkpoint)

    # PPO was trained on normalized observations, so the same statistics have to
    # be applied at inference. Without this the policy sees inputs on a scale it
    # never saw in training and the recording shows a far worse agent than the
    # evaluation logs report.
    normalizer = None
    stats_path = os.path.join(run_dir, "vecnormalize.pkl")
    if os.path.exists(stats_path):
        normalizer = VecNormalize.load(
            stats_path, DummyVecEnv([lambda: gym.make(env_id)]))
        normalizer.training = False
        normalizer.norm_reward = False
        print("Applied saved observation normalization")

    return model, normalizer


def rollout(env_id, model, normalizer, episodes, seed):
    """Run episodes and return the frames and the return of each."""
    env = gym.make(env_id, render_mode="rgb_array")
    fps = env.metadata.get("render_fps", 30)

    frames, returns = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done, total = False, 0.0
        while not done:
            frames.append(env.render())
            model_obs = normalizer.normalize_obs(obs) if normalizer else obs
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        print(f"Episode {episode + 1}: return {total:.1f}, "
              f"{len(frames)} frames so far")

    env.close()
    return frames, returns, fps


def to_gif(frames, source_fps, path, target_fps=25, width=320, max_frames=150):
    """Subsample and downscale, because a full rate GIF is enormous.

    MuJoCo renders at 125 frames per second, so an untouched episode would be a
    thousand frames of 480 pixels. Browsers also clamp very high GIF rates, so
    the result would play at the wrong speed as well as being large.
    """
    stride = max(1, round(source_fps / target_fps))
    selected = frames[::stride][:max_frames]

    scaled = []
    for frame in selected:
        image = Image.fromarray(frame)
        height = round(image.height * width / image.width)
        scaled.append(np.asarray(image.resize((width, height), Image.LANCZOS)))

    imageio.mimsave(path, scaled, fps=source_fps / stride, loop=0)
    size_mb = os.path.getsize(path) / 1e6
    print(f"GIF: {path}  {len(scaled)} frames, {width}px, {size_mb:.1f} MB")
    if size_mb > 5:
        print("Over 5 MB. Lower --gif-width or --gif-max-frames before committing it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--arm", default="sac",
                        help="results folder name, such as sac or sac_ls1000")
    parser.add_argument("--seed", type=int, default=None,
                        help="default: the seed with the highest final score")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="videos")
    parser.add_argument("--gif-width", type=int, default=320)
    parser.add_argument("--gif-fps", type=int, default=25)
    parser.add_argument("--gif-max-frames", type=int, default=150)
    parser.add_argument("--no-mp4", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed = args.seed if args.seed is not None else best_seed(
        args.results, args.env, args.arm)

    model, normalizer = load_policy(args.results, args.env, args.arm, seed)
    # Offset so the recorded episodes are not the ones used for evaluation.
    frames, returns, fps = rollout(args.env, model, normalizer,
                                   args.episodes, seed=seed + 20_000)

    print(f"Mean return over {len(returns)} episode(s): {np.mean(returns):.1f}")

    stem = os.path.join(args.out, f"{args.env.lower()}_{args.arm}_seed{seed}")
    if not args.no_mp4:
        mp4 = f"{stem}.mp4"
        imageio.mimsave(mp4, frames, fps=fps, macro_block_size=1)
        print(f"MP4: {mp4}  {len(frames)} frames at {fps} fps")

    to_gif(frames, fps, f"{stem}.gif", target_fps=args.gif_fps,
           width=args.gif_width, max_frames=args.gif_max_frames)


if __name__ == "__main__":
    main()
