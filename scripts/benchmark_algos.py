"""🅱  Algorithm benchmark: SAC / TD3 / DDPG / PPO on FetchPickAndPlace.

Off-policy algorithms use HER + sparse reward; PPO uses the dense reward. We log
greedy success-rate vs timesteps for each and overlay them into one figure -- a
sample-efficiency comparison at a fixed compute budget.

Usage:
    python scripts/benchmark_algos.py --algos SAC TD3 DDPG PPO --timesteps 60000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc.common.envs import make_env, DEFAULT_TASK
from mc.common.training import build_model, needs_her, make_vec_env
from mc.common.callbacks import SuccessRateEvalCallback, CurveLoggerCallback
from mc.common.plotting import plot_learning_curves, plot_training_curves
from mc.sim2real import PushDisturbanceWrapper


def train_one(algo, timesteps, eval_freq, n_eval, max_steps, out_dir, seed,
              task=DEFAULT_TASK, disturb=False, n_envs=1):
    algo = algo.upper()
    reward_type = "sparse" if needs_her(algo) else "dense"

    def make_train_env(rank=0):
        e = make_env(task, reward_type=reward_type)
        if disturb:  # train with random shoves (disturbance domain randomization)
            e = PushDisturbanceWrapper(e, push_step="random", duration=3, strength=1.0,
                                       seed=(seed or 0) + rank)
        return e

    train_env = make_vec_env(make_train_env, n_envs=n_envs, seed=seed)

    def eval_env_fn():
        return make_env(task, reward_type="sparse")

    # Parallel collection -> keep one gradient update per collected transition.
    grad_steps = -1 if (needs_her(algo) and n_envs > 1) else 1
    learning_starts = max(100, (max_steps + 1) * n_envs)
    model = build_model(algo, train_env, use_her=needs_her(algo), seed=seed,
                        verbose=0, gradient_steps=grad_steps,
                        learning_starts=learning_starts)

    suffix = "_disturb" if disturb else ""
    algo_dir = os.path.join(out_dir, algo.lower() + suffix)
    os.makedirs(algo_dir, exist_ok=True)
    csv_path = os.path.join(algo_dir, "success_curve.csv")
    succ_cb = SuccessRateEvalCallback(
        eval_env_fn, eval_freq=eval_freq, n_eval_episodes=n_eval,
        max_steps=max_steps, csv_path=csv_path,
    )
    curve_cb = CurveLoggerCallback(log_freq=max(500, eval_freq // 2),
                                   csv_path=os.path.join(algo_dir, "train_curve.csv"))
    mode = " +disturbance" if disturb else ""
    print(f"\n===== training {algo} ({'HER+sparse' if needs_her(algo) else 'dense'})"
          f"{mode} on {task} for {timesteps} steps =====")
    model.learn(total_timesteps=timesteps, callback=[succ_cb, curve_cb])
    model.save(os.path.join(algo_dir, "model"))
    train_env.close()
    plot_training_curves(os.path.join(algo_dir, "train_curve.csv"),
                         os.path.join(algo_dir, "training_curves.png"),
                         title=f"{algo} training curves ({task})")
    return csv_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algos", nargs="+", default=["SAC", "TD3", "DDPG", "PPO"])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--timesteps", type=int, default=60000)
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--out-dir", default="results/benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1,
                   help="parallel envs for sample collection (SubprocVecEnv)")
    p.add_argument("--disturb", action="store_true",
                   help="train with random shoves (disturbance domain randomization)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    curves = {}
    for algo in args.algos:
        csv_path = train_one(algo, args.timesteps, args.eval_freq, args.n_eval,
                             args.max_steps, args.out_dir, args.seed,
                             task=args.task, disturb=args.disturb,
                             n_envs=args.n_envs)
        tag = algo.upper() + (" +disturb" if args.disturb else "")
        curves[tag] = csv_path

    plot_learning_curves(
        curves,
        os.path.join(args.out_dir, "algo_benchmark_success.png"),
        title=f"Sample efficiency on {args.task} ({args.timesteps} steps)",
    )


if __name__ == "__main__":
    main()
