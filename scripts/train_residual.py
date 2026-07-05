"""🅰  Train Residual RL: scripted base controller + learned SAC residual (+HER).

The agent only learns a small correction on top of the scripted controller, so
it starts from a competent policy and converges far faster than learning from
scratch. Saves the residual model + a success-rate curve.

Usage:
    python scripts/train_residual.py --timesteps 40000 --residual-scale 0.3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc.common.envs import make_env, DEFAULT_TASK
from mc.common.training import build_model, make_vec_env
from mc.common.callbacks import SuccessRateEvalCallback, CurveLoggerCallback
from mc.common.plotting import plot_training_curves
from mc.controllers import base_for_task
from mc.controllers.residual import ResidualRLWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--timesteps", type=int, default=40000)
    p.add_argument("--residual-scale", type=float, default=0.3)
    p.add_argument("--eval-freq", type=int, default=2000)
    p.add_argument("--out-dir", default="results/residual")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1,
                   help="parallel envs for sample collection (SubprocVecEnv)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def make_residual(scale, seed=args.seed):
        e = make_env(args.task, reward_type="sparse")
        return ResidualRLWrapper(e, base_for_task(args.task), residual_scale=scale)

    train_env = make_vec_env(lambda r: make_residual(args.residual_scale, seed=(args.seed + r)),
                             n_envs=args.n_envs, seed=args.seed)
    grad_steps = -1 if args.n_envs > 1 else 1
    learning_starts = max(100, 61 * args.n_envs)
    model = build_model("SAC", train_env, use_her=True, seed=args.seed, verbose=0,
                        gradient_steps=grad_steps, learning_starts=learning_starts)

    succ_cb = SuccessRateEvalCallback(
        lambda: make_residual(args.residual_scale, seed=args.seed + 777),
        eval_freq=args.eval_freq, n_eval_episodes=20, max_steps=60,
        csv_path=os.path.join(args.out_dir, "success_curve.csv"),
    )
    curve_cb = CurveLoggerCallback(log_freq=max(500, args.eval_freq // 2),
                                   csv_path=os.path.join(args.out_dir, "train_curve.csv"))
    print(f"===== training Residual RL (scale={args.residual_scale}) for "
          f"{args.timesteps} steps =====")
    model.learn(total_timesteps=args.timesteps, callback=[succ_cb, curve_cb])
    model.save(os.path.join(args.out_dir, "model"))
    train_env.close()
    plot_training_curves(os.path.join(args.out_dir, "train_curve.csv"),
                         os.path.join(args.out_dir, "training_curves.png"),
                         title=f"Residual RL training curves ({args.task})")
    print(f"saved residual model to {os.path.join(args.out_dir, 'model.zip')}")


if __name__ == "__main__":
    main()
