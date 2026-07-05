"""Record demo gifs for the slides (headless, rgb_array -> gif).

Records the model-based baseline (always works: PD for Reach / scripted for
manipulation) plus any trained policies whose paths are given.

Usage:
    python scripts/record_demo.py --task FetchReach-v4 \
        --sac results/benchmark/sac/model.zip --residual results/residual/model.zip
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc.common.envs import make_env, DEFAULT_TASK
from mc.common.eval import model_policy
from mc.common.video import record_rollout
from mc.controllers import ScriptedPickPlace, TaskSpacePID
from mc.controllers.residual import ResidualController


def model_based_base(task: str):
    if "Reach" in task:
        return TaskSpacePID(kp=8.0, ki=0.0, kd=1.0, gripper=0.0)
    return ScriptedPickPlace()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--out-dir", default="results/gifs")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--sac", default=None)
    p.add_argument("--ppo", default=None)
    p.add_argument("--residual", default=None)
    p.add_argument("--residual-scale", type=float, default=0.3)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(args.task, reward_type="sparse", render_mode="rgb_array")
    reset_on_success = "Reach" not in args.task  # Reach episodes are short; keep them

    base = model_based_base(args.task)
    tag = "pd" if "Reach" in args.task else "scripted"
    record_rollout(env, base.act, os.path.join(args.out_dir, f"model_based_{tag}.gif"),
                   n_episodes=args.episodes, max_steps=args.max_steps,
                   reset_on_success=reset_on_success)

    if args.sac and os.path.exists(args.sac):
        from stable_baselines3 import SAC
        model = SAC.load(args.sac, env=make_env(args.task, reward_type="sparse"))
        record_rollout(env, model_policy(model), os.path.join(args.out_dir, "sac.gif"),
                       n_episodes=args.episodes, max_steps=args.max_steps,
                       reset_on_success=reset_on_success)

    if args.ppo and os.path.exists(args.ppo):
        from stable_baselines3 import PPO
        model = PPO.load(args.ppo)
        record_rollout(env, model_policy(model), os.path.join(args.out_dir, "ppo.gif"),
                       n_episodes=args.episodes, max_steps=args.max_steps,
                       reset_on_success=reset_on_success)

    if args.residual and os.path.exists(args.residual):
        from stable_baselines3 import SAC
        model = SAC.load(args.residual, env=make_env(args.task, reward_type="sparse"))
        ctrl = ResidualController(model_based_base(args.task), model, args.residual_scale)
        record_rollout(env, ctrl.act, os.path.join(args.out_dir, "residual.gif"),
                       n_episodes=args.episodes, max_steps=args.max_steps,
                       reset_on_success=reset_on_success)
    env.close()


if __name__ == "__main__":
    main()
