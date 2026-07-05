"""🅰  Controller matrix: evaluate classical -> RL -> hybrid on one task.

Evaluates every available controller on FetchPickAndPlace with the same metrics
and produces: a CSV table, a grouped bar chart, an error-vs-time overlay, and a
demo gif. Trained models (pure RL / residual) are included automatically if
their files exist.

Usage:
    python scripts/controller_matrix.py --episodes 50
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc.common.envs import make_env, DEFAULT_TASK
from mc.common.eval import evaluate, model_policy
from mc.common.metrics import summarize
from mc.common.plotting import plot_metric_bars, plot_error_curves
from mc.common.video import record_rollout
from mc.controllers import TaskSpacePID, ScriptedPickPlace, base_for_task
from mc.controllers.residual import ResidualController

METRICS = ["success_rate", "tracking_rmse", "final_error", "control_energy",
           "action_jerk", "path_length"]


def classical_controllers(task):
    """Group 1: the classical baselines (PID / model-based state machine).

    Reach -> a single PD controller. PickAndPlace -> the scripted state machine.
    """
    if "Reach" in task:
        return [("Model-based (PD)", TaskSpacePID(kp=8.0, ki=0.0, kd=1.0, gripper=0.0))]
    return [("Scripted (model-based)", ScriptedPickPlace())]


def build_controllers(task, sac_path, ppo_path, residual_path, residual_scale):
    controllers = list(classical_controllers(task))

    if sac_path and os.path.exists(sac_path):
        from stable_baselines3 import SAC
        model = SAC.load(sac_path, env=make_env(task, reward_type="sparse"))
        controllers.append(("SAC (+HER)", model_policy(model)))
    else:
        print(f"[matrix] no SAC model at {sac_path}, skipping")

    if ppo_path and os.path.exists(ppo_path):
        from stable_baselines3 import PPO
        model = PPO.load(ppo_path)
        controllers.append(("PPO (dense)", model_policy(model)))
    else:
        print(f"[matrix] no PPO model at {ppo_path}, skipping")

    if residual_path and os.path.exists(residual_path):
        from stable_baselines3 import SAC
        model = SAC.load(residual_path, env=make_env(task, reward_type="sparse"))
        ctrl = ResidualController(base_for_task(task), model, residual_scale)
        controllers.append(("Residual RL (hybrid)", ctrl))
    else:
        print(f"[matrix] no residual model at {residual_path}, skipping")
    return controllers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--sac", default="results/benchmark/sac/model.zip")
    p.add_argument("--ppo", default="results/benchmark/ppo/model.zip")
    p.add_argument("--residual", default="results/residual/model.zip")
    p.add_argument("--residual-scale", type=float, default=0.3)
    p.add_argument("--out-dir", default="results/controller_matrix")
    p.add_argument("--gif", action="store_true", help="also record demo gifs")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def eval_env_factory(seed=0, render_mode=None):
        return make_env(args.task, reward_type="sparse", render_mode=render_mode)

    env = eval_env_factory(seed=123)
    controllers = build_controllers(args.task, args.sac, args.ppo,
                                    args.residual, args.residual_scale)

    rows = []
    error_curves = {}
    for name, ctrl in controllers:
        trace, curve = evaluate(env, ctrl, n_episodes=args.episodes, max_steps=args.max_steps)
        m = summarize(trace)
        m["controller"] = name
        rows.append(m)
        if curve:
            error_curves[name] = curve
        print(f"{name:24s} success={m['success_rate']:.0%} "
              f"rmse={m.get('tracking_rmse', float('nan')):.3f} "
              f"final={m.get('final_error', float('nan')):.3f} "
              f"energy={m.get('control_energy', float('nan')):.3f} "
              f"jerk={m.get('action_jerk', float('nan')):.3f}")
    env.close()

    # --- CSV table ---
    csv_path = os.path.join(args.out_dir, "controller_metrics.csv")
    keys = ["controller"] + [k for k in rows[0] if k != "controller"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[matrix] wrote {csv_path}")

    plot_metric_bars(rows, METRICS, os.path.join(args.out_dir, "controller_bars.png"),
                     title=f"Controller matrix on {args.task}")
    if error_curves:
        plot_error_curves(error_curves, os.path.join(args.out_dir, "error_vs_time.png"))

    # --- demo gifs ---
    if args.gif:
        render_env = eval_env_factory(seed=7, render_mode="rgb_array")
        for name, ctrl in controllers:
            policy = ctrl.act if hasattr(ctrl, "act") else ctrl
            if hasattr(ctrl, "reset"):
                ctrl.reset()
            safe = name.split(" ")[0].lower().replace("(", "").replace(")", "")
            record_rollout(render_env, policy,
                           os.path.join(args.out_dir, f"demo_{safe}.gif"),
                           n_episodes=2, max_steps=args.max_steps)
        render_env.close()


if __name__ == "__main__":
    main()
