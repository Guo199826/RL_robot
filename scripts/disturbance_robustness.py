"""Disturbance-rejection experiment: shove the end-effector mid-reach.

For each controller we inject a fixed-timing exogenous push (see
``PushDisturbanceWrapper``) and record the goal-distance error over time,
averaged across episodes. This exposes each strategy's *disturbance rejection*
characteristic -- peak deviation when shoved and how fast it recovers.

Usage:
    python scripts/disturbance_robustness.py --episodes 30 --push-step 18 --strength 1.0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mc.common.envs import make_env, parse_obs, DEFAULT_TASK
from mc.common.eval import model_policy
from mc.controllers import TaskSpacePID, ScriptedPickPlace
from mc.controllers.residual import ResidualController
from mc.sim2real import PushDisturbanceWrapper

TOL = 0.05


def model_based_base(task):
    if "Reach" in task:
        return TaskSpacePID(kp=8.0, ki=0.0, kd=1.0, gripper=0.0)
    return ScriptedPickPlace()


def build_controllers(task, sac_path, ppo_path, residual_path, residual_scale,
                      sac_disturb_path=None):
    ctrls = []
    label = "Model-based (PD)" if "Reach" in task else "Scripted (model-based)"
    ctrls.append((label, model_based_base(task)))
    if sac_path and os.path.exists(sac_path):
        from stable_baselines3 import SAC
        m = SAC.load(sac_path, env=make_env(task, reward_type="sparse"))
        ctrls.append(("SAC (clean-trained)", model_policy(m)))
    if sac_disturb_path and os.path.exists(sac_disturb_path):
        from stable_baselines3 import SAC
        m = SAC.load(sac_disturb_path, env=make_env(task, reward_type="sparse"))
        ctrls.append(("SAC (push-trained)", model_policy(m)))
    if ppo_path and os.path.exists(ppo_path):
        from stable_baselines3 import PPO
        ctrls.append(("PPO (dense)", model_policy(PPO.load(ppo_path))))
    if residual_path and os.path.exists(residual_path):
        from stable_baselines3 import SAC
        m = SAC.load(residual_path, env=make_env(task, reward_type="sparse"))
        ctrls.append(("Residual RL (hybrid)", ResidualController(model_based_base(task), m, residual_scale)))
    return ctrls


def rollout(env, controller, episodes, max_steps):
    """Fixed-length rollouts (no early stop) -> mean error curve + metrics."""
    act = controller.act if hasattr(controller, "act") else controller
    curves = []
    final_ok = []
    for _ in range(episodes):
        if hasattr(controller, "reset"):
            controller.reset()
        obs, _ = env.reset()
        errs = []
        for t in range(max_steps):
            s = parse_obs(obs)
            errs.append(float(np.linalg.norm(s.desired_goal - s.achieved_goal)))
            obs, _, term, trunc, _ = env.step(act(obs))
        curves.append(errs)
        final_ok.append(errs[-1] < TOL)
    mean_curve = np.mean(np.array(curves), axis=0)
    return mean_curve, float(np.mean(final_ok))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=45)
    p.add_argument("--push-step", type=int, default=18)
    p.add_argument("--duration", type=int, default=3)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--sac", default="results/benchmark/sac/model.zip")
    p.add_argument("--ppo", default="results/benchmark/ppo/model.zip")
    p.add_argument("--sac-disturb", default=None)
    p.add_argument("--residual", default="results/residual/model.zip")
    p.add_argument("--residual-scale", type=float, default=0.3)
    p.add_argument("--out-dir", default="results/disturbance")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    controllers = build_controllers(args.task, args.sac, args.ppo,
                                    args.residual, args.residual_scale,
                                    sac_disturb_path=args.sac_disturb)
    push_end = args.push_step + args.duration

    rows, curves = [], {}
    for name, ctrl in controllers:
        env = PushDisturbanceWrapper(
            make_env(args.task, reward_type="sparse"),
            push_step=args.push_step, duration=args.duration,
            strength=args.strength, random_dir=True, seed=0,
        )
        mean_curve, recovered = rollout(env, ctrl, args.episodes, args.max_steps)
        env.close()
        curves[name] = mean_curve

        post = mean_curve[args.push_step:]
        peak = float(np.max(post))
        # recovery: steps after push-end until error stays < TOL
        rec = args.max_steps - push_end
        for i in range(push_end, args.max_steps):
            if np.all(mean_curve[i:] < TOL):
                rec = i - push_end
                break
        rows.append({"controller": name, "peak_deviation": round(peak, 4),
                     "recovery_steps": rec, "recovered_rate": round(recovered, 3)})
        print(f"{name:24s} peak_dev={peak:.3f}m  recovery={rec} steps  recovered={recovered:.0%}")

    # CSV
    with open(os.path.join(args.out_dir, "disturbance_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["controller", "peak_deviation", "recovery_steps", "recovered_rate"])
        w.writeheader(); w.writerows(rows)

    # error-vs-time overlay with shaded push window
    plt.figure(figsize=(9.5, 5.5))
    for name, c in curves.items():
        plt.plot(c, lw=2, label=name)
    plt.axvspan(args.push_step, push_end, color="red", alpha=0.15, label="push (shove)")
    plt.axhline(TOL, ls=":", color="gray", label=f"success tol {TOL}m")
    plt.xlabel("Time step"); plt.ylabel("Goal-distance error (m)")
    plt.title(f"Disturbance rejection on {args.task}: sudden push at step {args.push_step}")
    plt.grid(True, alpha=0.3); plt.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "disturbance_recovery.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
