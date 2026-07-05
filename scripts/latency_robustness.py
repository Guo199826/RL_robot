"""Where RL *does* beat fixed classical control: latency + model uncertainty.

On FetchReach a closed-loop PD is intrinsically robust to gain/disturbance, so
RL has no edge on the nominal task. The edge appears under **control latency +
wide model uncertainty**: a fixed-gain linear PD faces a hard speed-vs-stability
tradeoff -- tune it "hot" and it oscillates under delay, tune it "soft" and it
is sluggish. *No single fixed gain is good across the whole uncertainty range.*
A policy trained across that distribution (DR + random latency) learns a
nonlinear, context-sensitive law that is more robust across the range.

This script:
  1. trains a SAC under DR (wide gain + noise) + random latency (if absent);
  2. sweeps the deployment latency and reports tracking RMSE for
     {PD-compromise, PD-hot, SAC-clean, SAC-DR+latency};
  3. plots RMSE-vs-latency (lower & flatter = more robust).

Usage:
    python scripts/latency_robustness.py --timesteps 25000 --episodes 40
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
from mc.common.training import build_model
from mc.common.callbacks import SuccessRateEvalCallback
from mc.controllers import TaskSpacePID
from mc.sim2real import (
    DomainRandomizationWrapper,
    ObservationNoiseWrapper,
    ActionLatencyWrapper,
    RandomActionLatencyWrapper,
)

GAIN_RANGE = (0.4, 1.6)
PROC_NOISE = 0.02
OBS_SIGMA = 0.01
TOL = 0.05


def perturbed_env(seed, delay, train=False, max_delay=3):
    """DR (wide gain + noise) + obs noise + latency. Train uses random latency."""
    e = make_env(DEFAULT_TASK, reward_type="sparse")
    e = DomainRandomizationWrapper(e, action_gain_range=GAIN_RANGE,
                                   process_noise=PROC_NOISE, seed=seed)
    e = ObservationNoiseWrapper(e, obs_sigma=OBS_SIGMA, seed=seed)
    if train:
        e = RandomActionLatencyWrapper(e, max_delay=max_delay, seed=seed)
    elif delay > 0:
        e = ActionLatencyWrapper(e, delay=delay)
    return e


def train_dr_latency(path, timesteps, seed):
    if os.path.exists(path):
        print(f"[train] reuse existing {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = perturbed_env(seed, delay=0, train=True)
    model = build_model("SAC", env, use_her=True, seed=seed, verbose=0)
    cb = SuccessRateEvalCallback(lambda: perturbed_env(seed + 7, delay=0, train=True),
                                 eval_freq=2500, n_eval_episodes=20, max_steps=50,
                                 csv_path=os.path.join(os.path.dirname(path), "dr_latency_curve.csv"))
    print(f"===== training SAC under DR + random latency for {timesteps} steps =====")
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(path[:-4] if path.endswith(".zip") else path)
    env.close()


def eval_rmse(controller, delay, episodes):
    act = controller.act if hasattr(controller, "act") else controller
    rmses, successes = [], []
    for ep in range(episodes):
        env = perturbed_env(1000 + ep, delay=delay)
        if hasattr(controller, "reset"):
            controller.reset()
        obs, _ = env.reset(seed=1000 + ep)
        errs, ok = [], False
        for _ in range(50):
            s = parse_obs(obs)
            errs.append(float(np.linalg.norm(s.desired_goal - s.achieved_goal)))
            obs, _, _, _, info = env.step(act(obs))
            ok = ok or bool(info.get("is_success"))
        env.close()
        rmses.append(float(np.sqrt(np.mean(np.square(errs)))))
        successes.append(ok)
    return float(np.mean(rmses)), float(np.mean(successes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=25000)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--sac-clean", default="results/benchmark/sac/model.zip")
    p.add_argument("--out-dir", default="results/latency")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dr_path = os.path.join(args.out_dir, "sac_dr_latency.zip")
    train_dr_latency(dr_path, args.timesteps, args.seed)

    from stable_baselines3 import SAC
    controllers = {
        "PD (compromise kp=8)": TaskSpacePID(kp=8.0, kd=1.0, gripper=0.0),
        "PD (hot kp=15)": TaskSpacePID(kp=15.0, kd=1.0, gripper=0.0),
    }
    if os.path.exists(args.sac_clean):
        controllers["SAC (clean-trained)"] = model_policy(
            SAC.load(args.sac_clean, env=make_env(DEFAULT_TASK, reward_type="sparse")))
    controllers["SAC (DR + latency-trained)"] = model_policy(
        SAC.load(dr_path, env=make_env(DEFAULT_TASK, reward_type="sparse")))

    rmse_curves, succ_curves = {}, {}
    rows = []
    for name, ctrl in controllers.items():
        rmse_curves[name], succ_curves[name] = [], []
        for d in args.delays:
            r, s = eval_rmse(ctrl, d, args.episodes)
            rmse_curves[name].append(r); succ_curves[name].append(s)
            rows.append({"controller": name, "delay": d, "rmse": round(r, 4), "success": round(s, 3)})
        print(f"{name:30s} RMSE@delays = " +
              " ".join(f"{d}:{r:.3f}" for d, r in zip(args.delays, rmse_curves[name])))

    with open(os.path.join(args.out_dir, "latency_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["controller", "delay", "rmse", "success"])
        w.writeheader(); w.writerows(rows)

    styles = {"PD (compromise kp=8)": ("tab:green", "--", "s"),
              "PD (hot kp=15)": ("tab:olive", ":", "^"),
              "SAC (clean-trained)": ("tab:orange", "-", "o"),
              "SAC (DR + latency-trained)": ("tab:blue", "-", "D")}
    plt.figure(figsize=(9, 5.6))
    for name, ys in rmse_curves.items():
        c, ls, mk = styles.get(name, ("gray", "-", "o"))
        plt.plot(args.delays, ys, color=c, ls=ls, marker=mk, lw=2, ms=6, label=name)
    plt.xlabel("Deployment control latency (steps)")
    plt.ylabel("Tracking RMSE (m)  — lower & flatter = more robust")
    plt.title("FetchReach under latency + model uncertainty:\n"
              "feedback-dominated → well-tuned PD still wins; DR only makes RL the *most robust RL*")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    out = os.path.join(args.out_dir, "latency_robustness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
