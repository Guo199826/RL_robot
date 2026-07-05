"""🅲  Sim2real robustness: train with vs. without domain randomization (DR).

Trains two SAC+HER policies (clean env vs. DR env), then cross-evaluates both on
clean and perturbed dynamics. The DR policy should keep a much higher success
rate under perturbation -- that gap is the sim2real robustness result.

Usage:
    python scripts/sim2real_robustness.py --timesteps 60000
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

from mc.common.envs import make_env, DEFAULT_TASK
from mc.common.training import build_model
from mc.common.callbacks import SuccessRateEvalCallback
from mc.common.eval import evaluate, model_policy
from mc.common.metrics import summarize
from mc.sim2real import make_randomized_env


def clean_env():
    return make_env(DEFAULT_TASK, reward_type="sparse")


def perturbed_env(seed=None):
    return make_randomized_env(make_env(DEFAULT_TASK, reward_type="sparse"),
                               dr=True, obs_noise=True, latency=1, seed=seed)


def train(tag, env_fn, timesteps, out_dir, seed):
    env = env_fn()
    model = build_model("SAC", env, use_her=True, seed=seed, verbose=0)
    cb = SuccessRateEvalCallback(clean_env, eval_freq=5000, n_eval_episodes=20,
                                 csv_path=os.path.join(out_dir, f"{tag}_curve.csv"))
    print(f"===== training [{tag}] for {timesteps} steps =====")
    model.learn(total_timesteps=timesteps, callback=cb)
    path = os.path.join(out_dir, f"{tag}_model")
    model.save(path)
    env.close()
    return path


def eval_success(model_path, env_fn, episodes=50):
    from stable_baselines3 import SAC
    model = SAC.load(model_path)
    env = env_fn()
    trace, _ = evaluate(env, model_policy(model), n_episodes=episodes, max_steps=60)
    env.close()
    return summarize(trace)["success_rate"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=60000)
    p.add_argument("--out-dir", default="results/sim2real")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    no_dr = train("no_dr", clean_env, args.timesteps, args.out_dir, args.seed)
    dr = train("dr", lambda: perturbed_env(args.seed), args.timesteps, args.out_dir, args.seed)

    results = {
        "no_dr": {"clean": eval_success(no_dr, clean_env),
                  "perturbed": eval_success(no_dr, lambda: perturbed_env(123))},
        "dr": {"clean": eval_success(dr, clean_env),
               "perturbed": eval_success(dr, lambda: perturbed_env(123))},
    }

    csv_path = os.path.join(args.out_dir, "robustness.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_setting", "eval_clean", "eval_perturbed"])
        for k, v in results.items():
            w.writerow([k, v["clean"], v["perturbed"]])
    print(f"[sim2real] {results}")

    # grouped bar chart
    labels = ["Eval: clean", "Eval: perturbed"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5.5))
    plt.bar(x - width / 2, [results["no_dr"]["clean"], results["no_dr"]["perturbed"]],
            width, label="Trained w/o DR")
    plt.bar(x + width / 2, [results["dr"]["clean"], results["dr"]["perturbed"]],
            width, label="Trained w/ DR")
    plt.xticks(x, labels)
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    plt.title("Sim2real robustness: domain randomization closes the gap")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "robustness.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[sim2real] saved {out_png}")


if __name__ == "__main__":
    main()
