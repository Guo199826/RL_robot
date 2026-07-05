"""Pendulum swing-up study: classical model-based vs RL under uncertainty.

Produces exactly the three interview figures:
  1. training_curves.png   -- SAC vs PPO sample efficiency (eval return vs steps)
  2. nominal_test.png      -- PID / Energy+LQR / PPO / SAC / residual on the
                              *nominal* model (classical control shines here)
  3. random_test.png       -- same controllers under mass/length/damping
                              randomization (RL stays robust, classical degrades)

Thesis: classical control is great under the nominal model but brittle under
parameter uncertainty; RL (trained with domain randomization) is robust; SAC is
more sample-efficient than PPO for this continuous-control task.

Run:
  python scripts/pendulum_study.py --out-dir results/pendulum
  python scripts/pendulum_study.py --quick      # fast smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_num_threads(4)  # small MLPs: fewer threads avoids CPU oversubscription

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback

from mc.pendulum.env import make_pendulum
from mc.pendulum.controllers import PID, EnergyLQR, ResidualController, ResidualEnv
from mc.common.callbacks import CurveLoggerCallback
from mc.common.plotting import (plot_learning_curves, plot_metric_bars,
                                plot_training_curves)

EVAL_SEED = 777


# --------------------------------------------------------------------------- #
#  evaluation
# --------------------------------------------------------------------------- #
def eval_controller(kind, obj, randomize, n_episodes=60, seed=EVAL_SEED,
                    residual_base=None):
    """kind in {'classical','rl','residual'}. Returns (mean_return, upright_frac)."""
    env = make_pendulum(randomize=randomize, seed=seed)
    u = env.unwrapped
    rets, uprs = [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        if kind == "classical":
            obj.reset()
        elif kind == "residual":
            residual_base.reset()
        obs = u._obs()
        R, ups = 0.0, []
        while True:
            if kind == "classical":
                a = obj.act(u.control_state())
            elif kind == "rl":
                a, _ = obj.predict(obs, deterministic=True)
            else:  # residual
                a = obj.act_with_obs(u.control_state(), obs)
            obs, r, term, trunc, info = env.step(a)
            R += r
            ups.append(info["upright"])
            if term or trunc:
                break
        rets.append(R)
        uprs.append(float(np.mean(ups[-60:])))
    env.close()
    return float(np.mean(rets)), float(np.mean(uprs))


class EvalReturnCallback(BaseCallback):
    """Log mean eval return + upright fraction to CSV every ``eval_freq`` steps."""

    def __init__(self, csv_path, eval_freq=2000, n_eval=10, residual_base=None):
        super().__init__()
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.residual_base = residual_base
        self._next = eval_freq
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "mean_return", "upright_frac"])

    def _evaluate(self):
        env = make_pendulum(randomize=True, seed=EVAL_SEED)
        u = env.unwrapped
        rets, uprs = [], []
        for ep in range(self.n_eval):
            env.reset(seed=EVAL_SEED + ep)
            if self.residual_base is not None:
                self.residual_base.reset()
            obs = u._obs()
            R, ups = 0.0, []
            while True:
                act, _ = self.model.predict(obs, deterministic=True)
                if self.residual_base is not None:
                    base = self.residual_base.act(u.control_state())
                    act = np.clip(base + 0.5 * np.asarray(act, np.float32), -1, 1)
                obs, r, term, trunc, info = env.step(act)
                R += r
                ups.append(info["upright"])
                if term or trunc:
                    break
            rets.append(R)
            uprs.append(float(np.mean(ups[-60:])))
        env.close()
        return float(np.mean(rets)), float(np.mean(uprs))

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            mr, uf = self._evaluate()
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, f"{mr:.3f}", f"{uf:.4f}"])
            if self.verbose:
                print(f"[eval] step={self.num_timesteps:7d} return={mr:8.1f} upright={uf:.2f}")
        return True


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
def train_sac(out_dir, timesteps, seed, residual_base=None):
    tag = "residual" if residual_base is not None else "sac"
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    if residual_base is not None:
        env = ResidualEnv(make_pendulum(randomize=True, seed=seed), EnergyLQR())
    else:
        env = make_pendulum(randomize=True, seed=seed)
    model = SAC("MlpPolicy", env, learning_rate=1e-3, verbose=0, seed=seed,
                gamma=0.98, buffer_size=200_000)
    cb = EvalReturnCallback(os.path.join(d, "train_curve.csv"), eval_freq=2000,
                            residual_base=EnergyLQR() if residual_base is not None else None)
    loss_cb = CurveLoggerCallback(log_freq=1000, csv_path=os.path.join(d, "loss_curve.csv"))
    print(f"[train] {tag.upper()} for {timesteps} steps (domain-randomized)")
    model.learn(total_timesteps=timesteps, callback=[cb, loss_cb])
    model.save(os.path.join(d, "model.zip"))
    plot_training_curves(os.path.join(d, "loss_curve.csv"),
                         os.path.join(d, "training_diagnostics.png"),
                         title=f"{tag.upper()} training diagnostics (pendulum swing-up)")
    return model


def train_ppo(out_dir, timesteps, seed, n_envs=4):
    """PPO tuned for Pendulum (RL-Zoo style: gamma=0.9 + gSDE exploration).

    Pendulum has a short effective horizon, so a high discount hurts; gSDE gives
    PPO the temporally-correlated exploration it needs for swing-up. Several
    parallel envs make the on-policy budget cheap in wall-clock.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    d = os.path.join(out_dir, "ppo")
    os.makedirs(d, exist_ok=True)
    env = DummyVecEnv([(lambda i=i: make_pendulum(randomize=True, seed=seed + i))
                       for i in range(n_envs)])
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, learning_rate=1e-3,
                gamma=0.9, gae_lambda=0.95, n_steps=1024, batch_size=256,
                n_epochs=10, ent_coef=0.0, clip_range=0.2,
                use_sde=True, sde_sample_freq=4)
    cb = EvalReturnCallback(os.path.join(d, "train_curve.csv"), eval_freq=2000)
    loss_cb = CurveLoggerCallback(log_freq=1000, csv_path=os.path.join(d, "loss_curve.csv"))
    print(f"[train] PPO for {timesteps} steps (domain-randomized, {n_envs} envs)")
    model.learn(total_timesteps=timesteps, callback=[cb, loss_cb])
    model.save(os.path.join(d, "model.zip"))
    plot_training_curves(os.path.join(d, "loss_curve.csv"),
                         os.path.join(d, "training_diagnostics.png"),
                         title="PPO training diagnostics (pendulum swing-up)")
    return model


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/pendulum")
    p.add_argument("--sac-steps", type=int, default=100_000)
    p.add_argument("--ppo-steps", type=int, default=200_000)
    p.add_argument("--residual-steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse-rl", action="store_true",
                   help="load existing SAC/residual models; only (re)train PPO")
    p.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test")
    args = p.parse_args()
    if args.quick:
        args.sac_steps = args.ppo_steps = args.residual_steps = 6_000
    os.makedirs(args.out_dir, exist_ok=True)

    if args.reuse_rl:
        print("[reuse] loading existing SAC + residual models")
        sac = SAC.load(os.path.join(args.out_dir, "sac", "model.zip"))
        residual_model = SAC.load(os.path.join(args.out_dir, "residual", "model.zip"))
    else:
        sac = train_sac(args.out_dir, args.sac_steps, args.seed)
        residual_model = train_sac(args.out_dir, args.residual_steps, args.seed,
                                   residual_base=EnergyLQR())
    ppo = train_ppo(args.out_dir, args.ppo_steps, args.seed)

    # ---- Figure 1: SAC vs PPO training curves --------------------------------
    plot_learning_curves(
        {"SAC": os.path.join(args.out_dir, "sac", "train_curve.csv"),
         "PPO": os.path.join(args.out_dir, "ppo", "train_curve.csv")},
        os.path.join(args.out_dir, "training_curves.png"),
        title="Sample efficiency: SAC vs PPO (pendulum swing-up, domain-randomized)",
        ylabel="Mean eval return", y="mean_return",
    )

    # ---- Figures 2 & 3: nominal and randomized evaluation --------------------
    controllers = [
        ("PID (naive)", "classical", PID(), None),
        ("Energy+LQR (model-based)", "classical", EnergyLQR(), None),
        ("PPO", "rl", ppo, None),
        ("SAC", "rl", sac, None),
        ("Model-based + RL residual", "residual",
         ResidualController(EnergyLQR(), residual_model, residual_scale=0.5), EnergyLQR()),
    ]
    for tag, randomize in [("nominal", False), ("random", True)]:
        rows = []
        print(f"\n=== {tag} evaluation ===")
        for name, kind, obj, rbase in controllers:
            mr, uf = eval_controller(kind, obj, randomize, residual_base=rbase)
            rows.append({"controller": name, "return": round(mr, 1),
                         "upright_frac": round(uf, 3)})
            print(f"  {name:28s} return={mr:8.1f} upright={uf:.2f}")
        csv_path = os.path.join(args.out_dir, f"{tag}_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["controller", "return", "upright_frac"])
            w.writeheader(); w.writerows(rows)
        title = ("Nominal model: classical control is excellent" if tag == "nominal"
                 else "Randomized mass/length/damping: RL stays robust, classical degrades")
        plot_metric_bars(rows, ["return", "upright_frac"],
                         os.path.join(args.out_dir, f"{tag}_test.png"), title=title)
    print("\n[done] figures + metrics in", args.out_dir)


if __name__ == "__main__":
    main()
