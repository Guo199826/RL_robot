"""Torque-vs-time: the SAC-entropy effect is a *stochastic-deployment* effect.

Honest finding (verified across seeds):
  * Deterministic policy (the mode, what we render in the GIFs): BOTH SAC and PPO
    settle to a near-constant hold torque -- SAC is in fact perfectly flat
    (std~0). Neither chatters. Both hold a small steady-state angle offset
    (~0.076 rad) with a constant bias torque (no integral action -> residual
    error).
  * Stochastic policy (sampled actions): SAC keeps a wide action distribution
    (ent_coef ~ 0.072 by design) so its torque chatters (+-0.33) even at the
    goal; PPO collapsed its entropy to ~0 (ent_coef=0) so it stays smooth.

So the max-entropy "chatter" is a property of *how you deploy* SAC (sampling vs
mode), not of the deterministic controller. This 2-row figure shows both.

Run:
  python scripts/pendulum_torque_curve.py --out-dir results/pendulum
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, PPO

from mc.pendulum.env import make_pendulum, angle_normalize


def rollout(model, seed, deterministic, max_steps):
    env = make_pendulum(randomize=False, seed=seed, max_episode_steps=max_steps)
    u = env.unwrapped
    obs, _ = env.reset()
    t, torque, angle = [], [], []
    for k in range(max_steps):
        a, _ = model.predict(obs, deterministic=deterministic)
        torque.append(float(np.clip(a, -1.0, 1.0)[0]))
        angle.append(abs(float(angle_normalize(u.state[0]))))
        t.append(k * u.dt)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    env.close()
    return np.array(t), np.array(angle), np.array(torque)


def steady_std(t, x, window=2.0):
    return float(np.std(x[t >= (t[-1] - window)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/pendulum")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    sac = SAC.load(os.path.join(args.out_dir, "sac", "model.zip"))
    ppo = PPO.load(os.path.join(args.out_dir, "ppo", "model.zip"))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True)
    for ax, det, label in [(ax0, True, "Deterministic (mode) — used in the GIFs"),
                           (ax1, False, "Stochastic (sampled actions)")]:
        ts, _, us = rollout(sac, args.seed, det, args.max_steps)
        tp, _, up = rollout(ppo, args.seed, det, args.max_steps)
        ax.plot(ts, us, color="tab:blue", lw=1.3,
                label=f"SAC  (steady std={steady_std(ts, us):.3f})")
        ax.plot(tp, up, color="tab:orange", lw=1.3,
                label=f"PPO  (steady std={steady_std(tp, up):.3f})")
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.axvspan(8.0, 10.0, color="0.92", zorder=0)  # steady-state window
        ax.set_ylabel("normalized torque  u / u_max")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
    ax1.set_xlabel("time (s)   (grey band = steady-state window)")
    fig.suptitle("SAC keeps entropy (chatters only when sampled); PPO collapses entropy (smooth either way)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "torque_comparison.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[done] saved {out}")


if __name__ == "__main__":
    main()
