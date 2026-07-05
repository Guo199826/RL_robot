"""Render a result GIF for every pendulum controller.

Rolls out each of PID / Energy+LQR / PPO / SAC / Model-based+residual on the
pendulum and saves an animated GIF, so the deck can show *what each policy
actually does* (PID stalls; Energy+LQR / SAC / residual swing up and balance).

By default the demo is on the *nominal* model; pass ``--randomize`` to show the
robustness contrast (classical Energy+LQR mistunes and falls, RL stays up).

Run:
  python scripts/pendulum_gifs.py --out-dir results/pendulum
  python scripts/pendulum_gifs.py --out-dir results/pendulum --randomize
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

from mc.pendulum.env import make_pendulum, angle_normalize, MAX_TORQUE
from mc.pendulum.controllers import PID, EnergyLQR, ResidualController
from mc.common.video import save_frames


def render_frame(fig, ax, theta, u_norm, name, upright):
    """Draw one pendulum frame (theta=0 is upright) -> RGB uint8 array."""
    ax.clear()
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    # target (upright) reference
    ax.plot([0, 0], [0, 1.18], color="0.8", lw=1.2, ls="--", zorder=0)
    tip_x, tip_y = np.sin(theta), np.cos(theta)        # l = 1 in display units
    color = "tab:green" if upright else "tab:red"
    ax.plot([0, tip_x], [0, tip_y], color=color, lw=6, solid_capstyle="round", zorder=2)
    ax.scatter([tip_x], [tip_y], s=320, color=color, zorder=3, edgecolors="k", linewidths=0.8)
    ax.scatter([0], [0], s=60, color="k", zorder=4)
    # torque indicator (signed bar at the pivot)
    ax.barh(-1.25, 1.1 * float(np.clip(u_norm, -1, 1)), height=0.12,
            left=0.0, color="tab:orange", alpha=0.8, zorder=1)
    ax.text(0, -1.25, "torque", ha="center", va="center", fontsize=7, color="0.3")
    ax.set_title(name, fontsize=12, pad=8)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def rollout_frames(fig, ax, env, step_action, name, max_steps, linger=12):
    u = env.unwrapped
    obs, _ = env.reset()
    frames = []
    for _ in range(max_steps):
        th = u.state[0]
        upright = abs(angle_normalize(th)) < 0.2
        frames.append(render_frame(fig, ax, th, u.last_u / MAX_TORQUE, name, upright))
        a = step_action(obs, u)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    # linger on the final pose so the GIF ends on the result
    th = u.state[0]
    upright = abs(angle_normalize(th)) < 0.2
    for _ in range(linger):
        frames.append(render_frame(fig, ax, th, u.last_u / MAX_TORQUE, name, upright))
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/pendulum")
    p.add_argument("--randomize", action="store_true",
                   help="demo under mass/length/damping randomization (robustness contrast)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    sac = SAC.load(os.path.join(args.out_dir, "sac", "model.zip"))
    ppo = PPO.load(os.path.join(args.out_dir, "ppo", "model.zip"))
    residual_model = SAC.load(os.path.join(args.out_dir, "residual", "model.zip"))
    residual = ResidualController(EnergyLQR(), residual_model, residual_scale=0.5)

    def classical(ctrl):
        return lambda obs, u: ctrl.act(u.control_state())

    def rl(model):
        return lambda obs, u: model.predict(obs, deterministic=True)[0]

    controllers = [
        ("pid", "PID (naive)", PID(), classical(PID())),
        ("energy_lqr", "Energy+LQR (model-based)", EnergyLQR(), classical(EnergyLQR())),
        ("ppo", "PPO", None, rl(ppo)),
        ("sac", "SAC", None, rl(sac)),
        ("residual", "Model-based + RL residual", residual,
         lambda obs, u: residual.act_with_obs(u.control_state(), obs)),
    ]

    gif_dir = os.path.join(args.out_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    suffix = "_random" if args.randomize else ""
    fig, ax = plt.subplots(figsize=(3.2, 3.4), dpi=110)
    title_tag = " [randomized]" if args.randomize else ""
    for key, name, ctrl, step_action in controllers:
        env = make_pendulum(randomize=args.randomize, seed=args.seed,
                            max_episode_steps=args.max_steps)
        if ctrl is not None and hasattr(ctrl, "reset"):
            ctrl.reset()
        frames = rollout_frames(fig, ax, env, step_action, name + title_tag,
                                args.max_steps)
        env.close()
        save_frames(frames, os.path.join(gif_dir, f"{key}{suffix}.gif"), fps=args.fps)
    plt.close(fig)
    print(f"[done] gifs in {gif_dir}")


if __name__ == "__main__":
    main()
