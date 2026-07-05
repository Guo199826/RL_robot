"""Headless-friendly rollout recording (rgb_array -> gif/mp4).

On a CPU/headless box ``render_mode="human"`` is useless for a slide deck, so we
roll out with ``render_mode="rgb_array"`` and dump frames to a gif (always
available via matplotlib/Pillow) or mp4 (if imageio is installed).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np


def record_rollout(
    env,
    policy: Callable,
    out_path: str,
    n_episodes: int = 3,
    max_steps: int = 80,
    fps: int = 20,
    reset_on_success: bool = True,
) -> Optional[str]:
    """Roll ``policy`` out in ``env`` (rgb_array) and save an animation.

    ``policy(obs) -> action``. ``env`` must be created with
    ``render_mode="rgb_array"``.
    """
    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            frames.append(env.render())
            action = policy(obs)
            obs, _, term, trunc, info = env.step(action)
            if reset_on_success and info.get("is_success", 0) > 0:
                # linger a few frames on success, then move on
                for _ in range(5):
                    frames.append(env.render())
                break
            if term or trunc:
                break
    return save_frames(frames, out_path, fps=fps)


def save_frames(frames, out_path: str, fps: int = 20) -> Optional[str]:
    if not frames:
        print("[video] no frames captured")
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frames = [np.asarray(f) for f in frames]

    # Prefer imageio (handles both gif and mp4); fall back to Pillow for gif.
    try:
        import imageio.v2 as imageio

        if out_path.endswith(".mp4"):
            imageio.mimsave(out_path, frames, fps=fps)
        else:
            imageio.mimsave(out_path, frames, duration=1.0 / fps)
        print(f"[video] saved {out_path} ({len(frames)} frames)")
        return out_path
    except Exception:
        pass

    try:
        from PIL import Image

        gif_path = out_path if out_path.endswith(".gif") else out_path + ".gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"[video] saved {gif_path} ({len(frames)} frames, Pillow)")
        return gif_path
    except Exception as e:  # pragma: no cover
        print(f"[video] failed to save animation: {e}")
        return None
