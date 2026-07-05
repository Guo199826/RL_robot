"""Environment factory and observation parsing for the Fetch tasks.

We standardise on ``FetchPickAndPlace-v4`` (``v3`` is deprecated in
gymnasium-robotics 1.4.x). Everything in the repo goes through ``make_env`` so
the task id, reward type and wrappers are configured in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

DEFAULT_TASK = "FetchReach-v4"


# --- Fetch observation layouts ------------------------------------------------
# The ``observation`` vector layout differs by task (see gymnasium-robotics
# MujocoFetchEnv._get_obs). We support both so the framework is task-agnostic.
#
# Manipulation tasks (PickAndPlace / Push, 25-dim):
#   [0:3] grip_pos  [3:6] object_pos  [6:9] object_rel_pos  [9:11] gripper_state
#   [11:14] object_rot  [14:17] object_velp  [17:20] object_velr
#   [20:23] grip_velp  [23:25] gripper_vel
PICK_SLICES = {
    "grip_pos": slice(0, 3),
    "object_pos": slice(3, 6),
    "object_rel_pos": slice(6, 9),
    "gripper_state": slice(9, 11),
    "grip_velp": slice(20, 23),
}
# Reach task (10-dim, no object):
#   [0:3] grip_pos  [3:5] gripper_state  [5:8] grip_velp  [8:10] gripper_vel
REACH_SLICES = {
    "grip_pos": slice(0, 3),
    "gripper_state": slice(3, 5),
    "grip_velp": slice(5, 8),
}
# Backwards-compat alias used elsewhere.
SLICES = PICK_SLICES


@dataclass
class FetchObs:
    """Typed view over a Fetch observation dict (no copies of the big array).

    ``object_pos``/``object_rel_pos`` are ``None`` on the Reach task (no object).
    """

    grip_pos: np.ndarray
    gripper_state: np.ndarray
    grip_velp: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray
    object_pos: "np.ndarray | None" = None
    object_rel_pos: "np.ndarray | None" = None

    @property
    def gripper_opening(self) -> float:
        """Distance between the two fingers (~0.05 fully open, ~0.0 closed)."""
        return float(self.gripper_state[0] + self.gripper_state[1])


def parse_obs(obs: dict) -> FetchObs:
    """Turn a raw Fetch observation dict into a typed, named view.

    The layout is selected from the observation length (10 = Reach, 25 = manip).
    """
    o = obs["observation"]
    ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
    dg = np.asarray(obs["desired_goal"], dtype=np.float32)

    if len(o) <= 10:  # Reach
        s = REACH_SLICES
        return FetchObs(
            grip_pos=o[s["grip_pos"]],
            gripper_state=o[s["gripper_state"]],
            grip_velp=o[s["grip_velp"]],
            achieved_goal=ag,
            desired_goal=dg,
        )

    s = PICK_SLICES  # manipulation
    return FetchObs(
        grip_pos=o[s["grip_pos"]],
        gripper_state=o[s["gripper_state"]],
        grip_velp=o[s["grip_velp"]],
        achieved_goal=ag,
        desired_goal=dg,
        object_pos=o[s["object_pos"]],
        object_rel_pos=o[s["object_rel_pos"]],
    )


def make_env(
    task: str = DEFAULT_TASK,
    reward_type: str = "sparse",
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """Create a Fetch env with consistent defaults.

    Parameters
    ----------
    reward_type: ``"sparse"`` (0 on success, -1 otherwise; pair with HER) or
        ``"dense"`` (negative distance; works without HER but is harder to tune).
    render_mode: ``None`` for headless training, ``"rgb_array"`` for recording,
        ``"human"`` for an interactive window (needs a display).
    """
    kwargs = dict(reward_type=reward_type)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(task, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
