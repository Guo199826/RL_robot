"""Pure task-space PID feedback controller (classical baseline).

This is the simplest "traditional control" baseline: a PID loop on the
Cartesian error between the end-effector and the *goal*. It has no notion of
grasping, so on PickAndPlace it can only reach toward the target -- which is
exactly why it makes a good lower bound that motivates the smarter controllers.
For a fair reaching comparison it is also usable on FetchReach.
"""
from __future__ import annotations

import numpy as np

from .base import Controller
from ..common.envs import parse_obs


class TaskSpacePID(Controller):
    name = "PID"

    def __init__(self, kp=5.0, ki=0.0, kd=0.0, integral_clip=1.0, gripper=-1.0,
                 track="goal"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_clip = integral_clip
        self.gripper = gripper  # fixed gripper command (-1 closed, +1 open)
        self.track = track      # "goal" (desired_goal) or "object"
        self.reset()

    def reset(self) -> None:
        self._integral = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)

    def act(self, obs: dict) -> np.ndarray:
        s = parse_obs(obs)
        target = s.desired_goal if self.track == "goal" else s.object_pos
        error = target - s.grip_pos

        self._integral = np.clip(self._integral + error, -self.integral_clip, self.integral_clip)
        derivative = error - self._prev_error
        self._prev_error = error

        vel = self.kp * error + self.ki * self._integral + self.kd * derivative
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(vel, -1.0, 1.0)
        action[3] = self.gripper
        return action
