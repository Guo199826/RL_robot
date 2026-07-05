"""RL-tuned PID (gain scheduling): RL outputs [Kp, Ki, Kd], a PID loop drives the arm.

This is a *milder* hybrid than Residual RL: instead of replacing the controller,
RL only adapts its gains online. It keeps the full interpretability of a PID law
while letting the policy schedule gains per state. Most meaningful on reaching /
trajectory-tracking tasks (there is no explicit grasp phase), so for the
PickAndPlace matrix it serves as the "feedback-tuning, no task structure"
reference point.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from ..common.envs import parse_obs

# Map the policy output in [-1, 1] to physical gain ranges.
GAIN_SCALE = np.array([2.5, 0.5, 1.0])   # Kp in [0,5], Ki in [0,1], Kd in [0,2]
GAIN_BIAS = np.array([2.5, 0.5, 1.0])


def action_to_gains(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).reshape(-1)[:3]
    return a * GAIN_SCALE + GAIN_BIAS


class RLPIDTunerWrapper(gym.Wrapper):
    """Action space becomes [Kp, Ki, Kd]; an internal PID computes the velocity."""

    def __init__(self, env, gripper=-1.0, integral_clip=1.0):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.gripper = gripper
        self.integral_clip = integral_clip
        self._reset_pid()

    def _reset_pid(self):
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)
        self._last_obs = None

    def reset(self, **kwargs):
        self._reset_pid()
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, gain_action):
        kp, ki, kd = action_to_gains(gain_action)
        s = parse_obs(self._last_obs)
        error = s.desired_goal - s.grip_pos

        self._integral = np.clip(self._integral + error, -self.integral_clip, self.integral_clip)
        derivative = error - self._prev_error
        self._prev_error = error

        vel = kp * error + ki * self._integral + kd * derivative
        robot_action = np.zeros(4, dtype=np.float32)
        robot_action[:3] = np.clip(vel, -1.0, 1.0)
        robot_action[3] = self.gripper

        obs, reward, terminated, truncated, info = self.env.step(robot_action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info
