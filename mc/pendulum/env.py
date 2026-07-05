"""Torque-limited pendulum swing-up with optional parameter randomization.

Mirrors Gymnasium's ``Pendulum-v1`` (uniform rod, inertia ``I = m l^2 / 3``,
com at ``l/2``), with two additions that matter for this study:

* a viscous **joint damping** term ``b`` (Pendulum-v1 has none), so we can
  randomize damping alongside mass/length;
* the physical parameters ``m, l, g, b`` are plain attributes, so a
  domain-randomization wrapper can resample them per episode and the classical
  controllers can be built from a *fixed nominal* model (the mismatch is the
  whole point).

Angle convention: ``theta = 0`` is **upright** (goal, unstable); the pendulum
starts hanging near ``theta = pi``. Reward (dense, identical to Pendulum-v1):
``-(theta_norm^2 + 0.1*thetadot^2 + 0.001*u^2)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# nominal physical parameters (what the classical controllers assume)
NOMINAL = dict(m=1.0, l=1.0, g=10.0, b=0.0)
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None):
        self.m = NOMINAL["m"]
        self.l = NOMINAL["l"]
        self.g = NOMINAL["g"]
        self.b = NOMINAL["b"]
        self.dt = DT
        self.max_torque = MAX_TORQUE
        self.max_speed = MAX_SPEED
        self.render_mode = render_mode
        self.state = np.zeros(2)
        self.last_u = 0.0

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    # --- dynamics -----------------------------------------------------------
    def _obs(self):
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)

    def step(self, action):
        th, thdot = self.state
        u = float(np.clip(action, -1.0, 1.0)[0]) * self.max_torque
        self.last_u = u

        I = self.m * self.l ** 2 / 3.0                 # uniform-rod inertia
        grav = self.m * self.g * (self.l / 2.0) * np.sin(th)
        thddot = (grav + u - self.b * thdot) / I

        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2

        thdot = float(np.clip(thdot + thddot * self.dt, -self.max_speed, self.max_speed))
        th = th + thdot * self.dt
        self.state = np.array([th, thdot])

        info = {"theta": angle_normalize(th), "thetadot": thdot,
                "upright": float(abs(angle_normalize(th)) < 0.2)}
        return self._obs(), -float(cost), False, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # start hanging down (theta ~ pi) with small perturbation
        th = np.pi + self.np_random.uniform(-0.3, 0.3)
        thdot = self.np_random.uniform(-0.5, 0.5)
        self.state = np.array([th, thdot])
        self.last_u = 0.0
        return self._obs(), {}

    def control_state(self):
        """True (theta-from-upright, thetadot) for the model-based controllers."""
        th, thdot = self.state
        return np.array([angle_normalize(th), thdot])


class DomainRandomizationWrapper(gym.Wrapper):
    """Resample mass / length / damping per episode (multiplicative on m, l)."""

    def __init__(self, env, mass_range=(0.6, 1.4), length_range=(0.7, 1.3),
                 damping_range=(0.0, 0.15), seed=None):
        super().__init__(env)
        self.mass_range = mass_range
        self.length_range = length_range
        self.damping_range = damping_range
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        u = self.unwrapped
        u.m = NOMINAL["m"] * self.rng.uniform(*self.mass_range)
        u.l = NOMINAL["l"] * self.rng.uniform(*self.length_range)
        u.b = self.rng.uniform(*self.damping_range)
        return self.env.reset(**kwargs)


def make_pendulum(randomize: bool = False, max_episode_steps: int = 200,
                  seed: Optional[int] = None, **dr_kwargs):
    env = PendulumEnv()
    if randomize:
        env = DomainRandomizationWrapper(env, seed=seed, **dr_kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env
