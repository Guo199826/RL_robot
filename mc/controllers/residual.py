"""Residual RL: classical base controller + learned residual correction.

This is the canonical "traditional control + RL" hybrid (Johannink et al.,
"Residual Reinforcement Learning for Robot Control", ICRA 2019). A hand-designed
base controller does most of the work; the RL policy only learns a small
*residual* action that corrects for what the model cannot capture (contacts,
friction, calibration error). Benefits to highlight in the interview:

- Training starts from a competent policy -> far better sample efficiency and
  safety than learning from scratch.
- The base controller bounds worst-case behaviour (interpretability/safety).
- The residual absorbs the sim2real gap the analytic model misses.

``ResidualRLWrapper`` is used at *training* time (the agent sees the normal
goal-conditioned obs and emits a residual). ``ResidualController`` wraps a
trained model for *evaluation* through the uniform ``Controller`` interface.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from .base import Controller


class ResidualRLWrapper(gym.Wrapper):
    """Add the agent's action as a residual on top of a base controller.

    The action space stays ``Box(-1, 1, (4,))`` so any off-policy algo + HER
    works unchanged; ``residual_scale`` caps how much the policy may deviate
    from the base controller.
    """

    def __init__(self, env, base_controller, residual_scale: float = 0.3):
        super().__init__(env)
        self.base = base_controller
        self.residual_scale = residual_scale
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.base.reset()
        self._last_obs = obs
        return obs, info

    def step(self, residual):
        base_action = self.base.act(self._last_obs)
        action = np.clip(base_action + self.residual_scale * np.asarray(residual), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action.astype(np.float32))
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


class ResidualController(Controller):
    """Evaluation-time view: base action + trained-model residual."""

    name = "Residual RL (hybrid)"

    def __init__(self, base_controller, model, residual_scale: float = 0.3):
        self.base = base_controller
        self.model = model
        self.residual_scale = residual_scale

    def reset(self) -> None:
        self.base.reset()

    def act(self, obs: dict) -> np.ndarray:
        base_action = self.base.act(obs)
        residual, _ = self.model.predict(obs, deterministic=True)
        return np.clip(base_action + self.residual_scale * np.asarray(residual), -1.0, 1.0).astype(np.float32)
