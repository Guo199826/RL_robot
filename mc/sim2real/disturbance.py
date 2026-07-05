"""External-disturbance ("sudden push") wrapper for the Fetch tasks.

Why this exists: a key robustness question is *disturbance rejection* -- if the
end-effector is suddenly shoved, can the controller bring it back? On Fetch the
gripper is rigidly welded to a position-controlled mocap target, so external
forces (``xfrc_applied``) are almost fully cancelled by the weld. The faithful,
interpretable way to model a push here is therefore a short burst of strong,
exogenous velocity command in a random direction that the controller cannot
prevent. After the burst the controller sees the displaced state and must
re-converge -- exactly a disturbance-rejection test.

Two modes:
- eval (reproducible): fix ``push_step`` and ``direction``.
- training (domain randomization): ``push_step="random"`` + ``random_dir=True``
  re-samples timing/direction every episode so the policy learns to be robust.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import gymnasium as gym


class PushDisturbanceWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        push_step: Union[int, Sequence[int], str] = "random",
        duration: int = 3,
        strength: float = 1.0,
        direction: Optional[Sequence[float]] = None,
        random_dir: bool = True,
        window: Sequence[int] = (8, 35),  # random push happens within this step range
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.push_step = push_step
        self.duration = duration
        self.strength = strength
        self.fixed_direction = None if direction is None else np.asarray(direction, dtype=np.float32)
        self.random_dir = random_dir
        self.window = window
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._push_starts = []
        self._cur_dir = None

    def _sample_dir(self) -> np.ndarray:
        if self.fixed_direction is not None and not self.random_dir:
            d = self.fixed_direction
        else:
            d = self.rng.normal(size=3)
        n = np.linalg.norm(d)
        return (d / n).astype(np.float32) if n > 1e-8 else np.array([1, 0, 0], np.float32)

    def reset(self, **kwargs):
        self._t = 0
        if self.push_step == "random":
            self._push_starts = [int(self.rng.integers(self.window[0], self.window[1]))]
        elif isinstance(self.push_step, int):
            self._push_starts = [self.push_step]
        else:
            self._push_starts = list(self.push_step)
        self._cur_dir = self._sample_dir()
        return self.env.reset(**kwargs)

    def is_pushing(self) -> bool:
        return any(s <= self._t < s + self.duration for s in self._push_starts)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        pushing = self.is_pushing()
        if pushing:
            # exogenous shove: override the translational command for this burst
            action[:3] = self._cur_dir * self.strength
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["disturbance_active"] = bool(pushing)
        self._t += 1
        return obs, reward, terminated, truncated, info
