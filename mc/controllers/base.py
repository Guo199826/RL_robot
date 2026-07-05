"""Common controller interface so the eval harness can treat them uniformly."""
from __future__ import annotations

import abc

import numpy as np


class Controller(abc.ABC):
    """A controller maps a Fetch observation dict to a 4-D action.

    Action layout: ``[dx, dy, dz, gripper]`` in ``[-1, 1]`` (mocap displacement
    command + gripper open(+1)/close(-1)).
    """

    name: str = "controller"

    def reset(self) -> None:  # stateful controllers (PID integral) override this
        pass

    @abc.abstractmethod
    def act(self, obs: dict) -> np.ndarray:
        ...

    def __call__(self, obs: dict) -> np.ndarray:
        return self.act(obs)
