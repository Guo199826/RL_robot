"""Scripted pick-and-place controller: a model-based state machine.

This is the "traditional motion control" upper baseline. It encodes the task
structure as an explicit finite-state machine (approach -> descend -> grasp ->
transport -> release) and uses proportional task-space feedback within each
phase. No learning involved -- it works out of the box at ~100% success and
serves two purposes:

1. a strong classical baseline in the controller matrix, and
2. the *base policy* that Residual RL and BC build on top of.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from .base import Controller
from ..common.envs import parse_obs


class Phase(Enum):
    APPROACH = 0   # move above the object, gripper open
    DESCEND = 1    # lower onto the object, gripper open
    GRASP = 2      # close the gripper
    TRANSPORT = 3  # carry the object to the goal, gripper closed


class ScriptedPickPlace(Controller):
    name = "Scripted (model-based)"

    def __init__(self, gain=6.0, hover_height=0.05, reach_tol=0.02,
                 align_tol=0.02, grasp_open_thresh=0.05):
        self.gain = gain
        self.hover_height = hover_height
        self.reach_tol = reach_tol
        self.align_tol = align_tol
        self.grasp_open_thresh = grasp_open_thresh
        self.reset()

    def reset(self) -> None:
        self._phase = Phase.APPROACH

    def act(self, obs: dict) -> np.ndarray:
        s = parse_obs(obs)
        grip, obj, rel = s.grip_pos, s.object_pos, s.object_rel_pos
        goal = s.desired_goal
        above = obj + np.array([0.0, 0.0, self.hover_height])

        action = np.zeros(4, dtype=np.float32)

        # --- finite-state machine ---------------------------------------
        if self._phase == Phase.APPROACH:
            # align in XY above the object before going down
            action[:3] = (above - grip) * self.gain
            action[3] = 1.0  # open
            if np.linalg.norm((above - grip)[:2]) < self.align_tol:
                self._phase = Phase.DESCEND

        elif self._phase == Phase.DESCEND:
            action[:3] = rel * self.gain
            action[3] = 1.0
            if np.linalg.norm(rel) < self.reach_tol:
                self._phase = Phase.GRASP

        elif self._phase == Phase.GRASP:
            action[:3] = rel * self.gain
            action[3] = -1.0  # close
            if s.gripper_opening < self.grasp_open_thresh:
                self._phase = Phase.TRANSPORT

        else:  # TRANSPORT
            action[:3] = (goal - obj) * self.gain
            action[3] = -1.0  # keep closed

        action[:3] = np.clip(action[:3], -1.0, 1.0)
        return action

    def base_action(self, obs: dict) -> np.ndarray:
        """Alias used by ResidualRLWrapper (the scripted action is the base)."""
        return self.act(obs)
