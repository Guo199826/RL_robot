"""Quantitative motion-control metrics.

A controls interview cares about *numbers*, not just "it reached the goal".
This module turns a rollout (the per-step traces collected by the evaluation
harness) into the metrics a motion-control engineer would report:

- success rate           : task-level success (from ``info['is_success']``)
- tracking RMSE          : root-mean-square of the goal-distance error
- final / steady error   : error at the end of the episode
- settling time          : steps until the error stays within a tolerance band
- overshoot              : how far past the target the trajectory swings
- control energy         : sum of squared actions (actuation effort)
- action smoothness/jerk : RMS of consecutive action differences (chattering)
- path length            : end-effector path length (efficiency)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List

import numpy as np


@dataclass
class RolloutTrace:
    """Per-step traces collected during one (or many) evaluation episode(s)."""

    errors: List[float] = field(default_factory=list)       # goal distance per step
    actions: List[np.ndarray] = field(default_factory=list)  # action per step
    ee_pos: List[np.ndarray] = field(default_factory=list)   # end-effector position
    successes: List[bool] = field(default_factory=list)      # one bool per episode
    ep_lengths: List[int] = field(default_factory=list)      # steps per episode


def settling_time(errors: np.ndarray, tol: float = 0.05) -> int:
    """First step after which the error stays within ``tol`` forever.

    Returns ``len(errors)`` (i.e. "never settled") if it never stabilises.
    """
    n = len(errors)
    for i in range(n):
        if np.all(errors[i:] <= tol):
            return i
    return n


def overshoot(errors: np.ndarray) -> float:
    """Overshoot heuristic for a monotone-decreasing error signal.

    If the error ever increases again after first getting close to zero, the
    rise above the running minimum is reported as overshoot.
    """
    if len(errors) < 2:
        return 0.0
    running_min = np.minimum.accumulate(errors)
    return float(np.max(errors - running_min))


def path_length(ee_pos: np.ndarray) -> float:
    if len(ee_pos) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)))


def summarize(trace: RolloutTrace, settle_tol: float = 0.05) -> Dict[str, float]:
    """Reduce a (possibly multi-episode) trace to a flat dict of metrics."""
    errors = np.asarray(trace.errors, dtype=np.float64)
    actions = np.asarray(trace.actions, dtype=np.float64) if trace.actions else np.zeros((0, 1))
    ee_pos = np.asarray(trace.ee_pos, dtype=np.float64) if trace.ee_pos else np.zeros((0, 3))

    out: Dict[str, float] = {}
    out["success_rate"] = float(np.mean(trace.successes)) if trace.successes else float("nan")
    out["n_episodes"] = float(len(trace.successes))
    out["mean_ep_len"] = float(np.mean(trace.ep_lengths)) if trace.ep_lengths else float("nan")

    if len(errors):
        out["tracking_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        out["final_error"] = float(errors[-1])
        out["settling_time"] = float(settling_time(errors, settle_tol))
        out["overshoot"] = float(overshoot(errors))
    if len(actions):
        out["control_energy"] = float(np.mean(np.sum(actions ** 2, axis=1)))
        if len(actions) > 1:
            jerk = np.diff(actions, axis=0)
            out["action_jerk"] = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))
    if len(ee_pos):
        out["path_length"] = path_length(ee_pos)
    return out


def to_row(name: str, metrics: Dict[str, float]) -> Dict[str, float]:
    row = {"controller": name}
    row.update(metrics)
    return row


def asdict_trace(trace: RolloutTrace) -> dict:  # convenience for serialisation
    d = asdict(trace)
    d["actions"] = [np.asarray(a).tolist() for a in trace.actions]
    d["ee_pos"] = [np.asarray(p).tolist() for p in trace.ee_pos]
    return d
