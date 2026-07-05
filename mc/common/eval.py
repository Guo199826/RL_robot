"""Unified evaluation harness shared by every controller and policy.

Anything that maps ``obs -> action`` (a ``Controller`` instance, a bare
callable, or a Stable-Baselines3 model via ``model_policy``) can be evaluated
here, producing a :class:`~mc.common.metrics.RolloutTrace` that the metrics and
plotting modules consume.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .envs import parse_obs
from .metrics import RolloutTrace, summarize


def model_policy(model, deterministic: bool = True) -> Callable:
    """Adapt an SB3 model into a ``policy(obs) -> action`` callable."""
    def _policy(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    return _policy


def evaluate(
    env,
    controller,
    n_episodes: int = 20,
    max_steps: int = 60,
    collect_first_error_trace: bool = True,
) -> Tuple[RolloutTrace, Optional[list]]:
    """Run ``controller`` for ``n_episodes`` and collect a metrics trace.

    ``controller`` may be a ``Controller`` (has ``.reset()``/``.act()``) or a
    plain callable ``policy(obs) -> action``.

    Returns ``(trace, first_episode_error_curve)`` where the error curve is the
    per-step goal-distance of the first episode (handy for the error-vs-time
    overlay plot).
    """
    has_reset = hasattr(controller, "reset")
    act = controller.act if hasattr(controller, "act") else controller

    trace = RolloutTrace()
    first_error_curve = None

    for ep in range(n_episodes):
        if has_reset:
            controller.reset()
        obs, _ = env.reset()
        ep_errors = []
        success = False
        steps = 0
        for t in range(max_steps):
            s = parse_obs(obs)
            err = float(np.linalg.norm(s.desired_goal - s.achieved_goal))
            action = np.asarray(act(obs), dtype=np.float32)

            trace.errors.append(err)
            trace.actions.append(action.copy())
            trace.ee_pos.append(s.grip_pos.copy())
            ep_errors.append(err)

            obs, _, term, trunc, info = env.step(action)
            steps = t + 1
            if info.get("is_success", 0) > 0:
                success = True
                break
            if term or trunc:
                break

        trace.successes.append(success)
        trace.ep_lengths.append(steps)
        if ep == 0 and collect_first_error_trace:
            first_error_curve = ep_errors

    return trace, first_error_curve


def evaluate_and_summarize(env, controller, name: str, **kw):
    trace, curve = evaluate(env, controller, **kw)
    metrics = summarize(trace)
    metrics["controller"] = name
    return metrics, curve
