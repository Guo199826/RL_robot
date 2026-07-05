"""mc: a small motion-control + RL framework for the FetchPickAndPlace demo.

This package collects the reusable pieces that the interview demo is built on:

- ``mc.common``      : env factory, metrics, training callbacks, plotting, video.
- ``mc.controllers`` : the controller matrix (PID / scripted / RL-tuned PID / Residual RL).
- ``mc.sim2real``    : domain-randomization and perturbation wrappers.
- ``mc.imitation``   : scripted-demo collection + BC pretraining helpers.

The story line is "classical control -> RL -> hybrid (best of both)".
"""

__all__ = ["common", "controllers", "sim2real", "imitation"]
