"""Underactuated pendulum swing-up: the torque-control case where classical
model-based control (energy shaping + LQR) is excellent under the *nominal*
model but degrades under parameter uncertainty, while RL stays robust.

This is the honest "RL beats model-based" regime: the task is torque-limited and
underactuated, so a plain feedback controller provably *cannot* swing up -- you
need either an accurate model (energy shaping) or learning.
"""
from .env import make_pendulum, PendulumEnv, NOMINAL  # noqa: F401
