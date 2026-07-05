"""The controller matrix: classical -> RL -> hybrid.

- ``base.Controller``        : common interface (``reset`` + ``act``).
- ``pid.TaskSpacePID``       : pure feedback PID in Cartesian task space.
- ``scripted.ScriptedPickPlace`` : a model-based state machine (plan + feedback).
- ``rl_pid.RLPIDTunerWrapper``   : RL outputs the PID gains (gain scheduling).
- ``residual.ResidualRLWrapper`` : classical base action + learned RL residual.
"""
from .base import Controller  # noqa: F401
from .pid import TaskSpacePID  # noqa: F401
from .scripted import ScriptedPickPlace  # noqa: F401
from .residual import ResidualRLWrapper  # noqa: F401
from .rl_pid import RLPIDTunerWrapper  # noqa: F401


def base_for_task(task: str) -> Controller:
    """Task-aware classical base controller (used as residual base + baseline).

    - Reach -> well-damped task-space PD (no object to grasp).
    - PickAndPlace (and anything else with an object) -> scripted pick-and-place.
    """
    if "Reach" in task:
        return TaskSpacePID(kp=8.0, ki=0.0, kd=1.0, gripper=0.0)
    return ScriptedPickPlace()


__all__ = [
    "Controller",
    "TaskSpacePID",
    "ScriptedPickPlace",
    "ResidualRLWrapper",
    "RLPIDTunerWrapper",
    "base_for_task",
]
