"""Model factory: off-policy algorithms with HER, plus on-policy PPO.

Goal-conditioned manipulation with *sparse* rewards is the textbook use case for
Hindsight Experience Replay (HER). HER only works with off-policy algorithms
(SAC/TD3/DDPG), so PPO is trained on the *dense* reward instead -- a deliberate,
interview-worthy distinction handled here in one place.
"""
from __future__ import annotations

from typing import Callable, Optional

from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np

OFF_POLICY = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}
ON_POLICY = {"PPO": PPO}
ALL_ALGOS = list(OFF_POLICY) + list(ON_POLICY)


def needs_her(algo: str) -> bool:
    return algo in OFF_POLICY


def make_vec_env(env_fn: Callable[[int], "object"], n_envs: int = 1,
                 seed: Optional[int] = None):
    """Build a (Subproc/Dummy) VecEnv from ``env_fn(rank) -> gym.Env``.

    With 20 CPU cores available, running several envs in subprocesses massively
    speeds up sample collection (the main wall-clock bottleneck on CPU). Each
    rank gets a distinct seed so domain randomization differs across workers.
    HER works unchanged on a goal-conditioned VecEnv.
    """
    base = 0 if seed is None else int(seed)
    fns = [(lambda r=r: env_fn(base + r)) for r in range(n_envs)]
    if n_envs <= 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def build_model(
    algo: str,
    env,
    use_her: bool = True,
    learning_rate: float = 1e-3,
    seed: Optional[int] = None,
    verbose: int = 0,
    tensorboard_log: Optional[str] = None,
    buffer_size: int = 300_000,
    gradient_steps: int = 1,
    learning_starts: int = 100,
):
    """Construct a model for ``algo`` on a goal-conditioned ``env``.

    ``buffer_size`` caps the (pre-allocated) replay buffer. SB3 allocates the
    full buffer up front, so the 1e6 default eats a lot of RAM -- 3e5 is plenty
    for the modest step budgets here and keeps several runs memory-safe.
    """
    algo = algo.upper()
    common = dict(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        verbose=verbose,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )

    if algo in OFF_POLICY:
        cls = OFF_POLICY[algo]
        kwargs = dict(common)
        kwargs["buffer_size"] = buffer_size
        # With a VecEnv, gradient_steps=-1 does one update per collected
        # transition (preserves sample efficiency while collection runs in
        # parallel). Default 1 keeps the single-env behaviour unchanged.
        kwargs["gradient_steps"] = gradient_steps
        # HER can only sample once each env has finished >=1 episode, so
        # learning_starts must exceed one episode per parallel env.
        kwargs["learning_starts"] = learning_starts
        if use_her:
            from stable_baselines3 import HerReplayBuffer

            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            )
        # DDPG/TD3 benefit from exploration noise on deterministic policies
        if algo in ("TD3", "DDPG"):
            n_actions = env.action_space.shape[-1]
            kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        return cls(**kwargs)

    if algo in ON_POLICY:
        # PPO can't use HER; it is trained on the dense-reward env.
        return PPO(n_steps=2048, batch_size=256, **common)

    raise ValueError(f"unknown algo {algo!r}; choose from {ALL_ALGOS}")
