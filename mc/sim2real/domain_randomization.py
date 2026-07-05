"""Domain randomization & hardware-defect wrappers for sim2real studies.

The sim2real claim in the JD is backed by a concrete experiment: train a policy
with vs. without domain randomization (DR), then evaluate *both* under perturbed
dynamics/sensing. DR-trained policies should degrade far less -- that gap is the
"robustness" story.

Wrappers (compose them as needed):

- ``DomainRandomizationWrapper`` : per-episode physics randomization
  (object mass & friction via MuJoCo when reachable, plus a model-agnostic
  per-episode actuator-gain mismatch + per-step process noise).
- ``ObservationNoiseWrapper``    : Gaussian sensing noise on the state.
- ``ActionLatencyWrapper``       : fixed control delay (action buffering).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        mass_range=(0.5, 2.0),       # multiplicative on object mass
        friction_range=(0.5, 1.5),  # multiplicative on sliding friction
        action_gain_range=(0.8, 1.2),  # actuator gain mismatch per episode
        process_noise=0.01,         # per-step Gaussian noise added to action
        object_body="object0",
        seed=None,
    ):
        super().__init__(env)
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.action_gain_range = action_gain_range
        self.process_noise = process_noise
        self.object_body = object_body
        self.rng = np.random.default_rng(seed)
        self._action_gain = 1.0

        self._model = getattr(env.unwrapped, "model", None)
        self._body_id = self._find_body_id()
        self._nominal_mass = None
        # Snapshot nominal physics so per-episode randomization always scales
        # from the *original* values (otherwise repeated multiplicative scaling
        # compounds and friction/mass drifts over many resets).
        self._nominal_friction = None
        if self._model is not None:
            if self._body_id is not None:
                self._nominal_mass = float(self._model.body_mass[self._body_id])
            self._nominal_friction = self._model.geom_friction[:, 0].copy()

    def _find_body_id(self):
        if self._model is None:
            return None
        try:
            import mujoco

            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.object_body)
        except Exception:
            return None

    def _randomize_physics(self):
        if self._model is None:
            return
        try:
            if self._body_id is not None and self._nominal_mass:
                m = self.rng.uniform(*self.mass_range)
                self._model.body_mass[self._body_id] = self._nominal_mass * m
            if self._nominal_friction is not None:
                fr = self.rng.uniform(*self.friction_range)
                self._model.geom_friction[:, 0] = np.clip(
                    self._nominal_friction * fr, 1e-4, None
                )
        except Exception:
            pass

    def reset(self, **kwargs):
        self._action_gain = self.rng.uniform(*self.action_gain_range)
        self._randomize_physics()
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32) * self._action_gain
        if self.process_noise > 0:
            action = action + self.rng.normal(0.0, self.process_noise, size=action.shape)
        return self.env.step(np.clip(action, -1.0, 1.0).astype(np.float32))


class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_sigma=0.005, goal_sigma=0.002, seed=None):
        super().__init__(env)
        self.obs_sigma = obs_sigma
        self.goal_sigma = goal_sigma
        self.rng = np.random.default_rng(seed)

    def observation(self, obs):
        out = dict(obs)
        out["observation"] = (
            obs["observation"] + self.rng.normal(0, self.obs_sigma, obs["observation"].shape)
        ).astype(np.float32)
        out["achieved_goal"] = (
            obs["achieved_goal"] + self.rng.normal(0, self.goal_sigma, obs["achieved_goal"].shape)
        ).astype(np.float32)
        # desired_goal is a planned target -> kept noise-free
        return out


class ActionLatencyWrapper(gym.Wrapper):
    """Apply each action ``delay`` steps late (control/communication latency)."""

    def __init__(self, env, delay=1):
        super().__init__(env)
        self.delay = delay
        self._buffer = deque(maxlen=delay + 1)

    def reset(self, **kwargs):
        self._buffer.clear()
        for _ in range(self.delay):
            self._buffer.append(np.zeros(self.env.action_space.shape, dtype=np.float32))
        return self.env.reset(**kwargs)

    def step(self, action):
        self._buffer.append(np.asarray(action, dtype=np.float32))
        delayed = self._buffer.popleft()
        return self.env.step(delayed)


class RandomActionLatencyWrapper(gym.Wrapper):
    """Per-episode random control delay in ``[0, max_delay]`` steps.

    Used for *training* a policy that is robust across a range of latencies
    (a fixed-gain linear controller cannot be optimal for the whole range).
    """

    def __init__(self, env, max_delay=3, seed=None):
        super().__init__(env)
        self.max_delay = max_delay
        self.rng = np.random.default_rng(seed)
        self._buffer = deque()
        self._delay = 0

    def reset(self, **kwargs):
        self._delay = int(self.rng.integers(0, self.max_delay + 1))
        self._buffer = deque(maxlen=self._delay + 1)
        for _ in range(self._delay):
            self._buffer.append(np.zeros(self.env.action_space.shape, dtype=np.float32))
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._delay == 0:
            return self.env.step(action)
        self._buffer.append(action)
        return self.env.step(self._buffer.popleft())


def make_randomized_env(
    env,
    dr=True,
    obs_noise=True,
    latency=0,
    seed=None,
):
    """Convenience composition used by the sim2real experiment."""
    if dr:
        env = DomainRandomizationWrapper(env, seed=seed)
    if obs_noise:
        env = ObservationNoiseWrapper(env, seed=seed)
    if latency and latency > 0:
        env = ActionLatencyWrapper(env, delay=latency)
    return env
