"""Classical controllers for the pendulum, built from a fixed *nominal* model.

- ``PID``        : naive feedback torque ``-(Kp*theta + Kd*thetadot)``. Being
  torque-limited and underactuated, it *cannot* swing up -- it just saturates
  and stalls near the bottom. The point: plain feedback is not enough here.
- ``EnergyLQR``  : the textbook model-based swing-up. Energy shaping pumps the
  pendulum to the homoclinic (upright) energy, then an LQR (linearized about
  upright) catches and balances it. Excellent under the nominal model; it
  mistunes its energy target and gains when m/l/g differ -> degrades under
  randomization.
- ``ResidualController`` : EnergyLQR base + a learned residual torque (hybrid).

All ``act`` methods return a normalized torque in ``[-1, 1]``.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from .env import NOMINAL, MAX_TORQUE


def _lqr_gain(m, l, g, q=(4.0, 0.3), r=1.0):
    """LQR gain for the pendulum linearized about upright (theta=0)."""
    from scipy.linalg import solve_continuous_are

    I = m * l ** 2 / 3.0
    A = np.array([[0.0, 1.0], [3.0 * g / (2.0 * l), 0.0]])
    B = np.array([[0.0], [1.0 / I]])
    Q = np.diag(q)
    R = np.array([[r]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K.ravel()  # [k_theta, k_thetadot]


class PID:
    name = "PID (naive)"

    def __init__(self, kp=8.0, kd=2.0):
        self.kp, self.kd = kp, kd

    def reset(self):
        pass

    def act(self, state):
        theta, thetadot = state
        u = -(self.kp * theta + self.kd * thetadot)
        return np.array([np.clip(u / MAX_TORQUE, -1.0, 1.0)], dtype=np.float32)


class EnergyLQR:
    name = "Energy+LQR (model-based)"

    def __init__(self, ke=0.9, switch_angle=0.5, switch_speed=1.5, params=None):
        p = params or NOMINAL
        self.m, self.l, self.g = p["m"], p["l"], p["g"]
        self.ke = ke
        self.switch_angle = switch_angle
        self.switch_speed = switch_speed
        self.I = self.m * self.l ** 2 / 3.0
        self.E_des = self.m * self.g * (self.l / 2.0)      # upright potential
        self.K = _lqr_gain(self.m, self.l, self.g)

    def reset(self):
        pass

    def _energy(self, theta, thetadot):
        return 0.5 * self.I * thetadot ** 2 + self.m * self.g * (self.l / 2.0) * np.cos(theta)

    def act(self, state):
        theta, thetadot = state
        if abs(theta) < self.switch_angle and abs(thetadot) < self.switch_speed:
            u = -float(self.K @ np.array([theta, thetadot]))      # LQR balance
        else:
            E = self._energy(theta, thetadot)
            sign = np.sign(thetadot) if thetadot != 0 else 1.0
            u = self.ke * (self.E_des - E) * sign                  # energy pumping
        return np.array([np.clip(u / MAX_TORQUE, -1.0, 1.0)], dtype=np.float32)


class ResidualController:
    name = "Model-based + RL residual"

    def __init__(self, base, model, residual_scale=0.5):
        self.base = base
        self.model = model
        self.residual_scale = residual_scale

    def reset(self):
        self.base.reset()

    def act_with_obs(self, state, obs):
        base = self.base.act(state)
        residual, _ = self.model.predict(obs, deterministic=True)
        u = base + self.residual_scale * np.asarray(residual, dtype=np.float32)
        return np.clip(u, -1.0, 1.0).astype(np.float32)


class ResidualEnv(gym.Wrapper):
    """Training env for residual RL: total torque = base(state) + scale*action.

    The RL agent only has to learn the *correction* to a model-based controller,
    which is easy to learn and inherits the base's nominal competence -- the
    hybrid that is robust both at nominal and under randomization.
    """

    def __init__(self, env, base, residual_scale=0.5):
        super().__init__(env)
        self.base = base
        self.residual_scale = residual_scale

    def reset(self, **kwargs):
        self.base.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        state = self.env.unwrapped.control_state()
        base = self.base.act(state)
        total = np.clip(base + self.residual_scale * np.asarray(action, dtype=np.float32),
                        -1.0, 1.0)
        return self.env.step(total)
