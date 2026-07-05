"""Reusable Stable-Baselines3 callbacks.

The old scripts each carried a near-identical ~90-line ``PlottingCallback``.
This module replaces all of them with two small, reusable callbacks:

- ``SuccessRateEvalCallback`` : periodically evaluates the greedy policy, logs
  the success rate vs timesteps and writes it to CSV. This is the single most
  important curve for comparing algorithms on a goal-conditioned task.
- ``CurveLoggerCallback``     : records reward + algo losses to CSV/PNG without
  any interactive ``plt.ion`` dependency (works headless).
"""
from __future__ import annotations

import csv
import os
from typing import Callable, List, Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateEvalCallback(BaseCallback):
    """Evaluate greedy success rate every ``eval_freq`` steps and log to CSV."""

    def __init__(
        self,
        eval_env_fn: Callable[[], "object"],
        eval_freq: int = 5000,
        n_eval_episodes: int = 20,
        max_steps: int = 60,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.csv_path = csv_path
        self.timesteps: List[int] = []
        self.success_rates: List[float] = []
        self._eval_env = None

    def _evaluate(self) -> float:
        if self._eval_env is None:
            self._eval_env = self.eval_env_fn()
        succ = 0
        for ep in range(self.n_eval_episodes):
            obs, _ = self._eval_env.reset()
            for _ in range(self.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = self._eval_env.step(action)
                if info.get("is_success", 0) > 0:
                    succ += 1
                    break
                if term or trunc:
                    break
        return succ / self.n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rate = self._evaluate()
            self.timesteps.append(int(self.num_timesteps))
            self.success_rates.append(rate)
            if self.verbose:
                print(f"[eval] step={self.num_timesteps:>8d}  success_rate={rate:.2%}")
            if self.csv_path:
                self._write_csv()
        return True

    def _write_csv(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "success_rate"])
            for t, r in zip(self.timesteps, self.success_rates):
                w.writerow([t, r])

    def _on_training_end(self) -> None:
        if self.csv_path:
            self._write_csv()
        if self._eval_env is not None:
            self._eval_env.close()


class CurveLoggerCallback(BaseCallback):
    """Log mean step reward and algo-specific losses to CSV (headless-safe)."""

    LOSS_KEYS = {
        "actor_loss": "train/actor_loss",
        "critic_loss": "train/critic_loss",
        "ent_coef": "train/ent_coef",
        "value_loss": "train/value_loss",
        "policy_loss": "train/policy_gradient_loss",
    }

    def __init__(self, log_freq: int = 1000, csv_path: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.csv_path = csv_path
        self.rows: List[dict] = []
        self._recent_rewards: List[float] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._recent_rewards.extend(np.asarray(rewards).reshape(-1).tolist())
        if self.n_calls % self.log_freq == 0:
            row = {"timesteps": int(self.num_timesteps)}
            if self._recent_rewards:
                row["mean_reward"] = float(np.mean(self._recent_rewards[-self.log_freq:]))
            logger = getattr(self.model, "logger", None)
            if logger is not None:
                for name, key in self.LOSS_KEYS.items():
                    val = logger.name_to_value.get(key)
                    if val is not None:
                        row[name] = float(val)
            self.rows.append(row)
        return True

    def _on_training_end(self) -> None:
        if self.csv_path and self.rows:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
            keys = sorted({k for r in self.rows for k in r})
            keys = ["timesteps"] + [k for k in keys if k != "timesteps"]
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
