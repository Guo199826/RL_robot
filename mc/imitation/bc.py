"""Behavior cloning into a Stable-Baselines3 SAC actor.

We supervise the SAC actor's (squashed) mean action to match the scripted
demonstrations, then hand the warm-started model to ``model.learn`` for RL
fine-tuning. This is the "imitation + RL" recipe: BC gets the policy into a good
region cheaply, RL then improves beyond the (sub-optimal) demonstrator.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch as th


def _make_obs_batch(demos: Dict[str, np.ndarray], idx: np.ndarray) -> dict:
    return {
        "observation": demos["observation"][idx],
        "achieved_goal": demos["achieved_goal"][idx],
        "desired_goal": demos["desired_goal"][idx],
    }


def behavior_clone(
    model,
    demos: Dict[str, np.ndarray],
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: int = 1,
):
    """Warm-start ``model`` (SAC w/ MultiInputPolicy) by cloning ``demos``."""
    actor = model.actor
    device = model.device
    optimizer = th.optim.Adam(actor.parameters(), lr=lr)

    actions = th.as_tensor(demos["actions"], dtype=th.float32, device=device)
    n = actions.shape[0]
    n_batches = max(1, n // batch_size)

    actor.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            obs_np = _make_obs_batch(demos, idx)
            obs_t, _ = model.policy.obs_to_tensor(obs_np)
            mean_actions, _, _ = actor.get_action_dist_params(obs_t)
            pred = th.tanh(mean_actions)  # squashed action in [-1, 1]
            target = actions[idx]
            loss = th.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"[bc] epoch {epoch:>3d}  mse={epoch_loss / n_batches:.4f}")
    actor.eval()
    return model


def behavior_clone_ppo(
    model,
    demos: Dict[str, np.ndarray],
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    verbose: int = 1,
):
    """Warm-start a PPO ``ActorCriticPolicy`` by cloning ``demos``.

    PPO uses an unsquashed diagonal-Gaussian over Box actions, so we supervise
    the distribution *mean* (MSE to the demo action). This trains the shared
    feature extractor + policy net; the value head is left to RL fine-tuning.
    Note: PPO is on-policy and cannot reuse the demos in its own updates, so the
    BC init tends to drift once fine-tuning starts (cf. off-policy SAC).
    """
    policy = model.policy
    device = model.device
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)

    actions = th.as_tensor(demos["actions"], dtype=th.float32, device=device)
    n = actions.shape[0]
    n_batches = max(1, n // batch_size)

    policy.set_training_mode(True)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            obs_t, _ = policy.obs_to_tensor(_make_obs_batch(demos, idx))
            dist = policy.get_distribution(obs_t)
            pred = dist.distribution.mean  # Gaussian mean action
            loss = th.nn.functional.mse_loss(pred, actions[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"[bc-ppo] epoch {epoch:>3d}  mse={epoch_loss / n_batches:.4f}")
    policy.set_training_mode(False)
    return model
