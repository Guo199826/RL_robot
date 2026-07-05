"""Seed an off-policy replay buffer with scripted demonstrations (SACfD-style).

Why this exists: BC-warm-starting *only* the actor is fragile -- the critic
starts random, so the first RL updates pull the actor away from the cloned
(good) policy and success collapses. The robust fix for off-policy RL is to put
the demonstrations *in the replay buffer* so the critic is grounded in
successful transitions from step 0. We reuse SB3's own rollout-collection
pattern (VecEnv step -> replay_buffer.add) so HER relabeling works unchanged.
"""
from __future__ import annotations

import numpy as np

from ..controllers import ScriptedPickPlace


def seed_replay_buffer_with_scripted(model, n_episodes: int = 200, max_steps: int = 50,
                                     verbose: int = 1) -> int:
    """Fill ``model.replay_buffer`` with scripted-expert transitions.

    Steps the model's own VecEnv with the scripted controller and adds each
    transition exactly like ``OffPolicyAlgorithm.collect_rollouts`` does, so the
    (HER) replay buffer stays internally consistent.
    """
    venv = model.get_env()
    ctrl = ScriptedPickPlace()
    obs = venv.reset()
    ctrl.reset()
    added = 0
    successes = 0
    for _ in range(n_episodes):
        for _ in range(max_steps):
            single = {k: v[0] for k, v in obs.items()}
            a = np.asarray(ctrl.act(single), dtype=np.float32)
            action = a[None, :]
            new_obs, reward, done, infos = venv.step(action)
            model.replay_buffer.add(obs, new_obs, action, reward, done, infos)
            added += 1
            obs = new_obs
            if done[0]:
                successes += int(infos[0].get("is_success", 0) > 0)
                ctrl.reset()  # VecEnv has already auto-reset obs
                break
    if verbose:
        print(f"[demo-buffer] seeded {added} transitions from {n_episodes} scripted episodes "
              f"({successes} successful); buffer size now {model.replay_buffer.size()}")
    return added
