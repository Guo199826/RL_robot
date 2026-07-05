"""Collect demonstrations from the scripted controller.

The scripted pick-and-place controller is a free source of (near-)optimal
demonstrations -- we use it instead of teleoperation. Demonstrations are stored
as parallel arrays of the flat ``observation`` dict and the 4-D actions.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

from ..common.envs import make_env
from ..controllers import ScriptedPickPlace


def collect_demos(
    n_episodes: int = 200,
    max_steps: int = 60,
    task: str = "FetchPickAndPlace-v4",
    only_successful: bool = True,
    out_path: str = "results/demos/scripted_demos.npz",
) -> Dict[str, np.ndarray]:
    env = make_env(task, reward_type="sparse")
    ctrl = ScriptedPickPlace()

    obs_buf, ag_buf, dg_buf, act_buf = [], [], [], []
    n_kept = 0
    for ep in range(n_episodes):
        ctrl.reset()
        obs, _ = env.reset()
        ep_o, ep_ag, ep_dg, ep_a = [], [], [], []
        success = False
        for _ in range(max_steps):
            a = ctrl.act(obs)
            ep_o.append(obs["observation"].copy())
            ep_ag.append(obs["achieved_goal"].copy())
            ep_dg.append(obs["desired_goal"].copy())
            ep_a.append(np.asarray(a, dtype=np.float32))
            obs, _, term, trunc, info = env.step(a)
            if info.get("is_success", 0) > 0:
                success = True
                break
            if term or trunc:
                break
        if success or not only_successful:
            obs_buf.extend(ep_o)
            ag_buf.extend(ep_ag)
            dg_buf.extend(ep_dg)
            act_buf.extend(ep_a)
            n_kept += 1
    env.close()

    data = {
        "observation": np.asarray(obs_buf, dtype=np.float32),
        "achieved_goal": np.asarray(ag_buf, dtype=np.float32),
        "desired_goal": np.asarray(dg_buf, dtype=np.float32),
        "actions": np.asarray(act_buf, dtype=np.float32),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(out_path, **data)
        print(f"[demos] kept {n_kept}/{n_episodes} episodes, "
              f"{len(data['actions'])} transitions -> {out_path}")
    return data


if __name__ == "__main__":
    collect_demos()
