# debug_rollout.py
import argparse
import time
import numpy as np
from stable_baselines3 import SAC

from assistive_fetch.envs import make_assistive_fetch_env


def zero_policy(obs, env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def random_policy(obs, env):
    return env.action_space.sample().astype(np.float32)


def run_episode(env, policy_fn, max_steps=200, sleep=0.03, verbose=True):
    obs, info = env.reset()
    ep_reward = 0.0

    for t in range(max_steps):
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if verbose:
            print(
                f"[t={t:03d}] "
                f"reward={reward: .3f} | "
                f"dist={info.get('assist_reward/dist', np.nan): .3f} | "
                f"success={info.get('assist_reward/success', np.nan): .1f} | "
                f"phase={info.get('human_phase', np.nan): .1f} | "
                f"xy_prepush={info.get('xy_dist_to_prepush', np.nan): .3f} | "
                f"gobj_xy={info.get('gripper_object_xy_dist', np.nan): .3f} | "
                f"safe_z_err={info.get('safe_z_error', np.nan): .3f} | "
                f"push_z_err={info.get('push_z_error', np.nan): .3f} | "
                f"assist={info.get('assist_action_norm', np.nan): .3f} | "
                f"human={info.get('human_action_norm', np.nan): .3f} | "
                f"full={info.get('full_action_norm', np.nan): .3f}"
            )

        if sleep > 0:
            time.sleep(sleep)

        if terminated or truncated:
            break

    print(f"\nEpisode return: {ep_reward:.3f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="FetchPush-v4")
    parser.add_argument("--mode", type=str, default="heuristic", choices=["heuristic", "random", "model"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.03)

    parser.add_argument("--human_gain", type=float, default=1.0)
    parser.add_argument("--assist_scale", type=float, default=0.15)
    parser.add_argument("--success_bonus", type=float, default=10.0)
    parser.add_argument("--assist_cost_coef", type=float, default=0.005)
    parser.add_argument("--dist_weight", type=float, default=1.0)

    parser.add_argument("--wrapper_type", type=str, default="fetchpush_two_stage")
    args = parser.parse_args()

    env = make_assistive_fetch_env(
        env_id=args.env_id,
        render_mode="human",
        wrapper_type=args.wrapper_type,
        human_gain=args.human_gain,
        assist_scale=args.assist_scale,
        success_bonus=args.success_bonus,
        assist_cost_coef=args.assist_cost_coef,
        dist_weight=args.dist_weight,
    )

    if args.mode == "heuristic":
        policy_fn = zero_policy
    elif args.mode == "random":
        policy_fn = random_policy
    else:
        assert args.model_path is not None, "--model_path is required in model mode"
        model = SAC.load(args.model_path, env=env)

        def policy_fn(obs, env):
            action, _ = model.predict(obs, deterministic=True)
            return action.astype(np.float32)

    for ep in range(args.episodes):
        print(f"\n========== Episode {ep + 1} / {args.episodes} ==========")
        run_episode(env, policy_fn, max_steps=args.max_steps, sleep=args.sleep, verbose=True)

    env.close()


if __name__ == "__main__":
    main()