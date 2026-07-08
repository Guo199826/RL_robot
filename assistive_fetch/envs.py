# assistive_fetch/envs.py
import gymnasium as gym
import gymnasium_robotics

from assistive_fetch.wrappers import AssistiveSharedControlWrapper
from assistive_fetch.fetch_push_wrapper import FetchPushTwoStageAssistiveWrapper


def make_assistive_fetch_env(
    env_id="FetchPush-v4",
    render_mode=None,
    wrapper_type="fetchpush_two_stage",
    human_gain=1.0,
    assist_scale=0.15,
    smoothness_coef=0.05,
    effort_coef=0.05,   
    overassist_coef=0.1,    
    success_bonus=10.0,
    assist_cost_coef=0.005,
    dist_weight=1.0,
    use_dense_reward=True,
):
    gym.register_envs(gymnasium_robotics)

    env = gym.make(env_id, render_mode=render_mode)

    if wrapper_type == "fetchpush_two_stage":
        env = FetchPushTwoStageAssistiveWrapper(
            env=env,
            human_gain=human_gain,
            assist_scale=assist_scale,
            smoothness_coef=smoothness_coef,
            effort_coef=effort_coef,
            overassist_coef=overassist_coef,
            success_bonus=success_bonus,
            assist_cost_coef=assist_cost_coef,
            dist_weight=dist_weight,
            use_dense_reward=use_dense_reward,
        )
    elif wrapper_type == "generic":
        env = AssistiveSharedControlWrapper(
            env=env,
            human_gain=human_gain,
            assist_scale=assist_scale,
            smoothness_coef=smoothness_coef,
            effort_coef=effort_coef,
            overassist_coef=overassist_coef,
            success_bonus=success_bonus,
            assist_cost_coef=assist_cost_coef,
            dist_weight=dist_weight,
        )
    else:
        raise ValueError(f"Unknown wrapper_type: {wrapper_type}")

    return env