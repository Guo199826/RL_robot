# assistive_fetch/wrappers.py
import gymnasium as gym
import numpy as np


class AssistiveSharedControlWrapper(gym.Wrapper):
    """
    Shared-control wrapper for goal-based Fetch envs.
    Agent outputs assist action only.
    Final action = human_intent_action + assist_action, with clipping.
    """

    def __init__(
        self,
        env,
        human_gain=0.6,
        assist_scale=0.5,
        smoothness_coef=0.05,
        effort_coef=0.05,
        overassist_coef=0.1,
        success_bonus=5.0,
        dist_weight=1.0,
        assist_cost_coef=0.01,
        use_dense_reward=True,
    ):
        super().__init__(env)
        self.human_gain = human_gain
        self.assist_scale = assist_scale
        self.smoothness_coef = smoothness_coef
        self.effort_coef = effort_coef
        self.overassist_coef = overassist_coef
        self.success_bonus = success_bonus
        self.dist_weight = dist_weight
        self.assist_cost_coef = assist_cost_coef
        self.use_dense_reward = use_dense_reward

        self.prev_assist_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_human_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_full_action = np.zeros(self.action_space.shape, dtype=np.float32)

        obs_space = self.observation_space
        assert isinstance(obs_space, gym.spaces.Dict), "Fetch env should be Dict observation."

        base_obs_dim = obs_space["observation"].shape[0]
        goal_dim = obs_space["desired_goal"].shape[0]
        act_dim = self.action_space.shape[0]

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(base_obs_dim + act_dim + act_dim + goal_dim,),
                    dtype=np.float32,
                ),
                "achieved_goal": obs_space["achieved_goal"],
                "desired_goal": obs_space["desired_goal"],
            }
        )

    # def _compute_human_intent_action(self, obs_dict):
    #     """
    #     Simple heuristic human intent:
    #     move gripper/object toward desired goal in Cartesian xyz;
    #     gripper open/close stays near zero for simplicity.
    #     """
    #     achieved = obs_dict["achieved_goal"]
    #     desired = obs_dict["desired_goal"]

    #     delta = desired - achieved
    #     human_xyz = self.human_gain * delta

    #     human_action = np.zeros(self.action_space.shape, dtype=np.float32)

    #     # Fetch action dim is typically 4: dx, dy, dz, gripper
    #     n = min(3, human_action.shape[0])
    #     human_action[:n] = human_xyz[:n]

    #     # optional simple gripper heuristic
    #     if human_action.shape[0] > 3:
    #         human_action[3] = 0.0

    #     return np.clip(human_action, -1.0, 1.0)
    def _compute_human_intent_action(self, obs_dict):
        obs_vec = obs_dict["observation"]
        gripper_pos = obs_vec[:3].copy()
        object_pos = obs_dict["achieved_goal"].copy()
        target_pos = obs_dict["desired_goal"].copy()

        obj_to_goal = target_pos - object_pos
        obj_to_goal_xy = obj_to_goal[:2]
        norm_xy = np.linalg.norm(obj_to_goal_xy)

        if norm_xy < 1e-6:
            push_dir_xy = np.zeros(2, dtype=np.float32)
        else:
            push_dir_xy = obj_to_goal_xy / norm_xy

        behind_offset = 0.06
        approach_threshold = 0.04
        gripper_height_target = 0.425

        pre_push_xy = object_pos[:2] - behind_offset * push_dir_xy
        pre_push_pos = np.array(
            [pre_push_xy[0], pre_push_xy[1], gripper_height_target],
            dtype=np.float32,
        )

        approach_error = pre_push_pos - gripper_pos
        approach_dist = np.linalg.norm(approach_error[:2])

        human_action = np.zeros(self.action_space.shape, dtype=np.float32)

        if approach_dist > approach_threshold:
            xyz_cmd = self.human_gain * approach_error
        else:
            push_cmd = np.zeros(3, dtype=np.float32)
            push_cmd[0] = self.human_gain * obj_to_goal_xy[0]
            push_cmd[1] = self.human_gain * obj_to_goal_xy[1]
            push_cmd[2] = 0.5 * (gripper_height_target - gripper_pos[2])

            align_xy = object_pos[:2] - gripper_pos[:2]
            push_cmd[0] += 0.3 * align_xy[0]
            push_cmd[1] += 0.3 * align_xy[1]

            xyz_cmd = push_cmd

        human_action[:3] = xyz_cmd[:3]

        if human_action.shape[0] > 3:
            human_action[3] = 0.0

        return np.clip(human_action, -1.0, 1.0)


    def _augment_obs(self, obs_dict):
        human_action = self._compute_human_intent_action(obs_dict)
        goal_error = obs_dict["desired_goal"] - obs_dict["achieved_goal"]

        aug_obs = np.concatenate(
            [
                obs_dict["observation"],
                human_action,
                self.prev_assist_action,
                goal_error,
            ],
            axis=0,
        ).astype(np.float32)

        self.last_human_action = human_action

        return {
            "observation": aug_obs,
            "achieved_goal": obs_dict["achieved_goal"].astype(np.float32),
            "desired_goal": obs_dict["desired_goal"].astype(np.float32),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_assist_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_human_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_full_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self._augment_obs(obs), info

    def step(self, assist_action):
        assist_action = np.asarray(assist_action, dtype=np.float32)
        assist_action = np.clip(assist_action, -1.0, 1.0)

        raw_obs = self.env.unwrapped._get_obs()
        human_action = self._compute_human_intent_action(raw_obs)

        scaled_assist = self.assist_scale * assist_action
        full_action = np.clip(human_action + scaled_assist, -1.0, 1.0)

        obs, _, terminated, truncated, info = self.env.step(full_action)

        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        dist = np.linalg.norm(desired - achieved)

        success = float(info.get("is_success", dist < 0.05))

        # dense_term = -dist if self.use_dense_reward else base_reward
        # smoothness_penalty = self.smoothness_coef * np.linalg.norm(scaled_assist - self.prev_assist_action)
        # effort_penalty = self.effort_coef * np.linalg.norm(scaled_assist)
        # overassist_penalty = self.overassist_coef * np.linalg.norm(
        #     np.maximum(np.abs(scaled_assist) - np.abs(human_action), 0.0)
        # )

        dist_term = -self.dist_weight * dist
        success_term = self.success_bonus * success
        assist_cost_term = -self.assist_cost_coef * np.linalg.norm(scaled_assist)

        reward = dist_term + success_term + assist_cost_term

        # reward = (
        #     dense_term
        #     + self.success_bonus * success
        #     - smoothness_penalty
        #     - effort_penalty
        #     - overassist_penalty
        # )

        # info["assist_reward/base_reward"] = float(base_reward)
        info["assist_reward/dist"] = float(dist)
        info["assist_reward/success"] = float(success)
        info["assist_reward/total"] = float(reward)
        info["assist_reward/dist_term"] = float(dist_term)
        info["assist_reward/success_term"] = float(success_term)
        info["assist_reward/assist_cost_term"] = float(assist_cost_term)
        info["assist_reward/stage1_shaping_term"] = 0.0
        # info["assist_reward/smoothness_penalty"] = float(smoothness_penalty)
        # info["assist_reward/effort_penalty"] = float(effort_penalty)
        # info["assist_reward/overassist_penalty"] = float(overassist_penalty)
        info["assist_action_norm"] = float(np.linalg.norm(scaled_assist))
        info["human_action_norm"] = float(np.linalg.norm(human_action))
        info["full_action_norm"] = float(np.linalg.norm(full_action))

        self.prev_assist_action = scaled_assist.copy()
        self.last_full_action = full_action.copy()

        return self._augment_obs(obs), reward, terminated, truncated, info