# assistive_fetch/fetch_push_wrapper.py
import numpy as np
from assistive_fetch.wrappers import AssistiveSharedControlWrapper


class FetchPushTwoStageAssistiveWrapper(AssistiveSharedControlWrapper):
    def __init__(
        self,
        env,
        human_gain=1.0,
        assist_scale=0.15,
        smoothness_coef=0.05,
        effort_coef=0.05,
        overassist_coef=0.1,
        success_bonus=10.0,
        assist_cost_coef=0.002,
        dist_weight=1.0,
        use_dense_reward=True,
        behind_offset=0.08,
        enter_push_threshold=0.03,
        leave_push_threshold=0.05,
        lateral_align_threshold=0.035,
        z_align_threshold=0.02,
        push_gain=1.5,
        gripper_height_target=0.425,
        z_gain=1.0,
        lateral_gain=0.5,
        stage1_shaping_coef=0.2,
    ):
        super().__init__(
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

        self.behind_offset = behind_offset
        self.enter_push_threshold = enter_push_threshold
        self.leave_push_threshold = leave_push_threshold
        self.lateral_align_threshold = lateral_align_threshold
        self.z_align_threshold = z_align_threshold
        self.push_gain = push_gain
        self.gripper_height_target = gripper_height_target
        self.z_gain = z_gain
        self.lateral_gain = lateral_gain
        self.stage1_shaping_coef = stage1_shaping_coef

        self.current_phase = 0
        self.debug_info = {}

    def _extract_positions(self, obs_dict):
        obs_vec = obs_dict["observation"]
        gripper_pos = obs_vec[:3].copy()
        object_pos = obs_dict["achieved_goal"].copy()
        target_pos = obs_dict["desired_goal"].copy()
        return gripper_pos, object_pos, target_pos

    def _compute_human_intent_action(self, obs_dict):
        gripper_pos, object_pos, target_pos = self._extract_positions(obs_dict)

        obj_to_goal = target_pos - object_pos
        obj_to_goal_xy = obj_to_goal[:2]
        norm_xy = np.linalg.norm(obj_to_goal_xy)

        if norm_xy < 1e-6:
            push_dir_xy = np.zeros(2, dtype=np.float32)
        else:
            push_dir_xy = obj_to_goal_xy / norm_xy

        pre_push_xy = object_pos[:2] - self.behind_offset * push_dir_xy

        safe_z = 0.50
        push_z = self.gripper_height_target

        xy_to_prepush = pre_push_xy - gripper_pos[:2]
        xy_dist_to_prepush = np.linalg.norm(xy_to_prepush)

        z_to_safe = safe_z - gripper_pos[2]
        z_to_push = push_z - gripper_pos[2]

        gripper_object_xy = object_pos[:2] - gripper_pos[:2]
        gripper_object_xy_dist = np.linalg.norm(gripper_object_xy)

        human_action = np.zeros(self.action_space.shape, dtype=np.float32)
        xyz_cmd = np.zeros(3, dtype=np.float32)

        lift_done_threshold = 0.015
        xy_done_threshold = 0.03
        descend_done_threshold = 0.015

        if not hasattr(self, "current_phase"):
            self.current_phase = 0.0

        # -------- state transitions with memory --------
        if self.current_phase == 0.0:
            if abs(z_to_safe) < lift_done_threshold:
                self.current_phase = 1.0

        elif self.current_phase == 1.0:
            if xy_dist_to_prepush < xy_done_threshold:
                self.current_phase = 2.0

        elif self.current_phase == 2.0:
            if abs(z_to_push) < descend_done_threshold:
                self.current_phase = 3.0

        elif self.current_phase == 3.0:
            # only fall back if badly misaligned
            if gripper_object_xy_dist > 0.12:
                self.current_phase = 1.0

        # -------- control per phase --------
        if self.current_phase == 0.0:
            xyz_cmd[0] = 1.0 * self.human_gain * xy_to_prepush[0]
            xyz_cmd[1] = 1.0 * self.human_gain * xy_to_prepush[1]
            xyz_cmd[2] = 6.0 * z_to_safe

        elif self.current_phase == 1.0:
            xyz_cmd[0] = 3.0 * self.human_gain * xy_to_prepush[0]
            xyz_cmd[1] = 3.0 * self.human_gain * xy_to_prepush[1]
            xyz_cmd[2] = 4.0 * z_to_safe

        elif self.current_phase == 2.0:
            xyz_cmd[0] = 0.5 * self.human_gain * xy_to_prepush[0]
            xyz_cmd[1] = 0.5 * self.human_gain * xy_to_prepush[1]
            xyz_cmd[2] = 3.0 * z_to_push

        else:  # phase 3 push
            push_cmd = np.zeros(3, dtype=np.float32)
            push_cmd[0] = self.push_gain * obj_to_goal_xy[0]
            push_cmd[1] = self.push_gain * obj_to_goal_xy[1]
            push_cmd[2] = self.z_gain * z_to_push

            push_cmd[0] += self.lateral_gain * gripper_object_xy[0]
            push_cmd[1] += self.lateral_gain * gripper_object_xy[1]

            xyz_cmd = push_cmd

        human_action[:3] = xyz_cmd[:3]

        if human_action.shape[0] > 3:
            human_action[3] = 0.0

        human_action = np.clip(human_action, -1.0, 1.0)

        self.debug_info = {
            "human_phase": float(self.current_phase),
            "xy_dist_to_prepush": float(xy_dist_to_prepush),
            "gripper_object_xy_dist": float(gripper_object_xy_dist),
            "safe_z_error": float(abs(z_to_safe)),
            "push_z_error": float(abs(z_to_push)),
        }

        return human_action

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

        dist_term = -self.dist_weight * dist
        success_term = self.success_bonus * success
        assist_cost_term = -self.assist_cost_coef * np.linalg.norm(scaled_assist)

        reward = dist_term + success_term + assist_cost_term

        info["assist_reward/dist"] = float(dist)
        info["assist_reward/success"] = float(success)
        info["assist_reward/total"] = float(reward)
        info["assist_reward/dist_term"] = float(dist_term)
        info["assist_reward/success_term"] = float(success_term)
        info["assist_reward/assist_cost_term"] = float(assist_cost_term)
        info["assist_reward/stage1_shaping_term"] = 0.0
        info["assist_action_norm"] = float(np.linalg.norm(scaled_assist))
        info["human_action_norm"] = float(np.linalg.norm(human_action))
        info["full_action_norm"] = float(np.linalg.norm(full_action))

        for k, v in self.debug_info.items():
            info[k] = v

        self.prev_assist_action = scaled_assist.copy()
        self.last_full_action = full_action.copy()

        return self._augment_obs(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_assist_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_human_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_full_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_dist = None

        self.current_phase = 0.0
        self.debug_info = {}

        return self._augment_obs(obs), info