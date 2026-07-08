# assistive_fetch/callbacks.py
import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback


class PlottingCallback(BaseCallback):
    """实时绘制 Assistive-Fetch 训练关键指标的回调 (SAC + HER)。

    仿照 robot_arm/robot_arm_v1.py 里的 PlottingCallback，但针对
    共享控制 (shared-control) 任务额外监控：成功率、到目标的距离、
    以及机器人辅助力 / 人类意图力的大小和各项奖励惩罚。
    """

    def __init__(
        self,
        check_freq: int = 1000,
        window: int = 100,
        save_path: str = "assistive_fetch_training_curves.png",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.window = window
        self.save_path = save_path

        # 时间轴
        self.all_steps = []

        # 主要指标
        self.all_rewards = []
        self.all_success = []
        self.all_dist = []
        self.all_assist_norm = []
        self.all_human_norm = []
        self.all_full_norm = []

        # reward 分项（每步 info 里的各项贡献，滑动窗口平滑后记录）
        self.reward_terms = ["dist_term", "success_term", "assist_cost_term", "stage1_shaping_term"]
        self.all_reward_terms = {term: [] for term in self.reward_terms}

        # 算法内部指标（loss 记录点可能与主指标不同步，单独存 step）
        self.loss_steps = []
        self.all_actor_loss = []
        self.all_critic_loss = []
        self.all_ent_coef = []

        # 滑动窗口，用来平滑 per-step 的 info 指标
        self._succ_buf = deque(maxlen=window)
        self._dist_buf = deque(maxlen=window)
        self._assist_buf = deque(maxlen=window)
        self._human_buf = deque(maxlen=window)
        self._full_buf = deque(maxlen=window)
        self._reward_term_bufs = {term: deque(maxlen=window) for term in self.reward_terms}

        plt.ion()
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 15))
        self.fig.suptitle("Assistive-Fetch SAC+HER Training Progress", fontsize=16)

    def _collect_infos(self):
        """从 self.locals['infos'] 里累积每一步的辅助指标到滑动窗口。"""
        infos = self.locals.get("infos")
        if not infos:
            return
        for info in infos:
            if "assist_reward/success" in info:
                self._succ_buf.append(float(info["assist_reward/success"]))
            if "assist_reward/dist" in info:
                self._dist_buf.append(float(info["assist_reward/dist"]))
            if "assist_action_norm" in info:
                self._assist_buf.append(float(info["assist_action_norm"]))
            if "human_action_norm" in info:
                self._human_buf.append(float(info["human_action_norm"]))
            if "full_action_norm" in info:
                self._full_buf.append(float(info["full_action_norm"]))
            for term in self.reward_terms:
                key = f"assist_reward/{term}"
                if key in info:
                    self._reward_term_bufs[term].append(float(info[key]))

    def _on_step(self):
        self._collect_infos()

        if self.n_calls % self.check_freq != 0:
            return True

        # ---- 主指标：平均 episode reward ----
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                self.all_rewards.append(float(np.mean(ep_rewards)))
                self.all_steps.append(self.num_timesteps)

                self.all_success.append(np.mean(self._succ_buf) if self._succ_buf else np.nan)
                self.all_dist.append(np.mean(self._dist_buf) if self._dist_buf else np.nan)
                self.all_assist_norm.append(np.mean(self._assist_buf) if self._assist_buf else np.nan)
                self.all_human_norm.append(np.mean(self._human_buf) if self._human_buf else np.nan)
                self.all_full_norm.append(np.mean(self._full_buf) if self._full_buf else np.nan)

                for term in self.reward_terms:
                    buf = self._reward_term_bufs[term]
                    self.all_reward_terms[term].append(np.mean(buf) if buf else np.nan)

        # ---- 算法内部指标：losses / ent_coef ----
        logger = getattr(self.model, "logger", None)
        if logger is not None:
            actor = logger.name_to_value.get("train/actor_loss")
            critic = logger.name_to_value.get("train/critic_loss")
            ent = logger.name_to_value.get("train/ent_coef")
            if actor is not None or critic is not None or ent is not None:
                self.loss_steps.append(self.num_timesteps)
                self.all_actor_loss.append(actor if actor is not None else np.nan)
                self.all_critic_loss.append(critic if critic is not None else np.nan)
                self.all_ent_coef.append(ent if ent is not None else np.nan)

        self._redraw()
        return True

    def _redraw(self):
        for ax in self.axes.flat:
            ax.clear()

        # [0,0] Mean episode reward
        if self.all_rewards:
            self.axes[0, 0].plot(self.all_steps, self.all_rewards, "b-", linewidth=2)
            self.axes[0, 0].set_title(f"Mean Ep Reward (Current: {self.all_rewards[-1]:.1f})")
            self.axes[0, 0].set_xlabel("Timesteps")
            self.axes[0, 0].set_ylabel("Reward")
            self.axes[0, 0].grid(True, alpha=0.3)

        # [0,1] Success rate
        if self.all_success:
            self.axes[0, 1].plot(self.all_steps, self.all_success, "c-", linewidth=2)
            cur = self.all_success[-1]
            self.axes[0, 1].set_title(f"Success Rate (Current: {cur:.1%})" if not np.isnan(cur) else "Success Rate")
            self.axes[0, 1].set_xlabel("Timesteps")
            self.axes[0, 1].set_ylabel(f"Success (window={self.window})")
            self.axes[0, 1].set_ylim(-0.05, 1.05)
            self.axes[0, 1].grid(True, alpha=0.3)

        # [0,2] Distance to goal
        if self.all_dist:
            self.axes[0, 2].plot(self.all_steps, self.all_dist, "orange", linewidth=2)
            cur = self.all_dist[-1]
            self.axes[0, 2].set_title(f"Dist to Goal (Current: {cur:.3f})" if not np.isnan(cur) else "Dist to Goal")
            self.axes[0, 2].set_xlabel("Timesteps")
            self.axes[0, 2].set_ylabel("Distance (m)")
            self.axes[0, 2].grid(True, alpha=0.3)

        # [1,0] Actor loss
        if self.all_actor_loss:
            self.axes[1, 0].plot(self.loss_steps, self.all_actor_loss, "r-", linewidth=2)
            self.axes[1, 0].set_title(f"Actor Loss (Current: {self.all_actor_loss[-1]:.2f})")
            self.axes[1, 0].set_xlabel("Timesteps")
            self.axes[1, 0].set_ylabel("Actor Loss")
            self.axes[1, 0].grid(True, alpha=0.3)

        # [1,1] Critic loss + ent_coef (twin axis)
        if self.all_critic_loss:
            ax = self.axes[1, 1]
            ax.plot(self.loss_steps, self.all_critic_loss, "g-", linewidth=2, label="critic_loss")
            ax.set_title(f"Critic Loss (Current: {self.all_critic_loss[-1]:.2f})")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Critic Loss", color="g")
            ax.grid(True, alpha=0.3)
            if self.all_ent_coef:
                ax2 = ax.twinx()
                ax2.plot(self.loss_steps, self.all_ent_coef, "m--", linewidth=1.5, label="ent_coef")
                ax2.set_ylabel("Ent Coef", color="m")

        # [1,2] Assist / Human / Full action effort
        # assist_cost_coef 直接惩罚 ‖scaled_assist‖，所以这里重点看 assist 曲线
        if self.all_assist_norm or self.all_human_norm or self.all_full_norm:
            ax = self.axes[1, 2]
            if self.all_assist_norm:
                ax.plot(self.all_steps, self.all_assist_norm, "b-", linewidth=2, label="assist")
            if self.all_human_norm:
                ax.plot(self.all_steps, self.all_human_norm, "k--", linewidth=2, label="human")
            if self.all_full_norm:
                ax.plot(self.all_steps, self.all_full_norm, "g:", linewidth=2, label="full")
            title = "Action Effort (‖assist‖ / ‖human‖ / ‖full‖)"
            if self.all_assist_norm and not np.isnan(self.all_assist_norm[-1]):
                title += f"  assist={self.all_assist_norm[-1]:.3f}"
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Action Norm")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # [2,0] Reward 分项贡献（per-step, 窗口平滑）
        has_terms = any(self.all_reward_terms[t] for t in self.reward_terms)
        if has_terms:
            ax = self.axes[2, 0]
            colors = {
                "dist_term": "tab:orange",
                "success_term": "tab:green",
                "assist_cost_term": "tab:red",
                "stage1_shaping_term": "tab:purple",
            }
            for term in self.reward_terms:
                ys = self.all_reward_terms[term]
                if ys and not np.all(np.isnan(ys)):
                    ax.plot(self.all_steps, ys, linewidth=2,
                            label=term, color=colors.get(term))
            ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
            ax.set_title("Reward Breakdown (per-step, windowed)")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean Reward Contribution")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        # 关闭未使用的子图
        self.axes[2, 1].axis("off")
        self.axes[2, 2].axis("off")

        plt.tight_layout()
        plt.pause(0.01)

    def _on_training_end(self):
        if self.save_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
            self.fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"\n训练曲线已保存到: {self.save_path}")
        plt.ioff()
