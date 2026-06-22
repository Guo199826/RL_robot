import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt

gym.register_envs(gymnasium_robotics)

class PlottingCallback(BaseCallback):
    """实时绘制训练曲线的回调 - SAC"""
    def __init__(self, check_freq=100, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        
        self.all_rewards = []
        self.all_actor_loss = []
        self.all_critic_loss = []
        self.all_ent_coef = []
        self.all_steps = []
        
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('FetchReach SAC Training Progress', fontsize=16)
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                if ep_rewards:
                    mean_reward = np.mean(ep_rewards)
                    self.all_rewards.append(mean_reward)
                    self.all_steps.append(self.num_timesteps)
                    
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        try:
                            loss1 = self.model.logger.name_to_value.get('train/actor_loss', None)
                            loss2 = self.model.logger.name_to_value.get('train/critic_loss', None)
                            loss3 = self.model.logger.name_to_value.get('train/ent_coef', None)
                            
                            if loss1 is not None:
                                self.all_actor_loss.append(loss1)
                            if loss2 is not None:
                                self.all_critic_loss.append(loss2)
                            if loss3 is not None:
                                self.all_ent_coef.append(loss3)
                        except Exception:
                            pass
                    
                    for ax in self.axes.flat:
                        ax.clear()
                    
                    self.axes[0, 0].plot(self.all_steps, self.all_rewards, 'b-', linewidth=2)
                    self.axes[0, 0].set_xlabel('Timesteps')
                    self.axes[0, 0].set_ylabel('Mean Episode Reward')
                    self.axes[0, 0].set_title(f'Reward (Current: {mean_reward:.1f})')
                    self.axes[0, 0].grid(True, alpha=0.3)
                    
                    if len(self.all_actor_loss) > 0:
                        plot_steps = self.all_steps[-len(self.all_actor_loss):]
                        self.axes[0, 1].plot(plot_steps, self.all_actor_loss, 'r-', linewidth=2)
                        self.axes[0, 1].set_xlabel('Timesteps')
                        self.axes[0, 1].set_ylabel('Actor Loss')
                        self.axes[0, 1].set_title(f'Actor Loss (Current: {self.all_actor_loss[-1]:.2f})')
                        self.axes[0, 1].grid(True, alpha=0.3)
                    
                    if len(self.all_critic_loss) > 0:
                        plot_steps = self.all_steps[-len(self.all_critic_loss):]
                        self.axes[1, 0].plot(plot_steps, self.all_critic_loss, 'g-', linewidth=2)
                        self.axes[1, 0].set_xlabel('Timesteps')
                        self.axes[1, 0].set_ylabel('Critic Loss')
                        self.axes[1, 0].set_title(f'Critic Loss (Current: {self.all_critic_loss[-1]:.2f})')
                        self.axes[1, 0].grid(True, alpha=0.3)
                    
                    if len(self.all_ent_coef) > 0:
                        plot_steps = self.all_steps[-len(self.all_ent_coef):]
                        self.axes[1, 1].plot(plot_steps, self.all_ent_coef, 'm-', linewidth=2)
                        self.axes[1, 1].set_xlabel('Timesteps')
                        self.axes[1, 1].set_ylabel('Ent Coef')
                        self.axes[1, 1].set_title(f'Ent Coef (Current: {self.all_ent_coef[-1]:.3f})')
                        self.axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.pause(0.01)
        
        return True
    
    def _on_training_end(self):
        save_path = 'fetch_reach_training_curves.png'
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {save_path}")
        plt.ioff()
        plt.show()

# 创建一个自定义 Wrapper 来修改 Reward
class SmoothFetchReachWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # 1. 先执行原本的 step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. 我们拿到的是 Dense Reward (负的欧式距离)，这是我们的 P 项
        distance_penalty = reward
        
        # 3. 添加 Action 惩罚 (让动作更平滑，最小化速度指令)
        # action 是一个长度为 4 的数组 [vx, vy, vz, gripper]
        action_penalty = -0.05 * np.sum(np.square(action))

        # 注意！如何得到0.05这个系数：量纲分析（dimensional analysis）
        # 调参公式是： 次要指标（平滑性/能耗）的最大可能惩罚值 ～= 主要指标（距离误差）在此阶段你能容忍的稳态误差。
        # e.g. 在极端的动作下：惩罚是 $-4.0 \times 0.05 = -0.2$。这和“偏离目标 0.2 米”的惩罚力度相当。
        # 这意味着如果为了靠近 0.2 米而把力矩拉满，模型会觉得得不偿失，从而抑制极端行为。
        
        # 0.05 只是一个起点。在真实的工业项目中，我们通常会：
        # 1. 从很小的值（比如 0.001）开始加，直到机械臂刚好不发生抖动。
        # 2. 将这个系数改为动态的（比如随着训练步数或者距离的缩小逐渐变大）。这就是强化学习中常说的 Reward Annealing（奖励退火）技术。
        
        # 4. 组合新的 Reward
        custom_reward = distance_penalty + action_penalty
        
        return obs, custom_reward, terminated, truncated, info

# ========= 训练部分 =========

# 记得加上 reward_type="dense"
base_env = gym.make("FetchReach-v4", reward_type="dense")

# 套上我们自己写的 Reward Wrapper
env = SmoothFetchReachWrapper(base_env)

# 目标： 控制机械臂，让它的末端执行器（夹爪）移动到一个随机生成的空间红点（目标点）。

# Action： 4维连续向量（控制夹爪在 X、Y、Z 三个方向的速度，以及夹爪开合）。
# Reward： 如果末端距离目标点 > 5cm，reward 为 -1；如果到达 5cm 以内，reward 为 0。
# State（重点突破）： 与 Pendulum 的一维数组不同，真实机械臂的 State 通常是一个字典 (Dict)。它包含：
    # observation: 机械臂当前状态（夹爪位置、各个关节速度等）。
    # desired_goal: 红点的位置（你要去的地方）。
    # achieved_goal: 夹爪当前的位置。

# 注意：这里改成了 MultiInputPolicy，专门处理字典类型的 State
model = SAC("MultiInputPolicy", env, verbose=1, learning_rate=1e-3)

# 机械臂任务稍微复杂一点，我们让它跑 5 万步 (在 MuJoCo 下也就几分钟)
print("开始训练机械臂...")
plot_callback = PlottingCallback(check_freq=200)
model.learn(total_timesteps=30_000, callback=plot_callback)

model.save("sac_fetch_reach")
print("训练完成！模型已保存。")
env.close()

# ======== 渲染看效果 ========
print("开始演示...")
env = gym.make("FetchReach-v4", render_mode="human")
model = SAC.load("sac_fetch_reach")

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 因为机械臂没有所谓的"失败倒地"，通常只在超时或达到目标后 reset
    if terminated or truncated:
        obs, info = env.reset()

env.close()