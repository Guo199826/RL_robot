import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

gym.register_envs(gymnasium_robotics)

class PlottingCallback(BaseCallback):
    """实时绘制训练曲线 - 直接从logger读取，不依赖ep_info_buffer"""
    def __init__(self, check_freq=200, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        
        # 存储历史数据
        self.all_actor_loss = []
        self.all_critic_loss = []
        self.all_ent_coef = []
        self.all_reward = []
        self.all_steps = []
        
        # 累积 reward 计算（因为 episode 不会结束，我们手动累加窗口均值）
        self.recent_rewards = []
        
        # 设置交互式绘图 (2x2 布局)
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Training Progress - Feedforward Tracking', fontsize=16)
        
    def _on_step(self):
        # 累积每步的 reward
        if self.locals.get('rewards') is not None:
            self.recent_rewards.extend(self.locals['rewards'].tolist())
        
        if self.n_calls % self.check_freq == 0:
            self.all_steps.append(self.num_timesteps)
            
            # 计算最近 check_freq 步的平均 reward
            if len(self.recent_rewards) > 0:
                mean_rew = np.mean(self.recent_rewards[-self.check_freq:])
                self.all_reward.append(mean_rew)
            
            # 从 logger 直接读取训练指标
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    actor_loss = self.model.logger.name_to_value.get('train/actor_loss', None)
                    critic_loss = self.model.logger.name_to_value.get('train/critic_loss', None)
                    ent_coef = self.model.logger.name_to_value.get('train/ent_coef', None)
                    
                    if actor_loss is not None:
                        self.all_actor_loss.append(actor_loss)
                    if critic_loss is not None:
                        self.all_critic_loss.append(critic_loss)
                    if ent_coef is not None:
                        self.all_ent_coef.append(ent_coef)
                except:
                    pass
            
            # 清空并重绘
            for ax in self.axes.flat:
                ax.clear()
            
            # 图1: Mean Step Reward
            if len(self.all_reward) > 0:
                plot_steps = self.all_steps[:len(self.all_reward)]
                self.axes[0, 0].plot(plot_steps, self.all_reward, 'b-', linewidth=2)
                self.axes[0, 0].set_title(f'Mean Step Reward (Current: {self.all_reward[-1]:.3f})')
            self.axes[0, 0].set_xlabel('Timesteps')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].grid(True, alpha=0.3)
            
            # 图2: Actor Loss
            if len(self.all_actor_loss) > 0:
                plot_steps = self.all_steps[-len(self.all_actor_loss):]
                self.axes[0, 1].plot(plot_steps, self.all_actor_loss, 'r-', linewidth=2)
                self.axes[0, 1].set_title(f'Actor Loss (Current: {self.all_actor_loss[-1]:.2f})')
            self.axes[0, 1].set_xlabel('Timesteps')
            self.axes[0, 1].set_ylabel('Actor Loss')
            self.axes[0, 1].grid(True, alpha=0.3)
            
            # 图3: Critic Loss
            if len(self.all_critic_loss) > 0:
                plot_steps = self.all_steps[-len(self.all_critic_loss):]
                self.axes[1, 0].plot(plot_steps, self.all_critic_loss, 'g-', linewidth=2)
                self.axes[1, 0].set_title(f'Critic Loss (Current: {self.all_critic_loss[-1]:.2f})')
            self.axes[1, 0].set_xlabel('Timesteps')
            self.axes[1, 0].set_ylabel('Critic Loss')
            self.axes[1, 0].grid(True, alpha=0.3)
            
            # 图4: Entropy Coefficient
            if len(self.all_ent_coef) > 0:
                plot_steps = self.all_steps[-len(self.all_ent_coef):]
                self.axes[1, 1].plot(plot_steps, self.all_ent_coef, 'm-', linewidth=2)
                self.axes[1, 1].set_title(f'Ent Coef (Current: {self.all_ent_coef[-1]:.4f})')
            self.axes[1, 1].set_xlabel('Timesteps')
            self.axes[1, 1].set_ylabel('Entropy Coef')
            self.axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
        
        return True
    
    def _on_training_end(self):
        # 保存最终图表
        save_path = 'robot_arm/result_data/training_curves_feedforward.png'
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {save_path}")
        plt.ioff()
        plt.show()

class FeedforwardTrackingWrapper(gym.Wrapper):
    def __init__(self, env, radius=0.1, speed=0.05):
        super().__init__(env)
        self.t = 0
        self.radius = radius
        self.speed = speed
        self.center = None
        
        # 【核心操作1】：扩展 Observation 空间，给目标速度留出 3 个维度的位置
        old_obs_space = self.observation_space.spaces['observation']
        low = np.concatenate([old_obs_space.low, np.full(3, -np.inf, dtype=np.float32)])
        high = np.concatenate([old_obs_space.high, np.full(3, np.inf, dtype=np.float32)])
        
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=low, high=high, dtype=np.float32),
            'achieved_goal': self.observation_space.spaces['achieved_goal'],
            'desired_goal': self.observation_space.spaces['desired_goal']
        })

    def get_target_pos_vel(self):
        # 计算当前时刻红点的位置
        pos = self.center.copy()
        pos[0] += self.radius * np.cos(self.speed * self.t)
        pos[1] += self.radius * np.sin(self.speed * self.t)
        
        # 计算当前时刻红点的移动速度 (就是对位置求导)
        vel = np.zeros(3, dtype=np.float32)
        vel[0] = -self.radius * self.speed * np.sin(self.speed * self.t)
        vel[1] = self.radius * self.speed * np.cos(self.speed * self.t)
        return pos, vel

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        self.center = obs['desired_goal'].copy()
        
        pos, vel = self.get_target_pos_vel()
        self.env.unwrapped.goal = pos
        obs['desired_goal'] = pos
        # 【核心操作2】：把速度拼接到观测值里喂给神经网络
        obs['observation'] = np.concatenate([obs['observation'], vel])
        return obs, info

    def step(self, action):
        self.t += 1
        pos, vel = self.get_target_pos_vel()
        self.env.unwrapped.goal = pos
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 继承上一轮的优良传统：加上平滑控制惩罚
        reward -= 0.05 * np.sum(np.square(action))
        
        obs['desired_goal'] = pos
        # 同样把速度拼接到观测值里
        obs['observation'] = np.concatenate([obs['observation'], vel])
        
        # 强制不结束回合，让它能一直画圈跟踪
        return obs, reward, False, False, info


# ================= 1. 开始训练 =================
print("正在训练具备前馈追踪能力的机械臂...")
base_env = gym.make("FetchReach-v4", reward_type="dense")
env = FeedforwardTrackingWrapper(base_env, radius=0.1, speed=0.05)

# 训练网络 (由于任务变复杂了，画圆需要更多数据，我们跑 5万 步)
model = SAC("MultiInputPolicy", env, verbose=1, learning_rate=1e-3)

# 创建绘图回调（每200步更新一次）
plot_callback = PlottingCallback(check_freq=200)

model.learn(total_timesteps=50_000, callback=plot_callback)

model.save("robot_arm/model/sac_feedforward_tracking")
env.close()

# ================= 2. 演示成果 =================
print("训练完成！开始演示真正的零滞后跟踪...")
base_env = gym.make("FetchReach-v4", reward_type="dense", render_mode="human")
env = FeedforwardTrackingWrapper(base_env, radius=0.1, speed=0.05)
model = SAC.load("robot_arm/model/sac_feedforward_tracking")

obs, info = env.reset()
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
env.close()