# 理解state的物理含义（这很关键）：

# obs[0] = cos(θ)，摆角余弦
# obs[1] = sin(θ)，摆角正弦
# obs[2] = θ̇，角速度
# action = 力矩，范围 [-2, 2]
# Pendulum-v1原始reward已经是：-（θ² + 0.1·θ̇² + 0.001·τ²）

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from collections import deque

class PlottingCallback(BaseCallback):
    """实时绘制训练曲线的回调 - 支持 SAC 和 PPO"""
    def __init__(self, check_freq=100, algo_name="SAC", verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.algo_name = algo_name
        
        # 用于存储所有历史数据
        self.all_rewards = []
        self.all_loss1 = []  # SAC: actor_loss, PPO: policy_loss
        self.all_loss2 = []  # SAC: critic_loss, PPO: value_loss
        self.all_loss3 = []  # SAC: ent_coef, PPO: entropy_loss
        self.all_steps = []
        
        # 设置交互式绘图 (2x2 布局)
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(f'Training Progress - {algo_name}', fontsize=16)
        
    def _on_step(self):
        # 每 check_freq 步更新一次图表
        if self.n_calls % self.check_freq == 0:
            # 从 logger 获取最新数据
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                if ep_rewards:
                    mean_reward = np.mean(ep_rewards)
                    self.all_rewards.append(mean_reward)
                    self.all_steps.append(self.num_timesteps)
                    
                    # 根据算法类型获取不同的指标
                    if hasattr(self.model, 'logger') and self.model.logger is not None:
                        try:
                            if self.algo_name == "SAC":
                                loss1 = self.model.logger.name_to_value.get('train/actor_loss', None)
                                loss2 = self.model.logger.name_to_value.get('train/critic_loss', None)
                                loss3 = self.model.logger.name_to_value.get('train/ent_coef', None)
                                labels = ['Actor Loss', 'Critic Loss', 'Ent Coef']
                            else:  # PPO
                                loss1 = self.model.logger.name_to_value.get('train/policy_gradient_loss', None)
                                if loss1 is None:
                                    loss1 = self.model.logger.name_to_value.get('train/loss', None)
                                loss2 = self.model.logger.name_to_value.get('train/value_loss', None)
                                loss3 = self.model.logger.name_to_value.get('train/entropy_loss', None)
                                labels = ['Policy Loss', 'Value Loss', 'Entropy Loss']
                            
                            if loss1 is not None:
                                self.all_loss1.append(loss1)
                            if loss2 is not None:
                                self.all_loss2.append(loss2)
                            if loss3 is not None:
                                self.all_loss3.append(loss3)
                        except Exception as e:
                            pass
                    
                    # 清空并重绘所有子图
                    for ax in self.axes.flat:
                        ax.clear()
                    
                    # 图1: Episode Reward
                    self.axes[0, 0].plot(self.all_steps, self.all_rewards, 'b-', linewidth=2)
                    self.axes[0, 0].set_xlabel('Timesteps')
                    self.axes[0, 0].set_ylabel('Mean Episode Reward')
                    self.axes[0, 0].set_title(f'Reward (Current: {mean_reward:.1f})')
                    self.axes[0, 0].grid(True, alpha=0.3)
                    
                    # 图2: Loss 1 (Actor/Policy)
                    if len(self.all_loss1) > 0:
                        plot_steps = self.all_steps[-len(self.all_loss1):]
                        self.axes[0, 1].plot(plot_steps, self.all_loss1, 'r-', linewidth=2)
                        self.axes[0, 1].set_xlabel('Timesteps')
                        self.axes[0, 1].set_ylabel(labels[0])
                        self.axes[0, 1].set_title(f'{labels[0]} (Current: {self.all_loss1[-1]:.2f})')
                        self.axes[0, 1].grid(True, alpha=0.3)
                    
                    # 图3: Loss 2 (Critic/Value)
                    if len(self.all_loss2) > 0:
                        plot_steps = self.all_steps[-len(self.all_loss2):]
                        self.axes[1, 0].plot(plot_steps, self.all_loss2, 'g-', linewidth=2)
                        self.axes[1, 0].set_xlabel('Timesteps')
                        self.axes[1, 0].set_ylabel(labels[1])
                        self.axes[1, 0].set_title(f'{labels[1]} (Current: {self.all_loss2[-1]:.2f})')
                        self.axes[1, 0].grid(True, alpha=0.3)
                    
                    # 图4: Loss 3 (Ent Coef/Entropy)
                    if len(self.all_loss3) > 0:
                        plot_steps = self.all_steps[-len(self.all_loss3):]
                        self.axes[1, 1].plot(plot_steps, self.all_loss3, 'm-', linewidth=2)
                        self.axes[1, 1].set_xlabel('Timesteps')
                        self.axes[1, 1].set_ylabel(labels[2])
                        self.axes[1, 1].set_title(f'{labels[2]} (Current: {self.all_loss3[-1]:.3f})')
                        self.axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.pause(0.01)
        
        return True
    
    def _on_training_end(self):
        # 保存最终的图表
        save_path = 'training_curves.png'
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {save_path}")
        plt.ioff()
        plt.show()

class CustomPendulum(gym.Wrapper):
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cos_th, sin_th, thdot = obs
        theta = np.arctan2(sin_th, cos_th)
        
        # 实验1: 只惩罚角度->会到位，但是剧烈摆动
        # reward = -theta**2
        
        # 实验2: 角度 + 速度 (稳定性更好)->结果是在指定位置小幅度摆动
        # reward = -(theta**2 + 0.1 * thdot**2)
        
        # 实验3: 加上控制量惩罚 (平滑性)
        reward = -(theta**2 + 0.1*thdot**2 + 0.01*action[0]**2)
        # Position penalty 给系统提供向目标运动的**“弹簧力” (P)**。
        # Velocity penalty 给系统提供防止超调的**“阻尼力” (D)**。
        # Action penalty 则是节能和平滑性的要求，防止控制器过于激进。
        
        return obs, reward, terminated, truncated, info

def train_and_evaluate(algo_name, total_steps):
    print(f"\n================ 开始 {algo_name} 实验 ================")
    # 1. 训练环境不开渲染（速度快），只用曲线观察
    train_env = CustomPendulum(gym.make("Pendulum-v1"))

    # 2. 根据指定的名称初始化对应的模型
    if algo_name == "SAC":
        model = SAC("MlpPolicy", train_env, verbose=1, learning_rate=1e-3)
        # multilayer perception策略网络结构（对于states是float的

        # learning_rate: 智能体每次发现自己的错误后，对神经网络的“脑回路”修改的步伐有多大
        # SAC更稳定，所以可以用更大的学习率；PPO通常需要更小的学习率（比如3e-4）才能稳定训练
        # 调参诀窍： 如果你发现 TensorBoard 上的 actor_loss 在几十步之内直接飞到了几万或者变成了 NaN，
        # 通常是因为学习率太大，此时你该把它缩小 10 倍（比如从 1e-3 变成 1e-4）。
    elif algo_name == "PPO":
        # PPO 跑连续动作收敛慢，我们加上官方推荐的探索优化参数 (use_sde=True)
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=1e-3, use_sde=True, sde_sample_freq=4)
    else:
        raise ValueError("不支持的算法")
    
    # 3. 开始训练
    print(f"正在训练 {algo_name} 模型，步数: {total_steps}...")
    
    # 创建绘图回调（每200步更新一次图表，传入算法名称）
    plot_callback = PlottingCallback(check_freq=200, algo_name=algo_name)
    
    model.learn(total_timesteps=total_steps, callback=plot_callback)

    # 4. 保存模型
    model_path = f"{algo_name.lower()}_pendulum_model"
    model.save(model_path)
    print(f"{algo_name} 模型已保存为 {model_path}.zip")
    train_env.close()

    # 5. 验证和演示环境 (加上 human 渲染)
    print(f"开始演示 {algo_name} 训练成果...")
    eval_env = gym.make("Pendulum-v1", render_mode="human")
    
    # 根据算法加载模型
    if algo_name == "SAC":
        eval_model = SAC.load(model_path)
    elif algo_name == "PPO":
        eval_model = PPO.load(model_path)

    obs, info = eval_env.reset()
    for i in range(1000):
        action, _states = eval_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()


# ====== 主程序执行 ======

# 先跑 SAC (3万步足够收敛)
# train_and_evaluate("SAC", total_steps=30_000)

# 然后跑 PPO (需要更多步数才能看到好效果，这里设为10万步)
train_and_evaluate("PPO", total_steps=100_000)