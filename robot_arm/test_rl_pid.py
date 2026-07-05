import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os

gym.register_envs(gymnasium_robotics)

class PlottingCallback(BaseCallback):
    """实时绘制训练曲线 - 直接从 logger 读取，不依赖 ep_info_buffer"""
    def __init__(self, check_freq=200, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq

        # 存储历史数据
        self.all_actor_loss = []
        self.all_critic_loss = []
        self.all_ent_coef = []
        self.all_reward = []
        self.all_steps = []

        # 存储 PID 增益历史 (Kp, Ki, Kd)
        self.all_kp = []
        self.all_ki = []
        self.all_kd = []

        # 累积 reward 计算（因为 episode 不会结束，我们手动累加窗口均值）
        self.recent_rewards = []

        # 累积每步的增益，用于计算窗口均值
        self.recent_kp = []
        self.recent_ki = []
        self.recent_kd = []

        # 设置交互式绘图 (2x3 布局)
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Training Progress - RL-Tuned PID', fontsize=16)

    def _on_step(self):
        # 累积每步的 reward
        if self.locals.get('rewards') is not None:
            self.recent_rewards.extend(self.locals['rewards'].tolist())

        # 累积每步的 PID 增益 (从策略输出的 action 映射回真实增益)
        actions = self.locals.get('actions', None)
        if actions is not None:
            actions = np.asarray(actions)
            # 兼容单环境 (3,) 与多环境 (n_envs, 3)
            flat = actions.reshape(-1, 3)
            for a in flat:
                self.recent_kp.append((a[0] + 1.0) * 2.5)
                self.recent_ki.append((a[1] + 1.0) * 0.5)
                self.recent_kd.append((a[2] + 1.0) * 1.0)

        if self.n_calls % self.check_freq == 0:
            self.all_steps.append(self.num_timesteps)

            # 计算最近 check_freq 步的平均 reward
            if len(self.recent_rewards) > 0:
                mean_rew = np.mean(self.recent_rewards[-self.check_freq:])
                self.all_reward.append(mean_rew)

            # 计算最近 check_freq 步的平均增益
            if len(self.recent_kp) > 0:
                self.all_kp.append(np.mean(self.recent_kp[-self.check_freq:]))
                self.all_ki.append(np.mean(self.recent_ki[-self.check_freq:]))
                self.all_kd.append(np.mean(self.recent_kd[-self.check_freq:]))

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

            # 图5: PID 增益 Kp, Ki, Kd（随训练变化）
            if len(self.all_kp) > 0:
                plot_steps = self.all_steps[-len(self.all_kp):]
                self.axes[1, 2].plot(plot_steps, self.all_kp, 'r-', linewidth=2, label='Kp')
                self.axes[1, 2].plot(plot_steps, self.all_ki, 'g-', linewidth=2, label='Ki')
                self.axes[1, 2].plot(plot_steps, self.all_kd, 'b-', linewidth=2, label='Kd')
                self.axes[1, 2].set_title(
                    f'PID Gains (Kp:{self.all_kp[-1]:.2f} Ki:{self.all_ki[-1]:.2f} Kd:{self.all_kd[-1]:.2f})'
                )
                self.axes[1, 2].legend(loc='best')
            self.axes[1, 2].set_xlabel('Timesteps')
            self.axes[1, 2].set_ylabel('Gain Value')
            self.axes[1, 2].grid(True, alpha=0.3)

            # 隐藏未使用的子图
            self.axes[0, 2].axis('off')

            plt.tight_layout()
            plt.pause(0.01)

        return True

    def _on_training_end(self):
        # 保存最终图表
        os.makedirs('robot_arm/result_data', exist_ok=True)
        save_path = 'robot_arm/result_data/training_curves_rl_pid_noise.png'
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {save_path}")
        plt.ioff()
        plt.show()

class RLPIDTunerWrapper(gym.Wrapper):
    """
    这个 Wrapper 把环境的 Action Space 从 [速度指令] 改成了 [Kp, Ki, Kd]
    然后在内部运行一个 PID 循环来驱动机械臂。
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 1. 重新定义 Action Space (神经网络只输出 Kp, Ki, Kd)
        # 我们假设这三个参数的合理范围是 [0.0, 5.0]
        # 但 RL 习惯输出 [-1, 1]，所以稍后在 step 里会做映射
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # PID 的积分和微分状态
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        # 连续停留在目标半径内的步数（用于稳态奖励）
        self.in_goal_steps = 0
        
    def reset(self, **kwargs):
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.in_goal_steps = 0
        return self.env.reset(**kwargs)

    def step(self, rl_action):
        """
        rl_action 就是神经网络输出的 [Kp, Ki, Kd]，在 [-1, 1] 之间。
        """
        # 1. 把 RL 的输出映射到真实的 PID 参数范围
        # 例如: Kp [0, 5], Ki [0, 1], Kd [0, 2]
        Kp = (rl_action[0] + 1.0) * 2.5      # 映射到 0~5
        Ki = (rl_action[1] + 1.0) * 0.5      # 映射到 0~1
        Kd = (rl_action[2] + 1.0) * 1.0      # 映射到 0~2

        # 2. 获取当前状态，计算误差 (Error = 目标点 - 夹爪当前点)
        # 实际上我们不调用底层的 step，而是先从底层的变量读取状态
        # (Fetch 环境底层保存了当前 goal 和 grip_pos)
        obs = self.env.unwrapped._get_obs()
        current_pos = obs['observation'][:3] # 夹爪位置 (xyz)
        target_pos = obs['desired_goal']     # 目标位置 (xyz)
        
        error = target_pos - current_pos
        
        # 3. 计算 PID
        self.integral_error += error
        # 为了防止积分饱和 (Windup)，限制积分项的上限
        self.integral_error = np.clip(self.integral_error, -1.0, 1.0)
        
        derivative_error = error - self.prev_error
        self.prev_error = error

        # PID 输出：我们要发送给机械臂底层的速度指令
        pid_velocity = (Kp * error) + (Ki * self.integral_error) + (Kd * derivative_error)
        
        # 机械臂的 action 需要是 4维 [vx, vy, vz, gripper]
        # 我们把 PID 算出的 3维速度补上一个不动的夹爪维度 (0.0)
        final_robot_action = np.zeros(4, dtype=np.float32)
        # 防止 PID 算出来的速度太大，做个裁剪？？
        final_robot_action[:3] = np.clip(pid_velocity, -1.0, 1.0) 

        # 4. 把真正的控制指令发给底层环境执行
        next_obs, reward, terminated, truncated, info = self.env.step(final_robot_action)
        
        # 解析位置和速度
        full_obs = self.env.unwrapped._get_obs()
        current_pos = full_obs['observation'][:3]
        target_pos = full_obs['desired_goal']
        gripper_vel = full_obs['observation'][3:6]

        dist = np.linalg.norm(target_pos - current_pos)
        speed = np.linalg.norm(gripper_vel)

        # 1. 距离惩罚（主目标）
        reward = -dist

        # 2. 靠近目标时的速度惩罚（防 overshoot／抖动）
        near_threshold = 0.03  #3cm
        if dist < near_threshold:
            reward -= 0.1 * speed**2

        # 3. 停留奖励（鼓励稳态）
        goal_radius = 0.015
        if dist < goal_radius:
            self.in_goal_steps += 1
            reward += 0.02
        else:
            self.in_goal_steps = 0

        # # 4. RL 调参幅值惩罚（你之前的那一项）
        # reward -= 0.01 * np.sum(np.square(rl_action))

        return next_obs, reward, terminated, truncated, info

class ObsNoiseWrapper(gym.ObservationWrapper):
    """
    在 observation 和 achieved_goal 上加固定高斯噪声。
    desired_goal 保持不变（目标点是真实已知的）。
    """
    def __init__(self, env, obs_sigma=0.005, goal_sigma=0.002):
        super().__init__(env)
        self.obs_sigma = obs_sigma      # state 噪声强度 (m)
        self.goal_sigma = goal_sigma    # 末端位置观测噪声

    def observation(self, obs):
        # obs 是字典：{'observation', 'achieved_goal', 'desired_goal'}
        noisy_obs = obs.copy()

        # 对机械臂状态 observation 加噪声
        noisy_obs['observation'] = (
            noisy_obs['observation'] +
            np.random.normal(0.0, self.obs_sigma, size=noisy_obs['observation'].shape)
        ).astype(np.float32)

        # 对 achieved_goal（末端测量位置）加噪声
        noisy_obs['achieved_goal'] = (
            noisy_obs['achieved_goal'] +
            np.random.normal(0.0, self.goal_sigma, size=noisy_obs['achieved_goal'].shape)
        ).astype(np.float32)

        # desired_goal 不加噪声：目标点一般是规划给出的“真值”
        return noisy_obs

if __name__ == '__main__':
    # ================= 开始训练 =================
    print("开始训练 RL-Tuned PID 控制器...")
    # 训练时不渲染，否则 50k 步会被实时窗口拖慢几个数量级
    base_env = gym.make("FetchReach-v4", reward_type="dense", render_mode=None)

    env = RLPIDTunerWrapper(base_env)

    # 固定观测噪声
    env = ObsNoiseWrapper(env, obs_sigma=0.005, goal_sigma=0.002)

    os.makedirs("./tensorboard_logs_pid/", exist_ok=True)
    env = Monitor(env, "./tensorboard_logs_pid/") 

    # 初始化模型
    model = SAC("MultiInputPolicy", env, verbose=1, learning_rate=1e-3, tensorboard_log="./tensorboard_logs_pid/")

    # 创建绘图回调（每200步更新一次）
    plot_callback = PlottingCallback(check_freq=200)

    # 训练它自己找最优参数
    model.learn(total_timesteps=50_000, tb_log_name="SAC_PID", callback=plot_callback)
    model.save("sac_pid_tuner_noise")
    env.close()

    # ================= 演示效果 =================
    print("训练完成！开始演示 PID 自动调参控制...")
    base_env = gym.make("FetchReach-v4", reward_type="dense", render_mode="human")
    eval_env = RLPIDTunerWrapper(base_env)
    # 加载本次训练保存的同一个模型（带噪声训练版）
    model = SAC.load("sac_pid_tuner_noise")

    obs, info = eval_env.reset()
    for _ in range(1000):
        # RL 实时输出当前的 Kp, Ki, Kd
        action, _states = model.predict(obs, deterministic=True)
        
        Kp = (action[0] + 1.0) * 2.5
        Ki = (action[1] + 1.0) * 0.5
        Kd = (action[2] + 1.0) * 1.0
        # print(f"当前 RL 设定的参数 -> Kp: {Kp:.2f}, Ki: {Ki:.2f}, Kd: {Kd:.2f}")
        
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()
