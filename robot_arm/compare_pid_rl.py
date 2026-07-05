import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

# ================= 配置（集中管理，避免图例/代码不一致）=================
MODEL_PATH = "model/sac_pid_tuner_noise"   # 与 test_rl_pid.py 训练保存的名字保持一致
FIXED_GAINS = dict(Kp=3.0, Ki=0.3, Kd=1.0)   # 调好的固定 PID（更强的基线）
EP_LEN = 200         # 跑 200 步；需在 gym.make 时把时限放宽到 200（默认是 50）
N_EPISODES = 20      # 用多个 episode 取平均，降低单次随机目标带来的偶然性
SEEDS = list(range(N_EPISODES))


class FixedPIDController:
    """一个传统的固定参数 PID 控制器"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

    def reset(self):
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

    def get_action(self, current_pos, target_pos):
        error = target_pos - current_pos

        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -1.0, 1.0)

        derivative_error = error - self.prev_error
        self.prev_error = error

        pid_velocity = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative_error)

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(pid_velocity, -1.0, 1.0)
        return action


def run_episode(env, controller_type, seed, model=None, fixed_pid=None, steps=EP_LEN):
    """用给定 seed 跑一个 episode，返回每一步的距离误差列表。"""
    obs, info = env.reset(seed=seed)
    errors = []

    if controller_type == "Fixed_PID":
        fixed_pid.reset()

    for _ in range(steps):
        current_pos = obs['observation'][:3]
        target_pos = obs['desired_goal']

        # 记录当前的欧氏距离误差
        errors.append(np.linalg.norm(target_pos - current_pos))

        if controller_type == "RL_PID":
            # RL 模型输出 [-1,1] 的增益动作，由 RLPIDTunerWrapper 内部转成速度
            action, _ = model.predict(obs, deterministic=True)
        else:
            # 纯 PID 自己算速度
            action = fixed_pid.get_action(current_pos, target_pos)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # episode 结束就停，不在已结束环境上越界 step
            break

    # 长度不足 steps（提前结束）时用最后一个误差补齐，方便对齐做平均
    while len(errors) < steps:
        errors.append(errors[-1])
    return errors


def run_all(env, controller_type, model=None, fixed_pid=None):
    """对所有 seed 跑一遍，返回 (n_episodes, steps) 的误差矩阵。"""
    runs = [run_episode(env, controller_type, s, model=model, fixed_pid=fixed_pid)
            for s in SEEDS]
    return np.asarray(runs)


# ================= 1. 环境准备（关闭渲染，跑多个 episode 才不会卡）=================
from test_rl_pid import RLPIDTunerWrapper  # 复用包含 PID 逻辑的 Wrapper

print("正在加载环境和模型...")
base_env_rl = gym.make("FetchReach-v4", reward_type="dense", render_mode=None,
                       max_episode_steps=EP_LEN)
env_rl = RLPIDTunerWrapper(base_env_rl)
model_rl = SAC.load(MODEL_PATH)

base_env_fixed = gym.make("FetchReach-v4", reward_type="dense", render_mode=None,
                          max_episode_steps=EP_LEN)
fixed_pid = FixedPIDController(**FIXED_GAINS)

# ================= 2. 运行对比（两者用完全相同的 SEEDS，目标点一一对应）=========
print("正在运行纯 PID 控制测试...")
errors_fixed = run_all(base_env_fixed, "Fixed_PID", fixed_pid=fixed_pid)

print("正在运行 RL-PID 控制测试...")
errors_rl = run_all(env_rl, "RL_PID", model=model_rl)

# ================= 3. 绘制对比图表（均值 ± 标准差）=================
print("正在生成对比图表...")
t = np.arange(EP_LEN)
mean_fixed, std_fixed = errors_fixed.mean(axis=0), errors_fixed.std(axis=0)
mean_rl, std_rl = errors_rl.mean(axis=0), errors_rl.std(axis=0)

g = FIXED_GAINS
fixed_label = f"Fixed PID (Kp={g['Kp']}, Ki={g['Ki']}, Kd={g['Kd']})"

plt.figure(figsize=(10, 6))
plt.plot(t, mean_fixed, label=fixed_label, color='red', linestyle='--')
plt.fill_between(t, mean_fixed - std_fixed, mean_fixed + std_fixed, color='red', alpha=0.15)
plt.plot(t, mean_rl, label='RL-Tuned PID (Adaptive)', color='blue', linewidth=2)
plt.fill_between(t, mean_rl - std_rl, mean_rl + std_rl, color='blue', alpha=0.15)

plt.title(f'Performance Comparison: Fixed PID vs. RL-Tuned PID (mean of {N_EPISODES} eps)', fontsize=15)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Distance Error (m)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

import os
os.makedirs('result_data', exist_ok=True)
out_path = 'result_data/pid_vs_rl_pid_comparison.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"对比完成！图表已保存为 {out_path}")

env_rl.close()
base_env_fixed.close()
