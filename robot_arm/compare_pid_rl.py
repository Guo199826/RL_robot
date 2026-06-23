import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

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

def run_evaluation(env, controller_type, model=None, fixed_pid=None, steps=200):
    """运行评估并收集误差数据"""
    obs, info = env.reset()
    errors = []
    
    if controller_type == "Fixed_PID":
        fixed_pid.reset()
        
    for _ in range(steps):
        current_pos = obs['observation'][:3]
        target_pos = obs['desired_goal']
        
        # 记录当前的欧氏距离误差
        error_dist = np.linalg.norm(target_pos - current_pos)
        errors.append(error_dist)
        
        if controller_type == "RL_PID":
            # RL 模型预测的是 [-1, 1] 的增益动作
            rl_action, _ = model.predict(obs, deterministic=True)
            # 在之前写的 RLPIDTunerWrapper 中，这个动作会被内部的 PID 计算为最终速度
            action = rl_action 
        else:
            # 纯 PID 靠自己算速度
            action = fixed_pid.get_action(current_pos, target_pos)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
    return errors

# ================= 1. 环境准备 =================
# 我们使用你上一步写的包含 PID 逻辑的 Wrapper
from test_rl_pid import RLPIDTunerWrapper # 确保 test_rl_pid.py 在同目录下并去掉了演示循环

print("正在加载环境和模型...")
base_env_rl = gym.make("FetchReach-v4", reward_type="dense", render_mode="human")
env_rl = RLPIDTunerWrapper(base_env_rl)
# 加载你之前训练好的 RL-PID 模型
model_rl = SAC.load("sac_pid_tuner")

base_env_fixed = gym.make("FetchReach-v4", reward_type="dense", render_mode="human")
# 我们这里手动选一组看似合理的固定 PID 参数来对比
# 如果你之前观察到 RL 的输出，可以挑一组它平均给出的值
fixed_pid = FixedPIDController(Kp=1.5, Ki=0.1, Kd=0.5)

# ================= 2. 运行对比 =================
print("正在运行纯 PID 控制测试...")
# 纯 PID 不需要 Wrapper，因为它直接输出速度 action 给底层
errors_fixed = run_evaluation(base_env_fixed, "Fixed_PID", fixed_pid=fixed_pid)

print("正在运行 RL-PID 控制测试...")
errors_rl = run_evaluation(env_rl, "RL_PID", model=model_rl)

# ================= 3. 绘制对比图表 =================
print("正在生成对比图表...")
plt.figure(figsize=(10, 6))
plt.plot(errors_fixed, label='Fixed PID (Kp=3, Ki=1, Kd=0.3)', color='red', linestyle='--')
plt.plot(errors_rl, label='RL-Tuned PID (Adaptive)', color='blue', linewidth=2)

plt.title('Performance Comparison: Fixed PID vs. RL-Tuned PID', fontsize=16)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Distance Error (m)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

# 保存图表
plt.savefig('pid_vs_rl_pid_comparison.png', dpi=300, bbox_inches='tight')
print("对比完成！图表已保存为 pid_vs_rl_pid_comparison.png")

env_rl.close()
base_env_fixed.close()