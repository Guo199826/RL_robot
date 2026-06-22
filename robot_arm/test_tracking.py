import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

# 我们写一个测试专用的 Wrapper，让红点（目标）动起来
class CircleTrajectoryWrapper(gym.Wrapper):
    def __init__(self, env, radius=0.05, speed=0.1):
        super().__init__(env)
        self.t = 0
        self.radius = radius
        self.speed = speed
        self.center = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        # 记录初始目标点作为圆心
        self.center = obs['desired_goal'].copy()
        return obs, info

    def step(self, action):
        self.t += 1
        
        # 计算新时刻的目标点（在 XY 平面上画圆）
        new_goal = self.center.copy()
        new_goal[0] += self.radius * np.cos(self.speed * self.t)
        new_goal[1] += self.radius * np.sin(self.speed * self.t)
        
        # 强行修改底层引擎的目标点，这样画面的红点也会跟着动
        self.env.unwrapped.goal = new_goal
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 替换观察值中的目标点，让神经网络追踪新点
        obs['desired_goal'] = new_goal.copy()
        
        # 为了持续跟踪，我们强行让它永不结束
        return obs, reward, False, False, info


print("加载训练好的平滑模型进行轨迹跟踪...")
# 加载环境
base_env = gym.make("FetchReach-v4", reward_type="dense", render_mode="human")
# 套上画圆轨迹的 Wrapper
env = CircleTrajectoryWrapper(base_env, radius=0.2, speed=0.05)

# 加载你刚刚训练出来的 100% 成功率模型
model = SAC.load("robot_arm/model/sac_fetch_reach_action_30k")

obs, info = env.reset()
# 跑一个长循环（10000步），观察它的持续跟踪能力
for _ in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()