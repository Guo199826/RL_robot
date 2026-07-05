# 设计笔记 / 实施思路（面试讲解用）

> 这份文档把**最初代码里就有的设计哲学**（reward 怎么设计、plots 观测哪些数据）和
> **这次重构成框架的实施思路**串成一条主线，方便面试时按图讲解。
> 一句话主线：**用「经典控制的语言」去设计 RL 的奖励与评估，再把传统控制器和 RL 结合起来。**

---

## Part 1. 最初的核心思想：reward 整形 = 经典控制项的类比

这是整个项目最值得讲的「一以贯之」的洞察，从 `pendulum` 阶段就定了调：

```138:141:pendulum/pendulum_v1.py
        reward = -(theta**2 + 0.1*thdot**2 + 0.01*action[0]**2)
        # Position penalty 给系统提供向目标运动的**“弹簧力” (P)**。
        # Velocity penalty 给系统提供防止超调的**“阻尼力” (D)**。
        # Action penalty 则是节能和平滑性的要求，防止控制器过于激进。
```

把奖励的三项与 PD 控制一一对应：

| reward 项 | 物理含义 | 对应控制概念 |
|---|---|---|
| `-θ²`（位置误差²） | 拉回目标的「弹簧力」 | **P（比例）** |
| `-0.1·θ̇²`（速度²） | 抑制超调的「阻尼力」 | **D（微分）** |
| `-0.01·τ²`（控制量²） | 节能 / 平滑 / 防止激进 | 控制能耗约束 |

**面试讲法**：我做 reward shaping 不是拍脑袋加项，而是按控制论的结构去配——
误差项决定「能不能到」，速度项决定「稳不稳（超调/震荡）」，控制量项决定「平不平滑（能耗/抖动）」。
这套思路后面在机械臂上完全复用。

---

## Part 2. reward 设计演进史（逐任务，含数值与理由）

### 2.1 倒立摆 Pendulum：三段递进实验
我是用「消融实验」的方式确定 reward 的（代码里把三版都留了注释）：

```131:138:pendulum/pendulum_v1.py
        # 实验1: 只惩罚角度->会到位，但是剧烈摆动
        # reward = -theta**2
        # 实验2: 角度 + 速度 (稳定性更好)->结果是在指定位置小幅度摆动
        # reward = -(theta**2 + 0.1 * thdot**2)
        # 实验3: 加上控制量惩罚 (平滑性)
        reward = -(theta**2 + 0.1*thdot**2 + 0.01*action[0]**2)
```

- 实验1（只有 P）：能到位但剧烈摆动 → 缺阻尼。
- 实验2（P+D）：稳定，小幅摆动 → 加了速度阻尼。
- 实验3（P+D+控制量）：更平滑、更省力。

**讲法**：先验证主目标，再逐步加稳定性/平滑性约束——这就是工程上「先让它动起来，再让它动得好」。

### 2.2 机械臂到达 FetchReach：dense 奖励 + 动作惩罚（量纲分析定系数）
```108:113:robot_arm/robot_arm_v1.py
        # 3. 添加 Action 惩罚 (让动作更平滑，最小化速度指令)
        action_penalty = -0.05 * np.sum(np.square(action))
        # 注意！如何得到0.05这个系数：量纲分析（dimensional analysis）
        # 调参公式是： 次要指标（平滑性/能耗）的最大可能惩罚值 ～= 主要指标（距离误差）在此阶段你能容忍的稳态误差。
```

**关键讲点（系数怎么定的，不是瞎调）**：动作惩罚系数 0.05 是用**量纲分析**定的——
动作最大时惩罚 ≈ `4 × 0.05 = 0.2`，刚好和「偏离目标 0.2m」的惩罚同量级；
这样模型不会为了多靠近 0.2m 而把速度拉满。即：
**次要指标的最大惩罚 ≈ 主要指标能容忍的稳态误差**。还提到可以做 reward annealing（奖励退火）动态调大。

### 2.3 前馈轨迹跟踪 FeedforwardTracking：把「前馈」喂进观测
跟踪动目标时，纯反馈控制必然有**滞后**。我的做法是把目标的**速度**（位置对时间求导）拼进观测，
让网络拿到前馈信息，实现近似零滞后跟踪：

```132:142:robot_arm/test_tracking_ff.py
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
```

**讲法**：这是经典控制里「前馈 + 反馈」思想在 RL 观测设计上的体现——
反馈靠误差纠偏，前馈靠模型/目标导数提前量补偿，消除跟踪滞后。

### 2.4 RL 自整定 PID：RL 出增益、内部跑 PID
这是「传统+RL 结合」的第一个混合体：神经网络不直接出速度，而是输出 `[Kp,Ki,Kd]`，
内部用真正的 PID 环驱动机械臂；reward 在「距离 + 近距速度惩罚 + 停留奖励」上做了精细化：

```232:246:robot_arm/test_rl_pid.py
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
```

增益映射（把 RL 的 `[-1,1]` 输出映回物理范围）：`Kp∈[0,5], Ki∈[0,1], Kd∈[0,2]`。

**讲法**：保留 PID 的可解释性与稳定性，让 RL 只做「在线增益调度（gain scheduling）」；
reward 用「远处看距离、近处看速度、到了给停留奖励」分段塑形，专治超调和稳态抖动。

> 注：原 `test_rl_pid.py` 里 `self.in_goal_steps` 在 `reset` 中漏初始化（小 bug）；
> 新框架 `mc/controllers/rl_pid.py` 已重写、规避。

---

## Part 3. plots 观测了哪些数据，以及「为什么看它」

原始脚本里每个训练都画 4~6 张子图（`PlottingCallback`）。**面试要能说清每条曲线诊断什么**：

| 观测量 | 出处 | 看它用来判断什么 |
|---|---|---|
| **Mean Episode/Step Reward** | 所有脚本 | 是否在学、收敛到什么水平 |
| **Actor Loss**（SAC） | SAC | 策略更新幅度；**飞到上万/NaN = 学习率太大**（代码注释里写了这条诊断经验） |
| **Critic Loss**（SAC） | SAC | 值函数拟合是否稳定 |
| **Ent Coef**（SAC 自动熵） | SAC | 探索强度随训练衰减情况 |
| **Policy/Value/Entropy Loss**（PPO） | PPO 分支 | PPO 对应的策略/值/熵诊断 |
| **Kp / Ki / Kd 随训练演化** | `test_rl_pid.py` | RL 学到的增益是否合理、是否收敛——**可解释性** |
| **Distance Error vs Time** | `compare_pid_rl.py` | 控制性能：到位速度、稳态误差、超调 |

代码里那条很实用的调参经验（直接来自注释）：

```156:158:pendulum/pendulum_v1.py
        # 调参诀窍： 如果你发现 TensorBoard 上的 actor_loss 在几十步之内直接飞到了几万或者变成了 NaN，
        # 通常是因为学习率太大，此时你该把它缩小 10 倍（比如从 1e-3 变成 1e-4）。
```

**这次重构对「观测数据」的升级**：原来评估只看「距离误差」一条；新框架 `mc/common/metrics.py`
把它扩成了**运控工程师的完整指标体系**：成功率 / 跟踪RMSE / 稳态误差 / 调节时间(settling) /
超调(overshoot) / 控制能耗 / 动作 jerk(平滑度) / 路径长度。——用数字而非「看起来到了」去对比控制器。

新框架保留了原 `result_data` 那套训练诊断图：`CurveLoggerCallback` 记录 reward/actor/critic/熵系数,
`plot_training_curves` 复刻 2×2 子图,每次训练自动产出 `training_curves.png`。

---

## Part 3.5. 抗扰实验：怎么在 Fetch 上"忠实地推一下"

**踩坑**：第一反应是 `data.xfrc_applied` 给末端施加外力,但 200N 只让夹爪动了 0.001m——
Fetch 夹爪刚性焊接到位置控制的 mocap 上,外力被 weld 约束几乎完全抵消(qvel kick 同理被位置作动器立刻纠正)。

**解法**：这个平台上"推一下"最忠实、最可解释的建模是**一段强外源速度指令**——
在随机/固定时刻往随机方向注入 `action[:3]=dir*1.0` 持续几步,把末端"撞"开 9–18cm,
控制器无法阻止,之后看它如何纠回(`PushDisturbanceWrapper`)。这正是控制理论里的**扰动抑制**测试。

**三个可直接讲的发现**：
1. reach 是反馈主导的简单任务,**所有稳定控制器都能抑制推力**;区分度在"稳态精度+恢复平滑度",PD 最优。
2. **Residual(scale 0.3) 在近最优 base 上反而最差**(稳态~4cm、恢复最慢、强推下只 75% 复位)——
   说明残差不是免费的:base 够好时它是负担,base 不足时(PickAndPlace 2%→82%)才是增益。这是个有深度的 nuance。
3. 把推力作为**域随机化**注入训练,SAC 收敛慢约 3×(6k→18k 步),而 reach 上鲁棒性收益甚微——
   **"为鲁棒性付训练成本"是要看任务的工程取舍**。

---

## Part 3.6. 给 SAC/PPO 加模仿学习(IL):一条层层递进的诚实结论链

实验:脚本示教 200 条 → BC 预热 → RL 微调(`scripts/imitation_bc_rl.py`,PAP 成功率)。

| 配方 | SAC | PPO |
|---|---|---|
| Scratch | 4% | 8% |
| BC only | 17% | 10% |
| BC + RL(朴素) | 7% | 3% |
| SACfD(BC + 示教灌 buffer) | 27%(峰值 45%) | — |

**踩坑与洞察**:
1. BC 预热有效(4%→17%),但**朴素 BC+RL 会灾难性遗忘**——actor 被克隆、critic 还随机,
   首批 RL 更新最大化的是"垃圾 Q",把 actor 拉离 BC 好区域。on-policy PPO 无法在更新里复用示教,遗忘更狠。
2. **正确 off-policy 做法 = 示教灌入 replay buffer(SACfD)**(`mc/imitation/demo_buffer.py`,
   复用 SB3 的 `VecEnv.step→replay_buffer.add` 模式,HER 重标记照常工作),让 critic 从第 0 步见到成功轨迹
   → 27%,是唯一相对 BC 还在涨的配方。**IL 更适合配 off-policy 而非 on-policy。**
3. **但 Residual(88%)在同等预算下仍碾压所有 IL**:差别在**先验注入位置**——
   Residual 是**动作级结构先验**(base 每步都在干活),IL 是**数据级先验**(易被 RL 冲淡)。
   **接触丰富 + 算力受限 → 结构先验 > 数据先验。**

---

## Part 4. 这次重构的实施思路（设计决策，逐条可讲）

### 4.1 为什么要重构成框架 `mc/`
原来 4 个脚本各自带一份近乎相同的 ~90 行 `PlottingCallback`，路径硬编码、指标只有距离误差，
无法公平对比。重构目标：**一套环境工厂 + 一套指标 + 一套评估 harness = 任意控制器/策略可复现地公平对比**。

- `mc/common/envs.py`：环境工厂 + 观测解析（统一从这里 make_env，并把 Fetch 观测按字段命名解析）。
- `mc/common/eval.py`：统一评估 harness——只要是 `obs->action`（控制器、闭包、SB3 模型）都能评。
- `mc/common/training.py`：算法工厂（off-policy+HER / PPO dense）集中一处。
- `mc/common/{metrics,plotting,video,callbacks}.py`：指标 / 出图 / 录 gif / 回调。

### 4.2 控制器矩阵：经典 → RL → 混合（核心叙事）
把「传统 vs RL vs 结合」放到**同一任务、同一指标**下对比：

- **纯 PID / PD（任务空间反馈）**：经典反馈下界。
- **脚本化状态机（model-based）**：approach→descend→grasp→transport，规划+反馈，~100%。
- **端到端 RL（SAC+HER）**：数据驱动、泛化强但样本贵。
- **RL 自整定 PID**：保留 PID 可解释性，RL 做增益调度。
- **Residual RL（经典基线 + RL 残差）★ 核心**：`a = a_base(s) + α·a_RL(s)`。

### 4.3 为什么 Residual RL 是重点
- 从「已经能用」的控制器起步 → 样本效率/安全性远好于从零学。
- base 控制器兜底 worst-case（可解释、可控）。
- 残差专门吸收模型没建准的部分（接触、摩擦、标定误差、sim2real gap）。
- 出处：Johannink et al., *Residual RL for Robot Control*, ICRA 2019。
- 实测（PickAndPlace，4 万步同等预算）：从零 RL ~2%，**Residual RL 82%**——证明的是
  「结构化先验带来的样本效率」，不是某个算法更强（这点要主动点破，避免被说「对比不公平」）。

### 4.4 为什么 off-policy 用 sparse+HER、PPO 用 dense
- 目标条件 + 稀疏奖励的抓取/到达，是 **HER** 的教科书场景：把失败轨迹按「事后达到的目标」重标，
  凭空造出成功样本。
- **HER 只适用于 off-policy（有 replay buffer）**，所以 SAC/TD3/DDPG 用 HER+sparse；
  PPO 是 on-policy、没法用 HER，只能用 dense 奖励——这是个有意为之、可讲的取舍。

### 4.5 任务选择：PickAndPlace → Reach 的取舍
- PickAndPlace 适合展示「脚本化 model-based + Residual」的深度，但**极难、收敛慢**（CPU+4万步远不够，
  纯 RL 看起来像没训起来）。
- 因此对照演示切到 **FetchReach**：几千步即可收敛，能**干净地展示不同策略的「特性」**而非「谁没训好」：
  - 纯 model-based（PD）：零训练、确定性、能耗/jerk 最低、即时最优——但手工调、任务专用。
  - SAC（off-policy+HER）：样本效率高，几千步收敛。
  - PPO（on-policy）：稳定但更吃样本，收敛更慢。
  - Residual RL：从 base ~100% 起步，曲线一开始就高——安全先验 + 学习能力兼得。

---

## Part 5. 工程踩坑与解决（体现工程/系统能力）

| 问题 | 现象 | 解决 |
|---|---|---|
| 4 个 HER 进程并发 | 内存被 1M×4 的 replay buffer 撑爆，进程被 OOM 静默杀死（无 traceback） | `build_model` 加 `buffer_size=300k` 上限；并行更安全 |
| 环境版本 | `FetchPickAndPlace-v3` 在 gymnasium-robotics 1.4.x 已废弃 | 统一用 `-v4` |
| 加载模型报错 | `SAC.load` 用了 HER → 必须传 env | 加载时传一个同构 env |
| headless 渲染 | 服务器无显示器，`render_mode="human"` 没用 | `rgb_array` + imageio 录 gif，PPT 可直接用 |
| 后台日志丢失 | stdout 被块缓冲，进程被杀时日志没落盘 | `PYTHONUNBUFFERED=1`；CSV 即时 flush |
| 多任务观测不一致 | Reach 观测 10 维、PickAndPlace 25 维，字段布局不同 | `parse_obs` 按观测长度自动选布局 |

---

## Part 6. 面试三分钟讲解脚本（浓缩话术）

1. **一句话**：我做的是「用经典控制的语言设计 RL 的奖励与评估，再把传统控制器和 RL 结合」。
2. **reward 哲学**：从倒立摆开始，我的 reward 一直是 P/D/控制量三段式——误差项保证到位、
   速度项抑制超调、控制量项保证平滑；机械臂上的动作惩罚系数还是用量纲分析定的，不是瞎调。
3. **观测什么**：训练看 reward + actor/critic loss + 熵系数（loss 飞了就是学习率太大）；
   评估我建了一套运控指标——成功率/RMSE/超调/调节时间/能耗/jerk/路径，用数字说话。
4. **核心实验**：同一任务、同一指标，对比 PID / 脚本化 model-based / SAC / PPO / Residual RL。
   结论是 **Residual RL 把传统控制器当先验，用同样的 RL 算法拿到远更高的样本效率**——
   这正是「传统运控 + RL 结合」的价值，而且 base 控制器还提供了安全下界与可解释性。
5. **工程**：HER+sparse 与 PPO+dense 的取舍、OOM/版本/渲染等坑都踩过并解决，框架可一键复现。
6. **延伸**：关节力矩级计算力矩/逆动力学、策略蒸馏与真机部署、灵巧手与快慢系统（system 0/1）。
