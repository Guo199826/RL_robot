# RL × Classical Motion Control for Robot Manipulation
### 强化学习 × 传统运控：机器人操作运动控制 Demo

> 面向「传统运控 + 强化学习结合」的机器人运动控制算法岗位的面试 Demo。
> 主线：**经典控制 → 端到端 RL → 两者结合（取长补短）**，在 MuJoCo 机械臂任务上做系统对照。
> 主演示任务 `FetchReach-v4`（快速收敛、干净展示策略特性），更难的 `FetchPickAndPlace-v4` 展示
> manipulation 深度（脚本化状态机 + Residual 抓取）。

---

## 0. TL;DR / 一句话

我在同一个机械臂任务上，把 **经典 PID/PD 反馈 / 脚本化 model-based 控制器 / 端到端 RL(SAC,PPO) /
RL 自整定 PID / Residual RL(经典控制+RL残差)** 放在统一指标体系下对比，并配套
**Sim2Real 域随机化鲁棒性**、**模仿学习+RL** 两组实验，说明「为什么要把传统运控和 RL 结合，
以及怎么结合」。核心发现：当各策略成功率都达 100% 时，**经典控制最平滑省力、纯 RL 抖动大，
而 Residual RL 兼得平滑与可学习性**。

## 1. 为什么这样设计（对照岗位 JD）

| JD 要求 | 本仓库对应 |
|---|---|
| 机械臂 / 全身运动控制的 RL | `FetchReach` / `FetchPickAndPlace`，SAC+HER 端到端 |
| PPO/SAC/TD3/DDPG 主流算法实践 | 🅱 算法横评 `scripts/benchmark_algos.py`（演示用 SAC/PPO，TD3/DDPG 已支持） |
| Isaac/MuJoCo 仿真 + sim2real | MuJoCo + 🅲 域随机化鲁棒性 `scripts/sim2real_robustness.py` |
| 模仿学习与 RL 结合 | 🅳 脚本示教→BC→RL 微调 `scripts/imitation_bc_rl.py` |
| 运动学/动力学、manipulation、运控 | 🅰 控制器矩阵：PID / 状态机 / Residual RL |
| model-based 传统运控 | 任务空间 PID + 轨迹跟踪 + 脚本化状态机（规划+反馈）作为 model-based 基线 |
| 真机/反馈 RL、提升泛化与精准 | Residual RL（经典控制器兜底 + RL 残差修正）、观测噪声/延迟 |
| 蒸馏与真机部署 | 见 §6 Roadmap（已留接口） |

> 说明：Fetch 是 **mocap 笛卡尔速度控制**（动作 = 末端位移 + 夹爪），不是关节力矩控制，
> 因此这里的 "model-based" = **任务空间规划 + 前馈/反馈**。关节力矩级的 **计算力矩/逆动力学**
> 控制更适合在力矩控制臂（如自建 2–3 连杆臂 / Reacher）上展示，已列入 Roadmap。

## 2. 仓库结构

```
mc/                         # 复用框架（motion control）
  common/                   # 基建：环境工厂 / 指标 / 回调 / 绘图 / 录像 / 评估 / 训练
    envs.py                 #   make_env + Fetch 观测解析（兼容 Reach 10维 / 抓取 25维）
    metrics.py              #   成功率/跟踪RMSE/超调/调节时间/能耗/jerk/路径长度
    eval.py                 #   统一评估 harness（任何 obs->action 都能评）
    training.py             #   算法工厂：off-policy+HER / PPO dense
    callbacks.py            #   成功率评估回调 + 曲线日志回调
    plotting.py / video.py  #   出图 / rgb_array->gif（headless 友好）
  controllers/              # 🅰 控制器矩阵
    pid.py                  #   任务空间 PID/PD（经典反馈，reach 的 model-based 基线）
    scripted.py            #   脚本化状态机（pick-and-place，model-based，~100%）
    rl_pid.py              #   RL 自整定 PID（增益调度）
    residual.py           #   Residual RL（经典基线 + RL 残差）★ 核心混合范式
    __init__.py           #   base_for_task(task)：按任务自动选经典基线
  sim2real/                 # 🅲 域随机化 / 观测噪声 / 外扰
  imitation/                # 🅳 脚本示教采集 + BC 预训练
scripts/                    # 可直接运行的实验入口
  benchmark_algos.py        # 🅱 SAC/TD3/DDPG/PPO 横评（--task/--disturb）
  controller_matrix.py      # 🅰 控制器矩阵评估 -> 表/柱状图/误差曲线/gif
  train_residual.py         # 训练 Residual RL（--task）
  sim2real_robustness.py    # 🅲 有无域随机化 + 扰动下交叉评估
  imitation_bc_rl.py        # 🅳 BC 预训练 -> RL 微调
  record_demo.py            # 录制 PPT 用 gif
results/                    # 所有产出（csv / png / gif / 模型）
pendulum/ , robot_arm/      # 早期学习笔记（保留，未改动）
```

## 3. 安装

```bash
conda env create -f environment.yml      # 或 pip install -r requirements.txt
conda activate rl_robot
```

## 4. 复现实验

默认任务是 **`FetchReach-v4`**（几千步即收敛，适合干净地展示不同策略的"特性"）；
框架同样支持更难的 `FetchPickAndPlace-v4`（脚本化状态机 + Residual 抓取，见 §5.2）。

```bash
# 🅱 算法横评（off-policy 用 HER+sparse，PPO 用 dense）
python scripts/benchmark_algos.py --algos SAC --timesteps 15000 --eval-freq 2000
python scripts/benchmark_algos.py --algos PPO --timesteps 60000 --eval-freq 3000

# 🅰 训练 Residual RL（PD 基线 + RL 残差），再评估整个控制器矩阵
python scripts/train_residual.py --task FetchReach-v4 --timesteps 12000
python scripts/controller_matrix.py --task FetchReach-v4 --episodes 50 --gif

# 🅴 抗扰实验：reach 途中突然"推一下"，看各控制器的扰动抑制特性
python scripts/disturbance_robustness.py --episodes 40 --push-step 28 --duration 6 \
    --sac-disturb results/benchmark/sac_disturb/model.zip
# 训练带随机推力的鲁棒 SAC（看干扰如何影响收敛）
python scripts/benchmark_algos.py --algos SAC --timesteps 20000 --disturb

# 🅲 Sim2Real / 🅳 模仿学习（代码就绪，按需运行；PickAndPlace 更能体现）
python scripts/sim2real_robustness.py --timesteps 60000
python scripts/imitation_bc_rl.py --demos 200 --bc-epochs 30 --timesteps 30000

# 录制 demo gif（PPT 用）
python scripts/record_demo.py --task FetchReach-v4 \
    --sac results/benchmark/sac/model.zip --ppo results/benchmark/ppo/model.zip \
    --residual results/residual/model.zip
```

产出位置：`results/benchmark/`, `results/controller_matrix/`, `results/gifs/`。

## 5. 关键结果

### 5.1 主任务 FetchReach-v4（全部收敛，展示「策略特性」）

> 重点：reach 简单到**四种控制器都 100% 成功**——于是差异体现在「**怎么到**」：
> 控制能耗与动作平滑度（jerk）。这正是运控工程师真正关心的维度。
> 图：`results/benchmark/algo_benchmark_success.png`、`results/controller_matrix/controller_bars.png`。

**样本效率 / 收敛特性**（成功率 vs 步数）：

| 策略 | 收敛到 100% 所需步数 | 特性 |
|---|---|---|
| Model-based（PD 反馈） | 0（无需训练） | 确定性、即时最优，但手工调、任务专用 |
| Residual RL（PD 基线 + RL 残差） | ~2k（开局即满分） | 继承先验，安全 + 可学习 ★ |
| SAC（off-policy + HER） | ~8k | 样本效率高，快速收敛 |
| PPO（on-policy, dense） | ~18k | 稳定但更吃样本 |

**控制质量对比**（50 回合实测，成功率均 100%）：

| 控制器 | 跟踪RMSE | 控制能耗 | 动作 jerk(抖动) | 说明 |
|---|---|---|---|---|
| **Model-based (PD)** | **0.104** | **0.69** | **0.67** | 最平滑、最省力（经典控制优势） |
| SAC (+HER) | 0.122 | 2.31 | 1.35 | 能到位但激进/抖动（RL 易 bang-bang） |
| PPO (dense) | 0.119 | 1.63 | 1.11 | 居中 |
| **Residual RL** | 0.109 | 1.05 | 0.82 | 继承 PD 的平滑，远好于纯 RL ★ |

> **核心结论（面试可直接讲）**：当"能不能到"不再有区分度时，比的是"控制质量"。
> 经典 PD 最平滑省力；纯 RL 虽到位但抖动大、能耗高；**Residual RL 把经典控制器当先验，
> 既保留了平滑/低能耗，又具备学习/泛化能力**——这正是「传统运控 + RL 结合」的价值。

**训练诊断曲线**（reward / actor-loss / critic-loss / 熵系数，类经典 SB3 诊断图）：
每次训练自动产出 `results/benchmark/<algo>/training_curves.png`、`results/residual/training_curves.png`。
SAC 的熵系数从 0.4 自适应衰减到 ~0.006（探索→利用），actor-loss(=−Q) 随 Q 上升而抬高，
是判断"是否真的在学/已收敛"的关键依据。

### 5.2 抗扰特性：reach 途中"突然被推一下" 🅴

> Fetch 夹爪刚性焊接到位置控制 mocap 上，外力(`xfrc`)几乎被约束抵消；
> 故用**一段强外源速度指令把末端"撞"开 ~9–18cm**（控制器无法阻止）来忠实建模"推一下"，
> 之后控制器必须自行纠回——即经典的**扰动抑制**测试。图：`results/disturbance/`。

**1) 扰动抑制（用已训练好的策略，无重训）** —— `disturbance_recovery.png`

| 控制器 | 推后峰值偏差 | 恢复步数 | 稳态精度(推前) | 观察 |
|---|---|---|---|---|
| **Model-based (PD)** | 0.18m | 5 | **最优(~0)** | 反馈天然抗扰，恢复最干净、稳态最准 |
| SAC (clean) | 0.18m | 4 | ~0.02m | 快速恢复，有小稳态偏差 |
| SAC (push-trained) | 0.19m | 4 | ~0.02m | 与 clean 几乎一致(reach 太简单，DR 收益有限) |
| Residual RL (scale 0.3) | 0.18m | 7 | ~0.04m | 残差给近最优 base 添了噪声→稳态/恢复反而最差 |

> **结论**：在 reach（反馈主导的简单任务）上，**所有稳定控制器都能抑制推力**；
> 区分度在"稳态精度 + 恢复平滑度"，PD 最优。**这也暴露了一个 nuanced 洞察:
> 当 base 已接近最优(PD on reach)，Residual 的残差是"负担"而非增益**；
> 它真正的价值在 base 不足的硬任务(PickAndPlace: 2%→82%)。

**2) 干扰如何影响"收敛"** —— `convergence_disturb_vs_clean.png`

把随机推力作为**域随机化**注入训练后,SAC 收敛明显变慢:
clean 训练 ~6k 步到 100%,带推力训练需 ~18k 步——因为策略要学会覆盖更宽的(被推后)状态分布。
代价是收敛慢约 3×,而在 reach 上鲁棒性收益甚微——**说明"为鲁棒性付训练成本"要看任务是否真需要**。

**3) 一个诚实的负结果:为什么 reach 体现不出 RL 优势(即便加延迟+域随机化)** —— `results/latency/latency_robustness.png`

进一步追问"换更难的扰动能否让 RL 反超?",做了控制延迟 + 宽域随机化(增益 0.4–1.6×+噪声)下的
跟踪 RMSE 扫描(`scripts/latency_robustness.py`):**调好的 PD(kp=8)在所有延迟下都赢**;
DR+延迟训练的 SAC 只是"最鲁棒的 RL"(高延迟下比 clean-SAC/热调-PD 平),但始终不及简单 PD。

> **核心洞察(强烈建议面试讲)**:FetchReach 是**反馈主导**的任务——动作是末端笛卡尔速度,
> 环境内置控制器已解了 IK+关节动力学,留给 agent 的几乎是个线性 P 问题,闭环反馈对增益/扰动天生鲁棒。
> 所以**reach 适合展示"经典控制的优越性 + RL 的代价",而非 RL 的优势**。RL 的不可替代性在
> *接触丰富/模型失配/力矩级非线性* 的任务(本仓库 PickAndPlace:脚本 2% → Residual 82%)。
> 知道"何时不该用 RL",本身是运控工程师的成熟判断。

### 5.3 更难任务 FetchPickAndPlace-v4（contact-rich，**RL/混合的不可替代性**）

reach 上经典控制赢;**换到接触丰富的抓取任务,结论反转**——这正是 RL 的主场。
图:`results/controller_matrix_pap/{controller_bars.png,error_vs_time.png,demo_*.gif}`。

| 控制器 | 成功率 | final_error | 说明 |
|---|---|---|---|
| Scripted（model-based 状态机） | **100%** | 0.06 | 可靠,但**全靠手工编码、任务专用、脆** |
| SAC (+HER) 纯 RL | **4%** | 0.31 | 从零端到端**学不会接触抓取**(稀疏奖励+CPU 预算) |
| PPO 纯 RL | **8%** | 0.15 | 同上,且路径冗长(乱探索) |
| **Residual RL（脚本先验 + RL 残差）** | **88%** | 0.05 | **继承先验的可靠性 + 可学习** ★ |

> **核心结论**:抓取需要离散的接触/grasp 阶段,**纯端到端 RL 在 CPU 预算内无法发现(4–8%)**;
> 手工状态机能解(100%)但脆且不泛化;**Residual RL 把经典控制器当先验、用 RL 学修正,拿到 88%**——
> 兼得"可靠性 + 可学习/可适应"。这就是 JD 要的「传统运控 + 强化学习结合」的价值。

**两任务合起来构成完整设计空间(面试主线)**:
- **FetchReach(良建模、反馈主导)** → 经典 PD 最优,RL 是负担(§5.1/5.2);
- **FetchPickAndPlace(接触丰富)** → 纯 RL 失败、经典脆,**Residual RL 不可替代**(§5.3)。
- 一句话:**知道每种方法的适用边界,按任务选型**——这是运控工程师的核心判断力。

> 完整算法横评(含 TD3/DDPG)及更早的训练曲线见 `results/archive_pickandplace/`。

### 5.4 给 SAC/PPO 加模仿学习(IL):BC 预热 + RL 微调 🅳

用脚本控制器免费产生 200 条近最优示教,做行为克隆(BC)预热再 RL 微调
(`scripts/imitation_bc_rl.py`)。图:`results/imitation/il_comparison.png`。成功率(PAP):

| 配方 | SAC | PPO | 解读 |
|---|---|---|---|
| Scratch(纯 RL) | 4% | 8% | 从零学不会抓取 |
| **BC only**(只克隆,不 RL) | **17%** | 10% | IL 一步把策略拉进"会抓"的区域 |
| BC + RL(朴素微调) | 7% | **3%** | **灾难性遗忘**:critic 冷启动→把 BC 成果冲掉(on-policy PPO 尤其严重) |
| **SACfD**(BC + 示教灌入 replay buffer) | **27%**(峰值 45%) | — | **唯一"比 BC 更好"的 RL 配方**:critic 一开始就见到成功轨迹 |

> **三个层层递进的洞察(面试可讲)**:
> 1. **BC 预热确实有效**(SAC 4%→17%),但**朴素 BC+RL 会遗忘**——actor 被克隆、critic 还随机,
>    早期 RL 更新把 actor 拉离好区域;**on-policy 的 PPO 无法在更新里复用示教,遗忘更彻底**(10%→3%)。
> 2. **正确的 off-policy 做法是把示教灌进 replay buffer(SACfD)**,让 critic 从第 0 步就有成功信号 →
>    27%、且是唯一相对 BC 还在涨的配方。**IL 与 off-policy 的组合远好于 on-policy**。
> 3. **但在这个 CPU 预算下,Residual(88%)仍碾压所有 IL 配方**——关键在于**先验注入的位置**:
>    Residual 把经典控制器作为**动作级结构先验**(每步都在干活),IL 只是**数据级先验**(易被冲淡)。
>    **结论:接触丰富 + 算力受限时,结构先验(Residual)> 数据先验(IL)。**

**🅲 Sim2Real** / **🅳 模仿学习+RL**：代码就绪（`scripts/sim2real_robustness.py`、
`scripts/imitation_bc_rl.py`），在 PickAndPlace 上更能体现，按需运行。

## 6. Roadmap（面试可口头延伸）

- 关节力矩级 **计算力矩/逆动力学控制** 对照（力矩控制臂 / Reacher）。
- **策略蒸馏**：大策略→轻量策略，性能 vs 推理时延对照，面向真机部署。
- 灵巧手 / 接触丰富任务、快慢系统（system 0/1）结合。
- Isaac Lab 上的并行采样与分布式训练。
