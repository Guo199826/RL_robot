# 面试 PPT 大纲：传统运控 × 强化学习

> 配合本仓库的 demo / 图表使用。每页给出【标题 / 要讲的点 / 对应素材】。
> 主线一句话：**经典控制保证下界与可解释性，RL 突破上界与泛化，二者结合取长补短。**

---

### Slide 1 — 封面 / 自我定位
- 标题：强化学习 × 传统运控：机器人操作运动控制
- 一句话定位：做「传统 model-based 运控 + RL」结合的机器人运动控制。
- 素材：`results/gifs/scripted.gif`（机械臂抓取放置）。

### Slide 2 — 问题与动机（Why hybrid）
- 纯传统控制：可解释、稳定，但对接触/摩擦/标定误差建模困难，难泛化。
- 纯端到端 RL：泛化强、能处理接触，但样本消耗大、训练不稳、缺安全保证。
- 结论：**把结构化先验（控制）和数据驱动（RL）结合** → 安全 + 高效 + 泛化。
- 素材：一张三栏对比图（手画即可）。

### Slide 3 — 任务与平台
- 平台：MuJoCo + gymnasium-robotics。**主任务 `FetchReach-v4`**（快速收敛，干净展示策略特性）；
  **难任务 `FetchPickAndPlace-v4`**（展示 manipulation 深度）。
- 动作 4 维 `[dx,dy,dz,gripper]`；观测 reach 10 维 / 抓取 25 维（末端/物体位姿、速度等）。
- 评估指标体系：成功率 / 跟踪 RMSE / 稳态误差 / 调节时间 / 超调 / 控制能耗 / 动作 jerk / 路径长度。
- 素材：`mc/common/metrics.py`，`mc/common/envs.py`（讲观测分解）。

### Slide 4 — 🅰 控制器矩阵（核心）
- 同一任务、同一指标，对比四类控制器：
  1. **Model-based（PD 反馈）**：经典控制基线，无需训练。
  2. **SAC（off-policy + HER）**：端到端 RL。
  3. **PPO（on-policy, dense）**：端到端 RL。
  4. **Residual RL（PD 基线 + RL 残差）★**：传统 + RL 结合。
  （+ RL 自整定 PID、脚本化状态机在框架里也有）
- **核心讲点（reach 上的关键洞察）**：四者**成功率都 100%**——区分度不在"能不能到"，
  而在"**怎么到**"：看控制能耗与动作 jerk（抖动）。
  → Model-based 最平滑省力(energy 0.69 / jerk 0.67)；纯 SAC 抖动大(2.31 / 1.35)；
  **Residual RL 继承 PD 的平滑(1.05 / 0.82)又能学习** ★。
- 素材：`results/controller_matrix/controller_bars.png` + demo gif（`results/gifs/`）。

### Slide 5 — Residual RL 原理（重点深入）
- 公式：`a = a_base(s) + α · a_residual_RL(s)`，α 限幅保证安全。
- 好处：从「能用的策略」起步 → 样本效率/安全性远好于从零学；base 兜底 worst-case；
  残差专门吸收模型没建出来的部分（接触/摩擦/sim2real gap）。
- 出处：Johannink et al., Residual RL for Robot Control, ICRA 2019。
- 素材：`mc/controllers/residual.py`（现场可翻代码）。

### Slide 6 — 🅱 算法横评（样本效率 / 收敛特性）
- 成功率 vs 步数曲线，体现不同策略的收敛特性：
  - Model-based PD：0 步即 100%（参考线）；Residual RL：~2k 步即满分（继承先验）；
    SAC+HER：~8k 收敛（off-policy 样本效率高）；PPO：~18k 收敛（on-policy 更吃样本）。
- 讲点：HER 只适用于 off-policy，故 SAC 用 HER+sparse、PPO 用 dense；
  → off-policy+HER 在目标条件任务上更高效；Residual 因有先验最省样本。
- 素材：`results/benchmark/algo_benchmark_success.png`。
- （进阶难任务）PickAndPlace 上从零 RL 4 万步仅 ~2%、Residual RL 达 82%，
  见 `results/archive_pickandplace/`——更突出"先验带来的样本效率"。

### Slide 6.7 — 🅰′ 换任务:接触丰富的 PickAndPlace（**RL/混合不可替代**,主线高潮）
- 同一套控制器矩阵,换到需要 grasp 的抓取任务,**结论反转**:
  | Scripted 100% | 纯 SAC 4% | 纯 PPO 8% | **Residual RL 88%** |
- 讲点:接触/grasp 是离散非线性,**纯端到端 RL 在 CPU 预算内学不会**;手工状态机能解但脆;
  **Residual 把经典当先验、RL 学修正 → 兼得可靠 + 可学习**。这就是 JD 的"传统运控 + RL"。
- 素材:`results/controller_matrix_pap/controller_bars.png` + `demo_{scripted,sac,residual}.gif`。
- **两任务合起来 = 完整设计空间(本 PPT 主线)**:
  reach(良建模)经典赢 → PickAndPlace(接触丰富)RL/混合赢 → **"知道何时用/不用 RL"是核心判断力**。

### Slide 6.5 — 🅴 抗扰特性 / 扰动抑制（"突然被推一下"）
- 建模：Fetch 夹爪焊接到 mocap，外力被抵消 → 用一段强外源速度把末端"撞"开 9–18cm 来忠实模拟推力。
- 图1 `disturbance_recovery.png`（error-vs-time，红色窗口=推力）：所有稳定控制器都能抑制推力；
  **PD 稳态最准、恢复最干净；Residual(0.3) 反而稳态/恢复最差**——揭示"base 已近最优时残差是负担"。
- 图2 `convergence_disturb_vs_clean.png`：把推力作为域随机化注入训练 → SAC 收敛慢约 3×(6k→18k)，
  reach 上鲁棒性收益甚微 → **"为鲁棒性付训练成本"要看任务是否真需要**。
- 讲点：反馈控制器天然抗扰；扰动主要考验"稳态精度 + 恢复平滑度"；DR 是有成本的工程取舍。
- 素材：`mc/sim2real/disturbance.py`、`scripts/disturbance_robustness.py`、`results/disturbance/`。

### Slide 6.6 — 训练诊断曲线（怎么判断"真的在学/已收敛"）
- 每次训练自动产出 `training_curves.png`：reward / actor-loss / critic-loss / 熵系数。
- 讲点：SAC 熵系数 0.4→0.006(探索→利用)、actor-loss(=−Q) 随 Q 上升而抬高、critic-loss 收敛 →
  这些是判断训练健康度的关键，而非只看成功率。
- 素材：`results/benchmark/<algo>/training_curves.png`、`mc/common/callbacks.py`、`mc/common/plotting.py`。

### Slide 7 — 🅲 Sim2Real 鲁棒性
- 域随机化（物体质量/摩擦/作动增益）+ 观测噪声 + 控制延迟。
- 实验：有/无 DR 训练，各自在「干净」与「扰动」动力学下评估。
- 讲点：DR 训练的策略在扰动下成功率下降更小 → sim2real 迁移更稳。
- 素材：`results/sim2real/robustness.png`，`mc/sim2real/domain_randomization.py`。

### Slide 8 — 🅳 模仿学习 + RL
- 脚本控制器免费产生示教 → BC 预训练策略 → SAC+HER 微调。
- 讲点：BC 把策略放到好区域，RL 再超越（次优）示教者；样本效率优于从零。
- 素材：`results/imitation/`（BC 后成功率 + 微调曲线）。

### Slide 9 — 工程化 / 框架能力
- 统一框架 `mc/`：环境工厂、指标库、评估 harness、算法工厂、headless 录像。
- 一套指标 + 一套评估接口 = 任意控制器/策略可公平对比、可复现。
- 讲点：体现系统/框架开发能力（JD 加分项）。

### Slide 10 — 总结 & Roadmap
- 结论：传统运控 + RL 的结合（Residual RL / RL 自整定 / IL+RL）在效率、稳定、
  泛化、安全间取得更好折中。
- Roadmap：关节力矩级计算力矩/逆动力学对照、策略蒸馏与真机部署、灵巧手/接触丰富任务、
  快慢系统(system 0/1)、Isaac Lab 分布式训练。

---

## 现场 Demo 顺序建议
1. 放 `algo_benchmark_success.png` → 讲收敛特性（PD 即时 / Residual 开局即满分 / SAC 快 / PPO 慢）。
2. 翻 `controller_bars.png` → **关键转折**："成功率都 100%，那比什么？比能耗和 jerk"
   → PD 最平滑、纯 RL 最抖、Residual 兼得。用数字说话。
3. 放 demo gif（`results/gifs/`）→ 直观看四种策略的运动。
4. 翻代码 `mc/controllers/residual.py`（`a=a_base+α·a_RL`）+ `mc/common/training.py`（HER 取舍）
   → 体现工程功底。
5. 放 `disturbance/disturbance_recovery.png` → "途中推一下"，讲扰动抑制 + Residual 残差是把双刃剑；
   再翻 `convergence_disturb_vs_clean.png` → 带干扰训练收敛慢 3×，引出 DR 成本权衡。
6. 翻任一 `training_curves.png` → 讲怎么判断训练健康（熵系数衰减/actor-critic loss）。
7. **（主线高潮）换任务 PickAndPlace**：放 `controller_matrix_pap/controller_bars.png`
   + 三个 gif（scripted 稳但脆 / 纯 SAC 抓不起来 4% / residual 88%）→
   "reach 经典赢、这里 RL/混合赢"，落到"按任务选型"的判断力。
8. （可选）`robustness.png` 讲 sim2real（需先运行 `sim2real_robustness.py`）。
