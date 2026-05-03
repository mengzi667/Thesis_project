# 第五章《实验设计》内容结构要求

## 5.1 实验目标与比较逻辑

本节说明第五章的总体目标：评估 trip-level refinement layer 是否能改善 OR-guided relocation execution 的执行效果。

需要明确以下内容：

- 上游 OR model 保持固定。
- 实验只改变 downstream execution policy。
- 比较对象包括三种策略：
  - `no-offer`
  - `always-offer`
  - `checkpoint`
- `checkpoint` 表示训练后保存的 RL policy，并在评估阶段以 greedy 方式执行。
- `always-offer` 是执行层基线，表示所有可用 OR offer 都被激活；它不等价于 OR objective value 本身。

---

## 5.2 统一实验环境与数据口径

本节说明所有实验共享的环境设置、数据来源和需求生成机制。

需要包括：

- 所有实验使用同一个 event-driven simulator。
- 所有策略共享：
  - 同一 OR 输入表
  - 同一 user choice model
  - 同一 battery transition model
  - 同一 episode length
  - 同一 seed control 规则
- 每个 episode 长度为 2 小时。
- OR planning slot 为 15 分钟。
- Trip arrival 使用 NB2：
  - NB2 负责采样每个 OD-slot 中发生多少 trips。
  - 每个 slot 内的具体 arrival timestamp 使用 uniform distribution 采样。
- 定义三种 demand profile：
  - `sparse`
  - `main`
  - `dense`
- 三种 profile 只改变 simulator-side trip realization intensity。
- 三种 profile 不改变 OR output、OR interface 或 OR planning logic。

---

## 5.3 Scenario 1: Binary Offer Activation

本节是当前已经完整实现的主要实验对象。

### 5.3.1 Decision Scope and Action Space

需要说明：

- RL 只在 eligible OR-matched trip event 上触发。
- 一个 trip event 必须同时满足：
  - 当前 trip 与 OR output 中某条 relocation opportunity 匹配。
  - origin zone 有可租用 scooter。
- 动作空间是二值：
  - `0 = no_offer`
  - `1 = offer`
- Scenario 1 中 RL 不选择车辆。
- Scooter assignment 保持 highest-battery-first 规则。

### 5.3.2 State Representation and User Outcome

需要说明 state vector 包含哪些信息：

- temporal context
- origin / original destination / recommended destination identifiers
- origin、original destination、recommended destination 三个 zone 的库存状态
- EDL-related local context
- trip and offer attributes
- quota / budget context

需要说明 user behavior：

- 当 RL 选择 `no_offer`：
  - 用户选择集合为 `base / opt-out`
- 当 RL 选择 `offer`：
  - 用户选择集合为 `offer / base / opt-out`
- 用户最终选择由 single-layer choice model 生成。
- RL 只决定是否激活 offer，不直接决定用户是否接受。

### 5.3.3 Reward Design

需要说明 reward 由以下部分组成：

- realized loss
- delta EDL
- accept term
- cost term
- reject term

需要解释：

- `L_i` 表示 realized loss。
- `L_i` 的统计窗口为：
  - 从当前 decision step 到下一个 decision step；
  - 如果没有下一个 decision step，则到 episode end。
- `L_i` 的统计对象为当前 decision 相关的三个 zone：
  - origin zone `O_i`
  - original destination zone `D_i`
  - recommended destination zone `R_i`
- `L_i` 只统计 no-supply loss。
- `Delta EDL_i` 表示 realized outcome 相对于 no-relocation counterfactual 的 expected demand loss 改善。
- Reward 是 delayed reward：
  - 当前 decision 的 reward 在下一个 relevant decision state 出现时 finalized；
  - 或在 episode end finalized。
- `L_ref` 和 `E_ref` 是数值归一化锚点：
  - 用 decision-level samples 的分位数标定；
  - 用于稳定不同 reward 项的尺度；
  - 不是物理常数。

### 5.3.4 Training Protocol

需要说明：

- Scenario 1 使用 DDQN。
- 训练只在 `main` demand profile 下进行。
- 使用 binary action space。
- OR input 固定。
- 训练和评估使用不同 seed range。
- 训练过程中保存 checkpoint。
- 训练过程中记录：
  - episode-level metrics
  - transition logs
  - reward decomposition

### 5.3.5 Evaluation Protocol

需要说明：

- 使用同一个 frozen checkpoint 进行评估。
- 评估在三个 profile 下分别进行：
  - `sparse`
  - `main`
  - `dense`
- 每个 profile 内都比较三种策略：
  - `no-offer`
  - `always-offer`
  - `checkpoint`
- 三种策略在同一 profile 内共享相同：
  - simulator settings
  - OR input
  - demand sampling profile
  - seed schedule
- `main` profile 用于主结论。
- `sparse` 和 `dense` 用于 robustness / stress testing。

---

## 5.4 Scenario 2: Joint Offer-Option Refinement (Planned)

本节保留 Scenario 2 的结构，但当前不报告实验结果。

需要明确：

- Scenario 2 是后续扩展方向。
- 当前 thesis 阶段只保留设计位置。
- 本节不应写成已经完成的实验。

### 5.4.1 Decision Scope and Action Space (Planned)

需要说明：

- Scenario 2 中 RL 不只决定是否给 offer。
- RL 还可能决定 offer option，例如：
  - vehicle / battery class
  - candidate relocation option
  - offer package
- 当前阶段不实现，只作为扩展方向。

### 5.4.2 State Representation and User Choice Coupling (Planned)

需要说明：

- Scenario 2 需要更完整的 state representation。
- State 可能包括：
  - candidate scooters
  - scooter battery class
  - option-level attributes
  - user choice set information
- User choice model 会与 RL action 更紧密耦合。

### 5.4.3 Reward and Constraints (Planned)

需要说明：

- Reward 仍然围绕以下目标：
  - service loss
  - EDL improvement
  - incentive cost
  - user acceptance
- 但 Scenario 2 需要额外考虑：
  - option feasibility
  - battery-class effects
  - budget constraints
  - quota constraints

### 5.4.4 Training and Evaluation Protocol (Planned)

需要说明：

- Scenario 2 需要重新定义：
  - action space
  - baseline policies
  - state vector
  - evaluation tables
- 当前章节只保留 protocol 位置，不报告数值结果。

---

## 5.5 Metrics and Reporting Protocol

本节说明结果章节将如何汇报实验结果。

需要包括两类指标。

### Operational KPIs

包括：

- service rate
- unserved requests / no-supply count
- number of activated offers
- number of accepted offers
- acceptance rate

### Reward and Mechanism Diagnostics

包括：

- mean reward
- reward realized-loss term
- reward EDL term
- reward accept term
- realized loss
- delta EDL

### Aggregation Rule

需要说明：

- 所有关键指标至少报告 mean。
- 对 reward decomposition 和 physical quantities，必要时报告 sum。
- 可以报告 seed-level dispersion 或 confidence interval。

---

## 5.6 Sensitivity Analysis Design

本节说明敏感性分析设计。

需要说明：

- 在主模型固定后进行敏感性分析。
- 敏感性分析关注：
  - reward coefficients
  - normalization anchors
  - learning settings
  - demand scaling settings
- 使用 short screening + longer confirmation 的流程：
  - 先用短训练/短评估筛掉明显较差组合；
  - 再对表现较好的组合进行更长评估。
- 敏感性分析的目标不是寻找绝对最优参数。
- 目标是验证主结论是否对合理参数扰动保持稳定。

---

## 5.7 Reproducibility and Implementation Notes

本节说明实验如何复现。

需要包括：

- 训练入口：
  - `python -m rl.train`
- 评估入口：
  - `python -m rl.evaluate`
- 每次 run 记录：
  - demand profile parameters
  - seed range
  - checkpoint path
  - reward parameters
  - OR input path
- 输出文件包括：
  - episode-level metrics
  - transition logs
  - summary CSV
  - checkpoint files

---

## 5.8 Chapter Summary

本节总结第五章内容。

需要说明：

- 本章定义了 Scenario 1 的完整训练与评估协议。
- 本章保留了 Scenario 2 的扩展结构。
- 下一章将基于本章定义的协议报告 Scenario 1 的实验结果。
