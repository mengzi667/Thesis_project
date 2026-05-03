# Chapter 6 Results

## 6.1 Result Reporting Overview

说明本章的统一汇报口径：

- 两个 scenario 都按同一套结果逻辑报告
- Scenario 1 报告当前完整结果
- Scenario 2 目前保留结构占位，后续补充数值
- 固定 OR output，改变 downstream execution policy
- 使用 NB2 trip generation
- 使用 sparse / main / dense 三个 demand profiles
- 对比策略保持统一：no-offer / always-offer / learned policy


## 6.2 Simulation and Data Sanity Check

目的：证明后续结果的环境和数据输入是可信的。

内容：

- 每个 profile 的 generated trips 数量
- decision opportunities / OR matches 数量
- transitions 数量
- offer hit 情况
- sparse / main / dense 是否按预期递增
- NB2 是否体现 demand uncertainty

推荐表：

- Trip and opportunity statistics by profile
- Mean / variance / Fano-like dispersion of generated trips

图表插入建议：

- 表 6.1（放在 6.2 第一段“目的”后）  
  标题建议：`Trip and Decision Opportunity Statistics by Profile`  
  列建议：`profile, episodes, trip_count_mean, trip_count_var, fano_like, mean_transitions, mean_offers`  
  数据来源：`results/*/metrics/eval_summary.csv`（三 profile）。

- 图 6.2（放在表 6.1 之后）  
  标题建议：`Generated Trip Intensity Across Sparse/Main/Dense Profiles`  
  形式：柱状图（trip_count_mean），可加误差线（方差或标准差）。
  用途：证明 profile 强度递增。

- 图 6.3（放在图 6.2 之后）  
  标题建议：`Offer Opportunities and Transitions by Profile`  
  形式：分组柱状图（`mean_offers` 与 `mean_transitions`）。
  用途：证明决策样本密度变化，不是空跑。

## 6.3 Scenario 1 Results: Binary Offer Activation

### 6.3.1 Training Dynamics

内容：

- DDQN training reward curve
- loss curve
- epsilon decay
- offers during training
- acceptance rate during training
- reward component evolution

目的：

- 说明训练是否稳定
- 是否避免 always no-offer collapse
- 是否产生有效 offer activation 行为

图表插入建议：

- 图 6.4（放在 6.3.1 小节第一段后）  
  标题建议：`Training Reward Curve (Scenario 1, Main Profile)`  
  字段：episode-level `mean_reward`（可加 moving average）。

- 图 6.5（图 6.4 之后）  
  标题建议：`Training Loss and Epsilon Decay`  
  形式：双 y 轴折线（loss, epsilon）。

- 图 6.6（图 6.5 之后）  
  标题建议：`Offer Activation and Acceptance During Training`  
  字段：每轮 `offers`, `accept_rate`。  
  用途：直接判定是否塌缩到 no-offer。

### 6.3.2 Main Profile Policy Comparison

这是 Scenario 1 主结果。

对比：

- no-offer
- always-offer
- RL checkpoint

指标：

- service rate
- offers
- accepted relocations
- acceptance rate
- mean reward
- realized loss
- delta EDL
- reward decomposition

解释重点：

- RL 是否减少无效 offer
- RL 是否保持或提升 service rate
- RL 是否改善 EDL/reward
- 如果 realized loss 很低，要说明是当前供需条件导致的信号稀疏

图表插入建议：

- 表 6.2（放在 6.3.2 小节开头）  
  标题建议：`Main Profile Policy Comparison (Scenario 1)`  
  行：`no-offer / always-offer / checkpoint`  
  列：`service_rate, offers, accepted, acceptance_rate, mean_reward, mean_realized_loss, mean_delta_edl`。

- 图 6.7（表 6.2 后）  
  标题建议：`Main Profile KPI Comparison by Policy`  
  形式：分组柱状图，建议分两张子图：  
  子图 A：`service_rate, offers, accepted`；  
  子图 B：`mean_reward, mean_delta_edl, mean_realized_loss`。

### 6.3.3 Robustness Across Demand Profiles

内容：

- 同一个 frozen checkpoint 在 sparse / main / dense 下评估
- 不重新训练，只改变 simulator-side trip realization intensity
- 比较三策略在不同需求强度下的表现

推荐表：

- 三策略 × 三 profile 的 KPI 对比

解释重点：

- sparse 下机会少，策略差异可能较小
- dense 下机会更多，更容易观察 selective activation 的效果
- profile scaling 是 stress test，不改变 OR plan

图表插入建议：

- 表 6.3（放在 6.3.3 小节开头）  
  标题建议：`Policy Robustness Across Sparse/Main/Dense`  
  行：`profile × policy`（共 9 行）  
  列：`service_rate, offers, mean_reward, mean_delta_edl, mean_realized_loss`。

- 图 6.8（表 6.3 后）  
  标题建议：`Reward and EDL Robustness Across Profiles`  
  形式：折线图（x=profile，三条线=三策略，y 可分为 `mean_reward` 和 `mean_delta_edl` 两张子图）。

- 图 6.9（图 6.8 后）  
  标题建议：`Offer Activation Robustness Across Profiles`  
  形式：折线或柱图（x=profile，y=`offers`，三策略对比）。

### 6.3.4 Reward Decomposition and Operational Interpretation

内容：

- realized-loss term
- EDL term
- accept term
- cost term
- reject term

解释：

- 总 reward 的变化由哪一项驱动
- RL 是否只是少发 offer，还是改善了 EDL 或接受效果
- cost/reject 是否成功约束滥发 offer

图表插入建议：

- 表 6.4（放在 6.3.4 小节开头）  
  标题建议：`Reward Decomposition by Policy (Main Profile)`  
  列：`mean_reward_realized_term, mean_reward_edl_term, mean_reward_accept_term, mean_reward`（必要时加 sum）。

- 图 6.10（表 6.4 后）  
  标题建议：`Reward Component Breakdown`  
  形式：堆叠柱状图（按策略分组），展示 reward 各项贡献方向与大小。

### 6.3.5 Sensitivity Analysis

内容：

- reward 参数敏感性
- demand scaling 敏感性
- supply stress setting 如有结果可放这里

写法：

- 不需要报告所有组合
- 只报告关键组合和趋势
- 解释最终参数选择依据

图表插入建议：

- 表 6.5（放在 6.3.5 小节开头）  
  标题建议：`Sensitivity Test Matrix and Outcomes`  
  列：`run_id, parameter_changes, profile, episodes, mean_reward, offers, service_rate, mean_delta_edl`。

- 图 6.11（表 6.5 后）  
  标题建议：`Sensitivity Outcome Ranking`  
  形式：条形图（按 `mean_reward` 或综合评分排序）。  
  用途：给出“为什么最终用这组参数”的可视证据。

## 6.4 Scenario 2 Results: Joint Offer-Option Refinement

### 6.4.1 Training Dynamics Placeholder

占位内容：

- Scenario 2 will report training reward, loss, action distribution, and option-selection behavior.
- 当前无数值结果。

图表占位建议：

- 图 6.12（占位）`Scenario 2 Training Curves (to be added)`  
- 表 6.6（占位）`Scenario 2 Training Summary (to be added)`

### 6.4.2 Main Profile Policy Comparison Placeholder

占位内容：

- 后续将比较 no-offer / always-offer / Scenario 2 learned policy。
- 指标口径与 Scenario 1 保持一致。
- 额外增加 option-level 指标，例如 selected battery class 或 selected offer type。

图表占位建议：

- 表 6.7（占位）`Scenario 2 Main Profile Policy Comparison`  
- 图 6.13（占位）`Option-Level Action Distribution under Scenario 2`

### 6.4.3 Robustness Across Demand Profiles Placeholder

占位内容：

- 后续使用同样的 sparse / main / dense profiles。
- 检查 Scenario 2 在不同 demand intensity 下是否仍能选择合理 offer option。

图表占位建议：

- 表 6.8（占位）`Scenario 2 Robustness Across Profiles`  
- 图 6.14（占位）`Scenario 2 Reward/EDL Robustness`

### 6.4.4 Reward Decomposition and Operational Interpretation Placeholder

占位内容：

- 沿用 Scenario 1 reward decomposition。
- 后续增加 battery/option-level interpretation。

图表占位建议：

- 表 6.9（占位）`Scenario 2 Reward Decomposition`  
- 图 6.15（占位）`Scenario 2 Reward Component Breakdown`

### 6.4.5 Sensitivity Analysis Placeholder

占位内容：

- 后续分析 action-space expansion、battery option design、reward weights 对结果的影响。

图表占位建议：

- 表 6.10（占位）`Scenario 2 Sensitivity Matrix`  
- 图 6.16（占位）`Scenario 2 Sensitivity Ranking`

## 6.5 Cross-Scenario Comparison

目的：让两个 scenario 的结果最终能放在同一个逻辑框架下比较。

当前写法：

- Scenario 1 已实现并报告完整结果。
- Scenario 2 保留相同结果结构，等待后续实现。

后续填数时比较：

- Scenario 2 是否比 Scenario 1 提供更高 flexibility
- 是否带来更高 reward / lower EDL / better acceptance
- 是否因为 action space 更大导致训练更难

图表插入建议：

- 表 6.11（放在 6.5 小节末尾）  
  标题建议：`Cross-Scenario Comparison (Aligned Metric Schema)`  
  行：`Scenario 1 / Scenario 2`（Scenario 2 当前填占位）  
  列：`action_space, policy_complexity, mean_reward, mean_delta_edl, service_rate, training_stability`。

- 图 6.17（可选，占位）  
  标题建议：`Scenario-Level Comparison Radar/Bar`  
  用途：最终版本可直观展示两个 scenario 的 trade-off。

## 6.6 Chapter Summary

总结：

- 本章采用统一结果口径报告两个 scenario。
- Scenario 1 展示当前主实验结果。
- Scenario 2 保留完整结果结构，作为后续扩展。
- 主要结论围绕 service rate、offer efficiency、EDL improvement、reward decomposition 和 demand-profile robustness。

图表插入建议（可选）：

- 图 6.18（放在 6.6 最后一段后）  
  标题建议：`Chapter 6 Key Findings Snapshot`  
  形式：一个小型汇总图（例如 3-4 个关键指标的对比条图），用于帮助读者在章末快速回看核心发现。

---

## 图表命名与落地规则（执行时避免混乱）
  
- Scenario 2 占位图表标题中统一加 `to be added`，避免审稿误解为漏图。  
- 主文中首次引用建议顺序：先表后图（先给数字，再给可视化）。  
- 每张图/表标题都写 profile 与 policy 范围（如 `Main profile`, `Sparse/Main/Dense`, `Scenario 1`）。  
