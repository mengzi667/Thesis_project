# Master Sample Consistent Chain (生成-聚合-回放一致链路)

目标：让 OR 求解输入与仿真回放来自同一份样本，避免 `(o,d,t)` 失配。

## 1) 从 omega 生成主样本（master trips）

```powershell
cd D:\TUD\Thesis_project
python pipeline\build_master_trips_from_omega.py `
  --omega-csv sara_repo/data/30sep-omega_h.csv `
  --output data/input/master_trips.csv `
  --is-weekend 1 `
  --slot-minutes 15 `
  --slot0-hour 6 `
  --t-begin 0 `
  --t-end 56 `
  --seed 42 `
  --sample-mode poisson
```

输出：`data/input/master_trips.csv`

## 2) 由同一份 master trips 反聚合 OR 输入 omega

```powershell
python pipeline\aggregate_master_trips_to_omega.py `
  --master data/input/master_trips.csv `
  --output data/generated/omega_from_master.csv `
  --is-weekend 1 `
  --slot-minutes 15 `
  --slot0-hour 6
```

输出：`data/generated/omega_from_master.csv`

将该文件替换/拷贝到 Sara OR 运行所用的 `30sep-omega_h.csv`（或在 Sara 侧加参数读该文件）。

## 3) 运行 Sara OR（得到 incentive_plan）

示例：

```powershell
cd D:\TUD\Thesis_project\sara_repo
python main.py --config RH_hyb --t-begin 0 --t-end 56 --data-dir data
```

## 4) 把 Sara 输出转成 U_odit

```powershell
cd D:\TUD\Thesis_project
python -m or_model.prepare_uodit_from_sara `
  --input sara_repo/results/RH_hyb/detailed_results_0_56.xlsx `
  --output data/input/u_odit.csv `
  --slot-minutes 15 `
  --sheet incentive_plan
```

## 5) 仿真回放同一份 master trips

在 `config.py` 设置：

```python
TRIP_SOURCE = "replay"
TRIP_REPLAY_PATH = "data/input/master_trips.csv"
```

然后运行：

```powershell
python main.py
```

## 关键检查

- OR 与仿真必须共享同一个 `master_trips.csv` 来源链路
- `slot0_hour`、`slot_minutes` 必须一致
- zone 编码必须一致（1..110）

