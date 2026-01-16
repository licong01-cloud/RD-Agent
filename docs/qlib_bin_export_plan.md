# Qlib Snapshot → bin 导出功能设计（AIstock FastAPI + Next.js）

版本：v1.0（规划稿）  
范围：仅在 AIstock 项目内开发（FastAPI 后端 + Next.js 前端），不修改 rdagent 目录。

## 0. 目标与约束

- 保留现有 Qlib Snapshot 导出能力，新建“导出 Qlib bin（CSV→bin）”页签/流程。
- 流程：数据库 → 过滤（ST/退市/暂停）→ 导出 CSV → 调用 Qlib 工具 dump 为 bin → 校验。
- 不改 rdagent 代码；仅在 AIstock 后端/前端实现。

## 1. 数据要求（符合 Qlib dump）

- CSV 字段（日线示例）：`instrument,date,open,high,low,close,volume,amount`，日期格式 `YYYY-MM-DD`。  
  分钟线（预留）：`instrument,datetime,open,high,low,close,volume,amount`。
- instruments 文件：`instruments/all.txt` 与 CSV instrument 集合一致（过滤后再生成）。
- 过滤选项：
  1) 剔除曾经/当前 ST：查 `stock_st` 表，去除所有曾有 ST 记录及当前 ST 的证券。
  2) 剔除退市/暂停上市：查 `stock_basic`，去除退市或有暂停上市记录的证券。
- freq 与 dump 参数一致：`symbol_field=instrument`，`date_field=date|datetime`，`freq=day|1min...`。

## 2. 后端设计（FastAPI）

- 新增路由（示例）：
  - `POST /qlib/export/bin`：参数（频率/日期范围/过滤开关/输出路径可选）；返回 job_id 或同步结果。
  - `GET /qlib/export/bin/status/{job_id}`（如采用异步）。
  - `POST /qlib/export/bin/validate`（可选）：对生成的 bin 跑 qlib.init + D.features 快速校验。
- 处理流程：
  1) 根据过滤选项查询证券集合（stock_st、stock_basic）；统计样本数。
  2) 导出 CSV（字段符合 1.1），生成 instruments/all.txt。
  3) 调用 Qlib dump 工具（`dump_bin.py` 或官方 API）生成 bin 目录。
  4) 返回生成的 bin 路径、样本数、日期范围、字段列表；若异步则提供状态轮询。
- 校验（可选自动化）：`qlib.init(provider_uri=..., region="cn")` → `D.instruments` → `D.features([...], ["$close"], start/end, freq)`。

## 3. 前端设计（Next.js）

- 在 Qlib Snapshot 管理中新增页签“导出 Qlib bin（CSV→bin）”。
- 表单字段：日期范围、频率、市场、导出路径（可选）、过滤开关（ST、退市/暂停）。
- 交互：
  - “预览/统计”按钮：显示过滤后样本数、日期范围校验。
  - “执行导出”按钮：调用后端导出接口；显示进度/结果（bin 路径、样本数、字段列表）。
  - 可选：展示校验结果（D.instruments / D.features 通过与否）。

## 4. 校验方案

- 自动脚本（可复用现有 `test_qlib_snapshot_official_style.py` 逻辑）：  
  `qlib.init(provider_uri=bin_dir, region="cn"); D.instruments(...); D.features(..., freq="day")`。
- 校验通过条件：instruments 非空，features 能返回数据且行数>0。

## 5. 任务清单（进度占位）

- [ ] 后端：新增 `POST /qlib/export/bin`（含过滤逻辑、CSV 导出、dump 调用）。
- [ ] 后端（可选）：状态/校验接口。
- [ ] 前端：新增页签 UI（表单、过滤、预览、执行导出、结果展示）。
- [ ] 校验脚本/接口：对生成的 bin 做快速检测。
- [ ] 联调与文档更新：记录生成路径、参数示例、验证结果。

## 6. 需确认与输入

- CSV 输出目录与 bin 目标目录的默认路径。
- 是否需要异步任务（job_id + status）或同步执行即可。
- 使用的 Qlib 环境/命令（dump_bin.py 路径或调用方式）。
