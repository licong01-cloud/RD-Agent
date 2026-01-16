# RD-Agent 调度系统设计与进度跟踪

## 0. 当前状态（进度总览）

- [ ] 后端 API / Worker：任务创建、查询、取消、日志、结果解析
- [x] 配置中心基础：`.env` + 5 个 Qlib 模板读写接口，写前自动备份（`config_service.py` 已落地）
- [x] 模板历史与回滚（基础版）：history 记录落地 JSONL，占位回滚函数（`storage.py`，`record_history` 钩子已接入）
- [x] 本地持久化占位：任务/数据集 JSONL 存储与操作（`task_service.py`）
- [x] Worker 与任务创建占位联通：`api_stub.py` 触发队列占位 `queue_stub.submit_task` → `worker_stub.run_rdagent_task`
- [x] 结果记录占位：`worker_stub` 调用 `record_result` 落地 returncode/工作目录/命令（`results.jsonl`）
- [x] 前端导航：左侧新增 “RD-Agent 调度” 按钮与路由
- [x] 前端页面（初版占位）：Streamlit UI（任务列表/新建、日志查看、结果查看、数据集管理），依赖 `rdagent.app.scheduler.server`
- [ ] 联调与验证：备份命名、回滚、基于历史版本运行任务

> 说明：本文件将随开发进度更新复选框，便于后续查阅。

## 1. 设计目标

- 在 AIstock 平台内调度 RD-Agent：创建/执行/监控任务。
- 保持 RD-Agent 源码不改动，通过 CLI/API 调用。
- 自动备份 YAML/.env，每次变更可回溯、可复用、可回滚。

## 2. 架构概览

- **前端**：AIstock 左侧导航新增 “RD-Agent 调度”，进入多 Tab 界面（任务管理 / 配置中心 / 数据集管理 / 运行监控）。
- **后端**：
  - `Scheduler API`：任务创建/查询/取消、日志拉取。
  - `Config Service`：统一读写 `.env` 与 5 个 Qlib 模板，写前自动备份。
  - `Worker/Queue`：执行 `rdagent fin_quant`，收集日志与结果。
  - `Backup Manager`：管理备份、版本记录、回滚/复用。
- **存储**：数据库记录任务、数据集、配置历史；文件系统存放 YAML/.env 备份与任务输出。

## 3. 配置与模板范围

**需管理的运行配置：**

1) `.env`（LLM Key、provider_uri、知识库路径等）  
2) Qlib 模板（5 个）  
   - `rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml`  
   - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml`  
   - `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml`  
   - `rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml`  
   - `rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml`  
3) Prompt 相关（仅展示/备份，不常改）  
   - `rdagent/scenarios/qlib/experiment/prompts.yaml`  
   - `rdagent/scenarios/qlib/prompts.yaml`  
   - `rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml`  
4) 备份文件：`history/YYYYMMDD/{timestamp}_{taskId}_{filename}`，已有 `.backup_*.yaml` 仅作参考。

## 4. 自动备份与回滚

- **触发**：任何通过 Config Service 的写操作（.env 或 5 个模板）。
- **命名**：`history/YYYYMMDD/{timestamp}_{taskId}_{filename}`；若无 taskId，用 `manual`。
- **元数据**：`template_history` 表（taskId、操作者、hash、备注、时间）。
- **回滚/复用**：
  - 配置中心/任务详情提供“版本历史”“一键回滚”。
  - 新建任务可选“基于历史版本”启动，后端将指定备份作为本次任务模板。

## 5. 后端设计

### 5.1 API（REST 示例）

- 任务：  
  - `GET /api/rdagent/tasks` 列表  
  - `POST /api/rdagent/tasks` 创建（dataset_ids、loop_n、all_duration、evolving_mode、sourceHistoryId）  
  - `GET /api/rdagent/tasks/{id}` 详情  
  - `POST /api/rdagent/tasks/{id}/cancel` 取消
- 配置：  
  - `GET /api/rdagent/env` / `PUT /api/rdagent/env`（保存前备份+校验）  
  - `GET /api/rdagent/templates`（5 个模板 hash/mtime）  
  - `GET /api/rdagent/templates/history?file=...`（历史列表）  
  - `POST /api/rdagent/templates/rollback`（回滚到历史版本）
- 数据集：  
  - `GET /api/rdagent/datasets`  
  - `POST /api/rdagent/datasets/upload-instruments`
- 日志与监控：  
  - `GET /api/rdagent/tasks/{id}/logs`（SSE/WebSocket/轮询）  
  - `GET /api/rdagent/tasks/{id}/artifacts`（结果文件列表/下载）

### 5.2 数据库表

- `rdagent_task`：参数、状态、配置 hash、workspace_path、结果摘要。  
- `rdagent_dataset`：name、provider_uri、instruments_file。  
- `rdagent_template_history`：file、history_path、taskId、hash、操作者、时间。  
- （可选）`rdagent_env_history`：.env 历史。

### 5.3 Worker 流程

1) 取 `pending` 任务 → 写 `.env`（如需更新）。  
2) 复制模板到临时 workspace（若指定历史版本则先回滚到临时副本）。  
3) 运行：

   ```bash
   cd /mnt/c/.../RD-Agent-main
   dotenv run --workspace {workspace} -- rdagent fin_quant --loop-n {loop_n} --all-duration "{duration}" --trace {task_id}
   ```

4) 捕获 stdout/stderr → 写 `rdagent_logs`。  
5) 解析 `ret.pkl/qlib_res.csv` → 生成指标摘要，更新任务状态。  
6) 打包产出文件/提供下载链接。

## 6. 前端设计

### 6.1 导航与路由

- 左侧菜单新增 “RD-Agent 调度”，路由 `/rdagent`，内含 Tabs。

### 6.2 Tabs 与功能

1) **任务管理**：表格（任务名、数据集、loop/duration、演进模式、状态、操作）；新建任务弹窗（数据集多选、loop、duration、演进模式、n_epochs/batch_size/n_jobs、知识库开关、配置来源：当前模板/历史版本）。  
2) **配置中心**：`.env` 关键字段编辑（LLM、Embedding、Qlib 路径、知识库路径等）；显示 5 个模板 hash/mtime；历史版本查看与回滚。  
3) **数据集管理**：dataset 列表；上传股票池 txt 生成 dataset；设置默认数据集。  
4) **运行监控/日志**：展示 loop 进度、指标、实时日志（SSE/WebSocket/轮询）、结果下载。

### 6.3 交互要点

- 保存配置需校验并 toast 反馈；LLM/Qlib 提供“测试”按钮。  
- 历史版本支持 diff/metadata；回滚需二次确认。  
- 日志视图支持暂停滚动/搜索。

## 7. 开发步骤与进度

### 阶段 1：后端基础

- [ ] 建表（任务、数据集、模板历史、可选 .env 历史）
- [ ] 实现 ConfigService（备份+写入+回滚）
- [ ] 任务 API、队列/Worker、日志采集、结果解析

### 阶段 2：前端页面

- [ ] 导航与路由
- [ ] 任务列表 + 新建任务表单（含历史配置选项）
- [ ] 配置中心（.env + 模板 hash/历史/回滚）
- [ ] 数据集管理
- [ ] 运行监控 + 日志视图

### 阶段 3：联调与验收

- [ ] 验证备份命名/存储、回滚与基于历史版本创建任务
- [ ] 校验 market/provider_uri/segments 在 5 个模板的一致性
- [ ] 失败日志可视化、任务状态回传、结果下载

## 8. 运行/操作规范

- 修改 `.env` 和 5 个模板一律通过后端接口，自动备份；禁止直接手改生产文件。  
- 新建任务前：若需复用历史配置，先在表单中选择对应版本；无需手动回滚主模板。  
- 若发现配置改坏：通过配置中心“版本历史”执行回滚，并记录原因。

---

> 维护说明：后续开发中完成某项任务时，请勾选对应复选框并简要记录变更，确保可追溯。
