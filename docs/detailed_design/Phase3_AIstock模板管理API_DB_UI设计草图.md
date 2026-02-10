# Phase 3：AIstock 模板管理 API/DB/UI 设计草图

## 1. 目标与范围

- AIstock 侧提供模板版本化管理能力（展示、编辑、发布、回滚、审计）。
- RD-Agent 侧仅负责模板落盘与运行加载（app_tpl），不改核心执行链路。
- AIstock 通过 API 与 RD-Agent 对接，实现模板发布与回滚。

## 2. 架构与职责

### 2.1 责任边界

- **AIstock**：模板版本数据管理、内容编辑、审计、发布编排。
- **RD-Agent**：接收模板包、写入 app_tpl、生成 manifest、历史记录与回滚。

### 2.2 数据流

1. AIstock 创建或编辑模板版本（保存结构化内容）。
2. AIstock 点击发布，调用 RD-Agent `/templates/publish`。
3. RD-Agent 校验并落盘 `app_tpl/<scenario>/<version>`，生成 `manifest.json`。
4. 发布结果回传 AIstock（manifest hash / 版本状态）。

## 3. 数据模型设计（AIstock DB）

### 3.1 TemplateVersion

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | UUID | 主键 |
| scenario | string | 场景（如 qlib） |
| version | string | 版本号（v1/v2...） |
| base_version | string | 基线版本（v0） |
| description | string | 版本说明 |
| changed_files | json | 变更文件列表 |
| status | enum | draft/published/archived |
| created_by | string | 创建人 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

### 3.2 TemplateFile

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | UUID | 主键 |
| template_version_id | UUID | 外键 |
| path | string | 相对路径（rdagent/...） |
| content | text | 文件内容 |
| sha256 | string | 内容哈希 |
| size | int | 内容大小 |
| file_type | enum | prompt/config/other |

### 3.3 PublishHistory

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | UUID | 主键 |
| template_version_id | UUID | 外键 |
| publish_status | enum | success/failed |
| manifest_hash | string | RD-Agent 回传 hash |
| error_message | text | 失败原因 |
| created_at | datetime | 发布时间 |

## 4. API 设计（AIstock <-> RD-Agent）

### 4.1 RD-Agent 侧 API（Phase 2）

- `POST /templates/publish`
  - 请求：`{scenario, version, description, base_version, changed_files, files[]}`
  - files[]：`{path, content, sha256}`
  - 响应：`{status, manifest_hash, manifest_path}`

- `GET /templates/history`
  - 返回模板发布历史

- `POST /templates/rollback`
  - 请求：`{scenario, version}` 或 `backup_path`

### 4.2 AIstock 侧 API（面向 UI）

- `POST /template_versions`
- `GET /template_versions?scenario=`
- `GET /template_versions/{id}`
- `PUT /template_versions/{id}`
- `POST /template_versions/{id}/publish`
- `POST /template_versions/{id}/rollback`

## 5. UI 设计草图

### 5.1 模板版本列表

- 列表字段：scenario、version、description、status、updated_at
- 支持筛选与搜索

### 5.2 模板详情

- 左侧：文件树（路径 + 类型）
- 右侧：内容预览（YAML/Prompt 语法高亮）
- 支持 diff（与 base_version 或上一版本比较）

### 5.3 编辑与发布

- 编辑页面：内容编辑 + schema 校验
- 发布弹窗：版本说明、变更摘要、发布确认

## 6. 校验与安全

- 发布前校验：YAML/Jinja 语法、路径白名单（只允许 rdagent/scenarios 下的模板）。
- 发布后校验：manifest hash 回传与存档。
- 审计日志：记录用户、时间、变更文件。

## 7. 版本策略

- v0 仅作为基线快照，不可改。
- v1/v2/v3 等按版本递增，不建议覆盖同版本。
- UI 强制填写 description 与 changed_files（可自动从 diff 生成）。

## 8. 最小落地顺序

1. AIstock 侧数据模型落地（TemplateVersion/TemplateFile/PublishHistory）。
2. UI 列表与详情只读展示。
3. 发布 API 对接 RD-Agent（publish/rollback）。
4. 编辑与 diff 功能。
