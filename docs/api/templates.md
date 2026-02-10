# 模板管理 API（Phase 2）

> 适用范围：RD-Agent 调度器模板发布/历史/回滚接口。

## 1. 发布模板

### POST /templates/publish

#### 请求体

```json
{
  "scenario": "qlib",
  "version": "v2",
  "task_id": "task_001",
  "description": "实验版本说明",
  "base_version": "v1",
  "changed_files": [
    "rdagent/scenarios/qlib/prompts.yaml"
  ],
  "files": [
    {
      "path": "rdagent/scenarios/qlib/prompts.yaml",
      "content": "..."
    }
  ]
}
```

#### 错误响应

```json
{
  "detail": "Missing scenario or version"
}
```

#### 响应体

```json
{
  "status": "ok",
  "scenario": "qlib",
  "version": "v2",
  "output_dir": ".../app_tpl/qlib/v2",
  "manifest_path": ".../app_tpl/qlib/v2/manifest.json",
  "manifest_hash": "...",
  "backup_path": ".../history/template_bundles/20260120_153000_qlib_v2"
}
```

#### 校验规则

- `files` 为非空数组。
- `path` 必须以 `rdagent/` 开头，禁止 `..`。
- 仅支持 `.yaml/.yml/.json`。
- YAML/JSON 语法校验 + Jinja 语法校验。

---

## 2. 历史列表

### POST /templates/history

#### 请求体（历史）

```json
{
  "scenario": "qlib",
  "version": "v2"
}
```

#### 响应体（历史）

```json
{
  "items": [
    {
      "id": null,
      "file_name": "manifest.json",
      "backup_path": "...",
      "task_id": "task_001",
      "user": null,
      "hash": null,
      "created_at": "2026-01-20T15:30:00+08:00",
      "extra": {
        "action": "publish",
        "scenario": "qlib",
        "version": "v2",
        "manifest_path": ".../manifest.json"
      }
    }
  ]
}
```

---

## 3. 回滚模板

### POST /templates/rollback

#### 请求体（方式一：指定备份路径）

```json
{
  "backup_path": ".../history/template_bundles/20260120_153000_qlib_v2"
}
```

#### 请求体（方式二：按 scenario/version 回滚最新）

```json
{
  "scenario": "qlib",
  "version": "v2"
}
```

#### 响应体（回滚）

```json
{
  "status": "ok",
  "scenario": "qlib",
  "version": "v2",
  "output_dir": ".../app_tpl/qlib/v2",
  "backup_path": ".../history/template_bundles/20260120_153000_qlib_v2"
}
```

---

## 4. 备注

- 所有时间戳统一为北京时间（UTC+08:00）。
- 发布时会自动备份旧版本目录到 `history/template_bundles`。
- 该接口仅影响 `app_tpl` 与 history 目录，不影响核心 RD-Agent 执行链路。

## 5. 常见错误码

| 场景 | 错误消息示例 | 说明 |
| --- | --- | --- |
| 参数缺失 | `Missing scenario or version` | `scenario/version` 未传 |
| 文件空 | `files must be a non-empty list` | 未提供模板文件 |
| 路径非法 | `Template path must be under rdagent/` | 路径不在模板根目录 |
| 类型不支持 | `Unsupported template file type` | 非 YAML/JSON |
| 内容无效 | `Invalid .yaml content for template` | 语法解析失败 |
| Jinja 错误 | `Invalid Jinja template syntax` | 模板语法错误 |
| 历史缺失 | `No history records found for rollback` | 无可用回滚记录 |
