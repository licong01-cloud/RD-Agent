# log-only同步脚本分析报告

## 一、当前实现分析

### 1.1 关键代码位置

**文件**：`f:/Dev/AIstock/backend/services/rdagent_task_sync_service.py`

**函数**：`sync_task_from_log()`

**关键代码段**（第1873-1875行）：

```python
# combined_factors_df.parquet 属于回测离线特征文件，log-only 同步不再拷贝（避免误用）。
combined_relpath = None
diagnostics["combined_factors"] = {"mode": "skipped", "reason": "backtest_artifact_not_synced"}
```

### 1.2 当前实现的问题

**问题1：跳过 combined_factors_df.parquet 同步**

- **现状**：第1873-1875行明确跳过 `combined_factors_df.parquet` 的同步
- **原因**：注释说明"属于回测离线特征文件，log-only 同步不再拷贝（避免误用）"
- **问题**：虽然跳过了 parquet 文件的同步，但**没有读取 parquet 的列顺序并写入 factor_order.json**
- **影响**：无法获取动态因子顺序，实盘选股时无法保证因子顺序与训练时一致

**问题2：manifest 中 combined_factors_relpath 为 None**

- **现状**：第1908行 `combined_factors_relpath` 设置为 `None`
- **问题**：manifest 中缺少 `factor_order_relpath` 字段
- **影响**：AIstock 侧无法知道是否有 `factor_order.json` 文件可用

**问题3：缺少 factor_order.json 生成逻辑**

- **现状**：第1873-1875行之后没有读取 parquet 列顺序的逻辑
- **问题**：没有生成 `factor_order.json` 文件
- **影响**：无法获取动态因子顺序

## 二、需要修改的内容

### 2.1 添加 factor_order.json 生成逻辑

**位置**：第1873-1875行之后

**需要添加的代码**：

```python
# combined_factors_df.parquet 属于回测离线特征文件，log-only 同步不再拷贝（避免误用）。
combined_relpath = None
diagnostics["combined_factors"] = {"mode": "skipped", "reason": "backtest_artifact_not_synced"}

# 新增：读取 combined_factors_df.parquet 的列顺序，写入 factor_order.json
factor_order_relpath = None
try:
    combined_parquet_path = ws_dir / "combined_factors_df.parquet"
    if combined_parquet_path.exists() and combined_parquet_path.is_file():
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(combined_parquet_path)
            all_cols = meta.schema.names

            # 提取因子顺序（排除索引列）
            factor_order = [name for name in all_cols
                           if name not in ("datetime", "instrument", "index", "level_0", "level_1")]

            # 写入 factor_order.json
            factor_order_payload = {
                "version": "v1",
                "task_run_id": task_run_id,
                "loop_id": loop_id,
                "generated_at_utc": _utc_now_iso(),
                "source_file": "combined_factors_df.parquet",
                "factor_order": factor_order,
                "alpha158_count": sum(1 for name in factor_order if "alpha158" in name.lower()),
                "dynamic_factor_count": sum(1 for name in factor_order if "alpha158" not in name.lower()),
            }
            factor_order_json_path = task_dir / "factor_order.json"
            factor_order_json_path.write_text(json.dumps(factor_order_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            factor_order_relpath = "factor_order.json"
            diagnostics["factor_order"] = {
                "mode": "extracted_from_parquet",
                "source": str(combined_parquet_path),
                "factor_count": len(factor_order),
                "alpha158_count": factor_order_payload["alpha158_count"],
                "dynamic_factor_count": factor_order_payload["dynamic_factor_count"],
            }
        except Exception as e:
            diagnostics["factor_order"] = {"mode": "extract_failed", "error": str(e)}
    else:
        diagnostics["factor_order"] = {"mode": "skipped", "reason": "parquet_not_found"}
except Exception as e:
    diagnostics["factor_order"] = {"mode": "skipped", "error": str(e)}
```

### 2.2 更新 manifest

**位置**：第1889-1911行

**需要修改的代码**：

```python
aistock_manifest: JsonDict = {
    "schema_version": 1,
    "task_id": tid,
    "task_run_id": task_run_id,
    "loop_id": loop_id,
    "generated_at_utc": _utc_now_iso(),
    "source": "aistock_task_sync_log_only",
    "task_only": {
        "mode": "log_only",
        "session": diagnostics.get("session"),
        "sota_factor": diagnostics.get("sota_factor"),
        "workspace": diagnostics.get("workspace"),
        "factor_source": str(factor_src) if factor_src is not None else None,
        "model_weight_source": str(weight_src) if weight_src is not None else None,
    },
    "primary_assets": {
        "factor_entry_relpath": "factor_entry.py",
        "model_weight_relpath": "model.pkl",
        "config_relpath": None,
        "combined_factors_relpath": combined_relpath,
        "model_meta_relpath": model_meta_relpath,
        "factor_order_relpath": factor_order_relpath,  # 新增
    },
}
```

## 三、逻辑错误分析

### 3.1 错误1：跳过 parquet 同步但未提取列顺序

**错误描述**：
- 第1873-1875行跳过了 `combined_factors_df.parquet` 的同步
- 但没有读取 parquet 的列顺序并写入 `factor_order.json`
- 导致无法获取动态因子顺序

**错误原因**：
- 代码注释说"避免误用"，但没有考虑到需要提取列顺序
- 缺少对动态因子顺序的处理逻辑

**影响**：
- 实盘选股时无法保证因子顺序与训练时一致
- 无法满足"实盘选股禁止使用回测数据"的要求

### 3.2 错误2：manifest 缺少 factor_order_relpath 字段

**错误描述**：
- 第1908行 `combined_factors_relpath` 设置为 `None`
- 但没有添加 `factor_order_relpath` 字段
- manifest 中缺少因子顺序信息的引用

**错误原因**：
- 没有考虑到需要记录 `factor_order.json` 的路径
- manifest schema 不完整

**影响**：
- AIstock 侧无法知道是否有 `factor_order.json` 文件可用
- 无法正确读取因子顺序信息

### 3.3 错误3：缺少 factor_order.json 生成逻辑

**错误描述**：
- 第1873-1875行之后没有读取 parquet 列顺序的逻辑
- 没有生成 `factor_order.json` 文件
- 没有更新 diagnostics 信息

**错误原因**：
- 没有考虑到动态因子顺序的重要性
- 缺少对因子顺序的处理逻辑

**影响**：
- 无法获取动态因子顺序
- 实盘选股时无法保证因子顺序与训练时一致

## 四、修改建议

### 4.1 修改优先级

**高优先级**：
1. 添加 `factor_order.json` 生成逻辑（第1873-1875行之后）
2. 更新 manifest，添加 `factor_order_relpath` 字段（第1908行）

**中优先级**：
3. 更新 diagnostics，添加 `factor_order` 字段
4. 添加错误处理逻辑

**低优先级**：
5. 添加单元测试
6. 添加日志记录

### 4.2 修改步骤

**步骤1**：添加 `factor_order.json` 生成逻辑

- 在第1873-1875行之后添加代码
- 读取 `combined_factors_df.parquet` 的列顺序
- 写入 `factor_order.json` 文件

**步骤2**：更新 manifest

- 在第1908行之后添加 `factor_order_relpath` 字段
- 确保 manifest schema 完整

**步骤3**：更新 diagnostics

- 添加 `factor_order` 字段
- 记录因子顺序提取的成功/失败状态

**步骤4**：添加错误处理

- 添加 try-except 块
- 确保即使出错也不会中断同步流程

### 4.3 测试建议

**测试1**：验证 `factor_order.json` 生成

- 运行 `sync_task_from_log()` 函数
- 检查 `factor_order.json` 是否生成
- 验证 `factor_order.json` 的内容是否正确

**测试2**：验证 manifest 更新

- 检查 manifest 中是否包含 `factor_order_relpath` 字段
- 验证 `factor_order_relpath` 的值是否正确

**测试3**：验证 diagnostics 更新

- 检查 diagnostics 中是否包含 `factor_order` 字段
- 验证 `factor_order` 的内容是否正确

**测试4**：验证错误处理

- 模拟 `combined_factors_df.parquet` 不存在的情况
- 模拟读取 parquet 失败的情况
- 确保不会中断同步流程

## 五、总结

### 5.1 当前实现的问题

1. **跳过 parquet 同步但未提取列顺序**
2. **manifest 缺少 factor_order_relpath 字段**
3. **缺少 factor_order.json 生成逻辑**

### 5.2 需要修改的内容

1. 添加 `factor_order.json` 生成逻辑
2. 更新 manifest，添加 `factor_order_relpath` 字段
3. 更新 diagnostics，添加 `factor_order` 字段
4. 添加错误处理逻辑

### 5.3 修改优先级

**高优先级**：
- 添加 `factor_order.json` 生成逻辑
- 更新 manifest，添加 `factor_order_relpath` 字段

**中优先级**：
- 更新 diagnostics
- 添加错误处理逻辑

**低优先级**：
- 添加单元测试
- 添加日志记录

### 5.4 建议

1. **立即实施**：添加 `factor_order.json` 生成逻辑
2. **测试验证**：确保 `factor_order.json` 正确生成
3. **更新文档**：更新《模型权重文件定位方案_v2.md》，明确问题已解决
