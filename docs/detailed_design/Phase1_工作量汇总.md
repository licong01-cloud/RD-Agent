# Phase 1 工作量与风险汇总

## Phase 1 总览

**预计总工作量**: 30人日（约6周，单人）  
**优先级**: P0（必须完成）  
**核心目标**: 建立多模型支持、因子优化和API基础

## 详细工作量分解

### 1.1 ML模型适配器（9.5人日）

| 子任务 | 文件 | 工作量 | 风险 |
|--------|------|--------|------|
| base.py基类 | adapters/base.py | 1人日 | 低 |
| ml_adapter.py | adapters/ml_adapter.py | 3人日 | 中 |
| pytorch_adapter.py重构 | adapters/pytorch_adapter.py | 2人日 | 低 |
| registry.py注册表 | registry.py | 0.5人日 | 低 |
| 提示词修改 | prompts.yaml | 1人日 | 低 |
| ModelCoder集成 | model_coder/model.py | 1人日 | 低 |
| ModelRunner集成 | developer/model_runner.py | 1人日 | 低 |

### 1.2 因子库管理（11.5人日）

| 子任务 | 文件 | 工作量 | 风险 |
|--------|------|--------|------|
| models.py数据模型 | factor_library/models.py | 0.5人日 | 低 |
| database.py数据库 | factor_library/database.py | 2人日 | 低 |
| factor_lib.py接口 | factor_library/factor_lib.py | 1.5人日 | 低 |
| classifier.py分类器 | factor_library/classifier.py | 1人日 | 低 |
| optimizer.py优化器 | factor_library/optimizer.py | 3人日 | 中 |
| Alpha因子导入 | importers/alpha158.py等 | 2人日 | 低 |
| FactorRunner集成 | developer/factor_runner.py | 1人日 | 低 |
| Summarizer集成 | feedback/summarizer.py | 0.5人日 | 低 |

### 1.3 API扩展（9人日）

| 子任务 | 文件 | 工作量 | 风险 |
|--------|------|--------|------|
| 扩展server.py | app/scheduler/server.py | 1人日 | 低 |
| Pydantic模型 | app/api/models.py | 1.5人日 | 低 |
| 因子管理路由 | app/api/routes/factors.py | 2人日 | 低 |
| 模型管理路由 | app/api/routes/models.py | 1.5人日 | 低 |
| 实验管理路由 | app/api/routes/experiments.py | 1.5人日 | 低 |
| 任务进度查询 | app/scheduler/api_stub.py | 1人日 | 低 |
| API文档生成 | - | 0.5人日 | 低 |

### 测试与文档（总计：8人日）

| 子任务 | 工作量 | 风险 |
|--------|--------|------|
| ML适配器单元测试 | 2人日 | 低 |
| ML适配器集成测试 | 1人日 | 低 |
| 因子库单元测试 | 1.5人日 | 低 |
| 因子库集成测试 | 1人日 | 低 |
| API测试 | 2人日 | 低 |
| 文档编写 | 0.5人日 | 低 |

## 总工作量汇总表

| 模块 | 开发工作量 | 测试工作量 | 小计 |
|------|-----------|-----------|------|
| ML模型适配器 | 9.5人日 | 3人日 | 12.5人日 |
| 因子库管理 | 11.5人日 | 2.5人日 | 14人日 |
| API扩展 | 9人日 | 2.5人日 | 11.5人日 |
| **总计** | **30人日** | **8人日** | **38人日** |

**调整后总计**（考虑10%缓冲）：**42人日 ≈ 8.5周（单人）或 4周（双人）**

## 风险评估矩阵

### 高风险项（需重点关注）

| 风险项 | 影响 | 概率 | 缓解措施 | 责任人 |
|--------|------|------|---------|--------|
| 依赖库版本冲突 | 中 | 中 | 使用conda环境隔离，锁定版本 | 开发者 |
| 因子组合优化效果不佳 | 中 | 中 | 先用简单贪心，Phase 2优化 | 开发者 |

### 中风险项

| 风险项 | 影响 | 概率 | 缓解措施 |
|--------|------|------|---------|
| API接口设计不合理 | 中 | 低 | 与AIStock侧对接确认需求 |
| 因子分类不准确 | 低 | 中 | 启发式规则+人工校验 |
| 破坏现有PyTorch逻辑 | 高 | 低 | 充分测试，保持向后兼容 |

### 低风险项（常规处理）

- SQLite性能问题
- 提示词修改错误
- 测试覆盖不全

## 依赖关系与开发顺序

```
开发顺序（可并行的标记*）：

Week 1-2: 基础设施
  ├─ [1.1] ML模型适配器开发 *
  └─ [1.2] 因子库数据模型+数据库 *

Week 3-4: 核心功能
  ├─ [1.1] ModelCoder/Runner集成
  ├─ [1.2] 因子库分类器+优化器 *
  └─ [1.3] API基础框架 *

Week 5-6: 集成与测试
  ├─ [1.2] Alpha因子导入
  ├─ [1.3] API路由开发 *
  └─ 单元测试（并行进行）

Week 7: 集成测试与文档
  ├─ 端到端测试
  ├─ 回归测试
  └─ 文档编写

Week 8-8.5: 缓冲与优化
  └─ Bug修复，性能优化
```

## 团队配置建议

### 理想配置（2人，4周完成）

**开发者A（后端专家）**:
- ML模型适配器开发
- 因子库开发
- 集成到演进循环

**开发者B（API专家）**:
- API扩展开发
- Pydantic模型设计
- API测试

### 最小配置（1人，8.5周完成）

按上述开发顺序依次完成。

## 里程碑与验收标准

### Milestone 1: 基础设施完成（Week 2结束）

**验收标准**:
- ✅ MLAdapter可创建XGBoost/LightGBM模型
- ✅ FactorDatabase可执行CRUD操作
- ✅ 单元测试通过率 > 90%

### Milestone 2: 核心功能完成（Week 4结束）

**验收标准**:
- ✅ 演进循环可使用XGBoost进行模型演进
- ✅ 因子演进结果自动入库
- ✅ API基础端点可正常访问

### Milestone 3: 集成完成（Week 6结束）

**验收标准**:
- ✅ Alpha158/360因子成功导入
- ✅ 因子组合优化API正常工作
- ✅ 所有API端点测试通过

### Milestone 4: Phase 1完成（Week 8结束）

**验收标准**:
- ✅ 端到端测试通过
- ✅ 现有PyTorch模型演进不受影响
- ✅ API文档完整
- ✅ 代码review完成

## 质量保证

### 代码质量要求

- 单元测试覆盖率 > 80%
- 集成测试通过率 100%
- 代码review必须通过
- 遵循PEP8代码规范

### 文档要求

- 每个新增模块有README
- API自动生成Swagger文档
- 关键函数有docstring
- 变更记录（CHANGELOG）

## 后续Phase依赖

Phase 1完成后，以下Phase可并行开始：

- **Phase 2.1** (多模型协同选股): 依赖因子库和模型注册表
- **Phase 2.2** (HMM大盘分析): 依赖模型适配器
- **Phase 3** (策略演进): 依赖API扩展

## 成本估算

假设人日成本为X：

| 项目 | 工作量 | 成本 |
|------|--------|------|
| 开发工作量 | 30人日 | 30X |
| 测试工作量 | 8人日 | 8X |
| 缓冲（10%） | 4人日 | 4X |
| **总计** | **42人日** | **42X** |

## Phase 1完成标志

✅ 所有验收标准达成  
✅ 无P0/P1级别bug  
✅ 文档完整  
✅ 代码review通过  
✅ 与AIStock侧对接测试通过
