# PR Review Checklist（RD-Agent × AIstock）

## 1. 设计文档 & REQ 对齐

- [ ] PR 描述中列出了关联的设计文档（顶层 / Phase1 / Phase2 / Phase3 / DataServiceLayer）。
- [ ] PR 描述中列出了本次改动涉及的 REQ ID（如 REQ-FACTOR-P2-001）。
- [ ] 代码实现与对应 REQ / 设计章节内容一一对应，没有“隐性设计”。
- [ ] 若有设计偏离 / 放宽 / 新增：
  - [ ] PR 中明确说明了偏离点；
  - [ ] 有对应设计文档更新（已合并或作为同一批 PR 提交）。

## 2. 不得精简原则

- [ ] 未删除 / 合并 / 隐藏任何设计文档中要求必须存在的字段：
  - Catalog JSON 中字段仍然完整（model / factor / strategy / loop）。
  - Registry 表（SQLite）中的关键字段仍存在（task_runs / loops / workspaces / artifacts / artifact_files）。
  - Results API / DataService 等对外接口仍包含所有约定字段，未做“字段精简”。
- [ ] 未以“PoC / 最小可用版本 / 临时路径”为理由将简化实现引入生产链路：
  - PoC 代码是否被限制在本地脚本或实验性配置，不在主执行栈 / 正式 API 中出现？
  - 执行栈消费数据时是否一律通过 DataService，而不是直接读取 HDF5 / 临时 DB？

## 3. 契约 & CI 覆盖

- [ ] 如修改了 JSON / Catalog / Registry / Results API / DataService：
  - [ ] 对应 contract tests（结构/字段契约）是否已更新？
  - [ ] `docs/req_test_map.yaml` 是否已同步更新，且相关 REQ 仍然有对应测试用例？
  - [ ] CI 中的 contract tests 是否通过？
- [ ] 新增的功能 / REQ 是否补足了测试覆盖，而不是只写了代码没有测试？

## 4. 业务效果与验收（Phase2 / Phase3）

- **Phase 2 相关改动：**
  - [ ] 若涉及成果导出 / registry / catalog / AIstock 导入视图：
    - [ ] Phase 2 文档 8.2 节中的验证点是否仍可通过（任务→loop→workspace→artifacts→AIstock UI 闭环）？
    - [ ] 小规模回测样例在 CI 中是否仍能完整打通 Phase 2 资产链路？

- **Phase 3 相关改动：**
  - [ ] 若涉及执行迁移 / Strategy Preview / 归档 / LiveFeedback：
    - [ ] 执行迁移对齐测试是否仍然通过（收益曲线相关性 & 指标偏差在预设阈值内）？
    - [ ] Strategy Preview 与归档相关 acceptance tests 是否已更新并通过？

## 5. 兼容性 & 回滚

- [ ] 是否考虑了已有数据 / 配置的兼容性？
  - 例如：老版本 JSON / SQLite 记录是否仍然可被新代码正确读取或迁移？
- [ ] 若为不兼容变更：
  - [ ] 是否有清晰的数据迁移脚本？
  - [ ] 是否有可行的回滚方案（如何在出问题时快速恢复）？
