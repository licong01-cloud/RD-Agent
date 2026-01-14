<!--- Thank you for submitting a Pull Request! In order to make our work smoother. -->## 1. 变更类型

- [ ] Feature
- [ ] Bugfix
- [ ] Refactor
- [ ] Docs
- [ ] Infra/CI

## 2. 关联设计文档 & REQ ID（必填）

- 相关设计文档（可多选）：
  - [ ] 顶层：`docs/2025-12-26_RD-Agent_AIstock_TopLevel_Architecture_Design_v2.md`
  - [ ] Phase 1：`docs/2025-12-23_Phase1_Detail_Design_RD-Agent_AIstock.md`
  - [ ] Phase 2：`docs/2025-12-29_Phase2_Detail_Design_RD-Agent_AIstock_Final.md`
  - [ ] Phase 3：`docs/2025-12-29_Phase3_Detail_Design_RD-Agent_AIstock_Final.md`
  - [ ] DataServiceLayer：`docs/2025-12-24_DataServiceLayer_Detail_Design_RD-Agent_AIstock.md`

- 本 PR 主要涉及的 REQ ID（请逐条列出，如 REQ-FACTOR-P2-001 等）：
  - REQ-XXXX-XXX：
  - REQ-YYYY-YYY：

## 3. 是否偏离现有设计？（不得精简原则）

- [ ] 不偏离，仅按照现有 REQ 实现。
- [ ] 存在偏离/放宽/新增设计（**必须说明**）：

  - 偏离点说明（涉及哪些章节/REQ，原设计是什么，现在怎么改）：

  - 对应设计文档更新链接（MR/PR/commit）：

## 4. 契约与数据结构影响

是否影响以下契约？（如勾选任一项，下面两行必须为 Yes）

- [ ] JSON / Catalog 结构（factor/model/strategy/loop_catalog, `*_meta.json` 等）
- [ ] Registry / SQLite 结构或字段
- [ ] Results API 响应结构
- [ ] DataService 接口或返回结构

如上有勾选：

- [ ] 已同步更新对应设计文档中的 REQ（含 REQ ID）。
- [ ] 已同步更新 `docs/req_test_map.yaml`，并新增/调整对应 contract/acceptance tests。

## 5. 测试执行情况（填具体命令/结果）

- [ ] 单元测试（命令 & 结果）：
- [ ] Contract Tests（命令 & 结果，例如 `pytest tests/contract`）：
- [ ] Phase 2 验收测试（如适用，命令 & 结果）：
- [ ] Phase 3 验收测试（如适用，命令 & 结果）：

## 6. 风险 & 回滚

- 主要风险点：
- 回滚策略（如合入后发现问题，如何快速回退）：

<!--- please make sure your Pull Request meets the following requirements: -->
<!---   1. Provide a general summary of your changes in the Title above; -->
<!---   2. Add appropriate prefixes to titles, such as `build:`, `chore:`, `ci:`, `docs:`, `feat:`, `fix:`, `perf:`, `refactor:`, `revert:`, `style:`, `test:`(Ref: https://www.conventionalcommits.org/). -->
<!--- Category: -->
<!--- Patch Updates: `fix:` -->
<!---   Example: fix(auth): correct login validation issue -->
<!--- minor update (introduces new functionality): `feat` -->
<!---   Example: feature(parser): add ability to parse arrays -->
<!--- major update(destructive update): Include BREAKING CHANGE in the commit message footer, or add `! ` in the commit footer to indicate that there is a destructive update. -->
<!---   Example: feat(auth)! : remove support for old authentication method -->
<!--- Other updates: `build:`, `chore:`, `ci:`, `docs:`, `perf:`, `refactor:`, `revert:`, `style:`, `test:`. -->

## Description
<!--- Describe your changes in detail -->

## Motivation and Context
<!--- Are there any related issues? If so, please put the link here. -->
<!--- Why is this change required? What problem does it solve? -->

## How Has This Been Tested?
<!---  Put an `x` in all the boxes that apply: --->
- [ ] If you are adding a new feature, test on your own test scripts.

<!--- **ATTENTION**: If you are adding a new feature, please make sure your codes are **correctly tested**. If our test scripts do not cover your cases, please provide your own test scripts under the `tests` folder and test them. More information about test scripts can be found [here](https://docs.python.org/3/library/unittest.html#basic-example), or you could refer to those we provide under the `tests` folder. -->

## Screenshots of Test Results (if appropriate):
1. Your own tests:

## Types of changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Fix bugs
- [ ] Add new feature
- [ ] Update documentation
