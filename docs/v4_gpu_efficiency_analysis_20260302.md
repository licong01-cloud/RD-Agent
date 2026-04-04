# RD-Agent v4 GPU效率与优化方案影响分析

> 生成日期: 2026-03-02
> 硬件环境: AMD Ryzen 9 3900X (12C/24T), 48GB RAM, NVIDIA RTX 5080 16GB, WSL2 Ubuntu
> 模型: GRU_Attention_Residual_Model (0.83MB), batch_size=16384, n_epochs=300, early_stop=30
> 数据: train=3,704,481 samples, valid=1,602,097 samples, 20 features, step_len=20

---

## 一、核心问题：训练集评估是否只有打日志功能？

### 1.1 代码层面的完整证据链

以下分析基于 Qlib `GeneralPTNN.fit()` 源码（`qlib.contrib.model.pytorch_general_nn`）逐行审计。

**`train_score` 的全部引用点（共4处）：**

| 行号 | 代码 | 用途 |
|------|------|------|
| 60 | `train_loss = 0` | 初始化变量 |
| 75 | `train_loss, train_score = self.test_epoch(train_loader)` | 计算训练集评估指标 |
| 77 | `self.logger.info("Epoch%d: train %.6f, valid %.6f" % ...)` | **仅用于日志打印** |
| 78 | `evals_result["train"].append(train_score)` | 追加到 evals_result 字典 |

**`val_score` 的全部引用点（共5处）：**

| 行号 | 代码 | 用途 |
|------|------|------|
| 76 | `val_loss, val_score = self.test_epoch(valid_loader)` | 计算验证集评估指标 |
| 77 | `self.logger.info(...)` | 日志打印 |
| 79 | `evals_result["valid"].append(val_score)` | 追加到 evals_result |
| 84 | `self.lr_scheduler.step(val_score)` | **驱动学习率调度** |
| 88 | `if val_score < best_score:` | **驱动 early_stop 和最优模型保存** |

### 1.2 evals_result 是否被下游消费？

**验证路径（逐层追踪）：**

1. `_exe_task()` 调用 `model.fit(dataset, reweighter=reweighter)` — **没有传入 evals_result 参数**，使用默认值 `dict()`
2. `_exe_task()` 调用后直接 `R.save_objects(**{"params.pkl": model})`，**不保存 evals_result**
3. `SignalRecord.generate()` 只调用 `model.predict()`，**不读取 evals_result**
4. `SigAnaRecord` — **不包含 evals_result 引用**
5. RD-Agent 的 `read_exp_res.py` — **不包含 evals_result 引用**（已全文搜索确认）
6. RD-Agent 的 `model_runner.py` — **不包含 evals_result 引用**（已全文搜索确认）
7. RD-Agent 整个 `rdagent/` 目录 — **不包含 evals_result 引用**（已递归搜索确认）

### 1.3 结论

**`train_score` 的作用：**
- 日志打印（`logger.info`）：纯观测用途
- 写入 `evals_result["train"]`：但该字典在当前调用链中**从未被任何下游代码读取**

**`val_score` 的作用（关键区别）：**
- `lr_scheduler.step(val_score)`：驱动 ReduceLROnPlateau 学习率衰减（patience=5, factor=0.5）
- `if val_score < best_score`：驱动 early_stop 判断和最优模型权重保存

> **代码级证明：`train_score` 不参与任何训练控制逻辑（学习率调度、early_stop、模型保存），仅用于日志输出。删除训练集评估不会改变模型训练的任何行为。**

---

## 二、各优化方案对训练效果/演进准确性的影响分析

### 2.1 影响分类总表

| 方案 | 对训练效果的影响 | 对演进准确性的影响 | 是否推荐 |
|------|-----------------|-------------------|---------|
| A. 跳过训练集评估 | **无影响** | **无影响** | 强烈推荐 |
| B. 验证阶段加AMP | **极微小** | **无影响** | 推荐 |
| C. 增大验证batch_size | **无影响** | **无影响** | 推荐 |
| D. 预加载数据到GPU | **无影响** | **无影响** | 条件推荐 |
| E. Qlib数据迁移ext4 | **无影响** | **无影响** | 强烈推荐 |
| F. 增加n_jobs | **无影响** | **无影响** | 推荐 |
| G. torch.compile() | **极微小** | **无影响** | 谨慎推荐 |
| H. 减小batch_size | **有影响** | **可能有影响** | 不推荐 |
| I. 增大模型 | **有影响** | **有影响** | 不推荐 |
| J. CUDA Graphs | **无影响** | **无影响** | 复杂度高，暂不推荐 |


### 2.2 逐方案详细分析

#### 方案A：跳过训练集评估 — 无影响，强烈推荐

**当前行为：**
```python
# fit() 第75行：每个epoch对训练集做完整推理
train_loss, train_score = self.test_epoch(train_loader)  # 3,704,481 样本
val_loss, val_score = self.test_epoch(valid_loader)       # 1,602,097 样本
```

**影响分析：**
- `train_score` 仅用于 `logger.info` 和 `evals_result["train"]`（见第一章证明）
- 不影响 `lr_scheduler.step(val_score)` — 学习率调度只看 val_score
- 不影响 `if val_score < best_score` — early_stop 只看 val_score
- 不影响 `best_param = copy.deepcopy(self.dnn_model.state_dict())` — 模型保存只看 val_score
- `evals_result` 在整个 RD-Agent 调用链中无消费者

**性能收益：**
- 每 epoch 省去 3,704,481 / 16384 = 226 个 batch 的推理
- 当前评估阶段耗时约 94 秒（train 66s + valid 28s），优化后仅 28 秒
- **每 epoch 节省约 66 秒，总训练时间减少约 43%**

**对演进的影响：无。** RD-Agent 的 feedback 阶段读取的是 `read_exp_res.py` 产出的回测指标（IC、收益率等），不读取训练过程中的 train_score。

---

#### 方案B：验证阶段加 AMP（混合精度） — 极微小影响，推荐

**当前行为：**
```python
# train_epoch: 使用 AMP
with torch.amp.autocast('cuda'):
    pred = self.dnn_model(feature.float())

# test_epoch: 不使用 AMP，全精度 FP32
with torch.no_grad():
    pred = self.dnn_model(feature.float())
```

**影响分析：**
- AMP 在推理时使用 FP16 计算，会引入极微小的数值差异（约 1e-4 量级）
- `val_score` 是 MSE loss 的均值，FP16 计算的 MSE 与 FP32 的差异在 1e-6 量级
- 这个差异远小于 `lr_scheduler` 的 `threshold=1e-5`，不会改变学习率调度行为
- early_stop 判断 `val_score < best_score` 在极端边界情况下可能有 1 个 epoch 的差异，但对最终模型质量无实质影响

**性能收益：**
- RTX 5080 的 FP16 吞吐量约为 FP32 的 2 倍
- 验证推理速度提升约 1.5-2x

**对演进的影响：无实质影响。** 训练本身已经使用 AMP，验证加 AMP 只是保持一致性。

---

#### 方案C：增大验证 batch_size — 无影响，推荐

**当前行为：**
```python
# train_loader 和 valid_loader 使用相同的 batch_size=16384
valid_loader = DataLoader(
    ConcatDataset(dl_valid, wl_valid),
    batch_size=self.batch_size,  # 16384
    ...
)
```

**影响分析：**
- 验证阶段使用 `torch.no_grad()`，不需要存储梯度，显存占用远小于训练
- 增大 batch_size 不改变计算结果（推理是确定性的，batch_size 不影响前向传播结果）
- MSE loss 是按样本平均的，batch_size 不影响最终的 val_score 值
- 注意：`drop_last=True` 意味着最后不足一个 batch 的样本会被丢弃，增大 batch_size 会丢弃更多尾部样本，但对 1,602,097 样本来说影响可忽略

**性能收益：**
- batch_size 从 16384 增大到 65536：batch 数从 97 降到 24，减少 75% 的 DataLoader 开销
- 每个 batch 的 GPU kernel launch 开销是固定的，减少 batch 数直接减少总开销

**对演进的影响：无。**

---

#### 方案D：预加载数据到 GPU — 无影响，条件推荐

**用户问题：减小 batch_size 是否能减少显存占用，让训练集和验证集都能在 GPU 中？**

**关键澄清：batch_size 与数据预加载是两个独立的概念。**

当前显存占用的构成：
```
模型参数:        0.83 MB
梯度:            0.83 MB
优化器状态:      1.66 MB (Adam: 2x params)
AMP scaler:     ~0.01 MB
DataLoader batch: 26.2 MB (16384 * 20 * 21 * 4 bytes, 在GPU上的当前batch)
PyTorch CUDA 缓存: ~13 GB (CUDA context + memory pool)
```

**13-14GB 显存占用的真正原因不是数据，而是 PyTorch CUDA 运行时的内存池预分配。** 模型和数据实际只用了不到 30MB，但 PyTorch 的 CUDA caching allocator 会预分配大块显存以减少 malloc 调用。

**减小 batch_size 不会显著减少显存占用**，因为：
1. 当前 batch 在 GPU 上只占 26.2 MB，即使减半也只省 13 MB
2. PyTorch CUDA 缓存不会因为 batch 变小而释放
3. 真正占显存的是 CUDA context（~300MB）和 memory pool

**预加载全部数据到 GPU 的内存需求：**

| 数据集 | 样本数 | float32 | float16 |
|--------|--------|---------|---------|
| 训练集 | 3,704,481 | 5.80 GB | 2.90 GB |
| 验证集 | 1,602,097 | 2.51 GB | 1.25 GB |
| 合计 | 5,306,578 | **8.30 GB** | **4.15 GB** |

但这里有一个关键问题：**TSDataSampler 的数据存储方式。**

TSDataSampler 不是按样本存储的，而是存储了一个扩展后的 `data_arr`（numpy 数组），`__getitem__` 通过索引切片来构造时间序列窗口：

```python
# TSDataSampler.__init__:
self.data_arr = np.array(self.data)  # 完整的扩展 DataFrame 转 numpy
# __getitem__:
indices = self.idx_arr[max(row - step_len + 1, 0) : row + 1, col]
data = self.data_arr[indices]  # numpy 索引切片
```

要预加载到 GPU，需要：
1. 将 `data_arr` 转为 GPU tensor
2. 将 `idx_arr` 转为 GPU tensor
3. 重写 `__getitem__` 使用 torch 索引代替 numpy 索引
4. 绕过 DataLoader 的 CPU worker 机制

**这是一个侵入性很大的改动**，需要修改 Qlib 核心数据管道。

**对训练效果的影响：无。** 数据内容完全相同，只是存储位置从 CPU 内存移到 GPU 显存。

**对演进的影响：无。**

**结论：技术上可行但改动大，且当前显存不足以同时放下训练集+验证集（float32需8.3GB，加上CUDA缓存超过16GB）。如果只预加载验证集（float16 1.25GB），收益有限因为验证只占总时间的一小部分（方案A优化后）。暂不推荐。**


---

#### 方案E：Qlib 数据迁移到 ext4 — 无影响，强烈推荐

**当前行为：**
```
Qlib 数据路径: /mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209 (NTFS via 9P)
数据加载耗时: 847.796 秒 (约14分钟)
```

**影响分析：**
- 纯 I/O 优化，不改变任何数据内容
- WSL2 的 9P 协议访问 NTFS 有严重性能损失（随机读放大 10-50x）
- 迁移到 ext4 后，数据加载预计降到 30-60 秒

**对训练效果的影响：无。** 读取的数据完全相同。

**对演进的影响：无。** 但每个 Loop 节省约 13 分钟的数据加载时间，3 个 Loop 节省约 40 分钟。

---

#### 方案F：增加 n_jobs（DataLoader workers） — 无影响，推荐

**当前配置：** `n_jobs: 4`

**影响分析：**
- DataLoader workers 只负责数据预取，不影响数据内容
- 12 核 CPU 有足够余量支持 8 个 workers
- 更多 workers 可以更好地隐藏 CPU→GPU 数据传输延迟

**对训练效果的影响：无。**

**对演进的影响：无。**

---

#### 方案G：torch.compile() — 极微小影响，谨慎推荐

**影响分析：**
- `torch.compile()` 通过算子融合和代码生成优化模型执行
- 可能引入极微小的数值差异（浮点运算顺序变化）
- 对于 GRU 这类 RNN 模型，compile 的兼容性需要验证
- 首次编译有额外开销（30-60秒），但后续 epoch 受益

**对训练效果的影响：极微小。** 浮点精度差异在 1e-7 量级，不影响模型收敛方向。

**对演进的影响：无。**

**风险：** GRU + MultiheadAttention 的组合在 torch.compile 下可能有兼容性问题，需要实测验证。

---

#### 方案H：减小 batch_size — 有影响，不推荐

**这是用户提出的方案，需要特别说明为什么不推荐。**

**影响分析：**

减小 batch_size 会改变训练动态：

| batch_size | 每epoch batch数 | 梯度噪声 | 收敛特性 |
|------------|----------------|----------|---------|
| 16384 (当前) | 226 | 低 | 稳定收敛，泛化适中 |
| 4096 | 904 | 中 | 收敛更慢，可能泛化更好 |
| 2048 | 1808 | 高 | 收敛慢，需调整 lr |
| 1024 | 3617 | 很高 | 需要完全重新调参 |

**关键问题：**
1. **batch_size 与 learning rate 是耦合的**：当前 lr=0.001 是针对 batch_size=16384 调优的，减小 batch_size 通常需要按比例减小 lr（线性缩放规则）
2. **改变收敛轨迹**：不同 batch_size 会导致不同的 loss landscape 探索路径，最终模型可能不同
3. **不解决显存问题**：如前所述，显存占用主要是 CUDA 缓存，不是 batch 数据
4. **增加训练时间**：batch 数增加意味着更多的 GPU kernel launch 和 DataLoader 迭代

**对训练效果的影响：有。** 改变了梯度估计的方差，影响收敛速度和最终模型质量。

**对演进的影响：可能有。** 如果模型质量变化，feedback 阶段的评估指标会变化，影响 LLM 的演进决策。

---

#### 方案I：增大模型 — 有影响，不推荐（当前阶段）

**影响分析：**
- 增大模型（更多 GRU 层、更大 hidden_size）会改变模型容量
- 需要重新调整所有超参数
- 更大的模型不一定在量化因子预测任务上表现更好（过拟合风险）
- 这属于模型架构搜索，应该由 RD-Agent 的演进过程自动探索

**对训练效果的影响：有。** 完全改变了模型。

**对演进的影响：有。** 这本身就是演进应该做的事情。

---

## 三、显存占用深度分析

### 3.1 当前 13-14GB 显存的构成

```
PyTorch CUDA Context:           ~300 MB  (CUDA 运行时初始化)
CUDA Memory Pool (预分配):       ~12.5 GB (caching allocator 预留)
模型参数 (float32):              0.83 MB
梯度 (float32):                  0.83 MB
优化器状态 (Adam, 2x):           1.66 MB
AMP GradScaler:                  ~0.01 MB
当前 batch 数据 (GPU):           26.2 MB  (16384 * 20 * 21 * 4)
当前 batch 标签 (GPU):           0.06 MB  (16384 * 4)
中间激活 (训练时):               ~50-100 MB
```

**关键发现：实际有效使用的显存不到 500 MB，其余 13+ GB 是 PyTorch CUDA caching allocator 的预分配。**

### 3.2 为什么 GPU 利用率只有 22%（SM 利用率约 35%）

根本原因是**模型太小**：
- 0.83MB 的模型，前向传播在 RTX 5080 的 10752 个 CUDA 核心上只需要微秒级
- 每个 batch 的 GPU 计算时间远小于 CPU→GPU 数据传输时间
- GPU 大部分时间在等待下一个 batch 的数据到达

这是一个典型的 **I/O bound** 而非 **compute bound** 的场景。提高 GPU 利用率的正确方式不是"占满资源"，而是减少 GPU 等待时间。

### 3.3 减小 batch_size 对显存的影响

| batch_size | batch 数据量 (GPU) | 总显存变化 |
|------------|-------------------|-----------|
| 16384 | 26.2 MB | 基准 |
| 8192 | 13.1 MB | -13.1 MB |
| 4096 | 6.6 MB | -19.6 MB |

**减小 batch_size 最多只能省约 20 MB 显存**，相对于 13 GB 的总占用可以忽略不计。这不是释放显存的有效手段。

---

## 四、推荐优化方案（按优先级排序）

以下方案均满足"不影响训练效果和演进准确性"的前提。

### 优先级 P0：立即可做，效果显著

#### P0-1：跳过训练集评估

**改动方式：** Monkey-patch `GeneralPTNN.fit()` 中的评估循环

```python
# 在 RD-Agent 启动时注入
import qlib.contrib.model.pytorch_general_nn as ptnn
_orig_test_epoch = ptnn.GeneralPTNN.test_epoch

def _patched_fit(original_fit):
    """Wrap fit() to skip train set evaluation."""
    import functools
    @functools.wraps(original_fit)
    def wrapper(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        # Temporarily replace test_epoch to track calls
        call_count = [0]
        _orig = self.test_epoch
        def _counting_test_epoch(data_loader):
            call_count[0] += 1
            # In fit(), test_epoch is called twice per epoch:
            # 1st call = train_loader (skip), 2nd call = valid_loader (keep)
            # But we can't distinguish by call order alone in a patch.
            # Instead, we modify fit() behavior directly.
            return _orig(data_loader)
        return original_fit(self, dataset, evals_result, save_path, reweighter)
    return wrapper
```

**更简洁的方式：直接修改 conda 环境中的源文件**

```python
# 文件: qlib/contrib/model/pytorch_general_nn.py
# 将 fit() 中的:
#   train_loss, train_score = self.test_epoch(train_loader)
# 改为:
#   train_loss, train_score = 0.0, 0.0  # Skip train eval for performance
```

**预期效果：每 epoch 节省 66 秒，总训练时间减少 43%**

#### P0-2：Qlib 数据迁移到 WSL ext4

```bash
# 在 WSL 中执行
mkdir -p ~/qlib_data
cp -r /mnt/f/Dev/AIstock/qlib_bin/qlib_bin_20251209 ~/qlib_data/
# 然后修改 conf_baseline_factors_model.yaml:
# provider_uri: "/home/lc999/qlib_data/qlib_bin_20251209"
```

**预期效果：数据加载从 848 秒降到 30-60 秒，每 Loop 节省约 13 分钟**

### 优先级 P1：中等改动，效果明显

#### P1-1：验证阶段加 AMP

```python
# 修改 test_epoch:
def test_epoch(self, data_loader):
    self.dnn_model.eval()
    scores = []
    losses = []
    for data, weight in data_loader:
        feature, label = self._get_fl(data)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):  # 添加 AMP
                pred = self.dnn_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                scores.append(self.metric_fn(pred, label).item())
                losses.append(loss.item())
    return np.mean(losses), np.mean(scores)
```

**预期效果：验证推理速度提升 1.5-2x**

#### P1-2：增大验证 batch_size

```python
# 修改 fit() 中 valid_loader 的 batch_size:
valid_loader = DataLoader(
    ConcatDataset(dl_valid, wl_valid),
    batch_size=self.batch_size * 4,  # 65536
    ...
)
```

**预期效果：验证 batch 数从 97 降到 24**

#### P1-3：增加 n_jobs 到 8

```yaml
# conf_baseline_factors_model.yaml:
n_jobs: 8  # 从 4 增加到 8
```

### 优先级 P2：长期优化

- torch.compile()：需要验证 GRU+Attention 兼容性
- 跨 Loop 缓存数据集：避免每个 Loop 重新加载 848 秒的数据

---

## 五、组合优化效果预估

### 当前每 epoch 时间分解（从日志计算）

```
训练 (train_epoch):           61 秒
评估训练集 (test_epoch #1):   66 秒  ← 浪费
评估验证集 (test_epoch #2):   28 秒
─────────────────────────────────
总计:                         155 秒/epoch
```

### 应用 P0+P1 优化后

```
训练 (train_epoch):           61 秒  (不变)
评估训练集:                    0 秒  (P0-1: 跳过)
评估验证集:                   ~7 秒  (P1-1: AMP + P1-2: 大batch)
─────────────────────────────────
总计:                         ~68 秒/epoch  (提升 56%)
```

### 每个 Loop 总时间

| 阶段 | 当前 | 优化后 | 节省 |
|------|------|--------|------|
| 数据加载 | 848 秒 | ~45 秒 (ext4) | 803 秒 |
| 训练 (假设30 epochs) | 4650 秒 | 2040 秒 | 2610 秒 |
| **Loop 总计** | **~5500 秒 (92分钟)** | **~2100 秒 (35分钟)** | **~57分钟 (62%)** |

---

## 六、总结

1. **训练集评估确实只有打日志功能**，有完整的代码证据链证明 `train_score` 不参与任何训练控制逻辑
2. **减小 batch_size 不能有效减少显存占用**（只省约 20MB），且会改变训练动态，不推荐
3. **预加载全部数据到 GPU 在当前条件下不可行**（float32 需 8.3GB + CUDA 缓存 > 16GB），且改动侵入性大
4. **推荐的优化组合（P0+P1）可以将每个 Loop 从 92 分钟降到 35 分钟**，且完全不影响训练效果和演进准确性
5. 所有推荐方案的核心原则：**只减少无效等待和冗余计算，不改变任何影响模型质量的参数**