# LightGBM GPU/CUDA 备忘录

## 背景
当前 RD-Agent/Qlib 配置中包含 `LGBModel`（底层为 LightGBM）。用户目标是尽量使用 GPU；若 LightGBM GPU 无法启用，则允许 LightGBM 继续使用 CPU，其他模型（例如 PyTorch 模型）按原逻辑执行。

## 已验证结论（本次会话）
### 1. OpenCL 路线（device_type: gpu）在当前 WSL2 环境不可用
- 已创建 `/etc/OpenCL/vendors/nvidia.icd`，其内容为 `libnvidia-opencl.so.1`
- 系统已存在 OpenCL loader 与 NVIDIA OpenCL 用户态库：
  - `libOpenCL.so*`
  - `libnvidia-opencl.so.1`（可见具体版本例如 `libnvidia-opencl.so.580.95.05`）
- 但 `clinfo` 结果为：
  - `Number of platforms 0`
- 结论：即使库存在，WSL2 环境下 OpenCL 平台未暴露/不可用，因此 LightGBM 的 OpenCL GPU（`device_type: gpu`）无法正常工作。

### 2. CUDA 路线（device_type: cuda）在当前 LightGBM build 未启用
使用最小训练脚本验证：
- `device_type: cuda` 报错：
  - `CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1`
- 结论：当前安装的 `lightgbm` 为 CPU build，不包含 CUDA backend。

### 3. conda-forge 存在 CUDA build，可作为后续 A1 尝试方向
- `conda search -c conda-forge "lightgbm*"` 显示存在：
  - `lightgbm 4.6.0 cuda_py_*`
  - `lightgbm 4.6.0 cuda129_py_*`
  - `lightgbm 4.6.0 cuda130_py_*`
- 后续可通过 `conda install --dry-run` 评估是否会破坏现有 `rdagent-gpu` 环境（尤其是 PyTorch/CUDA 版本）。

## 当前策略（暂时）
### LightGBM 强制 CPU
为了避免每次 qrun 都先因 LightGBM GPU/OpenCL/CUDA backend 报错再回退，本次将程序行为调整为：
- 在运行 `qrun` 之前，主动从 workspace 的 qlib config 中移除 LightGBM 的 GPU/CUDA 相关参数（如 `device_type: gpu/cuda`、`gpu_use_dp`、`max_bin` 等），使 LightGBM 直接以 CPU 模式运行。

### 兜底策略保持
- 若未来配置仍包含 GPU/CUDA 参数，程序仍保留“检测到 LightGBM GPU/CUDA/OpenCL 失败后自动移除参数并重跑”的兜底逻辑。

## 下次继续（A1 优先）
### A1: 通过 conda-forge 安装 LightGBM CUDA build
1. 先 dry-run（不改环境）：
   - `conda install -n rdagent-gpu -c conda-forge --dry-run "lightgbm=4.6.0=cuda_py_*"`
   - 如无解，再尝试 `cuda129_py_*` / `cuda130_py_*`
2. dry-run 通过后再正式安装。
3. 安装完成后复测脚本：
   - `device_type: cuda` 小训练应输出 OK。

### A2: 源码编译 LightGBM CUDA backend
若 A1 因依赖冲突/不可用失败，再考虑 A2（需要 nvcc、cmake、gcc 等工具链，并在 WSL 内编译安装）。
