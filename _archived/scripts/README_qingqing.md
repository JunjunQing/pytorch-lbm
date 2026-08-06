# qingqing 部署与运行指南

## 一、硬件环境

| 项目 | 规格 |
|------|------|
| CPU | Intel Core i5/i7 或更高 |
| GPU | Intel Arc A750 8GB GDDR6 |
| RAM | 建议 ≥16GB |
| 系统 | Linux (Ubuntu 22.04/24.04 推荐) |
| 驱动 | Intel GPU driver + oneAPI runtime |

## 二、软件安装

### 2.1 系统依赖

```bash
# Ubuntu
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

# 安装 Intel GPU 驱动和 oneAPI runtime
# 参考: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install -y intel-oneapi-runtime-libs intel-oneapi-runtime-opencl intel-oneapi-runtime-level-zero
```

### 2.2 验证 Intel GPU 可见

```bash
# 检查 GPU 设备
sudo apt install -y intel-opencl-icd
clinfo | grep "Device Name"
# 应该看到: Intel(R) Arc(TM) A750

# 或使用 oneAPI 工具
sudo apt install -y intel-oneapi-runtime-sycl
sycl-ls
# 应该看到包含 "ext_oneapi_level_zero" 或 "opencl" 的设备
```

### 2.3 创建虚拟环境

```bash
cd ~/simulation-projects
python3 -m venv venv
source venv/bin/activate
```

### 2.4 安装 PyTorch (XPU)

**关键**：必须安装 XPU 版本的 PyTorch，标准版不支持 Intel GPU。

```bash
# XPU 专用 PyTorch（必须使用 --index-url 指定 XPU 源）
pip install torch --index-url https://download.pytorch.org/whl/xpu

# 验证版本号
pip show torch | grep Version
# 应显示类似: Version: 2.x.x+ipx
```

> **注意**：不要使用标准 `pip install torch`，那个只有 CUDA 版本。
> 也不能用 `pip install torch-directml`（那是 AMD/通用 GPU 方案，已弃用）。

### 2.5 验证 XPU 可用

```bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('XPU available:', torch.xpu.is_available())
if torch.xpu.is_available():
    print('GPU name:', torch.xpu.get_device_name(0))
    props = torch.xpu.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1024**3:.1f} GB')
    # 快速计算测试
    x = torch.randn(1000, 1000, device='xpu')
    y = x @ x
    print('XPU compute test: OK')
else:
    print('ERROR: XPU not available!')
    print('Troubleshooting:')
    print('  1. Check driver: sycl-ls')
    print('  2. Reinstall: pip install torch --index-url https://download.pytorch.org/whl/xpu')
    print('  3. Check oneAPI: dpkg -l | grep oneapi')
"
```

### 2.6 获取代码

```bash
# 从 Gitee 克隆（推荐，国内速度快）
git clone https://gitee.com/qmingjun/simulation-projects.git
cd simulation-projects/pytorch_lbm

# 或从本机拷贝
# scp -r user@local_ip:/mnt/simulation-projects/pytorch_lbm ~/simulation-projects/pytorch_lbm
```

## 三、运行前验证

在正式运行仿真之前，先用小网格验证求解器工作正常：

```bash
cd ~/simulation-projects/pytorch_lbm
python3 -c "
import torch
import numpy as np
from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_fe_droplet_with_impact
from geometry.substrate import create_substrate

# 小网格快速测试 (约30秒)
config = FEConfig(
    nx=60, ny=60, nz=40,
    rho_l=1.0, rho_g=1.0/828.0,
    sigma=0.0127, xi=4.0, beta=0.0381, kappa=0.305,
    M=0.000524, tau_l=0.53, tau_g=0.53, tau_h=0.57,
    theta_eq=162.0, device='xpu',
    max_steps=100, output_interval=101,
    g_force=(0.0, 0.0, 0.0),
)
solver = AllenCahnSolver(config, dtype=torch.float32,
                         stab_mode='fakhari', boundary_relax=0.0,
                         geometric_wetting=True, geo_amplification=1.8)

solid = create_substrate(60, 60, 40, 'flat', R_star=None, R_d=15.0)
solver.set_solid(solid)

C, _, u = create_fe_droplet_with_impact(
    60, 60, 40, center=(30,30,25), radius=15.0, xi=4.0,
    rho_l=1.0, rho_g=1.0/828.0, u_impact=(0,0,-0.05))
solver.init_fields(C, u)

for i in range(100):
    solver.step()

phi = solver.phi.cpu().numpy()
print(f'phi range: [{phi.min():.3f}, {phi.max():.3f}]')
print(f'phi sum: {phi.sum():.0f}')
assert not np.isnan(phi).any(), 'NaN detected!'
print('VALIDATION PASSED: solver works correctly on Arc A750 XPU')
"
```

## 四、执行仿真任务

qingqing 需要运行 **4 组任务，共 47 个 cases**：

| 顺序 | 脚本 | Study | Cases | 预计时间 |
|------|------|-------|-------|----------|
| 1 | `run_thesis_qingqing.sh` | 4a/4b/6 | 36 | ~6.6h |
| 2 | `run_thesis_multi_droplet.sh` | 10 | 11 | ~2.8h |
| **合计** | | | **47** | **~9.4h** |

### 4.1 运行方式

```bash
cd ~/simulation-projects/pytorch_lbm

# 创建结果目录
mkdir -p results

# 方式 A：逐个运行（推荐，便于观察）
bash scripts/run_thesis_qingqing.sh
bash scripts/run_thesis_multi_droplet.sh

# 方式 B：串联一键运行
bash scripts/run_thesis_qingqing.sh && bash scripts/run_thesis_multi_droplet.sh

# 方式 C：后台运行（断开SSH不中断）
nohup bash -c "bash scripts/run_thesis_qingqing.sh && bash scripts/run_thesis_multi_droplet.sh" \
    > results/qingqing_full.log 2>&1 &
# 查看进度
tail -f results/qingqing_full.log
```

### 4.2 使用 screen/tmux 长时间运行

```bash
# screen 方式
screen -S gpu_sim
cd ~/simulation-projects/pytorch_lbm
source venv/bin/activate
bash scripts/run_thesis_qingqing.sh && bash scripts/run_thesis_multi_droplet.sh
# Ctrl+A D 分离会话
# screen -r gpu_sim 重新连接

# tmux 方式
tmux new -s gpu_sim
cd ~/simulation-projects/pytorch_lbm
source venv/bin/activate
bash scripts/run_thesis_qingqing.sh && bash scripts/run_thesis_multi_droplet.sh
# Ctrl+B D 分离会话
# tmux attach -t gpu_sim 重新连接
```

## 五、结果收集

### 5.1 输出文件

仿真完成后，`results/` 目录下应有以下文件：

```
results/
├── qingqing_run.log                    # Study 4a/4b/6 运行日志
├── thesis_qingqing_results.json        # Study 4a/4b/6 数值结果
├── thesis_qingqing_history.json        # Study 4a/4b/6 时间序列
├── multi_droplet_run.log               # Study 10 运行日志
├── thesis_multi_droplet_results.json   # Study 10 数值结果
└── thesis_multi_droplet_history.json   # Study 10 时间序列
```

### 5.2 传回本机

```bash
# 在本机执行
scp -r user@qingqing_ip:~/simulation-projects/pytorch_lbm/results/thesis_*qingqing* \
    /mnt/simulation-projects/pytorch_lbm/results/
scp -r user@qingqing_ip:~/simulation-projects/pytorch_lbm/results/thesis_*multi* \
    /mnt/simulation-projects/pytorch_lbm/results/
```

### 5.3 结果验证

```bash
python3 -c "
import json
for f in ['thesis_qingqing_results.json', 'thesis_multi_droplet_results.json']:
    data = json.load(open(f'results/{f}'))
    ok = sum(1 for r in data if r['stable'])
    fail = len(data) - ok
    print(f'{f}: {ok} OK, {fail} FAIL, total {len(data)} cases')
"
```

## 六、VRAM 使用估算

| 网格规模 | VRAM 占用 | 说明 |
|----------|-----------|------|
| 150×150×82 (flat) | ~3.5 GB | Study 4a flat |
| 150×150×120 (ridge R*=1.0) | ~5.0 GB | 大部分 cases |
| 150×150×200 (多液滴) | ~5.8 GB | Study 10c (5 drops) |

> Arc A750 8GB 可以运行所有 cases。如果 OOM，将 `n_base` 从 150 降到 120。

## 七、常见问题

### Q1: `torch.xpu.is_available()` 返回 False

这是最常见的问题，按以下顺序排查：

```bash
# 1. 检查 PyTorch 是否为 XPU 版本
python3 -c "import torch; print(torch.__version__)"
# 版本号应包含 "xpu" 或 "ipx" 后缀
# 如果没有，重新安装:
pip install torch --index-url https://download.pytorch.org/whl/xpu

# 2. 检查 Intel 驱动
sycl-ls
# 应该看到 Arc A750 设备

# 3. 检查 Level Zero 运行时
ls /usr/lib/x86_64-linux-gnu/intel-opencl/
# 应该有 libze_loader.so 等文件

# 4. 重新安装驱动
sudo apt install --reinstall intel-opencl-icd intel-level-zero-gpu
```

### Q2: `RuntimeError: XPU out of memory`
```
RuntimeError: XPU out of memory.
```
**解决**：减小网格分辨率。在脚本中将 `n_base=150` 改为 `n_base=120`。

### Q3: 运行速度比 NVIDIA 慢很多
- Arc A750 的 PyTorch XPU 后端相比 CUDA 后端确实有一定性能差距
- 预计每个 case 约 15-20 分钟（vs NVIDIA 的 ~11 分钟）
- 这是正常的，不影响结果精度

### Q4: `pip install` 速度慢
```bash
# 使用清华镜像
pip install torch --index-url https://download.pytorch.org/whl/xpu \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q5: 某个 case 出现 NaN (FAIL)
- 这是正常的，某些极端参数组合可能导致数值不稳定
- FAIL 的 case 不需要重跑
- 如果 FAIL 率超过 20%，请联系检查配置

## 八、Study 10 特殊说明（多液滴连续沉积）

Study 10 使用 `solver.inject_droplet()` 在运行中注入新液滴。这是本机的特有功能。

**注入机制**：
1. 第一个液滴通过 `init_fields()` 正常初始化
2. 后续液滴在指定时间步通过 `inject_droplet()` 注入
3. 注入时自动更新 phi、f、g 分布，重新平衡质量守恒

**注意事项**：
- 注入液滴时终端会打印 `[inject]` 信息，确认注入成功
- 注入后可能需要 ~100 步达到稳态，前 100 步的 k 值可能波动
- 多液滴模拟的总步数更长（N=6000-15000），耐心等待

## 九、与 bigblack 的分工对比

| 项目 | bigblack (GTX 1080) | qingqing (Arc A750) |
|------|--------------------|--------------------|
| GPU后端 | CUDA | XPU (oneAPI) |
| Study 3a/3b/5 | ✓ We扫描+R*扫描 | — |
| Study 9 | ✓ 双液滴合并 | — |
| Study 4a/4b | — | ✓ 接触角扫描 |
| Study 6 | — | ✓ We-θ相图 |
| Study 10 | — | ✓ 多液滴连续沉积 |
| 总 cases | 53 | 47 |
| 预计时间 | ~10.5h | ~9.4h |
