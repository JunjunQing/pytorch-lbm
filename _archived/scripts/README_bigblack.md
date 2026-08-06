# bigblack 部署与运行指南

## 一、硬件环境

| 项目 | 规格 |
|------|------|
| CPU | AMD Ryzen 9 5900x (12核24线程) |
| GPU | NVIDIA GTX 1080 8GB GDDR5X |
| RAM | 建议 ≥16GB |
| 系统 | Linux (Ubuntu 20.04/22.04 推荐) |
| CUDA Driver | ≥ 450.80 (需支持 GTX 1080) |

## 二、软件安装

### 2.1 系统依赖

```bash
# Ubuntu
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

### 2.2 创建虚拟环境

```bash
cd ~/simulation-projects
python3 -m venv venv
source venv/bin/activate
```

### 2.3 安装 PyTorch (CUDA)

GTX 1080 支持 CUDA Compute Capability 6.1，所有现代 PyTorch 版本兼容。

```bash
# 方案 A：标准安装（自动匹配最新CUDA版本）
pip install torch numpy

# 方案 B：指定 CUDA 11.8（如果方案A不兼容）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 方案 C：指定 CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 验证 GPU 可用

```bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1024**3:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
    # 快速计算测试
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print('GPU compute test: OK')
else:
    print('ERROR: CUDA not available!')
    print('Check: nvidia-smi')
"
```

### 2.5 获取代码

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
    theta_eq=162.0, device='cuda',
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
print('VALIDATION PASSED: solver works correctly on GTX 1080')
"
```

## 四、执行仿真任务

bigblack 需要运行 **4 组任务，共 53 个 cases**：

| 顺序 | 脚本 | Study | Cases | 预计时间 |
|------|------|-------|-------|----------|
| 1 | `run_thesis_bigblack.sh` | 3a/3b/5 | 41 | ~7.5h |
| 2 | `run_thesis_dual_droplet.sh` | 9 | 12 | ~3h |
| **合计** | | | **53** | **~10.5h** |

### 4.1 运行方式

```bash
cd ~/simulation-projects/pytorch_lbm

# 创建结果目录
mkdir -p results

# 方式 A：逐个运行（推荐，便于观察）
bash scripts/run_thesis_bigblack.sh
bash scripts/run_thesis_dual_droplet.sh

# 方式 B：串联一键运行
bash scripts/run_thesis_bigblack.sh && bash scripts/run_thesis_dual_droplet.sh

# 方式 C：后台运行（断开SSH不中断）
nohup bash -c "bash scripts/run_thesis_bigblack.sh && bash scripts/run_thesis_dual_droplet.sh" \
    > results/bigblack_full.log 2>&1 &
# 查看进度
tail -f results/bigblack_full.log
```

### 4.2 使用 screen/tmux 长时间运行

```bash
# screen 方式
screen -S gpu_sim
cd ~/simulation-projects/pytorch_lbm
source venv/bin/activate
bash scripts/run_thesis_bigblack.sh && bash scripts/run_thesis_dual_droplet.sh
# Ctrl+A D 分离会话
# screen -r gpu_sim 重新连接

# tmux 方式
tmux new -s gpu_sim
cd ~/simulation-projects/pytorch_lbm
source venv/bin/activate
bash scripts/run_thesis_bigblack.sh && bash scripts/run_thesis_dual_droplet.sh
# Ctrl+B D 分离会话
# tmux attach -t gpu_sim 重新连接
```

## 五、结果收集

### 5.1 输出文件

仿真完成后，`results/` 目录下应有以下文件：

```
results/
├── bigblack_run.log                    # Study 3a/3b/5 运行日志
├── thesis_bigblack_results.json        # Study 3a/3b/5 数值结果
├── thesis_bigblack_history.json        # Study 3a/3b/5 时间序列
├── dual_droplet_run.log                # Study 9 运行日志
├── thesis_dual_droplet_results.json    # Study 9 数值结果
└── thesis_dual_droplet_history.json    # Study 9 时间序列
```

### 5.2 传回本机

```bash
# 在本机执行
scp -r user@bigblack_ip:~/simulation-projects/pytorch_lbm/results/thesis_*bigblack* \
    /mnt/simulation-projects/pytorch_lbm/results/
scp -r user@bigblack_ip:~/simulation-projects/pytorch_lbm/results/thesis_*dual* \
    /mnt/simulation-projects/pytorch_lbm/results/
```

### 5.3 结果验证

```bash
python3 -c "
import json
for f in ['thesis_bigblack_results.json', 'thesis_dual_droplet_results.json']:
    data = json.load(open(f'results/{f}'))
    ok = sum(1 for r in data if r['stable'])
    fail = len(data) - ok
    print(f'{f}: {ok} OK, {fail} FAIL, total {len(data)} cases')
"
```

## 六、VRAM 使用估算

| 网格规模 | VRAM 占用 | 说明 |
|----------|-----------|------|
| 150×150×82 (flat) | ~3.5 GB | Study 3a flat |
| 150×150×120 (ridge R*=1.0) | ~5.0 GB | 大部分 cases |
| 150×150×200 (ridge R*=2.0) | ~5.8 GB | Study 5 大 R* |
| 250×150×120 (dual droplet) | ~7.5 GB | Study 9 双液滴 |

> GTX 1080 8GB 可以运行所有 cases。如果 OOM，将 `n_base` 从 150 降到 120。

## 七、常见问题

### Q1: `CUDA out of memory`
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```
**解决**：减小网格分辨率。在脚本中将 `n_base=150` 改为 `n_base=120`。

### Q2: `torch.cuda.is_available()` 返回 False
```bash
# 检查驱动
nvidia-smi
# 如果没有输出，重新安装 NVIDIA 驱动
sudo apt install nvidia-driver-535  # 或对应版本
```

### Q3: 仿真速度异常慢
```bash
# 确认 GPU 正在工作
nvidia-smi -l 1  # 每秒刷新，看 GPU 利用率
# 如果利用率低，检查是否有其他进程占用 GPU
sudo fuser -v /dev/nvidia*
```

### Q4: 某个 case 出现 NaN (FAIL)
- 这是正常的，某些极端参数组合（如高We+小R*）可能导致数值不稳定
- FAIL 的 case 不需要重跑，在论文中标注即可
- 如果 FAIL 率超过 20%，请联系检查配置
