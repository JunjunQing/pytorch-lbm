#!/usr/bin/env python3
"""GPU benchmark: single flat case at 150^3, measure steps/s."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
from lbm.mc_config import MCConfig
from lbm.mc_solver import MCSSolver
from lbm.equilibrium import equilibrium as feq_func
from geometry.droplet import create_mc_droplet

NX = NY = NZ = 150
R_DROP = 30
U = 0.20
G_ADS = 3.46
G_12 = 4.0
SIGMA = 0.47
N_WARMUP = 200
N_BENCH = 1000

print(f"GPU Benchmark: {NX}x{NY}x{NZ}, R_drop={R_DROP}")
print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Setup solver
config = MCConfig(
    nx=NX, ny=NY, nz=NZ, dx=1.0, dt=1.0,
    rho1_l=1.0, rho1_g=0.02, tau1=1.0,
    rho2_l=0.02, rho2_g=1.0, tau2=1.0,
    G_12=G_12, G_21=G_12, G_11=0.0, G_22=0.0,
    G_ads_1=G_ADS, G_ads_2=0.0, s_wall=1.0,
    g_force=(0, 0, -1e-6), collision='bgk', forcing='velocity_shift',
    device='cuda',
)
solver = MCSSolver(config)

# Flat substrate
solid = np.zeros((NX, NY, NZ), dtype=bool)
solid[:, :, 0] = True; solid[:, :, NZ-1] = True
solid[0, :, :] = True; solid[NX-1, :, :] = True
solid[:, 0, :] = True; solid[:, NY-1, :] = True
solver.set_solid(solid)

# Droplet
z_center = 1 + 3 + R_DROP
rho1_init, rho2_init = create_mc_droplet(NX, NY, nz=NZ,
    center=(NX/2.0, NY/2.0, z_center), radius=R_DROP,
    rho1_l=1.0, rho1_g=0.02, rho2_l=0.02, rho2_g=1.0, dx=1.0, width=3.0)
solver.init_equilibrium(rho1_init, rho2_init, np.zeros((3, NX, NY, NZ), dtype=np.float32))

# Warmup
print(f"\n  Warmup: {N_WARMUP} steps...")
for _ in range(N_WARMUP):
    solver.step()
torch.cuda.synchronize()

# Benchmark
print(f"  Benchmark: {N_BENCH} steps...")
t0 = time.time()
for _ in range(N_BENCH):
    solver.step()
torch.cuda.synchronize()
t1 = time.time()

elapsed = t1 - t0
steps_per_sec = N_BENCH / elapsed
nodes = NX * NY * NZ
throughput = steps_per_sec * nodes / 1e6  # Mnode-steps/s

print(f"\n  RESULT:")
print(f"    Grid: {NX}x{NY}x{NZ} = {nodes:,} nodes")
print(f"    Steps: {N_BENCH}")
print(f"    Time: {elapsed:.2f}s")
print(f"    Speed: {steps_per_sec:.1f} steps/s")
print(f"    Throughput: {throughput:.1f} Mnode-steps/s")

# Memory
mem_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"    GPU memory: {mem_mb:.0f} MB")

del solver
torch.cuda.empty_cache()
