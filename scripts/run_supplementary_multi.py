#!/usr/bin/env python3
"""补充多液滴仿真 (Study 10d-10i, 12 cases).

补全多液滴参数空间:
  10d: θ扫描 ridge n=2 (3 cases)
  10e: convex/concave n=2 (2 cases)
  10f: R*效应 ridge n=2 (2 cases)
  10g: n=4 ridge (1 case)
  10h: We=15 基底交叉 n=2 (3 cases)
  10i: n=3 dt*=2.0 ridge (1 case)

Machine: Intel Arc A750 (8GB VRAM) 或其他GPU
预估: ~8-9h

Usage: python3 scripts/run_supplementary_multi.py
"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import time
import json

# 自动检测设备: CUDA (NVIDIA) / XPU (Intel Arc) / CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.empty_cache()
    print(f"GPU (CUDA): {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = 'xpu'
    torch.xpu.empty_cache()
    print(f"GPU (XPU): Intel Arc")
    try:
        print(f"VRAM: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception:
        pass
else:
    DEVICE = 'cpu'
    print("WARNING: No GPU detected, using CPU (very slow)")

D0 = 45.0
R_drop = D0 / 2.0
rho_l = 1.0
rho_g = 1.0 / 828.0
xi = 4.0
tau = 0.53
U0 = -0.05


def get_surface_height(solid, nx, ny, nz):
    height = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1, -1, -1):
                if solid[i, j, k]:
                    height[i, j] = k
                    break
    return height


def run_sequential_case(substrate_type='ridge', R_star=None, We=7.9,
                        theta_eq=162.0, amp=1.8,
                        n_drops=2, dt_star=1.0,
                        N=6000, n_base=150, dtype=torch.float32, label=None,
                        save_snapshots=False):
    """运行单个多液滴顺序沉积案例。"""
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    tau_l, tau_g = tau, tau

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nz = min(int(R_drop + 2 + (1 + n_drops) * R_drop + 20), 300)
        nx, ny = n_base, n_base
    else:
        R_g = abs(R_star) * R_drop
        nx, ny = n_base, n_base
        nz = min(int(R_g + 2 + R_drop + (1 + n_drops) * R_drop + 20), 300)

    dt_inject = int(dt_star * D0 / abs(U0))
    inject_steps = [k * dt_inject for k in range(1, n_drops)]

    from lbm.fe_config import FEConfig
    from lbm.fe_ac_solver import AllenCahnSolver
    from lbm.fe_droplet import create_fe_droplet_with_impact
    from geometry.substrate import create_substrate, create_substrate_with_fraction

    config = FEConfig(
        nx=nx, ny=ny, nz=nz,
        rho_l=rho_l, rho_g=rho_g,
        sigma=sigma, xi=xi, beta=beta, kappa=kappa,
        M=M, tau_l=tau_l, tau_g=tau_g, tau_h=tau_l + 0.04,
        theta_eq=theta_eq, device=DEVICE,
        max_steps=N, output_interval=N + 1,
        g_force=(0.0, 0.0, 0.0),
    )
    solver = AllenCahnSolver(
        config, dtype=dtype, stab_mode='fakhari',
        boundary_relax=0.0, geometric_wetting=(amp > 0),
        geo_amplification=amp,
    )

    if is_flat:
        solid = create_substrate(nx, ny, nz, substrate_type='flat',
                                 R_star=None, R_d=R_drop)
        solver.set_solid(solid)
    else:
        solid, fraction = create_substrate_with_fraction(
            nx, ny, nz, substrate_type=substrate_type,
            R_star=R_star, R_d=R_drop)
        solver.set_solid(solid, solid_fraction=fraction)

    surface_height = get_surface_height(solid, nx, ny, nz)
    cx, cy = nx / 2.0, ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    C_init, _, u_init = create_fe_droplet_with_impact(
        nx, ny, nz, center=(cx, cy, cz), radius=R_drop, xi=xi,
        rho_l=rho_l, rho_g=rho_g, u_impact=(0.0, 0.0, U0))
    solver.init_fields(C_init, u_init)

    # 快照目录
    snap_dir = os.path.join(_project_root, 'results', 'thesis_snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    snap_prefix = label if label else f"seq_{substrate_type}_R{R_star or 'flat'}_n{n_drops}"
    snap_interval = max(500, N // 10)
    snap_steps = set()
    if save_snapshots:
        for s in range(snap_interval, N + 1, snap_interval):
            snap_steps.add(s)

    max_Dx, max_Dy, max_Dx_step = 0, 0, 0
    k_history = []
    stable = True
    inject_idx = 0
    t0 = time.time()

    for step in range(1, N + 1):
        if inject_idx < len(inject_steps) and step == inject_steps[inject_idx]:
            solver.inject_droplet(center=(cx, cy, cz), radius=R_drop, xi=xi,
                                  u_impact=(0.0, 0.0, U0))
            inject_idx += 1
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
            else:
                Dx = Dy = 0
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy,
                              'n_deposited': 1 + inject_idx})
            if Dx > max_Dx:
                max_Dx, max_Dy, max_Dx_step = Dx, Dy, step
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

        if save_snapshots and step in snap_steps:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            mid_j = ny // 2
            xz_slice = phi_np[:, mid_j, :]
            xz_solid = solid_np[:, mid_j, :]
            snap_path = os.path.join(snap_dir, f'{snap_prefix}_step{step}.npz')
            np.savez_compressed(snap_path, xz_slice=xz_slice, xz_solid=xz_solid,
                                step=step, nx=nx, ny=ny, nz=nz)

    elapsed = time.time() - t0
    if DEVICE == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    elif DEVICE == 'xpu':
        try:
            mem_mb = torch.xpu.max_memory_allocated() / 1024 / 1024
            torch.xpu.reset_peak_memory_stats()
        except Exception:
            mem_mb = 0
    else:
        mem_mb = 0

    if label is None:
        label = f"seq_{substrate_type}_R{R_star or 'flat'}_n{n_drops}_dt{dt_star:.1f}"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>45} n={n_drops} dt*={dt_star:.1f} We={We:5.1f} θ={theta_eq:.0f} "
          f"| Dx={max_Dx:.0f} Dy={max_Dy:.0f} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")
    del solver
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'xpu':
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp,
        'n_drops': n_drops, 'dt_star': dt_star,
        'max_Dx': max_Dx, 'max_Dy': max_Dy, 'max_Dx_step': max_Dx_step,
        'stable': stable, 'grid': f"{nx}x{ny}x{nz}",
        'elapsed_s': round(elapsed, 1), 'mem_mb': round(mem_mb, 0),
        'machine': 'a750_supplementary',
        'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # STUDY 10d: 接触角扫描（脊状, n=2, dt*=1.0, We=7.9）
    # 探究润湿性对多液滴沉积的影响
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10d: 接触角扫描 (ridge R*=1.0, n=2, We=7.9)")
    print("=" * 70)
    for theta in [90, 120, 150]:
        results.append(run_sequential_case(
            substrate_type='ridge', R_star=1.0, We=7.9,
            theta_eq=theta, amp=1.8, n_drops=2, dt_star=1.0,
            N=6000, n_base=150,
            label=f"s10d_ridge_R1.0_theta{theta:.0f}",
            save_snapshots=(theta in [90, 150]),
        ))

    # =================================================================
    # STUDY 10e: convex/concave n=2（补全基底类型）
    # 对比曲面基底上的多液滴沉积行为
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10e: convex/concave 多液滴沉积 (n=2, We=7.9, θ=162°)")
    print("=" * 70)
    for sub in ['convex', 'concave']:
        results.append(run_sequential_case(
            substrate_type=sub, R_star=1.0, We=7.9,
            theta_eq=162.0, amp=1.8, n_drops=2, dt_star=1.0,
            N=6000, n_base=150,
            label=f"s10e_{sub}_R1.0_n2",
            save_snapshots=True,
        ))

    # =================================================================
    # STUDY 10f: R* 效应（脊状, n=2, dt*=1.0, We=7.9）
    # 曲率对多液滴沉积不对称性的影响
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10f: R* 效应 (ridge, n=2, We=7.9, θ=162°)")
    print("=" * 70)
    for rstar in [0.7, 1.4]:
        results.append(run_sequential_case(
            substrate_type='ridge', R_star=rstar, We=7.9,
            theta_eq=162.0, amp=1.8, n_drops=2, dt_star=1.0,
            N=6000, n_base=150,
            label=f"s10f_ridge_R{rstar:.1f}_n2",
            save_snapshots=(rstar == 1.4),
        ))

    # =================================================================
    # STUDY 10g: 液滴数 n=4（脊状, 扩展沉积序列）
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10g: n=4 扩展沉积 (ridge R*=1.0, We=7.9, θ=162°)")
    print("=" * 70)
    results.append(run_sequential_case(
        substrate_type='ridge', R_star=1.0, We=7.9,
        theta_eq=162.0, amp=1.8, n_drops=4, dt_star=1.0,
        N=12000, n_base=150,
        label=f"s10g_ridge_R1.0_n4",
        save_snapshots=True,
    ))

    # =================================================================
    # STUDY 10h: 高We基底交叉（We=15, n=2）
    # 对比不同基底在高冲击速度下的沉积行为
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10h: 高We基底交叉 (We=15, n=2, θ=162°)")
    print("=" * 70)
    for sub, R, amp in [('flat', None, 0.0), ('convex', 1.0, 1.8), ('concave', 1.0, 1.8)]:
        results.append(run_sequential_case(
            substrate_type=sub, R_star=R, We=15.0,
            theta_eq=162.0, amp=amp, n_drops=2, dt_star=1.0,
            N=6000, n_base=150,
            label=f"s10h_{sub}_R{R or 'flat'}_We15",
            save_snapshots=False,
        ))

    # =================================================================
    # STUDY 10i: n=3 dt*=2.0（大间隔三液滴沉积）
    # 对比不同沉积间隔的累积效应
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 10i: n=3 dt*=2.0 (ridge R*=1.0, We=7.9, θ=162°)")
    print("=" * 70)
    results.append(run_sequential_case(
        substrate_type='ridge', R_star=1.0, We=7.9,
        theta_eq=162.0, amp=1.8, n_drops=3, dt_star=2.0,
        N=9000, n_base=150,
        label=f"s10i_ridge_R1.0_n3_dt2.0",
        save_snapshots=True,
    ))

    # =================================================================
    # 汇总
    # =================================================================
    print("\n" + "=" * 70)
    print("  补充多液滴仿真结果汇总")
    print("=" * 70)
    ok = sum(1 for r in results if r['stable'])
    print(f"\n  总计: {len(results)} cases, {ok} OK, {len(results)-ok} FAIL")

    for study in ['s10d', 's10e', 's10f', 's10g', 's10h', 's10i']:
        study_cases = [r for r in results if r['label'].startswith(study)]
        if study_cases:
            print(f"\n  --- {study} ---")
            for r in study_cases:
                tag = "OK" if r['stable'] else "FAIL"
                n = r.get('n_drops', '-')
                dt = r.get('dt_star', '-')
                theta = r.get('theta_eq', 162.0)
                print(f"    {r['label']:>45} n={n} dt*={dt} θ={theta:3.0f} We={r['We']:5.1f} "
                      f"Dx={r['max_Dx']:.0f} Dy={r['max_Dy']:.0f} [{tag}]")

    # 保存结果
    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)

    main_path = os.path.join(save_dir, 'studies_9_10.json')
    if os.path.exists(main_path):
        with open(main_path) as f:
            main_data = json.load(f)
    else:
        main_data = []

    existing_labels = {r['label'] for r in main_data}
    new_count = 0
    for r in results:
        if r['label'] not in existing_labels:
            main_data.append({k: v for k, v in r.items() if k != 'k_history'})
            existing_labels.add(r['label'])
            new_count += 1

    with open(main_path, 'w') as f:
        json.dump(main_data, f, indent=2, default=str)
    print(f"\n  追加 {new_count} 条新结果到 {main_path}")

    hist_path = os.path.join(save_dir, 'supplementary_multi_history.json')
    hist_data = {r['label']: r['k_history'] for r in results if r['k_history']}
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            existing_hist = json.load(f)
        existing_hist.update(hist_data)
        hist_data = existing_hist
    with open(hist_path, 'w') as f:
        json.dump(hist_data, f)
    print(f"  时间序列保存到 {hist_path} ({len(hist_data)} 条)")

    elapsed_total = sum(r['elapsed_s'] for r in results)
    print(f"\n  总耗时: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")


if __name__ == "__main__":
    main()
