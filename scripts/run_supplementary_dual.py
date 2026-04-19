#!/usr/bin/env python3
"""补充双液滴仿真 (Study 9d-9h, 22 cases).

补全双液滴参数空间:
  9d: θ扫描 (ridge, 4 cases)
  9e: convex/concave 间距扫描 (4 cases)
  9f: ridge 扩展间距 (5 cases)
  9g: R* 效应 (3 cases)
  9h: 高We基底交叉 (4 cases)
  9i: 长时动力学追踪 (2 cases, N=6000)

Machine: GTX 1080 (8GB VRAM)
预估: ~22 cases × ~18min ≈ 6.6h

Usage: python3 scripts/run_supplementary_dual.py
"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
import time
import json

from lbm.fe_config import FEConfig
from lbm.fe_ac_solver import AllenCahnSolver
from lbm.fe_droplet import create_multi_droplet_with_impact
from geometry.substrate import create_substrate, create_substrate_with_fraction

DEVICE = 'cuda'
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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


def run_dual_case(substrate_type='ridge', R_star=None, We=7.9,
                  theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
                  N=4000, n_base=150, dtype=torch.float32, label=None,
                  save_snapshots=False):
    """运行单个双液滴案例，可选快照保存。"""
    sigma = rho_l * U0 ** 2 * D0 / We
    beta = 12.0 * sigma / xi
    kappa = beta * xi ** 2 / 8.0
    M = 0.02 / beta
    tau_l, tau_g = tau, tau

    is_flat = substrate_type == 'flat' or R_star is None
    if is_flat:
        nz_min = int(R_drop + 2 + 2 * R_drop + 15)
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny, nz = n_base, nz_min
    else:
        R_g = abs(R_star) * R_drop
        nx = int(spacing_ratio * D0 + D0 + 20)
        ny = n_base
        nz = min(int(R_g + 2 + R_drop + 2 * R_drop + 15), 300)

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
    cx_total = nx / 2.0
    cy = ny / 2.0
    if substrate_type == 'concave':
        ridge_top = int(surface_height.max())
    else:
        ridge_top = surface_height[nx // 2, ny // 2]
    gap = 2.0
    cz = ridge_top + gap + R_drop
    max_cz = nz - R_drop - 2
    if cz > max_cz:
        cz = max_cz

    d_center = spacing_ratio * D0
    cx1 = max(R_drop + 1, cx_total - d_center / 2.0)
    cx2 = min(nx - R_drop - 1, cx_total + d_center / 2.0)
    centers = [(cx1, cy, cz), (cx2, cy, cz)]
    radii = [R_drop, R_drop]
    u_impacts = [(0.0, 0.0, U0), (0.0, 0.0, U0)]

    C_init, _, u_init = create_multi_droplet_with_impact(
        nx, ny, nz, centers=centers, radii=radii,
        xi=xi, rho_l=rho_l, rho_g=rho_g, u_impacts=u_impacts,
    )
    solver.init_fields(C_init, u_init)

    # 快照目录
    snap_dir = os.path.join(_project_root, 'results', 'thesis_snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    snap_prefix = label if label else f"dual_{substrate_type}_R{R_star or 'flat'}_d{spacing_ratio:.1f}"
    snap_steps = set()
    if save_snapshots:
        # 每隔约500步保存一次，加上关键时刻
        for s in range(500, N + 1, 500):
            snap_steps.add(s)

    max_k, max_k_step, Dx_peak, Dy_peak = 0, 0, 0, 0
    k_history = []
    coalescence_step = None
    stable = True
    t0 = time.time()

    for step in range(1, N + 1):
        solver.step()
        if step % 50 == 0:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            interface = (phi_np > 0.5) & ~solid_np
            if interface.any():
                coords = np.argwhere(interface)
                Dx = float(coords[:, 0].max() - coords[:, 0].min() + 1)
                Dy = float(coords[:, 1].max() - coords[:, 1].min() + 1)
                k_val = Dx / Dy if Dy > 0 else 0
                x_range = coords[:, 0].max() - coords[:, 0].min()
                merged = x_range < (d_center + R_drop * 0.5)
            else:
                Dx = Dy = k_val = 0
                merged = False
            if coalescence_step is None and merged:
                coalescence_step = step
            k_history.append({'step': step, 'Dx': Dx, 'Dy': Dy, 'k': k_val, 'merged': merged})
            if k_val > max_k:
                max_k, max_k_step, Dx_peak, Dy_peak = k_val, step, Dx, Dy
            if np.isnan(phi_np).any() or phi_np.max() < 0.01:
                stable = False
                break

        # 保存快照
        if save_snapshots and step in snap_steps:
            phi_np = solver.phi.detach().cpu().numpy()
            solid_np = solver.solid.cpu().numpy()
            mid_j = ny // 2
            xz_slice = phi_np[:, mid_j, :]
            xz_solid = solid_np[:, mid_j, :]
            snap_path = os.path.join(snap_dir, f'{snap_prefix}_step{step}.npz')
            np.savez_compressed(snap_path, xz_slice=xz_slice, xz_solid=xz_solid,
                                step=step, nx=nx, ny=ny, nz=nz)

    # 保存最终状态快照（对长时动力学案例特别重要）
    if save_snapshots:
        phi_np = solver.phi.detach().cpu().numpy()
        solid_np = solver.solid.cpu().numpy()
        mid_j = ny // 2
        xz_slice = phi_np[:, mid_j, :]
        xz_solid = solid_np[:, mid_j, :]
        snap_path = os.path.join(snap_dir, f'{snap_prefix}_step{N}.npz')
        if N not in snap_steps:
            np.savez_compressed(snap_path, xz_slice=xz_slice, xz_solid=xz_solid,
                                step=step, nx=nx, ny=ny, nz=nz)
        # 保存最大k时刻的快照
        if max_k_step not in snap_steps and max_k_step > 0:
            print(f"    WARNING: max_k at step {max_k_step}, not captured in snapshots")

    elapsed = time.time() - t0
    mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    torch.cuda.reset_peak_memory_stats()
    if label is None:
        label = f"dual_{substrate_type}_R{R_star or 'flat'}_d{spacing_ratio:.1f}"
    coal_str = f"merge@{coalescence_step}" if coalescence_step else "separate"
    tag = "OK" if stable else "FAIL"
    print(f"  {label:>45} We={We:5.1f} d/D={spacing_ratio:.1f} θ={theta_eq:.0f} "
          f"| k={max_k:.4f} {coal_str:15s} [{tag}] [{elapsed:.0f}s, {mem_mb:.0f}MB] {nx}x{ny}x{nz}")
    del solver
    torch.cuda.empty_cache()
    return {
        'label': label, 'substrate_type': substrate_type, 'R_star': R_star,
        'We': We, 'theta_eq': theta_eq, 'amp': amp, 'spacing_ratio': spacing_ratio,
        'max_k': max_k, 'max_k_step': max_k_step, 'Dx': Dx_peak, 'Dy': Dy_peak,
        'coalescence_step': coalescence_step, 'stable': stable,
        'grid': f"{nx}x{ny}x{nz}", 'elapsed_s': round(elapsed, 1),
        'mem_mb': round(mem_mb, 0), 'machine': 'gtx1080_supplementary',
        'k_history': k_history,
    }


def main():
    results = []

    # =================================================================
    # STUDY 9d: 接触角扫描（脊状, d/D=1.2, We=7.9）
    # 探究不同润湿性对双液滴聚结行为的影响
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9d: 接触角扫描 (ridge R*=1.0, d/D=1.2, We=7.9)")
    print("=" * 70)
    for theta in [90, 120, 140, 150]:
        results.append(run_dual_case(
            substrate_type='ridge', R_star=1.0, We=7.9,
            theta_eq=theta, amp=1.8, spacing_ratio=1.2,
            N=4000, n_base=150,
            label=f"s9d_ridge_R1.0_theta{theta:.0f}",
            save_snapshots=(theta in [90, 120, 150]),
        ))

    # =================================================================
    # STUDY 9e: convex/concave 间距扫描（补全 s9b 的基底缺口）
    # 对比曲面基底的间距效应
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9e: convex/concave 间距扫描 (We=7.9, θ=162°)")
    print("=" * 70)
    for sub in ['convex', 'concave']:
        for d in [1.0, 1.5]:
            results.append(run_dual_case(
                substrate_type=sub, R_star=1.0, We=7.9,
                theta_eq=162.0, amp=1.8, spacing_ratio=d,
                N=4000, n_base=150,
                label=f"s9e_{sub}_R1.0_d{d:.1f}",
                save_snapshots=(d == 1.5),
            ))

    # =================================================================
    # STUDY 9f: ridge 扩展间距（补全 <1.0 近接触区和 >2.0 远场区）
    # 完整刻画间距-不对称因子关系曲线
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9f: ridge 扩展间距扫描 (We=7.9, θ=162°)")
    print("=" * 70)
    for d in [0.8, 0.9, 1.8, 2.5, 3.0]:
        results.append(run_dual_case(
            substrate_type='ridge', R_star=1.0, We=7.9,
            theta_eq=162.0, amp=1.8, spacing_ratio=d,
            N=4000, n_base=150,
            label=f"s9f_ridge_R1.0_d{d:.1f}",
            save_snapshots=(d in [0.8, 3.0]),
        ))

    # =================================================================
    # STUDY 9g: R* 效应（脊状, d/D=1.2, We=7.9）
    # 曲率对双液滴不对称性的影响
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9g: R* 效应 (ridge, d/D=1.2, We=7.9, θ=162°)")
    print("=" * 70)
    for rstar in [0.7, 1.4, 2.0]:
        results.append(run_dual_case(
            substrate_type='ridge', R_star=rstar, We=7.9,
            theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
            N=4000, n_base=150,
            label=f"s9g_ridge_R{rstar:.1f}_d1.2",
            save_snapshots=(rstar in [0.7, 2.0]),
        ))

    # =================================================================
    # STUDY 9h: 高We基底交叉（We=15, d/D=1.2）
    # 对比不同基底在高We下的聚结行为
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9h: 高We基底交叉 (We=15, d/D=1.2, θ=162°)")
    print("=" * 70)
    for sub, R, amp in [('flat', None, 0.0), ('convex', 1.0, 1.8), ('concave', 1.0, 1.8)]:
        results.append(run_dual_case(
            substrate_type=sub, R_star=R, We=15.0,
            theta_eq=162.0, amp=amp, spacing_ratio=1.2,
            N=4000, n_base=150,
            label=f"s9h_{sub}_R{R or 'flat'}_We15",
            save_snapshots=False,
        ))
    # 额外: We=25 ridge, 看极端We下的行为
    results.append(run_dual_case(
        substrate_type='ridge', R_star=1.0, We=25.0,
        theta_eq=162.0, amp=1.8, spacing_ratio=1.2,
        N=4000, n_base=150,
        label=f"s9h_ridge_R1.0_We25",
        save_snapshots=True,
    ))

    # =================================================================
    # STUDY 9i: 长时动力学追踪（N=6000）
    # 捕获聚结后的完整演化过程
    # =================================================================
    print("\n" + "=" * 70)
    print("  STUDY 9i: 长时动力学追踪 (N=6000)")
    print("=" * 70)
    # flat 在 d/D=1.2 发生聚结, 延长看聚结后演化
    results.append(run_dual_case(
        substrate_type='flat', R_star=None, We=7.9,
        theta_eq=162.0, amp=0.0, spacing_ratio=1.2,
        N=6000, n_base=150,
        label=f"s9i_flat_Rflat_d1.2_long",
        save_snapshots=True,
    ))
    # ridge d/D=2.0 (最远间距), 看是否会最终聚结
    results.append(run_dual_case(
        substrate_type='ridge', R_star=1.0, We=7.9,
        theta_eq=162.0, amp=1.8, spacing_ratio=2.0,
        N=6000, n_base=150,
        label=f"s9i_ridge_R1.0_d2.0_long",
        save_snapshots=True,
    ))

    # =================================================================
    # 汇总
    # =================================================================
    print("\n" + "=" * 70)
    print("  补充双液滴仿真结果汇总")
    print("=" * 70)
    ok = sum(1 for r in results if r['stable'])
    print(f"\n  总计: {len(results)} cases, {ok} OK, {len(results)-ok} FAIL")

    # 按研究分组打印
    for study in ['s9d', 's9e', 's9f', 's9g', 's9h', 's9i']:
        study_cases = [r for r in results if r['label'].startswith(study)]
        if study_cases:
            print(f"\n  --- {study} ---")
            for r in study_cases:
                tag = "OK" if r['stable'] else "FAIL"
                coal = r.get('coalescence_step')
                coal_str = f"merge@{coal}" if coal else "separate"
                theta = r.get('theta_eq', 162.0)
                d = r.get('spacing_ratio', 1.2)
                print(f"    {r['label']:>45} θ={theta:3.0f} d/D={d:.1f} We={r['We']:5.1f} "
                      f"k={r['max_k']:.4f} {coal_str:15s} [{tag}]")

    # 保存结果
    save_dir = os.path.join(_project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)

    # 合并到主文件
    main_path = os.path.join(save_dir, 'studies_9_10.json')
    if os.path.exists(main_path):
        with open(main_path) as f:
            main_data = json.load(f)
    else:
        main_data = []

    # 去重: 按label去重
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

    # 保存历史数据到单独文件
    hist_path = os.path.join(save_dir, 'supplementary_dual_history.json')
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
