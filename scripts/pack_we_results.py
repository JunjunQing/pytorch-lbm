#!/usr/bin/env python3
"""精简We扫频快照并打包，用于从1080机器传输到本地。

用法: python3 scripts/pack_we_results.py
输出: supplementary_we_pack.tar.gz (约6-7MB)
"""
import numpy as np
import glob
import tarfile
import os

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
snap_dir = os.path.join(results_dir, 'thesis_snapshots')

# 收集快照文件
snap_files = sorted(
    glob.glob(os.path.join(snap_dir, 'morph_We*_step*.npz'))
    + glob.glob(os.path.join(snap_dir, 'sup_We*_step*.npz'))
)
print(f'找到 {len(snap_files)} 个快照文件')

# 精简：只保留 xz_slice, xz_solid, step, nx, ny, nz
thinned_dir = os.path.join(results_dir, 'thesis_snapshots_thinned')
os.makedirs(thinned_dir, exist_ok=True)

for i, f in enumerate(snap_files, 1):
    basename = os.path.basename(f)
    out_path = os.path.join(thinned_dir, basename)
    d = dict(np.load(f))
    thin = {k: v for k, v in d.items() if k in ('xz_slice', 'xz_solid', 'step', 'nx', 'ny', 'nz')}
    np.savez_compressed(out_path, **thin)
    if i % 20 == 0:
        print(f'  精简进度: {i}/{len(snap_files)}')

print(f'精简完成，输出目录: {thinned_dir}')

# 打包
tar_path = os.path.join(results_dir, 'supplementary_we_pack.tar.gz')
with tarfile.open(tar_path, 'w:gz') as tar:
    # JSON结果
    for name in ['supplementary_we_results.json', 'supplementary_we_history.json']:
        full = os.path.join(results_dir, name)
        if os.path.exists(full):
            tar.add(full, arcname=f'results/{name}')
            print(f'  添加 {name}')
        else:
            print(f'  警告: {name} 不存在，跳过')
    # 精简后的快照
    for f in sorted(glob.glob(os.path.join(thinned_dir, '*.npz'))):
        tar.add(f, arcname=f'results/thesis_snapshots/{os.path.basename(f)}')

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f'\n打包完成: {tar_path} ({size_mb:.1f}MB)')
print(f'包含 {len(snap_files)} 个精简快照 + JSON结果')
print(f'\n传输到本地后解压: tar xzf supplementary_we_pack.tar.gz -C /mnt/simulation-projects/pytorch_lbm/')

# 清理临时目录
import shutil
shutil.rmtree(thinned_dir)
print('临时文件已清理')
