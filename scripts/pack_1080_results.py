#!/usr/bin/env python3
"""精简1080补充双液滴结果并打包传输。

用法: python3 scripts/pack_1080_results.py
输出: ~/data_1080.tar.gz
"""
import numpy as np
import glob
import tarfile
import json
import os

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
pack_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pytorch_lbm root

files_to_pack = []

# 1. studies_9_10.json (JSON直接打包)
f = os.path.join(results_dir, 'studies_9_10.json')
if os.path.exists(f):
    files_to_pack.append(('results/studies_9_10.json', f))
    with open(f) as fh:
        data = json.load(fh)
    print(f'studies_9_10.json: {len(data)} cases')

# 2. supplementary_dual_history.json (可能损坏，容错处理)
f = os.path.join(results_dir, 'supplementary_dual_history.json')
if os.path.exists(f):
    try:
        with open(f) as fh:
            data = json.load(fh)
        files_to_pack.append(('results/supplementary_dual_history.json', f))
        print(f'supplementary_dual_history.json: {len(data)} entries')
    except json.JSONDecodeError as e:
        print(f'WARNING: supplementary_dual_history.json 损坏 ({e})，跳过')
        print(f'  提示: 该文件不影响主要结果，studies_9_10.json 已包含所有数据')

# 3. 快照精简 (只保留xz切片)
snap_dir = os.path.join(results_dir, 'thesis_snapshots')
snap_files = sorted(glob.glob(os.path.join(snap_dir, 's9*.npz')))
print(f'\n1080快照: {len(snap_files)} 个')

thinned_dir = os.path.join(results_dir, '_thinned_tmp')
os.makedirs(thinned_dir, exist_ok=True)

for i, f in enumerate(snap_files, 1):
    basename = os.path.basename(f)
    out_path = os.path.join(thinned_dir, basename)
    d = dict(np.load(f))
    thin = {k: v for k, v in d.items() if k in ('xz_slice', 'xz_solid', 'step', 'nx', 'ny', 'nz')}
    np.savez_compressed(out_path, **thin)
    if i % 10 == 0:
        print(f'  精简: {i}/{len(snap_files)}')

for f in sorted(glob.glob(os.path.join(thinned_dir, 's9*.npz'))):
    files_to_pack.append((f'results/thesis_snapshots/{os.path.basename(f)}', f))

# 打包
tar_path = os.path.join(pack_dir, 'data_1080.tar.gz')
with tarfile.open(tar_path, 'w:gz') as tar:
    for arcname, fullpath in files_to_pack:
        tar.add(fullpath, arcname=arcname)

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f'\n打包完成: {tar_path} ({size_mb:.1f}MB)')
print(f'传输到本机: scp user@1080:~/data_1080.tar.gz /tmp/')

# 清理
import shutil
shutil.rmtree(thinned_dir)
