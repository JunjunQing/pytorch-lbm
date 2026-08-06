#!/usr/bin/env python3
"""精简A750补充多液滴结果并打包传输。

用法: python3 scripts/pack_a750_results.py
输出: ~/data_a750.tar.gz
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

# 2. supplementary_multi_history.json
f = os.path.join(results_dir, 'supplementary_multi_history.json')
if os.path.exists(f):
    files_to_pack.append(('results/supplementary_multi_history.json', f))
    with open(f) as fh:
        data = json.load(fh)
    print(f'supplementary_multi_history.json: {len(data)} entries')

# 3. 快照精简 (只保留xz切片)
snap_dir = os.path.join(results_dir, 'thesis_snapshots')
snap_files = sorted(glob.glob(os.path.join(snap_dir, 's10*.npz')))
print(f'\nA750快照: {len(snap_files)} 个')

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

for f in sorted(glob.glob(os.path.join(thinned_dir, 's10*.npz'))):
    files_to_pack.append((f'results/thesis_snapshots/{os.path.basename(f)}', f))

# 打包
tar_path = os.path.join(pack_dir, 'data_a750.tar.gz')
with tarfile.open(tar_path, 'w:gz') as tar:
    for arcname, fullpath in files_to_pack:
        tar.add(fullpath, arcname=arcname)

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f'\n打包完成: {tar_path} ({size_mb:.1f}MB)')
print(f'传输到本机: scp user@a750:~/data_a750.tar.gz /tmp/')

# 清理
import shutil
shutil.rmtree(thinned_dir)
