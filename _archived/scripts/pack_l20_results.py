#!/usr/bin/env python3
"""精简L20第四章线形成结果并打包传输。

用法: python3 scripts/pack_l20_results.py
输出: pytorch_lbm/data_l20.tar.gz
"""
import numpy as np
import glob
import tarfile
import json
import os

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
pack_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pytorch_lbm root

files_to_pack = []

# 1. chapter4_line_results.json
f = os.path.join(results_dir, 'chapter4_line_results.json')
if os.path.exists(f):
    files_to_pack.append(('results/chapter4_line_results.json', f))
    with open(f) as fh:
        data = json.load(fh)
    print(f'chapter4_line_results.json: {len(data)} cases')

# 2. chapter4_line_history.json
f = os.path.join(results_dir, 'chapter4_line_history.json')
if os.path.exists(f):
    try:
        with open(f) as fh:
            data = json.load(fh)
        files_to_pack.append(('results/chapter4_line_history.json', f))
        print(f'chapter4_line_history.json: {len(data)} entries')
    except json.JSONDecodeError as e:
        print(f'WARNING: chapter4_line_history.json 损坏 ({e})，跳过')

# 3. 快照精简 (chapter4_snapshots)
snap_dir = os.path.join(results_dir, 'chapter4_snapshots')
snap_files = sorted(glob.glob(os.path.join(snap_dir, '*.npz')))
print(f'\nL20快照: {len(snap_files)} 个')

thinned_dir = os.path.join(results_dir, '_thinned_tmp')
os.makedirs(thinned_dir, exist_ok=True)

for i, f in enumerate(snap_files, 1):
    basename = os.path.basename(f)
    out_path = os.path.join(thinned_dir, basename)
    d = dict(np.load(f))
    thin = {k: v for k, v in d.items()
            if k in ('xz_slice', 'xz_solid', 'xy_slice', 'xy_solid',
                     'step', 'nx', 'ny', 'nz', 'p_ratio', 'n_drops')}
    np.savez_compressed(out_path, **thin)
    if i % 10 == 0:
        print(f'  精简: {i}/{len(snap_files)}')

for f in sorted(glob.glob(os.path.join(thinned_dir, '*.npz'))):
    files_to_pack.append((f'results/chapter4_snapshots/{os.path.basename(f)}', f))

# 打包
tar_path = os.path.join(pack_dir, 'data_l20.tar.gz')
with tarfile.open(tar_path, 'w:gz') as tar:
    for arcname, fullpath in files_to_pack:
        tar.add(fullpath, arcname=arcname)

size_mb = os.path.getsize(tar_path) / 1024 / 1024
print(f'\n打包完成: {tar_path} ({size_mb:.1f}MB)')

# 清理
import shutil
shutil.rmtree(thinned_dir)
