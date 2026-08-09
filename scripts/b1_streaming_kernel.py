#!/usr/bin/env python3
"""B1 prototype: fused streaming kernel for D2Q9/D3Q19 (A-A pattern).

Replaces the per-direction torch.roll loop in AllenCahnSolver.step()
(9-19 rolls for f + g per step, each a full-grid copy) with ONE kernel
that streams all directions in a single pass (periodic semantics, same as
torch.roll; bounce-back still handles solid nodes afterwards).

Correctness: bit-identical to the torch.roll version (periodic shift).
Build: torch.utils.cpp_extension.load_inline (JIT, first call compiles).

Usage: venv/bin/python scripts/b1_streaming_kernel.py [--bench]
"""
import os
import sys
import time
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Stream all Q directions in one pass: f_new[q][x] = f[q][x - e_q]
// (periodic, matching torch.roll semantics; e stored per direction,
//  only the first ndim entries are used).
__global__ void stream_fused_kernel(
    const float* __restrict__ f,
    float* __restrict__ f_new,
    const int* __restrict__ e,
    int Q, int nx, int ny, int nz, int ndim)
{
    int nodes = nx * ny * nz;
    long total = (long)Q * nodes;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int q = (int)(t / nodes);
    int node = (int)(t % nodes);
    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (nz * ny);

    // Match the solver's streaming convention: torch.roll(shift=-e)
    // yields f_new[x] = f_old[x + e] (verified against the solver).
    int si = i, sj = j, sk = k;
    if (ndim >= 1) si = (i + e[q * ndim] + nx) % nx;
    if (ndim >= 2) sj = (j + e[q * ndim + 1] + ny) % ny;
    if (ndim >= 3) sk = (k + e[q * ndim + 2] + nz) % nz;
    int src = (si * ny + sj) * nz + sk;
    f_new[t] = f[(long)q * nodes + src];
}

torch::Tensor stream_fused(torch::Tensor f, torch::Tensor e, int64_t ndim) {
    TORCH_CHECK(f.is_cuda(), "f must be CUDA");
    TORCH_CHECK(f.scalar_type() == torch::kFloat32, "f must be float32");
    auto f_new = torch::empty_like(f);
    int Q = f.size(0);
    int nx = f.size(1), ny = f.size(2), nz = f.size(3);
    long total = (long)Q * nx * ny * nz;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    stream_fused_kernel<<<blocks, threads>>>(
        f.data_ptr<float>(), f_new.data_ptr<float>(),
        e.data_ptr<int>(), Q, nx, ny, nz, (int)ndim);
    return f_new;
}
"""

CPP_SRC = """
torch::Tensor stream_fused(torch::Tensor f, torch::Tensor e, int64_t ndim);
"""


def load_kernel():
    return load_inline(
        name='b1_stream_fused',
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=['stream_fused'],
        verbose=False,
    )


def roll_reference(f, e, ndim):
    """Reference: per-direction torch.roll (current solver behavior)."""
    Q = f.shape[0]
    out = f.clone()
    for q in range(Q):
        for d in range(ndim):
            s = int(e[q, d])
            if s != 0:
                out[q] = torch.roll(out[q], shifts=-s, dims=d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--q', type=int, default=9)
    ap.add_argument('--iters', type=int, default=200)
    args = ap.parse_args()

    print(f"Compiling fused streaming kernel (JIT, first call takes ~1 min)...",
          flush=True)
    t0 = time.time()
    mod = load_kernel()
    print(f"  compiled in {time.time()-t0:.0f}s", flush=True)

    torch.manual_seed(0)
    n = args.n
    f = torch.randn(args.q, n, n, 64, device='cuda', dtype=torch.float32)
    e = torch.tensor([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, -1], [1, -1], [-1, 1],
    ][:args.q], dtype=torch.int32, device='cuda')

    # correctness
    fused = mod.stream_fused(f, e, 2)
    ref = roll_reference(f, e, 2)
    ok = torch.equal(fused, ref)
    print(f"correctness vs torch.roll: {'BIT-IDENTICAL' if ok else 'MISMATCH'}")
    if not ok:
        diff = (fused - ref).abs().max().item()
        print(f"  max abs diff: {diff}")
        return

    # benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        ref = roll_reference(f, e, 2)
    torch.cuda.synchronize()
    t_roll = (time.perf_counter() - t0) / args.iters

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        fused = mod.stream_fused(f, e, 2)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / args.iters

    print(f"torch.roll loop : {t_roll*1e3:.3f} ms (18 copies for f+g)")
    print(f"fused kernel    : {t_fused*1e3:.3f} ms (1 kernel)")
    print(f"streaming speedup: {t_roll/t_fused:.1f}x")

    # end-to-end estimate: current step 46.4ms @64^3; streaming share = 2*18 rolls
    print(f"note: step() has 2x this streaming cost (f and g distributions)")


if __name__ == '__main__':
    main()
