"""Benchmarking utilities for sparse-matrix dense-matrix multiply (SpMM)."""

import time
import torch

from time_fvm.utils.ell_kernel import ell_spmm, ell_spmv


def torch_csr(A, x, iters):
    """Run iters iterations of PyTorch sparse CSR x dense SpMM."""
    for _ in range(iters):
        y = torch.sparse.mm(A, x)
    torch.cuda.synchronize()
    return y


def triton_fn(A, x, iters):
    """Run iters iterations of Triton ELL-packed SpMM."""
    vals, cols = A

    for _ in range(iters):
        Y = ell_spmm(vals, cols, x)
    return Y


@torch.inference_mode()
def benchmark_pytorch_spmm(
    A, spmm_fn, shape, bs,
    iters=1000,
    warmup=10,
    dtype=torch.float32, device="cuda"
):
    """Benchmark an SpMM function and optionally export a Chrome trace."""
    torch.manual_seed(0)

    m, n = shape
    x = torch.randn((n, bs), device=device, dtype=dtype)

    # Warmup
    spmm_fn(A, x, warmup)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    y = spmm_fn(A, x, iters)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000 / iters
    # nnz = A.values().numel()

    print(f"avg time: {avg_ms:.4f} ms")

    return y
