"""Entry point for running SpMM benchmarks."""

import torch
from cprint import c_print

from base_cfg import BASE_DIR
from benchmark.bench_utils import benchmark_pytorch_spmm, torch_csr, triton_fn
from benchmark.ell_kernel import csr_to_ell, print_best_config

def main():
    A = torch.load(BASE_DIR / "grad_mat.pth").cuda()
    shape = A.shape
    device = A.device

    # c_print("Starting pytorch SpMM benchmark...", "green")
    # # --- PyTorch CSR SpMM ---
    # y_torch = benchmark_pytorch_spmm(
    #     A, torch_csr, shape,
    #     dtype=torch.float32, device=device,
    # )
    # print()
    c_print("Starting triton SpMM benchmark...", "green")

    # --- Triton ELL SpMM ---
    vals, cols = csr_to_ell(A, K=4)
    y_triton = benchmark_pytorch_spmm(
        (vals, cols), triton_fn, shape,
        dtype=torch.float32, device=device,
    )
    print_best_config()

    # c_print("Testing allclose:", "bright_green")
    # print(torch.allclose(y_torch, y_triton))


if __name__ == "__main__":
    main()
