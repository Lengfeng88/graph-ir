"""
compare_matmul.py — Cross-check kernels.py's Phase 3a triton_matmul (which
internally casts to fp16) against kernels/triton/matmul.py's Phase 4 version
(which stays in fp32).

These are NOT directly comparable by just subtracting their outputs — one is
fp16-precision, the other is fp32-precision, so a nonzero diff between them
is expected and uninformative on its own (it doesn't tell you which one, if
either, has a logic bug). What actually matters:
  1. Does the Phase 3a fp16 result match an fp16-precision reference?
  2. Does the Phase 4 fp32 result match an fp32-precision reference?
If both hold, the two kernels are doing the same underlying computation,
just at different precisions — the diff between them is precision, not a bug.
If either fails its own precision-appropriate reference, THAT is a real bug
in that specific kernel, independent of the other one.

Run from the project root (so both `kernels.py` and `kernels/triton/matmul.py`
are importable):
    python kernels/triton/compare_matmul.py
"""

import sys
import os

sys.path.insert(0, os.getcwd())  # for `kernels.py` at the project root
sys.path.insert(0, os.path.dirname(__file__))  # for local matmul.py

import torch

from kernels import triton_matmul as matmul_phase3a
from matmul import matmul as matmul_phase4

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

test_cases = [(64, 64, 64), (1000, 700, 300), (4096, 4096, 4096)]

print(
    f"{'M':>5} {'N':>5} {'K':>5} | "
    f"{'phase3a vs fp16 ref':>20} | {'phase4 vs fp32 ref':>19} | "
    f"{'phase3a vs phase4 (raw)':>24}"
)
print("-" * 90)

for M, N, K in test_cases:
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Phase 3a's own precision contract: it casts to fp16 regardless of
    # input dtype, so its correctness oracle has to be an fp16 matmul, not
    # an fp32 one.
    ref_fp16 = (a.half() @ b.half()).float()
    c_phase3a = matmul_phase3a(a, b).float()
    diff_phase3a = (c_phase3a - ref_fp16).abs().max().item()

    # Phase 4's precision contract: stays fp32 throughout.
    ref_fp32 = a @ b
    c_phase4 = matmul_phase4(a, b)
    diff_phase4 = (c_phase4 - ref_fp32).abs().max().item()

    # This is expected to be nonzero and roughly fp16-epsilon-sized — NOT
    # evidence of a bug in either kernel on its own.
    diff_cross = (c_phase3a - c_phase4).abs().max().item()

    print(
        f"{M:>5} {N:>5} {K:>5} | "
        f"{diff_phase3a:>20.3e} | {diff_phase4:>19.3e} | {diff_cross:>24.3e}"
    )
