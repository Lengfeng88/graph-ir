"""
Real SDPA measurement at num_heads=1 -- the regime pass4_bridge.py's
decision now depends on, but that calibrate.py never actually measured
(every calibration scenario used H=8 or H=32). Run this before trusting
the "dense wins at H=1" result from the extrapolated launch_us model.
"""
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from triton.testing import do_bench

BACKENDS = {"dense": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}

def measure(B, H, N, D, backend):
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    with sdpa_kernel(backend):
        return do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False))

for H in (1, 8):
    for B, N, D in [(2, 128, 64), (4, 256, 32)]:
        row = f"B={B} H={H} N={N} D={D}: "
        for name, be in BACKENDS.items():
            try:
                ms = measure(B, H, N, D, be)
                row += f"{name}={ms:.5f}ms  "
            except Exception as e:
                row += f"{name}=FAILED({type(e).__name__})  "
        print(row)
