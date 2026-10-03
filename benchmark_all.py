import torch
import torch.nn.functional as F
import subprocess, time, os

device = torch.device("cuda")

# ── PyTorch kernels ──────────────────────────────────────────────
def pt_naive(Q, K, V, scale):
    S = torch.matmul(Q, K.T) * scale
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)

def pt_sdpa(Q, K, V):
    Q2 = Q.half().unsqueeze(0).unsqueeze(0)
    K2 = K.half().unsqueeze(0).unsqueeze(0)
    V2 = V.half().unsqueeze(0).unsqueeze(0)
    return F.scaled_dot_product_attention(Q2, K2, V2)

def time_fn(fn, runs=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000

# ── CUDA kernel runner ───────────────────────────────────────────
def run_cuda_kernel(binary, N, D):
    """运行 CUDA binary，parse 输出中的 ms 数字"""
    try:
        result = subprocess.run(
            [f"./{binary}"], capture_output=True, text=True, timeout=30)
        # binary 输出格式：N=XXX ... YYY ms
        for line in result.stdout.split('\n'):
            if f'N={N}' in line and 'ms' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if 'ms' in p:
                        return float(parts[i-1])
    except Exception as e:
        pass
    return None

# ── Main benchmark ───────────────────────────────────────────────
def bench_all(N, D=64):
    scale = 1.0 / (D ** 0.5)
    Q = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1
    K = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1
    V = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1

    results = {}

    # PyTorch naive FP32
    results['pt_naive'] = time_fn(lambda: pt_naive(Q, K, V, scale))

    # PyTorch SDPA FP16
    results['pt_sdpa'] = time_fn(lambda: pt_sdpa(Q, K, V))

    # 理论 HBM traffic
    s_matrix_mb = N * N * 4 / 1e6
    qkvo_mb     = 4 * N * D * 4 / 1e6

    print(f"N={N:5d} D={D}  S_matrix={s_matrix_mb:.0f}MB")
    print(f"  {'kernel':<20} {'ms':>8}  {'vs_sdpa':>8}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}")

    sdpa_ms = results['pt_sdpa']
    for name, ms in results.items():
        ratio = ms / sdpa_ms
        print(f"  {name:<20} {ms:>8.3f}  {ratio:>7.1f}x")

    # FA v1, v2 从之前的测量结果读取（hardcoded）
    fa_v1 = {128: 0.452, 512: 1.807, 1024: 3.558, 2048: 7.157}
    fa_v2 = {128: 0.044, 512: 0.171, 1024: 0.615, 2048: 1.796}

    if N in fa_v1:
        ms = fa_v1[N]
        print(f"  {'fa_v1_FP32':<20} {ms:>8.3f}  {ms/sdpa_ms:>7.1f}x")
    if N in fa_v2:
        ms = fa_v2[N]
        print(f"  {'fa_v2_FP32':<20} {ms:>8.3f}  {ms/sdpa_ms:>7.1f}x")

    print(f"  {'pt_sdpa (target)':<20} {sdpa_ms:>8.3f}  {'1.0x':>8}")
    print()

print("="*55)
print("FlashAttention Benchmark — Phase 5C")
print("="*55)
for N in [128, 512, 1024, 2048]:
    bench_all(N)
