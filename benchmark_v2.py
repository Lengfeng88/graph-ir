import torch
import torch.nn.functional as F
import time

device = torch.device("cuda")

def bench(N, D=64, runs=50):
    scale = 1.0 / (D ** 0.5)

    # FP32 naive
    Q = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1
    K = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1
    V = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1

    for _ in range(5):
        S = torch.matmul(Q, K.T) * scale
        P = torch.softmax(S, dim=-1)
        out = torch.matmul(P, V)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        S = torch.matmul(Q, K.T) * scale
        P = torch.softmax(S, dim=-1)
        out = torch.matmul(P, V)
    torch.cuda.synchronize()
    ms_naive = (time.perf_counter() - t) / runs * 1000

    # FP16 SDPA (FlashAttention path)
    Qh = Q.half().unsqueeze(0).unsqueeze(0)
    Kh = K.half().unsqueeze(0).unsqueeze(0)
    Vh = V.half().unsqueeze(0).unsqueeze(0)

    for _ in range(5):
        out2 = F.scaled_dot_product_attention(Qh, Kh, Vh)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        out2 = F.scaled_dot_product_attention(Qh, Kh, Vh)
    torch.cuda.synchronize()
    ms_sdpa = (time.perf_counter() - t) / runs * 1000

    # HBM traffic
    attn_matrix_gb = N * N * 4 / 1e9   # S matrix，FP32
    qkvo_gb = 4 * N * D * 4 / 1e9      # Q+K+V+O，FP32

    print(f"N={N:5d}  "
          f"naive={ms_naive:.3f}ms  "
          f"sdpa_fp16={ms_sdpa:.3f}ms  "
          f"speedup={ms_naive/ms_sdpa:.1f}x  "
          f"S_matrix={attn_matrix_gb*1000:.1f}MB  "
          f"QKVO={qkvo_gb*1000:.1f}MB")

print(f"{'N':>7}  {'naive FP32':>12}  {'SDPA FP16':>11}  "
      f"{'speedup':>8}  {'S(N²)':>8}  {'QKVO':>8}")
for N in [512, 1024, 2048, 4096, 8192]:
    bench(N)
