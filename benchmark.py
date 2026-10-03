import torch
import torch.nn.functional as F
import time

device = torch.device("cuda")

def attention_ref(Q, K, V, scale):
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    return torch.matmul(P, V)

def bench(N, D=64, runs=100):
    scale = 1.0 / (D ** 0.5)
    Q = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1
    K = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1
    V = torch.randn(N, D, device=device, dtype=torch.float32) * 0.1

    # warmup
    for _ in range(10):
        out = attention_ref(Q, K, V, scale)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        out = attention_ref(Q, K, V, scale)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / runs * 1000

    # PyTorch SDPA (FlashAttention backend)
    Q2 = Q.unsqueeze(0).unsqueeze(0)
    K2 = K.unsqueeze(0).unsqueeze(0)
    V2 = V.unsqueeze(0).unsqueeze(0)
    for _ in range(10):
        out2 = F.scaled_dot_product_attention(Q2, K2, V2)
    torch.cuda.synchronize()
    start2 = time.perf_counter()
    for _ in range(runs):
        out2 = F.scaled_dot_product_attention(Q2, K2, V2)
    torch.cuda.synchronize()
    ms2 = (time.perf_counter() - start2) / runs * 1000

    print(f"N={N:5d}  naive_attn={ms:.3f}ms  sdpa={ms2:.3f}ms  speedup={ms/ms2:.1f}x")

print("PyTorch reference (FP32, D=64)")
print(f"{'N':>7}  {'naive':>12}  {'SDPA':>8}  {'speedup':>8}")
for N in [128, 512, 1024, 2048]:
    bench(N)
