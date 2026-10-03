import torch, triton
from flash_attention import flash_attention

torch.manual_seed(0)
seq_len, head_dim = 2048, 64
q = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
scale = 1.0 / (head_dim**0.5)

for _ in range(5):
    t_false = triton.testing.do_bench(lambda: flash_attention(q, k, v, scale=scale, is_causal=False))
    t_true = triton.testing.do_bench(lambda: flash_attention(q, k, v, scale=scale, is_causal=True))
    print(f"non-causal={t_false:.4f}ms  causal={t_true:.4f}ms  ratio={t_false/t_true:.2f}x")
