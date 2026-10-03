# AUTO-GENERATED causal=True
import torch, triton, triton.language as tl, time

@triton.jit
def flash_attn_causal(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N: tl.constexpr, D: tl.constexpr,
    Br: tl.constexpr, Bc: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_start = q_block * Br
    scale   = 1.0 / tl.sqrt(float(D))
    q_offs  = q_start + tl.arange(0, Br)
    d_offs  = tl.arange(0, D)
    k_offs  = tl.arange(0, Bc)
    Q = tl.load(Q_ptr + q_offs[:, None] * D + d_offs[None, :],
                mask=q_offs[:, None] < N, other=0.0)
    m = tl.full([Br], -3.4e38, dtype=tl.float32)
    l = tl.zeros([Br], dtype=tl.float32)
    O = tl.zeros([Br, D], dtype=tl.float32)
    for kv in range(tl.cdiv(N, Bc)):
        k_rows = kv * Bc + k_offs
        K = tl.load(K_ptr + k_rows[:, None] * D + d_offs[None, :],
                    mask=k_rows[:, None] < N, other=0.0)
        V = tl.load(V_ptr + k_rows[:, None] * D + d_offs[None, :],
                    mask=k_rows[:, None] < N, other=0.0)
        S = tl.dot(Q, tl.trans(K)) * scale
        S = tl.where(k_rows[None, :] < N, S, -3.4e38)
        S = tl.where(q_offs[:, None] >= k_rows[None, :], S, -3.4e38)
        m_new = tl.maximum(m, tl.max(S, axis=1))
        alpha = tl.exp(m - m_new)
        P     = tl.exp(S - m_new[:, None])
        l     = alpha * l + tl.sum(P, axis=1)
        O     = alpha[:, None] * O + tl.dot(P.to(tl.float32), V)
        m     = m_new
    O = O / l[:, None]
    tl.store(O_ptr + q_offs[:, None] * D + d_offs[None, :],
             O, mask=q_offs[:, None] < N)

def run(Q, K, V, Br=16, Bc=16):
    N, D = Q.shape
    O = torch.empty_like(Q)
    flash_attn_causal[(triton.cdiv(N, Br),)](
        Q, K, V, O, N=N, D=D, Br=Br, Bc=Bc)
    return O

if __name__ == "__main__":
    torch.manual_seed(42)
    dev = "cuda"
    print(f"Phase 5E causal=True  Br=16  Bc=16  D=64")
    print("=" * 50)
    for N in [128, 512, 1024]:
        Q = torch.randn(N, 64, device=dev) * 0.1
        K = torch.randn(N, 64, device=dev) * 0.1
        V = torch.randn(N, 64, device=dev) * 0.1
        scale = 1.0 / (64 ** 0.5)
        S_ref = torch.matmul(Q, K.T) * scale
        if True:
            mask = torch.triu(torch.ones(N, N, device=dev), diagonal=1).bool()
            S_ref = S_ref.masked_fill(mask, -3.4e38)
        O_ref = torch.matmul(torch.softmax(S_ref, dim=-1), V)
        O_out = run(Q, K, V)
        abs_err = (O_out - O_ref).abs()
        bad = (abs_err > 5e-5 + 1e-3 * O_ref.abs().mean().item()).sum().item()
        for _ in range(5): run(Q, K, V)
        torch.cuda.synchronize()
        import time
        t = time.perf_counter()
        for _ in range(100): run(Q, K, V)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 100 * 1000
        status = "PASSED" if bad == 0 else f"FAILED(bad={bad})"
        print(f"N={N:5d}: {status}  max_abs={abs_err.max():.2e}  {ms:.3f}ms")
