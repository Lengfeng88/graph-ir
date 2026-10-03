from attention_ir import AttentionOp, lower_to_flash
import os

def codegen_triton_causal(ir, causal):
    op, Br, Bc, D = ir.op, ir.Br, ir.Bc, ir.op.head_dim
    fname_suffix = "causal" if causal else "full"
    mask_line = ""
    if causal:
        mask_line = "        S = tl.where(q_offs[:, None] >= k_rows[None, :], S, -3.4e38)\n"

    src = f'''# AUTO-GENERATED causal={causal}
import torch, triton, triton.language as tl, time

@triton.jit
def flash_attn_{fname_suffix}(
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
{mask_line}        m_new = tl.maximum(m, tl.max(S, axis=1))
        alpha = tl.exp(m - m_new)
        P     = tl.exp(S - m_new[:, None])
        l     = alpha * l + tl.sum(P, axis=1)
        O     = alpha[:, None] * O + tl.dot(P.to(tl.float32), V)
        m     = m_new
    O = O / l[:, None]
    tl.store(O_ptr + q_offs[:, None] * D + d_offs[None, :],
             O, mask=q_offs[:, None] < N)

def run(Q, K, V, Br={Br}, Bc={Bc}):
    N, D = Q.shape
    O = torch.empty_like(Q)
    flash_attn_{fname_suffix}[(triton.cdiv(N, Br),)](
        Q, K, V, O, N=N, D=D, Br=Br, Bc=Bc)
    return O

if __name__ == "__main__":
    torch.manual_seed(42)
    dev = "cuda"
    print(f"Phase 5E causal={causal}  Br={Br}  Bc={Bc}  D={D}")
    print("=" * 50)
    for N in [128, 512, 1024]:
        Q = torch.randn(N, {D}, device=dev) * 0.1
        K = torch.randn(N, {D}, device=dev) * 0.1
        V = torch.randn(N, {D}, device=dev) * 0.1
        scale = 1.0 / ({D} ** 0.5)
        S_ref = torch.matmul(Q, K.T) * scale
        if {causal}:
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
        status = "PASSED" if bad == 0 else f"FAILED(bad={{bad}})"
        print(f"N={{N:5d}}: {{status}}  max_abs={{abs_err.max():.2e}}  {{ms:.3f}}ms")
'''
    return src, f"gen_triton_{fname_suffix}.py"

if __name__ == "__main__":
    op = AttentionOp(seq_len=1024, head_dim=64, causal=False, dtype="fp32")
    ir = lower_to_flash(op, Br=16, Bc=16, backend="triton")
    for causal in [False, True]:
        src, fname = codegen_triton_causal(ir, causal)
        with open(fname, "w") as f:
            f.write(src)
        print(f"Generated: {fname}")
        os.system(f"python3 {fname}")
        print()
