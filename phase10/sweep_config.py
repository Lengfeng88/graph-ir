SHAPES = []

# prefill: q_len == kv_len == N, causal
for B in [1, 4, 16]:
    for N in [128, 512, 2048, 8192]:
        for H in [1, 8, 32]:
            SHAPES.append(dict(batch=B, num_heads=H, head_dim=64,
                                q_len=N, kv_len=N, causal=True))

# decode: q_len=1, kv_len varies
for B in [1, 16, 64]:
    for KV in [128, 2048, 8192]:
        SHAPES.append(dict(batch=B, num_heads=8, head_dim=64,
                            q_len=1, kv_len=KV, causal=False))

# sparse-only: batch must be 1 (BlockSparseAttentionWrapper limitation), needs sparse_density
SPARSE_SHAPES = []
for N in [2048, 4096, 8192, 16384]:
    for density in [0.02, 0.05, 0.1]:
        SPARSE_SHAPES.append(dict(batch=1, num_heads=8, head_dim=64,
                                   q_len=N, kv_len=N, causal=True,
                                   sparse_density=density, allow_approx=True))
