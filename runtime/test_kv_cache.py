import torch
from kv_cache import KVCacheConfig, KVCacheManager

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = KVCacheConfig(
    num_layers=4, num_heads=4, head_dim=32,
    block_size=4, num_blocks=64,
    dtype=torch.float16, device=device,
)
mgr = KVCacheManager(cfg)
print(f"init: {mgr.stats()}\n")

# ── prefill: seq=0，写6个token ──
mgr.allocate(seq_id=0)
k6 = torch.randn(6, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device)
v6 = torch.randn(6, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device)
mgr.append(0, layer=0, k=k6, v=v6)
print(f"after prefill(6 tokens): {mgr.stats()}")
# 6 tokens, block_size=4 → 需要2个block
assert mgr.stats()["used_blocks"] == 2, "should use 2 blocks for 6 tokens"
print("  block count ✓")

# ── decode: 再写3个token ──
for i in range(3):
    k1 = torch.randn(1, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device)
    v1 = torch.randn(1, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device)
    mgr.append(0, layer=0, k=k1, v=v1)
print(f"after decode(3 tokens): {mgr.stats()}")
# 9 tokens → ceil(9/4)=3 blocks
assert mgr.stats()["used_blocks"] == 3, "should use 3 blocks for 9 tokens"
print("  block count ✓")

# ── 读回，验证shape和数值 ──
k_all, v_all = mgr.get_kv(seq_id=0, layer=0)
print(f"get_kv: k={k_all.shape}, v={v_all.shape}")
assert k_all.shape == (9, cfg.num_heads, cfg.head_dim), f"wrong shape {k_all.shape}"
print("  shape ✓")

# 验证prefill的6个token数值保存正确
assert torch.allclose(k_all[:6], k6, atol=1e-3), "prefill K mismatch"
assert torch.allclose(v_all[:6], v6, atol=1e-3), "prefill V mismatch"
print("  values match original ✓")

# ── 第二条序列 ──
mgr.allocate(seq_id=1)
k2 = torch.randn(2, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device)
v2 = torch.randn(2, cfg.num_heads, cfg.head_dim, dtype=cfg.dtype, device=device)
mgr.append(1, layer=0, k=k2, v=v2)
print(f"\nafter seq1 prefill(2 tokens): {mgr.stats()}")

# ── 释放seq=0 ──
mgr.free(seq_id=0)
print(f"after free seq0: {mgr.stats()}")
assert mgr.stats()["used_blocks"] == 1, "seq0's 3 blocks should be freed"
print("  free ✓")

# ── OOM guard ──
print("\n=== OOM Guard ===")
tiny_cfg = KVCacheConfig(
    num_layers=1, num_heads=1, head_dim=8,
    block_size=4, num_blocks=2,
    dtype=torch.float16, device=device,
)
tiny = KVCacheManager(tiny_cfg)
tiny.allocate(0)
try:
    # 塞9个token → 需要3个block，但只有2个
    k9 = torch.randn(9, 1, 8, dtype=torch.float16, device=device)
    v9 = torch.randn(9, 1, 8, dtype=torch.float16, device=device)
    tiny.append(0, layer=0, k=k9, v=v9)
    print("  OOM: MISSED (bug!)")
except RuntimeError as e:
    print(f"  OOM caught ✓  {e}")

print("\n✓ All KV Cache tests passed")
