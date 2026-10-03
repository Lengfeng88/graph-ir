import torch
from allocator import BumpAllocator, PoolAllocator, CUDAAllocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

# ── BumpAllocator ──
print("=== BumpAllocator ===")
bump = BumpAllocator(capacity_mb=64, device=device)

t1 = bump.malloc((1024, 1024), torch.float16, device)
t2 = bump.malloc((512, 512),   torch.float32, device)
print(f"  t1: {t1.shape} {t1.dtype}")
print(f"  t2: {t2.shape} {t2.dtype}")
print(f"  used={bump.used_mb:.3f}MB  free={bump.free_mb:.3f}MB")

bump.reset()
print(f"  after reset: used={bump.used_mb:.3f}MB")

# OOM guard
try:
    small = BumpAllocator(capacity_mb=1, device=device)
    small.malloc((1024, 1024), torch.float32, device)   # 4MB into 1MB
    print("  OOM: MISSED (bug!)")
except MemoryError as e:
    print(f"  OOM caught ✓  {e}")

# ── PoolAllocator ──
print("\n=== PoolAllocator ===")
pool = PoolAllocator()

t3 = pool.malloc((2, 8, 128, 64), torch.float16, device)
print(f"  1st malloc: {pool.stats()}")
pool.free(t3)

t4 = pool.malloc((2, 8, 128, 64), torch.float16, device)
print(f"  2nd malloc: {pool.stats()}")
assert t4.data_ptr() == t3.data_ptr(), "buffer should be reused"
print(f"  same buffer reused ✓  ptr={hex(t4.data_ptr())}")

# ── CUDAAllocator ──
print("\n=== CUDAAllocator ===")
if torch.cuda.is_available():
    stream = torch.cuda.Stream()
    alloc  = CUDAAllocator(stream=stream)
    t5 = alloc.malloc((4096, 4096), torch.float16, device)
    print(f"  t5: {t5.shape} on {t5.device}")
else:
    alloc = CUDAAllocator()
    t5 = alloc.malloc((256, 256), torch.float32, device)
    print(f"  t5 (CPU fallback): {t5.shape} on {t5.device}")

print("\n✓ All allocator tests passed")
