import torch
from executor import Executor, register_kernel
from scheduler import OpNode
from allocator import PoolAllocator, BumpAllocator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}\n")

# ══════════════════════════════════════════════════════
# T1: 简单3-op线性链  x->matmul->relu->matmul->out
# ══════════════════════════════════════════════════════
print("=== T1: linear chain (matmul→relu→matmul) ===")

ops = [
    OpNode("mm0",  "matmul", ["x", "w0"],  ["h0"]),
    OpNode("relu", "relu",   ["h0"],        ["a0"]),
    OpNode("mm1",  "matmul", ["a0", "w1"], ["out"]),
]
B, D = 4, 64
x  = torch.randn(B, D,  device=device, dtype=torch.float32)
w0 = torch.randn(D, D,  device=device, dtype=torch.float32)
w1 = torch.randn(D, 16, device=device, dtype=torch.float32)

exe = Executor(allocator=PoolAllocator(), device=device, verbose=True)
result = exe.run(ops,
                 inputs={"x": x, "w0": w0, "w1": w1},
                 graph_outputs={"out"})

out = result["out"]
print(f"  output shape: {out.shape}")
assert out.shape == (B, 16), f"wrong shape {out.shape}"

# 手动验证数值
ref = torch.matmul(torch.relu(torch.matmul(x, w0)), w1)
assert torch.allclose(out, ref, atol=1e-5), "T1 value mismatch"
print("  value correct ✓")
print(exe.stats.summary())

# ══════════════════════════════════════════════════════
# T2: attention block  (q,k,v → flash_attn → linear → out)
# ══════════════════════════════════════════════════════
print("\n=== T2: attention block ===")

ops2 = [
    OpNode("attn", "flash_attn", ["q","k","v"], ["ctx"],
           attrs={"causal": True}),
    OpNode("proj", "linear",     ["ctx","w_o"], ["out2"]),
]
S, H, D_h = 8, 4, 16
q  = torch.randn(1, H, S, D_h, device=device, dtype=torch.float32)
k  = torch.randn(1, H, S, D_h, device=device, dtype=torch.float32)
v  = torch.randn(1, H, S, D_h, device=device, dtype=torch.float32)
w_o = torch.randn(D_h, D_h,    device=device, dtype=torch.float32)

exe2 = Executor(device=device, verbose=True)
result2 = exe2.run(ops2,
                   inputs={"q":q, "k":k, "v":v, "w_o":w_o},
                   graph_outputs={"out2"})
print(f"  output shape: {result2['out2'].shape}")
print(exe2.stats.summary())

# ══════════════════════════════════════════════════════
# T3: 未注册op → 清晰报错
# ══════════════════════════════════════════════════════
print("\n=== T3: unknown op error ===")
ops3 = [OpNode("bad", "unknown_op", ["x"], ["y"])]
try:
    Executor(device=device).run(ops3, {"x": x}, {"y"})
    print("  error: MISSED (bug!)")
except NotImplementedError as e:
    print(f"  NotImplementedError caught ✓  {e}")

# ══════════════════════════════════════════════════════
# T4: BumpAllocator集成
# ══════════════════════════════════════════════════════
print("\n=== T4: BumpAllocator integration ===")
from allocator import BumpAllocator
bump = BumpAllocator(capacity_mb=64, device=device)
exe4 = Executor(allocator=bump, device=device)
result4 = exe4.run(
    [OpNode("mm", "matmul", ["x","w0"], ["out"])],
    inputs={"x": x, "w0": w0},
    graph_outputs={"out"},
)
print(f"  output shape: {result4['out'].shape}")
print(f"  bump used: {bump.used_mb:.3f} MB")
print("  BumpAllocator ✓")

print("\n✓ All Executor tests passed")
