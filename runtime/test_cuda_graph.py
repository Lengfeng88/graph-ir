"""
Phase 9 扩展 — CUDA Graph 测试
T1: 正确性（eager vs graph结果一致）
T2: 性能（replay比eager快）
T3: 多次replay（输入变化时结果跟着变）
T4: reset后重新capture
"""
import torch, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if not torch.cuda.is_available():
    print("SKIP: no CUDA")
    sys.exit(0)

from scheduler import OpNode
from allocator import PoolAllocator
from executor  import Executor
from cuda_graph import CUDAGraphExecutor

device = "cuda"
print(f"device: {torch.cuda.get_device_name(0)}\n")

# 共用op图：matmul → relu → matmul
B, D = 32, 256
ops = [
    OpNode("mm0",  "matmul", ["x","w0"], ["h0"]),
    OpNode("relu", "relu",   ["h0"],     ["a0"]),
    OpNode("mm1",  "matmul", ["a0","w1"],["out"]),
]
w0 = torch.randn(D, D,  device=device)
w1 = torch.randn(D, 16, device=device)

def make_inputs(seed=0):
    torch.manual_seed(seed)
    return {
        "x":  torch.randn(B, D, device=device),
        "w0": w0,
        "w1": w1,
    }

# ══════════════════════════════════════════════════════
# T1: 正确性
# ══════════════════════════════════════════════════════
print("=== T1: correctness (eager vs graph) ===")
inputs = make_inputs(42)

# eager reference
eager = Executor(device=device)
ref = eager.run(ops, inputs, {"out"})["out"]

# graph
gexe = CUDAGraphExecutor(ops, {"out"}, device=device, verbose=True)
out_g = gexe.run(inputs)["out"]

diff = (out_g - ref).abs().max().item()
print(f"  max diff eager vs graph: {diff:.2e}")
assert diff < 1e-4, f"correctness fail: diff={diff}"
print("  correctness ✓")

# ══════════════════════════════════════════════════════
# T2: 性能对比
# ══════════════════════════════════════════════════════
print("\n=== T2: performance (eager vs replay) ===")
RUNS = 200

# eager warmup + bench
for _ in range(10):
    Executor(device=device).run(ops, inputs, {"out"})
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS):
    Executor(device=device).run(ops, inputs, {"out"})
torch.cuda.synchronize()
eager_ms = (time.perf_counter() - t0) * 1000 / RUNS

# graph replay bench（已经captured）
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS):
    gexe.run(inputs)
torch.cuda.synchronize()
graph_ms = (time.perf_counter() - t0) * 1000 / RUNS

speedup = eager_ms / graph_ms
print(f"  eager  avg: {eager_ms:.3f} ms")
print(f"  graph  avg: {graph_ms:.3f} ms")
print(f"  speedup:    {speedup:.2f}x")
assert speedup > 1.5, f"expected >1.5x speedup, got {speedup:.2f}x"
print("  speedup ✓")

# ══════════════════════════════════════════════════════
# T3: 多次replay，输入变化结果跟着变
# ══════════════════════════════════════════════════════
print("\n=== T3: replay with changing inputs ===")
for seed in range(3):
    inp = make_inputs(seed)
    ref_i = Executor(device=device).run(ops, inp, {"out"})["out"]
    out_i = gexe.run(inp)["out"]
    diff_i = (out_i - ref_i).abs().max().item()
    print(f"  seed={seed}  diff={diff_i:.2e}")
    assert diff_i < 1e-4, f"seed={seed} fail: diff={diff_i}"
print("  all seeds ✓")

# ══════════════════════════════════════════════════════
# T4: reset → 重新capture
# ══════════════════════════════════════════════════════
print("\n=== T4: reset and recapture ===")
gexe.reset()
assert not gexe._captured
inp2 = make_inputs(99)
out2 = gexe.run(inp2)["out"]
ref2 = Executor(device=device).run(ops, inp2, {"out"})["out"]
diff2 = (out2 - ref2).abs().max().item()
print(f"  after reset diff: {diff2:.2e}")
assert diff2 < 1e-4
print("  recapture ✓")

print("\n✓ All CUDA Graph tests passed")
