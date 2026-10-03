from scheduler import OpNode, TopologicalScheduler, LivenessAnalyzer, MemoryScheduler

# DAG:  x,w0 -> matmul_0 -> h0 -> relu_0 -> a0 -> matmul_1(+w1) -> out
#       故意乱序传入，验证拓扑排序
ops = [
    OpNode("matmul_1", "matmul", ["a0", "w1"], ["out"]),   # 故意放最前
    OpNode("relu_0",   "relu",   ["h0"],        ["a0"]),
    OpNode("matmul_0", "matmul", ["x", "w0"],   ["h0"]),
]

print("=== TopologicalScheduler ===")
sched = TopologicalScheduler(ops)
order = sched.schedule()
for i, op in enumerate(order):
    print(f"  [{i}] {op.name}: {op.inputs} -> {op.outputs}")

# 验证顺序必须是 matmul_0 → relu_0 → matmul_1
names = [op.name for op in order]
assert names == ["matmul_0", "relu_0", "matmul_1"], f"wrong order: {names}"
print("  order correct ✓")

print("\n=== LivenessAnalyzer ===")
liveness = LivenessAnalyzer()
lifetimes = liveness.analyze(order)
for name, lt in sorted(lifetimes.items()):
    print(f"  {name:6s}: alive [{lt.first_use}, {lt.last_use}]")

# h0 产生于step0，消费于step1，所以last_use=1
assert lifetimes["h0"].first_use == 0 and lifetimes["h0"].last_use == 1
# a0 产生于step1，消费于step2
assert lifetimes["a0"].first_use == 1 and lifetimes["a0"].last_use == 2
print("  lifetimes correct ✓")

print("\n=== MemoryScheduler ===")
mem = MemoryScheduler(lifetimes, graph_outputs={"out"})
plan = mem.free_schedule()
for step, tensors in sorted(plan.items()):
    print(f"  after step {step}: free {sorted(tensors)}")

# h0最后在step1用完 → step1后free
assert "h0" in plan[1]
# a0最后在step2用完 → step2后free
assert "a0" in plan[2]
# out是graph output → 不在plan里
assert not any("out" in ts for ts in plan.values())
print("  free schedule correct ✓")

# 环检测
print("\n=== Cycle Detection ===")
cycle_ops = [
    OpNode("A", "relu", ["b"], ["a"]),
    OpNode("B", "relu", ["a"], ["b"]),
]
try:
    TopologicalScheduler(cycle_ops).schedule()
    print("  cycle: MISSED (bug!)")
except RuntimeError as e:
    print(f"  cycle caught ✓  {e}")

print("\n✓ All scheduler tests passed")
