"""
test_phase8.py  —  Phase 8 test suite (6 tests)

T1  TP   matmul        → AllReduce after
T2  CP   flash_attn    → AllGather before
T3  EP   expert_mlp    → AllToAll before + after
T4  FSDP matmul        → AllGather before + ReduceScatter after
T5  TP+EP              → AllReduce(attn) + AllToAll x2(expert)
T6  Full transformer   → Phase-7 fused ops, all 4 strategies
"""

import sys
sys.path.insert(0, '.')

from distributed_types import DistConfig, ParallelStrategy
from pass8_partition   import PartitionPass
from pass8_comm        import CommunicationPass, CommNode
from compiler_ir       import GraphBuilder, DType, TensorType, Shape, OpCode
from pass7_whole_graph import FUSED_NORM_QKV, FUSED_ATTN_PROJ, FUSED_MLP, P7Op

PASS = "\033[32mv\033[0m"
FAIL = "\033[31mx\033[0m"


# ── helpers ──────────────────────────────────────────────────────────

def run(ops_spec: list, config: DistConfig):
    """
    ops_spec: list of (opcode, operands_spec)
    Builds a minimal graph, runs P8 partition + comm, returns expanded list.
    """
    gb = GraphBuilder()
    X  = gb.param('X', TensorType(Shape.of(2,512,512), DType.f16))
    W  = gb.param('W', TensorType(Shape.of(512,512),   DType.f16))
    b  = gb.param('b', TensorType(Shape.of(512),       DType.f16))
    Q  = gb.param('Q', TensorType(Shape.of(2,8,512,64),  DType.f16))
    K  = gb.param('K', TensorType(Shape.of(2,8,512,64),  DType.f16))
    V  = gb.param('V', TensorType(Shape.of(2,8,512,64),  DType.f16))

    last = X
    for opcode in ops_spec:
        if opcode == OpCode.MATMUL:
            last = gb.matmul(last, W)
        elif opcode == OpCode.LAYER_NORM:
            last = gb.layer_norm(last, W, b)
        elif opcode == OpCode.FLASH_ATTN:
            last = gb.flash_attn(Q, K, V)
        elif opcode == OpCode.GELU:
            last = gb.gelu(last)
        elif opcode == OpCode.ADD:
            last = gb.add(last, X)
        elif isinstance(opcode, P7Op):
            # inject P7Op node directly into block
            from compiler_ir import Op
            fake_op = Op.__new__(Op)
            fake_op.opcode   = opcode
            fake_op.operands = []
            fake_op.results  = []
            fake_op.attrs    = {}
            fake_op.name     = str(opcode)
            fake_op._id      = 0
            fake_op.shard_specs = {}
            gb.graph.block.ops.append(fake_op)
        else:
            last = gb.matmul(last, W)  # fallback

    gb.mark_result(last)
    g = gb.graph

    PartitionPass(config).run(g)
    expanded = CommunicationPass(config).run(g)
    return expanded


def comm_nodes(expanded):
    return [n for n in expanded if isinstance(n, CommNode)]


def has_comm(expanded, comm_op, strategy, direction, anchor):
    for n in expanded:
        if (isinstance(n, CommNode) and
                n.comm_op   == comm_op and
                n.strategy  == strategy and
                n.direction == direction and
                n.anchor_op == anchor):
            return True
    return False


def no_comm_for(expanded, strategy):
    return not any(isinstance(n, CommNode) and n.strategy == strategy
                   for n in expanded)


# ── tests ────────────────────────────────────────────────────────────

def test_t1_tp_matmul():
    cfg = DistConfig(tp_size=4)
    exp = run([OpCode.MATMUL], cfg)
    assert has_comm(exp, "AllReduce", ParallelStrategy.TP, "after", "matmul")
    assert len(comm_nodes(exp)) == 1
    print(f"  {PASS} T1  TP/matmul -> 1x AllReduce after")


def test_t2_cp_flash_attn():
    cfg = DistConfig(cp_size=2)
    exp = run([OpCode.FLASH_ATTN], cfg)
    assert has_comm(exp, "AllGather", ParallelStrategy.CP, "before", "flash_attn")
    after = [n for n in comm_nodes(exp) if n.direction == "after"]
    assert len(after) == 0
    print(f"  {PASS} T2  CP/flash_attn -> 1x AllGather before, none after")


def test_t3_ep_expert():
    cfg = DistConfig(ep_size=8)
    exp = run([P7Op("EXPERT_MLP")], cfg)
    # P7Op name lowercases to "expert_mlp"
    anchor = "expert_mlp"
    assert has_comm(exp, "AllToAll", ParallelStrategy.EP, "before", anchor)
    assert has_comm(exp, "AllToAll", ParallelStrategy.EP, "after",  anchor)
    ep_comms = [n for n in comm_nodes(exp) if n.strategy == ParallelStrategy.EP]
    assert len(ep_comms) == 2
    print(f"  {PASS} T3  EP/expert_mlp -> AllToAll before + AllToAll after")


def test_t4_fsdp_matmul():
    cfg = DistConfig(fsdp_size=8)
    exp = run([OpCode.MATMUL], cfg)
    assert has_comm(exp, "AllGather",     ParallelStrategy.FSDP, "before", "matmul")
    assert has_comm(exp, "ReduceScatter", ParallelStrategy.FSDP, "after",  "matmul")
    fsdp = [n for n in comm_nodes(exp) if n.strategy == ParallelStrategy.FSDP]
    assert len(fsdp) == 2
    print(f"  {PASS} T4  FSDP/matmul -> AllGather before + ReduceScatter after")


def test_t5_tp_ep():
    cfg = DistConfig(tp_size=4, ep_size=8)
    exp = run([OpCode.FLASH_ATTN, P7Op("EXPERT_MLP")], cfg)
    assert has_comm(exp, "AllReduce", ParallelStrategy.TP, "after", "flash_attn")
    assert has_comm(exp, "AllToAll",  ParallelStrategy.EP, "before", "expert_mlp")
    assert has_comm(exp, "AllToAll",  ParallelStrategy.EP, "after",  "expert_mlp")
    assert no_comm_for(exp, ParallelStrategy.CP)
    print(f"  {PASS} T5  TP+EP: AllReduce(attn) + AllToAll x2(expert)")


def test_t6_full_transformer():
    cfg = DistConfig(tp_size=2, cp_size=2, ep_size=2, fsdp_size=4)
    exp = run([FUSED_NORM_QKV, FUSED_ATTN_PROJ, FUSED_MLP,
               OpCode.ADD, OpCode.ADD], cfg)

    assert has_comm(exp, "AllReduce",     ParallelStrategy.TP,   "after",  "fused_norm_qkv")
    assert has_comm(exp, "AllGather",     ParallelStrategy.FSDP, "before", "fused_norm_qkv")
    assert has_comm(exp, "AllGather",     ParallelStrategy.CP,   "before", "fused_attn_proj")
    assert has_comm(exp, "AllReduce",     ParallelStrategy.TP,   "after",  "fused_attn_proj")
    assert has_comm(exp, "AllReduce",     ParallelStrategy.TP,   "after",  "fused_mlp")
    assert has_comm(exp, "AllGather",     ParallelStrategy.FSDP, "before", "fused_mlp")

    add_comms = [n for n in comm_nodes(exp)
                 if n.anchor_op == "add" and n.strategy != ParallelStrategy.FSDP]
    assert len(add_comms) == 0

    c = comm_nodes(exp)
    print(f"  {PASS} T6  Full transformer: {len(c)} comm nodes across "
          f"{len([n for n in exp if not isinstance(n, CommNode)])} compute ops")

    print("\n  === Distributed IR ===")
    for n in exp:
        if isinstance(n, CommNode):
            print(f"    |  {n.comm_op:<16} [{n.strategy.value}]  {n.direction}  {n.anchor_op}")
        else:
            base = str(n.opcode).lower().replace("opcode.", "")
            print(f"    o  {base}")


# ── runner ───────────────────────────────────────────────────────────

def main():
    tests = [
        ("T1  TP/matmul",              test_t1_tp_matmul),
        ("T2  CP/flash_attn",          test_t2_cp_flash_attn),
        ("T3  EP/expert_mlp",          test_t3_ep_expert),
        ("T4  FSDP/matmul",            test_t4_fsdp_matmul),
        ("T5  TP+EP combined",         test_t5_tp_ep),
        ("T6  Full transformer block", test_t6_full_transformer),
    ]
    print("=" * 50)
    print("  Phase 8 — Distributed Compiler (6 tests)")
    print("=" * 50)
    passed = failed = 0
    for name, fn in tests:
        print(f"\n-- {name} --")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {FAIL}  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"  Result: {passed}/{passed+failed} passed")
    print(f"{'='*50}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
