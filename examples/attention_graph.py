"""
Build the attention diagram from the original prompt using the minimal
IR in ir/:

    x
    |- Linear(Q) --,
    |- Linear(K) --+-- MatMul -> Scale -> Softmax --,
    `- Linear(V) --+-------------------------------- MatMul -> Output
                   (Q,K meet at the first MatMul; V meets at the second)

This is not a plain chain -- x fans out into three (Q/K/V), the first
MatMul merges Q and K, and the second MatMul merges the softmax result
with V. This is a real DAG, which exercises Use-Def / DAG structure
better than a plain chain would.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir import Graph, OpKind


def build_attention_graph() -> Graph:
    g = Graph(name="attention")

    x = g.add_input(shape=(1, 128, 512), name="%x")

    q = g.add_node(OpKind.LINEAR, [x], out_name="%q")
    k = g.add_node(OpKind.LINEAR, [x], out_name="%k")
    v = g.add_node(OpKind.LINEAR, [x], out_name="%v")

    qk = g.add_node(OpKind.MATMUL, [q, k], out_name="%qk")
    scaled = g.add_node(OpKind.SCALE, [qk], attrs={"factor": "1/sqrt(d_k)"}, out_name="%scaled")
    attn = g.add_node(OpKind.SOFTMAX, [scaled], out_name="%attn")

    out = g.add_node(OpKind.MATMUL, [attn, v], out_name="%out")
    g.set_output(out)

    return g


if __name__ == "__main__":
    g = build_attention_graph()

    print("===== graph.print()  (full SSA text form) =====")
    g.print()

    print("\n===== graph.print_flow()  (simplified arrow view) =====")
    g.print_flow()

    print("\n===== manual Use-Def walk =====")
    last_node = g.nodes[-1]
    print(f"{last_node} depends on:")
    for d in last_node.defs():
        print(f"  - {d}")

    x_input = g.inputs[0]
    print(f"\n{x_input.name} is used by:")
    for u in x_input.users:
        print(f"  - {u}")
