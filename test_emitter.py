import torch
import torch.nn.functional as F

from emitter import Node, CodeEmitter


def test_add() -> None:
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")

    node_x = Node(op="input", name="x", value=x)
    node_y = Node(op="input", name="y", value=y)
    node_out = Node(op="add", name="out", operands=[node_x, node_y])

    result = CodeEmitter().run(node_out)
    expected = x + y
    torch.testing.assert_close(result, expected)
    print("add: OK")


def test_gelu() -> None:
    x = torch.randn(4096, device="cuda")

    node_x = Node(op="input", name="x", value=x)
    node_out = Node(op="gelu", name="out", operands=[node_x])

    result = CodeEmitter().run(node_out)
    expected = F.gelu(x, approximate="tanh")
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
    print("gelu: OK")


def test_matmul() -> None:
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)

    node_a = Node(op="input", name="a", value=a)
    node_b = Node(op="input", name="b", value=b)
    node_out = Node(op="matmul", name="out", operands=[node_a, node_b])

    result = CodeEmitter().run(node_out)
    # reference computed in fp32 then cast down, to isolate our kernel's
    # error from fp16 accumulation error in a naive torch.matmul baseline
    expected = (a.float() @ b.float()).half()
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    print("matmul: OK")


def test_fused_matmul_bias_gelu() -> None:
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    bias = torch.randn(256, device="cuda", dtype=torch.float16)

    node_a = Node(op="input", name="a", value=a)
    node_b = Node(op="input", name="b", value=b)
    node_bias = Node(op="input", name="bias", value=bias)
    node_out = Node(
        op="fused_matmul_bias_gelu",
        name="out",
        operands=[node_a, node_b, node_bias],
    )

    result = CodeEmitter().run(node_out)
    expected = F.gelu(
        (a.float() @ b.float()) + bias.float(), approximate="tanh"
    ).half()
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)
    print("fused_matmul_bias_gelu: OK")


def test_attention() -> None:
    M, N, d = 128, 128, 64
    q = torch.randn(M, d, device="cuda", dtype=torch.float16)
    k = torch.randn(N, d, device="cuda", dtype=torch.float16)
    v = torch.randn(N, d, device="cuda", dtype=torch.float16)

    node_q = Node(op="input", name="q", value=q)
    node_k = Node(op="input", name="k", value=k)
    node_v = Node(op="input", name="v", value=v)
    node_out = Node(op="attention", name="out", operands=[node_q, node_k, node_v])

    result = CodeEmitter().run(node_out)

    scale = 1.0 / (d ** 0.5)
    scores = (q.float() @ k.float().t()) * scale
    probs = torch.softmax(scores, dim=-1)
    expected = (probs @ v.float()).half()

    torch.testing.assert_close(result, expected, atol=3e-2, rtol=3e-2)
    print("attention: OK")


if __name__ == "__main__":
    test_add()
    test_gelu()
    test_matmul()
    test_fused_matmul_bias_gelu()
    test_attention()
