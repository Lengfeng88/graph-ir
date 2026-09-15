"""
torch_fx.py -- turn a PyTorch model into a torch.fx.GraphModule.

This layer's only job is "call torch.fx" -- it doesn't try to
understand semantics. Keeping this separate matters: after
trace_model() you still have PyTorch's own graph (nodes are
placeholder/call_module/call_function/output); actually translating
that into our own IR is importer.py's job, one layer up.

Why symbolic_trace instead of torch.export / make_fx: symbolic_trace
is the simplest option and is enough for "static, no control flow"
models (a DAG like an attention block is fine). Once a model has
if/for branches that depend on tensor values, symbolic_trace will
fail -- that's a problem for later (probably torch.export), but for
now the simplest path is the one worth getting working first.
"""

from __future__ import annotations
import torch
import torch.fx as fx


def trace_model(model: torch.nn.Module, example_inputs) -> fx.GraphModule:
    """
    Symbolically trace the model and return a torch.fx.GraphModule.

    example_inputs isn't actually used by symbolic_trace (unlike
    torch.jit.trace, it doesn't need a real forward pass), but the
    parameter is kept -- once we switch to a trace backend that does
    need a real forward pass (e.g. torch.export), this will be used,
    and callers above this layer won't need to change.
    """
    model.eval()
    graph_module = fx.symbolic_trace(model)
    return graph_module


def dump_fx_graph(graph_module: fx.GraphModule) -> None:
    """Debug helper: print the raw FX node table -- handy for looking
    up each node's opcode/target/args while writing the importer."""
    graph_module.graph.print_tabular()
