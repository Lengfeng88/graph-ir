#!/usr/bin/env python3
"""
mini_compiler.py -- the acceptance-test entry point for Phase 1 Frontend.

Usage:
    python3 mini_compiler.py path/to/model.py

The flow maps directly onto the frontend's three layers:
    PyTorch model  --[torch_fx.trace_model]-->  fx.GraphModule
                   --[importer.import_from_fx]--> our own ir.Graph
                   --[graph.print_ops]-->        print the op list
"""

import sys
import os
import importlib.util
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontend import trace_model, import_from_fx


def load_model_module(model_path: str):
    """Dynamically import model.py as a module, without requiring it
    to live inside a package."""
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(prog="mini-compiler")
    parser.add_argument("model_file", help="a Python file defining get_model()")
    parser.add_argument("--dump-fx", action="store_true",
                         help="also print the raw FX graph -- useful when debugging the importer")
    args = parser.parse_args()

    module = load_model_module(args.model_file)
    if not hasattr(module, "get_model"):
        print(f"error: no get_model() found in {args.model_file}", file=sys.stderr)
        sys.exit(1)

    model, example_inputs = module.get_model()

    fx_graph_module = trace_model(model, example_inputs)
    if args.dump_fx:
        fx_graph_module.graph.print_tabular()
        print()

    graph = import_from_fx(fx_graph_module, graph_name=type(model).__name__)
    graph.print_ops()


if __name__ == "__main__":
    main()
