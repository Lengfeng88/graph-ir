# graph-ir

An AI compiler built from scratch — no MLIR, no LLVM. Custom Graph IR,
progressing through codegen, distributed passes, a runtime layer, and
auto-tuning.

## Progress

- [x] Phase 1 — Graph IR (SSA, Use-Def, DAG) + Frontend (PyTorch -> torch.fx -> Graph IR)
- [ ] Phase 2 — Optimization passes (constant folding, DCE, operator fusion, attention rewrite)
- [ ] Phase 3 — MLIR dialect (Attention Dialect + a Toy Dialect as a warm-up)
- [ ] Phase 4 — Triton codegen (MatMul / Softmax / LayerNorm / Flash Attention)
- [ ] Phase 5 — TBD
- [ ] Phase 6 — Cost model (roofline-based; decision layer over Dense/Flash/Sparse/Paged)
- [ ] Phase 7 — TBD
- [ ] Phase 8 — TBD
- [ ] Phase 9 — TBD
- [ ] Phase 10 — Auto search / auto-tuning

See [docs/roadmap.md](docs/roadmap.md) for the detailed plan.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 mini_compiler.py examples/model.py
```
