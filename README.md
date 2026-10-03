# graph-ir

An AI compiler built from scratch — no MLIR, no LLVM. Custom Graph IR,
progressing through codegen, distributed passes, a runtime layer, and
auto-tuning.

## Phase docs

See [`docs/phases/`](docs/phases/) for per-phase writeups:

- Phase 1 — Graph IR + Frontend ([`phase1.md`](docs/phases/phase1.md))
- Phase 2 — CUDA/Triton kernel exploration ([`README_phase2.md`](docs/phases/README_phase2.md))
- Phase 3 — Codegen + fusion pass ([`phase3.md`](docs/phases/phase3.md), [`README_phase3.md`](docs/phases/README_phase3.md))
- Phase 4 — Attention rewrite pass ([`phase4.md`](docs/phases/phase4.md))
- Phase 5 — Sparse rewrite + emitter ([`PHASE5_REPORT.md`](docs/phases/PHASE5_REPORT.md))
- Phase 6 — Cost model, calibration, autotuning ([`phase6.md`](docs/phases/phase6.md))
- Phase 7 — Whole-graph optimization ([`PHASE7_README.md`](docs/phases/PHASE7_README.md))
- Phase 8 — Distributed partitioning + comm passes ([`PHASE8_README.md`](docs/phases/PHASE8_README.md))
- Phase 9 — Execution runtime ([`PHASE9_README.md`](docs/phases/PHASE9_README.md))
- Phase 10 — Auto-search sweep + cost model validation (`phase10/`)
