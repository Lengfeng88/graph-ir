"""
calibrate_flashinfer.py -- run on the GPU machine, inside the env that has
flashinfer installed (your FlashInfer dev checkout, since you're already
working against VariableBlockSparseAttentionWrapper for #2879).

Measures real latency for the two candidates calibrate.py CANNOT reach
through torch SDPA:
  * paged  -> flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
              (heterogeneous per-sequence kv_lens, real page tables)
  * sparse -> flashinfer.sparse.BlockSparseAttentionWrapper
              (random block-sparse CSR pattern at a target density)

Fits (eff, launch_us) the same way calibrate.py does, then MERGES the
result into calibration.json's "sparse"/"paged" keys -- it does not touch
whatever calibrate.py already wrote for dense/flash.

UNVERIFIED: I could not run this against real flashinfer (no GPU / no
flashinfer in this sandbox). The plan()/run() signatures below are taken
verbatim from https://docs.flashinfer.ai/api/decode.html and
.../api/sparse.html (checked 2026-09-28), but if your dev checkout is
ahead of those docs, `plan()`'s kwarg names are the most likely mismatch --
if it throws a TypeError there, check `help(wrapper.plan)` first.
"""
import json, math, os, random
import torch
from triton.testing import do_bench
from cost_model import (AttentionWorkload, AttentionCostModel, Calibration,
                        HARDWARE, CANDIDATES, REGIMES)

try:
    import flashinfer
except ImportError:
    raise SystemExit("flashinfer not importable in this env -- activate the "
                     "env / dev checkout you use for the FlashInfer PRs")

WORKSPACE_MB = 128

# ── paged decode ──────────────────────────────────────────────────

def build_page_table(kv_lens, page_size, device="cuda"):
    """CSR-style page table for BatchDecodeWithPagedKVCacheWrapper: pages
    are handed out sequentially per sequence (fine for a latency benchmark;
    a real allocator's page IDs wouldn't be contiguous, but page ID order
    doesn't affect this kernel's cost)."""
    indptr, last_len, n_pages = [0], [], 0
    for L in kv_lens:
        pages = math.ceil(L / page_size)
        n_pages += pages
        indptr.append(n_pages)
        last_len.append((L - 1) % page_size + 1)
    return (torch.tensor(indptr, dtype=torch.int32, device=device),
           torch.arange(n_pages, dtype=torch.int32, device=device),
           torch.tensor(last_len, dtype=torch.int32, device=device), n_pages)

def measure_paged(w: AttentionWorkload, device="cuda"):
    assert w.q_len == 1, "paged decode benchmark assumes q_len==1"
    H, D, pg = w.num_heads, w.head_dim, w.page_size
    dtype = torch.float16 if w.dtype in ("fp16",) else torch.bfloat16
    indptr, indices, last_len, n_pages = build_page_table(w.kv_lens, pg, device=device)
    kv_cache = torch.randn(n_pages, 2, pg, H, D, dtype=dtype, device=device)
    q = torch.randn(w.batch, H, D, dtype=dtype, device=device)

    ws = torch.empty(WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    wrapper.plan(indptr, indices, last_len, H, H, D, pg,
                pos_encoding_mode="NONE", data_type=dtype)
    return do_bench(lambda: wrapper.run(q, kv_cache))

PAGED_SCENARIOS = [
    # H=8,D=64 to match phase10 sweep's decode shapes; (batch,kv_len) pairs
    # deliberately disjoint from the sweep's (1/16/64, 128/2048/8192) grid so
    # the sweep stays an out-of-sample agreement check, not a self-fit.
    ("paged decode B4 kv256",    AttentionWorkload.decode([256] * 4, 8, 64)),
    ("paged decode B4 kv1024",   AttentionWorkload.decode([1024] * 4, 8, 64)),
    ("paged decode B4 kv4096",   AttentionWorkload.decode([4096] * 4, 8, 64)),
    ("paged decode B4 kv16384",  AttentionWorkload.decode([16384] * 4, 8, 64)),
    ("paged decode B32 kv256",   AttentionWorkload.decode([256] * 32, 8, 64)),
    ("paged decode B32 kv1024",  AttentionWorkload.decode([1024] * 32, 8, 64)),
    ("paged decode B32 kv4096",  AttentionWorkload.decode([4096] * 32, 8, 64)),
    ("paged decode B32 kv16384", AttentionWorkload.decode([16384] * 32, 8, 64)),
]

# ── block-sparse prefill ──────────────────────────────────────────

def random_block_csr(m_blocks, n_blocks, density, causal, seed, device="cuda"):
    rng = random.Random(seed)
    rows = []
    for r in range(m_blocks):
        cols = range(min(r + 1, n_blocks)) if causal else range(n_blocks)
        cols = list(cols)
        k = max(1, round(density * len(cols)))
        rows.append(sorted(rng.sample(cols, min(k, len(cols)))))
    indptr = [0]
    indices = []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    return (torch.tensor(indptr, dtype=torch.int32, device=device),
           torch.tensor(indices, dtype=torch.int32, device=device))

def measure_sparse(w: AttentionWorkload, seed=0, device="cuda"):
    assert w.q_len == w.kv_len, "sparse benchmark assumes prefill (q_len==kv_len)"
    assert w.batch == 1, "BlockSparseAttentionWrapper takes one (M,N) sequence, not a batch"
    H, D, blk = w.num_heads, w.head_dim, w.sparse_block_size
    M = N = w.q_len
    assert M % blk == 0, f"seq_len {M} must be a multiple of block size {blk}"
    m_blocks = n_blocks = M // blk
    dtype = torch.float16 if w.dtype in ("fp16",) else torch.bfloat16

    indptr, indices = random_block_csr(m_blocks, n_blocks, w.sparse_density, w.causal, seed, device=device)
    ws = torch.empty(WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BlockSparseAttentionWrapper(ws)
    wrapper.plan(indptr, indices, M, N, blk, blk, H, H, D)

    q = torch.randn(M, H, D, dtype=dtype, device=device)
    k = torch.randn(N, H, D, dtype=dtype, device=device)
    v = torch.randn(N, H, D, dtype=dtype, device=device)
    return do_bench(lambda: wrapper.run(q, k, v))

SPARSE_SCENARIOS = [
    ("sparse N8192 density0.05",  AttentionWorkload.prefill(1, 8192, 8, 64, sparse_density=0.05, allow_approx=True)),
    ("sparse N8192 density0.02",  AttentionWorkload.prefill(1, 8192, 8, 64, sparse_density=0.02, allow_approx=True)),
    ("sparse N16384 density0.02", AttentionWorkload.prefill(1, 16384, 8, 64, sparse_density=0.02, allow_approx=True)),
    ("sparse N4096 density0.1",   AttentionWorkload.prefill(1, 4096, 8, 64, sparse_density=0.1, allow_approx=True)),
]

# ── fit + merge (same weighted-LS as calibrate.py) ────────────────

def fit_launch_eff(pts):
    w = [1 / real**2 for _, _, real in pts]
    sw = sum(w); sx = sum(wi * i for wi, (_, i, _) in zip(w, pts))
    sy = sum(wi * real for wi, (_, _, real) in zip(w, pts))
    sxx = sum(wi * i * i for wi, (_, i, _) in zip(w, pts))
    sxy = sum(wi * i * real for wi, (_, i, real) in zip(w, pts))
    b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
    a = (sy - b * sx) / sw
    if a < 0 or len(pts) < 2:
        a, b = 0.0, sxy / sxx
    return max(a, 0.0) * 1000, 1 / b

def main():
    path = "calibration.json"
    out = json.load(open(path)) if os.path.exists(path) else {"hardware": {}, "candidates": {}}
    hw = HARDWARE["rtx4080"]
    if out.get("hardware"):
        from dataclasses import replace
        hw = replace(hw, **out["hardware"])
    ideal = Calibration(eff={c: {r: 1.0 for r in REGIMES} for c in CANDIDATES},
                        launch_us={c: {r: 0.0 for r in REGIMES} for c in CANDIDATES})
    model = AttentionCostModel(hw, ideal)

    print("── paged (decode) ──")
    pts = []
    for name, w in PAGED_SCENARIOS:
        ideal_ms = {e.candidate: e for e in model.evaluate(w).estimates}["paged"].latency_ms
        real = measure_paged(w)
        pts.append((name, ideal_ms, real))
        print(f"  {name:<32} ideal={ideal_ms:8.3f}ms real={real:8.3f}ms real/ideal={real/ideal_ms:5.2f}")
    launch, eff = fit_launch_eff(pts)
    out["candidates"].setdefault("paged", {})["decode"] = {"eff": round(eff, 4), "launch_us": round(launch, 2)}
    print(f"  fitted: eff={eff:.3f} launch={launch:.1f}us\n")

    print("── sparse (prefill) ──")
    pts = []
    for name, w in SPARSE_SCENARIOS:
        ideal_ms = {e.candidate: e for e in model.evaluate(w).estimates}["sparse"].latency_ms
        real = measure_sparse(w)
        pts.append((name, ideal_ms, real))
        print(f"  {name:<32} ideal={ideal_ms:8.3f}ms real={real:8.3f}ms real/ideal={real/ideal_ms:5.2f}")
    launch, eff = fit_launch_eff(pts)
    out["candidates"].setdefault("sparse", {})["prefill"] = {"eff": round(eff, 4), "launch_us": round(launch, 2)}
    print(f"  fitted: eff={eff:.3f} launch={launch:.1f}us")

    json.dump(out, open(path, "w"), indent=2)
    print(f"\nmerged into {path} (dense/flash entries, if present, are untouched)")

if __name__ == "__main__":
    main()
