"""
Adapter: wraps the OFFICIAL triton-lang/triton fused-attention tutorial
kernel (06-fused-attention.py, tag v3.2.0 -- matching the installed
triton==3.2.0 -- downloaded from
raw.githubusercontent.com/triton-lang/triton/v3.2.0/python/tutorials/)
into autotune.py's run(q,k,v,causal,cfg) interface.

Requires 06-fused-attention.py (the v3.2.0 version) to sit next to this file.

How the override works: _attn_fwd is wrapped by
@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"]) in
the tutorial file, and forward(ctx, q, k, v, causal, sm_scale) does NOT
expose BLOCK_M/BLOCK_N/num_warps/num_stages as call arguments -- they're
entirely internal to that autotune wrapper's own config search. So to
force a specific TileConfig we temporarily replace _attn_fwd.configs with
a single-element list (v3.2.0's own configs carry no pre_hook, so ours
doesn't either) and clear _attn_fwd.cache so the autotuner can't replay a
previously-cached winner instead of running our forced config.

KNOWN RISK, not yet exercised: the tutorial's own `keep(conf)` filters out
some (BLOCK_M, BLOCK_N, num_warps, num_stages) combinations before they
ever reach @triton.autotune -- presumably because they're invalid for this
kernel (e.g. blow SRAM, illegal shapes). DEFAULT_SEARCH_SPACE in
autotune.py was NOT built through that same `keep()` filter, so it's
possible some of our forced configs are ones the tutorial authors
deliberately excluded. Failures from that should surface as normal
exceptions caught by autotune.py's search() loop and shown as
per-config errors -- but if instead the run dies hard (a raw "CUDA
error" rather than a clean Python traceback, or every config AFTER a
specific one starts failing too), that's a poisoned CUDA context from
one bad config, not a new bug in each subsequent one; re-run excluding
whatever config came right before things went bad.

This mutates tutorial module state for the duration of the call and is
NOT thread-safe -- fine for a single-threaded autotune sweep only.
"""
import importlib.util
import math
import threading

_LOCK = threading.Lock()   # the .configs/.cache monkeypatch below is global
                           # module state -- serialize calls through it

def _load_tutorial(path="06-fused-attention.py"):
    spec = importlib.util.spec_from_file_location("fused_attention_tutorial", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_tutorial = _load_tutorial()

def run(q, k, v, causal, cfg):
    import triton
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    forced = triton.Config(
        {"BLOCK_M": cfg.block_m, "BLOCK_N": cfg.block_n},
        num_stages=cfg.num_stages, num_warps=cfg.num_warps,
    )
    with _LOCK:
        orig_configs = _tutorial._attn_fwd.configs
        orig_cache = dict(_tutorial._attn_fwd.cache)
        try:
            _tutorial._attn_fwd.configs = [forced]
            _tutorial._attn_fwd.cache.clear()
            out = _tutorial._attention.apply(q, k, v, causal, sm_scale)
        finally:
            _tutorial._attn_fwd.configs = orig_configs
            _tutorial._attn_fwd.cache.clear()
            _tutorial._attn_fwd.cache.update(orig_cache)
    return out
