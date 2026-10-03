"""
Patches pass4_attention_rewrite.py to call pass4_bridge.decide_dense_or_flash()
instead of unconditionally emitting FLASH_ATTN. Safety: each replacement
requires an EXACT, UNIQUE match in the target file; if the file's actual
text differs from what this script expects (different whitespace, already
edited, etc.), it reports which piece didn't match and writes NOTHING --
never partially patches.

Usage:
    python3 apply_pass4_patch.py pass4_attention_rewrite.py
Writes a .bak of the original before overwriting, and prints a diff.
"""
import sys, shutil, difflib

PATCHES = [
    # 1. import the bridge, anchored right before the class it modifies
    (
        'class AttentionRewriter:\n    """\n    Given a confirmed AttnMatch, rewrites the graph:',
        'from pass4_bridge import decide_dense_or_flash\n\n\n'
        'class AttentionRewriter:\n    """\n    Given a confirmed AttnMatch, rewrites the graph:',
    ),
    # 2. insert the dense/flash decision at the top of rewrite(), before
    #    it unconditionally starts building a flash_op
    (
        '    def rewrite(self, match: AttnMatch, graph: Graph) -> Op:\n'
        '        Q, K, V = match.Q, match.K, match.V\n'
        '\n'
        '        # build result name\n',
        '    def rewrite(self, match: AttnMatch, graph: Graph) -> Optional[Op]:\n'
        '        Q, K, V = match.Q, match.K, match.V\n'
        '\n'
        '        decision = decide_dense_or_flash(\n'
        '            batch=Q.type.shape[0].static_value,\n'
        '            seq_len=Q.type.shape[1].static_value,\n'
        '            head_dim=match.d_head,\n'
        '        )\n'
        '        if decision["variant"] == "dense":\n'
        '            return None   # leave the original qk_mm/scale/softmax/av_mm chain untouched\n'
        '\n'
        '        # build result name\n',
    ),
    # 3. run() must handle rewrite() now returning None
    (
        '            flash = self.rewriter.rewrite(m, graph)\n'
        '            self._log(\n'
        '                f"  -> {flash.result.name}: {flash.result.type}"\n'
        '                f"  [flash_attn]"\n'
        '            )\n'
        '\n'
        '            consumed |= ids\n'
        '            rewrites += 1\n',
        '            flash = self.rewriter.rewrite(m, graph)\n'
        '            if flash is None:\n'
        '                self._log(f"  -> kept dense (cost model): "\n'
        '                         f"QK={m.ops[\'qk_mm\'].result.name}")\n'
        '                continue\n'
        '            self._log(\n'
        '                f"  -> {flash.result.name}: {flash.result.type}"\n'
        '                f"  [flash_attn]"\n'
        '            )\n'
        '\n'
        '            consumed |= ids\n'
        '            rewrites += 1\n',
    ),
]

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 apply_pass4_patch.py pass4_attention_rewrite.py")
    path = sys.argv[1]
    original = open(path).read()
    text = original

    for i, (old, new) in enumerate(PATCHES, 1):
        count = text.count(old)
        if count != 1:
            sys.exit(f"ABORT: patch {i}/{len(PATCHES)} matched {count} times "
                     f"(need exactly 1) -- file differs from what this script "
                     f"expects. No changes written. Paste the actual "
                     f"surrounding lines and I'll fix the patch.")
        text = text.replace(old, new)

    shutil.copy(path, path + ".bak")
    with open(path, "w") as f:
        f.write(text)

    print(f"patched {path} (backup at {path}.bak)\n")
    diff = difflib.unified_diff(original.splitlines(keepends=True),
                                text.splitlines(keepends=True),
                                fromfile=path + ".bak", tofile=path)
    print("".join(diff))

if __name__ == "__main__":
    main()
