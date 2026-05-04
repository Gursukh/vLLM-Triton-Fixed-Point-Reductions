#!/usr/bin/env python3
"""Bit-exact check across batch compositions: fxpr_vllm vs FLASH_ATTN baseline."""

from __future__ import annotations

import gc
import os
import struct

TEST_PROMPTS = [
    ("short",  "What is 17 * 23?"),                    # decode-dominated
    ("medium", "The capital of France is " * 50),      # ~250 tok
    ("long",   "The capital of France is " * 200),     # ~1000 tok
]
HEAVY_FILLERS = [
    "Once upon a time, in a land far far away, there lived a wizard who " * 80,
    "It was a dark and stormy night, the wind howled through the trees "  * 80,
    "All work and no play makes for a very boring afternoon, said the "    * 80,
    "The quick brown fox jumps over the lazy dog, again and again and "    * 80,
    "In the beginning was the Word, and the Word was with everyone, and "  * 80,
    "A long time ago in a galaxy far, far away, there were many stars "    * 80,
    "Two roads diverged in a yellow wood, and sorry I could not travel "   * 80,
]
BATCH_SIZES = [1, 4, 8]


def bits(x):
    return struct.pack("<d", float(x))


def extract(req):
    co = req.outputs[0]
    prompt = [
        None if p is None else sorted((int(t), float(lp.logprob)) for t, lp in p.items())
        for p in req.prompt_logprobs
    ]
    decode = [
        sorted((int(t), float(lp.logprob)) for t, lp in step.items())
        for step in co.logprobs
    ]
    return prompt, list(co.token_ids), decode


def diff_pairs(ref, cand):
    if [t for t, _ in ref] != [t for t, _ in cand]:
        return "top-K token set differs"
    n = sum(1 for (_, r), (_, c) in zip(ref, cand) if bits(r) != bits(c))
    if n:
        d = max(abs(r - c) for (_, r), (_, c) in zip(ref, cand))
        return f"{n}/{len(ref)} top-K logprobs differ; max |delta|={d:.3e}"
    return None


def compare(ref, cand, label):
    rp, rt, rd = ref
    cp, ct, cd = cand
    out = []
    if len(rp) != len(cp):
        out.append(f"{label}: prompt_logprobs length {len(rp)} vs {len(cp)}")
    for i, (a, b) in enumerate(zip(rp, cp)):
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            out.append(f"{label}: prompt_logprob@{i}: one side None")
            continue
        d = diff_pairs(a, b)
        if d:
            out.append(f"{label}: prompt_logprob@{i}: {d}")
    if rt != ct:
        for i, (a, b) in enumerate(zip(rt, ct)):
            if a != b:
                out.append(f"{label}: decode_token@{i}: ref={a} cand={b}")
                break
        if len(rt) != len(ct):
            out.append(f"{label}: decode_token length {len(rt)} vs {len(ct)}")
    for i, (a, b) in enumerate(zip(rd, cd)):
        d = diff_pairs(a, b)
        if d:
            out.append(f"{label}: decode_logprob@{i}: {d}")
    return out


COMMON_LLM_KWARGS = dict(
    model="Qwen/Qwen3-0.6B",
    dtype="float16",
    enforce_eager=True,
    enable_prefix_caching=False,
    max_model_len=4096,
    max_num_seqs=max(BATCH_SIZES),
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.70,  
    seed=0,
)


def shutdown(llm):
    # v1 leaves engine subprocesses around; the second LLM OOMs without this.
    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_matrix(llm):
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=32,
        logprobs=20, prompt_logprobs=5, seed=0,
    )
    results = {}
    for label, prompt in TEST_PROMPTS:
        for bs in BATCH_SIZES:
            batch = [prompt] + HEAVY_FILLERS[: bs - 1]
            outs = llm.generate(batch, sp, use_tqdm=False)
            results[(label, bs)] = extract(outs[0])

    divergences = []
    for label, _ in TEST_PROMPTS:
        ref = results[(label, 1)]
        for bs in BATCH_SIZES:
            if bs == 1:
                continue
            divergences.extend(compare(ref, results[(label, bs)], f"{label}@BS={bs}"))
    return divergences


def main():
    # Baseline first: register() mutates vLLM state irreversibly. The plugin
    # entry point auto-runs on every LLM construction, so we set
    # FXPR_DISABLE_PATCHES to keep this arm vanilla.
    os.environ["FXPR_DISABLE_PATCHES"] = "1"
    from vllm import LLM
    baseline = LLM(attention_backend="FLASH_ATTN", **COMMON_LLM_KWARGS)
    try:
        baseline_div = run_matrix(baseline)
    finally:
        shutdown(baseline)
    os.environ.pop("FXPR_DISABLE_PATCHES", None)

    from fxpr_vllm.register import register
    register()
    fxpr = LLM(
        quantization="fixed_point_det",
        attention_backend="CUSTOM",
        **COMMON_LLM_KWARGS,
    )
    try:
        fxpr_div = run_matrix(fxpr)
    finally:
        shutdown(fxpr)

    n_cells = len(TEST_PROMPTS) * (len(BATCH_SIZES) - 1)
    print()
    print(f"FLASH_ATTN baseline: {len(baseline_div):>4} divergence(s) / {n_cells} cells (informational)")
    print(f"fxpr_vllm:           {len(fxpr_div):>4} divergence(s) / {n_cells} cells (must be 0)")
    if baseline_div:
        print("\nFLASH_ATTN baseline diverged. This is the problem fxpr_vllm fixes.")
        for d in baseline_div[:10]:
            print(f"  [baseline] {d}")
        if len(baseline_div) > 10:
            print(f"  ... and {len(baseline_div) - 10} more")
    else:
        print("\nFLASH_ATTN baseline was bit-equal on this hardware; cuBLAS picked "
              "the same kernel across BS. fxpr still gives invariance by construction.")
    if fxpr_div:
        print("\nfxpr_vllm divergences:")
        for d in fxpr_div[:30]:
            print(f"  [fxpr] {d}")
        if len(fxpr_div) > 30:
            print(f"  ... and {len(fxpr_div) - 30} more")

    print()
    print("PASS" if not fxpr_div else "FAIL")
    return 0 if not fxpr_div else 1


if __name__ == "__main__":
    raise SystemExit(main())
