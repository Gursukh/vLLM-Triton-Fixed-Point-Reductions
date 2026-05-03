from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks import ncu, ops


class _L2Flusher:
    def __init__(self):
        size = getattr(torch.cuda.get_device_properties(0),
                       "L2_cache_size", None) or (40 << 20)
        self.buf = torch.empty(size + (1 << 20), device="cuda",
                               dtype=torch.int8)

    def flush(self):
        self.buf.zero_()


def _percentile(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def time_kernel(fn: Callable, *, warmup: int, iters: int,
                cache_mode: str, flusher: _L2Flusher) -> dict[str, float]:
    flush = flusher.flush if cache_mode == "cold" else (lambda: None)

    for _ in range(warmup):
        flush()
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))

    s = sorted(samples)
    return {
        "n_iters": len(samples),
        "min_ms": s[0],
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(s, 0.95),
    }


def _measure_hbm_gbps(*, size_bytes: int = 256 << 20, iters: int = 25) -> float:
    n = size_bytes // 4
    src = torch.empty(n, device="cuda", dtype=torch.float32).normal_()
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record(); dst.copy_(src); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return (2.0 * size_bytes) / (best * 1e-3) / 1e9


def _measure_sgemm_tflops(*, n: int = 4096, iters: int = 10) -> float:
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for _ in range(3):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(iters):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record(); torch.matmul(a, b); e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return (2.0 * n * n * n) / (best * 1e-3) / 1e12


def measure_peaks(*, force: bool = False) -> dict[str, float]:
    p = torch.cuda.get_device_properties(0)
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME",
                                    Path.home() / ".cache")) / "fxpr-bench"
    cache_dir.mkdir(parents=True, exist_ok=True)
    short = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.name)
    cache_file = cache_dir / f"peaks_{short}_cc{p.major}.{p.minor}.json"
    if cache_file.exists() and not force:
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            pass

    out = {
        "hbm_gbps": _measure_hbm_gbps(),
        "sgemm_tflops": _measure_sgemm_tflops(),
    }
    cache_file.write_text(json.dumps(out, indent=2))
    return out


def env_columns() -> dict[str, str]:
    p = torch.cuda.get_device_properties(0)
    drv = "n/a"
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            text=True, timeout=5).strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    sha = "n/a"
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, timeout=5,
            cwd=Path(__file__).parent.parent).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return {
        "gpu_name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "cuda_version": torch.version.cuda or "n/a",
        "torch_version": torch.__version__,
        "driver_version": drv,
        "commit_sha": sha,
    }


def roofline_columns(flops: int, bytes_: int, median_ms: float,
                     peak_gflops: float, peak_gbps: float) -> dict[str, float | str]:
    sec = median_ms * 1e-3
    gflops = flops / sec / 1e9
    gbps = bytes_ / sec / 1e9
    ai = flops / bytes_ if bytes_ else 0.0
    roof = min(peak_gflops, peak_gbps * ai)
    pct = (gflops / roof * 100.0) if roof else 0.0
    ridge_ai = (peak_gflops / peak_gbps) if peak_gbps else 0.0
    bound = "compute" if ai >= ridge_ai else "memory"
    return {
        "gflops": gflops,
        "gbps": gbps,
        "arith_intensity": ai,
        "roof_gflops": roof,
        "pct_of_roof": pct,
        "bound": bound,
    }


def _parse_configs(text: str) -> list[dict[str, int]]:
    out = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        d: dict[str, int] = {}
        for kv in chunk.split(","):
            k, v = kv.split("=")
            d[k.strip()] = int(v)
        out.append(d)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ops", default="all",
                   help=f"Comma list or 'all'. Available: {','.join(ops.list_ops())}")
    p.add_argument("--configs", default="int_bits=32",
                   help="Semicolon-separated, e.g. 'int_bits=32;int_bits=64'")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--ncu", action="store_true")
    p.add_argument("--ncu-iters", type=int, default=3)
    p.add_argument("--out", default="outputs/bench")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    op_names = ops.list_ops() if args.ops == "all" else [
        s.strip() for s in args.ops.split(",") if s.strip()]
    for n in op_names:
        if n not in ops.OPS:
            print(f"unknown op: {n}", file=sys.stderr)
            return 2
    configs = _parse_configs(args.configs)

    peaks = measure_peaks()
    peak_gflops = peaks["sgemm_tflops"] * 1000.0
    peak_gbps = peaks["hbm_gbps"]
    print(f"hbm={peak_gbps:.0f} GB/s  sgemm={peaks['sgemm_tflops']:.1f} TFLOPS",
          flush=True)

    env = env_columns()
    flusher = _L2Flusher()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_gpu = re.sub(r"[^A-Za-z0-9]+", "", env["gpu_name"]).lower()[:16]
    csv_path = out_root / f"bench_{ts}_{short_gpu}.csv"

    fieldnames = [
        "op", "shape_label", "shape", "config",
        "n_iters", "min_ms", "median_ms", "p95_ms",
        "flops", "bytes", "gflops", "gbps", "arith_intensity",
        "peak_gflops", "peak_gbps", "roof_gflops", "pct_of_roof", "bound",
        "cache_mode",
        *env.keys(),
        "sm_active_pct", "achieved_occupancy_pct", "dram_bytes",
        "l1_hit_pct", "l2_hit_pct", "ncu_status",
    ]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for op_name in op_names:
            spec = ops.OPS[op_name]
            for shape in spec["shapes"]:
                for cfg in configs:
                    row = _run_one(op_name, spec, shape, cfg,
                                   warmup=args.warmup, iters=args.iters,
                                   ncu_iters=args.ncu_iters,
                                   want_ncu=args.ncu,
                                   peak_gflops=peak_gflops,
                                   peak_gbps=peak_gbps,
                                   flusher=flusher, env=env)
                    w.writerow(row)
                    f.flush()
                    print(_short_summary(row), flush=True)

    print(f"\nCSV: {csv_path.resolve()}")
    return 0


def _run_one(op_name, spec, shape, cfg, *, warmup, iters, ncu_iters, want_ncu,
             peak_gflops, peak_gbps, flusher, env) -> dict[str, Any]:
    fn, flops, bytes_ = ops.build_one(op_name, shape, cfg)
    t = time_kernel(fn, warmup=warmup, iters=iters,
                    cache_mode=spec["cache_mode"], flusher=flusher)
    rl = roofline_columns(flops, bytes_, t["median_ms"], peak_gflops, peak_gbps)

    ncu_metrics: dict[str, Any] = {
        "sm_active_pct": "", "achieved_occupancy_pct": "",
        "dram_bytes": "", "l1_hit_pct": "", "l2_hit_pct": "",
        "ncu_status": "skipped",
    }
    if want_ncu:
        result = ncu.run(op_name=op_name, kernel_regex=spec["ncu_regex"],
                         shape=shape, config=cfg,
                         warmup=2, iters=ncu_iters)
        if result is None:
            ncu_metrics["ncu_status"] = "ncu-not-available"
        else:
            ncu_metrics["ncu_status"] = result.pop("ncu_status", "ok")
            for k, v in result.items():
                ncu_metrics[k] = v

    return {
        "op": op_name,
        "shape_label": shape.get("label", ""),
        "shape": json.dumps(shape, separators=(",", ":")),
        "config": json.dumps(cfg, separators=(",", ":")),
        "flops": flops,
        "bytes": bytes_,
        "peak_gflops": peak_gflops,
        "peak_gbps": peak_gbps,
        "cache_mode": spec["cache_mode"],
        **t, **rl, **env, **ncu_metrics,
    }


def _short_summary(r: dict[str, Any]) -> str:
    return (f"  {r['op']:<20} {r['shape_label']:<16} "
            f"med={r['median_ms']:7.3f}ms  "
            f"{r['gflops']:8.1f} GF/s  "
            f"{r['gbps']:7.1f} GB/s  "
            f"AI={r['arith_intensity']:7.2f}  "
            f"{r['pct_of_roof']:5.1f}% of roof ({r['bound']})")


if __name__ == "__main__":
    sys.exit(main())
