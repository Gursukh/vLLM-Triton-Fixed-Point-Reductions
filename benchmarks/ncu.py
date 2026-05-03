from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any


_METRIC_MAP: dict[str, list[str]] = {
    "sm_active_pct": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "achieved_occupancy_pct": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    "dram_bytes": [
        "dram__bytes.sum",
        "dram__bytes_read.sum",
    ],
    "l1_hit_pct": ["l1tex__t_sector_hit_rate.pct"],
    "l2_hit_pct": ["lts__t_sector_hit_rate.pct"],
}

_DURATION_KEYS = ("gpu__time_duration.sum", "gpu__time_active.sum")
_DRAM_RATE_KEYS = ("dram__bytes.sum.per_second",)

_SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy"]


def is_available() -> bool:
    return shutil.which("ncu") is not None


def _child_script(op_name: str, shape_json: str, config_json: str,
                  iters: int) -> str:
    return (
        "import json, torch\n"
        "from benchmarks.ops import build_one\n"
        f"shape = json.loads({shape_json!r})\n"
        f"config = json.loads({config_json!r})\n"
        f"fn, _, _ = build_one({op_name!r}, shape, config)\n"
        f"for _ in range({iters}):\n"
        "    fn()\n"
        "torch.cuda.synchronize()\n"
    )


def _parse_csv(text: str) -> dict[str, float]:
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.startswith('"ID"') or l.startswith("ID,")), None)
    if start is None:
        return {}
    reader = csv.reader(lines[start:])
    try:
        header = next(reader)
    except StopIteration:
        return {}

    # Skip the units row that ncu --page raw emits after the header.
    id_idx = header.index("ID") if "ID" in header else 0
    rows: list[list[str]] = []
    for r in reader:
        if id_idx < len(r) and r[id_idx].strip():
            rows.append(r)
    if not rows:
        return {}

    # Long form (--page details / older ncu).
    if "Metric Name" in header and "Metric Value" in header:
        name_i = header.index("Metric Name")
        val_i = header.index("Metric Value")
        by_metric: dict[str, list[float]] = {}
        for r in rows:
            if name_i >= len(r) or val_i >= len(r):
                continue
            name = r[name_i].strip()
            val = r[val_i].strip().replace(",", "")
            if not name or not val:
                continue
            try:
                by_metric.setdefault(name, []).append(float(val))
            except ValueError:
                continue
        out: dict[str, float] = {}
        for field, candidates in _METRIC_MAP.items():
            for m in candidates:
                if m in by_metric:
                    vs = by_metric[m]
                    out[field] = sum(vs) / len(vs)
                    break
        return out

    # Wide form (--page raw): metric per column, kernel per row.
    def _avg(col_name: str) -> float | None:
        if col_name not in header:
            return None
        i = header.index(col_name)
        vals: list[float] = []
        for r in rows:
            if i >= len(r):
                continue
            v = r[i].strip().replace(",", "")
            if not v:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        return sum(vals) / len(vals) if vals else None

    out = {}
    for field, candidates in _METRIC_MAP.items():
        for m in candidates:
            v = _avg(m)
            if v is not None:
                out[field] = v
                break

    # Fall back to rate * duration if absolute bytes weren't collected.
    if "dram_bytes" not in out:
        rate_gbps = next((_avg(k) for k in _DRAM_RATE_KEYS if _avg(k) is not None),
                         None)
        dur_ms = next((_avg(k) for k in _DURATION_KEYS if _avg(k) is not None),
                      None)
        if rate_gbps is not None and dur_ms is not None:
            out["dram_bytes"] = rate_gbps * 1e9 * dur_ms * 1e-3 * len(rows)
    return out


def run(*, op_name: str, kernel_regex: str, shape: dict, config: dict,
        warmup: int, iters: int) -> dict[str, Any] | None:
    if not is_available():
        return None

    shape_json, config_json = json.dumps(shape), json.dumps(config)
    with tempfile.TemporaryDirectory() as tmp:
        rep = os.path.join(tmp, "out")
        section_args: list[str] = []
        for s in _SECTIONS:
            section_args += ["--section", s]
        cmd = [
            "ncu", *section_args, "--csv",
            "--kernel-name", f"regex:{kernel_regex}",
            "--launch-skip", str(warmup),
            "--launch-count", str(iters),
            "--target-processes", "all",
            "--force-overwrite",
            "--replay-mode", "kernel",
            "-o", rep,
            sys.executable, "-c", _child_script(op_name, shape_json,
                                                config_json, warmup + iters),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[:300].replace("\n", " ")
            return {"ncu_status": f"ncu-failed rc={proc.returncode}: {err}"}

        rep_file = rep + ".ncu-rep"
        if not os.path.exists(rep_file):
            return {"ncu_status": "no-report-produced"}

        imp = subprocess.run(
            ["ncu", "--import", rep_file, "--csv", "--page", "raw"],
            capture_output=True, text=True)
        if imp.returncode != 0:
            return {"ncu_status": f"ncu-import-failed rc={imp.returncode}"}

        metrics = _parse_csv(imp.stdout)
        if not metrics:
            label = shape.get("label", "noshape")
            dump = os.path.join(tempfile.gettempdir(),
                                f"ncu_dump_{op_name}_{label}_{os.getpid()}.csv")
            try:
                with open(dump, "w") as f:
                    f.write(imp.stdout)
            except OSError:
                dump = "<dump-failed>"
            return {"ncu_status": f"no-metrics-parsed (dump={dump})"}
        metrics["ncu_status"] = "ok"
        return metrics
