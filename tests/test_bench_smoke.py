from __future__ import annotations

import csv

from tests.fixed_point_helpers import requires_cuda


@requires_cuda
def test_bench_runs_one_op(tmp_path):
    from benchmarks.bench import main
    rc = main([
        "--ops", "rms_norm",
        "--configs", "int_bits=32",
        "--iters", "5",
        "--warmup", "1",
        "--out", str(tmp_path),
    ])
    assert rc == 0

    csvs = list(tmp_path.glob("bench_*.csv"))
    assert len(csvs) == 1
    with csvs[0].open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "no rows written"
    r = rows[0]
    for col in ("median_ms", "p95_ms", "gflops", "gbps", "arith_intensity",
                "roof_gflops", "pct_of_roof", "bound", "gpu_name"):
        assert r.get(col), f"missing {col}"
    assert r["ncu_status"] == "skipped"


def test_flop_byte_models_sane():
    from benchmarks.ops import OPS
    assert set(OPS) == {"cast_f2i", "cast_i2f", "rms_norm", "rms_norm_residual",
                        "log_softmax", "gemm", "attn"}
    for name, spec in OPS.items():
        assert spec["shapes"], f"{name} has no shapes"
        assert spec["cache_mode"] in ("cold", "warm")
        assert spec["ncu_regex"]
