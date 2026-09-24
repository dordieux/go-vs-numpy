"""Float64 kernel benchmarks: pure Python and NumPy.

Emits one JSON line per (op, impl, n) measurement on stdout, using the same
schema and deterministic input generator as the Go and C++ harnesses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ALPHA = 1.0000001

# Pure-Python loops above this size are too slow to sweep; NumPy has no cap.
PURE_PYTHON_MAX_N = 1_000_000


def make_input(n: int) -> np.ndarray:
    """Knuth multiplicative hash of the index mapped into [-100, 100)."""
    i = np.arange(n, dtype=np.uint64)
    h = (i * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return h.astype(np.float64) / 4294967296.0 * 200 - 100


# ---- pure python kernels (operate on Python lists) ----


def clip_pure(dst: list, src: list) -> None:
    for i, v in enumerate(src):
        dst[i] = 0.0 if v < 0 else v


def add_pure(dst: list, a: list, b: list) -> None:
    for i in range(len(a)):
        dst[i] = a[i] + b[i]


def axpy_pure(dst: list, x: list, y: list) -> None:
    for i in range(len(x)):
        dst[i] = ALPHA * x[i] + y[i]


def sqrt_pure(dst: list, src: list) -> None:
    for i, v in enumerate(src):
        dst[i] = math.sqrt(v)


def exp_pure(dst: list, src: list) -> None:
    for i, v in enumerate(src):
        dst[i] = math.exp(v)


def sum_pure(src: list) -> float:
    total = 0.0
    for v in src:
        total += v
    return total


# ---- numpy kernels (preallocated out= to measure compute, not allocation) ----


def clip_np(dst, src) -> None:
    np.maximum(src, 0.0, out=dst)


def add_np(dst, a, b) -> None:
    np.add(a, b, out=dst)


def axpy_np(dst, x, y) -> None:
    # Idiomatic in-place NumPy: two passes, no temporary allocation.
    np.multiply(x, ALPHA, out=dst)
    np.add(dst, y, out=dst)


def sqrt_np(dst, src) -> None:
    np.sqrt(src, out=dst)


def exp_np(dst, src) -> None:
    np.exp(src, out=dst)


def sum_np(src) -> float:
    return float(src.sum())


# ---- numpy + ThreadPoolExecutor kernels ----
#
# NumPy の ufunc は計算中に GIL を解放するため、配列をスライス分割して
# 複数スレッドに渡せば実際にコア数ぶんスケールする。Go の goroutine 版と
# 条件を揃えるための実装で、ワーカー数は Go の GOMAXPROCS と同じ値を使う。
# Go 側と同様、小さい n でもスレッド分割を省略しない（オーバーヘッドを隠さない）。


def _chunks(n: int, workers: int):
    if workers > n:
        workers = max(n, 1)
    chunk = (n + workers - 1) // workers
    return [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]


def clip_np_mt(pool, bounds, dst, src) -> None:
    list(pool.map(lambda b: np.maximum(src[b[0]:b[1]], 0.0, out=dst[b[0]:b[1]]), bounds))


def add_np_mt(pool, bounds, dst, a, b_) -> None:
    list(pool.map(lambda b: np.add(a[b[0]:b[1]], b_[b[0]:b[1]], out=dst[b[0]:b[1]]), bounds))


def axpy_np_mt(pool, bounds, dst, x, y) -> None:
    def work(b):
        lo, hi = b
        np.multiply(x[lo:hi], ALPHA, out=dst[lo:hi])
        np.add(dst[lo:hi], y[lo:hi], out=dst[lo:hi])
    list(pool.map(work, bounds))


def sqrt_np_mt(pool, bounds, dst, src) -> None:
    list(pool.map(lambda b: np.sqrt(src[b[0]:b[1]], out=dst[b[0]:b[1]]), bounds))


def exp_np_mt(pool, bounds, dst, src) -> None:
    list(pool.map(lambda b: np.exp(src[b[0]:b[1]], out=dst[b[0]:b[1]]), bounds))


def sum_np_mt(pool, bounds, src) -> float:
    return float(sum(pool.map(lambda b: src[b[0]:b[1]].sum(), bounds)))


def measure(fn, min_time_s: float, min_reps: int, max_reps: int):
    fn()  # warmup
    times: list[float] = []
    total = 0.0
    while (total < min_time_s or len(times) < min_reps) and len(times) < max_reps:
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        total += elapsed / 1e9
        times.append(float(elapsed))
    times.sort()
    return times[0], statistics.median(times), len(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="clip,add,axpy,sqrt,exp,sum")
    parser.add_argument("--impls", default="pure,numpy,numpy_mt")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("BENCH_WORKERS", "0")),
                        help="threads for numpy_mt (0 = os.cpu_count())")
    parser.add_argument(
        "--sizes", default="1000,10000,100000,1000000,10000000,100000000"
    )
    parser.add_argument("--mintime", type=float, default=0.3, help="seconds")
    parser.add_argument("--minreps", type=int, default=5)
    parser.add_argument("--maxreps", type=int, default=10000)
    parser.add_argument("--pure-max-n", type=int, default=PURE_PYTHON_MAX_N)
    args = parser.parse_args()

    want_ops = args.ops.split(",")
    want_impls = args.impls.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]

    workers = args.workers or os.cpu_count() or 1
    meta = (
        f"python={platform.python_version()} numpy={np.__version__} "
        f"arch={platform.machine()} workers={workers}"
    )
    print("# " + meta, file=sys.stderr)

    # プールは作り直さず使い回す。OSスレッドの起動は goroutine と違って重く、
    # 実務でも ThreadPoolExecutor は使い回すのが自然な書き方のため。
    pool = ThreadPoolExecutor(max_workers=workers)

    for n in sizes:
        x = make_input(n)
        y = make_input(n)[::-1].copy()
        dst = np.empty(n, dtype=np.float64)

        for op in want_ops:
            src = x
            if op == "sqrt":
                src = np.abs(x)
            elif op == "exp":
                src = x / 100

            # jobs: (impl, timed_fn, checksum_fn); the checksum pass runs
            # once after measurement, never inside the timed loop.
            jobs = []
            if "numpy" in want_impls:
                np_kernels = {
                    "clip": (lambda: clip_np(dst, src), lambda: float(dst.sum())),
                    "add": (lambda: add_np(dst, x, y), lambda: float(dst.sum())),
                    "axpy": (lambda: axpy_np(dst, x, y), lambda: float(dst.sum())),
                    "sqrt": (lambda: sqrt_np(dst, src), lambda: float(dst.sum())),
                    "exp": (lambda: exp_np(dst, src), lambda: float(dst.sum())),
                    "sum": (lambda: sum_np(src), lambda: sum_np(src)),
                }
                jobs.append(("numpy", *np_kernels[op]))

            if "numpy_mt" in want_impls:
                bounds = _chunks(n, workers)
                mt_kernels = {
                    "clip": (lambda: clip_np_mt(pool, bounds, dst, src), lambda: float(dst.sum())),
                    "add": (lambda: add_np_mt(pool, bounds, dst, x, y), lambda: float(dst.sum())),
                    "axpy": (lambda: axpy_np_mt(pool, bounds, dst, x, y), lambda: float(dst.sum())),
                    "sqrt": (lambda: sqrt_np_mt(pool, bounds, dst, src), lambda: float(dst.sum())),
                    "exp": (lambda: exp_np_mt(pool, bounds, dst, src), lambda: float(dst.sum())),
                    "sum": (lambda: sum_np_mt(pool, bounds, src), lambda: sum_np_mt(pool, bounds, src)),
                }
                jobs.append(("numpy_mt", *mt_kernels[op]))

            if "pure" in want_impls and n <= args.pure_max_n:
                src_l, x_l, y_l = src.tolist(), x.tolist(), y.tolist()
                dst_l = [0.0] * n
                pure_kernels = {
                    "clip": (lambda: clip_pure(dst_l, src_l), lambda: sum(dst_l)),
                    "add": (lambda: add_pure(dst_l, x_l, y_l), lambda: sum(dst_l)),
                    "axpy": (lambda: axpy_pure(dst_l, x_l, y_l), lambda: sum(dst_l)),
                    "sqrt": (lambda: sqrt_pure(dst_l, src_l), lambda: sum(dst_l)),
                    "exp": (lambda: exp_pure(dst_l, src_l), lambda: sum(dst_l)),
                    "sum": (lambda: sum_pure(src_l), lambda: sum_pure(src_l)),
                }
                jobs.append(("pure", *pure_kernels[op]))

            for impl, fn, check in jobs:
                ns_min, ns_median, reps = measure(
                    fn, args.mintime, args.minreps, args.maxreps
                )
                checksum = check()
                print(
                    json.dumps(
                        {
                            "lang": "python",
                            "impl": impl,
                            "op": op,
                            "n": n,
                            "ns_min": ns_min,
                            "ns_median": ns_median,
                            "reps": reps,
                            "checksum": checksum,
                            "meta": meta,
                        }
                    ),
                    flush=True,
                )
                print(
                    f"python {impl:<8} {op:<5} n={n:<10} min={ns_min:12.0f}ns reps={reps}",
                    file=sys.stderr,
                )


    pool.shutdown()


if __name__ == "__main__":
    main()
