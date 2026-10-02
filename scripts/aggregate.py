# /// script
# requires-python = ">=3.12"
# ///
"""Aggregate several benchmark runs into one view.

Usage: python3 scripts/aggregate.py run1.jsonl run2.jsonl ... [--n 100000000]

A single run on a shared CI runner is not enough to call a winner. Memory-bound
kernels compete with other tenants for bandwidth, so their wall times move by
~10% between runs while compute-bound kernels stay within a fraction of a
percent. This script reports the spread so that claims can be scoped to what
the data actually supports:

  - median across runs, plus the min/max envelope
  - coefficient of variation, which separates stable from noisy measurements
  - a verdict that only calls a winner when the envelopes do not overlap
"""

import json
import statistics
import sys
from collections import defaultdict

N_DEFAULT = 100_000_000
OPS = ["clip", "add", "axpy", "sqrt", "exp", "sum"]


def load(paths):
    runs = []
    for p in paths:
        by = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                by[(r["op"], r["n"], r["lang"], r["impl"])] = r["ns_min"]
        runs.append(by)
    return runs


def stats(values):
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    cv = (statistics.stdev(values) / med * 100) if len(values) > 1 else 0.0
    return med, lo, hi, cv


def best_go_parallel(run, op, n):
    vals = [run.get((op, n, "go", i)) for i in ("parallel", "parallel_simd")]
    vals = [v for v in vals if v]
    return min(vals) if vals else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = N_DEFAULT
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    runs = load(args)
    print(f"{len(runs)} runs, n={n:,}\n")

    series = defaultdict(list)
    for run in runs:
        for op in OPS:
            npv = run.get((op, n, "python", "numpy_mt"))
            gov = best_go_parallel(run, op, n)
            if npv:
                series[(op, "NumPy")].append(npv / 1e6)
            if gov:
                series[(op, "Go")].append(gov / 1e6)

    print(f"{'op':6} {'side':6} {'median':>9} {'min':>9} {'max':>9} {'CV':>7}")
    print("-" * 52)
    for op in OPS:
        for side in ("NumPy", "Go"):
            v = series[(op, side)]
            if not v:
                continue
            med, lo, hi, cv = stats(v)
            print(f"{op:6} {side:6} {med:8.1f}ms {lo:8.1f} {hi:8.1f} {cv:6.1f}%")
        print()

    print("=== verdict (a winner only when the min/max envelopes stay apart) ===")
    for op in OPS:
        npv, gov = series[(op, "NumPy")], series[(op, "Go")]
        if not npv or not gov:
            continue
        n_med, n_lo, n_hi, _ = stats(npv)
        g_med, g_lo, g_hi, _ = stats(gov)
        if g_hi < n_lo:
            v = f"Go が速い (中央値で {n_med / g_med:.2f} 倍)"
        elif n_hi < g_lo:
            v = f"NumPy が速い (中央値で {g_med / n_med:.2f} 倍)"
        else:
            v = "測定の揺れに埋もれる差"
        print(f"  {op:6} {v}")


if __name__ == "__main__":
    main()
