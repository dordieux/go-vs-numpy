# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Plot benchmark results: one throughput chart per op.

Usage: uv run scripts/plot.py results/all.jsonl [plots/] [--include-cpp]

The C++ -O3 baseline is measured but hidden by default: the talk's main
charts are the Go vs NumPy duel, and C++ appears only on the dedicated
auto-vectorization slide.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# bytes moved per element: reads + writes (dst counts as one write)
BYTES_PER_ELEM = {
    "clip": 16,  # read src, write dst
    "add": 24,  # read a, b, write dst
    "axpy": 24,
    "sqrt": 16,
    "exp": 16,
    "sum": 8,  # read only
}

SERIES_STYLE = {
    ("python", "pure"): dict(color="#999999", ls=":", label="Python pure loop"),
    ("python", "numpy"): dict(color="#306998", ls="-", label="NumPy (1 thread)"),
    ("python", "numpy_mt"): dict(color="#1f4468", ls="--", label="NumPy (threaded)"),
    ("cpp", "scalar_O3"): dict(color="#f34b7d", ls="-", label="C++ -O3"),
    ("go", "scalar"): dict(color="#8cc5e3", ls="-", label="Go scalar"),
    ("go", "simd"): dict(color="#00add8", ls="-", label="Go SIMD"),
    ("go", "parallel"): dict(color="#7fd13b", ls="--", label="Go parallel"),
    ("go", "parallel_simd"): dict(color="#0f7038", ls="-", label="Go parallel+SIMD"),
}


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--include-cpp"]
    include_cpp = "--include-cpp" in sys.argv[1:]
    path = args[0] if len(args) > 0 else "results/all.jsonl"
    out_dir = Path(args[1] if len(args) > 1 else "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    series: dict = defaultdict(list)  # (op, lang, impl) -> [(n, gib_s)]
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["lang"] == "cpp" and not include_cpp:
                continue
            op, n = row["op"], row["n"]
            gib_s = BYTES_PER_ELEM[op] * n / row["ns_min"] * 1e9 / 2**30
            series[(op, row["lang"], row["impl"])].append((n, gib_s))

    ops = sorted({op for (op, _, _) in series})
    for op in ops:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (s_op, lang, impl), points in sorted(series.items()):
            if s_op != op:
                continue
            style = SERIES_STYLE.get((lang, impl), dict(label=f"{lang}/{impl}"))
            points.sort()
            ax.plot(
                [n for n, _ in points],
                [g for _, g in points],
                marker="o",
                markersize=4,
                **style,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("elements (n)")
        ax.set_ylabel("effective throughput (GiB/s)")
        ax.set_title(f"{op}: float64 throughput vs size (higher is better)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = out_dir / f"{op}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
