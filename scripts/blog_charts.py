# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Charts for the accompanying blog post.

Usage: uv run scripts/blog_charts.py <results.jsonl> <out_dir>

Three figures, each carrying one point of the article:
  1. parallel_showdown — NumPy threads vs Go goroutines at 1e8, per op
  2. bound_by_what     — how improvements stack up for a memory-bound op
                         (clip) versus a compute-bound one (sqrt)
  3. parallel_overhead — the cost of going parallel at small sizes, where
                         goroutines and OS threads differ by two orders of
                         magnitude
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 100_000_000
GO_BLUE = "#00add8"
GO_DARK = "#0f7038"
PY_BLUE = "#306998"
PY_DARK = "#1f4468"
GRAY = "#999999"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "font.family": ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "sans-serif"],
    "axes.unicode_minus": False,
})


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return {(r["op"], r["n"], r["lang"], r["impl"]): r["ns_min"] for r in rows}


def go_parallel_best(by, op, n):
    vals = [by.get((op, n, "go", i)) for i in ("parallel", "parallel_simd")]
    return min(v for v in vals if v)


def chart_showdown(by, out):
    ops = ["clip", "add", "axpy", "sqrt", "exp", "sum"]
    numpy_ms = [by[(op, N, "python", "numpy_mt")] / 1e6 for op in ops]
    go_ms = [go_parallel_best(by, op, N) / 1e6 for op in ops]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = range(len(ops))
    h = 0.38
    ax.barh([i + h / 2 for i in y], numpy_ms, height=h, color=PY_BLUE, label="NumPy (並列)")
    ax.barh([i - h / 2 for i in y], go_ms, height=h, color=GO_DARK, label="Go (並列)")
    ax.set_yticks(list(y), ops)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("処理時間 ms  (対数目盛・短いほど速い)")
    ax.set_title("1億要素・4スレッド同士の勝負")
    ax.legend(loc="upper right", framealpha=0.95)
    for i, (nm, gm) in enumerate(zip(numpy_ms, go_ms)):
        winner = "互角" if 0.97 < nm / gm < 1.03 else ("Go" if nm > gm else "NumPy")
        color = {"Go": GO_DARK, "NumPy": PY_BLUE, "互角": "#555555"}[winner]
        ax.text(max(nm, gm) * 1.15, i, winner, va="center", fontsize=12,
                color=color, fontweight="bold")
    ax.set_xlim(right=max(max(numpy_ms), max(go_ms)) * 3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "parallel-showdown.png", dpi=200, facecolor="white")
    plt.close(fig)


def chart_bound(by, out):
    impls = [("scalar", "素朴なループ"), ("simd", "SIMD"),
             ("parallel", "並列"), ("parallel_simd", "並列 + SIMD")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, op, title, color in [
        (axes[0], "clip", "clip: メモリ帯域が上限", GRAY),
        (axes[1], "sqrt", "sqrt: 計算が上限", GO_BLUE),
    ]:
        base = by[(op, N, "go", "scalar")]
        ratios = [base / by[(op, N, "go", i)] for i, _ in impls]
        labels = [lbl for _, lbl in impls]
        bars = ax.bar(labels, ratios, color=color, width=0.62)
        bars[-1].set_color(GO_DARK)
        ax.set_title(title)
        ax.axhline(1.0, color="#cccccc", lw=1)
        for b, r in zip(bars, ratios):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.12, f"{r:.2f}倍",
                    ha="center", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("素朴なループの何倍速いか")
    axes[0].set_ylim(0, 6.6)
    fig.tight_layout()
    fig.savefig(out / "bound-by-what.png", dpi=200, facecolor="white")
    plt.close(fig)


def chart_overhead(by, out):
    sizes = [1000, 10000, 100000, 1000000]
    go_ratio, np_ratio = [], []
    for n in sizes:
        go_ratio.append(by[("clip", n, "go", "parallel_simd")] / by[("clip", n, "go", "scalar")])
        np_ratio.append(by[("clip", n, "python", "numpy_mt")] / by[("clip", n, "python", "numpy")])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(sizes, go_ratio, marker="o", markersize=7, lw=2.5, color=GO_DARK, label="Go (goroutine)")
    ax.plot(sizes, np_ratio, marker="s", markersize=7, lw=2.5, color=PY_BLUE, label="NumPy (OSスレッド)")
    ax.axhline(1.0, color="#888888", ls="--", lw=1.5)
    ax.text(1100, 1.15, "この線より下なら並列化が得", fontsize=11, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("要素数")
    ax.set_ylabel("並列化で何倍になるか (小さいほど良い)")
    ax.set_title("小さいデータで並列化したときのコスト (clip)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.annotate(f"{np_ratio[0]:.0f}倍に悪化", xy=(1000, np_ratio[0]),
                xytext=(2600, np_ratio[0] * 1.1), fontsize=12,
                color=PY_BLUE, fontweight="bold")
    ax.annotate(f"{go_ratio[0]:.1f}倍で済む", xy=(1000, go_ratio[0]),
                xytext=(2600, go_ratio[0] * 0.78), fontsize=12,
                color=GO_DARK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "parallel-overhead.png", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/all.jsonl")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "plots/blog")
    out.mkdir(parents=True, exist_ok=True)
    by = load(src)
    chart_showdown(by, out)
    chart_bound(by, out)
    chart_overhead(by, out)
    for p in sorted(out.glob("*.png")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
