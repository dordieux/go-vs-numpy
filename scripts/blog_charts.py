# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Charts for the accompanying blog post.

Usage: uv run scripts/blog_charts.py <out_dir> <run1.jsonl> <run2.jsonl> ...

Pass several runs from the SAME cpu model. A single run on a shared CI runner
is not enough: memory-bound kernels move by several percent between runs, so
every figure here plots the median with a min/max envelope and only claims a
difference when the envelopes stay apart.

Four figures, each carrying one point of the article:
  1. three_outcomes    — the six ops split into "no difference", "Go ahead"
                         and "NumPy ahead", with the spread drawn in
  2. bandwidth_wall    — why SIMD helps clip but does nothing for add
  3. parallel_overhead — the cost of going parallel at small sizes
  4. stability         — NumPy is steadier than Go even where Go is faster
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 100_000_000
OPS = ["clip", "add", "axpy", "sqrt", "exp", "sum"]
BYTES = {"clip": 16, "add": 24, "axpy": 24, "sqrt": 16, "exp": 16, "sum": 8}

GO = "#0f7038"
GO_LIGHT = "#7fd13b"
PY = "#306998"
GRAY = "#9a9a9a"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "font.family": ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "sans-serif"],
    "axes.unicode_minus": False,
})


def load_runs(paths):
    runs = []
    for p in paths:
        by = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                by[(r["op"], r["n"], r["lang"], r["impl"])] = r["ns_min"]
        runs.append(by)
    return runs


def go_parallel(run, op, n):
    vals = [run.get((op, n, "go", i)) for i in ("parallel", "parallel_simd")]
    vals = [v for v in vals if v]
    return min(vals) if vals else None


def collect(runs):
    """-> {(op, side): [ms, ...]} across runs, at n=1e8"""
    out = defaultdict(list)
    for run in runs:
        for op in OPS:
            npv = run.get((op, N, "python", "numpy_mt"))
            gov = go_parallel(run, op, N)
            if npv:
                out[(op, "NumPy")].append(npv / 1e6)
            if gov:
                out[(op, "Go")].append(gov / 1e6)
    return out


def envelope(vals):
    return statistics.median(vals), min(vals), max(vals)


def verdict(series, op):
    n_med, n_lo, n_hi = envelope(series[(op, "NumPy")])
    g_med, g_lo, g_hi = envelope(series[(op, "Go")])
    if g_hi < n_lo:
        return "go", n_med / g_med
    if n_hi < g_lo:
        return "numpy", g_med / n_med
    return "tie", 1.0


def chart_three_outcomes(series, out):
    order, labels, verdicts = [], [], {}
    for op in OPS:
        v, ratio = verdict(series, op)
        verdicts[op] = (v, ratio)
    for want in ("tie", "go", "numpy"):
        for op in OPS:
            if verdicts[op][0] == want:
                order.append(op)
    labels = order

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    h = 0.36
    for i, op in enumerate(labels):
        for side, off, color in (("NumPy", h / 2, PY), ("Go", -h / 2, GO)):
            med, lo, hi = envelope(series[(op, side)])
            ax.barh(i + off, med, height=h, color=color,
                    xerr=[[med - lo], [hi - med]],
                    error_kw=dict(ecolor="#333333", capsize=3, lw=1.2),
                    label=side if i == 0 else None)
        v, ratio = verdicts[op]
        text = {"tie": "差は揺れに埋もれる",
                "go": f"Go が {ratio:.2f} 倍",
                "numpy": f"NumPy が {ratio:.2f} 倍"}[v]
        color = {"tie": "#555555", "go": GO, "numpy": PY}[v]
        xmax = max(envelope(series[(op, s)])[2] for s in ("NumPy", "Go"))
        ax.text(xmax * 1.1, i, text, va="center", fontsize=11,
                color=color, fontweight="bold")

    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("処理時間 ms  (対数目盛・短いほど速い。横棒は4回の最小〜最大)")
    ax.set_title("1億要素・並列同士の比較")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(right=max(envelope(series[(op, s)])[2]
                          for op in OPS for s in ("NumPy", "Go")) * 3.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "three-outcomes.png", dpi=200, facecolor="white")
    plt.close(fig)


def chart_bandwidth_wall(runs, out):
    """clipはSIMDで帯域の壁に届く。addは素朴ループの時点ですでに壁にいる。"""
    def gbs(op, impl):
        vals = [r[(op, N, "go", impl)] for r in runs if (op, N, "go", impl) in r]
        med = statistics.median(vals)
        return BYTES[op] * N / (med / 1e9) / 1e9

    pairs = [("clip", "素朴なループ", "scalar"), ("clip", "SIMD", "simd"),
             ("add", "素朴なループ", "scalar"), ("add", "SIMD", "simd")]
    vals = [gbs(op, impl) for op, _, impl in pairs]
    wall = max(vals)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = [0, 1, 2.6, 3.6]
    colors = [GRAY, GO, GO, GO]
    bars = ax.bar(xs, vals, width=0.8, color=colors)
    ax.axhline(wall, color="#cc3333", ls="--", lw=1.6)
    ax.text(3.95, wall, f"  帯域の壁\n  {wall:.0f} GB/s", color="#cc3333",
            fontsize=12, va="center", fontweight="bold")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.7, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs, [lbl for _, lbl, _ in pairs])
    ax.set_ylabel("実効メモリ帯域 (GB/s)")
    ax.set_title("clipはSIMDで壁に届く。addは最初から壁にいる")
    ax.set_ylim(0, wall * 1.35)
    ax.text(0.5, -0.22, "clip (16B/要素)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=12, color="#555555")
    ax.text(3.1, -0.22, "add (24B/要素)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=12, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out / "bandwidth-wall.png", dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


def chart_parallel_overhead(runs, out):
    sizes = [1000, 10000, 100000, 1000000]

    def ratio(lang, par, seq, n):
        vals = []
        for r in runs:
            a, b = r.get(("clip", n, lang, par)), r.get(("clip", n, lang, seq))
            if a and b:
                vals.append(a / b)
        return statistics.median(vals)

    go_r = [ratio("go", "parallel_simd", "scalar", n) for n in sizes]
    np_r = [ratio("python", "numpy_mt", "numpy", n) for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(sizes, go_r, marker="o", markersize=7, lw=2.5, color=GO, label="Go (goroutine)")
    ax.plot(sizes, np_r, marker="s", markersize=7, lw=2.5, color=PY,
            label="NumPy (ThreadPoolExecutor)")
    ax.axhline(1.0, color="#888888", ls="--", lw=1.5)
    ax.text(1100, 1.12, "この線より下なら並列化が得", fontsize=11, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("要素数")
    ax.set_ylabel("並列化で何倍になるか (小さいほど良い)")
    ax.set_title("小さいデータで並列化したときのコスト (clip)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.annotate(f"{np_r[0]:.0f}倍に悪化", xy=(1000, np_r[0]),
                xytext=(2500, np_r[0] * 1.05), fontsize=12, color=PY, fontweight="bold")
    ax.annotate(f"{go_r[0]:.1f}倍で済む", xy=(1000, go_r[0]),
                xytext=(2500, go_r[0] * 0.75), fontsize=12, color=GO, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "parallel-overhead.png", dpi=200, facecolor="white")
    plt.close(fig)


def chart_stability(series, out):
    """速さではなく「測るたびのブレ」を見る。Goのほうが揺れている。"""
    cvs = {}
    for op in OPS:
        for side in ("NumPy", "Go"):
            v = series[(op, side)]
            cvs[(op, side)] = statistics.stdev(v) / statistics.median(v) * 100

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = range(len(OPS))
    h = 0.36
    ax.bar([i + h / 2 for i in x], [cvs[(op, "NumPy")] for op in OPS],
           width=h, color=PY, label="NumPy")
    ax.bar([i - h / 2 for i in x], [cvs[(op, "Go")] for op in OPS],
           width=h, color=GO, label="Go")
    ax.set_xticks(list(x), OPS)
    ax.set_ylabel("4回の測定のばらつき (変動係数 %)")
    ax.set_title("速さとは別に、Goのほうが測るたびに揺れる")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    i = OPS.index("sqrt")
    ax.annotate("sqrtはGoのほうが速いが、\nNumPyのほうが安定している",
                xy=(i - h / 2, cvs[("sqrt", "Go")]),
                xytext=(i - 1.6, cvs[("sqrt", "Go")] + 1.6),
                fontsize=11, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4))
    fig.tight_layout()
    fig.savefig(out / "stability.png", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    out = Path(sys.argv[1])
    paths = sys.argv[2:]
    if not paths:
        sys.exit("usage: blog_charts.py <out_dir> <run1.jsonl> ...")
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(paths)
    series = collect(runs)
    chart_three_outcomes(series, out)
    chart_bandwidth_wall(runs, out)
    chart_parallel_overhead(runs, out)
    chart_stability(series, out)
    for p in sorted(out.glob("*.png")):
        print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
