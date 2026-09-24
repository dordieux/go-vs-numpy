# go-vs-numpy

How far can Go go on large `float64` arrays, now that it has a SIMD API?

Go 1.26 added an experimental `simd/archsimd` package (amd64 only), and Go 1.27
turned it into a portable `simd` package that also works on arm64 NEON. This
repository measures what that actually buys you, with NumPy as the yardstick.

## What is measured

Six elementwise kernels over `float64` arrays, swept from 1e3 to 1e8 elements:

| op   | definition                | character     |
| ---- | ------------------------- | ------------- |
| clip | `dst[i] = max(src[i], 0)` | memory-bound  |
| add  | `dst[i] = a[i] + b[i]`    | memory-bound  |
| axpy | `dst[i] = α·x[i] + y[i]`  | memory-bound  |
| sqrt | `dst[i] = sqrt(src[i])`   | mixed         |
| exp  | `dst[i] = exp(src[i])`    | compute-bound |
| sum  | `Σ src[i]`                | reduction     |

Implementations:

| language | variants |
| --- | --- |
| **Go** | naive loop, goroutine-parallel, portable `simd` package, parallel + SIMD |
| **Python** | naive loop (capped at 1e6), NumPy single-threaded, NumPy across a `ThreadPoolExecutor` |
| **C++** | naive loops at `-O3`, single-threaded — a reference ceiling and a demo of compiler auto-vectorization |

The threaded NumPy variant matters: NumPy's ufuncs release the GIL, so slicing
an array across threads really does scale. Comparing goroutine-parallel Go
against single-threaded NumPy would flatter Go for no good reason.

## Methodology

- **Same answer everywhere.** All harnesses share one deterministic input
  generator (a Knuth multiplicative hash of the index mapped into [-100, 100)),
  and `scripts/verify.py` cross-checks every implementation's checksum.
- **Only the kernel is timed.** Checksums are computed once after measurement,
  and destination buffers are preallocated, so allocators stay out of the numbers.
- **Repeat until stable.** Each measurement accumulates at least 0.3 s (minimum
  5 repetitions) and reports the minimum and median wall time.
- **No hidden thresholds.** Neither the Go parallel variants nor the threaded
  NumPy variant skip splitting at small sizes. Parallelism overhead at small `n`
  is part of what this measures.
- **Refuses to run on a busy machine.** `run_all.sh` aborts when the 1-minute
  load average exceeds 2.0. Background indexing and security scanners can skew
  wall-clock numbers by 2x, and not uniformly across implementations.

Reference numbers come from the CI workflow, which runs on an idle amd64 runner
with AVX2. A laptop is a poor benchmark host.

## Run it

```bash
./scripts/run_all.sh                        # everything -> results/all.jsonl
python3 scripts/verify.py results/all.jsonl # cross-language parity
uv run scripts/plot.py results/all.jsonl plots/
```

Environment knobs: `SIZES`, `OPS`, `MINTIME_S`, `OUT_DIR`, `MAX_LOAD`,
`BENCH_WORKERS` (threads for the NumPy variant), `GO` (path to a Go ≥ 1.27
toolchain).

Requirements: Go ≥ 1.27 (fetched automatically via the module's `toolchain`
directive), `uv`, and `clang++`.

Note: the `go` command from Go 1.24 crashes when it sees `GOEXPERIMENT=simd`,
so `run_all.sh` locates a downloaded ≥ 1.27 toolchain binary and invokes it
directly.

## A note on lane counts

The portable `simd` package picks its vector width at runtime:

| hardware | vector width | float64 lanes |
| --- | --- | --- |
| arm64 NEON | 128-bit (fixed; SVE is "still TBD" in the Go source) | 2 |
| amd64 + AVX2 | 256-bit | 4 |
| amd64 + AVX-512 | 512-bit | 8 |

So the same binary vectorizes differently depending on where it runs, and an
Apple Silicon Mac is the narrowest case. `GODEBUG=gosimd=<bits>` can narrow the
width for testing, but cannot exceed what the hardware offers.
