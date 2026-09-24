#!/usr/bin/env bash
# Runs the Go, Python, and C++ benchmarks with a shared configuration and
# merges the JSON-lines results into results/all.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZES="${SIZES:-1000,10000,100000,1000000,10000000,100000000}"
OPS="${OPS:-clip,add,axpy,sqrt,exp,sum}"
MINTIME_S="${MINTIME_S:-0.3}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
mkdir -p "$OUT_DIR"

# Refuse to measure on a busy machine. Background load (Spotlight indexing,
# security scanners, other builds) skews wall-clock numbers by 2x or more,
# and the skew is not uniform across implementations. Override with
# MAX_LOAD=99 if you really mean it.
MAX_LOAD="${MAX_LOAD:-2.0}"
load1=$(uptime | sed 's/.*load averages*: *//' | awk '{print $1}' | tr -d ',')
if awk "BEGIN{exit !($load1 > $MAX_LOAD)}"; then
  echo "ERROR: 1-minute load average is $load1 (limit $MAX_LOAD)." >&2
  echo "       Wait for the machine to settle, or set MAX_LOAD to override." >&2
  echo "       Top consumers:" >&2
  ps -Ao %cpu,comm -r | head -4 | sed 's/^/         /' >&2
  exit 1
fi
echo "load average ok: $load1" >&2

# Resolve a Go >= 1.27 toolchain (GOEXPERIMENT=simd needs it; the go1.24
# wrapper crashes when it sees an unknown experiment, so call it directly).
if [[ -z "${GO:-}" ]]; then
  GO=$(ls -d "$HOME"/go/pkg/mod/golang.org/toolchain@*go1.2[7-9]*/bin/go 2>/dev/null | sort -V | tail -1 || true)
  GO="${GO:-go}"
fi
echo "using go: $GO ($($GO version))" >&2

echo "== build ==" >&2
(cd "$ROOT/go" && GOEXPERIMENT=simd "$GO" build -o "$ROOT/bench-go" ./cmd/bench)
(cd "$ROOT/cpp" && make -s bench)
(cd "$ROOT/python" && uv sync -q)

echo "== go ==" >&2
"$ROOT/bench-go" -ops "$OPS" -sizes "$SIZES" -mintime "${MINTIME_S}s" \
  > "$OUT_DIR/go.jsonl"

echo "== cpp ==" >&2
"$ROOT/cpp/bench" --ops "$OPS" --sizes "$SIZES" --mintime "$MINTIME_S" \
  > "$OUT_DIR/cpp.jsonl"

echo "== python ==" >&2
(cd "$ROOT/python" && uv run python bench.py --ops "$OPS" --sizes "$SIZES" \
  --mintime "$MINTIME_S") > "$OUT_DIR/python.jsonl"

cat "$OUT_DIR/go.jsonl" "$OUT_DIR/cpp.jsonl" "$OUT_DIR/python.jsonl" \
  > "$OUT_DIR/all.jsonl"
echo "wrote $OUT_DIR/all.jsonl ($(wc -l < "$OUT_DIR/all.jsonl" | tr -d ' ') rows)" >&2
