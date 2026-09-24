// Command bench runs the float64 kernel benchmarks and emits one JSON line
// per (op, impl, n) measurement. The schema is shared with the Python and
// C++ harnesses so results can be merged and plotted together.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/hyunseung-park/go-vs-numpy/ops"
)

const alpha = 1.0000001

// makeInput mirrors the deterministic generator used by all harnesses:
// a Knuth multiplicative hash of the index mapped into [-100, 100).
func makeInput(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		h := (uint64(i) * 2654435761) & 0xFFFFFFFF
		out[i] = float64(h)/4294967296.0*200 - 100
	}
	return out
}

type result struct {
	Lang     string  `json:"lang"`
	Impl     string  `json:"impl"`
	Op       string  `json:"op"`
	N        int     `json:"n"`
	NsMin    float64 `json:"ns_min"`
	NsMedian float64 `json:"ns_median"`
	Reps     int     `json:"reps"`
	Checksum float64 `json:"checksum"`
	Meta     string  `json:"meta"`
}

// measure runs fn repeatedly until minTime is accumulated (at least minReps
// runs, at most maxReps) and returns min/median ns plus the rep count.
func measure(fn func(), minTime time.Duration, minReps, maxReps int) (nsMin, nsMedian float64, reps int) {
	fn() // warmup, also faults in the destination pages

	var times []float64
	var total time.Duration
	for (total < minTime || len(times) < minReps) && len(times) < maxReps {
		start := time.Now()
		fn()
		d := time.Since(start)
		total += d
		times = append(times, float64(d.Nanoseconds()))
	}
	slices.Sort(times)
	return times[0], times[len(times)/2], len(times)
}

func main() {
	opsFlag := flag.String("ops", "clip,add,axpy,sqrt,exp,sum", "comma-separated ops to run")
	implsFlag := flag.String("impls", "scalar,parallel,simd,parallel_simd", "comma-separated impls to run")
	sizesFlag := flag.String("sizes", "1000,10000,100000,1000000,10000000,100000000", "comma-separated sizes")
	minTime := flag.Duration("mintime", 300*time.Millisecond, "minimum accumulated time per measurement")
	minReps := flag.Int("minreps", 5, "minimum repetitions per measurement")
	maxReps := flag.Int("maxreps", 10000, "maximum repetitions per measurement")
	flag.Parse()

	wantOps := strings.Split(*opsFlag, ",")
	wantImpls := strings.Split(*implsFlag, ",")
	var sizes []int
	for _, s := range strings.Split(*sizesFlag, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad size:", s)
			os.Exit(1)
		}
		sizes = append(sizes, n)
	}

	meta := fmt.Sprintf("go=%s simd=%v lanes=%d gomaxprocs=%d arch=%s",
		runtime.Version(), ops.SIMDEnabled, ops.Lanes(), runtime.GOMAXPROCS(0), runtime.GOARCH)
	fmt.Fprintln(os.Stderr, "# "+meta)

	enc := json.NewEncoder(os.Stdout)

	// The timed function must contain the kernel only; checksums are computed
	// once after measurement so the digest pass never pollutes the timing.
	// sink defeats dead-code elimination for reduction kernels.
	var sink float64
	digest := func(dst []float64) float64 {
		s := 0.0
		for _, v := range dst {
			s += v
		}
		return s
	}
	type kernel struct {
		impl  string
		run   func(dst, x, y []float64)
		check func(dst, x, y []float64) float64
	}
	fromDst := func(dst, _, _ []float64) float64 { return digest(dst) }
	fromSink := func(_, _, _ []float64) float64 { return sink }

	kernels := map[string][]kernel{
		"clip": {
			{"scalar", func(dst, x, _ []float64) { ops.ClipScalar(dst, x) }, fromDst},
			{"parallel", func(dst, x, _ []float64) { ops.ClipParallel(dst, x) }, fromDst},
			{"simd", func(dst, x, _ []float64) { ops.ClipSIMD(dst, x) }, fromDst},
			{"parallel_simd", func(dst, x, _ []float64) { ops.ClipParallelSIMD(dst, x) }, fromDst},
		},
		"add": {
			{"scalar", func(dst, x, y []float64) { ops.AddScalar(dst, x, y) }, fromDst},
			{"parallel", func(dst, x, y []float64) { ops.AddParallel(dst, x, y) }, fromDst},
			{"simd", func(dst, x, y []float64) { ops.AddSIMD(dst, x, y) }, fromDst},
			{"parallel_simd", func(dst, x, y []float64) { ops.AddParallelSIMD(dst, x, y) }, fromDst},
		},
		"axpy": {
			{"scalar", func(dst, x, y []float64) { ops.AxpyScalar(dst, x, y, alpha) }, fromDst},
			{"parallel", func(dst, x, y []float64) { ops.AxpyParallel(dst, x, y, alpha) }, fromDst},
			{"simd", func(dst, x, y []float64) { ops.AxpySIMD(dst, x, y, alpha) }, fromDst},
			{"parallel_simd", func(dst, x, y []float64) { ops.AxpyParallelSIMD(dst, x, y, alpha) }, fromDst},
		},
		"sqrt": {
			{"scalar", func(dst, x, _ []float64) { ops.SqrtScalar(dst, x) }, fromDst},
			{"parallel", func(dst, x, _ []float64) { ops.SqrtParallel(dst, x) }, fromDst},
			{"simd", func(dst, x, _ []float64) { ops.SqrtSIMD(dst, x) }, fromDst},
			{"parallel_simd", func(dst, x, _ []float64) { ops.SqrtParallelSIMD(dst, x) }, fromDst},
		},
		"exp": {
			{"scalar", func(dst, x, _ []float64) { ops.ExpScalar(dst, x) }, fromDst},
			{"parallel", func(dst, x, _ []float64) { ops.ExpParallel(dst, x) }, fromDst},
		},
		"sum": {
			{"scalar", func(_, x, _ []float64) { sink = ops.SumScalar(x) }, fromSink},
			{"parallel", func(_, x, _ []float64) { sink = ops.SumParallel(x) }, fromSink},
			{"simd", func(_, x, _ []float64) { sink = ops.SumSIMD(x) }, fromSink},
		},
	}

	for _, n := range sizes {
		x := makeInput(n)
		y := makeInput(n)
		slices.Reverse(y)
		dst := make([]float64, n)

		for _, opName := range wantOps {
			ks, ok := kernels[opName]
			if !ok {
				fmt.Fprintln(os.Stderr, "unknown op:", opName)
				os.Exit(1)
			}
			// per-op input domain adjustments (shared across harnesses)
			in := x
			switch opName {
			case "sqrt":
				in = make([]float64, n)
				for i, v := range x {
					in[i] = math.Abs(v)
				}
			case "exp":
				in = make([]float64, n)
				for i, v := range x {
					in[i] = v / 100
				}
			}

			for _, k := range ks {
				if !slices.Contains(wantImpls, k.impl) {
					continue
				}
				if strings.Contains(k.impl, "simd") && !ops.SIMDEnabled {
					continue
				}
				nsMin, nsMedian, reps := measure(func() {
					k.run(dst, in, y)
				}, *minTime, *minReps, *maxReps)
				checksum := k.check(dst, in, y)
				r := result{
					Lang: "go", Impl: k.impl, Op: opName, N: n,
					NsMin: nsMin, NsMedian: nsMedian, Reps: reps,
					Checksum: checksum, Meta: meta,
				}
				if err := enc.Encode(r); err != nil {
					panic(err)
				}
				fmt.Fprintf(os.Stderr, "go %-14s %-5s n=%-10d min=%12.0fns reps=%d\n",
					k.impl, opName, n, nsMin, reps)
			}
		}
	}
}
