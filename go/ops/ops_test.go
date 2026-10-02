package ops

import (
	"math"
	"testing"
)

// testInput mirrors the deterministic input generator used by all three
// language harnesses (see cmd/bench and the python/cpp counterparts).
func testInput(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		h := (uint64(i) * 2654435761) & 0xFFFFFFFF
		out[i] = float64(h)/4294967296.0*200 - 100
	}
	return out
}

// sizes deliberately include 0, tiny, odd, and non-lane-aligned lengths.
var testSizes = []int{0, 1, 2, 3, 7, 8, 1023, 4096, 100003}

func almostEqual(a, b, relTol float64) bool {
	if a == b {
		return true
	}
	diff := math.Abs(a - b)
	scale := math.Max(math.Abs(a), math.Abs(b))
	return diff <= relTol*scale
}

func assertSliceEqual(t *testing.T, name string, got, want []float64, relTol float64) {
	t.Helper()
	for i := range want {
		if !almostEqual(got[i], want[i], relTol) {
			t.Fatalf("%s: index %d: got %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestElementwiseVariantsMatchScalar(t *testing.T) {
	type variant struct {
		name string
		run  func(dst []float64, x, y []float64)
		skip bool
	}
	type op struct {
		name     string
		scalar   func(dst []float64, x, y []float64)
		variants []variant
		prep     func(x []float64) // adjust input domain (e.g. sqrt needs >= 0)
	}

	const alpha = 1.0000001
	ops := []op{
		{
			name:   "clip",
			scalar: func(dst, x, _ []float64) { ClipScalar(dst, x) },
			variants: []variant{
				{"parallel", func(dst, x, _ []float64) { ClipParallel(dst, x) }, false},
				{"simd", func(dst, x, _ []float64) { ClipSIMD(dst, x) }, !SIMDEnabled},
				{"parallel_simd", func(dst, x, _ []float64) { ClipParallelSIMD(dst, x) }, !SIMDEnabled},
			},
		},
		{
			name:   "add",
			scalar: func(dst, x, y []float64) { AddScalar(dst, x, y) },
			variants: []variant{
				{"parallel", func(dst, x, y []float64) { AddParallel(dst, x, y) }, false},
				{"simd", func(dst, x, y []float64) { AddSIMD(dst, x, y) }, !SIMDEnabled},
				{"parallel_simd", func(dst, x, y []float64) { AddParallelSIMD(dst, x, y) }, !SIMDEnabled},
			},
		},
		{
			name:   "axpy",
			scalar: func(dst, x, y []float64) { AxpyScalar(dst, x, y, alpha) },
			variants: []variant{
				{"parallel", func(dst, x, y []float64) { AxpyParallel(dst, x, y, alpha) }, false},
				{"simd", func(dst, x, y []float64) { AxpySIMD(dst, x, y, alpha) }, !SIMDEnabled},
				{"parallel_simd", func(dst, x, y []float64) { AxpyParallelSIMD(dst, x, y, alpha) }, !SIMDEnabled},
			},
		},
		{
			name:   "sqrt",
			scalar: func(dst, x, _ []float64) { SqrtScalar(dst, x) },
			prep: func(x []float64) {
				for i := range x {
					x[i] = math.Abs(x[i])
				}
			},
			variants: []variant{
				{"parallel", func(dst, x, _ []float64) { SqrtParallel(dst, x) }, false},
				{"simd", func(dst, x, _ []float64) { SqrtSIMD(dst, x) }, !SIMDEnabled},
				{"parallel_simd", func(dst, x, _ []float64) { SqrtParallelSIMD(dst, x) }, !SIMDEnabled},
			},
		},
		{
			name:   "exp",
			scalar: func(dst, x, _ []float64) { ExpScalar(dst, x) },
			prep: func(x []float64) {
				for i := range x {
					x[i] /= 100 // keep exp in a sane range
				}
			},
			variants: []variant{
				{"parallel", func(dst, x, _ []float64) { ExpParallel(dst, x) }, false},
			},
		},
	}

	for _, o := range ops {
		for _, n := range testSizes {
			x := testInput(n)
			y := testInput(n)
			for i, j := 0, len(y)-1; i < j; i, j = i+1, j-1 {
				y[i], y[j] = y[j], y[i]
			}
			if o.prep != nil {
				o.prep(x)
			}
			want := make([]float64, n)
			o.scalar(want, x, y)
			for _, v := range o.variants {
				if v.skip {
					continue
				}
				got := make([]float64, n)
				v.run(got, x, y)
				assertSliceEqual(t, o.name+"/"+v.name, got, want, 0)
			}
		}
	}
}

func TestSumVariantsMatchScalar(t *testing.T) {
	for _, n := range testSizes {
		x := testInput(n)
		want := SumScalar(x)
		// Different summation orders legitimately give slightly different
		// float results; allow a small relative tolerance.
		if got := SumParallel(x); !almostEqual(got, want, 1e-9) {
			t.Fatalf("sum/parallel: n=%d got %v want %v", n, got, want)
		}
		if SIMDEnabled {
			if got := SumSIMD(x); !almostEqual(got, want, 1e-9) {
				t.Fatalf("sum/simd: n=%d got %v want %v", n, got, want)
			}
			if got := SumParallelSIMD(x); !almostEqual(got, want, 1e-9) {
				t.Fatalf("sum/parallel_simd: n=%d got %v want %v", n, got, want)
			}
		}
	}
}
