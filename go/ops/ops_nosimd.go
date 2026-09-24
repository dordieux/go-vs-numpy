//go:build !goexperiment.simd

package ops

// SIMDEnabled reports whether this binary was built with GOEXPERIMENT=simd.
const SIMDEnabled = false

// Lanes returns 1 when SIMD is unavailable.
func Lanes() int { return 1 }

// Fallbacks so the benchmark binary still compiles without GOEXPERIMENT=simd.
// The harness skips simd impls when SIMDEnabled is false, so these never run
// in a measured loop.

func ClipSIMD(dst, src []float64)                     { ClipScalar(dst, src) }
func AddSIMD(dst, a, b []float64)                     { AddScalar(dst, a, b) }
func AxpySIMD(dst, x, y []float64, alpha float64)     { AxpyScalar(dst, x, y, alpha) }
func SqrtSIMD(dst, src []float64)                     { SqrtScalar(dst, src) }
func SumSIMD(src []float64) float64                   { return SumScalar(src) }
func ClipParallelSIMD(dst, src []float64)             { ClipParallel(dst, src) }
func AddParallelSIMD(dst, a, b []float64)             { AddParallel(dst, a, b) }
func AxpyParallelSIMD(dst, x, y []float64, alpha float64) {
	AxpyParallel(dst, x, y, alpha)
}
func SqrtParallelSIMD(dst, src []float64) { SqrtParallel(dst, src) }
