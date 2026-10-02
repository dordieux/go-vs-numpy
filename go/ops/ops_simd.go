//go:build goexperiment.simd

package ops

import "simd"

// SIMDEnabled reports whether this binary was built with GOEXPERIMENT=simd.
const SIMDEnabled = true

// Lanes returns the number of float64 lanes per vector on this platform
// (2 on arm64 NEON, 4 on amd64 AVX2, 8 on amd64 AVX-512).
func Lanes() int {
	return simd.BroadcastFloat64s(0).Len()
}

func ClipSIMD(dst, src []float64) {
	zero := simd.BroadcastFloat64s(0)
	lanes := zero.Len()
	i := 0
	for ; i+lanes <= len(src); i += lanes {
		simd.LoadFloat64s(src[i:]).Max(zero).Store(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = max(src[i], 0)
	}
}

func AddSIMD(dst, a, b []float64) {
	lanes := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+lanes <= len(a); i += lanes {
		simd.LoadFloat64s(a[i:]).Add(simd.LoadFloat64s(b[i:])).Store(dst[i:])
	}
	for ; i < len(a); i++ {
		dst[i] = a[i] + b[i]
	}
}

func AxpySIMD(dst, x, y []float64, alpha float64) {
	va := simd.BroadcastFloat64s(alpha)
	lanes := va.Len()
	i := 0
	for ; i+lanes <= len(x); i += lanes {
		simd.LoadFloat64s(x[i:]).MulAdd(va, simd.LoadFloat64s(y[i:])).Store(dst[i:])
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func SqrtSIMD(dst, src []float64) {
	lanes := simd.BroadcastFloat64s(0).Len()
	i := 0
	for ; i+lanes <= len(src); i += lanes {
		simd.LoadFloat64s(src[i:]).Sqrt().Store(dst[i:])
	}
	for ; i < len(src); i++ {
		dst[i] = sqrtScalarOne(src[i])
	}
}

func SumSIMD(src []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	lanes := acc.Len()
	i := 0
	for ; i+lanes <= len(src); i += lanes {
		acc = acc.Add(simd.LoadFloat64s(src[i:]))
	}
	buf := make([]float64, lanes)
	acc.Store(buf)
	s := 0.0
	for _, v := range buf {
		s += v
	}
	for ; i < len(src); i++ {
		s += src[i]
	}
	return s
}

func ClipParallelSIMD(dst, src []float64) {
	parallelChunks(len(src), func(lo, hi int) { ClipSIMD(dst[lo:hi], src[lo:hi]) })
}

func AddParallelSIMD(dst, a, b []float64) {
	parallelChunks(len(a), func(lo, hi int) { AddSIMD(dst[lo:hi], a[lo:hi], b[lo:hi]) })
}

func AxpyParallelSIMD(dst, x, y []float64, alpha float64) {
	parallelChunks(len(x), func(lo, hi int) { AxpySIMD(dst[lo:hi], x[lo:hi], y[lo:hi], alpha) })
}

func SqrtParallelSIMD(dst, src []float64) {
	parallelChunks(len(src), func(lo, hi int) { SqrtSIMD(dst[lo:hi], src[lo:hi]) })
}

// SumParallelSIMD reduces with the SIMD kernel on every chunk. Without this,
// the parallel sum silently fell back to scalar accumulation, which caps the
// achievable bandwidth well below what the hardware allows.
func SumParallelSIMD(src []float64) float64 {
	return parallelSum(len(src), func(lo, hi int) float64 {
		return SumSIMD(src[lo:hi])
	})
}
