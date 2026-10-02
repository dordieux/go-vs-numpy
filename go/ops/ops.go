// Package ops provides the float64 kernels being benchmarked.
//
// Each operation comes in up to four flavors:
//   - Scalar:       plain for loop
//   - Parallel:     scalar loop, chunked across GOMAXPROCS goroutines
//   - SIMD:         portable simd package (requires GOEXPERIMENT=simd, Go >= 1.27)
//   - ParallelSIMD: chunked goroutines running the SIMD kernel
package ops

import (
	"math"
	"runtime"
	"sync"
)

// ---- scalar kernels ----

func ClipScalar(dst, src []float64) {
	for i, v := range src {
		if v < 0 {
			dst[i] = 0
		} else {
			dst[i] = v
		}
	}
}

func AddScalar(dst, a, b []float64) {
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

// AxpyScalar computes dst = alpha*x + y (BLAS level-1 axpy).
func AxpyScalar(dst, x, y []float64, alpha float64) {
	for i := range x {
		dst[i] = alpha*x[i] + y[i]
	}
}

func sqrtScalarOne(v float64) float64 { return math.Sqrt(v) }

func SqrtScalar(dst, src []float64) {
	for i, v := range src {
		dst[i] = math.Sqrt(v)
	}
}

func ExpScalar(dst, src []float64) {
	for i, v := range src {
		dst[i] = math.Exp(v)
	}
}

func SumScalar(src []float64) float64 {
	s := 0.0
	for _, v := range src {
		s += v
	}
	return s
}

// ---- parallel harness ----

// parallelChunks splits [0, n) into GOMAXPROCS contiguous chunks and runs fn
// on each chunk in its own goroutine. No size threshold on purpose: the
// benchmark wants to expose goroutine overhead at small n, not hide it.
func parallelChunks(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	wg.Wait()
}

// ---- parallel kernels (scalar body) ----

func ClipParallel(dst, src []float64) {
	parallelChunks(len(src), func(lo, hi int) { ClipScalar(dst[lo:hi], src[lo:hi]) })
}

func AddParallel(dst, a, b []float64) {
	parallelChunks(len(a), func(lo, hi int) { AddScalar(dst[lo:hi], a[lo:hi], b[lo:hi]) })
}

func AxpyParallel(dst, x, y []float64, alpha float64) {
	parallelChunks(len(x), func(lo, hi int) { AxpyScalar(dst[lo:hi], x[lo:hi], y[lo:hi], alpha) })
}

func SqrtParallel(dst, src []float64) {
	parallelChunks(len(src), func(lo, hi int) { SqrtScalar(dst[lo:hi], src[lo:hi]) })
}

func ExpParallel(dst, src []float64) {
	parallelChunks(len(src), func(lo, hi int) { ExpScalar(dst[lo:hi], src[lo:hi]) })
}

// cacheLineFloats is how many float64 fit in a 64-byte cache line. Partial
// sums are spaced this far apart so that two workers never write into the
// same line; without the padding the stores contend even though each worker
// only writes once.
const cacheLineFloats = 8

// parallelSum splits [0, n) across GOMAXPROCS goroutines, reduces each chunk
// with sumChunk, and adds the partial results. sumChunk is what decides
// whether the per-chunk work is scalar or SIMD.
func parallelSum(n int, sumChunk func(lo, hi int) float64) float64 {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		return sumChunk(0, n)
	}
	chunk := (n + workers - 1) / workers
	nChunks := (n + chunk - 1) / chunk
	partials := make([]float64, nChunks*cacheLineFloats)

	var wg sync.WaitGroup
	for w, lo := 0, 0; lo < n; w, lo = w+1, lo+chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			partials[w*cacheLineFloats] = sumChunk(lo, hi)
		}(w, lo, hi)
	}
	wg.Wait()

	total := 0.0
	for i := 0; i < nChunks; i++ {
		total += partials[i*cacheLineFloats]
	}
	return total
}

func SumParallel(src []float64) float64 {
	return parallelSum(len(src), func(lo, hi int) float64 {
		return SumScalar(src[lo:hi])
	})
}
