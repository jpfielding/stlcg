package stlcg

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Benchmarks quantify the static-shape tax and formula-size scaling —
// Codex-flagged items from Phase J. Run with:
//
//	go test -bench=. -benchmem ./...
//
// Important benchmarks:
//
//  BenchmarkCompileAlways_T<n>_L<m> — one-time cost to build + JIT the graph
//  BenchmarkRunAlways_T<n>_L<m>    — steady-state per-call execution cost
//  BenchmarkShapeChurn             — cost of recompiling every call with new shape

func makeTraceTensor(b *testing.B, batch, tLen int) *tensors.Tensor {
	b.Helper()
	tn := tensors.FromShape(shapes.Make(dtypes.Float32, batch, tLen, 1))
	if err := tensors.MutableFlatData(tn, func(d []float32) {
		for i := range d {
			d[i] = float32(i%7) - 3.5 // deterministic ~[-3.5, 3.5]
		}
	}); err != nil {
		b.Fatal(err)
	}
	return tn
}

func benchCompileAlways(b *testing.B, T, L int) {
	phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, L-1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eval := NewEvaluator(testBackend, phi,
			WithMode(ModeSmooth), WithScale(5.0))
		// Trigger actual compilation by running once.
		trace := makeTraceTensor(b, 1, T)
		out := eval.RobustnessTrace(SignalMap{"x": trace})
		out.FinalizeAll()
		trace.FinalizeAll()
		eval.Close()
	}
}

func BenchmarkCompileAlways_T20_L5(b *testing.B)    { benchCompileAlways(b, 20, 5) }
func BenchmarkCompileAlways_T100_L10(b *testing.B)  { benchCompileAlways(b, 100, 10) }
func BenchmarkCompileAlways_T100_L50(b *testing.B)  { benchCompileAlways(b, 100, 50) }
func BenchmarkCompileAlways_T1000_L50(b *testing.B) { benchCompileAlways(b, 1000, 50) }

func benchRunAlways(b *testing.B, T, L int) {
	phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, L-1))
	eval := NewEvaluator(testBackend, phi,
		WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()

	trace := makeTraceTensor(b, 1, T)
	defer trace.FinalizeAll()
	sig := SignalMap{"x": trace}

	// Warm up JIT cache.
	warm := eval.RobustnessTrace(sig)
	warm.FinalizeAll()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := eval.RobustnessTrace(sig)
		out.FinalizeAll()
	}
}

func BenchmarkRunAlways_T20_L5(b *testing.B)    { benchRunAlways(b, 20, 5) }
func BenchmarkRunAlways_T100_L10(b *testing.B)  { benchRunAlways(b, 100, 10) }
func BenchmarkRunAlways_T100_L50(b *testing.B)  { benchRunAlways(b, 100, 50) }
func BenchmarkRunAlways_T1000_L50(b *testing.B) { benchRunAlways(b, 1000, 50) }

// BenchmarkShapeChurn recompiles on every iteration by alternating two
// distinct batch sizes. Upper bound on the static-shape tax.
func BenchmarkShapeChurn(b *testing.B) {
	phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, 9))
	eval := NewEvaluator(testBackend, phi,
		WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()
	eval.SetMaxCache(2) // ensure both shapes stay compiled

	traceA := makeTraceTensor(b, 1, 50)
	defer traceA.FinalizeAll()
	traceB := makeTraceTensor(b, 2, 50)
	defer traceB.FinalizeAll()

	// Warm both shapes.
	a0 := eval.RobustnessTrace(SignalMap{"x": traceA})
	a0.FinalizeAll()
	b0 := eval.RobustnessTrace(SignalMap{"x": traceB})
	b0.FinalizeAll()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out *tensors.Tensor
		if i%2 == 0 {
			out = eval.RobustnessTrace(SignalMap{"x": traceA})
		} else {
			out = eval.RobustnessTrace(SignalMap{"x": traceB})
		}
		out.FinalizeAll()
	}
}

// BenchmarkNestedFormula stresses deeper AST trees.
func BenchmarkNestedFormula(b *testing.B) {
	x := Var("x")
	// Depth-4 tree: Always(Eventually(And(Gt, Lt), [0,3]), [0,5])
	phi := Always(
		Eventually(
			And(
				Gt(x, Const(0.0)),
				Lt(x, Const(1.0)),
			),
			Bounds(0, 3),
		),
		Bounds(0, 5),
	)
	eval := NewEvaluator(testBackend, phi,
		WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()

	trace := makeTraceTensor(b, 1, 80)
	defer trace.FinalizeAll()
	sig := SignalMap{"x": trace}

	warm := eval.RobustnessTrace(sig)
	warm.FinalizeAll()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := eval.RobustnessTrace(sig)
		out.FinalizeAll()
	}
}

// BenchmarkUntilDPUnroll exercises the O(L^2) Until DP.
func BenchmarkUntilDPUnroll(b *testing.B) {
	phi := Until(
		Gt(Var("x"), Const(0.0)),
		Lt(Var("y"), Const(0.0)),
		Bounds(0, 15),
		true,
	)
	eval := NewEvaluator(testBackend, phi,
		WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()

	tx := makeTraceTensor(b, 1, 40)
	ty := makeTraceTensor(b, 1, 40)
	defer tx.FinalizeAll()
	defer ty.FinalizeAll()
	sig := SignalMap{"x": tx, "y": ty}

	warm := eval.RobustnessTrace(sig)
	warm.FinalizeAll()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := eval.RobustnessTrace(sig)
		out.FinalizeAll()
	}
}

// Sanity probe that labels benchmarks with formula stats for the
// reader. Not a real benchmark — runs once on b.N=1.
func BenchmarkFormulaStats(b *testing.B) {
	for _, T := range []int{20, 100, 1000} {
		for _, L := range []int{5, 10, 50} {
			phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, L-1))
			nodes := Walk(phi)
			b.ReportMetric(float64(len(nodes)), fmt.Sprintf("nodes_T%d_L%d", T, L))
		}
	}
	_ = b.N
}
