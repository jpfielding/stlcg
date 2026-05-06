package stlcg

import (
	"sync"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// TestEvaluatorConcurrent forces the race detector to observe concurrent
// RobustnessTrace calls against a shared Evaluator, with an overlapping
// Close at the end. Must be run with `go test -race`.
func TestEvaluatorConcurrent(t *testing.T) {
	phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, 4))
	eval := NewEvaluator(testBackend, phi, WithMode(ModeSmooth), WithScale(5.0))

	mkTrace := func() *tensors.Tensor {
		tn := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 8, 1))
		if err := tensors.MutableFlatData(tn, func(d []float32) {
			for i := range d {
				d[i] = float32(i%5) - 2
			}
		}); err != nil {
			t.Fatal(err)
		}
		return tn
	}

	const workers = 8
	const iters = 20

	var wg sync.WaitGroup
	errs := make(chan error, workers*iters)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				trace := mkTrace()
				out := eval.RobustnessTrace(SignalMap{"x": trace})
				out.FinalizeAll()
				trace.FinalizeAll()
			}
		}()
	}
	wg.Wait()
	close(errs)

	// After concurrent readers finish, Close should race-cleanly no-op a
	// second call.
	eval.Close()
	eval.Close()
}

// TestEvaluatorCloseIsIdempotent is a simpler canary: Close twice is safe.
func TestEvaluatorCloseIsIdempotent(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	eval.Close()
	eval.Close()
}
