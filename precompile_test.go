package stlcg

import (
	"errors"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// TestPrecompileWarmsCache drives graph construction for two distinct
// shapes via Precompile, then verifies that subsequent evaluations with
// those shapes do not error — proving the cache was populated.
// We can't directly introspect gomlx's cache size, so this is a smoke
// test that any shape-handling bug in Precompile would surface.
func TestPrecompileWarmsCache(t *testing.T) {
	phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, 3))
	eval := NewEvaluator(testBackend, phi, WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()

	if err := eval.Precompile([2]int{1, 8}, [2]int{2, 16}); err != nil {
		t.Fatalf("Precompile: %v", err)
	}

	// Re-run both shapes; should succeed using cached graphs.
	for _, sh := range [][2]int{{1, 8}, {2, 16}} {
		b, tLen := sh[0], sh[1]
		tn := tensors.FromShape(shapes.Make(dtypes.Float32, b, tLen, 1))
		if err := tensors.MutableFlatData(tn, func(d []float32) {
			for i := range d {
				d[i] = 0.5
			}
		}); err != nil {
			t.Fatal(err)
		}
		out := eval.RobustnessTrace(SignalMap{"x": tn})
		out.FinalizeAll()
		tn.FinalizeAll()
	}
}

func TestPrecompileRejectsBadShapes(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	defer eval.Close()

	for _, bad := range [][2]int{{0, 8}, {1, 0}, {-1, 8}} {
		if err := eval.Precompile(bad); err == nil {
			t.Errorf("Precompile(%v) expected error, got nil", bad)
		}
	}
}

// TestPrecompileSurfacesClosedError closes the evaluator, then calls
// Precompile and asserts the error surfaces (not panics). The prior
// implementation called panicking RobustnessTrace and leaked tensors.
func TestPrecompileSurfacesClosedError(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	eval.Close()

	err := eval.Precompile([2]int{1, 4})
	if err == nil {
		t.Fatal("Precompile on closed evaluator: expected error, got nil")
	}
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("Precompile on closed evaluator: expected ErrClosed, got %v", err)
	}
}
