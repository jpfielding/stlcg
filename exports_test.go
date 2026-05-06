package stlcg

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// TestPublicAPISmoke exercises every public method of Evaluator plus the
// convenience constructors so we notice if a refactor silently breaks
// the canary surface. This is deliberately end-to-end rather than
// per-unit.
func TestPublicAPISmoke(t *testing.T) {
	phi := Always(Gt(Var("speed"), Const(0.5)), Bounds(0, 2))

	eval := NewEvaluator(testBackend, phi, WithMode(ModeSmooth), WithScale(3.0))
	defer eval.Close()

	eval.SetMaxCache(4)

	if vars := eval.Vars(); len(vars) != 1 || vars[0] != "speed" {
		t.Fatalf("Vars() = %v, want [speed]", vars)
	}

	if err := eval.Precompile([2]int{1, 6}); err != nil {
		t.Fatalf("Precompile: %v", err)
	}

	tn := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 6, 1))
	if err := tensors.MutableFlatData(tn, func(d []float32) {
		for i := range d {
			d[i] = float32(i) * 0.2
		}
	}); err != nil {
		t.Fatal(err)
	}
	defer tn.FinalizeAll()

	sig := NewSignalMap(map[string]*tensors.Tensor{"speed": tn})

	trace := eval.RobustnessTrace(sig)
	if r := trace.Shape().Rank(); r != 3 {
		t.Errorf("RobustnessTrace shape rank = %d, want 3", r)
	}
	trace.FinalizeAll()

	single := eval.Robustness(sig, AtTime(0))
	if r := single.Shape().Rank(); r != 2 {
		t.Errorf("Robustness shape rank = %d, want 2", r)
	}
	single.FinalizeAll()

	// Negative AtTime should index from the end.
	last := eval.Robustness(sig, AtTime(-1))
	last.FinalizeAll()

	// Close is idempotent and safe after completion.
	eval.Close()
	eval.Close()
}
