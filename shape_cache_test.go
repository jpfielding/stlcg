package stlcg

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// TestShapeCacheRoundTrip verifies that the Evaluator correctly handles a
// sequence of distinct (batch, timeLen) pairs without errors, including
// returning to an earlier shape. A cache-key bug that conflated two
// shapes would surface as a shape-mismatch error on the first reuse.
//
// This does not directly measure compile count (gomlx does not expose
// that), but it exercises the path a misconfigured cache would break.
func TestShapeCacheRoundTrip(t *testing.T) {
	phi := Always(Gt(Var("x"), Const(0.0)), Bounds(0, 4))
	eval := NewEvaluator(testBackend, phi, WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()
	eval.SetMaxCache(8)

	mk := func(b, tLen int) *tensors.Tensor {
		tn := tensors.FromShape(shapes.Make(dtypes.Float32, b, tLen, 1))
		if err := tensors.MutableFlatData(tn, func(d []float32) {
			for i := range d {
				d[i] = float32(i%5) - 2
			}
		}); err != nil {
			t.Fatal(err)
		}
		return tn
	}

	seq := [][2]int{
		{1, 8}, {2, 8}, {1, 16}, {2, 16},
		{1, 8}, {1, 16}, // repeats hit cache
		{4, 12}, // new shape
	}

	for _, sh := range seq {
		tn := mk(sh[0], sh[1])
		out := eval.RobustnessTrace(SignalMap{"x": tn})
		// The output must match input shape (same batch, time, feature).
		outShape := out.Shape()
		if outShape.Dimensions[0] != sh[0] || outShape.Dimensions[1] != sh[1] {
			t.Errorf("shape %v: out dims = %v, want first two = %v", sh, outShape.Dimensions, sh)
		}
		out.FinalizeAll()
		tn.FinalizeAll()
	}
}

// TestShapeCacheMaxCapIsHard documents that SetMaxCache is a HARD cap,
// not an LRU: the underlying gomlx Exec panics when a third distinct
// shape arrives at SetMaxCache(2). If this ever becomes an LRU upstream
// the test will need to flip.
func TestShapeCacheMaxCapIsHard(t *testing.T) {
	phi := Eventually(Gt(Var("x"), Const(0.0)), Bounds(0, 3))
	eval := NewEvaluator(testBackend, phi, WithMode(ModeSmooth), WithScale(5.0))
	defer eval.Close()
	eval.SetMaxCache(2)

	mk := func(b int) *tensors.Tensor {
		tn := tensors.FromShape(shapes.Make(dtypes.Float32, b, 6, 1))
		if err := tensors.MutableFlatData(tn, func(d []float32) {
			for i := range d {
				d[i] = 0.1 * float32(i)
			}
		}); err != nil {
			t.Fatal(err)
		}
		return tn
	}

	// Fill the cap with shapes b=1 and b=2.
	for _, b := range []int{1, 2} {
		tn := mk(b)
		out := eval.RobustnessTrace(SignalMap{"x": tn})
		out.FinalizeAll()
		tn.FinalizeAll()
	}

	// Re-hits the populated cache; must succeed.
	for _, b := range []int{1, 2, 1, 2} {
		tn := mk(b)
		out := eval.RobustnessTrace(SignalMap{"x": tn})
		out.FinalizeAll()
		tn.FinalizeAll()
	}

	// A third distinct shape overflows the hard cap and panics. The test
	// succeeds if the panic comes through with a cache-size message.
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic on cache overflow, got none")
			return
		}
		msg := r.(error).Error()
		if !containsSubstr(msg, "cache") {
			t.Errorf("expected cache error, got: %v", msg)
		}
	}()

	tn := mk(3)
	defer tn.FinalizeAll()
	out := eval.RobustnessTrace(SignalMap{"x": tn})
	out.FinalizeAll()
}

func containsSubstr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
