package stlcg

import (
	"errors"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

func TestRobustnessTraceE_MissingSignal(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	defer eval.Close()

	// Deliberately supply the wrong variable name.
	y := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 4, 1))
	defer y.FinalizeAll()

	_, err := eval.RobustnessTraceE(SignalMap{"y": y})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrMissingSignal) {
		t.Errorf("err = %v, want ErrMissingSignal", err)
	}
}

func TestRobustnessTraceE_Closed(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	eval.Close()

	x := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 4, 1))
	defer x.FinalizeAll()

	_, err := eval.RobustnessTraceE(SignalMap{"x": x})
	if !errors.Is(err, ErrClosed) {
		t.Errorf("err = %v, want ErrClosed", err)
	}
}

func TestRobustnessE_TimeOutOfRange(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	defer eval.Close()

	x := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 4, 1))
	defer x.FinalizeAll()

	_, err := eval.RobustnessE(SignalMap{"x": x}, AtTime(99))
	if !errors.Is(err, ErrTimeOutOfRange) {
		t.Errorf("err = %v, want ErrTimeOutOfRange", err)
	}

	_, err = eval.RobustnessE(SignalMap{"x": x}, AtTime(-99))
	if !errors.Is(err, ErrTimeOutOfRange) {
		t.Errorf("neg OOB err = %v, want ErrTimeOutOfRange", err)
	}
}

func TestPanicsOnClosedForBackwardCompat(t *testing.T) {
	phi := Gt(Var("x"), Const(0.0))
	eval := NewEvaluator(testBackend, phi)
	eval.Close()

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		err, ok := r.(error)
		if !ok {
			t.Fatalf("panic was not an error: %v", r)
		}
		if !errors.Is(err, ErrClosed) {
			t.Errorf("panic err = %v, want wraps ErrClosed", err)
		}
	}()

	x := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 4, 1))
	defer x.FinalizeAll()
	_ = eval.RobustnessTrace(SignalMap{"x": x}) // should panic with wrapped ErrClosed
}
