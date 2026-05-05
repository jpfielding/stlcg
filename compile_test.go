package stlcg

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// testBackend returns a singleton gomlx backend. Backend creation is
// expensive; share across parity tests.
var testBackend = func() backends.Backend {
	return backends.MustNew()
}()

// tensorFromRow wraps a single [T] slice as a [1, T, 1] float32 tensor.
func tensorFromRow(row []float64) *tensors.Tensor {
	t := tensors.FromShape(shapes.Make(dtypes.Float32, 1, len(row), 1))
	if err := tensors.MutableFlatData(t, func(data []float32) {
		for i, v := range row {
			data[i] = float32(v)
		}
	}); err != nil {
		panic(err)
	}
	return t
}

// tensorRow extracts the [1, T, 1] tensor's row as a []float64.
func tensorRow(t *tensors.Tensor) []float64 {
	s := t.Shape()
	T := s.Dimensions[1]
	out := make([]float64, T)
	if err := tensors.ConstFlatData(t, func(data []float32) {
		for i := 0; i < T; i++ {
			out[i] = float64(data[i])
		}
	}); err != nil {
		panic(err)
	}
	return out
}

// assertClose compares two slices with absolute tolerance tol.
func assertClose(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d] = %g, want %g (|Δ|=%g)", name, i, got[i], want[i], math.Abs(got[i]-want[i]))
		}
	}
}

// runParity compiles phi and the reference evaluator under cfg, feeds
// signals, and asserts outputs match within tol.
func runParity(t *testing.T, phi Formula, signals map[string][]float64, cfg config, tol float64) {
	t.Helper()

	var opts []Option
	opts = append(opts, WithMode(cfg.mode), WithPScale(cfg.pscale), WithScale(cfg.scale))
	eval := NewEvaluator(testBackend, phi, opts...)
	defer eval.Close()

	sigMap := make(SignalMap, len(signals))
	for k, v := range signals {
		sigMap[k] = tensorFromRow(v)
	}

	trace := eval.RobustnessTrace(sigMap)
	defer trace.FinalizeAll()

	ref := newRefEvaluator(cfg).rho(phi, signals)
	got := tensorRow(trace)

	assertClose(t, phi.String(), got, ref, tol)
}

func TestPredicateParityExact(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{
		"x": {-1, 0, 1, 2, 3, 4, 5},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}
	cases := []Formula{
		Gt(x, Const(2.0)),
		Lt(x, Const(2.0)),
		Ge(x, Const(0.0)),
		Le(x, Const(4.0)),
		Eq(x, Const(3.0)),
		Identity(x),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestPredicateParityWithPScale(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{"x": {-2, -1, 0, 1, 2}}
	cfg := config{mode: ModeExact, pscale: 3.5, scale: 0}
	runParity(t, Gt(x, Const(0.0)), signals, cfg, 1e-5)
}

func TestLogicalParityExact(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {-1, 0, 1, 2, 3},
		"y": {5, 4, 3, 2, 1},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}
	cases := []Formula{
		And(Gt(x, Const(0.0)), Lt(y, Const(4.0))),
		Or(Gt(x, Const(0.0)), Lt(y, Const(0.0))),
		Not(Gt(x, Const(1.5))),
		Implies(Gt(x, Const(0.0)), Lt(y, Const(5.0))),
		And(And(Gt(x, Const(-5.0)), Lt(x, Const(5.0))), Gt(y, Const(0.0))),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestLogicalParitySmooth(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {-1.5, 0.5, 1.5, 2.5, 3.5},
		"y": {4.5, 3.5, 2.5, 1.5, 0.5},
	}
	// tau=5 gives a tight smooth approximation without overflow risk
	cfg := config{mode: ModeSmooth, pscale: 1.0, scale: 5.0}
	cases := []Formula{
		And(Gt(x, Const(0.0)), Lt(y, Const(4.0))),
		Or(Gt(x, Const(0.0)), Lt(y, Const(0.0))),
		Not(And(Gt(x, Const(0.0)), Lt(y, Const(4.0)))),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-4)
		})
	}
}

func TestDoubleNegationSemantics(t *testing.T) {
	// ¬¬φ must equal φ at the robustness level in both modes.
	x := Var("x")
	signals := map[string][]float64{"x": {-1, 0, 1, 2, 3}}
	phi := Gt(x, Const(1.0))
	nn := Not(Not(phi))

	for _, cfg := range []config{
		{mode: ModeExact, pscale: 1.0, scale: 0},
		{mode: ModeSmooth, pscale: 1.0, scale: 4.0},
	} {
		t.Run(cfg.mode.String(), func(t *testing.T) {
			e1 := NewEvaluator(testBackend, phi, WithMode(cfg.mode), WithScale(cfg.scale))
			defer e1.Close()
			e2 := NewEvaluator(testBackend, nn, WithMode(cfg.mode), WithScale(cfg.scale))
			defer e2.Close()

			sig := SignalMap{"x": tensorFromRow(signals["x"])}

			t1 := e1.RobustnessTrace(sig)
			defer t1.FinalizeAll()
			t2 := e2.RobustnessTrace(sig)
			defer t2.FinalizeAll()

			assertClose(t, "¬¬φ ≡ φ", tensorRow(t2), tensorRow(t1), 1e-5)
		})
	}
}

func TestDeMorganSmooth(t *testing.T) {
	// Smooth-mode De Morgan: And(a,b) = Not(Or(Not a, Not b)) at matched tau.
	// This holds exactly because Minish(a,b) = -Maxish(-a,-b) at any tau.
	x, y := Var("x"), Var("y")
	signals := SignalMap{
		"x": tensorFromRow([]float64{-1, 0, 1, 2, 3}),
		"y": tensorFromRow([]float64{4, 3, 2, 1, 0}),
	}
	a := Gt(x, Const(0.0))
	b := Lt(y, Const(3.0))

	lhs := NewEvaluator(testBackend, And(a, b), WithMode(ModeSmooth), WithScale(5.0))
	defer lhs.Close()
	rhs := NewEvaluator(testBackend, Not(Or(Not(a), Not(b))), WithMode(ModeSmooth), WithScale(5.0))
	defer rhs.Close()

	tl := lhs.RobustnessTrace(signals)
	defer tl.FinalizeAll()
	tr := rhs.RobustnessTrace(signals)
	defer tr.FinalizeAll()

	assertClose(t, "De Morgan", tensorRow(tr), tensorRow(tl), 1e-4)
}

func TestSmoothConvergesToExact(t *testing.T) {
	// As tau grows, smooth min/max approaches exact values for non-tied inputs.
	x := Var("x")
	rng := rand.New(rand.NewPCG(1, 2))
	row := make([]float64, 32)
	for i := range row {
		row[i] = rng.Float64()*4 - 2
	}
	signals := SignalMap{"x": tensorFromRow(row)}
	phi := And(Gt(x, Const(0.0)), Lt(x, Const(1.0)))

	exactEval := NewEvaluator(testBackend, phi, WithMode(ModeExact), WithScale(0))
	defer exactEval.Close()
	smoothEval := NewEvaluator(testBackend, phi, WithMode(ModeSmooth), WithScale(50.0))
	defer smoothEval.Close()

	exactT := exactEval.RobustnessTrace(signals)
	defer exactT.FinalizeAll()
	smoothT := smoothEval.RobustnessTrace(signals)
	defer smoothT.FinalizeAll()

	// tau=50 on values with |diff| >= 0.5 gets within ~1e-2 easily
	assertClose(t, "smooth→exact", tensorRow(smoothT), tensorRow(exactT), 5e-2)
}

func TestEvaluatorMissingVarPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on missing signal")
		}
	}()
	e := NewEvaluator(testBackend, Gt(Var("x"), Const(0.0)))
	defer e.Close()
	e.RobustnessTrace(SignalMap{}) // no "x" binding
}

func TestRobustnessAtTime(t *testing.T) {
	x := Var("x")
	phi := Gt(x, Const(0.0))
	e := NewEvaluator(testBackend, phi, WithMode(ModeExact), WithScale(0))
	defer e.Close()

	signals := SignalMap{"x": tensorFromRow([]float64{-1, 0, 1, 2, 3})}

	tr := e.RobustnessTrace(signals)
	defer tr.FinalizeAll()
	fullRow := tensorRow(tr)

	cases := []struct {
		t    int
		want float64
	}{
		{0, fullRow[0]},
		{2, fullRow[2]},
		{-1, fullRow[4]},
	}
	for _, c := range cases {
		out := e.Robustness(signals, AtTime(c.t))
		err := tensors.ConstFlatData(out, func(data []float32) {
			if math.Abs(float64(data[0])-c.want) > 1e-5 {
				t.Errorf("Robustness(AtTime(%d)) = %g, want %g", c.t, data[0], c.want)
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		out.FinalizeAll()
	}
}
