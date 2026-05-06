package stlcg

import (
	"math/rand/v2"
	"testing"
)

func TestUntilBoundedExact(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {2, 2, 2, 1, -1, -2, -3, -4, -5, -6},
		"y": {-3, -2, -1, 0, 1, 2, 1, 0, -1, -2},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	cases := []Formula{
		// phi (x>0) until psi (y>0) within [0,6]: at t=0, exists s with y>0 and x>0 held over [0,s].
		Until(Gt(x, Const(0.0)), Gt(y, Const(0.0)), Bounds(0, 6), true),
		Until(Gt(x, Const(0.0)), Gt(y, Const(0.0)), Bounds(2, 5), true),
		Until(Lt(x, Const(0.0)), Lt(y, Const(0.0)), Bounds(0, 5), true),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestUntilUnboundedExact(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {1, 1, 1, 1, 1, 0, -1, -2},
		"y": {-1, -1, -1, -1, 2, 1, 0, -1},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}
	phi := Until(Gt(x, Const(0.0)), Gt(y, Const(0.0)), AllTime(), true)
	runParity(t, phi, signals, cfg, 1e-5)
}

func TestThenExact(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {-1, -1, 2, -1, -1, -1, -1, -1},
		"y": {0, 0, 0, 0, 3, 0, 0, 0},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}
	// Then: phi must have held at some point in [t, s] (max prefix), psi at s.
	// At t=0, x>0 happens at s=2, y>0 happens at s=4; so some s∈[2,4] works.
	phi := Then(Gt(x, Const(0.0)), Gt(y, Const(0.0)), Bounds(0, 7), true)
	runParity(t, phi, signals, cfg, 1e-5)
}

func TestUntilThenSmoothParity(t *testing.T) {
	x, y := Var("x"), Var("y")
	rng := rand.New(rand.NewPCG(7, 11))
	row := func(n int) []float64 {
		r := make([]float64, n)
		for i := range r {
			r[i] = rng.Float64()*4 - 2
		}
		return r
	}
	signals := map[string][]float64{"x": row(16), "y": row(16)}
	cfg := config{mode: ModeSmooth, pscale: 1.0, scale: 5.0}

	cases := []Formula{
		Until(Gt(x, Const(0.0)), Lt(y, Const(0.0)), Bounds(0, 4), true),
		Then(Gt(x, Const(0.0)), Lt(y, Const(0.0)), Bounds(1, 3), true),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 5e-3)
		})
	}
}

func TestIntegralRiemannExact(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{"x": {1, 2, 3, 4, 5, 6, 7, 8}}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	cases := []Formula{
		Integral1d(Identity(x), Bounds(0, 2), Riemann),
		Integral1d(Identity(x), Bounds(1, 4), Riemann),
		Integral1d(Identity(x), Bounds(0, 0), Riemann),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestIntegralTrapezoidExact(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{"x": {1, 2, 3, 4, 5, 6, 7, 8}}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	phi := Integral1d(Identity(x), Bounds(0, 3), Trapezoid)
	runParity(t, phi, signals, cfg, 1e-5)
}

func TestIntegralOverPredicate(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{"x": {-1, 0, 1, 2, 3, 2, 1, 0, -1}}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	phi := Integral1d(Gt(x, Const(1.0)), Bounds(0, 4), Riemann)
	runParity(t, phi, signals, cfg, 1e-5)
}

// TestUntilOverlapFalse exercises the overlap=false branch. The runParity
// harness compares the compiled output to the reference evaluator, which
// now honors overlap per node. We also hand-compute a simple case at t=0
// to pin down the semantics: with overlap=false and a=0, psi[0] alone
// satisfies the until (phi prefix is empty, so its contribution is the
// min-identity +∞, dominated by psi).
func TestUntilOverlapFalse(t *testing.T) {
	x, y := Var("x"), Var("y")

	// Phi is always false on this trace; psi is true at index 2.
	// With overlap=true: Until fails because phi never holds — rho is
	//   dominated by negative phi prefix.
	// With overlap=false: at t=2 psi[2]=+3, phi prefix is empty (identity
	//   is +∞ for min), so inner_k=min(+∞, +3)=3 → positive robustness.
	signals := map[string][]float64{
		"x": {-1, -1, -1, -1, -1}, // phi = (x > 0) always false
		"y": {-1, -1, 3, -1, -1},  // psi = (y > 0) true at t=2
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	phiOverlapFalse := Until(Gt(x, Const(0.0)), Gt(y, Const(0.0)), Bounds(0, 4), false)
	phiOverlapTrue := Until(Gt(x, Const(0.0)), Gt(y, Const(0.0)), Bounds(0, 4), true)

	t.Run("overlap=false matches reference", func(t *testing.T) {
		runParity(t, phiOverlapFalse, signals, cfg, 1e-5)
	})
	t.Run("overlap=true matches reference", func(t *testing.T) {
		runParity(t, phiOverlapTrue, signals, cfg, 1e-5)
	})

	// Values must differ: overlap=false yields positive rho somewhere;
	// overlap=true stays negative everywhere.
	evalF := NewEvaluator(testBackend, phiOverlapFalse, WithMode(ModeExact), WithScale(0))
	defer evalF.Close()
	evalT := NewEvaluator(testBackend, phiOverlapTrue, WithMode(ModeExact), WithScale(0))
	defer evalT.Close()

	sigMap := make(SignalMap, 2)
	for k, v := range signals {
		sigMap[k] = tensorFromRow(v)
	}
	defer func() {
		for _, v := range sigMap {
			v.FinalizeAll()
		}
	}()

	tF := evalF.RobustnessTrace(sigMap)
	tT := evalT.RobustnessTrace(sigMap)
	defer tF.FinalizeAll()
	defer tT.FinalizeAll()

	rowF := tensorRow(tF)
	rowT := tensorRow(tT)

	// overlap=false should reach positive robustness somewhere in the trace
	// (the reachable psi dominates the empty phi-prefix).
	positiveF := false
	for _, v := range rowF {
		if v > 0 {
			positiveF = true
			break
		}
	}
	if !positiveF {
		t.Errorf("overlap=false should reach positive rho; got %v", rowF)
	}

	// overlap=true must never reach positive robustness (phi never holds).
	for i, v := range rowT {
		if v > 0 {
			t.Errorf("overlap=true rho[%d] = %g should be <= 0 (phi never holds)", i, v)
		}
	}
}

func TestComposedAlwaysUntil(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {2, 2, 2, 2, 2, 2, 0, 0},
		"y": {-1, -1, -1, 1, -1, -1, -1, -1},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}
	// Always[0,2] (x U_[0,3] (y > 0)).
	phi := Always(
		Until(Gt(x, Const(1.0)), Gt(y, Const(0.0)), Bounds(0, 3), true),
		Bounds(0, 2),
	)
	runParity(t, phi, signals, cfg, 1e-5)
}
