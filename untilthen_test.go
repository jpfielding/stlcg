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
