package stlcg

import (
	"math/rand/v2"
	"testing"
)

func TestAlwaysBoundedExact(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{
		"x": {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	cases := []Formula{
		Always(Gt(x, Const(0.0)), Bounds(0, 2)),
		Always(Gt(x, Const(0.0)), Bounds(1, 4)),
		Eventually(Gt(x, Const(5.0)), Bounds(0, 3)),
		Eventually(Lt(x, Const(0.0)), Bounds(0, 5)),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestAlwaysUnboundedExact(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{
		"x": {2, 3, 4, 5, 6, 5, 4, 3},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	cases := []Formula{
		Always(Gt(x, Const(0.0)), AllTime()),
		Always(Gt(x, Const(0.0)), From(2)),
		Eventually(Gt(x, Const(5.5)), AllTime()),
		Eventually(Lt(x, Const(3.5)), From(4)),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestAlwaysEventuallyParitySmooth(t *testing.T) {
	x := Var("x")
	rng := rand.New(rand.NewPCG(42, 17))
	row := make([]float64, 24)
	for i := range row {
		row[i] = rng.Float64()*4 - 2
	}
	signals := map[string][]float64{"x": row}
	cfg := config{mode: ModeSmooth, pscale: 1.0, scale: 5.0}

	cases := []Formula{
		Always(Gt(x, Const(0.0)), Bounds(0, 3)),
		Always(Gt(x, Const(0.0)), Bounds(2, 5)),
		Eventually(Gt(x, Const(1.0)), Bounds(0, 4)),
		Eventually(Lt(x, Const(-1.0)), Bounds(1, 3)),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 5e-3)
		})
	}
}

func TestNestedTemporalParity(t *testing.T) {
	// Asymmetric nested bounded ops — Codex-critical case.
	x := Var("x")
	signals := map[string][]float64{
		"x": {0.5, 1.5, 2.5, 1.0, 0.5, 2.0, 3.0, 2.5, 1.0, 0.0, -0.5, 0.5, 1.5, 2.0, 2.5},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	cases := []Formula{
		Always(Eventually(Gt(x, Const(2.0)), Bounds(0, 2)), Bounds(1, 3)),
		Eventually(Always(Gt(x, Const(0.0)), Bounds(0, 2)), Bounds(0, 4)),
		Always(And(Gt(x, Const(-1.0)), Lt(x, Const(3.0))), Bounds(0, 5)),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestTemporalEdgeIntervals(t *testing.T) {
	x := Var("x")
	signals := map[string][]float64{"x": {3, 2, 1, 0, -1, -2}}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	cases := []Formula{
		// Single-point interval: Always[2,2] phi at t = phi at t+2.
		Always(Gt(x, Const(0.0)), Bounds(2, 2)),
		// Eventually[0,0] phi at t = phi at t.
		Eventually(Gt(x, Const(0.0)), Bounds(0, 0)),
		// Window spanning the whole remaining trace.
		Always(Gt(x, Const(-10.0)), Bounds(0, 5)),
	}
	for _, phi := range cases {
		t.Run(phi.String(), func(t *testing.T) {
			runParity(t, phi, signals, cfg, 1e-5)
		})
	}
}

func TestTemporalWithLogicalInside(t *testing.T) {
	x, y := Var("x"), Var("y")
	signals := map[string][]float64{
		"x": {-1, 0, 1, 2, 3, 4, 5, 6},
		"y": {10, 8, 6, 4, 2, 0, -2, -4},
	}
	cfg := config{mode: ModeExact, pscale: 1.0, scale: 0}

	phi := Always(
		And(Gt(x, Const(0.0)), Lt(y, Const(10.0))),
		Bounds(1, 3),
	)
	runParity(t, phi, signals, cfg, 1e-5)
}
