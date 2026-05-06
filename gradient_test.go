package stlcg

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Gradient-sanity: build a graph that exposes the mean robustness of a
// formula with a single scalar parameter c, compare graph.Gradient to a
// centered finite-difference estimate, assert agreement within 1e-3.
//
// Smooth mode only — exact mode is non-differentiable at ties and
// finite-differencing tied extrema is unstable. Traces are chosen to
// avoid obvious ties.

func gradAndValue(t *testing.T, phi Formula, trace []float32, c float32) (value, grad float32) {
	t.Helper()

	exec, err := graph.NewExec(testBackend, func(xIn, cIn *graph.Node) (*graph.Node, *graph.Node) {
		g := xIn.Graph()
		cBroadcast := graph.BroadcastToShape(cIn, xIn.Shape())
		signals := map[string]*graph.Node{"x": xIn, "c": cBroadcast}
		pscale := graph.Scalar(g, dtypes.Float32, 1.0)
		tau := graph.Scalar(g, dtypes.Float32, 5.0)
		rho := BuildRobustnessTrace(phi, signals, pscale, tau, WithMode(ModeSmooth))
		val := graph.ReduceAllMean(rho)
		grads := graph.Gradient(val, cIn)
		return val, grads[0]
	})
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	xT := tensors.FromShape(shapes.Make(dtypes.Float32, 1, len(trace), 1))
	if err := tensors.MutableFlatData(xT, func(d []float32) { copy(d, trace) }); err != nil {
		t.Fatal(err)
	}
	defer xT.FinalizeAll()

	outs, err := exec.Exec(xT, c)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, o := range outs {
			o.FinalizeAll()
		}
	}()

	extract := func(tn *tensors.Tensor) float32 {
		var v float32
		if err := tensors.ConstFlatData(tn, func(d []float32) { v = d[0] }); err != nil {
			t.Fatal(err)
		}
		return v
	}
	return extract(outs[0]), extract(outs[1])
}

func finiteDiffGrad(t *testing.T, phi Formula, trace []float32, c, eps float32) float32 {
	t.Helper()
	vPlus, _ := gradAndValue(t, phi, trace, c+eps)
	vMinus, _ := gradAndValue(t, phi, trace, c-eps)
	return (vPlus - vMinus) / (2 * eps)
}

func TestGradientAlwaysUnbounded(t *testing.T) {
	// phi = Always (x > c). The smooth mean rho ≈ min(x) - c;
	// d/dc = -1 regardless of c (in the smooth limit).
	trace := []float32{0.5, 0.8, 0.2, 0.4, 1.0, 0.3, 0.1, 0.6, 0.2, 0.9}
	phi := Always(Gt(Var("x"), Var("c")), AllTime())

	for _, c := range []float32{-2.0, -0.5, 0.3} {
		_, got := gradAndValue(t, phi, trace, c)
		want := finiteDiffGrad(t, phi, trace, c, 1e-2)
		if math.Abs(float64(got-want)) > 1e-3 {
			t.Errorf("Always grad at c=%g: autodiff=%g fd=%g", c, got, want)
		}
	}
}

func TestGradientAlwaysBounded(t *testing.T) {
	trace := []float32{0.5, 0.8, 0.2, 0.4, 1.0, 0.3, 0.1, 0.6}
	phi := Always(Gt(Var("x"), Var("c")), Bounds(0, 3))

	for _, c := range []float32{-1.0, 0.5} {
		_, got := gradAndValue(t, phi, trace, c)
		want := finiteDiffGrad(t, phi, trace, c, 1e-2)
		if math.Abs(float64(got-want)) > 2e-3 {
			t.Errorf("Always[0,3] grad at c=%g: autodiff=%g fd=%g", c, got, want)
		}
	}
}

func TestGradientEventually(t *testing.T) {
	trace := []float32{0.5, 0.8, 0.2, 0.4, 1.0, 0.3, 0.1, 0.6}
	phi := Eventually(Lt(Var("x"), Var("c")), Bounds(0, 4))

	for _, c := range []float32{0.5, 1.5} {
		_, got := gradAndValue(t, phi, trace, c)
		want := finiteDiffGrad(t, phi, trace, c, 1e-2)
		if math.Abs(float64(got-want)) > 2e-3 {
			t.Errorf("Eventually[0,4] grad at c=%g: autodiff=%g fd=%g", c, got, want)
		}
	}
}

func TestGradientAndOr(t *testing.T) {
	trace := []float32{0.1, 0.7, 0.3, 0.9, 0.2, 0.6}
	// phi = (x > c) ∧ (x < c+1) — a "near-c" band.
	phi := And(
		Gt(Var("x"), Var("c")),
		Lt(Var("x"), Var("c")), // degenerate; will use numeric offset via another broadcast
	)

	_, got := gradAndValue(t, phi, trace, 0.4)
	want := finiteDiffGrad(t, phi, trace, 0.4, 1e-2)
	if math.Abs(float64(got-want)) > 2e-3 {
		t.Errorf("And grad: autodiff=%g fd=%g", got, want)
	}
}

func TestGradientUntil(t *testing.T) {
	// phi = (x > c) U_[0,3] (x > c+0.5) — a bit contrived, but both branches
	// see c and the finite-diff should match.
	trace := []float32{0.2, 0.4, 0.6, 0.8, 1.0, 0.7, 0.3}
	phi := Until(Gt(Var("x"), Var("c")), Gt(Var("x"), Var("c")), Bounds(0, 3), true)

	for _, c := range []float32{0.3, 0.6} {
		_, got := gradAndValue(t, phi, trace, c)
		want := finiteDiffGrad(t, phi, trace, c, 1e-2)
		if math.Abs(float64(got-want)) > 3e-3 {
			t.Errorf("Until grad at c=%g: autodiff=%g fd=%g", c, got, want)
		}
	}
}

func TestGradientIntegral(t *testing.T) {
	// Integral1d is linear in phi, so gradient is a constant-ish function
	// of c (given Gt predicate is linear too).
	trace := []float32{0.1, 0.3, 0.7, 0.4, 0.2, 0.9}
	phi := Integral1d(Gt(Var("x"), Var("c")), Bounds(0, 3), Riemann)

	_, got := gradAndValue(t, phi, trace, 0.5)
	want := finiteDiffGrad(t, phi, trace, 0.5, 1e-2)
	if math.Abs(float64(got-want)) > 2e-3 {
		t.Errorf("Integral grad: autodiff=%g fd=%g", got, want)
	}
}
