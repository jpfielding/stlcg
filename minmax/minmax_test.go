package minmax

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

var testBackend = func() backends.Backend {
	return backends.MustNew()
}()

// asFloat32 fills a 1-D tensor of length len(vals) with the given floats.
func asFloat32(vals []float64) *tensors.Tensor {
	tn := tensors.FromShape(shapes.Make(dtypes.Float32, len(vals)))
	if err := tensors.MutableFlatData(tn, func(d []float32) {
		for i, v := range vals {
			d[i] = float32(v)
		}
	}); err != nil {
		panic(err)
	}
	return tn
}

// runReduce builds `reduce(x, axis=0, tau, mode, tie)` and returns the
// scalar result.
func runReduce(t *testing.T, x []float64, tauVal float64, mode Mode, tie TiePolicy, wantMax bool) float64 {
	t.Helper()

	fn := func(xIn, tauIn *graph.Node) *graph.Node {
		if wantMax {
			return Maxish(xIn, 0, tauIn, mode, tie, false)
		}
		return Minish(xIn, 0, tauIn, mode, tie, false)
	}

	exec, err := graph.NewExec(testBackend, fn)
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	xT := asFloat32(x)
	defer xT.FinalizeAll()
	out, err := exec.Exec1(xT, float32(tauVal))
	if err != nil {
		t.Fatal(err)
	}
	defer out.FinalizeAll()

	var got float32
	if err := tensors.ConstFlatData(out, func(d []float32) { got = d[0] }); err != nil {
		t.Fatal(err)
	}
	return float64(got)
}

func TestSmoothApproximatesExactAtLargeTau(t *testing.T) {
	// smooth(max, tau=50) should be within 1e-2 of the true max for
	// well-separated inputs.
	vals := []float64{-1.5, 0.2, 3.7, 1.1, -2.0}
	exactMax := runReduce(t, vals, 1.0, Exact, Argmax, true)
	smoothMax := runReduce(t, vals, 50.0, Smooth, Argmax, true)
	if math.Abs(exactMax-3.7) > 1e-5 {
		t.Errorf("Exact max: got %g, want 3.7", exactMax)
	}
	if math.Abs(smoothMax-exactMax) > 1e-2 {
		t.Errorf("Smooth max @tau=50: got %g, exact %g (diff %g)", smoothMax, exactMax, math.Abs(smoothMax-exactMax))
	}

	exactMin := runReduce(t, vals, 1.0, Exact, Argmax, false)
	smoothMin := runReduce(t, vals, 50.0, Smooth, Argmax, false)
	if math.Abs(exactMin+2.0) > 1e-5 {
		t.Errorf("Exact min: got %g, want -2.0", exactMin)
	}
	if math.Abs(smoothMin-exactMin) > 1e-2 {
		t.Errorf("Smooth min @tau=50: got %g, exact %g (diff %g)", smoothMin, exactMin, math.Abs(smoothMin-exactMin))
	}
}

func TestSingleElementAxis(t *testing.T) {
	// Axis with one element should reduce to that element regardless of
	// mode, tie policy, or tau.
	for _, mode := range []Mode{Exact, Smooth} {
		for _, tie := range []TiePolicy{Argmax, Uniform} {
			for _, wantMax := range []bool{true, false} {
				got := runReduce(t, []float64{2.5}, 5.0, mode, tie, wantMax)
				if math.Abs(got-2.5) > 1e-4 {
					t.Errorf("L=1 (mode=%v tie=%v max=%v): got %g want 2.5", mode, tie, wantMax, got)
				}
			}
		}
	}
}

// TestSmoothTauNearZeroDiverges documents the tau -> 0 divergence:
// (1/tau) * log(sum(exp(0))) = log(N)/tau -> infinity. This is
// mathematically correct (tau=0 is not the arithmetic mean limit;
// it is degenerate). The test is documentation; callers who see
// blowups at tiny tau should switch to ModeExact or raise tau.
func TestSmoothTauNearZeroDiverges(t *testing.T) {
	vals := []float64{1, 2, 3}
	got := runReduce(t, vals, 1e-6, Smooth, Argmax, true)
	if !math.IsInf(got, 0) && math.Abs(got) < 1e5 {
		t.Errorf("tau near zero should diverge; got %g (expected |val| > 1e5 or Inf)", got)
	}
}

// TestExactTieUniformGradient exercises the custom stop-gradient tie mask
// path in exactExtremum. Inputs [2, 2, 2, 0, 0] under ReduceMax with
// TieUniform should split d(max)/dx_i uniformly across the tied maxima.
// Expected: [1/3, 1/3, 1/3, 0, 0].
func TestExactTieUniformGradient(t *testing.T) {
	fn := func(xIn *graph.Node) (*graph.Node, *graph.Node) {
		tau := graph.Scalar(xIn.Graph(), dtypes.Float32, 1.0)
		y := Maxish(xIn, 0, tau, Exact, Uniform, false)
		grads := graph.Gradient(y, xIn)
		return y, grads[0]
	}

	exec, err := graph.NewExec(testBackend, fn)
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	xT := asFloat32([]float64{2, 2, 2, 0, 0})
	defer xT.FinalizeAll()
	outs, err := exec.Exec(xT)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, o := range outs {
			o.FinalizeAll()
		}
	}()

	var y float32
	if err := tensors.ConstFlatData(outs[0], func(d []float32) { y = d[0] }); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y)-2) > 1e-5 {
		t.Errorf("forward: got %g, want 2", y)
	}

	grad := make([]float64, 5)
	if err := tensors.ConstFlatData(outs[1], func(d []float32) {
		for i := range grad {
			grad[i] = float64(d[i])
		}
	}); err != nil {
		t.Fatal(err)
	}

	want := []float64{1.0 / 3, 1.0 / 3, 1.0 / 3, 0, 0}
	for i := range want {
		if math.Abs(grad[i]-want[i]) > 5e-3 {
			t.Errorf("grad[%d] = %g, want %g", i, grad[i], want[i])
		}
	}
}

// TestExactArgmaxGradient documents the default TieArgmax path under
// gomlx/XLA. Empirically, ReduceMax's VJP routes gradient=1 to *each*
// tied maximum slot (total sum = number of ties), not to a single
// argmax slot. This differs from PyTorch's convention but is what the
// Go library actually does. TieUniform is the preferred policy when
// d(max)/dx should sum to 1.
func TestExactArgmaxGradient(t *testing.T) {
	fn := func(xIn *graph.Node) (*graph.Node, *graph.Node) {
		tau := graph.Scalar(xIn.Graph(), dtypes.Float32, 1.0)
		y := Maxish(xIn, 0, tau, Exact, Argmax, false)
		return y, graph.Gradient(y, xIn)[0]
	}
	exec, err := graph.NewExec(testBackend, fn)
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	xT := asFloat32([]float64{2, 2, 2, 0, 0})
	defer xT.FinalizeAll()
	outs, err := exec.Exec(xT)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, o := range outs {
			o.FinalizeAll()
		}
	}()

	grad := make([]float64, 5)
	if err := tensors.ConstFlatData(outs[1], func(d []float32) {
		for i := range grad {
			grad[i] = float64(d[i])
		}
	}); err != nil {
		t.Fatal(err)
	}

	// Each tied slot receives grad=1; non-tied slots receive 0.
	// Total sum = 3 (number of tied maxima).
	want := []float64{1, 1, 1, 0, 0}
	for i := range want {
		if math.Abs(grad[i]-want[i]) > 1e-4 {
			t.Errorf("grad[%d] = %g, want %g (documented TieArgmax behavior under gomlx/XLA)", i, grad[i], want[i])
		}
	}
}
