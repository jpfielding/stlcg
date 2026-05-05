// Command train is a tiny end-to-end gradient-descent demo that
// recovers the tightest threshold c* for Always_(x > c) by minimizing
// the squared robustness (pushing the rho trace toward zero at its
// tightest point). The expected answer is c* ≈ min(x); at that
// threshold rho = 0 everywhere except where x achieves its minimum.
//
// Structure:
//
//   1. Build a gomlx Exec whose graph function takes (x_trace, c_scalar)
//      and returns (loss, d(loss)/d(c)).
//   2. Loop: feed current c, take a gradient step, print.
//
// This doubles as a smoke test that autodiff passes through every piece
// of stlcg's temporal stack (Always → sliding reduce → Minish over a
// stacked axis → logsumexp → predicate arithmetic).
package main

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/jpfielding/stlcg"
)

func main() {
	// Target: phi = Always_[0, T-1] (x > c). Optimal c minimizes -rho,
	// which equals min(x) in the exact case and approaches min(x) as
	// τ → ∞ in the smooth case.
	const (
		T       = 20
		steps   = 400
		lr      = 0.05
		initial = -5.0
		tau     = 8.0
	)

	// A handcrafted trace whose true min is -0.3 at t=11.
	trace := []float32{
		0.5, 0.8, 0.2, 0.4, 1.0, 0.3, 0.1, 0.0, 0.2, 0.5, 0.4,
		-0.3, 0.1, 0.2, 0.3, 0.4, 0.7, 0.5, 0.2, 0.3,
	}
	if len(trace) != T {
		panic("trace length mismatch")
	}

	be := backends.MustNew()

	// Graph function: inputs are (x_trace [1, T, 1], c_scalar []).
	// Returns (loss = -mean(rho_trace), dLoss/dc).
	phi := stlcg.Always(
		stlcg.Gt(stlcg.Var("x"), stlcg.Var("c")),
		stlcg.AllTime(),
	)

	step := func(xIn, cIn *graph.Node) (*graph.Node, *graph.Node) {
		g := xIn.Graph()
		cBroadcast := graph.BroadcastToShape(cIn, xIn.Shape())
		signals := map[string]*graph.Node{"x": xIn, "c": cBroadcast}
		pscale := graph.Scalar(g, dtypes.Float32, 1.0)
		tauNode := graph.Scalar(g, dtypes.Float32, tau)
		rho := stlcg.BuildRobustnessTrace(phi, signals, pscale, tauNode,
			stlcg.WithMode(stlcg.ModeSmooth))
		// rho at t=0 ≈ min(x) - c for the unbounded Always. Target 0:
		// loss = (rho_t0 - 0)^2, minimized at c = min(x).
		rhoT0 := graph.Slice(rho,
			graph.AxisRange(),
			graph.AxisElem(0),
			graph.AxisRange(),
		)
		loss := graph.Mul(rhoT0, rhoT0)
		lossScalar := graph.ReduceAllMean(loss)
		grads := graph.Gradient(lossScalar, cIn)
		return lossScalar, grads[0]
	}

	exec, err := graph.NewExec(be, step)
	if err != nil {
		panic(err)
	}
	defer exec.Finalize()

	xT := tensors.FromShape(shapes.Make(dtypes.Float32, 1, T, 1))
	if err := tensors.MutableFlatData(xT, func(d []float32) { copy(d, trace) }); err != nil {
		panic(err)
	}
	defer xT.FinalizeAll()

	c := float32(initial)
	for i := 0; i < steps; i++ {
		outs, err := exec.Exec(xT, c)
		if err != nil {
			panic(err)
		}
		loss := extractScalar(outs[0])
		grad := extractScalar(outs[1])
		for _, o := range outs {
			o.FinalizeAll()
		}

		c -= lr * grad
		if i%20 == 0 || i == steps-1 {
			fmt.Printf("step=%3d  c=%+.5f  loss=%+.5f  grad=%+.5f\n", i, c, loss, grad)
		}
	}

	fmt.Printf("\nconverged c = %+.5f (target ≈ min(x) = %+.2f)\n", c, minOf(trace))
}

func extractScalar(t *tensors.Tensor) float32 {
	var v float32
	if err := tensors.ConstFlatData(t, func(d []float32) { v = d[0] }); err != nil {
		panic(err)
	}
	return v
}

func minOf(xs []float32) float32 {
	m := xs[0]
	for _, v := range xs[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
