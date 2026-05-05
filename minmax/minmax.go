// Package minmax provides smooth and exact min/max reductions used by
// stlcg's logical and temporal operators.
//
// Maxish / Minish reduce along a single axis of a graph node. Two modes are
// supported:
//
//   - Smooth: (1/tau)*logsumexp(tau*x, axis) for max, negated for min.
//     Differentiable everywhere. Tau is passed as a graph node so it can be
//     annealed at runtime without recompiling.
//
//   - Exact: true ReduceMax/Min. Not differentiable at ties.
//
// A TiePolicy knob controls how gradient mass is distributed at ties.
// In smooth mode softmax handles ties naturally; TieUniform in exact mode
// uses a stop-gradient tie mask to split gradient evenly.
package minmax

import (
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// Mode selects smooth vs. exact reduction.
type Mode int

const (
	Smooth Mode = iota
	Exact
)

// TiePolicy selects gradient distribution at ties.
type TiePolicy int

const (
	Argmax TiePolicy = iota
	Uniform
)

// Maxish returns the reduction of x along axis. If keepDim is true the
// reduced axis is preserved as size 1.
//
// In Smooth mode, tau must be a scalar graph node (same dtype as x).
// In Exact mode, tau is ignored.
func Maxish(x *graph.Node, axis int, tau *graph.Node, mode Mode, tie TiePolicy, keepDim bool) *graph.Node {
	return reduceExtremum(x, axis, tau, mode, tie, keepDim, true)
}

// Minish is the analogous minimum reduction.
func Minish(x *graph.Node, axis int, tau *graph.Node, mode Mode, tie TiePolicy, keepDim bool) *graph.Node {
	return reduceExtremum(x, axis, tau, mode, tie, keepDim, false)
}

func reduceExtremum(x *graph.Node, axis int, tau *graph.Node, mode Mode, tie TiePolicy, keepDim, wantMax bool) *graph.Node {
	switch mode {
	case Smooth:
		return smoothExtremum(x, axis, tau, keepDim, wantMax)
	case Exact:
		return exactExtremum(x, axis, tie, keepDim, wantMax)
	}
	panic("minmax: unknown mode")
}

// smoothExtremum implements (1/tau)*logsumexp(tau*x) with the shift trick
// for numerical stability. For min, flip sign on the way in and out.
func smoothExtremum(x *graph.Node, axis int, tau *graph.Node, keepDim, wantMax bool) *graph.Node {
	signed := x
	if !wantMax {
		signed = graph.Neg(x)
	}

	scaled := graph.Mul(signed, tau)

	// Shift by the per-axis max for numerical stability.
	shift := graph.StopGradient(graph.ReduceAndKeep(scaled, graph.ReduceMax, axis))
	shifted := graph.Sub(scaled, shift)

	sumExp := graph.ReduceAndKeep(graph.Exp(shifted), graph.ReduceSum, axis)
	lse := graph.Add(graph.Log(sumExp), shift) // shape keeps axis at size 1

	// Divide by tau to recover the temperature-scaled mean-ish quantity.
	result := graph.Div(lse, tau)

	if !wantMax {
		result = graph.Neg(result)
	}
	if !keepDim {
		result = graph.Squeeze(result, axis)
	}
	return result
}

// exactExtremum does ReduceMin/Max. For TieUniform it also builds a
// stop-gradient tie mask so the backward pass splits gradient across ties.
func exactExtremum(x *graph.Node, axis int, tie TiePolicy, keepDim, wantMax bool) *graph.Node {
	var reduced *graph.Node
	if wantMax {
		reduced = graph.ReduceAndKeep(x, graph.ReduceMax, axis)
	} else {
		reduced = graph.ReduceAndKeep(x, graph.ReduceMin, axis)
	}

	switch tie {
	case Argmax:
		if !keepDim {
			reduced = graph.Squeeze(reduced, axis)
		}
		return reduced

	case Uniform:
		// Build a mask of elements equal to the extremum along axis, then
		// express the extremum as sum(x * stopgrad(mask / count)). The
		// forward value is unchanged (mean of tied extrema = extremum) but
		// autodiff now routes dL/dy uniformly across ties.
		mask := graph.Equal(x, reduced) // broadcast
		onesLike := graph.OnesLike(x)
		maskOnes := graph.Where(mask, onesLike, graph.ZerosLike(x))
		count := graph.ReduceAndKeep(maskOnes, graph.ReduceSum, axis)
		weight := graph.StopGradient(graph.Div(maskOnes, count))
		y := graph.ReduceAndKeep(graph.Mul(x, weight), graph.ReduceSum, axis)
		if !keepDim {
			y = graph.Squeeze(y, axis)
		}
		return y
	}
	panic("minmax: unknown tie policy")
}
