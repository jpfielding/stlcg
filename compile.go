package stlcg

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/jpfielding/stlcg/minmax"
)

// compiler lowers an STL AST to a robustness-trace *graph.Node.
//
// Inputs to the compiled graph are, in order:
//
//  0..N-1) the N named variables, in the order of c.varOrder;
//  N)      the pscale scalar;
//  N+1)    the scale/tau scalar.
//
// Each variable is expected to be a [B, T, 1] tensor of the compiler's
// dtype. The produced robustness node is also [B, T, 1].
type compiler struct {
	cfg      config
	varOrder []string
	varNode  map[string]*graph.Node
	pscale   *graph.Node
	tau      *graph.Node
	dtype    dtypes.DType
}

// newCompiler returns a compiler bound to the given inputs.
//
// inputs must be arranged as [varNodes..., pscaleNode, tauNode]. The
// varOrder slice gives the name of each positional variable input.
func newCompiler(cfg config, varOrder []string, inputs []*graph.Node) *compiler {
	if len(inputs) != len(varOrder)+2 {
		panic(fmt.Sprintf("stlcg: compiler expected %d inputs, got %d", len(varOrder)+2, len(inputs)))
	}
	c := &compiler{
		cfg:      cfg,
		varOrder: varOrder,
		varNode:  make(map[string]*graph.Node, len(varOrder)),
		pscale:   inputs[len(varOrder)],
		tau:      inputs[len(varOrder)+1],
		dtype:    inputs[0].DType(),
	}
	for i, name := range varOrder {
		c.varNode[name] = inputs[i]
	}
	return c
}

func (c *compiler) graphOf() *graph.Graph {
	if len(c.varOrder) > 0 {
		return c.varNode[c.varOrder[0]].Graph()
	}
	return c.pscale.Graph()
}

// compileFormula lowers a Formula to a robustness-trace node of shape
// [B, T, 1].
func (c *compiler) compileFormula(f Formula) *graph.Node {
	switch n := f.(type) {
	case *predicate:
		return c.compilePredicate(n)
	case *identityFormula:
		return graph.Mul(c.compileExpr(n.signal), c.pscale)
	case *notFormula:
		return graph.Neg(c.compileFormula(n.sub))
	case *andFormula:
		return c.reducePair(c.compileFormula(n.left), c.compileFormula(n.right), false)
	case *orFormula:
		return c.reducePair(c.compileFormula(n.left), c.compileFormula(n.right), true)
	case *alwaysFormula:
		return c.slidingReduce(c.compileFormula(n.sub), n.interval, false)
	case *eventuallyFormula:
		return c.slidingReduce(c.compileFormula(n.sub), n.interval, true)
	case *untilFormula:
		return c.compileUntilThen(n.left, n.right, n.interval, n.overlap, false)
	case *thenFormula:
		return c.compileUntilThen(n.left, n.right, n.interval, n.overlap, true)
	case *integralFormula:
		return c.compileIntegral(n)
	}
	panic(fmt.Sprintf("stlcg: unknown formula type %T", f))
}

// compilePredicate lowers a leaf comparison node.
//
// Robustness definitions (all multiplied by pscale):
//
//	Lt, Le:  rhs - lhs
//	Gt, Ge:  lhs - rhs
//	Eq:      -|lhs - rhs|
func (c *compiler) compilePredicate(p *predicate) *graph.Node {
	l := c.compileExpr(p.lhs)
	r := c.compileExpr(p.rhs)

	var rho *graph.Node
	switch p.op {
	case opLt, opLe:
		rho = graph.Sub(r, l)
	case opGt, opGe:
		rho = graph.Sub(l, r)
	case opEq:
		rho = graph.Neg(graph.Abs(graph.Sub(l, r)))
	default:
		panic(fmt.Sprintf("stlcg: unknown predicate op %v", p.op))
	}
	return graph.Mul(rho, c.pscale)
}

// compileExpr lowers an arithmetic expression to a graph node. Vars become
// the bound Parameter nodes; Consts become broadcast scalar nodes.
func (c *compiler) compileExpr(e Expr) *graph.Node {
	switch v := e.(type) {
	case *variable:
		n, ok := c.varNode[v.name]
		if !ok {
			panic(fmt.Sprintf("stlcg: variable %q has no bound input", v.name))
		}
		return n
	case *constant:
		return graph.Scalar(c.graphOf(), c.dtype, v.value)
	}
	panic(fmt.Sprintf("stlcg: unknown expr type %T", e))
}

// reducePair stacks two robustness nodes along a new last axis and
// reduces via Minish (wantMax=false) or Maxish (wantMax=true).
func (c *compiler) reducePair(a, b *graph.Node, wantMax bool) *graph.Node {
	// a, b have shape [B, T, 1]. Stack on axis=rank to get [B, T, 1, 2],
	// reduce on that axis without keepDim to return to [B, T, 1].
	stackAxis := a.Rank()
	stacked := graph.Stack([]*graph.Node{a, b}, stackAxis)

	mode, tie := c.minmaxMode()
	if wantMax {
		return minmax.Maxish(stacked, stackAxis, c.tau, mode, tie, false)
	}
	return minmax.Minish(stacked, stackAxis, c.tau, mode, tie, false)
}

// padTimeAxis right-pads sub along axis 1 by padEnd values of the given
// sentinel fill. Unlike graph.Pad (no VJP in gomlx), this uses
// Concatenate with a StopGradient'd broadcast tensor so autodiff can flow
// through the rest of the trace unimpeded.
func padTimeAxisRight(sub, fill *graph.Node, padEnd int) *graph.Node {
	if padEnd <= 0 {
		return sub
	}
	shape := sub.Shape()
	rank := shape.Rank()
	padDims := make([]int, rank)
	for i := 0; i < rank; i++ {
		padDims[i] = shape.Dimensions[i]
	}
	padDims[1] = padEnd

	padShape := shape
	padShape.Dimensions = padDims
	sentinel := graph.StopGradient(graph.BroadcastToShape(fill, padShape))
	return graph.Concatenate([]*graph.Node{sub, sentinel}, 1)
}

// padTimeAxisLeft prepends padStart values of fill along axis 1, via
// Concatenate. Same VJP-safe reason as padTimeAxisRight.
func padTimeAxisLeft(sub, fill *graph.Node, padStart int) *graph.Node {
	if padStart <= 0 {
		return sub
	}
	shape := sub.Shape()
	rank := shape.Rank()
	padDims := make([]int, rank)
	for i := 0; i < rank; i++ {
		padDims[i] = shape.Dimensions[i]
	}
	padDims[1] = padStart

	padShape := shape
	padShape.Dimensions = padDims
	sentinel := graph.StopGradient(graph.BroadcastToShape(fill, padShape))
	return graph.Concatenate([]*graph.Node{sentinel, sub}, 1)
}

// slidingReduce computes a forward-time windowed min/max along the time
// axis (axis 1) of sub ([B, T, 1]). For bounded Interval [a, b], the
// output at time t is the reduction over sub[t+a : t+b+1] (inclusive
// upper index), clipped to the trace end. For unbounded [a, ∞) the
// window at time t is [t+a, T-1]. Truncated / empty windows are padded
// with a sentinel (+∞ for min, -∞ for max) that does not affect the
// reduction.
//
// Implementation is reshape-and-reduce: right-pad sub by L-1 sentinel
// values (via VJP-safe Concatenate+StopGradient), build L time-shifted
// slices at offsets a..a+L-1, stack them into a new axis, and reduce
// with Maxish/Minish. O(L) graph size.
func (c *compiler) slidingReduce(sub *graph.Node, iv Interval, wantMax bool) *graph.Node {
	shape := sub.Shape()
	if shape.Rank() < 2 {
		panic(fmt.Sprintf("stlcg: slidingReduce expects rank>=2 input, got %d", shape.Rank()))
	}
	T := shape.Dimensions[1]
	if iv.Lo >= T {
		panic(fmt.Sprintf("stlcg: interval lower bound %d >= trace length %d", iv.Lo, T))
	}

	a := iv.Lo
	var b int
	if iv.IsUnbounded() {
		b = T - 1
	} else {
		b = iv.Hi
		if b >= T {
			b = T - 1
		}
	}
	L := b - a + 1
	if L < 1 {
		panic(fmt.Sprintf("stlcg: empty interval after clipping: lo=%d hi=%d T=%d", iv.Lo, iv.Hi, T))
	}

	g := sub.Graph()
	rank := shape.Rank()

	sign := +1
	if wantMax {
		sign = -1 // sentinel = -∞ for max
	}
	fill := graph.Infinity(g, shape.DType, sign)

	padEnd := a + L - 1
	padded := padTimeAxisRight(sub, fill, padEnd)

	// Build L time-shifted slices along axis 1.
	slices := make([]*graph.Node, 0, L)
	for k := 0; k < L; k++ {
		axesSpec := make([]graph.SliceAxisSpec, rank)
		for ax := 0; ax < rank; ax++ {
			axesSpec[ax] = graph.AxisRange()
		}
		axesSpec[1] = graph.AxisRange(a+k, a+k+T)
		slices = append(slices, graph.Slice(padded, axesSpec...))
	}

	// Stack along a new last axis: shape [..., T, F, L]. Reduce with
	// keepDim=false so we collapse back to [..., T, F].
	stackAxis := rank
	stacked := graph.Stack(slices, stackAxis)

	mode, tie := c.minmaxMode()
	if wantMax {
		return minmax.Maxish(stacked, stackAxis, c.tau, mode, tie, false)
	}
	return minmax.Minish(stacked, stackAxis, c.tau, mode, tie, false)
}

// compileUntilThen lowers (phi U_iv psi) or (phi T_iv psi) into a
// [B, T, 1] robustness-trace node.
//
// Semantics (forward time; conventional STL):
//
//	until[t] = max_{s ∈ [t+a, t+b] ∩ [0, T-1]}
//	               min( prefix_{t..s} phi,   psi[s] )
//
// For Until, prefix_{t..s} phi = min over u of phi[u].
// For Then,  prefix_{t..s} phi = max over u of phi[u].
//
// The "overlap" parameter controls whether phi's prefix includes the
// matching time s:
//   - overlap=true  (conventional "strong until"): u ∈ [t, s].
//     phi is required to hold at s as well as earlier.
//   - overlap=false                                : u ∈ [t, s-1].
//     phi is only required before s; psi holding at t=s is sufficient
//     (the phi-prefix becomes the identity element of min/max).
//
// Implementation is an O(L) recurrence. Let upper_k = a+k (overlap=true)
// or a+k-1 (overlap=false). The seed phi_pfx_0 is the min/max of phi
// over window [t, t+upper_0]; each subsequent step extends by one slot:
//
//	phi_pfx_k = reducePair(phi_pfx_{k-1}, phi[t+upper_k], phiPrefixMax)
//
// This avoids rebuilding L sliding-window reductions (the prior
// O(L^2) graph-size cost), giving O(L) slices + O(L) pair reductions.
func (c *compiler) compileUntilThen(left, right Formula, iv Interval, overlap, phiPrefixMax bool) *graph.Node {
	phi := c.compileFormula(left)
	psi := c.compileFormula(right)

	shape := phi.Shape()
	T := shape.Dimensions[1]
	a := iv.Lo
	if a >= T {
		panic(fmt.Sprintf("stlcg: Until/Then interval lo=%d >= trace length %d", a, T))
	}
	var b int
	if iv.IsUnbounded() {
		b = T - 1
	} else {
		b = iv.Hi
		if b >= T {
			b = T - 1
		}
	}
	L := b - a + 1
	if L < 1 {
		panic(fmt.Sprintf("stlcg: Until/Then empty interval after clipping lo=%d hi=%d T=%d", iv.Lo, iv.Hi, T))
	}

	rank := shape.Rank()

	// Sentinels.
	// psi padding: -∞ (outer max identity). Past the trace end psi never
	// wins.
	// phi padding: +∞ for Until (min identity) or -∞ for Then (max
	// identity). Past the trace end phi never loses.
	psiFill := graph.Infinity(phi.Graph(), shape.DType, -1)
	phiFillSign := +1
	if phiPrefixMax {
		phiFillSign = -1
	}
	phiFill := graph.Infinity(phi.Graph(), shape.DType, phiFillSign)

	padRight := a + L - 1
	psiPad := padTimeAxisRight(psi, psiFill, padRight)
	phiPad := padTimeAxisRight(phi, phiFill, padRight)

	// sliceAt returns a [B, T, F] slice starting at offset `start` along
	// axis 1. Works on padded tensors where start+T is in range.
	sliceAt := func(node *graph.Node, start int) *graph.Node {
		spec := make([]graph.SliceAxisSpec, rank)
		for ax := 0; ax < rank; ax++ {
			spec[ax] = graph.AxisRange()
		}
		spec[1] = graph.AxisRange(start, start+T)
		return graph.Slice(node, spec...)
	}

	// Seed phi prefix.
	// upper_0 = a (overlap) or a-1 (no overlap).
	seedUpper := a
	if !overlap {
		seedUpper = a - 1
	}
	var phiPfx *graph.Node
	switch {
	case seedUpper < 0:
		// Empty prefix: broadcast the phi-reduction identity.
		phiPfx = graph.BroadcastToShape(phiFill, shape)
	case seedUpper == 0:
		// Single-sample prefix: no reduction needed, just phi[t].
		phiPfx = sliceAt(phiPad, 0)
	default:
		// Window [0, seedUpper] reduction. slidingReduce handles
		// sentinel padding internally; it is O(seedUpper+1) slices
		// paid ONCE, not per k.
		phiPfx = c.slidingReduce(phi, Bounds(0, seedUpper), phiPrefixMax)
	}

	inner := make([]*graph.Node, L)
	// k=0
	inner[0] = c.reducePair(phiPfx, sliceAt(psiPad, a), false)

	// k>=1: extend the prefix one sample and pair with the shifted psi.
	for k := 1; k < L; k++ {
		// nextIdx = upper_k: a+k (overlap) or a+k-1 (no overlap).
		nextIdx := a + k
		if !overlap {
			nextIdx = a + k - 1
		}
		phiShift := sliceAt(phiPad, nextIdx)
		phiPfx = c.reducePair(phiPfx, phiShift, phiPrefixMax)

		inner[k] = c.reducePair(phiPfx, sliceAt(psiPad, a+k), false)
	}

	stacked := graph.Stack(inner, rank)
	mode, tie := c.minmaxMode()
	return minmax.Maxish(stacked, rank, c.tau, mode, tie, false)
}

// compileIntegral lowers Integral1d(sub, [a,b], scheme) to a [B, T, 1]
// trace. Riemann: sum over s ∈ [t+a, t+b] of rho(sub, s), using cumsum
// differences. Trapezoid: Riemann minus 0.5 * (endpoints).
func (c *compiler) compileIntegral(f *integralFormula) *graph.Node {
	sub := c.compileFormula(f.sub)
	shape := sub.Shape()
	T := shape.Dimensions[1]
	a := f.interval.Lo
	b := f.interval.Hi
	if a >= T {
		panic(fmt.Sprintf("stlcg: Integral1d lo=%d >= trace length %d", a, T))
	}
	if b >= T {
		b = T - 1
	}
	L := b - a + 1
	if L < 1 {
		panic(fmt.Sprintf("stlcg: Integral1d empty window lo=%d hi=%d T=%d", f.interval.Lo, f.interval.Hi, T))
	}

	rank := shape.Rank()
	g := sub.Graph()

	// Reshape-and-reduce integral: right-pad by L-1 zeros (so slices
	// past the trace end contribute 0), build L shifted slices at
	// offsets a..a+L-1, stack, and ReduceSum along the stack axis.
	// ReduceSum has a VJP in gomlx; CumSum/SumPool does not.
	zero := graph.Scalar(g, shape.DType, 0)
	padded := padTimeAxisRight(sub, zero, a+L-1)

	rangeSpec := func(start, end int) []graph.SliceAxisSpec {
		s := make([]graph.SliceAxisSpec, rank)
		for ax := 0; ax < rank; ax++ {
			s[ax] = graph.AxisRange()
		}
		s[1] = graph.AxisRange(start, end)
		return s
	}

	slices := make([]*graph.Node, 0, L)
	for k := 0; k < L; k++ {
		slices = append(slices, graph.Slice(padded, rangeSpec(a+k, a+k+T)...))
	}
	stackAxis := rank
	stacked := graph.Stack(slices, stackAxis)
	integral := graph.ReduceSum(stacked, stackAxis)

	if f.scheme == Trapezoid {
		// Subtract half of the two endpoints: 0.5 * (sub[t+a] + sub[t+b]).
		lowEnd := graph.Slice(padded, rangeSpec(a, a+T)...)
		highEnd := graph.Slice(padded, rangeSpec(b, b+T)...)
		half := graph.Scalar(g, shape.DType, 0.5)
		adjust := graph.Mul(graph.Add(lowEnd, highEnd), half)
		integral = graph.Sub(integral, adjust)
	}

	return integral
}

// ensure dtypes import is used even if the compiler grows or shrinks.
var _ = dtypes.Float32

func (c *compiler) minmaxMode() (minmax.Mode, minmax.TiePolicy) {
	var mode minmax.Mode
	switch c.cfg.mode {
	case ModeSmooth:
		mode = minmax.Smooth
	case ModeExact:
		mode = minmax.Exact
	default:
		panic(fmt.Sprintf("stlcg: unknown mode %v", c.cfg.mode))
	}

	var tie minmax.TiePolicy
	switch c.cfg.tie {
	case TieArgmax:
		tie = minmax.Argmax
	case TieUniform:
		tie = minmax.Uniform
	default:
		panic(fmt.Sprintf("stlcg: unknown tie %v", c.cfg.tie))
	}
	return mode, tie
}
