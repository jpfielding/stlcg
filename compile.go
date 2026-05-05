package stlcg

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
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
	case *untilFormula, *thenFormula, *integralFormula:
		panic("stlcg: Until/Then/Integral1d not yet implemented (Phase E)")
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

// slidingReduce computes a forward-time windowed min/max along the time
// axis (axis 1) of sub ([B, T, 1]). For bounded Interval [a, b], the
// output at time t is the reduction over sub[t+a : t+b+1] (inclusive
// upper index), clipped to the trace end. For unbounded [a, ∞) the
// window at time t is [t+a, T-1]. Truncated / empty windows are padded
// with a sentinel (+∞ for min, -∞ for max) that does not affect the
// reduction.
//
// Implementation is reshape-and-reduce: right-pad sub by L-1 sentinel
// values, build L time-shifted slices at offsets a..a+L-1, stack them
// into a new axis, and reduce with Maxish/Minish. O(L) graph size.
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

	// Right-pad along the time axis with sentinel so later slices never
	// read past the original data. Need enough padding so sub[a+k : a+k+T]
	// is valid for all k in [0, L-1] -> need end >= a + L - 1 + T.
	padEnd := a + L - 1
	rank := shape.Rank()
	padAxes := make([]backends.PadAxis, rank)
	padAxes[1] = backends.PadAxis{Start: 0, End: padEnd, Interior: 0}

	sign := +1
	if wantMax {
		sign = -1 // sentinel = -∞ for max
	}
	fill := graph.Infinity(g, shape.DType, sign)
	padded := graph.Pad(sub, fill, padAxes...)

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
