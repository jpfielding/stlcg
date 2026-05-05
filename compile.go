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
	case *alwaysFormula, *eventuallyFormula, *untilFormula, *thenFormula, *integralFormula:
		panic("stlcg: temporal operators not yet implemented (Phase D/E)")
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
