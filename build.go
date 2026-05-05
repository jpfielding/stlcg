package stlcg

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/graph"
)

// BuildRobustnessTrace lowers formula into an existing gomlx graph and
// returns the robustness-trace node of shape [B, T, 1] (same shape as
// the signal inputs).
//
// This is the graph-level seam for embedding STL robustness into a
// larger model — for example, inside a train.Trainer loss function or a
// metric's UpdateGraph method. All supplied nodes (signals, pscale, tau)
// must belong to the same *graph.Graph. pscale and tau are scalar nodes;
// typical callers build them via graph.Scalar or as feedable parameters.
//
// Option values control topology-affecting knobs (Mode, TieGradient,
// KeepDim, AGM). WithPScale / WithScale option *values* are ignored —
// the tensor values come from the pscale/tau node arguments. WithAGM is
// not yet implemented and will panic.
func BuildRobustnessTrace(
	formula Formula,
	signals map[string]*graph.Node,
	pscale, tau *graph.Node,
	opts ...Option,
) *graph.Node {
	cfg := defaultConfig()
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.agm {
		panic("stlcg: WithAGM is not yet implemented")
	}

	varOrder := formula.Vars()
	inputs := make([]*graph.Node, 0, len(varOrder)+2)
	for _, name := range varOrder {
		n, ok := signals[name]
		if !ok {
			panic(fmt.Sprintf("stlcg: BuildRobustnessTrace missing variable %q", name))
		}
		inputs = append(inputs, n)
	}
	inputs = append(inputs, pscale, tau)

	c := newCompiler(cfg, varOrder, inputs)
	return c.compileFormula(formula)
}
