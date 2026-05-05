// Package stlcg computes the robustness of Signal Temporal Logic (STL)
// formulas on a differentiable computation graph, so that STL terms can be
// embedded in neural-network loss functions.
//
// It is a Go transliteration of stanfordASL/stlcg (PyTorch) onto
// github.com/gomlx/gomlx (XLA). See
// https://github.com/stanfordASL/stlcg for the original.
//
// # Time axis convention
//
// Unlike the Python original, stlcg.go uses **natural forward time**. A trace
// tensor has shape [batch, time, feature], where index 0 is the first
// observation in physical time and index T-1 is the most recent.
//
// The Python library expects time-reversed traces as input and reports
// results under the same reversed convention; transliterated code will
// silently disagree with stlcg.go if the reversal is not undone. When
// porting Python fixtures or expected values, re-derive them in forward time
// rather than blindly reversing — in particular, nested bounded temporal
// operators (for example Always[a,b] applied to Eventually[c,d]) have a
// composed horizon that a trace-level reversal will misalign.
//
// # Signals
//
// Signals are provided by name via SignalMap. This decouples formula
// construction from tensor-column layout, so a user writes
//
//	x := stlcg.Var("x")
//	y := stlcg.Var("y")
//	phi := stlcg.Always(stlcg.And(stlcg.Gt(x, stlcg.Const(5)),
//	                              stlcg.Not(stlcg.Lt(y, stlcg.Const(2)))),
//	                    stlcg.Interval(0, 50))
//
// and binds the tensors at evaluation time via a map keyed on the variable
// names.
//
// # Evaluation and compilation
//
// A Formula is pure, immutable Go data; no gomlx graph is built until a
// Formula is passed to NewEvaluator. The Evaluator compiles the formula to
// a gomlx graph and caches the compiled Exec by input-shape signature
// (dtype included). Changing batch size or trace length triggers a
// recompile; use Evaluator.Precompile to warm the cache for known shapes.
//
// # Smooth vs. exact min/max
//
// Maxish and Minish (used by And, Or, Always, Eventually, Until, Then) have
// two evaluation modes:
//
//   - ModeSmooth (default): (1/tau) * LogSumExp(tau*x) for max, negated
//     for min. Differentiable everywhere; tau = WithScale is tunable
//     without recompiling.
//   - ModeExact: true min/max. Non-differentiable at ties.
//
// A TieGradient policy is orthogonal to the mode and controls how
// gradient mass is distributed when multiple values tie for the extremum.
// TieArgmax (the XLA default) sends full gradient to the argmax;
// TieUniform splits it across ties. This replaces the Python `distributed`
// boolean with two independent knobs.
package stlcg
