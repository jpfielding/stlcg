# stlcg.go

Signal Temporal Logic robustness on a differentiable computation graph, in Go.

A transliteration of [stanfordASL/stlcg](https://github.com/stanfordASL/stlcg)
(PyTorch) onto [gomlx](https://github.com/gomlx/gomlx) (XLA). Use STL
specifications as differentiable terms in a neural-network loss.

## Status

Functional end-to-end for research use. All core operators compile and
differentiate. Ergonomics not fully frozen.

- ✅ Predicates: `Lt`, `Le`, `Gt`, `Ge`, `Eq`, `Identity`
- ✅ Logical: `Not`, `And`, `Or`, `Implies`
- ✅ Temporal: `Always`, `Eventually`, `Until`, `Then` (bounded + unbounded)
- ✅ `Integral1d` (Riemann, Trapezoid)
- ✅ Smooth (shifted log-sum-exp) and exact min/max
- ✅ `TieGradient` policy (Argmax, Uniform) orthogonal to mode
- ✅ Autodiff through every temporal op (gradient-sanity tests vs. finite-diff)
- ✅ `RobustnessMetric` implementing gomlx `metrics.Interface`
- ✅ Graphviz DOT emission
- ✅ CLI (`cmd/stlcg`) and two examples (`simple`, `train`)
- ⚠️ AGM robustness (`WithAGM`) — panics, not implemented
- ⚠️ TensorBoard summary output — JSONL only; pipe through a converter if needed
- ⚠️ Python stlcg parity fixtures — generator script committed, not run in CI

## Time axis

Forward time. Trace tensors are `[batch, time, feature]` with index 0 the
earliest observation. Python stlcg uses time-reversed input internally and
stlcg.go intentionally breaks from that convention — any ported Python code
must be re-derived in forward time, not blanket-reversed. See
[`doc.go`](doc.go) and [`testdata/generate_fixtures.py`](testdata/generate_fixtures.py)
for the formal index map.

## Quickstart

```go
import (
    "github.com/gomlx/gomlx/backends"
    _ "github.com/gomlx/gomlx/backends/default"
    "github.com/gomlx/gomlx/pkg/core/tensors"
    "github.com/jpfielding/stlcg"
)

x, y := stlcg.Var("x"), stlcg.Var("y")

phi := stlcg.Always(
    stlcg.And(
        stlcg.Gt(x, stlcg.Const(5.0)),
        stlcg.Not(stlcg.Lt(y, stlcg.Const(2.0))),
    ),
    stlcg.Bounds(0, 50),
)

be := backends.MustNew()
eval := stlcg.NewEvaluator(be, phi,
    stlcg.WithMode(stlcg.ModeSmooth),
    stlcg.WithScale(5.0),
    stlcg.WithPScale(1.0),
)
defer eval.Close()

trace := stlcg.NewSignalMap(map[string]*tensors.Tensor{
    "x": xTensor, "y": yTensor, // each [B, T, 1]
})
rho := eval.Robustness(trace, stlcg.AtTime(0))
```

## Embedding robustness in a loss

For direct integration into a gomlx training step, skip `Evaluator` and
use the graph-level seam:

```go
func lossFn(ctx *context.Context, xNode, cNode *graph.Node) *graph.Node {
    signals := map[string]*graph.Node{"x": xNode, "c": cNode}
    pscale := graph.Scalar(xNode.Graph(), xNode.DType(), 1.0)
    tau := graph.Scalar(xNode.Graph(), xNode.DType(), 5.0)
    rho := stlcg.BuildRobustnessTrace(phi, signals, pscale, tau,
        stlcg.WithMode(stlcg.ModeSmooth))
    return graph.Neg(graph.ReduceAllMean(rho))
}
```

`logger.RobustnessMetric` (see `logger/metric.go`) plugs the same builder
into gomlx `train.Trainer` via `metrics.Interface`.

## CLI

```
$ echo "x
0.5
0.3
-0.2
0.1" | stlcg -stdin -formula always-gt -threshold 0
formula: □ (x > 0)
t,rho
0,-0.200000
1,-0.200000
2,-0.200000
3,+0.100000
```

Flags: `-trace <csv>` (or `-stdin`), `-formula {always-gt,always-lt,eventually-gt,eventually-lt}`,
`-threshold`, `-a`, `-b` (`-1` = unbounded), `-scale` (`0` = exact), `-dot`.

## Design notes

- Formulas are pure immutable Go values (sealed sum type). `stlcg.Walk`
  exposes them for visualization or analysis without leaking concrete
  types.
- `Evaluator` owns a `graph.Exec` that caches compiled graphs by input
  shape signature. Changing batch size or trace length recompiles;
  `SetMaxCache` tunes the cache limit.
- Scale (τ) and PScale are fed as graph parameters, so annealing them
  does not trigger recompile.
- Sliding window min/max for `Always`/`Eventually` is implemented by
  right-padding with ±∞ sentinels (via `Concatenate` + `StopGradient` —
  `graph.Pad` lacks a VJP in gomlx v0.27.3) and reducing a stacked-slice
  tensor. `Until`/`Then` iterate the same construction across window
  offsets.
- `TieGradient` with `TieUniform` in exact mode uses a stop-gradient tie
  mask so gradient mass splits across tied extrema; in smooth mode it is
  free from the softmax derivative and the knob is purely semantic.

## Layout

```
.                    Formula/Expr AST + compiler + Evaluator + BuildRobustnessTrace
minmax/              Maxish / Minish reducers (smooth + exact, both tie policies)
viz/                 Graphviz DOT emitter
logger/              JSONL logger + RobustnessMetric for train.Trainer
cmd/stlcg/           CLI demo
examples/simple/     Build a formula, evaluate, print
examples/train/      SGD threshold recovery (end-to-end autodiff smoke)
testdata/            generate_fixtures.py + fixtures/ (not in CI)
```

## Tests

```
go test ./...
```

Coverage:

- Table tests on AST construction, `Vars()`, `String()`
- Parity tests (compiler vs. pure-Go reference evaluator) for every operator
  across exact + smooth modes
- Property tests (algebraic laws: ¬¬φ ≡ φ, De Morgan, Always/Eventually
  duality, single-point Always-as-shift, Or-as-pointwise-max, smooth→exact
  convergence)
- Gradient-sanity tests: finite-difference vs. `graph.Gradient` for
  `Always`, `Eventually`, `Until`, `And`, `Integral1d` in smooth mode
- Hand-authored parity with known-correct expected values for a handful
  of operators
- `logger` integration against a `graph.Exec` that stands in for
  `train.Trainer`
- `viz` golden-ish assertions on DOT output

## License

MIT. Portions derived from stanfordASL/stlcg, also MIT.
