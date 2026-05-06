# stlcg.go

**Signal Temporal Logic robustness on a differentiable computation graph, in Go.**

A transliteration of [stanfordASL/stlcg](https://github.com/stanfordASL/stlcg)
(PyTorch / WAFR 2020) onto [gomlx](https://github.com/gomlx/gomlx) (XLA).
Use STL specifications — safety envelopes, reach-avoid goals, sequencing
constraints — as differentiable terms in a neural-network loss and
backpropagate through them.

```
formula → AST → compile (gomlx graph) → Exec (cached per shape) → tensor
                                     └→ Gradient (autodiff end-to-end)
```

- [What is Signal Temporal Logic?](#what-is-signal-temporal-logic)
- [Why differentiable STL?](#why-differentiable-stl)
- [Status](#status)
- [Install](#install)
- [Quickstart](#quickstart)
- [Core concepts](#core-concepts)
- [Operators reference](#operators-reference)
- [Smooth vs. exact](#smooth-vs-exact-minmax)
- [Embedding robustness in neural-net training](#embedding-robustness-in-neural-net-training)
- [Time axis](#time-axis-forward-time-not-python-reversed)
- [CLI](#cli)
- [Visualization](#visualization)
- [Architecture](#architecture)
- [Performance](#performance)
- [Differences from Python stlcg](#differences-from-python-stlcg)
- [Reproducing Python parity](#reproducing-python-parity)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## What is Signal Temporal Logic?

STL is a formal specification language for time-varying real-valued signals.
A formula like

> **Always over [0, 50]**: (`x > 5`) AND NOT (`y < 2`)

reads as: for every time step from now to 50 steps out, `x` stays above 5
and `y` stays at or above 2.

STL's key feature for ML is **robustness**: every formula ϕ admits a
real-valued function ρ(ϕ, trace, t) ∈ ℝ that is

- **positive** when the trace satisfies ϕ at time t,
- **zero** at the boundary,
- **negative** (and proportional to how badly) when it violates ϕ.

So instead of "did the trajectory satisfy the spec?" (a boolean) you get
"by how much?" (a continuous quantity). Robustness is a continuous
relaxation of satisfaction; differentiate through it and you can
*optimize* for satisfaction.

## Why differentiable STL?

If you can compute ρ(ϕ, trace, t) as a differentiable function of the
trace, you can plug it into any gradient-based optimizer. Use cases:

- **Safe RL / safe control**: add `L = -ρ(safety_spec)` to your loss so
  the policy learns to satisfy safety constraints.
- **Trajectory optimization**: solve `min J(traj) − λ · ρ(spec, traj)`
  end-to-end with Adam rather than a mixed-integer solver.
- **Anomaly / monitoring loss**: train a model that predicts whether a
  signal will satisfy ϕ at runtime, supervising with robustness values.
- **Curriculum / shaping**: anneal the "how smooth" parameter τ during
  training so early iterations see a soft landscape and late ones see
  the true min/max.

The original stlcg paper is Leung & Arechiga, *Backpropagation through
STL specifications: Infusing logical structure into gradient-based
methods*, WAFR 2020.

## Status

Functional end-to-end for research use. All core operators compile and
differentiate; ergonomics not yet frozen for a 1.0 release.

| Area | State |
|---|---|
| Predicates `Lt Le Gt Ge Eq Identity` | ✅ |
| Logical `Not And Or Implies` | ✅ |
| Temporal `Always Eventually Until Then` (bounded + unbounded) | ✅ |
| `Integral1d` (Riemann / Trapezoid) | ✅ |
| Smooth (shifted log-sum-exp) and exact min/max | ✅ |
| `TieGradient` policy (`Argmax` / `Uniform`) orthogonal to mode | ✅ |
| Autodiff through every operator (finite-diff-validated) | ✅ |
| `RobustnessMetric` implementing gomlx `metrics.Interface` | ✅ |
| Graphviz DOT emission | ✅ |
| CLI + runnable examples | ✅ |
| AGM robustness (`WithAGM`) | ❌ panics (future work) |
| TensorBoard summary output | ❌ JSONL only |
| Python stlcg parity fixtures | ⚠️ generator committed; not run in CI |

## Install

```sh
go get github.com/jpfielding/stlcg@latest
```

First build pulls gomlx; the XLA PJRT plugin (~42 MB) is auto-downloaded
to `~/Library/Application Support/go-xla` (macOS) or the XDG cache dir
(Linux) on first use. CPU backend works out of the box; for GPU, follow
[gomlx's backend setup](https://github.com/gomlx/gomlx#install).

Requires Go 1.26.x or newer.

## Quickstart

Full runnable example (batch of one, T=12, two variables):

```go
package main

import (
    "fmt"

    "github.com/gomlx/gomlx/backends"
    _ "github.com/gomlx/gomlx/backends/default"
    "github.com/gomlx/gomlx/pkg/core/dtypes"
    "github.com/gomlx/gomlx/pkg/core/shapes"
    "github.com/gomlx/gomlx/pkg/core/tensors"
    "github.com/jpfielding/stlcg"
)

func main() {
    // 1. Build the formula: pure Go, no compute yet.
    x, y := stlcg.Var("x"), stlcg.Var("y")
    phi := stlcg.Always(
        stlcg.And(
            stlcg.Gt(x, stlcg.Const(0.0)),
            stlcg.Not(stlcg.Lt(y, stlcg.Const(-1.0))),
        ),
        stlcg.Bounds(0, 5),
    )
    fmt.Println(phi) // □[0,5] ((x > 0) ∧ ¬(y < -1))

    // 2. Compile the formula against a backend.
    be := backends.MustNew()
    eval := stlcg.NewEvaluator(be, phi,
        stlcg.WithMode(stlcg.ModeSmooth),
        stlcg.WithScale(5.0),   // τ: smooth-min temperature
        stlcg.WithPScale(1.0),  // predicate-robustness scale
    )
    defer eval.Close()

    // 3. Feed signals by name.
    signals := stlcg.NewSignalMap(map[string]*tensors.Tensor{
        "x": mkTensor([]float32{0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.9, 0.6, 1.0, 0.3, 0.7}),
        "y": mkTensor([]float32{0.5, 0.3, 0.0, -0.5, -0.8, -0.2, 0.1, 0.4, 0.0, -0.3, 0.2, 0.5}),
    })

    trace := eval.RobustnessTrace(signals)
    defer trace.FinalizeAll()

    _ = tensors.ConstFlatData(trace, func(data []float32) {
        for t, v := range data {
            fmt.Printf("t=%2d  ρ=%+0.4f\n", t, v)
        }
    })
}

func mkTensor(row []float32) *tensors.Tensor {
    t := tensors.FromShape(shapes.Make(dtypes.Float32, 1, len(row), 1))
    _ = tensors.MutableFlatData(t, func(d []float32) { copy(d, row) })
    return t
}
```

A working version lives in [`examples/simple/main.go`](examples/simple/main.go);
the [`examples/train/main.go`](examples/train/main.go) shows end-to-end
gradient descent recovering the tightest threshold `c*` for `Always (x > c)`.

## Core concepts

**Formula** — an immutable Go value representing one node of the STL AST.
Closed sum type (sealed interface); constructed only via the exported
constructor functions (`Var`, `Const`, `Gt`, `Always`, ...). Walk the
tree with `stlcg.Walk`.

**Expr** — an arithmetic expression referenced by predicates. Either a
`Var(name)` (a runtime-fed signal) or a `Const(value)` (a compile-time
scalar). `Gt(Var("x"), Const(5.0))` is the predicate "x > 5".

**Interval** — a struct with `Lo`, `Hi`. Constructed via `Bounds(lo, hi)`,
`From(lo)`, or `AllTime()`. Hi may be `Unbounded` for a half-open upper
bound. All intervals are inclusive both ends (forward time).

**Evaluator** — wraps a compiled gomlx `graph.Exec` for a fixed formula
under fixed topology (Mode, TieGradient, AGM, KeepDim). Caches compiled
graphs by input-shape signature. Create with `NewEvaluator(backend, phi,
opts...)`, call `RobustnessTrace(signals)` or `Robustness(signals,
AtTime(t))`, `Close()` when done.

**SignalMap** — a `map[string]*tensors.Tensor` from variable names to
tensors of shape `[B, T, 1]`. Decouples formula construction from
tensor-column layout.

**Robustness** — the output. `RobustnessTrace` returns shape `[B, T, 1]`;
`Robustness` picks a single time step and returns `[B, 1]`.

## Operators reference

```
Predicates (leaves)
-------------------
Lt(a, b), Le(a, b)       ρ = b − a        (a < b ⇒ ρ > 0)
Gt(a, b), Ge(a, b)       ρ = a − b
Eq(a, b)                 ρ = −|a − b|
Identity(a)              ρ = a            (pass-through for arbitrary scores)

Logical
-------
Not(ϕ)                   ρ = −ρ(ϕ)
And(ϕ, ψ)                ρ = Minish(ρ(ϕ), ρ(ψ))
Or(ϕ, ψ)                 ρ = Maxish(ρ(ϕ), ρ(ψ))
Implies(ϕ, ψ)            ≡ Or(Not(ϕ), ψ)

Temporal (forward-time, interval iv)
------------------------------------
Always(ϕ, iv)            ρ(t) = Minish over s∈[t+iv.Lo, t+iv.Hi ∩ T−1]
                                of ρ(ϕ, s)
Eventually(ϕ, iv)        ρ(t) = Maxish over the same window
Until(ϕ, ψ, iv, overlap) ρ(t) = Maxish over s∈[t+iv.Lo, t+iv.Hi] of
                                Minish(prefix-min ρ(ϕ) on [t, s], ρ(ψ, s))
Then(ϕ, ψ, iv, overlap)  Until with prefix-MAX on ϕ (i.e. ϕ must have
                         held at least once before ψ triggers)

Integral
--------
Integral1d(ϕ, iv, Riemann)    ρ(t) = Σ over s∈[t+iv.Lo, t+iv.Hi] of ρ(ϕ, s)
Integral1d(ϕ, iv, Trapezoid)  Riemann with half-weighted endpoints
```

All temporal operators apply a "shrinking window + sentinel" convention
at the trace's right edge: when `t+iv.Hi` exceeds `T-1` the window
truncates to `[t+iv.Lo, T-1]`, with padded slots filled by ±∞ so they
are absorbed by the min/max. For fully-past-end windows the result is
the sentinel (±∞) — a conservative signal of "undefined at this t".

## Smooth vs. exact min/max

Every min/max in the operator definitions above is really `Minish` /
`Maxish`, with two modes:

- **`ModeSmooth`** (default): `(1/τ) · logsumexp(τ · x)` for max,
  negated for min. Differentiable everywhere; τ = `WithScale(...)` is
  fed as a graph parameter, so you can anneal it during training without
  recompiling. As τ → ∞ the smooth reduction approaches the exact
  extremum.
- **`ModeExact`**: true `ReduceMin` / `ReduceMax`. Not differentiable at
  ties; gradient routes to a single argmin/argmax unless you ask for
  `TieGradient(TieUniform)`.

`TieGradient` is orthogonal to Mode:

- `TieArgmax` (default): XLA's native behavior — full gradient to one
  argmax/argmin, whichever backend picks.
- `TieUniform`: in smooth mode this is free from softmax; in exact mode
  it uses a stop-gradient tie mask to split gradient mass uniformly.

Pick `ModeSmooth` for training (gradient signal is non-zero everywhere)
and `ModeExact` for validation / reporting (you want the true
robustness value).

## Embedding robustness in neural-net training

Two seams:

### 1. Graph-level: `BuildRobustnessTrace`

For direct embedding in a gomlx loss function or a custom training step,
skip `Evaluator` and call the graph-level seam:

```go
import (
    "github.com/gomlx/gomlx/pkg/core/graph"
    "github.com/jpfielding/stlcg"
)

func stlLoss(xNode, cNode *graph.Node) *graph.Node {
    g := xNode.Graph()
    signals := map[string]*graph.Node{"x": xNode, "c": cNode}
    pscale := graph.Scalar(g, xNode.DType(), 1.0)
    tau    := graph.Scalar(g, xNode.DType(), 5.0)

    rho := stlcg.BuildRobustnessTrace(
        phi, signals, pscale, tau,
        stlcg.WithMode(stlcg.ModeSmooth),
    )
    // Minimize negative robustness ≡ maximize satisfaction.
    return graph.Neg(graph.ReduceAllMean(rho))
}
```

Autodiff through `BuildRobustnessTrace` "just works" — every internal op
has a VJP registered in gomlx (see [Architecture](#architecture) for
why that took care to guarantee).

### 2. Metric: `logger.RobustnessMetric`

For drop-in integration with `gomlx/pkg/ml/train.Trainer`:

```go
import (
    "github.com/gomlx/gomlx/pkg/ml/train"
    "github.com/jpfielding/stlcg"
    "github.com/jpfielding/stlcg/logger"
)

phi := stlcg.Always(stlcg.Gt(stlcg.Var("x"), stlcg.Const(0.5)), stlcg.Bounds(0, 10))
metric := logger.NewRobustnessMetric("safety_rho", phi).WithScale(5.0)

trainer := train.NewTrainer(..., []metrics.Interface{metric}, ...)
```

`RobustnessMetric` positionally binds the trainer's `predictions` slice
to the formula's variables in `phi.Vars()` order. For single-variable
formulas the mapping is trivial; for multi-variable formulas you'll
typically wrap your model output to produce the right slice.

### JSONL logger

Independent of the metric, `logger.JSONLLogger` writes one JSON record
per scalar / histogram call. Stdlib-only, mutex-guarded:

```go
f, _ := os.Create("run.jsonl")
l := logger.NewJSONL(f)
defer l.Close()

l.Scalar("loss", step, lossVal)
l.Histogram("grad_norms", step, gradNorms)
```

## Time axis: forward time, not Python-reversed

stlcg.go indexes traces in **forward time**: tensor index 0 is the
earliest observation, index T−1 is the most recent.

The Python stlcg library internally expects time-**reversed** inputs and
reports results under the same reversed convention. When porting a
Python formulation to stlcg.go, re-derive the semantics from the math,
not from blanket-reversing tensors — in particular, **nested bounded
temporal operators** have a composed horizon that a trace-level reversal
misaligns silently. See [`doc.go`](doc.go) and
[`testdata/generate_fixtures.py`](testdata/generate_fixtures.py) for
the formal index mapping used when porting Python fixtures.

## CLI

`cmd/stlcg` exercises the library over a single-column CSV trace:

```console
$ echo "x
0.5
0.3
-0.2
0.1
0.7" | stlcg -stdin -formula always-gt -threshold 0 -scale 0
formula: □ (x > 0)
t,rho
0,-0.200000
1,-0.200000
2,-0.200000
3,+0.100000
4,+0.700000
```

Flags:

- `-trace <path>` or `-stdin` — CSV source (header row required, column named `x`)
- `-formula {always-gt, always-lt, eventually-gt, eventually-lt}`
- `-threshold <float>` — predicate constant
- `-a N`, `-b N` — interval bounds; `-b -1` means unbounded upper
- `-scale <τ>` — smooth-min temperature; `0` switches to exact mode
- `-dot` — also emit the formula as Graphviz DOT on stdout

## Visualization

`viz.WriteDOT(w, phi)` emits Graphviz DOT for any formula. Hand-rolled,
no CGo, no external DOT library. Pipe through `dot`:

```sh
go run ./cmd/stlcg -stdin -formula always-gt -threshold 0 -dot \
    < trace.csv | dot -Tpng -o phi.png
```

Color palette: **lightblue** vars, **wheat** consts, **palegreen**
predicates, **orange** logical, **lightcoral** temporal, **plum**
integral.

`stlcg.Walk(phi)` exposes the AST as a flat `[]WalkNode` (id, kind,
label, children) — the neutral interface `viz` consumes. External
analyzers can use the same hook without touching the unexported
concrete types.

## Architecture

```
user code
    │
    ▼   stlcg.Var, Const, Gt, And, Always, Until, ...
Formula AST  (pure Go values; immutable; sealed sum type)
    │
    ▼   Evaluator.NewEvaluator(backend, phi, opts...)
compile.go  (walks AST, lowers to gomlx *graph.Node)
    │   predicates       → Sub / Neg / Abs * pscale
    │   And/Or/Not       → Minish / Maxish via minmax package
    │   Always/Eventually → sliding-window reshape-and-reduce
    │   Until/Then       → O(L) windows of pairwise Minish + outer Maxish
    │   Integral1d       → stacked slices + ReduceSum
    ▼
gomlx graph  (built once per (formula, input-shape) pair)
    │
    ▼   graph.NewExec caches by shape; graph.Gradient for VJPs
XLA executable  (CPU / GPU / TPU via gomlx backends)
    │
    ▼
tensors.Tensor output (robustness trace [B, T, 1])
```

**Why Concatenate + StopGradient instead of Pad.** `graph.Pad` has no
VJP registered in gomlx v0.27.3 — a training example caught this on
first run. Every past-end sentinel in the library is produced by
concatenating a `StopGradient`-wrapped broadcast tensor onto the trace,
keeping autodiff intact.

**Why reshape-and-reduce instead of ReduceWindow.** gomlx exposes
`BackendReduceWindow` but not a public graph-level wrapper, and its VJP
is limited. Reshape-and-reduce (L stacked slices → ReduceMin/Max) has a
clean VJP and O(L) graph size — acceptable for typical STL window
lengths.

**Why Until/Then iterate window offsets.** Proper segment-monoid DP
would be O(T log W); the current unroll is O(L²) per formula, which is
the same order as `ReduceWindow` but more op-dense. Documented as Risk
#5 / Phase-J follow-up — acceptable as long as L stays modest
(benchmarks show Until at L=16, T=40 = 509 µs per call).

## Performance

Benchmarks from Apple M2 Max, darwin-arm64, gomlx v0.27.3, XLA CPU
backend, `-benchtime=3x`:

| Benchmark | Time |
|---|---:|
| Compile T=20 L=5 (cold, includes JIT) | 26 ms |
| Compile T=100 L=10 | 37 ms |
| Compile T=100 L=50 | 75 ms |
| Compile T=1000 L=50 | 68 ms |
| Run T=20 L=5 (warm) | 68 µs |
| Run T=100 L=10 | 89 µs |
| Run T=100 L=50 | 207 µs |
| Run T=1000 L=50 | 320 µs |
| Nested `Always(Eventually(And(Gt,Lt)))`, T=80 | 102 µs |
| Until L=16, T=40 (O(L²) DP) | 509 µs |
| Shape churn, 2 pre-warmed shapes | 155 µs |

Takeaways:

- **JIT is a one-time 30-80 ms cost.** Amortizes immediately over a
  training run; annoying for one-shot evals.
- **Steady-state is sub-millisecond** for typical formula + trace sizes.
- **Shape cache works.** Alternating between two pre-warmed shapes has
  no penalty. Feed varying-length traces? Use `Pad(trace, targetLen)`
  to hit a fixed shape, or `SetMaxCache(n)` to keep the top-n shapes
  compiled.

Reproduce:

```sh
go test -bench=. -benchtime=10x -benchmem ./...
```

## Differences from Python stlcg

| | Python stlcg | stlcg.go |
|---|---|---|
| Backend | PyTorch | gomlx / XLA |
| Graph model | dynamic (PyTorch autograd) | static (XLA, traced once per shape) |
| Time axis | reversed internally | forward throughout |
| Formula construction | operator overloading on `Expression` | constructor functions (`Gt(x, Const(5))`) |
| `distributed=True` | one bool | `Mode` × `TieGradient` as orthogonal knobs |
| AGM robustness | `agm=True` | not yet implemented |
| Visualization | graphviz + IPython | stdlib DOT writer; pipe to `dot` |
| Training hookup | bare PyTorch autograd | gomlx `train.Trainer` metrics + graph-level seam |
| Logger | TF1 tensorboard | JSONL (+ user converts if TB needed) |

## Reproducing Python parity

Python stlcg parity fixtures are **not run in CI** — they live in
`testdata/fixtures/` for auditable diffs. Regenerate by hand:

```sh
pip install torch numpy
git clone https://github.com/stanfordASL/stlcg
cd stlcg && pip install -e . && cd -

python3 testdata/generate_fixtures.py > testdata/fixtures/all.jsonl

go test -run TestPythonParityFixtures ./...
```

Never silently regenerate — any change in expected values is a semantic
diff and must be human-reviewed.

Independent of Python parity, **hand-authored parity** in
[`parity_test.go`](parity_test.go) carries known-correct expected values
computed by hand for predicates, `Always[0,2]`, `Eventually[0,1]`, and
Integral Riemann. These run in CI.

## Testing

```sh
go test ./...                 # full suite
go test -race ./...           # with race detector
go test -bench=. -benchmem    # benchmarks
go test -run Property ./...   # just the property tests
go test -run Gradient ./...   # just the finite-diff vs. autodiff checks
```

Test inventory (12 test files):

- **Table tests** for AST construction, `Vars()`, `String()`
- **Parity tests** (compiler vs. pure-Go reference evaluator) for every
  operator in both exact and smooth modes
- **Property tests** (`testing/quick` on the reference evaluator):
  - `Not(Not φ) ≡ φ`
  - De Morgan smooth: `And(a,b) ≡ Not(Or(Not a, Not b))` at matched τ
  - Always/Eventually duality in exact mode
  - `Always[a,a] φ` ≡ `φ` shifted by a (sentinel past-end)
  - `Eventually[0,0] φ ≡ φ`
  - `Or(a,b)` is pointwise max in exact mode
  - Smooth τ=40 approaches exact on non-tied traces
- **Gradient-sanity tests**: centered finite-diff vs. `graph.Gradient`
  for Always, Eventually, Until, And, Integral1d (smooth mode,
  tolerance < 2e-3)
- **Hand-authored parity** (independent of both compiler and reference)
- **Logger integration** (stand-in for `train.Trainer`)
- **Viz DOT** assertions on labels, colors, and edge counts

## Roadmap

Shape of v1.0, in rough priority order:

1. AGM robustness (`WithAGM` currently panics).
2. Python stlcg parity fixtures committed & checked in CI.
3. Segment-monoid DP for Until/Then (O(T log W) vs. current O(L²)) if
   profiles justify it.
4. Multi-dtype support (currently float32-only).
5. Direct `train.Trainer` example demonstrating full training loop.
6. TensorBoard summary-proto writer (or confirm JSONL is enough).

## Citation

If you use stlcg.go in research, cite the original stlcg paper:

```bibtex
@inproceedings{leung2020back,
  title={Back-propagation through STL specifications: Infusing logical structure into gradient-based methods},
  author={Leung, Karen and Arechiga, Nikos and Pavone, Marco},
  booktitle={International Workshop on the Algorithmic Foundations of Robotics},
  year={2020}
}
```

And optionally link to this Go port: <https://github.com/jpfielding/stlcg>.

## License

MIT. Portions derived from [stanfordASL/stlcg](https://github.com/stanfordASL/stlcg),
also MIT.
