package stlcg

import (
	"fmt"
	"math"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// SignalMap binds variable names to tensors at evaluation time.
//
// Each tensor should have shape [B, T, 1] with the same dtype (currently
// float32). Multi-feature signals should be provided as separate named
// entries.
type SignalMap map[string]*tensors.Tensor

// NewSignalMap is a convenience wrapper for literal construction.
func NewSignalMap(m map[string]*tensors.Tensor) SignalMap { return SignalMap(m) }

// TimeSelector picks a specific time step from a robustness trace.
type TimeSelector struct{ t int }

// AtTime selects the time index t from a [B, T, 1] trace. Negative t
// counts from the end (e.g. -1 = last step).
func AtTime(t int) TimeSelector { return TimeSelector{t: t} }

// Evaluator owns a compiled gomlx Exec and evaluates a fixed Formula under
// fixed Mode/Tie topology. Mutable scalars (scale, pscale) are fed per call
// as graph parameters and do not invalidate the cache.
//
// Shape caching: the underlying graph.Exec rebuilds the graph only when the
// input tensor shapes or dtypes change. Use SetMaxCache to tune.
//
// Concurrency: RobustnessTrace / Robustness / Vars may be called
// concurrently from multiple goroutines after construction. Close
// serializes with in-flight evaluations — it is safe to call once, from
// any goroutine, at shutdown. A second Close is a no-op.
type Evaluator struct {
	formula  Formula
	cfg      config
	varOrder []string
	exec     *graph.Exec
	mu       sync.RWMutex
	closed   bool
}

// NewEvaluator compiles formula under the given options and returns an
// Evaluator bound to backend.
func NewEvaluator(be backends.Backend, formula Formula, opts ...Option) *Evaluator {
	cfg := defaultConfig()
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.agm {
		panic("stlcg: WithAGM is not yet implemented (Phase D follow-up)")
	}
	if cfg.scale == 0 {
		cfg.mode = ModeExact
	}

	varOrder := formula.Vars()
	varCount := len(varOrder)

	graphFn := func(inputs []*graph.Node) *graph.Node {
		if len(inputs) != varCount+2 {
			panic(fmt.Sprintf("stlcg: graph expected %d inputs, got %d", varCount+2, len(inputs)))
		}
		c := newCompiler(cfg, varOrder, inputs)
		return c.compileFormula(formula)
	}

	exec, err := graph.NewExec(be, graphFn)
	if err != nil {
		panic(fmt.Errorf("stlcg: NewExec failed: %w", err))
	}

	return &Evaluator{
		formula:  formula,
		cfg:      cfg,
		varOrder: varOrder,
		exec:     exec,
	}
}

// SetMaxCache caps the number of distinct input-shape graphs kept compiled.
func (e *Evaluator) SetMaxCache(n int) *Evaluator {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exec.SetMaxCache(n)
	return e
}

// Close finalizes the underlying Exec and releases backend resources.
// Further calls to RobustnessTrace or Robustness will panic. Close is
// idempotent and safe to call from any goroutine.
func (e *Evaluator) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.exec.Finalize()
	e.closed = true
}

// Vars returns the ordered variable names this Evaluator expects in
// incoming SignalMaps.
func (e *Evaluator) Vars() []string {
	out := make([]string, len(e.varOrder))
	copy(out, e.varOrder)
	return out
}

// RobustnessTrace evaluates the formula against signals and returns the
// robustness trace of shape [B, T, 1]. Per-call options currently only
// accept WithScale/WithPScale; other options are ignored.
func (e *Evaluator) RobustnessTrace(signals SignalMap, perCall ...Option) *tensors.Tensor {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		panic("stlcg: Evaluator is closed")
	}
	args := e.assembleArgs(signals, perCall)
	out, err := e.exec.Exec1(args...)
	if err != nil {
		panic(fmt.Errorf("stlcg: Exec failed: %w", err))
	}
	return out
}

// Robustness returns the single-step robustness at the selected time as a
// tensor of shape [B, 1].
func (e *Evaluator) Robustness(signals SignalMap, at TimeSelector, perCall ...Option) *tensors.Tensor {
	trace := e.RobustnessTrace(signals, perCall...)
	defer trace.FinalizeAll()
	return sliceAtTime(trace, at.t)
}

func (e *Evaluator) assembleArgs(signals SignalMap, perCall []Option) []any {
	cfg := e.cfg
	for _, o := range perCall {
		o(&cfg)
	}

	args := make([]any, 0, len(e.varOrder)+2)
	for _, name := range e.varOrder {
		t, ok := signals[name]
		if !ok {
			panic(fmt.Sprintf("stlcg: SignalMap missing required variable %q", name))
		}
		args = append(args, t)
	}
	pscale := float32(cfg.pscale)
	tau := float32(math.Abs(cfg.scale))
	if cfg.mode == ModeExact {
		tau = 1.0 // unused inside the compiled graph but must be a valid scalar
	}
	args = append(args, pscale, tau)
	return args
}

// sliceAtTime extracts the [B, 1] slice at time t of a [B, T, 1] trace.
// Negative t counts from the end. Materializes the data via ConstFlatData
// and builds a fresh Tensor — intentionally simple for v1.
func sliceAtTime(trace *tensors.Tensor, t int) *tensors.Tensor {
	s := trace.Shape()
	if s.Rank() < 2 {
		panic(fmt.Sprintf("stlcg: robustness trace rank must be >= 2, got %d", s.Rank()))
	}
	b := s.Dimensions[0]
	tDim := s.Dimensions[1]
	feat := 1
	if s.Rank() >= 3 {
		feat = s.Dimensions[2]
	}
	if t < 0 {
		t = tDim + t
	}
	if t < 0 || t >= tDim {
		panic(fmt.Sprintf("stlcg: AtTime(%d) out of range for T=%d", t, tDim))
	}

	out := tensors.FromShape(shapes.Make(s.DType, b, feat))
	err := tensors.ConstFlatData(trace, func(in []float32) {
		werr := tensors.MutableFlatData(out, func(o []float32) {
			for bi := 0; bi < b; bi++ {
				for fi := 0; fi < feat; fi++ {
					o[bi*feat+fi] = in[(bi*tDim+t)*feat+fi]
				}
			}
		})
		if werr != nil {
			panic(werr)
		}
	})
	if err != nil {
		panic(err)
	}
	return out
}
