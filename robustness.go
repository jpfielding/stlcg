package stlcg

import (
	"fmt"
	"math"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// defaultDType is the only dtype supported in v1. Multi-dtype support is
// tracked for v1.1+.
const defaultDType = dtypes.Float32

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
// Shape caching: the underlying graph.Exec rebuilds the graph only when
// the input tensor shapes or dtypes change, per gomlx's shapesMatch
// comparison. v1 is Float32-only, so in practice only batch and time
// dimensions drive cache misses. Use SetMaxCache to tune.
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

// SetMaxCache caps the number of distinct input-shape graphs kept
// compiled.
//
// The underlying gomlx cache is a HARD CAP, not an LRU: once n distinct
// shapes have been compiled, a further distinct shape causes Exec to
// error rather than evicting an older entry. Set n large enough to cover
// every shape the Evaluator will see, or set n=-1 for unbounded. When
// shapes are highly dynamic, pad traces to a fixed length instead.
func (e *Evaluator) SetMaxCache(n int) *Evaluator {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exec.SetMaxCache(n)
	return e
}

// Precompile warms the shape cache by running the Evaluator once for each
// (batch, timeLen) pair. Each pair triggers a graph build + JIT compile
// so that subsequent RobustnessTrace calls with matching shapes hit the
// cache instead of paying compile cost.
//
// All input tensors are assumed to share the [batch, timeLen, 1] layout
// that the library expects across every signal variable. Dtype defaults
// to Float32 (the only supported dtype in v1).
//
// Returns the first error encountered. Compile cache may be partially
// populated on error. Precompile is safe to call concurrently; it takes
// the same read lock as RobustnessTrace.
func (e *Evaluator) Precompile(shapes ...[2]int) error {
	for _, s := range shapes {
		b, tLen := s[0], s[1]
		if b <= 0 || tLen <= 0 {
			return fmt.Errorf("stlcg: Precompile: invalid shape [batch=%d, time=%d]", b, tLen)
		}
		if err := e.precompileOne(b, tLen); err != nil {
			return err
		}
	}
	return nil
}

// precompileOne drives a single (batch, time) shape through the graph
// and cleans up all allocated tensors even on error. Split out so the
// deferred cleanup scope is per-shape, not per-Precompile call.
func (e *Evaluator) precompileOne(b, tLen int) error {
	sig := make(SignalMap, len(e.varOrder))
	for _, name := range e.varOrder {
		sig[name] = zeroTraceTensor(b, tLen)
	}
	defer func() {
		for _, t := range sig {
			t.FinalizeAll()
		}
	}()
	out, err := e.RobustnessTraceE(sig)
	if err != nil {
		return fmt.Errorf("stlcg: Precompile shape [%d,%d]: %w", b, tLen, err)
	}
	out.FinalizeAll()
	return nil
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
//
// Panics on runtime error (closed evaluator, missing signal, exec failure).
// Use RobustnessTraceE for an error-returning variant suitable for
// library-embedded use.
func (e *Evaluator) RobustnessTrace(signals SignalMap, perCall ...Option) *tensors.Tensor {
	out, err := e.RobustnessTraceE(signals, perCall...)
	if err != nil {
		panic(err)
	}
	return out
}

// RobustnessTraceE is the error-returning counterpart of RobustnessTrace.
// Runtime errors (closed evaluator, missing signal variable, gomlx Exec
// failure, interval-exceeds-trace-length) are wrapped with the sentinel
// errors in errors.go. Programmer invariants (unknown AST types, arity
// mismatches) still panic.
func (e *Evaluator) RobustnessTraceE(signals SignalMap, perCall ...Option) (*tensors.Tensor, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed {
		return nil, ErrClosed
	}
	args, err := e.assembleArgs(signals, perCall)
	if err != nil {
		return nil, err
	}
	// Pre-flight: reject formulas whose bounded interval lower bound
	// exceeds the runtime trace length. The compiler would otherwise
	// panic from inside gomlx's graph-build goroutine, leaving the Exec
	// in a state that deadlocks on Finalize. Pre-validation avoids the
	// panic entirely.
	if T, ok := traceLength(signals); ok {
		if err := validateIntervalsForT(e.formula, T); err != nil {
			return nil, err
		}
	}
	out, err := e.exec.Exec1(args...)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrExec, err)
	}
	return out, nil
}

// traceLength peeks the axis-1 size of any signal tensor. All signals
// share the same [B, T, 1] layout. Returns ok=false if the map is empty
// or signals lack rank 2+ (covered by the normal ErrBadShape path).
func traceLength(signals SignalMap) (int, bool) {
	for _, t := range signals {
		sh := t.Shape()
		if sh.Rank() >= 2 {
			return sh.Dimensions[1], true
		}
	}
	return 0, false
}

// Robustness returns the single-step robustness at the selected time as a
// tensor of shape [B, 1].
//
// Panics on runtime error; see RobustnessE for the error-returning form.
func (e *Evaluator) Robustness(signals SignalMap, at TimeSelector, perCall ...Option) *tensors.Tensor {
	out, err := e.RobustnessE(signals, at, perCall...)
	if err != nil {
		panic(err)
	}
	return out
}

// RobustnessE is the error-returning counterpart of Robustness.
func (e *Evaluator) RobustnessE(signals SignalMap, at TimeSelector, perCall ...Option) (*tensors.Tensor, error) {
	trace, err := e.RobustnessTraceE(signals, perCall...)
	if err != nil {
		return nil, err
	}
	defer trace.FinalizeAll()
	return sliceAtTimeE(trace, at.t)
}

func (e *Evaluator) assembleArgs(signals SignalMap, perCall []Option) ([]any, error) {
	cfg := e.cfg
	for _, o := range perCall {
		o(&cfg)
	}

	args := make([]any, 0, len(e.varOrder)+2)
	for _, name := range e.varOrder {
		t, ok := signals[name]
		if !ok {
			return nil, fmt.Errorf("%w: %q", ErrMissingSignal, name)
		}
		if dt := t.Shape().DType; dt != defaultDType {
			return nil, fmt.Errorf("%w: signal %q has dtype %v, want %v (multi-dtype support is a v1.1+ roadmap item)",
				ErrBadShape, name, dt, defaultDType)
		}
		args = append(args, t)
	}
	pscale := float32(cfg.pscale)
	tau := float32(math.Abs(cfg.scale))
	if cfg.mode == ModeExact {
		tau = 1.0 // unused inside the compiled graph but must be a valid scalar
	}
	args = append(args, pscale, tau)
	return args, nil
}

// zeroTraceTensor builds a [batch, timeLen, 1] Float32 tensor filled with
// zeros — used by Precompile to drive graph construction without
// caring about tensor contents.
func zeroTraceTensor(batch, timeLen int) *tensors.Tensor {
	return tensors.FromShape(shapes.Make(defaultDType, batch, timeLen, 1))
}

// sliceAtTimeE extracts the [B, 1] slice at time t of a [B, T, 1] trace.
// Negative t counts from the end. v1 uses a host roundtrip (ConstFlatData
// + MutableFlatData); for hot training loops, prefer BuildRobustnessTrace
// + a manual graph.Slice.
func sliceAtTimeE(trace *tensors.Tensor, t int) (*tensors.Tensor, error) {
	s := trace.Shape()
	if s.Rank() < 2 {
		return nil, fmt.Errorf("%w: robustness trace rank must be >= 2, got %d", ErrBadShape, s.Rank())
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
		return nil, fmt.Errorf("%w: AtTime(%d) out of range for T=%d", ErrTimeOutOfRange, t, tDim)
	}

	out := tensors.FromShape(shapes.Make(s.DType, b, feat))
	var loopErr error
	err := tensors.ConstFlatData(trace, func(in []float32) {
		werr := tensors.MutableFlatData(out, func(o []float32) {
			for bi := 0; bi < b; bi++ {
				for fi := 0; fi < feat; fi++ {
					o[bi*feat+fi] = in[(bi*tDim+t)*feat+fi]
				}
			}
		})
		if werr != nil {
			loopErr = werr
		}
	})
	if err != nil {
		out.FinalizeAll()
		return nil, fmt.Errorf("%w: %v", ErrExec, err)
	}
	if loopErr != nil {
		out.FinalizeAll()
		return nil, fmt.Errorf("%w: %v", ErrExec, loopErr)
	}
	return out, nil
}
