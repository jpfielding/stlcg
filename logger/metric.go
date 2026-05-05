package logger

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/jpfielding/stlcg"
)

// RobustnessMetric is a gomlx train.Trainer-compatible metric (satisfies
// metrics.Interface) that reports the mean STL robustness of a formula
// across a batch.
//
// The Trainer passes `predictions []*graph.Node` to UpdateGraph; the
// metric maps each of the formula's Vars(), in sorted order, to one
// entry of that slice. In other words, wrap your model so its
// `predictions` output is a slice whose element i is the tensor for
// the i-th variable in `phi.Vars()`. The simplest case: single-variable
// formulas with predictions=[x_pred].
//
// PScale and Scale are compiled into the metric graph as constants; to
// anneal them at training time, construct a new metric between runs.
type RobustnessMetric struct {
	name       string
	shortName  string
	scopeName  string
	metricType string

	formula stlcg.Formula
	opts    []stlcg.Option
	pscale  float64
	scale   float64
}

// NewRobustnessMetric builds a metric with the given label and formula.
// Name defaults to "robustness" when empty.
func NewRobustnessMetric(name string, phi stlcg.Formula, opts ...stlcg.Option) *RobustnessMetric {
	if name == "" {
		name = "robustness"
	}
	return &RobustnessMetric{
		name:       name,
		shortName:  "rho",
		scopeName:  "stlcg/" + name,
		metricType: "robustness",
		formula:    phi,
		opts:       opts,
		pscale:     1.0,
		scale:      1.0,
	}
}

// WithPScale / WithScale set the compile-time scalar values. These are
// baked into the metric graph — adjust by constructing a new metric.
func (m *RobustnessMetric) WithPScale(v float64) *RobustnessMetric { m.pscale = v; return m }
func (m *RobustnessMetric) WithScale(v float64) *RobustnessMetric  { m.scale = v; return m }

// --- metrics.Interface ---

func (m *RobustnessMetric) Name() string       { return m.name }
func (m *RobustnessMetric) ShortName() string  { return m.shortName }
func (m *RobustnessMetric) ScopeName() string  { return m.scopeName }
func (m *RobustnessMetric) MetricType() string { return m.metricType }

// UpdateGraph builds a scalar node equal to the mean robustness of the
// formula when evaluated on predictions. Predictions are positionally
// aligned with the formula's Vars() (sorted).
func (m *RobustnessMetric) UpdateGraph(_ *context.Context, _ []*graph.Node, predictions []*graph.Node) *graph.Node {
	vars := m.formula.Vars()
	if len(predictions) < len(vars) {
		panic(fmt.Sprintf("stlcg/logger: RobustnessMetric needs %d prediction nodes for vars %v, got %d",
			len(vars), vars, len(predictions)))
	}

	signals := make(map[string]*graph.Node, len(vars))
	for i, name := range vars {
		signals[name] = predictions[i]
	}

	g := predictions[0].Graph()
	dtype := predictions[0].DType()
	pscale := graph.Scalar(g, dtype, m.pscale)
	tau := graph.Scalar(g, dtype, m.scale)

	trace := stlcg.BuildRobustnessTrace(m.formula, signals, pscale, tau, m.opts...)
	return graph.ReduceAllMean(trace)
}

// PrettyPrint renders a scalar tensor as a signed robustness value.
func (m *RobustnessMetric) PrettyPrint(value *tensors.Tensor) string {
	if !value.Shape().IsScalar() {
		return value.String()
	}
	switch v := value.Value().(type) {
	case float32:
		return fmt.Sprintf("%+.4g", v)
	case float64:
		return fmt.Sprintf("%+.4g", v)
	}
	return value.String()
}

// Reset is a no-op — RobustnessMetric is stateless per batch.
func (m *RobustnessMetric) Reset(_ *context.Context) {}
