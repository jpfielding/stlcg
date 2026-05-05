package logger_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/jpfielding/stlcg"
	"github.com/jpfielding/stlcg/logger"
)

func TestJSONLScalarAndHistogram(t *testing.T) {
	var buf bytes.Buffer
	l := logger.NewJSONL(&buf)

	l.Scalar("loss", 1, 0.5)
	l.Scalar("loss", 2, 0.4)
	l.Histogram("grad_norms", 2, []float64{0.1, 0.2, 0.3})
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}

	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 records, got %d:\n%s", len(lines), buf.String())
	}

	var first map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if first["kind"] != "scalar" || first["tag"] != "loss" {
		t.Errorf("unexpected first record: %v", first)
	}

	var hist map[string]any
	if err := json.Unmarshal([]byte(lines[2]), &hist); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if hist["kind"] != "hist" {
		t.Errorf("expected hist kind, got: %v", hist)
	}
	vs, ok := hist["values"].([]any)
	if !ok || len(vs) != 3 {
		t.Errorf("histogram values missing or wrong length: %v", hist["values"])
	}
}

func TestJSONLCloseIsIdempotent(t *testing.T) {
	l := logger.NewJSONL(&bytes.Buffer{})
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	// No panic, no error, and no more records accepted.
	l.Scalar("x", 1, 1)
}

func TestNopLogger(t *testing.T) {
	var l logger.Logger = logger.NopLogger{}
	l.Scalar("x", 1, 1)
	l.Histogram("x", 1, []float64{1})
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestRobustnessMetricBuildsGraph exercises the gomlx metrics.Interface
// hookup via the graph-level BuildRobustnessTrace seam. We don't require
// train.Trainer here (that's covered by users' own training loops); we
// just verify UpdateGraph produces a correct scalar mean robustness.
func TestRobustnessMetricBuildsGraph(t *testing.T) {
	be := backends.MustNew()

	x := stlcg.Var("x")
	phi := stlcg.Gt(x, stlcg.Const(0.5))
	metric := logger.NewRobustnessMetric("rho", phi, stlcg.WithMode(stlcg.ModeExact))

	// Build a one-shot Exec that treats the batched signal as the single
	// "prediction" and returns the metric's UpdateGraph output.
	fn := func(signal *graph.Node) *graph.Node {
		return metric.UpdateGraph(nil, nil, []*graph.Node{signal})
	}
	exec, err := graph.NewExec(be, fn)
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()

	// Batch of 1, T=4, F=1: signal = [0, 1, 2, 3]. Robustness = signal - 0.5.
	// Mean = (−0.5 + 0.5 + 1.5 + 2.5)/4 = 1.0.
	trace := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 4, 1))
	if err := tensors.MutableFlatData(trace, func(d []float32) {
		d[0] = 0
		d[1] = 1
		d[2] = 2
		d[3] = 3
	}); err != nil {
		t.Fatal(err)
	}

	out, err := exec.Exec1(trace)
	if err != nil {
		t.Fatal(err)
	}
	defer out.FinalizeAll()

	var got float32
	if err := tensors.ConstFlatData(out, func(d []float32) { got = d[0] }); err != nil {
		t.Fatal(err)
	}
	if got < 0.99 || got > 1.01 {
		t.Errorf("mean robustness = %g, want ≈ 1.0", got)
	}

	// PrettyPrint sanity.
	if got := metric.PrettyPrint(out); !strings.Contains(got, "+1") {
		t.Errorf("PrettyPrint = %q, want something with +1", got)
	}
}

func TestRobustnessMetricInterfaceShape(t *testing.T) {
	m := logger.NewRobustnessMetric("foo", stlcg.Gt(stlcg.Var("x"), stlcg.Const(0.0)))
	if m.Name() != "foo" {
		t.Errorf("Name = %q", m.Name())
	}
	if m.ShortName() == "" {
		t.Errorf("ShortName empty")
	}
	if m.ScopeName() == "" {
		t.Errorf("ScopeName empty")
	}
	if m.MetricType() == "" {
		t.Errorf("MetricType empty")
	}
	m.Reset(nil) // must be a no-op, not a panic
}
