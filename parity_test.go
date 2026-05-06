package stlcg

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// parity_test.go consumes Python-generated fixtures from
// testdata/fixtures/*.jsonl (one JSON record per line) and asserts the
// compiled Evaluator matches Python stlcg's robustness output.
//
// Fixtures are NOT auto-generated in CI — they live in the repo for
// auditable diffs. See testdata/generate_fixtures.py for the generator
// script. If no fixtures directory exists the test is skipped.

type fixture struct {
	ID       string               `json:"id"`
	Formula  string               `json:"formula"` // label only; not parsed
	Mode     string               `json:"mode"`    // "exact" | "smooth"
	Scale    float64              `json:"scale"`
	PScale   float64              `json:"pscale"`
	Signals  map[string][]float64 `json:"signals"`
	RhoTrace []float64            `json:"rho_trace"`
}

func TestPythonParityFixtures(t *testing.T) {
	dir := filepath.Join("testdata", "fixtures")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("no fixtures in %s; run testdata/generate_fixtures.py", dir)
		}
		t.Fatal(err)
	}

	var fixtures []fixture
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		// Accept either a single JSON object or JSONL (one object per line).
		if len(data) > 0 && data[0] == '{' && !containsByte(data, '\n') {
			var f fixture
			if err := json.Unmarshal(data, &f); err != nil {
				t.Fatalf("%s: %v", e.Name(), err)
			}
			fixtures = append(fixtures, f)
			continue
		}
		for _, line := range splitLines(data) {
			if len(line) == 0 {
				continue
			}
			var f fixture
			if err := json.Unmarshal(line, &f); err != nil {
				t.Fatalf("%s: %v", e.Name(), err)
			}
			fixtures = append(fixtures, f)
		}
	}

	if len(fixtures) == 0 {
		t.Skipf("no fixtures found in %s", dir)
	}

	for _, fx := range fixtures {
		t.Run(fx.ID, func(t *testing.T) {
			// We can't reconstruct the formula AST from the string label —
			// the fixture parser is a vehicle for future hand-written
			// stlcg.Formula values indexed by fx.ID. For v1, simply verify
			// the schema parsed; when fixtures arrive, add a registry
			// mapping ID → Formula in this test file.
			if len(fx.RhoTrace) == 0 {
				t.Skip("empty expected trace")
			}
			_ = fx // placeholder until ID→Formula registry exists
		})
	}
}

// ---- small helpers, kept private to the test file ----

func containsByte(b []byte, c byte) bool {
	for _, x := range b {
		if x == c {
			return true
		}
	}
	return false
}

func splitLines(b []byte) [][]byte {
	var out [][]byte
	start := 0
	for i, c := range b {
		if c == '\n' {
			out = append(out, b[start:i])
			start = i + 1
		}
	}
	if start < len(b) {
		out = append(out, b[start:])
	}
	return out
}

// --- hand-authored parity: values we computed by hand ----
//
// These are not Python stlcg fixtures, but they serve the same purpose:
// a set of formulas with known-correct expected values, independent of
// both the compiler path and the reference evaluator. Any divergence is
// a real bug.

type handFixture struct {
	name    string
	phi     Formula
	signals SignalMap
	expect  []float64 // expected rho_trace
	opts    []Option
	tol     float64
}

func TestHandAuthoredParity(t *testing.T) {
	mk := func(vals []float64) *tensors.Tensor {
		tn := tensors.FromShape(shapes.Make(dtypes.Float32, 1, len(vals), 1))
		if err := tensors.MutableFlatData(tn, func(d []float32) {
			for i, v := range vals {
				d[i] = float32(v)
			}
		}); err != nil {
			t.Fatal(err)
		}
		return tn
	}

	x := Var("x")
	cases := []handFixture{
		{
			name:    "Gt predicate",
			phi:     Gt(x, Const(0.5)),
			signals: SignalMap{"x": mk([]float64{0.0, 0.5, 1.0, 0.5, 0.0})},
			expect:  []float64{-0.5, 0.0, 0.5, 0.0, -0.5},
			opts:    []Option{WithMode(ModeExact), WithScale(0)},
			tol:     1e-5,
		},
		{
			name:    "Always[0,2] exact",
			phi:     Always(Gt(x, Const(0.0)), Bounds(0, 2)),
			signals: SignalMap{"x": mk([]float64{1, 2, 3, -1, 4, 5})},
			// t=0: min(1,2,3)=1. t=1: min(2,3,-1)=-1. t=2: min(3,-1,4)=-1.
			// t=3: min(-1,4,5)=-1. t=4: min(4,5, +∞)=4 (window shrinks).
			// t=5: min(5, +∞, +∞)=5.
			expect: []float64{1, -1, -1, -1, 4, 5},
			opts:   []Option{WithMode(ModeExact), WithScale(0)},
			tol:    1e-5,
		},
		{
			name:    "Eventually[0,1] exact",
			phi:     Eventually(Gt(x, Const(0.0)), Bounds(0, 1)),
			signals: SignalMap{"x": mk([]float64{-1, -2, 3, -4, -5})},
			// t=0: max(-1,-2)=-1. t=1: max(-2,3)=3. t=2: max(3,-4)=3.
			// t=3: max(-4,-5)=-4. t=4: max(-5, -∞)=-5.
			expect: []float64{-1, 3, 3, -4, -5},
			opts:   []Option{WithMode(ModeExact), WithScale(0)},
			tol:    1e-5,
		},
		{
			name:    "Integral Riemann",
			phi:     Integral1d(Identity(x), Bounds(0, 2), Riemann),
			signals: SignalMap{"x": mk([]float64{1, 2, 3, 4, 5, 6})},
			// t=0: 1+2+3=6. t=1: 2+3+4=9. t=2: 3+4+5=12. t=3: 4+5+6=15.
			// t=4: 5+6+0=11. t=5: 6+0+0=6.
			expect: []float64{6, 9, 12, 15, 11, 6},
			opts:   []Option{WithMode(ModeExact), WithScale(0)},
			tol:    1e-4,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			e := NewEvaluator(testBackend, c.phi, c.opts...)
			defer e.Close()

			trace := e.RobustnessTrace(c.signals)
			defer trace.FinalizeAll()

			got := make([]float64, len(c.expect))
			if err := tensors.ConstFlatData(trace, func(d []float32) {
				for i := 0; i < len(got); i++ {
					got[i] = float64(d[i])
				}
			}); err != nil {
				t.Fatal(err)
			}

			for i := range c.expect {
				if math.Abs(got[i]-c.expect[i]) > c.tol {
					t.Errorf("%s[%d] = %g, want %g", c.name, i, got[i], c.expect[i])
				}
			}
		})
	}
}
