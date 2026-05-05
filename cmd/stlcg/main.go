// Command stlcg is a small CLI demo for the stlcg.go library. It loads
// a CSV-encoded signal trace, evaluates one of a fixed set of STL
// formulas over it, and prints the robustness trace. Optionally emits
// the formula's graphviz DOT.
//
// CSV format: single column named "x", one float per line (a header
// row is required). Multi-variable CLI support is future work.
//
// Example:
//
//	echo -e "x\n0.1\n0.5\n0.3\n-0.2\n0.7" | \
//	  stlcg -formula always-gt -threshold 0 -stdin -dot
package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/jpfielding/stlcg"
	"github.com/jpfielding/stlcg/viz"
)

func main() {
	var (
		tracePath = flag.String("trace", "", "path to CSV trace file (single column named \"x\"); overrides -stdin")
		fromStdin = flag.Bool("stdin", false, "read trace CSV from stdin")
		formula   = flag.String("formula", "always-gt", "formula kind: always-gt, always-lt, eventually-gt, eventually-lt")
		threshold = flag.Float64("threshold", 0.0, "predicate threshold constant")
		aFlag     = flag.Int("a", 0, "interval lower bound (>= 0)")
		bFlag     = flag.Int("b", -1, "interval upper bound (-1 = unbounded)")
		scale     = flag.Float64("scale", 1.0, "smooth-min/max temperature τ (0 = exact mode)")
		emitDOT   = flag.Bool("dot", false, "also emit Graphviz DOT to stdout")
	)
	flag.Parse()

	if *tracePath == "" && !*fromStdin {
		fmt.Fprintln(os.Stderr, "stlcg: must supply -trace <path> or -stdin")
		flag.Usage()
		os.Exit(2)
	}

	var src io.Reader
	if *tracePath != "" {
		f, err := os.Open(*tracePath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stlcg:", err)
			os.Exit(1)
		}
		defer f.Close()
		src = f
	} else {
		src = bufio.NewReader(os.Stdin)
	}

	row, err := readSingleColumnCSV(src, "x")
	if err != nil {
		fmt.Fprintln(os.Stderr, "stlcg:", err)
		os.Exit(1)
	}
	if len(row) == 0 {
		fmt.Fprintln(os.Stderr, "stlcg: trace is empty")
		os.Exit(1)
	}

	iv := stlcg.AllTime()
	if *bFlag >= 0 {
		iv = stlcg.Bounds(*aFlag, *bFlag)
	} else if *aFlag > 0 {
		iv = stlcg.From(*aFlag)
	}

	phi, err := buildFormula(*formula, *threshold, iv)
	if err != nil {
		fmt.Fprintln(os.Stderr, "stlcg:", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "formula: %s\n", phi)

	be := backends.MustNew()
	var opts []stlcg.Option
	if *scale == 0 {
		opts = append(opts, stlcg.WithMode(stlcg.ModeExact), stlcg.WithScale(0))
	} else {
		opts = append(opts, stlcg.WithMode(stlcg.ModeSmooth), stlcg.WithScale(*scale))
	}
	eval := stlcg.NewEvaluator(be, phi, opts...)
	defer eval.Close()

	tr := tensors.FromShape(shapes.Make(dtypes.Float32, 1, len(row), 1))
	if err := tensors.MutableFlatData(tr, func(d []float32) {
		for i, v := range row {
			d[i] = float32(v)
		}
	}); err != nil {
		fmt.Fprintln(os.Stderr, "stlcg:", err)
		os.Exit(1)
	}
	defer tr.FinalizeAll()

	out := eval.RobustnessTrace(stlcg.SignalMap{"x": tr})
	defer out.FinalizeAll()

	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	_ = tensors.ConstFlatData(out, func(d []float32) {
		fmt.Fprintln(w, "t,rho")
		for t, v := range d {
			fmt.Fprintf(w, "%d,%+.6f\n", t, v)
		}
	})

	if *emitDOT {
		fmt.Fprintln(os.Stderr, "--- DOT ---")
		_ = viz.WriteDOT(os.Stdout, phi)
	}
}

func buildFormula(kind string, threshold float64, iv stlcg.Interval) (stlcg.Formula, error) {
	x := stlcg.Var("x")
	c := stlcg.Const(threshold)
	switch kind {
	case "always-gt":
		return stlcg.Always(stlcg.Gt(x, c), iv), nil
	case "always-lt":
		return stlcg.Always(stlcg.Lt(x, c), iv), nil
	case "eventually-gt":
		return stlcg.Eventually(stlcg.Gt(x, c), iv), nil
	case "eventually-lt":
		return stlcg.Eventually(stlcg.Lt(x, c), iv), nil
	}
	return nil, fmt.Errorf("unknown -formula %q", kind)
}

func readSingleColumnCSV(r io.Reader, expectedHeader string) ([]float64, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = 1

	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	if len(header) != 1 || header[0] != expectedHeader {
		return nil, fmt.Errorf("expected single column header %q, got %v", expectedHeader, header)
	}

	var out []float64
	for line := 2; ; line++ {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		v, err := strconv.ParseFloat(rec[0], 64)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		out = append(out, v)
	}
	return out, nil
}
