// Command simple demonstrates constructing an STL formula, evaluating
// its robustness on a synthetic trace, and printing the trace.
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
	// phi = Always_[0,5] ((x > 0) AND NOT (y < -1))
	x := stlcg.Var("x")
	y := stlcg.Var("y")
	phi := stlcg.Always(
		stlcg.And(
			stlcg.Gt(x, stlcg.Const(0.0)),
			stlcg.Not(stlcg.Lt(y, stlcg.Const(-1.0))),
		),
		stlcg.Bounds(0, 5),
	)
	fmt.Printf("formula: %s\n", phi)

	be := backends.MustNew()
	eval := stlcg.NewEvaluator(be, phi,
		stlcg.WithMode(stlcg.ModeSmooth),
		stlcg.WithScale(5.0),
		stlcg.WithPScale(1.0),
	)
	defer eval.Close()

	// Synthetic trace, T=12, B=1. x is noisy-positive, y stays around 0.
	xT := tensorFromRow([]float32{0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.9, 0.6, 1.0, 0.3, 0.7})
	yT := tensorFromRow([]float32{0.5, 0.3, 0.0, -0.5, -0.8, -0.2, 0.1, 0.4, 0.0, -0.3, 0.2, 0.5})
	defer xT.FinalizeAll()
	defer yT.FinalizeAll()

	signals := stlcg.SignalMap{"x": xT, "y": yT}
	trace := eval.RobustnessTrace(signals)
	defer trace.FinalizeAll()

	fmt.Println("robustness trace:")
	_ = tensors.ConstFlatData(trace, func(data []float32) {
		for t, v := range data {
			fmt.Printf("  t=%2d  %+.4f\n", t, v)
		}
	})

	rho := eval.Robustness(signals, stlcg.AtTime(0))
	defer rho.FinalizeAll()
	_ = tensors.ConstFlatData(rho, func(data []float32) {
		fmt.Printf("robustness at t=0: %+.4f\n", data[0])
	})
}

func tensorFromRow(row []float32) *tensors.Tensor {
	t := tensors.FromShape(shapes.Make(dtypes.Float32, 1, len(row), 1))
	if err := tensors.MutableFlatData(t, func(d []float32) { copy(d, row) }); err != nil {
		panic(err)
	}
	return t
}
