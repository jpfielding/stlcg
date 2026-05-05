# stlcg.go

Signal Temporal Logic robustness on a differentiable computation graph, in Go.

A transliteration of [stanfordASL/stlcg](https://github.com/stanfordASL/stlcg)
(PyTorch) onto [gomlx](https://github.com/gomlx/gomlx) (XLA). Use STL
specifications as differentiable terms in a neural-network loss.

## Status

Under construction. Phased roadmap lives in the plan file; public API not yet
frozen.

## Time axis

Unlike the Python original, stlcg.go uses **natural forward time**. Trace
tensors are `[batch, time, feature]` with index 0 the earliest observation.
Python's time-reversed convention has been dropped. See [`doc.go`](doc.go)
and any parity fixtures in `testdata/fixtures/` for the formal index map.

## Target API

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
    stlcg.Interval(0, 50),
)

be := backends.MustNew()
eval := stlcg.NewEvaluator(be, phi, stlcg.WithScale(1.0), stlcg.WithPScale(1.0))
defer eval.Close()

trace := stlcg.NewSignalMap(map[string]*tensors.Tensor{"x": xTensor, "y": yTensor})
rho := eval.Robustness(trace, stlcg.AtTime(0))
```

## License

MIT. Portions derived from stanfordASL/stlcg, also MIT.
