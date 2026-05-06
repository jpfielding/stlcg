# Changelog

All notable changes to stlcg.go are recorded here. Semver once v1.0.0
lands; pre-1.0 releases may break.

## [Unreleased]

Changes that have not yet been tagged.

## [0.1.1] — 2026-05-05

### Fixed

- `logger.NewRobustnessMetric` now honors scalar options
  (`stlcg.WithPScale`, `stlcg.WithScale`) passed through its variadic
  `opts` parameter. Previously such options were stored and forwarded to
  `BuildRobustnessTrace`, which intentionally ignores scalar option
  *values* — so callers got `pscale=1.0` and `scale=1.0` silently. The
  constructor now resolves scalars into the metric's own fields
  up-front. Regression covered by
  `TestRobustnessMetric_ConstructorScalarOptions`.

### Added

- Public `stlcg.Config` readonly view + `stlcg.ResolveConfig(opts ...Option) Config`
  helper exposing `Mode()`, `Tie()`, `PScale()`, and `Scale()` accessors.
  Used by `logger.NewRobustnessMetric` and available to any caller that
  needs to inspect the resolved effect of an `[]Option`.

### Removed

- Three unused internal helpers deleted (staticcheck `U1000`):
  `compile.padTimeAxisLeft` (O(L) Until/Then rewrite made it unreachable),
  `expr.varName` (superseded by direct AST walking in `walk.go`),
  `robustness.sliceAtTime` (panic-wrapper over `sliceAtTimeE`; the
  panic/`E` convention is already represented by the exported
  `Robustness`/`RobustnessE` pair). No public-API impact.

### Docs

- README JSONL example now propagates `os.Create` errors instead of
  discarding them with `_`.

### Test

- `TestRobustnessMetric_PrettyPrintNonScalar` asserts the non-scalar
  fallback of `RobustnessMetric.PrettyPrint` against
  `tensors.Tensor.String()`.

## [0.1.0] — 2026-05-05 (v1.0.0 candidate)

> CHANGELOG correction: the entries below marked `(post-0.1.0 CHANGELOG
> fold-in)` shipped in the v0.1.0 tag (`9451527`) but were mis-filed
> under `[Unreleased]` in that release's tree. They are folded in here
> retroactively in v0.1.1.

### Fixed (post-0.1.0 CHANGELOG fold-in)

- `robustness.go:sliceAtTimeE` now finalizes its output tensor on error
  paths (was leaked) and reports data-access failures as `ErrExec`
  rather than `ErrBadShape`.
- `Evaluator.Precompile` now surfaces Exec errors via the return value
  and finalizes signal tensors on all paths (previously panicked via
  the wrapping `RobustnessTrace` and leaked tensors on failure).

### Added (post-0.1.0 CHANGELOG fold-in)

- `RobustnessTraceE` pre-flight rejects formulas whose bounded
  temporal interval lower bound exceeds the runtime trace length,
  returning `ErrBadShape` instead of panicking through gomlx's lazy
  compile. Covered by `TestRobustnessTraceE_IntervalExceedsTrace`.
- `WithMode` and `WithTieGradient` now panic at option-construction on
  invalid enum values, not during graph compilation.
- `parity_test.go` enforces an ID→Formula registry and aggregates
  unregistered fixture IDs into a single test failure — closes the
  "parses fixtures but compares nothing" silent-pass bug.

### Changed (post-0.1.0 CHANGELOG fold-in)

- `go.mod` minimum Go version relaxed from 1.26.1 to 1.25 (gomlx's
  floor). CI matrix runs 1.25.x and 1.26.x.

### Breaking

- **Removed `WithAGM`** and the `config.agm` field. The option was
  non-functional: any evaluator constructed with `agm=true` panicked.
  AGM (arithmetic-geometric mean) robustness is a v2 roadmap item.
- **Removed `WithKeepDim`** and the `config.keep` field. The option was
  set by callers but never read by the compile path (reducePair,
  slidingReduce, compileUntilThen all hardcoded keepDim=false). Dead
  since Phase C.
- **`TieArgmax` docs corrected, not semantics.** The prior claim that
  `TieArgmax` "sends the full gradient to a single argmax (XLA default)"
  was wrong for gomlx: ReduceMax/ReduceMin's default VJP routes grad=1
  to EACH tied slot (so the gradient vector sums to the number of ties,
  not to 1). The code always did this; the documentation and
  expectations have been corrected. Use `TieUniform` when
  d(extremum)/dx must sum to 1.
- **Non-Float32 inputs now error early.** Passing a tensor with dtype
  other than Float32 to `RobustnessTrace*` or `Robustness*` returns
  (or panics with) `ErrBadShape` plus a message pointing to the
  multi-dtype roadmap. Previously the tensor was passed to gomlx and
  failed with a less helpful error.

### Added

- **`Evaluator.Precompile(shapes ...[2]int) error`** — warms the JIT
  cache for known (batch, timeLen) pairs so subsequent
  `RobustnessTrace` calls with those shapes skip the compile path.
- **Error-returning variants**: `RobustnessTraceE`, `RobustnessE`.
  Sentinel errors in `errors.go`: `ErrClosed`, `ErrMissingSignal`,
  `ErrTimeOutOfRange`, `ErrBadShape`, `ErrExec`. Panicking methods
  remain for example code and now wrap the same errors so recovery
  can use `errors.Is`.
- **`overlap` flag on Until/Then is now honored** (previously silently
  ignored). `overlap=true` is the conventional strong-until semantics
  (phi holds on `[t, s]`); `overlap=false` uses `[t, s-1]` so psi
  holding at `t=s` is sufficient.
- **Concurrency safety for `Evaluator`.** `RobustnessTrace*`, `Vars`,
  and `Precompile` take a read lock; `Close` takes a write lock.
  Evaluators can now be shared across goroutines (tested under
  `go test -race`).

### Changed

- **Until/Then graph size is now O(L)**, not O(L²). Prior implementation
  rebuilt phi's prefix reduction with a fresh `slidingReduce` at each
  offset k (L calls of O(L) slices each). The new compile path seeds
  the prefix once, then extends by a single sample per step.
  `Bounds(0, 50)` drops from ~1,300 stacked slices to ~100.
- **`SetMaxCache` is documented as a hard cap**, not an LRU. gomlx's
  underlying cache errors out on the (n+1)th distinct shape rather
  than evicting; docs now reflect that and the test suite locks down
  the observed behavior.
- **`BuildRobustnessTrace` option list** no longer mentions KeepDim /
  AGM.

### Fixed

- `doc.go` no longer references `stlcg.Interval(0, 50)` (not a
  function) or `Evaluator.Precompile` (doesn't exist — now
  implemented).
- `compile.go:compileUntilThen` no longer discards the `overlap`
  argument with `_ = overlap`.

### Test

- New tests: `concurrent_test.go`, `precompile_test.go`,
  `shape_cache_test.go`, `walk_test.go`, `exports_test.go`,
  `errors_test.go`, plus `minmax/minmax_test.go` (closes 0% coverage
  on the reduction kernel). `TestUntilOverlapFalse` exercises the
  new overlap branch against the reference evaluator and hand-computed
  expectations.
- CI workflow added at `.github/workflows/ci.yml`: `go vet`,
  `go test -race`, `go build`, `staticcheck` on push and PR to `main`.

### Known issues / deferred

- Python-parity fixtures (`testdata/fixtures/*.jsonl`) still need to
  be generated against a pinned upstream commit of
  `stanfordASL/stlcg`. `parity_test.go` skips when the directory is
  absent; the generator (`testdata/generate_fixtures.py`) is committed
  but unexecuted. Tracked at
  <https://github.com/jpfielding/stlcg/issues/1>.
- `sliceAtTime` still does a host roundtrip; in-graph slice path is a
  v1.1 item (does not affect training loops that use
  `BuildRobustnessTrace`).
- Multi-dtype support (Float64, BFloat16) is v1.1+.
- A `stlcg` binary (22 MB Mach-O arm64) was committed in
  `f3aa040` and removed in `ee9ae8c`; the blob remains in git
  history. Rewriting public history is deliberately left for a
  separately authorized BFG pass.

## [0.0.0] — initial implementation (pre-review)

The 10-phase port from `stanfordASL/stlcg` (Python/PyTorch) to
gomlx/XLA. See commits `6848693..d9f4eef` for phased deliverables:
skeleton (A), AST (B), predicates + logical (C), Always/Eventually
(D), Until/Then/Integral1d (E), viz (F), logger + metrics (G),
examples + CLI (H), docs + property/gradient tests (I), benchmarks
(J).
