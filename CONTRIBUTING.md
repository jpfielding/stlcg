# Contributing to stlcg.go

Thanks for wanting to help. This doc covers what we expect from patches.

## Ground rules

1. **Parity over novelty.** stlcg.go is a Go transliteration of
   `stanfordASL/stlcg`. A change that makes the Go library disagree
   with the Python one (without a deliberate, documented reason) will
   not land. If you find a semantic divergence, open an issue first.
2. **Tests are not optional.** Every semantic change needs either a
   new test or an extended existing one. See the "Testing" section.
3. **No silent API breaks.** If a change removes or renames a public
   symbol, flag it in `CHANGELOG.md` under the `### Breaking` section
   of the next release.
4. **Math is explained.** Anywhere you lower STL semantics into gomlx
   graph ops, the code comment must spell out the math: the invariant,
   the sentinel, the clipping rule. These are the bugs that cost hours
   to unwind.

## Development workflow

### Setup

```bash
git clone https://github.com/jpfielding/stlcg
cd stlcg
go mod download
```

You need Go **1.25 or later** (see `go.mod`). CI runs both 1.25.x and 1.26.x.

### Running the suite

```bash
go vet ./...
go test -race -count=1 ./...
staticcheck ./...          # optional locally, required in CI
```

The race detector must pass. CI enforces `-race`.

### Benchmarks

```bash
go test -bench=. -benchmem -run=^$ ./...
```

If a change touches `compile.go` or `minmax/`, include before/after
numbers in the PR.

## Testing conventions

Every operator in the library has three independent verifiers:

1. **Reference evaluator** (`reference_test.go`) — pure-Go, loop-based,
   forward-time robustness over `map[string][]float64`. Always the
   source of truth.
2. **Hand-authored parity** (`parity_test.go:TestHandAuthoredParity`)
   — values computed on paper. Independent of both the compiler and
   the reference evaluator. Catches bugs where compiler + reference
   agree on a wrong answer.
3. **Python stlcg parity** (`parity_test.go:TestPythonParityFixtures`)
   — auto-skips when `testdata/fixtures/` is absent. Generate fixtures
   with `testdata/generate_fixtures.py` against a pinned upstream
   commit.

New operators or semantic changes must extend all three layers.

Property tests (`property_test.go`) exercise algebraic laws
(De Morgan, time-shift invariance, Always/Eventually duality). When
adding an operator with a natural algebraic identity, add a property
test too.

Gradient tests (`gradient_test.go`) use centered finite differences
against `graph.Gradient`. Smooth mode only — exact mode is
non-differentiable at ties.

## Code style

- `gofmt -s` is enforced.
- Exported identifiers need godoc comments.
- Comments explain _why_ and _what math_, not _what the code does_.
- Error returns carry `errors.Is`-compatible sentinels from
  `errors.go`. Add a new one there if your code introduces a new
  failure category.
- Panics are reserved for programmer invariants (unknown AST type,
  arity mismatch). User-input failures return errors.

## Commit messages

Imperative first line, ≤ 72 chars. Body explains _why_ and any
non-obvious consequences. Example:

```
until/then: implement overlap=false semantics

The overlap flag on Until/Then was accepted by the constructors but
silently ignored in the compiler. Python stlcg honors it; the lie
was moved from the compiler up to the constructor.
...
```

## Filing issues

- Bugs: minimal reproducer (formula, trace, expected vs. observed
  output) or a failing test.
- Feature requests: cite the Python stlcg counterpart or the STL
  paper you want to implement.

## License

By contributing you agree that your changes are licensed under the
MIT License (see `LICENSE`).
