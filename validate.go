package stlcg

import "fmt"

// validateIntervalsForT walks the formula AST and returns an
// ErrBadShape-wrapped error if any bounded temporal interval is
// incompatible with the runtime trace length T.
//
// Specifically, if an interval has a lower bound >= T, the compile-time
// window would be empty or negative. Unbounded intervals are always
// valid because they clip to [lo, T-1] at compile time. Upper bound
// b >= T is benign — the compiler clips to T-1 — but lo >= T cannot
// be recovered.
//
// Returning an error here rather than panicking inside the graph-build
// path is necessary because gomlx's lazy compile runs inside an internal
// goroutine; a panic there leaves the Exec in a state that deadlocks on
// Finalize. See RobustnessTraceE.
func validateIntervalsForT(f Formula, T int) error {
	switch n := f.(type) {
	case *predicate, *identityFormula:
		return nil
	case *notFormula:
		return validateIntervalsForT(n.sub, T)
	case *andFormula:
		if err := validateIntervalsForT(n.left, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.right, T)
	case *orFormula:
		if err := validateIntervalsForT(n.left, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.right, T)
	case *alwaysFormula:
		if err := checkLoAgainstT("Always", n.interval, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.sub, T)
	case *eventuallyFormula:
		if err := checkLoAgainstT("Eventually", n.interval, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.sub, T)
	case *untilFormula:
		if err := checkLoAgainstT("Until", n.interval, T); err != nil {
			return err
		}
		if err := validateIntervalsForT(n.left, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.right, T)
	case *thenFormula:
		if err := checkLoAgainstT("Then", n.interval, T); err != nil {
			return err
		}
		if err := validateIntervalsForT(n.left, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.right, T)
	case *integralFormula:
		if err := checkLoAgainstT("Integral1d", n.interval, T); err != nil {
			return err
		}
		return validateIntervalsForT(n.sub, T)
	}
	// Unknown formula types are programmer invariants; fall through to
	// the compiler's panic.
	return nil
}

func checkLoAgainstT(op string, iv Interval, T int) error {
	if iv.Lo >= T {
		return fmt.Errorf("%w: %s interval lower bound %d >= trace length %d",
			ErrBadShape, op, iv.Lo, T)
	}
	return nil
}
