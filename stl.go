package stlcg

import (
	"math"
	"sort"
)

// Formula is an immutable STL abstract-syntax-tree node.
//
// The type is sealed: only this package can declare implementations.
// External users construct formulas via the exported constructor
// functions (Var, Gt, Always, etc.) and consume them as values.
type Formula interface {
	// Vars returns the free variables referenced by this subtree, in sorted
	// order with duplicates removed.
	Vars() []string

	// String returns a readable representation of the formula. The exact
	// format is not part of the API contract.
	String() string

	// sealed is the unexported marker that keeps Formula a closed sum.
	sealed()
}

// Expr is an immutable arithmetic expression referenced by predicates.
// Sealed in the same sense as Formula.
type Expr interface {
	Vars() []string
	String() string
	sealedExpr()
}

// Interval represents a time interval [Lo, Hi] (inclusive both ends).
// Hi may be Unbounded to denote [Lo, +∞).
type Interval struct {
	Lo, Hi int
}

// Unbounded is the Hi value used to denote an unbounded upper bound.
const Unbounded = math.MaxInt

// Bounds constructs a bounded interval [lo, hi]. Panics if lo < 0 or
// hi < lo (and hi != Unbounded).
func Bounds(lo, hi int) Interval {
	if lo < 0 {
		panic("stlcg: interval lower bound must be >= 0")
	}
	if hi != Unbounded && hi < lo {
		panic("stlcg: interval upper bound must be >= lower bound")
	}
	return Interval{Lo: lo, Hi: hi}
}

// AllTime returns the unbounded interval [0, +∞).
func AllTime() Interval { return Interval{Lo: 0, Hi: Unbounded} }

// From returns the interval [lo, +∞).
func From(lo int) Interval { return Bounds(lo, Unbounded) }

// IsUnbounded reports whether the interval's upper bound is infinite.
func (iv Interval) IsUnbounded() bool { return iv.Hi == Unbounded }

// sortedUnique returns a sorted deduplicated copy of ss.
func sortedUnique(ss []string) []string {
	if len(ss) == 0 {
		return nil
	}
	cp := append([]string(nil), ss...)
	sort.Strings(cp)
	out := cp[:0]
	for i, s := range cp {
		if i == 0 || s != cp[i-1] {
			out = append(out, s)
		}
	}
	return out
}

// mergeVars merges Vars() results from multiple nodes into a sorted,
// deduplicated slice.
func mergeVars(sets ...[]string) []string {
	var all []string
	for _, s := range sets {
		all = append(all, s...)
	}
	return sortedUnique(all)
}
