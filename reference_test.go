package stlcg

import (
	"fmt"
	"math"
)

// referenceEvaluator is a naive, loop-based, forward-time STL robustness
// evaluator used as the source of truth in unit tests.
//
// Semantics:
//   - Signals are map[string][]float64 of length T (per batch element B is
//     handled externally by calling this once per batch).
//   - Time runs forward: index 0 is earliest, T-1 is latest.
//   - Smooth mode uses (1/tau)*LogSumExp(tau*x); Exact uses true min/max.
//   - AGM and advanced TiePolicy effects are not implemented (the compiler
//     doesn't exercise them yet either).
type referenceEvaluator struct {
	mode   Mode
	pscale float64
	scale  float64
}

func newRefEvaluator(cfg config) *referenceEvaluator {
	return &referenceEvaluator{
		mode:   cfg.mode,
		pscale: cfg.pscale,
		scale:  math.Abs(cfg.scale),
	}
}

func (r *referenceEvaluator) rho(f Formula, signals map[string][]float64) []float64 {
	T := traceLen(signals)
	switch n := f.(type) {
	case *predicate:
		return r.rhoPredicate(n, signals, T)
	case *identityFormula:
		out := make([]float64, T)
		vals := r.evalExpr(n.signal, signals, T)
		for t := 0; t < T; t++ {
			out[t] = vals[t] * r.pscale
		}
		return out
	case *notFormula:
		sub := r.rho(n.sub, signals)
		out := make([]float64, T)
		for t := 0; t < T; t++ {
			out[t] = -sub[t]
		}
		return out
	case *andFormula:
		l := r.rho(n.left, signals)
		rr := r.rho(n.right, signals)
		return r.reducePair(l, rr, false)
	case *orFormula:
		l := r.rho(n.left, signals)
		rr := r.rho(n.right, signals)
		return r.reducePair(l, rr, true)
	}
	panic(fmt.Sprintf("ref: unsupported formula %T", f))
}

func (r *referenceEvaluator) rhoPredicate(p *predicate, signals map[string][]float64, T int) []float64 {
	lhs := r.evalExpr(p.lhs, signals, T)
	rhs := r.evalExpr(p.rhs, signals, T)
	out := make([]float64, T)
	for t := 0; t < T; t++ {
		var v float64
		switch p.op {
		case opLt, opLe:
			v = rhs[t] - lhs[t]
		case opGt, opGe:
			v = lhs[t] - rhs[t]
		case opEq:
			v = -math.Abs(lhs[t] - rhs[t])
		}
		out[t] = v * r.pscale
	}
	return out
}

func (r *referenceEvaluator) evalExpr(e Expr, signals map[string][]float64, T int) []float64 {
	switch v := e.(type) {
	case *variable:
		s, ok := signals[v.name]
		if !ok {
			panic(fmt.Sprintf("ref: signal %q missing", v.name))
		}
		if len(s) != T {
			panic(fmt.Sprintf("ref: signal %q length %d != %d", v.name, len(s), T))
		}
		return s
	case *constant:
		out := make([]float64, T)
		for i := range out {
			out[i] = v.value
		}
		return out
	}
	panic(fmt.Sprintf("ref: unsupported expr %T", e))
}

func (r *referenceEvaluator) reducePair(a, b []float64, wantMax bool) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		switch r.mode {
		case ModeExact:
			if wantMax {
				out[i] = math.Max(a[i], b[i])
			} else {
				out[i] = math.Min(a[i], b[i])
			}
		case ModeSmooth:
			out[i] = smoothExtremum2(a[i], b[i], r.scale, wantMax)
		}
	}
	return out
}

// smoothExtremum2 is the scalar two-argument (1/tau)*logsumexp(tau*x)
// (with numerical-stability shift) used by the reference evaluator.
func smoothExtremum2(x, y, tau float64, wantMax bool) float64 {
	sx, sy := x, y
	if !wantMax {
		sx, sy = -x, -y
	}
	m := math.Max(sx*tau, sy*tau)
	lse := m + math.Log(math.Exp(sx*tau-m)+math.Exp(sy*tau-m))
	r := lse / tau
	if !wantMax {
		r = -r
	}
	return r
}

func traceLen(signals map[string][]float64) int {
	var T int
	for _, v := range signals {
		if T == 0 {
			T = len(v)
		} else if len(v) != T {
			panic("ref: signal length mismatch across variables")
		}
	}
	if T == 0 {
		panic("ref: empty signal map")
	}
	return T
}
