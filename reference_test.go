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
	case *alwaysFormula:
		return r.slidingReduce(r.rho(n.sub, signals), n.interval, false)
	case *eventuallyFormula:
		return r.slidingReduce(r.rho(n.sub, signals), n.interval, true)
	case *untilFormula:
		return r.untilOrThen(r.rho(n.left, signals), r.rho(n.right, signals), n.interval, false)
	case *thenFormula:
		return r.untilOrThen(r.rho(n.left, signals), r.rho(n.right, signals), n.interval, true)
	case *integralFormula:
		return r.integral(r.rho(n.sub, signals), n.interval, n.scheme)
	}
	panic(fmt.Sprintf("ref: unsupported formula %T", f))
}

// untilOrThen implements the reference semantics that mirrors the
// compiler: at each t, for each offset k ∈ [0, L-1] build inner_k =
// min(prefix_{t..t+a+k} phi, psi[t+a+k]) with past-end slots replaced by
// the psi-side sentinel (-∞), and take the max (or smooth max) across k.
// phiPrefixMax=true yields Then semantics; false yields Until.
func (r *referenceEvaluator) untilOrThen(phi, psi []float64, iv Interval, phiPrefixMax bool) []float64 {
	T := len(phi)
	a := iv.Lo
	var b int
	if iv.IsUnbounded() {
		b = T - 1
	} else {
		b = iv.Hi
	}
	if b > T-1 {
		b = T - 1
	}
	L := b - a + 1
	if L < 1 {
		panic(fmt.Sprintf("ref: Until/Then empty interval lo=%d hi=%d T=%d", iv.Lo, iv.Hi, T))
	}

	out := make([]float64, T)
	for t := 0; t < T; t++ {
		inner := make([]float64, L)
		for k := 0; k < L; k++ {
			s := t + a + k
			if s >= T {
				inner[k] = math.Inf(-1)
				continue
			}
			// phi prefix over [t, s] inclusive.
			win := make([]float64, s-t+1)
			copy(win, phi[t:s+1])
			phiPfx := r.reduceWindow(win, phiPrefixMax)

			// Pairwise min with psi[s].
			inner[k] = r.reducePairScalar(phiPfx, psi[s], false)
		}
		out[t] = r.reduceWindow(inner, true) // outer max
	}
	return out
}

// reducePairScalar: scalar-valued pair reduce matching Evaluator semantics.
func (r *referenceEvaluator) reducePairScalar(x, y float64, wantMax bool) float64 {
	switch r.mode {
	case ModeExact:
		if wantMax {
			return math.Max(x, y)
		}
		return math.Min(x, y)
	case ModeSmooth:
		return smoothExtremum2(x, y, r.scale, wantMax)
	}
	panic("ref: unknown mode")
}

// integral is the reference implementation of Integral1d. Riemann:
// sum over s ∈ [t+a, t+b] of phi[s]; past-end slots contribute 0.
// Trapezoid: Riemann minus half the endpoints at t+a and t+b (or the
// clipped endpoint) — the compiler does not clip the upper endpoint
// for Trapezoid, so this matches.
func (r *referenceEvaluator) integral(phi []float64, iv Interval, scheme IntegrationScheme) []float64 {
	T := len(phi)
	a := iv.Lo
	b := iv.Hi
	if b > T-1 {
		b = T - 1
	}
	L := b - a + 1
	if L < 1 {
		panic(fmt.Sprintf("ref: Integral1d empty window lo=%d hi=%d T=%d", iv.Lo, iv.Hi, T))
	}

	out := make([]float64, T)
	for t := 0; t < T; t++ {
		var sum float64
		for k := 0; k < L; k++ {
			idx := t + a + k
			if idx < T {
				sum += phi[idx]
			}
		}
		if scheme == Trapezoid {
			var endLow, endHigh float64
			if t+a < T {
				endLow = phi[t+a]
			}
			if t+b < T {
				endHigh = phi[t+b]
			}
			sum -= 0.5 * (endLow + endHigh)
		}
		out[t] = sum
	}
	return out
}

// slidingReduce is the reference-evaluator counterpart of
// compiler.slidingReduce. At time t, the window is
// [t+a, min(t+b, T-1)]. If empty (t+a >= T) the sentinel value
// (+∞ for min, -∞ for max) is used — matching the compiler.
func (r *referenceEvaluator) slidingReduce(sub []float64, iv Interval, wantMax bool) []float64 {
	T := len(sub)
	a := iv.Lo
	var b int
	if iv.IsUnbounded() {
		b = T - 1
	} else {
		b = iv.Hi
	}
	sentinel := math.Inf(+1)
	if wantMax {
		sentinel = math.Inf(-1)
	}

	out := make([]float64, T)
	for t := 0; t < T; t++ {
		lo := t + a
		hi := t + b
		if hi > T-1 {
			hi = T - 1
		}
		// Collect window values with sentinel padding so the window length
		// is constant across t (matches compiler's padded reshape-and-reduce).
		L := b - a + 1
		win := make([]float64, L)
		for k := 0; k < L; k++ {
			idx := lo + k
			if idx > T-1 {
				win[k] = sentinel
			} else {
				win[k] = sub[idx]
			}
		}
		out[t] = r.reduceWindow(win, wantMax)
	}
	return out
}

// reduceWindow reduces a fixed-length window according to mode/wantMax.
func (r *referenceEvaluator) reduceWindow(win []float64, wantMax bool) float64 {
	switch r.mode {
	case ModeExact:
		v := win[0]
		for _, x := range win[1:] {
			if wantMax {
				if x > v {
					v = x
				}
			} else {
				if x < v {
					v = x
				}
			}
		}
		return v
	case ModeSmooth:
		return smoothExtremumN(win, r.scale, wantMax)
	}
	panic("ref: unknown mode")
}

// smoothExtremumN is the n-way scalar smooth extremum.
func smoothExtremumN(xs []float64, tau float64, wantMax bool) float64 {
	signed := make([]float64, len(xs))
	for i, x := range xs {
		if wantMax {
			signed[i] = tau * x
		} else {
			signed[i] = -tau * x
		}
	}
	m := math.Inf(-1)
	for _, v := range signed {
		if v > m {
			m = v
		}
	}
	// Handle all-(-inf) case: result is -inf (i.e. sentinel dominated).
	if math.IsInf(m, -1) {
		if wantMax {
			return math.Inf(-1)
		}
		return math.Inf(+1)
	}
	var sum float64
	for _, v := range signed {
		sum += math.Exp(v - m)
	}
	lse := m + math.Log(sum)
	out := lse / tau
	if !wantMax {
		out = -out
	}
	return out
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
