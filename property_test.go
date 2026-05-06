package stlcg

import (
	"math"
	"math/rand/v2"
	"testing"
	"testing/quick"
)

// Property tests over the pure-Go reference evaluator. These assert
// algebraic laws of STL robustness that must hold regardless of the
// gomlx backend — any failure here is a bug in the semantics.
//
// The reference evaluator is exercised because it's the oracle the
// compiler is validated against; equivalent laws over the compiler are
// left to the parity tests in compile_test.go and *_compile_test.go.

var propSeed = func() *rand.Rand {
	return rand.New(rand.NewPCG(0xBA5E, 0xC0DE))
}

func randTrace(rng *rand.Rand, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.Float64()*4 - 2 // [-2, +2]
	}
	return out
}

func approxEqual(a, b float64, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	if math.IsInf(a, 0) && math.IsInf(b, 0) {
		return math.Signbit(a) == math.Signbit(b)
	}
	return math.Abs(a-b) <= tol
}

func sliceApproxEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approxEqual(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

// PropertyNotNotIsIdentity: ¬¬φ ≡ φ in both modes.
func TestPropertyNotNotIsIdentity(t *testing.T) {
	f := func(seed uint64) bool {
		rng := rand.New(rand.NewPCG(seed, 0xD15EA5E))
		signals := map[string][]float64{"x": randTrace(rng, 12)}
		phi := Gt(Var("x"), Const(0.0))
		nn := Not(Not(phi))
		for _, cfg := range []config{
			{mode: ModeExact, pscale: 1, scale: 0},
			{mode: ModeSmooth, pscale: 1, scale: 4},
		} {
			r := newRefEvaluator(cfg)
			if !sliceApproxEqual(r.rho(phi, signals), r.rho(nn, signals), 1e-9) {
				return false
			}
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

// PropertyDeMorganSmooth: And(a,b) ≡ Not(Or(Not a, Not b)) at matched τ.
// Holds exactly in smooth mode because Minish(a,b) = -Maxish(-a,-b) and
// (1/τ)logsumexp is sign-symmetric under negation.
func TestPropertyDeMorganSmooth(t *testing.T) {
	rng := propSeed()
	for i := 0; i < 25; i++ {
		signals := map[string][]float64{
			"x": randTrace(rng, 10),
			"y": randTrace(rng, 10),
		}
		a := Gt(Var("x"), Const(rng.Float64()*2-1))
		b := Lt(Var("y"), Const(rng.Float64()*2-1))

		lhs := And(a, b)
		rhs := Not(Or(Not(a), Not(b)))

		cfg := config{mode: ModeSmooth, pscale: 1, scale: 5}
		r := newRefEvaluator(cfg)
		if !sliceApproxEqual(r.rho(lhs, signals), r.rho(rhs, signals), 1e-9) {
			t.Fatalf("De Morgan violated on iter %d", i)
		}
	}
}

// PropertyAlwaysEventuallyDuality: Always_iv phi ≡ Not(Eventually_iv (Not phi))
// in exact mode on non-degenerate traces.
func TestPropertyAlwaysEventuallyDualityExact(t *testing.T) {
	rng := propSeed()
	cfg := config{mode: ModeExact, pscale: 1, scale: 0}
	for i := 0; i < 30; i++ {
		T := 8 + rng.IntN(8)
		signals := map[string][]float64{"x": randTrace(rng, T)}

		a := 0
		b := a + rng.IntN(T-a)
		iv := Bounds(a, b)
		phi := Gt(Var("x"), Const(rng.Float64()*2-1))

		lhs := Always(phi, iv)
		rhs := Not(Eventually(Not(phi), iv))

		r := newRefEvaluator(cfg)
		if !sliceApproxEqual(r.rho(lhs, signals), r.rho(rhs, signals), 1e-9) {
			t.Fatalf("Always/Eventually duality violated on iter %d (iv=[%d,%d] T=%d)", i, a, b, T)
		}
	}
}

// PropertyAlwaysSinglePointIsShift: Always[a,a] phi at t = phi at t+a,
// with sentinel +∞ past the trace end.
func TestPropertyAlwaysSinglePointIsShift(t *testing.T) {
	rng := propSeed()
	cfg := config{mode: ModeExact, pscale: 1, scale: 0}
	r := newRefEvaluator(cfg)

	for i := 0; i < 20; i++ {
		T := 8 + rng.IntN(6)
		signals := map[string][]float64{"x": randTrace(rng, T)}
		phi := Gt(Var("x"), Const(0.0))
		a := rng.IntN(T)

		got := r.rho(Always(phi, Bounds(a, a)), signals)
		base := r.rho(phi, signals)

		want := make([]float64, T)
		for ti := 0; ti < T; ti++ {
			if ti+a < T {
				want[ti] = base[ti+a]
			} else {
				want[ti] = math.Inf(+1)
			}
		}
		if !sliceApproxEqual(got, want, 1e-12) {
			t.Fatalf("Always[%d,%d] != shift(phi, %d): got %v want %v", a, a, a, got, want)
		}
	}
}

// PropertyEventuallyZeroIntervalIsIdentity: Eventually[0,0] phi ≡ phi.
func TestPropertyEventuallyZeroIntervalIsIdentity(t *testing.T) {
	rng := propSeed()
	r := newRefEvaluator(config{mode: ModeExact, pscale: 1, scale: 0})
	for i := 0; i < 10; i++ {
		signals := map[string][]float64{"x": randTrace(rng, 10)}
		phi := Lt(Var("x"), Const(rng.Float64()))
		got := r.rho(Eventually(phi, Bounds(0, 0)), signals)
		want := r.rho(phi, signals)
		if !sliceApproxEqual(got, want, 1e-12) {
			t.Fatalf("◇[0,0] phi != phi on iter %d", i)
		}
	}
}

// PropertyOrIsPointwiseMaxExact: Or(a,b) = max(a,b) pointwise in exact mode.
func TestPropertyOrIsPointwiseMaxExact(t *testing.T) {
	rng := propSeed()
	r := newRefEvaluator(config{mode: ModeExact, pscale: 1, scale: 0})
	for i := 0; i < 20; i++ {
		signals := map[string][]float64{
			"x": randTrace(rng, 10),
			"y": randTrace(rng, 10),
		}
		a := Gt(Var("x"), Const(0))
		b := Gt(Var("y"), Const(0))
		got := r.rho(Or(a, b), signals)

		ra := r.rho(a, signals)
		rb := r.rho(b, signals)
		want := make([]float64, len(ra))
		for j := range want {
			want[j] = math.Max(ra[j], rb[j])
		}
		if !sliceApproxEqual(got, want, 1e-12) {
			t.Fatalf("Or != pointwise max on iter %d", i)
		}
	}
}

// PropertySmoothConvergesExact: at large τ, smooth reductions approach
// exact values (up to a small bias).
func TestPropertySmoothConvergesExact(t *testing.T) {
	rng := propSeed()
	for i := 0; i < 10; i++ {
		// Choose traces with non-tied values so the exact min/max is
		// well-defined and smooth can approach it cleanly.
		x := randTrace(rng, 12)
		signals := map[string][]float64{"x": x}
		phi := Always(Gt(Var("x"), Const(-3)), Bounds(0, 3))

		exact := newRefEvaluator(config{mode: ModeExact, pscale: 1, scale: 0}).rho(phi, signals)
		smooth := newRefEvaluator(config{mode: ModeSmooth, pscale: 1, scale: 40}).rho(phi, signals)

		if !sliceApproxEqual(exact, smooth, 0.2) {
			t.Fatalf("smooth (τ=40) did not approach exact on iter %d: exact=%v smooth=%v", i, exact, smooth)
		}
	}
}
