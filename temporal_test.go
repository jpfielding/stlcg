package stlcg

import (
	"reflect"
	"testing"
)

func TestIntervalConstructors(t *testing.T) {
	iv := Bounds(0, 50)
	if iv.Lo != 0 || iv.Hi != 50 || iv.IsUnbounded() {
		t.Errorf("Bounds(0,50) = %+v", iv)
	}
	if From(3).Lo != 3 || !From(3).IsUnbounded() {
		t.Errorf("From(3) = %+v", From(3))
	}
	if at := AllTime(); at.Lo != 0 || !at.IsUnbounded() {
		t.Errorf("AllTime = %+v", at)
	}
}

func TestBoundsPanics(t *testing.T) {
	cases := []struct {
		name   string
		lo, hi int
	}{
		{"negative lo", -1, 5},
		{"hi < lo", 5, 3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Bounds(%d,%d) did not panic", c.lo, c.hi)
				}
			}()
			_ = Bounds(c.lo, c.hi)
		})
	}
}

func TestAlwaysEventuallyString(t *testing.T) {
	x := Var("x")
	cases := []struct {
		f    Formula
		want string
	}{
		{Always(Gt(x, Const(0)), Bounds(0, 50)), "□[0,50] (x > 0)"},
		{Always(Gt(x, Const(0)), AllTime()), "□ (x > 0)"},
		{Always(Gt(x, Const(0)), From(3)), "□[3,∞) (x > 0)"},
		{Eventually(Lt(x, Const(1)), Bounds(2, 5)), "◇[2,5] (x < 1)"},
	}
	for _, c := range cases {
		if got := c.f.String(); got != c.want {
			t.Errorf("String() = %q, want %q", got, c.want)
		}
	}
}

func TestUntilThenVars(t *testing.T) {
	x, y := Var("x"), Var("y")
	u := Until(Gt(x, Const(0)), Lt(y, Const(1)), Bounds(0, 10), true)
	if got, want := u.Vars(), []string{"x", "y"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Until Vars = %v, want %v", got, want)
	}
	if got, want := u.String(), "((x > 0) U[0,10] (y < 1))"; got != want {
		t.Errorf("Until String = %q, want %q", got, want)
	}

	th := Then(Gt(x, Const(0)), Lt(y, Const(1)), AllTime(), false)
	if got, want := th.String(), "((x > 0) T (y < 1))"; got != want {
		t.Errorf("Then String = %q, want %q", got, want)
	}
}

func TestIntegral1dPanicsOnUnbounded(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Integral1d on unbounded interval did not panic")
		}
	}()
	_ = Integral1d(Identity(Var("x")), AllTime(), Riemann)
}

func TestIntegral1dString(t *testing.T) {
	f := Integral1d(Identity(Var("x")), Bounds(0, 5), Trapezoid)
	if got, want := f.String(), "∫[0,5] Identity(x)"; got != want {
		t.Errorf("Integral1d String = %q, want %q", got, want)
	}
}

func TestNestedFormula(t *testing.T) {
	x, y := Var("x"), Var("y")
	phi := Always(
		And(Gt(x, Const(5)), Not(Lt(y, Const(2)))),
		Bounds(0, 50),
	)
	if got, want := phi.Vars(), []string{"x", "y"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Vars = %v, want %v", got, want)
	}
	want := "□[0,50] ((x > 5) ∧ ¬(y < 2))"
	if got := phi.String(); got != want {
		t.Errorf("String = %q, want %q", got, want)
	}
}
