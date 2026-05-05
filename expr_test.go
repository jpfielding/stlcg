package stlcg

import (
	"reflect"
	"testing"
)

func TestVarConstBasics(t *testing.T) {
	x := Var("x")
	if got := x.Vars(); !reflect.DeepEqual(got, []string{"x"}) {
		t.Fatalf("Var(\"x\").Vars() = %v, want [x]", got)
	}
	if x.String() != "x" {
		t.Fatalf("Var(\"x\").String() = %q, want %q", x.String(), "x")
	}

	c := Const(3.5)
	if c.Vars() != nil {
		t.Fatalf("Const.Vars() = %v, want nil", c.Vars())
	}
	if c.String() != "3.5" {
		t.Fatalf("Const(3.5).String() = %q, want %q", c.String(), "3.5")
	}
}

func TestVarEmptyPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Var(\"\") did not panic")
		}
	}()
	_ = Var("")
}

func TestPredicateStringAndVars(t *testing.T) {
	x, y := Var("x"), Var("y")
	cases := []struct {
		f        Formula
		wantStr  string
		wantVars []string
	}{
		{Lt(x, Const(5)), "(x < 5)", []string{"x"}},
		{Le(x, y), "(x <= y)", []string{"x", "y"}},
		{Gt(x, Const(5)), "(x > 5)", []string{"x"}},
		{Ge(y, Const(0)), "(y >= 0)", []string{"y"}},
		{Eq(x, y), "(x == y)", []string{"x", "y"}},
		{Identity(x), "Identity(x)", []string{"x"}},
	}
	for _, c := range cases {
		if got := c.f.String(); got != c.wantStr {
			t.Errorf("String() = %q, want %q", got, c.wantStr)
		}
		if got := c.f.Vars(); !reflect.DeepEqual(got, c.wantVars) {
			t.Errorf("Vars() = %v, want %v", got, c.wantVars)
		}
	}
}

func TestLogicalCombinators(t *testing.T) {
	x, y, z := Var("x"), Var("y"), Var("z")

	phi := And(Gt(x, Const(5)), Not(Lt(y, Const(2))))
	if got, want := phi.Vars(), []string{"x", "y"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Vars() = %v, want %v", got, want)
	}
	if got, want := phi.String(), "((x > 5) ∧ ¬(y < 2))"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}

	psi := Or(phi, Gt(z, Const(0)))
	if got, want := psi.Vars(), []string{"x", "y", "z"}; !reflect.DeepEqual(got, want) {
		t.Errorf("nested Vars() = %v, want %v", got, want)
	}

	imp := Implies(Gt(x, Const(0)), Lt(y, Const(10)))
	// Implies lowers to Or(Not(...), ...).
	if got, want := imp.String(), "(¬(x > 0) ∨ (y < 10))"; got != want {
		t.Errorf("Implies String() = %q, want %q", got, want)
	}
}

func TestDoubleNegationStructurePreserved(t *testing.T) {
	// Not(Not(phi)) is structurally distinct from phi in the AST;
	// semantic collapse happens at the compiler/reference level.
	phi := Gt(Var("x"), Const(0))
	nn := Not(Not(phi))
	if got, want := nn.String(), "¬¬(x > 0)"; got != want {
		t.Errorf("Not(Not()) String() = %q, want %q", got, want)
	}
	if got, want := nn.Vars(), []string{"x"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Vars() = %v, want %v", got, want)
	}
}

func TestVarsDedupAndSort(t *testing.T) {
	x := Var("x")
	f := And(And(Gt(x, Const(0)), Lt(x, Const(1))), Gt(Var("a"), Var("b")))
	if got, want := f.Vars(), []string{"a", "b", "x"}; !reflect.DeepEqual(got, want) {
		t.Errorf("Vars() = %v, want %v", got, want)
	}
}
