package viz_test

import (
	"strings"
	"testing"

	"github.com/jpfielding/stlcg"
	"github.com/jpfielding/stlcg/viz"
)

func TestToDOTBasics(t *testing.T) {
	x := stlcg.Var("x")
	phi := stlcg.Always(stlcg.Gt(x, stlcg.Const(5.0)), stlcg.Bounds(0, 10))
	dot := viz.ToDOT(phi)

	for _, want := range []string{
		"digraph stlcg {",
		"rankdir=TB;",
		`label="□[0,10]"`,
		"lightcoral", // Always
		"palegreen",  // predicate
		"lightblue",  // x
		"wheat",      // 5
		"->",         // at least one edge
	} {
		if !strings.Contains(dot, want) {
			t.Errorf("DOT missing %q\nfull:\n%s", want, dot)
		}
	}
}

func TestToDOTNested(t *testing.T) {
	x, y := stlcg.Var("x"), stlcg.Var("y")
	phi := stlcg.Until(
		stlcg.And(stlcg.Gt(x, stlcg.Const(0.0)), stlcg.Lt(y, stlcg.Const(1.0))),
		stlcg.Eventually(stlcg.Gt(x, stlcg.Const(2.0)), stlcg.Bounds(0, 3)),
		stlcg.Bounds(0, 5),
		true,
	)
	dot := viz.ToDOT(phi)

	// All operator labels should appear.
	for _, want := range []string{
		`label="U[0,5]"`,
		`label="◇[0,3]"`,
		`label="∧"`,
		`label="x"`,
		`label="y"`,
		`label="0"`,
		`label="1"`,
		`label="2"`,
	} {
		if !strings.Contains(dot, want) {
			t.Errorf("nested DOT missing %q", want)
		}
	}
}

func TestToDOTIntegral(t *testing.T) {
	phi := stlcg.Integral1d(stlcg.Identity(stlcg.Var("x")), stlcg.Bounds(0, 4), stlcg.Trapezoid)
	dot := viz.ToDOT(phi)

	if !strings.Contains(dot, "plum") {
		t.Errorf("Integral node should be plum, got:\n%s", dot)
	}
	if !strings.Contains(dot, "Trapezoid") {
		t.Errorf("Integral scheme label missing, got:\n%s", dot)
	}
}

func TestDOTEdgeCount(t *testing.T) {
	// Simple sanity: n nodes => n-1 edges in a tree-structured AST.
	x := stlcg.Var("x")
	phi := stlcg.And(stlcg.Gt(x, stlcg.Const(0.0)), stlcg.Lt(x, stlcg.Const(5.0)))
	// Tree: And -> (Gt -> x,0) (Lt -> x,5) = 1 + 2 + 2 + 2 = 7 nodes, 6 edges.
	// (Vars are created fresh per predicate; not shared.)

	dot := viz.ToDOT(phi)
	edges := strings.Count(dot, "->")
	if edges != 6 {
		t.Errorf("expected 6 edges, got %d\n%s", edges, dot)
	}
}
