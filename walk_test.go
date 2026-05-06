package stlcg

import "testing"

// TestWalkKinds constructs one formula of each node kind and asserts that
// Walk produces the expected top-level NodeKind. Closes 0% in-package
// coverage of walk.go. The viz tests exercise it transitively but would
// miss a kind regression if viz also changed.
func TestWalkKinds(t *testing.T) {
	x := Var("x")
	y := Var("y")
	pred := Gt(x, Const(0.5))

	cases := []struct {
		name string
		phi  Formula
		want NodeKind
	}{
		{"predicate", pred, KindPredicate},
		{"identity", Identity(x), KindIdentity},
		{"not", Not(pred), KindNot},
		{"and", And(pred, Lt(y, Const(1.0))), KindAnd},
		{"or", Or(pred, Lt(y, Const(1.0))), KindOr},
		{"always", Always(pred, Bounds(0, 3)), KindAlways},
		{"eventually", Eventually(pred, Bounds(0, 3)), KindEventually},
		{"until", Until(pred, Gt(y, Const(0.0)), Bounds(0, 3), true), KindUntil},
		{"then", Then(pred, Gt(y, Const(0.0)), Bounds(0, 3), true), KindThen},
		{"integral", Integral1d(Identity(x), Bounds(0, 3), Riemann), KindIntegral},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			nodes := Walk(c.phi)
			if len(nodes) == 0 {
				t.Fatalf("Walk returned 0 nodes for %q", c.name)
			}
			if nodes[0].Kind != c.want {
				t.Errorf("root kind = %v, want %v (label=%q)", nodes[0].Kind, c.want, nodes[0].Label)
			}
			if nodes[0].ID != 0 {
				t.Errorf("root ID = %d, want 0", nodes[0].ID)
			}
			// Children indices must all be valid.
			for _, ch := range nodes[0].Children {
				if ch < 0 || ch >= len(nodes) {
					t.Errorf("invalid child index %d (len=%d)", ch, len(nodes))
				}
			}
		})
	}
}

// TestWalkNestedStructure checks that a small nested formula yields the
// expected flat structure and that every node (including leaves) is
// reachable from the root.
func TestWalkNestedStructure(t *testing.T) {
	x := Var("x")
	phi := Always(
		And(
			Gt(x, Const(0.0)),
			Lt(x, Const(1.0)),
		),
		Bounds(0, 5),
	)

	nodes := Walk(phi)
	if len(nodes) == 0 {
		t.Fatal("empty walk")
	}
	if nodes[0].Kind != KindAlways {
		t.Fatalf("root = %v, want KindAlways", nodes[0].Kind)
	}

	// Reachability: BFS from root should touch every node.
	reached := make([]bool, len(nodes))
	queue := []int{0}
	for len(queue) > 0 {
		i := queue[0]
		queue = queue[1:]
		if reached[i] {
			continue
		}
		reached[i] = true
		queue = append(queue, nodes[i].Children...)
	}
	for i, r := range reached {
		if !r {
			t.Errorf("node %d (kind=%v, label=%q) unreachable from root", i, nodes[i].Kind, nodes[i].Label)
		}
	}
}

// TestWalkVarAndConstLeaves exercises visitExpr directly by building a
// predicate with both a variable and a constant.
func TestWalkVarAndConstLeaves(t *testing.T) {
	phi := Gt(Var("speed"), Const(42.0))
	nodes := Walk(phi)

	var hasVar, hasConst bool
	var varLabel, constLabel string
	for _, n := range nodes {
		if n.Kind == KindVar {
			hasVar = true
			varLabel = n.Label
		}
		if n.Kind == KindConst {
			hasConst = true
			constLabel = n.Label
		}
	}
	if !hasVar {
		t.Error("no KindVar node in walk")
	}
	if !hasConst {
		t.Error("no KindConst node in walk")
	}
	if varLabel != "speed" {
		t.Errorf("var label = %q, want %q", varLabel, "speed")
	}
	if constLabel == "" {
		t.Errorf("const label is empty")
	}
}
