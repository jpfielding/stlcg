package stlcg

import "fmt"

// NodeKind classifies a visited AST node for visualization and other
// external consumers. It is intentionally coarse: Lt/Le/Gt/Ge/Eq all map
// to KindPredicate since the distinction is already in the label.
type NodeKind int

const (
	KindUnknown NodeKind = iota
	KindVar
	KindConst
	KindPredicate
	KindIdentity
	KindNot
	KindAnd
	KindOr
	KindAlways
	KindEventually
	KindUntil
	KindThen
	KindIntegral
)

// WalkNode is one entry in the flat tree returned by Walk. Children lists
// indices into the same slice, in visit order.
type WalkNode struct {
	ID       int
	Kind     NodeKind
	Label    string
	Children []int
}

// Walk returns the AST rooted at f as a flat slice. The root is at index
// 0; nodes are listed in depth-first post-order of assignment, with
// Children holding indices of direct descendants.
//
// This enables external packages (e.g. viz) to render or analyze the
// formula tree without depending on the unexported concrete types.
func Walk(f Formula) []WalkNode {
	w := &walker{}
	w.visitFormula(f)
	return w.nodes
}

type walker struct {
	nodes []WalkNode
}

func (w *walker) alloc(kind NodeKind, label string) int {
	id := len(w.nodes)
	w.nodes = append(w.nodes, WalkNode{ID: id, Kind: kind, Label: label})
	return id
}

func (w *walker) setChildren(id int, children []int) {
	w.nodes[id].Children = children
}

func (w *walker) visitFormula(f Formula) int {
	switch n := f.(type) {
	case *predicate:
		id := w.alloc(KindPredicate, predicateLabel(n))
		lhs := w.visitExpr(n.lhs)
		rhs := w.visitExpr(n.rhs)
		w.setChildren(id, []int{lhs, rhs})
		return id
	case *identityFormula:
		id := w.alloc(KindIdentity, "Identity")
		c := w.visitExpr(n.signal)
		w.setChildren(id, []int{c})
		return id
	case *notFormula:
		id := w.alloc(KindNot, "¬")
		c := w.visitFormula(n.sub)
		w.setChildren(id, []int{c})
		return id
	case *andFormula:
		id := w.alloc(KindAnd, "∧")
		l := w.visitFormula(n.left)
		r := w.visitFormula(n.right)
		w.setChildren(id, []int{l, r})
		return id
	case *orFormula:
		id := w.alloc(KindOr, "∨")
		l := w.visitFormula(n.left)
		r := w.visitFormula(n.right)
		w.setChildren(id, []int{l, r})
		return id
	case *alwaysFormula:
		id := w.alloc(KindAlways, fmt.Sprintf("□%s", intervalString(n.interval)))
		c := w.visitFormula(n.sub)
		w.setChildren(id, []int{c})
		return id
	case *eventuallyFormula:
		id := w.alloc(KindEventually, fmt.Sprintf("◇%s", intervalString(n.interval)))
		c := w.visitFormula(n.sub)
		w.setChildren(id, []int{c})
		return id
	case *untilFormula:
		id := w.alloc(KindUntil, fmt.Sprintf("U%s", intervalString(n.interval)))
		l := w.visitFormula(n.left)
		r := w.visitFormula(n.right)
		w.setChildren(id, []int{l, r})
		return id
	case *thenFormula:
		id := w.alloc(KindThen, fmt.Sprintf("T%s", intervalString(n.interval)))
		l := w.visitFormula(n.left)
		r := w.visitFormula(n.right)
		w.setChildren(id, []int{l, r})
		return id
	case *integralFormula:
		schemeLabel := "Riemann"
		if n.scheme == Trapezoid {
			schemeLabel = "Trapezoid"
		}
		id := w.alloc(KindIntegral, fmt.Sprintf("∫%s %s", intervalString(n.interval), schemeLabel))
		c := w.visitFormula(n.sub)
		w.setChildren(id, []int{c})
		return id
	}
	panic(fmt.Sprintf("stlcg: Walk encountered unknown formula %T", f))
}

func (w *walker) visitExpr(e Expr) int {
	switch v := e.(type) {
	case *variable:
		return w.alloc(KindVar, v.name)
	case *constant:
		return w.alloc(KindConst, v.String())
	}
	panic(fmt.Sprintf("stlcg: Walk encountered unknown expr %T", e))
}

func predicateLabel(p *predicate) string {
	return p.op.symbol()
}
