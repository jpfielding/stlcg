package stlcg

import (
	"fmt"
	"strconv"
)

// -------- Expr implementations --------

type variable struct{ name string }

func (v *variable) Vars() []string { return []string{v.name} }
func (v *variable) String() string { return v.name }
func (v *variable) sealedExpr()    {}

// Var returns a named free-variable expression. The same name is used to
// look up the tensor in a SignalMap at evaluation time.
func Var(name string) Expr {
	if name == "" {
		panic("stlcg: Var name must be non-empty")
	}
	return &variable{name: name}
}

// Name returns the variable name if e is a *variable, else "".
func varName(e Expr) string {
	if v, ok := e.(*variable); ok {
		return v.name
	}
	return ""
}

type constant struct{ value float64 }

func (c *constant) Vars() []string { return nil }
func (c *constant) String() string { return strconv.FormatFloat(c.value, 'g', -1, 64) }
func (c *constant) sealedExpr()    {}

// Const returns a scalar-constant expression.
func Const(v float64) Expr { return &constant{value: v} }

// -------- Predicate formulas (leaves) --------

type cmpOp int

const (
	opLt cmpOp = iota
	opLe
	opGt
	opGe
	opEq
)

func (o cmpOp) symbol() string {
	switch o {
	case opLt:
		return "<"
	case opLe:
		return "<="
	case opGt:
		return ">"
	case opGe:
		return ">="
	case opEq:
		return "=="
	}
	return "?"
}

type predicate struct {
	op       cmpOp
	lhs, rhs Expr
}

func (p *predicate) Vars() []string { return mergeVars(p.lhs.Vars(), p.rhs.Vars()) }
func (p *predicate) String() string {
	return fmt.Sprintf("(%s %s %s)", p.lhs.String(), p.op.symbol(), p.rhs.String())
}
func (*predicate) sealed() {}

// Lt is the predicate lhs < rhs. Robustness = rhs - lhs, scaled by PScale.
func Lt(lhs, rhs Expr) Formula { return &predicate{op: opLt, lhs: lhs, rhs: rhs} }

// Le is the predicate lhs <= rhs. (Same robustness semantics as Lt; the
// distinction is degenerate over reals and kept for readability.)
func Le(lhs, rhs Expr) Formula { return &predicate{op: opLe, lhs: lhs, rhs: rhs} }

// Gt is the predicate lhs > rhs. Robustness = lhs - rhs, scaled by PScale.
func Gt(lhs, rhs Expr) Formula { return &predicate{op: opGt, lhs: lhs, rhs: rhs} }

// Ge is the predicate lhs >= rhs.
func Ge(lhs, rhs Expr) Formula { return &predicate{op: opGe, lhs: lhs, rhs: rhs} }

// Eq is the predicate lhs == rhs. Robustness = -|lhs - rhs|.
func Eq(lhs, rhs Expr) Formula { return &predicate{op: opEq, lhs: lhs, rhs: rhs} }

// Identity is a pass-through predicate whose robustness is the raw signal
// value. Equivalent to GreaterThan(x, 0) with PScale = 1 if the user wants
// x > 0 semantics, but Identity preserves the sign of x directly.
type identityFormula struct{ signal Expr }

func (i *identityFormula) Vars() []string { return i.signal.Vars() }
func (i *identityFormula) String() string { return fmt.Sprintf("Identity(%s)", i.signal.String()) }
func (*identityFormula) sealed()          {}

func Identity(signal Expr) Formula { return &identityFormula{signal: signal} }

// -------- Logical operators --------

type notFormula struct{ sub Formula }

func (n *notFormula) Vars() []string { return n.sub.Vars() }
func (n *notFormula) String() string { return fmt.Sprintf("¬%s", n.sub.String()) }
func (*notFormula) sealed()          {}

// Not is logical negation. Robustness = -Robustness(sub).
func Not(sub Formula) Formula { return &notFormula{sub: sub} }

type andFormula struct{ left, right Formula }

func (a *andFormula) Vars() []string { return mergeVars(a.left.Vars(), a.right.Vars()) }
func (a *andFormula) String() string { return fmt.Sprintf("(%s ∧ %s)", a.left, a.right) }
func (*andFormula) sealed()          {}

// And is logical conjunction. Robustness = Minish(left, right).
func And(left, right Formula) Formula { return &andFormula{left: left, right: right} }

type orFormula struct{ left, right Formula }

func (o *orFormula) Vars() []string { return mergeVars(o.left.Vars(), o.right.Vars()) }
func (o *orFormula) String() string { return fmt.Sprintf("(%s ∨ %s)", o.left, o.right) }
func (*orFormula) sealed()          {}

// Or is logical disjunction. Robustness = Maxish(left, right).
func Or(left, right Formula) Formula { return &orFormula{left: left, right: right} }

// Implies returns (left → right), which equals (¬left ∨ right).
func Implies(left, right Formula) Formula { return Or(Not(left), right) }
