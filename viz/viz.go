// Package viz emits Graphviz DOT for stlcg formulas.
//
// The DOT is a hand-rolled string write — no CGo and no external DOT
// library. Render to PNG/SVG/PDF with the `dot` command:
//
//	WriteDOT(os.Stdout, phi) | dot -Tpng -o phi.png
//
// Node kinds are color-coded:
//
//	Variable         lightblue
//	Constant         wheat
//	Predicate / Eq   palegreen
//	Logical (∧ ∨ ¬)  orange
//	Temporal (□ ◇ U T) lightcoral
//	Integral         plum
package viz

import (
	"fmt"
	"io"
	"strings"

	"github.com/jpfielding/stlcg"
)

// WriteDOT renders phi as a Graphviz DOT graph to w. Returns the first
// write error encountered.
func WriteDOT(w io.Writer, phi stlcg.Formula) error {
	nodes := stlcg.Walk(phi)

	var b strings.Builder
	b.WriteString("digraph stlcg {\n")
	b.WriteString("  rankdir=TB;\n")
	b.WriteString("  node [shape=box, style=filled, fontname=\"Helvetica\"];\n")

	for _, n := range nodes {
		fmt.Fprintf(&b, "  n%d [label=%q, fillcolor=%q];\n",
			n.ID, n.Label, colorFor(n.Kind))
	}
	for _, n := range nodes {
		for _, c := range n.Children {
			fmt.Fprintf(&b, "  n%d -> n%d;\n", n.ID, c)
		}
	}

	b.WriteString("}\n")
	_, err := io.WriteString(w, b.String())
	return err
}

// ToDOT returns the DOT representation as a string.
func ToDOT(phi stlcg.Formula) string {
	var b strings.Builder
	_ = WriteDOT(&b, phi)
	return b.String()
}

func colorFor(k stlcg.NodeKind) string {
	switch k {
	case stlcg.KindVar:
		return "lightblue"
	case stlcg.KindConst:
		return "wheat"
	case stlcg.KindPredicate, stlcg.KindIdentity:
		return "palegreen"
	case stlcg.KindNot, stlcg.KindAnd, stlcg.KindOr:
		return "orange"
	case stlcg.KindAlways, stlcg.KindEventually, stlcg.KindUntil, stlcg.KindThen:
		return "lightcoral"
	case stlcg.KindIntegral:
		return "plum"
	}
	return "white"
}
