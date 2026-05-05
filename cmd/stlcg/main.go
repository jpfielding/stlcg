// Command stlcg is a placeholder demo for the stlcg.go library.
// Phase H will replace this with a CSV-trace reader that evaluates a
// configurable formula and optionally emits DOT.
package main

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	be := backends.MustNew()
	fmt.Printf("stlcg.go: gomlx backend = %s\n", be.Name())
}
