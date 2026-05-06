package stlcg

import "fmt"

// IntegrationScheme selects the numerical integration scheme for Integral1d.
type IntegrationScheme int

const (
	// Riemann is a left-endpoint Riemann sum (cumulative-sum difference).
	Riemann IntegrationScheme = iota
	// Trapezoid is the trapezoidal rule, with half-weight on the endpoints.
	//
	// Tail convention: when the window reaches past the trace end
	// (t+b >= T), the "high" endpoint defaults to zero rather than the
	// clipped last-valid sample. This keeps the Riemann-plus-correction
	// form simple and matches the library's reference evaluator, but
	// may differ from implementations that clip the interval per-t.
	// Near-end trapezoid values should be treated as approximate.
	Trapezoid
)

type integralFormula struct {
	sub      Formula
	interval Interval
	scheme   IntegrationScheme
}

func (i *integralFormula) Vars() []string { return i.sub.Vars() }
func (i *integralFormula) String() string {
	return fmt.Sprintf("∫%s %s", intervalString(i.interval), i.sub)
}
func (*integralFormula) sealed() {}

// Integral1d integrates sub's robustness over the interval at each time
// step. The interval must be bounded (panics otherwise).
func Integral1d(sub Formula, iv Interval, scheme IntegrationScheme) Formula {
	if iv.IsUnbounded() {
		panic("stlcg: Integral1d requires a bounded interval")
	}
	return &integralFormula{sub: sub, interval: iv, scheme: scheme}
}
