package stlcg

// Mode selects between smooth and exact min/max in Maxish/Minish reductions.
type Mode int

const (
	// ModeSmooth is the default: differentiable (1/tau)*LogSumExp(tau*x).
	ModeSmooth Mode = iota
	// ModeExact is the true min/max. Non-differentiable at ties.
	ModeExact
)

func (m Mode) String() string {
	switch m {
	case ModeSmooth:
		return "smooth"
	case ModeExact:
		return "exact"
	}
	return "unknown"
}

// TiePolicy controls how gradient mass is distributed when multiple values
// tie for an extremum. It is orthogonal to Mode. Smooth mode already
// distributes gradients naturally via softmax; the policy there is a label
// for user intent. Exact mode with TieUniform uses a stop-gradient tie mask.
type TiePolicy int

const (
	// TieArgmax sends the full gradient to a single argmax (XLA default).
	TieArgmax TiePolicy = iota
	// TieUniform splits gradient uniformly across tied extrema.
	TieUniform
)

func (t TiePolicy) String() string {
	switch t {
	case TieArgmax:
		return "argmax"
	case TieUniform:
		return "uniform"
	}
	return "unknown"
}

// config holds the compile-time + runtime configuration of an Evaluator.
type config struct {
	mode   Mode
	tie    TiePolicy
	agm    bool
	keep   bool
	pscale float64
	scale  float64 // tau = abs(scale); 0 forces ModeExact
}

func defaultConfig() config {
	return config{
		mode:   ModeSmooth,
		tie:    TieArgmax,
		agm:    false,
		keep:   true,
		pscale: 1.0,
		scale:  1.0,
	}
}

// Option customizes an Evaluator.
type Option func(*config)

// WithMode sets the min/max evaluation mode.
func WithMode(m Mode) Option { return func(c *config) { c.mode = m } }

// WithTieGradient sets the tie-gradient policy.
func WithTieGradient(t TiePolicy) Option { return func(c *config) { c.tie = t } }

// WithAGM enables arithmetic-geometric mean robustness.
// Currently unimplemented (Phase D follow-up); setting true will panic at
// compile time until support lands.
func WithAGM(on bool) Option { return func(c *config) { c.agm = on } }

// WithKeepDim retains the reduced axis as size 1. Default true, matching the
// Python original's convention.
func WithKeepDim(on bool) Option { return func(c *config) { c.keep = on } }

// WithPScale sets the predicate-robustness scale factor. Fed as a graph
// parameter; changing it does not trigger recompile.
func WithPScale(v float64) Option { return func(c *config) { c.pscale = v } }

// WithScale sets the smooth-approximation temperature. Interpreted as
// tau = |scale|. Fed as a graph parameter; changing it does not trigger
// recompile. If scale == 0, Mode is coerced to ModeExact at compile time.
func WithScale(v float64) Option { return func(c *config) { c.scale = v } }
