package stlcg

import "fmt"

// -------- Unary temporal operators --------

type alwaysFormula struct {
	sub      Formula
	interval Interval
}

func (a *alwaysFormula) Vars() []string { return a.sub.Vars() }
func (a *alwaysFormula) String() string {
	return fmt.Sprintf("□%s %s", intervalString(a.interval), a.sub)
}
func (*alwaysFormula) sealed() {}

// Always returns the temporal "always" operator: □_iv phi. At time t,
// robustness is the min of phi's robustness over [t+iv.Lo, t+iv.Hi]
// (forward time, inclusive both ends).
func Always(sub Formula, iv Interval) Formula {
	return &alwaysFormula{sub: sub, interval: iv}
}

type eventuallyFormula struct {
	sub      Formula
	interval Interval
}

func (e *eventuallyFormula) Vars() []string { return e.sub.Vars() }
func (e *eventuallyFormula) String() string {
	return fmt.Sprintf("◇%s %s", intervalString(e.interval), e.sub)
}
func (*eventuallyFormula) sealed() {}

// Eventually returns the temporal "eventually" operator: ◇_iv phi. At
// time t, robustness is the max of phi's robustness over
// [t+iv.Lo, t+iv.Hi].
func Eventually(sub Formula, iv Interval) Formula {
	return &eventuallyFormula{sub: sub, interval: iv}
}

// -------- Binary temporal operators --------

type untilFormula struct {
	left, right Formula
	interval    Interval
	overlap     bool
}

func (u *untilFormula) Vars() []string { return mergeVars(u.left.Vars(), u.right.Vars()) }
func (u *untilFormula) String() string {
	return fmt.Sprintf("(%s U%s %s)", u.left, intervalString(u.interval), u.right)
}
func (*untilFormula) sealed() {}

// Until returns (left U_iv right): left holds until right becomes true,
// with right required to hold at some time in iv. If overlap is true,
// left is allowed to still be true when right triggers (the conventional
// STL until). iv == AllTime() means unbounded.
func Until(left, right Formula, iv Interval, overlap bool) Formula {
	return &untilFormula{left: left, right: right, interval: iv, overlap: overlap}
}

type thenFormula struct {
	left, right Formula
	interval    Interval
	overlap     bool
}

func (t *thenFormula) Vars() []string { return mergeVars(t.left.Vars(), t.right.Vars()) }
func (t *thenFormula) String() string {
	return fmt.Sprintf("(%s T%s %s)", t.left, intervalString(t.interval), t.right)
}
func (*thenFormula) sealed() {}

// Then returns (left T_iv right): left must eventually hold before right
// becomes true. Dual of Until where the inner min-over-prefix is replaced
// by max-over-prefix.
func Then(left, right Formula, iv Interval, overlap bool) Formula {
	return &thenFormula{left: left, right: right, interval: iv, overlap: overlap}
}

// intervalString formats an interval for Stringer output.
func intervalString(iv Interval) string {
	if iv.IsUnbounded() {
		if iv.Lo == 0 {
			return ""
		}
		return fmt.Sprintf("[%d,∞)", iv.Lo)
	}
	return fmt.Sprintf("[%d,%d]", iv.Lo, iv.Hi)
}
