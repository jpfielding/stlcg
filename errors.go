package stlcg

import "errors"

// Sentinel errors returned by the *E variants of Evaluator methods. They
// let callers distinguish user-input failures from programmer bugs: the
// latter still panic (unknown AST types, arity mismatches).
var (
	// ErrClosed is returned when a method is called on a closed Evaluator.
	ErrClosed = errors.New("stlcg: Evaluator is closed")

	// ErrMissingSignal is returned when a SignalMap lacks a required
	// variable.
	ErrMissingSignal = errors.New("stlcg: SignalMap missing required variable")

	// ErrTimeOutOfRange is returned by Robustness when AtTime is outside
	// the trace time dimension.
	ErrTimeOutOfRange = errors.New("stlcg: AtTime index out of range")

	// ErrBadShape is returned when an input tensor has the wrong rank or
	// a non-positive dimension.
	ErrBadShape = errors.New("stlcg: invalid tensor shape")

	// ErrExec wraps a gomlx Exec failure.
	ErrExec = errors.New("stlcg: graph execution failed")
)
