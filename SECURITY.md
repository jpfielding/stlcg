# Security Policy

## Scope

stlcg.go is a library for computing Signal Temporal Logic robustness on
differentiable computation graphs. It does not handle credentials,
parse untrusted text formats (formulas are constructed in-Go), or open
network connections. The surface for traditional security bugs is
small.

The primary concerns are:

- **Denial of service via pathological inputs.** Extremely large
  traces, unbounded intervals, or adversarially chosen scale values
  can blow up graph size, compile time, or memory. These are bugs
  we care about and will fix.
- **Panics on user input.** Any evaluator method that panics when
  given well-formed but unexpected input (e.g. a missing signal
  variable) is treated as a bug. Use the `*E` variants to get error
  returns.
- **Gradient correctness.** If autodiff silently disagrees with a
  centered finite-difference estimate by more than the documented
  tolerance, that is a real bug — it can cause model training to
  silently converge to the wrong optimum.

## Reporting

Please report security-relevant issues privately so they can be
triaged before public disclosure:

- Open a **private security advisory** on
  <https://github.com/jpfielding/stlcg/security/advisories>, or
- Email the maintainer (see `go.mod` author / commit log).

Non-security bugs can go to the public issue tracker.

## Supported versions

Until v1.0.0 is tagged, only the `main` branch is supported.
Post-1.0.0, the latest minor release line gets fixes.

## Disclosure timeline

We aim for:

- Acknowledgement within 3 business days.
- A triage decision within 2 weeks.
- A fix in `main` before any public disclosure.

We will not silently regenerate parity fixtures to mask a behavior
change — any semantic diff is human-reviewed.
