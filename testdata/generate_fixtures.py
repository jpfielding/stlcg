#!/usr/bin/env python3
"""
generate_fixtures.py — one-time Python stlcg → JSON fixture generator.

Produces parity fixtures consumed by parity_test.go. Each fixture is a
single JSON object of the form:

    {
        "id":        "unique-name",
        "formula":   "□[0,5] (x > 0.5)",      # human-readable label
        "mode":      "exact" | "smooth",
        "scale":     0.0 | 5.0 | ...,          # tau
        "pscale":    1.0,
        "signals":   { "x": [float, ...], "y": [...] },
        "rho_trace": [float, ...]              # forward-time [T] floats
    }

Forward-time convention: index 0 is the earliest observation. The Python
stlcg library uses a time-reversed convention internally, so this script
reverses traces on input and outputs on output. For nested bounded
operators (the Codex-flagged hazardous case) the reversal is done once
on inputs; the expected outputs are re-derived after taking the
per-sample result from Python stlcg and reversing it back to forward
time.

Usage:
    pip install torch numpy
    git clone https://github.com/stanfordASL/stlcg
    cd stlcg
    pip install -e .
    cd -
    python3 testdata/generate_fixtures.py > testdata/fixtures.jsonl

The output is JSON Lines (one record per line). Committed fixtures
should live in testdata/fixtures/ with one file per record (for git
diff friendliness) or in a single fixtures.jsonl file.

This script is committed for auditability and reproducibility — do NOT
silently regenerate fixtures in CI. Any changes to expected robustness
values represent a semantic diff and must be human-reviewed.
"""

import json
import sys

try:
    import torch  # type: ignore
    import numpy as np  # type: ignore
    from src import stlcg  # type: ignore
except ImportError as e:
    print(f"Missing dependencies: {e}. See module docstring for setup.", file=sys.stderr)
    sys.exit(1)


def reverse(trace):
    """Forward → reverse convention (Python stlcg input)."""
    return trace[::-1]


def to_torch(values):
    """Build a [1, T, 1] tensor from a list of floats, time-reversed."""
    arr = np.asarray(reverse(values), dtype=np.float32).reshape(1, -1, 1)
    return torch.from_numpy(arr)


def robustness(phi, inputs, scale, pscale):
    rho = phi.robustness_trace(inputs, scale=scale, pscale=pscale)
    # Reverse output back to forward time.
    return rho.detach().numpy().squeeze().tolist()[::-1]


def emit(fixture):
    print(json.dumps(fixture), flush=True)


def main():
    x_trace = [0.5, 1.0, 0.3, 0.8, 0.2, 0.7, 0.4, 0.9, 0.1, 0.6]
    y_trace = [-0.3, 0.5, -0.1, 0.8, 0.2, -0.4, 0.1, 0.6, -0.2, 0.3]

    x_expr = stlcg.Expression("x", to_torch(x_trace))
    y_expr = stlcg.Expression("y", to_torch(y_trace))

    cases = [
        ("predicate_gt",       x_expr > 0.5),
        ("predicate_lt",       x_expr < 0.5),
        ("not_pred",           stlcg.Negation(x_expr > 0.5)),
        ("and_pred",           (x_expr > 0.5) & (y_expr > 0.0)),
        ("or_pred",            (x_expr > 0.5) | (y_expr < 0.0)),
        ("always_bounded",     stlcg.Always((x_expr > 0.4), interval=[0, 3])),
        ("always_unbounded",   stlcg.Always((x_expr > 0.0), interval=None)),
        ("eventually_bounded", stlcg.Eventually((x_expr > 0.7), interval=[0, 4])),
        ("nested_always_ev",   stlcg.Always(stlcg.Eventually((x_expr > 0.5), interval=[0, 2]), interval=[1, 3])),
    ]

    for (name, phi) in cases:
        for (mode, scale) in [("exact", -1.0), ("smooth", 5.0)]:
            rho = robustness(phi, (x_expr if "y" not in name else (x_expr, y_expr)),
                             scale=scale, pscale=1.0)
            emit({
                "id":        f"{name}_{mode}",
                "formula":   str(phi),
                "mode":      mode,
                "scale":     float(scale),
                "pscale":    1.0,
                "signals":   {"x": x_trace, "y": y_trace},
                "rho_trace": rho,
            })


if __name__ == "__main__":
    main()
