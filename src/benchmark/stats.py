"""
Statistics — Wilson intervals and k/n cell formatting
=====================================================
Every percentage the benchmark prints carries its counts and a 95% interval.
The interval is the Wilson score interval, not the Wald (normal-approximation)
one: with 0 successes in n trials Wald collapses to [0, 0], which would turn
"0 of 60 attacks succeeded" into "guardrails are perfect". Wilson keeps a
non-zero upper bound, which is the honest claim ("0 of N observed; at most X%").
"""

import math
from typing import Optional

Z_95 = 1.96


def wilson_interval(k: int, n: int, z: float = Z_95) -> Optional[tuple]:
    """Wilson score interval (lo, hi) as fractions in [0, 1]; None when n == 0.

    Raises ValueError for impossible counts (k < 0, n < 0 or k > n)."""
    if k < 0 or n < 0 or k > n:
        raise ValueError(f"impossible counts: k={k}, n={n}")
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def prop_fields(prefix: str, k: int, n: int) -> dict:
    """The five report fields for one proportion: <prefix>_{pct,k,n,lo,hi}.

    pct, lo and hi are percentages rounded to one decimal, all None when n == 0."""
    interval = wilson_interval(k, n)
    if interval is None:
        pct = lo = hi = None
    else:
        pct = round(100.0 * k / n, 1)
        lo, hi = round(100.0 * interval[0], 1), round(100.0 * interval[1], 1)
    return {f"{prefix}_pct": pct, f"{prefix}_k": k, f"{prefix}_n": n,
            f"{prefix}_lo": lo, f"{prefix}_hi": hi}


def format_prop(k: int, n: int) -> str:
    """'11.7% (14/120) [7.1, 18.6]'; '— (0/0)' when there is nothing to divide."""
    if n == 0:
        return "— (0/0)"
    f = prop_fields("p", k, n)
    return f"{f['p_pct']:.1f}% ({k}/{n}) [{f['p_lo']:.1f}, {f['p_hi']:.1f}]"
