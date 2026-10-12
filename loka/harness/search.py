"""Force bisection for the standing push-resistance trial.

The outcome is assumed monotone in force: a force that topples the robot
means every larger force does too. Each probe is the caller's responsibility
(a fresh rollout). ``FellBeforePush`` aborts the search: the mass alone
toppled the robot, so the push threshold is reported as 0.
"""

from __future__ import annotations


class FellBeforePush(Exception):
    """The robot fell before the push was applied."""


def bisect_force(
    probe,
    *,
    lo: float,
    hi: float,
    resolution: float,
    max_hi: float,
) -> float:
    """Largest probed force that did not fall, to within ``resolution``.

    ``probe(force)`` returns True when that force topples the robot. ``hi`` is
    doubled until a fall is found or ``max_hi`` is reached, then the bracket
    is halved until it is no wider than ``resolution``. The return value is
    the largest force that survived. A fall at ``lo`` returns 0.
    """
    lo = float(lo)
    hi = float(hi)
    resolution = float(resolution)
    max_hi = float(max_hi)
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    if max_hi < hi:
        raise ValueError("max_hi must be >= hi")
    if probe(lo):
        return 0.0
    while True:
        if probe(hi):
            break
        if hi >= max_hi - 1e-9:
            return float(hi)
        lo = float(hi)
        nxt = min(max_hi, max(hi * 2.0, hi + resolution))
        if nxt <= hi + 1e-12:
            return float(hi)
        hi = float(nxt)
    while hi - lo > resolution + 1e-9:
        mid = 0.5 * (lo + hi)
        if probe(mid):
            hi = float(mid)
        else:
            lo = float(mid)
    return float(lo)


def run_force_search(probe, **kwargs) -> float:
    """``bisect_force``, treating :class:`FellBeforePush` as a threshold of 0."""
    try:
        return bisect_force(probe, **kwargs)
    except FellBeforePush:
        return 0.0
