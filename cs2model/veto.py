"""
veto.py — turn seven per-map probabilities into one match probability.

THE POINT OF THIS FILE:
  Team A can be 60% against Team B on average and still be a dog in the actual
  series, because the veto decides WHICH maps get played and both teams are
  actively steering it. A model that skips this step is answering a question
  nobody asked: "who would win on a random map?"

  Nobody plays on a random map.

We Monte-Carlo the ban/pick sequence (teams are goal-directed but not perfect,
hence a temperature), then compute the series probability EXACTLY for each
sampled map set. Exact where we can be, sampled only where we must.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

from .data import ACTIVE_DUTY, VETO_SCRIPTS, series_prob

# How rational the veto is. 0 -> perfectly greedy, higher -> sloppier.
# Real vetoes are prep-driven and somewhat predictable, but not deterministic.
#
# NOTE ON UNITS: edges are measured in LOG-ODDS, not probability. Probability
# units squash everything into +/-0.5, and at any sane temperature the softmax
# then comes out nearly uniform — a team hopeless on Nuke would ban it only a
# quarter of the time, which is nonsense. Log-odds is the natural strength
# scale and makes the veto respond proportionally to the actual skill gap.
VETO_TEMPERATURE = 0.6


def _edge_logit(p: float) -> float:
    """Signed strength edge on a map, in log-odds, from A's point of view."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _weight_tables(
    probs: Sequence[float], temp: float
) -> Tuple[List[float], List[float]]:
    """
    Precompute both softmax weight tables ONCE per fixture.

    The trick: every score in the veto is either +logit or -logit, because
    "ban my worst" and "pick my best" are the same comparison with opposite
    sign, and team B sees the negation of team A's edge. So there are only two
    possible weights per map — exp(+logit/T) and its reciprocal — and the
    inner loop never has to call exp() at all.

        actor A, ban  -> exp(-logit/T)      actor A, pick -> exp(+logit/T)
        actor B, ban  -> exp(+logit/T)      actor B, pick -> exp(-logit/T)
    """
    pos, neg = [], []
    for p in probs:
        e = math.exp(_edge_logit(p) / temp)
        pos.append(e)
        neg.append(1.0 / e)
    return pos, neg


def _pick_index(weights: Sequence[float], live: Sequence[int], rng: random.Random) -> int:
    """Weighted choice over the still-available map indices."""
    total = 0.0
    for i in live:
        total += weights[i]
    r = rng.random() * total
    acc = 0.0
    for i in live:
        acc += weights[i]
        if r <= acc:
            return i
    return live[-1]


def _simulate_indices(
    pos: Sequence[float],
    neg: Sequence[float],
    best_of: int,
    rng: random.Random,
) -> List[Tuple[int, str]]:
    """One sampled veto, over map indices. Returns (index, picked_by)."""
    live = list(range(len(pos)))
    picks: List[Tuple[int, str]] = []

    for actor, action in VETO_SCRIPTS[best_of]:
        # A bans low / picks high; B is the mirror image.
        use_pos = (actor == "A") == (action == "pick")
        idx = _pick_index(pos if use_pos else neg, live, rng)
        live.remove(idx)
        if action == "pick":
            picks.append((idx, actor))

    for i in live:
        picks.append((i, "decider"))
    return picks[:best_of]


def simulate_veto(
    map_prob: Callable[[str], float],
    best_of: int,
    rng: random.Random,
    pool: Sequence[str] = ACTIVE_DUTY,
    temp: float = VETO_TEMPERATURE,
) -> List[Tuple[str, str]]:
    """
    One sampled veto. `map_prob(m)` is P(A wins map m).
    Returns the played maps in order, as (map_name, picked_by).
    """
    probs = [map_prob(m) for m in pool]
    pos, neg = _weight_tables(probs, temp)
    return [(pool[i], pb) for i, pb in _simulate_indices(pos, neg, best_of, rng)]


def match_win_prob(
    map_prob: Callable[[str], float],
    best_of: int,
    n_sims: int = 600,
    seed: int = 0,
    pool: Sequence[str] = ACTIVE_DUTY,
    pick_advantage: float = 0.02,
    temp: float = VETO_TEMPERATURE,
) -> float:
    """
    P(A wins the series), marginalising over how the veto might go.

    `pick_advantage` nudges each map toward whoever picked it — teams pick
    maps they have prepped, and the raw rating does not fully capture that.
    Small on purpose; it is a real but modest effect.
    """
    probs = [map_prob(m) for m in pool]
    pos, neg = _weight_tables(probs, temp)

    # Pre-adjust for pick advantage so the inner loop is pure arithmetic.
    p_apick = [min(0.99, p + pick_advantage) for p in probs]
    p_bpick = [max(0.01, p - pick_advantage) for p in probs]

    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_sims):
        played = _simulate_indices(pos, neg, best_of, rng)
        series = []
        for i, picked_by in played:
            if picked_by == "A":
                series.append(p_apick[i])
            elif picked_by == "B":
                series.append(p_bpick[i])
            else:
                series.append(probs[i])
        total += series_prob(series, best_of)
    return total / n_sims


def veto_report(
    map_prob: Callable[[str], float],
    best_of: int,
    n_sims: int = 2000,
    seed: int = 0,
    pool: Sequence[str] = ACTIVE_DUTY,
    temp: float = VETO_TEMPERATURE,
) -> Dict[str, float]:
    """How often each map actually reaches the server. Useful for sanity checks
    and for explaining a prediction to a human."""
    probs = [map_prob(m) for m in pool]
    pos, neg = _weight_tables(probs, temp)
    rng = random.Random(seed)
    counts = [0.0] * len(pool)
    for _ in range(n_sims):
        for i, _pb in _simulate_indices(pos, neg, best_of, rng):
            counts[i] += 1
    out = {pool[i]: counts[i] / n_sims for i in range(len(pool))}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
