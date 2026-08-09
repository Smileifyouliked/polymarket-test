"""
glicko.py — Glicko-2, implemented properly.

WHY NOT ELO:
  Elo gives you a number. Glicko gives you a number AND how much it trusts
  itself (the rating deviation, RD). That second number is what lets the model
  say "I don't know" about a team returning from a roster overhaul, instead of
  confidently quoting a rating built by five players who no longer play here.

Each match is treated as its own rating period against a single opponent.
That is a standard simplification and it is what you want for a scene where
teams play on wildly different schedules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCALE = 173.7178  # Glicko-1 -> Glicko-2 conversion factor
BASE = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
TAU = 0.5  # constrains volatility change; 0.3-1.2 is the sane band
EPSILON = 1e-6


@dataclass
class Rating:
    r: float = BASE
    rd: float = DEFAULT_RD
    vol: float = DEFAULT_VOL
    games: int = 0

    @property
    def mu(self) -> float:
        return (self.r - BASE) / SCALE

    @property
    def phi(self) -> float:
        return self.rd / SCALE

    def copy(self) -> "Rating":
        return Rating(self.r, self.rd, self.vol, self.games)


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def expected_score(a: Rating, b: Rating) -> float:
    """P(a beats b), accounting for how uncertain b's rating is."""
    return _E(a.mu, b.mu, b.phi)


def _solve_volatility(phi: float, v: float, delta: float, vol: float) -> float:
    """
    Illinois-variant regula falsi, per Glickman's paper. This is the fiddly
    part of Glicko-2 and the part most implementations get wrong or skip.
    """
    a = math.log(vol * vol)
    phi2, v_, d2 = phi * phi, v, delta * delta

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (d2 - phi2 - v_ - ex)
        den = 2.0 * (phi2 + v_ + ex) ** 2
        return num / den - (x - a) / (TAU * TAU)

    A = a
    if d2 > phi2 + v_:
        B = math.log(d2 - phi2 - v_)
    else:
        k = 1
        while f(a - k * TAU) < 0 and k < 100:
            k += 1
        B = a - k * TAU

    fA, fB = f(A), f(B)
    n = 0
    while abs(B - A) > EPSILON and n < 100:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
        n += 1
    return math.exp(A / 2.0)


def update(rating: Rating, opponent: Rating, score: float) -> Rating:
    """
    One rating period, one opponent. `score` is 1.0 win, 0.0 loss.
    Returns a NEW Rating — callers decide when to commit it.
    """
    mu, phi = rating.mu, rating.phi
    mu_j, phi_j = opponent.mu, opponent.phi

    g_j = _g(phi_j)
    E_j = _E(mu, mu_j, phi_j)

    v = 1.0 / (g_j * g_j * E_j * (1.0 - E_j))
    delta = v * g_j * (score - E_j)

    new_vol = _solve_volatility(phi, v, delta, rating.vol)
    phi_star = math.sqrt(phi * phi + new_vol * new_vol)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    new_mu = mu + new_phi * new_phi * g_j * (score - E_j)

    return Rating(
        r=BASE + SCALE * new_mu,
        rd=min(SCALE * new_phi, DEFAULT_RD),
        vol=new_vol,
        games=rating.games + 1,
    )


def decay_idle(rating: Rating, days_idle: float, per_day: float = 1.2) -> Rating:
    """
    A team that hasn't played in three months is less known than one that
    played yesterday. Inflate RD, leave the rating alone.
    """
    out = rating.copy()
    out.rd = min(DEFAULT_RD, math.sqrt(out.rd**2 + (per_day * max(0.0, days_idle)) ** 2))
    return out


def inflate_for_roster_change(rating: Rating, n_changed: int) -> Rating:
    """
    THE THING TEAM-LEVEL MODELS GET WRONG.

    A roster swap does not make the old rating wrong, it makes it UNCERTAIN.
    So we widen RD and pull the rating a fraction of the way back toward the
    population mean — more for each player replaced. Swap all five and you are
    essentially a new team, which is correct: you are.
    """
    if n_changed <= 0:
        return rating.copy()
    frac = min(1.0, n_changed / 5.0)
    out = rating.copy()
    out.r = rating.r + (BASE - rating.r) * (0.35 * frac)
    out.rd = min(DEFAULT_RD, math.sqrt(rating.rd**2 + (170.0 * frac) ** 2))
    out.vol = min(0.12, rating.vol * (1.0 + 0.5 * frac))
    return out
