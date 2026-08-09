"""
model.py — features, the classifier, calibration, and the decision to shut up.

THE SHAPE OF THE ANSWER:
  1. Ratings give per-map probabilities.
  2. The veto simulator turns those into a structural match probability.
     This is the single strongest feature and it does most of the work.
  3. Logistic regression corrects it using context the ratings cannot see —
     form, roster uncertainty, LAN split, head-to-head, rest.
  4. Isotonic regression CALIBRATES the output, so "72%" means 72%.
  5. An abstention threshold lets the model decline the coinflips.

Step 4 is the one people skip, and it is the one that makes step 5 work. If
the probabilities are honest, then "only answer when p > 0.72" produces ~72%
accuracy on the answered subset almost by definition. Your accuracy target is
purchased with coverage, not with cleverness.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .data import Match
from .ratings import RatingBook
from .veto import match_win_prob

FEATURE_NAMES: Tuple[str, ...] = (
    "veto_logit",       # structural probability from ratings + veto simulation
    "glicko_diff",      # overall rating gap, in units of 100
    "rd_sum",           # how uncertain we are about BOTH teams
    "form_diff",        # last-10 winrate gap
    "streak_diff",      # signed streak gap
    "h2h_diff",         # historical head-to-head
    "lan",              # is this LAN
    "lan_split_diff",   # LAN-vs-online tendency gap, only meaningful when lan=1
    "rest_diff",        # days since last match, gap (capped)
    "tier1",            # top-tier event
    "bo_len",           # 1, 3, or 5 — longer series favour the better team
    "games_min",        # min(matches seen) — proxy for "do we know these teams"
)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _logit_col(p: np.ndarray) -> np.ndarray:
    """Platt scaling operates on log-odds, not raw probabilities."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).reshape(-1, 1)


# Monte Carlo sims per fixture for the veto probability.
#
# Measured, not guessed: sweeping 200 / 600 / 1500 moved out-of-sample log
# loss and AUC by less than 0.001. The veto distribution over seven maps is
# simply not that rough, so this is a pure speed knob and 300 is plenty.
# (It was originally set high on the theory that sampling noise was shrinking
# the feature's coefficient. That theory was wrong — the real cause was
# feeding veto_logit in as a penalised predictor instead of an offset.)
VETO_SIMS = 300

# Every fixture uses the SAME random stream (common random numbers). The noise
# then lands the same way for every match instead of varying independently,
# which stops it behaving like per-row measurement error.
VETO_SEED = 12345


def build_features(
    book: RatingBook, match: Match, seed: int = VETO_SEED, n_sims: int = VETO_SIMS
) -> np.ndarray:
    """
    Everything here must be knowable BEFORE the match. `book` has not yet
    observed this match — that ordering is enforced by the caller.
    """
    a, b = match.team_a, match.team_b

    # Register the lineups BEFORE reading any rating. The five names are known
    # before the game starts, so a roster change must widen uncertainty for
    # this match — not for the one after it.
    book.apply_lineup(a, match.lineup_a)
    book.apply_lineup(b, match.lineup_b)

    sa, sb = book.team(a), book.team(b)

    # Compute the seven map probabilities once, then let the veto simulator
    # hammer a dict instead of re-deriving Glicko expectations thousands of
    # times per fixture.
    table = book.map_prob_table(a, b, match.date)
    veto_p = match_win_prob(table.__getitem__, match.best_of, n_sims=n_sims, seed=seed)

    h2h_w, h2h_l = book.h2h_record(a, b)
    h2h = 0.0 if (h2h_w + h2h_l) == 0 else (h2h_w - h2h_l) / (h2h_w + h2h_l)

    def rest(st) -> float:
        if st.last_played is None:
            return 14.0
        return min(60.0, (match.date - st.last_played).total_seconds() / 86400.0)

    return np.array(
        [
            _logit(veto_p),
            (sa.overall.r - sb.overall.r) / 100.0,
            (sa.overall.rd + sb.overall.rd) / 100.0,
            sa.form(10) - sb.form(10),
            (sa.streak() - sb.streak()) / 5.0,
            h2h,
            1.0 if match.lan else 0.0,
            (sa.lan_split() - sb.lan_split()) * (1.0 if match.lan else 0.0),
            (rest(sa) - rest(sb)) / 30.0,
            1.0 if match.tier == 1 else 0.0,
            float(match.best_of),
            min(60.0, min(sa.overall.games, sb.overall.games)) / 60.0,
        ],
        dtype=float,
    )


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray  # 1 if team_a won
    dates: List
    matches: List[Match]
    # Structural prediction alone, before the classifier — a baseline in itself.
    veto_prob: np.ndarray
    glicko_prob: np.ndarray


def build_dataset(
    matches: Sequence[Match], seed: int = VETO_SEED, n_sims: int = VETO_SIMS
) -> Dataset:
    """
    Walk the whole history forward once. For each match: extract features from
    what we knew, record the outcome, THEN let the book learn from it.

    This ordering is the entire defence against leakage.
    """
    book = RatingBook()
    X, y, dates, kept, vetos, gl = [], [], [], [], [], []

    ordered = sorted(matches, key=lambda m: m.date)
    for i, m in enumerate(ordered):
        if m.winner is None or not m.maps:
            continue
        feats = build_features(book, m, seed=seed, n_sims=n_sims)
        X.append(feats)
        y.append(1 if m.winner == m.team_a else 0)
        dates.append(m.date)
        kept.append(m)
        vetos.append(1.0 / (1.0 + math.exp(-feats[0])))
        sa, sb = book.team(m.team_a).overall, book.team(m.team_b).overall
        from . import glicko as _g

        gl.append(_g.expected_score(sa, sb))
        book.observe(m)

    return Dataset(
        X=np.array(X),
        y=np.array(y),
        dates=dates,
        matches=kept,
        veto_prob=np.array(vetos),
        glicko_prob=np.array(gl),
    )


class _OffsetLogistic:
    """
    Logistic regression with a fixed offset term:

        P(A wins) = sigmoid( veto_logit  +  b  +  X_rest . beta )

    WHY THIS EXISTS (it replaced a plain LogisticRegression over all features):
      The veto probability is the best single signal in the model. Fed in as
      just another column, ridge regularisation treated it like any other
      predictor, split its weight with the collinear rating gap, and the
      combined model scored WORSE than the veto feature on its own — AUC
      0.626 against 0.638. A model losing to one of its own inputs is not a
      model.

      As an offset it is not estimated and not penalised. The classifier's
      only job is the correction on top. At beta = 0 the output is exactly the
      structural probability, so the fitted model starts from the baseline and
      can only move away from it if the data pays for it.
    """

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2
        self.b_: float = 0.0
        self.beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> "_OffsetLogistic":
        from scipy.optimize import minimize

        n, k = X.shape
        yf = y.astype(float)

        def nll_and_grad(params):
            b, beta = params[0], params[1:]
            z = offset + b + X @ beta
            # logaddexp keeps this stable for large |z|
            loss = np.mean(yf * np.logaddexp(0, -z) + (1 - yf) * np.logaddexp(0, z))
            loss += self.l2 * float(beta @ beta) / n
            resid = 1.0 / (1.0 + np.exp(-z)) - yf
            g_b = float(np.mean(resid))
            g_beta = X.T @ resid / n + 2.0 * self.l2 * beta / n
            return loss, np.concatenate(([g_b], g_beta))

        res = minimize(
            nll_and_grad, np.zeros(k + 1), jac=True, method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.b_ = float(res.x[0])
        self.beta_ = res.x[1:]
        return self

    def predict_proba(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        z = offset + self.b_ + X @ self.beta_
        return 1.0 / (1.0 + np.exp(-z))


class CS2Model:
    """
    The structural veto probability, corrected by a small penalised logistic
    on everything the ratings cannot see, then optionally calibrated.

    Deliberately small. With a few thousand matches, a big model memorises
    noise and its probabilities stop meaning anything — which destroys the
    abstention trick that your accuracy target depends on.
    """

    def __init__(self, C: float = 0.05, calibration: str = "auto") -> None:
        """
        calibration:
          "auto"     — sigmoid below 1000 held-out rows, isotonic above.
          "sigmoid"  — Platt scaling. Two parameters. Safe on small samples.
          "isotonic" — free-form monotone fit. Needs a LOT of data.
          "none"     — trust the logistic output as-is.

        WHY THE DEFAULT IS NOT ISOTONIC:
          Isotonic was the first thing here and it made the model measurably
          worse — log loss 0.668 vs 0.658, AUC 0.644 vs 0.659. With only a few
          hundred held-out rows it fits a coarse step function; the flat steps
          collapse distinct probabilities into ties, and ties destroy ranking.
          Isotonic is the right tool once you have thousands of matches, and
          the wrong one before that.
        """
        # C is kept as the familiar inverse-regularisation knob; the offset
        # model wants a penalty, so invert it.
        self.clf = _OffsetLogistic(l2=1.0 / max(C, 1e-9))
        self.calibration = calibration
        self.iso: Optional[IsotonicRegression] = None
        self.platt: Optional[LogisticRegression] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    @staticmethod
    def _split(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Column 0 is veto_logit (the offset); the rest are corrections."""
        return X[:, 0], X[:, 1:]

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _raw(self, X: np.ndarray) -> np.ndarray:
        offset, rest = self._split(X)
        return self.clf.predict_proba(self._scale(rest), offset)

    def fit(self, X: np.ndarray, y: np.ndarray, calib_frac: float = 0.25) -> "CS2Model":
        """
        The last `calib_frac` of the training window (by row order, which is
        time order) is reserved to fit the calibrator. Calibrating on data the
        classifier trained on produces a calibrator that flatters a liar.
        """
        _, rest_all = self._split(X)
        self.mean_ = rest_all.mean(axis=0)
        self.std_ = rest_all.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0

        n = len(X)
        self.iso = self.platt = None
        want_calib = self.calibration != "none"
        cut = int(n * (1 - calib_frac)) if want_calib else n
        if n - cut < 80:  # too little to calibrate on; don't pretend
            cut, want_calib = n, False

        offset_tr, rest_tr = self._split(X[:cut])
        self.clf.fit(self._scale(rest_tr), y[:cut], offset_tr)
        if not want_calib:
            return self

        raw = self._raw(X[cut:])
        y_cal = y[cut:]

        method = self.calibration
        if method == "auto":
            method = "isotonic" if len(y_cal) >= 1000 else "sigmoid"

        if method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            self.iso.fit(raw, y_cal)
        else:
            self.platt = LogisticRegression(C=1e6, max_iter=1000)
            self.platt.fit(_logit_col(raw), y_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw(X)
        if self.iso is not None:
            return np.clip(self.iso.predict(raw), 0.01, 0.99)
        if self.platt is not None:
            return np.clip(self.platt.predict_proba(_logit_col(raw))[:, 1], 0.01, 0.99)
        return raw

    def coefficients(self) -> Dict[str, float]:
        """
        Correction weights. veto_logit is absent by design — it enters as a
        fixed offset with an implicit coefficient of exactly 1.
        """
        return dict(zip(FEATURE_NAMES[1:], self.clf.beta_))


def decide(probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    +1 = call team A, 0 = NO CALL, -1 = call team B.

    The no-call is a feature. A model that must answer every match is a model
    that is wrong about a third of them.

    `threshold` is a CONFIDENCE, so it is clamped at 0.5 — below that the two
    bands overlap and every call is simultaneously "A" and "B". The previous
    version assigned both masks unconditionally and let the second overwrite
    the first, so decide([0.55, 0.59], 0.45) confidently returned "team B"
    twice. Nothing in the codebase passes a sub-0.5 threshold today, but
    coverage_curve() accepts arbitrary ones, so a sweep starting at 0.4 would
    have silently inverted every call it made.
    """
    t = max(float(threshold), 0.5)
    return np.where(probs >= t, 1, np.where(probs <= 1.0 - t, -1, 0)).astype(int)
