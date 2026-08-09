"""
evaluate.py — walk-forward validation, baselines, and the coverage curve.

WHAT THIS FILE IS FOR:
  Making it impossible to fool yourself.

  Three ways people get a fake 75% and believe it:
    1. Random train/test split. Leaks the future. Always time-ordered here.
    2. No baseline. 65% sounds good until you learn "always pick the favourite"
       scores 65% on the same rows. Every run prints the baselines.
    3. Reporting accuracy on a filtered slate without saying what was filtered.
       The coverage curve reports BOTH numbers, always, side by side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import CS2Model, Dataset, decide


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5) == (y == 1)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC. No sklearn import needed and handles ties correctly."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    r = 1
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        r += j - i + 1
        i = j + 1
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class Scores:
    name: str
    n: int
    acc: float
    logloss: float
    brier: float
    auc: float

    def row(self) -> str:
        return (
            f"{self.name:<22} n={self.n:<6d} acc={self.acc:6.3f}  "
            f"logloss={self.logloss:6.3f}  brier={self.brier:6.3f}  auc={self.auc:6.3f}"
        )


def score(name: str, y: np.ndarray, p: np.ndarray) -> Scores:
    return Scores(name, len(y), accuracy(y, p), log_loss(y, p), brier(y, p), auc(y, p))


def coverage_curve(
    y: np.ndarray, p: np.ndarray, thresholds: Sequence[float] = ()
) -> List[Dict[str, float]]:
    """
    The honest version of "what accuracy can I get?".

    For each confidence threshold: how often the model is right ON THE MATCHES
    IT CHOOSES TO CALL, and what fraction of the slate that is. You buy
    accuracy with silence. This table is the price list.
    """
    if not thresholds:
        thresholds = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

    rows = []
    for t in thresholds:
        calls = decide(p, t)
        mask = calls != 0
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {"threshold": t, "coverage": 0.0, "accuracy": float("nan"), "n": 0}
            )
            continue
        pred_a = calls[mask] == 1
        correct = pred_a == (y[mask] == 1)
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(n / len(y)),
                "accuracy": float(correct.mean()),
                "n": n,
            }
        )
    return rows


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    """
    Predicted vs actual, bucketed. If the model says 70% and wins 55% of those,
    the abstention trick is broken and no threshold will save you.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(m.sum()),
                "predicted": float(p[m].mean()),
                "actual": float(y[m].mean()),
            }
        )
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    rows = reliability(y, p, bins)
    n = len(y)
    return float(sum(r["n"] / n * abs(r["predicted"] - r["actual"]) for r in rows))


@dataclass
class WalkForwardResult:
    y: np.ndarray
    p_model: np.ndarray
    p_veto: np.ndarray
    p_glicko: np.ndarray
    p_market: Optional[np.ndarray]
    folds: int
    # Row indices into the Dataset that these predictions correspond to.
    # Not cosmetic: folds shorter than the minimum are skipped, so the
    # out-of-sample set is NOT simply "the last N rows". Anything that lines
    # predictions up against per-match data must go through this.
    idx: np.ndarray = None  # type: ignore[assignment]


def walk_forward(
    ds: Dataset,
    initial_frac: float = 0.5,
    n_folds: int = 8,
    C: float = 0.05,
    calibration: str = "auto",
) -> WalkForwardResult:
    """
    Expanding window. Train on everything before the fold, predict the fold,
    never look forward. Concatenate all out-of-sample predictions and score
    them together.
    """
    n = len(ds.y)
    start = int(n * initial_frac)
    step = max(1, (n - start) // n_folds)

    ys, pm, pv, pg, idxs = [], [], [], [], []
    markets: List[float] = []
    have_market = all(m.closing_prob_a is not None for m in ds.matches)

    folds = 0
    for s in range(start, n, step):
        e = min(n, s + step)
        if e - s < 5 or s < 100:
            continue
        model = CS2Model(C=C, calibration=calibration).fit(ds.X[:s], ds.y[:s])
        p = model.predict_proba(ds.X[s:e])
        ys.append(ds.y[s:e])
        pm.append(p)
        pv.append(ds.veto_prob[s:e])
        pg.append(ds.glicko_prob[s:e])
        idxs.append(np.arange(s, e))
        if have_market:
            markets.extend(m.closing_prob_a for m in ds.matches[s:e])
        folds += 1

    return WalkForwardResult(
        y=np.concatenate(ys),
        p_model=np.concatenate(pm),
        p_veto=np.concatenate(pv),
        p_glicko=np.concatenate(pg),
        p_market=np.array(markets) if have_market and markets else None,
        folds=folds,
        idx=np.concatenate(idxs),
    )


def report(res: WalkForwardResult, true_p: Optional[np.ndarray] = None) -> str:
    """The whole story in one printable block."""
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"WALK-FORWARD EVALUATION  ({res.folds} folds, {len(res.y)} out-of-sample matches)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("BASELINES FIRST — the model has to beat these or it is decoration:")
    lines.append("")

    # "Always pick the favourite" = glicko probability thresholded at 0.5.
    fav = score("always-pick-favourite", res.y, np.where(res.p_glicko >= 0.5, 0.999, 0.001))
    lines.append("  " + f"{fav.name:<22} n={fav.n:<6d} acc={fav.acc:6.3f}  (accuracy only —")
    lines.append("  " + " " * 22 + "  it has no real probabilities, so log loss is meaningless)")
    lines.append("  " + score("glicko-only", res.y, res.p_glicko).row())
    lines.append("  " + score("veto-structural", res.y, res.p_veto).row())
    if res.p_market is not None:
        lines.append("  " + score("closing-line", res.y, res.p_market).row())
    lines.append("")
    lines.append("  " + score("FULL MODEL", res.y, res.p_model).row())
    lines.append("")

    ece = expected_calibration_error(res.y, res.p_model)
    lines.append(f"  calibration error (ECE): {ece:.4f}   <- lower is better; under ~0.03 is healthy")
    if true_p is not None and len(true_p) == len(res.y):
        rmse = float(np.sqrt(np.mean((res.p_model - true_p) ** 2)))
        lines.append(f"  RMSE vs TRUE probability: {rmse:.4f}   <- synthetic data only")
    lines.append("")

    lines.append("-" * 78)
    lines.append("COVERAGE CURVE — this is where your accuracy target lives")
    lines.append("-" * 78)
    lines.append(f"  {'confidence':>10}  {'accuracy':>9}  {'coverage':>9}  {'matches':>8}")
    for row in coverage_curve(res.y, res.p_model):
        if row["n"] == 0:
            continue
        lines.append(
            f"  {row['threshold']:>10.2f}  {row['accuracy']:>9.3f}  "
            f"{row['coverage']:>8.1%}  {row['n']:>8d}"
        )
    lines.append("")
    lines.append("  Read it like this: at confidence 0.75 the model is right ~75% of the")
    lines.append("  time on the matches it agrees to call, and it declines the rest.")
    lines.append("")

    lines.append("-" * 78)
    lines.append("RELIABILITY — does '70%' actually mean 70%?")
    lines.append("-" * 78)
    lines.append(f"  {'bucket':>12}  {'predicted':>10}  {'actual':>8}  {'n':>6}")
    for r in reliability(res.y, res.p_model):
        lines.append(
            f"  {r['bin_lo']:.2f}-{r['bin_hi']:.2f}  {r['predicted']:>10.3f}  "
            f"{r['actual']:>8.3f}  {r['n']:>6d}"
        )
    lines.append("")
    return "\n".join(lines)
