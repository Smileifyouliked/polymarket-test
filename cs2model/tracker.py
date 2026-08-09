"""
tracker.py — bankroll ledger and heartbeat.

WHAT THIS DOES
  Records every position you take, keeps them "open" until the match resolves,
  settles them against the result, and tracks your capital, PnL, win rate and
  exposure. It also records whether the model was WELL CALIBRATED on the bets
  you actually made, which is the only number that tells you if the forecaster
  is still working.

WHAT THIS DOES NOT DO
  It does not place bets. Nothing here talks to a bookmaker or moves money.
  It is a record of decisions you make, so you can see whether they were good.

DURABILITY
  The ledger is the only thing you cannot regenerate — ratings and models can
  be rebuilt from match data, but your position history cannot. So every write
  goes to a temp file and is atomically renamed over the real one. A crash
  mid-write leaves the previous ledger intact rather than a truncated JSON
  file that loses your entire history.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _atomic_write_json(path: str, payload) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    """One decision, from the moment you take it to the moment it resolves."""

    id: str
    team_a: str
    team_b: str
    pick: str                 # which team you backed
    model_prob: float         # what the model said about YOUR pick
    odds: float               # decimal odds you got (2.0 == even money)
    stake: float
    opened_at: str
    best_of: int = 3
    event: str = ""
    note: str = ""
    status: str = "open"      # open | won | lost | void
    settled_at: Optional[str] = None
    pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def implied_prob(self) -> float:
        """What the market thinks, stripped out of the odds."""
        return 1.0 / self.odds if self.odds > 0 else 0.0

    @property
    def edge(self) -> float:
        """Model probability minus market probability. Your claimed advantage."""
        return self.model_prob - self.implied_prob

    @property
    def to_win(self) -> float:
        return self.stake * (self.odds - 1.0)


def kelly_fraction(prob: float, odds: float, cap: float = 0.25) -> float:
    """
    Fraction of bankroll Kelly would stake, capped.

    Full Kelly is famously optimal and famously unusable — it assumes your
    probability is exactly right, and a model that is even slightly
    over-confident will happily bet you into ruin. The cap (default quarter
    Kelly) is the whole reason this is safe to look at.
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, min(f, 1.0)) * cap


# ─────────────────────────────────────────────────────────────────────────────
# LEDGER
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Stats:
    starting_capital: float
    capital: float
    realised_pnl: float
    open_count: int
    at_risk: float
    exposure_pct: float
    wins: int
    losses: int
    voids: int
    settled: int
    win_rate: float
    roi: float
    peak_capital: float
    drawdown: float
    drawdown_pct: float
    avg_edge: float
    brier: float
    calibration_gap: float
    biggest_win: float
    biggest_loss: float


class Ledger:
    def __init__(self, path: str = "data/ledger.json", starting_capital: float = 0.0):
        self.path = path
        self.starting_capital = starting_capital
        self.positions: List[Position] = []
        self.created_at = _now().isoformat()
        if os.path.exists(path):
            self.load()

    # ── persistence ──────────────────────────────────────────────────────────

    def load(self) -> "Ledger":
        with open(self.path) as f:
            d = json.load(f)
        self.starting_capital = d.get("starting_capital", self.starting_capital)
        self.created_at = d.get("created_at", self.created_at)
        self.positions = [Position(**p) for p in d.get("positions", [])]
        return self

    def save(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "starting_capital": self.starting_capital,
                "created_at": self.created_at,
                "updated_at": _now().isoformat(),
                "positions": [asdict(p) for p in self.positions],
            },
        )

    # ── mutation ─────────────────────────────────────────────────────────────

    def open_position(
        self,
        team_a: str,
        team_b: str,
        pick: str,
        model_prob: float,
        odds: float,
        stake: float,
        best_of: int = 3,
        event: str = "",
        note: str = "",
    ) -> Position:
        if pick not in (team_a, team_b):
            raise ValueError(f"pick {pick!r} must be either {team_a!r} or {team_b!r}")
        if odds <= 1.0:
            raise ValueError("decimal odds must be greater than 1.0")
        if stake <= 0:
            raise ValueError("stake must be positive")
        if not 0.0 < model_prob < 1.0:
            raise ValueError("model_prob must be strictly between 0 and 1")

        available = self.stats().capital - self.stats().at_risk
        if stake > available + 1e-9:
            raise ValueError(
                f"stake {stake:.2f} exceeds available capital {available:.2f} "
                f"(capital minus what is already at risk on open positions)"
            )

        p = Position(
            id=uuid.uuid4().hex[:8],
            team_a=team_a,
            team_b=team_b,
            pick=pick,
            model_prob=model_prob,
            odds=odds,
            stake=stake,
            opened_at=_now().isoformat(),
            best_of=best_of,
            event=event,
            note=note,
        )
        self.positions.append(p)
        self.save()
        return p

    def settle(self, position_id: str, result: str) -> Position:
        """result: 'won' | 'lost' | 'void'."""
        if result not in ("won", "lost", "void"):
            raise ValueError("result must be won, lost or void")
        p = self.get(position_id)
        if p is None:
            raise KeyError(f"no position with id {position_id!r}")
        if not p.is_open:
            raise ValueError(f"position {position_id} is already {p.status}")

        p.status = result
        p.settled_at = _now().isoformat()
        p.pnl = {"won": p.to_win, "lost": -p.stake, "void": 0.0}[result]
        self.save()
        return p

    def get(self, position_id: str) -> Optional[Position]:
        for p in self.positions:
            if p.id == position_id:
                return p
        return None

    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.is_open]

    def settled_positions(self) -> List[Position]:
        return [p for p in self.positions if not p.is_open]

    # ── analysis ─────────────────────────────────────────────────────────────

    def equity_curve(self) -> List[float]:
        """Capital after each settled position, in settlement order."""
        settled = sorted(self.settled_positions(), key=lambda p: p.settled_at or "")
        cap = self.starting_capital
        curve = [cap]
        for p in settled:
            cap += p.pnl
            curve.append(cap)
        return curve

    def stats(self) -> Stats:
        settled = self.settled_positions()
        opens = self.open_positions()

        realised = sum(p.pnl for p in settled)
        capital = self.starting_capital + realised
        at_risk = sum(p.stake for p in opens)

        wins = sum(1 for p in settled if p.status == "won")
        losses = sum(1 for p in settled if p.status == "lost")
        voids = sum(1 for p in settled if p.status == "void")
        decided = wins + losses

        staked = sum(p.stake for p in settled if p.status != "void")

        curve = self.equity_curve()
        peak = max(curve) if curve else self.starting_capital
        drawdown = peak - capital

        # Was the model right about how sure it was? Over settled, decided bets:
        # mean predicted probability versus the fraction that actually won.
        decided_ps = [p for p in settled if p.status in ("won", "lost")]
        if decided_ps:
            mean_pred = sum(p.model_prob for p in decided_ps) / len(decided_ps)
            actual = wins / len(decided_ps)
            calib_gap = mean_pred - actual
            brier = sum(
                (p.model_prob - (1.0 if p.status == "won" else 0.0)) ** 2
                for p in decided_ps
            ) / len(decided_ps)
        else:
            calib_gap = 0.0
            brier = 0.0

        pnls = [p.pnl for p in settled] or [0.0]

        return Stats(
            starting_capital=self.starting_capital,
            capital=capital,
            realised_pnl=realised,
            open_count=len(opens),
            at_risk=at_risk,
            exposure_pct=(at_risk / capital * 100.0) if capital > 0 else 0.0,
            wins=wins,
            losses=losses,
            voids=voids,
            settled=len(settled),
            win_rate=(wins / decided) if decided else 0.0,
            roi=(realised / staked * 100.0) if staked > 0 else 0.0,
            peak_capital=peak,
            drawdown=drawdown,
            drawdown_pct=(drawdown / peak * 100.0) if peak > 0 else 0.0,
            avg_edge=(sum(p.edge for p in self.positions) / len(self.positions))
            if self.positions
            else 0.0,
            brier=brier,
            calibration_gap=calib_gap,
            biggest_win=max(pnls),
            biggest_loss=min(pnls),
        )


# ─────────────────────────────────────────────────────────────────────────────
# HEARTBEAT
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Beat:
    ts: str
    pid: int
    status: str
    note: str
    counter: int

    @property
    def age_seconds(self) -> float:
        try:
            t = datetime.fromisoformat(self.ts)
        except ValueError:
            return float("inf")
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        return (_now() - t).total_seconds()


class Heartbeat:
    """
    Proof of life for a long-running process.

    The file is rewritten on every beat with a timestamp. Anything else — the
    dashboard, a cron job, you over SSH — reads it and compares the age to an
    expected interval. Stale means the process died or wedged.

    Why a file and not a log line: a log tells you what happened, a heartbeat
    tells you what is happening NOW, in one cheap read, with no parsing.
    """

    def __init__(self, path: str = "data/heartbeat.json", stale_after: float = 180.0):
        self.path = path
        self.stale_after = stale_after
        self._counter = 0

    def beat(self, status: str = "ok", note: str = "") -> Beat:
        self._counter += 1
        b = Beat(
            ts=_now().isoformat(),
            pid=os.getpid(),
            status=status,
            note=note,
            counter=self._counter,
        )
        _atomic_write_json(self.path, asdict(b))
        return b

    def read(self) -> Optional[Beat]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path) as f:
                return Beat(**json.load(f))
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def is_alive(self) -> bool:
        b = self.read()
        return b is not None and b.age_seconds <= self.stale_after

    def describe(self) -> str:
        b = self.read()
        if b is None:
            return "NEVER STARTED — no heartbeat file yet"
        age = b.age_seconds
        if age <= self.stale_after:
            return f"ALIVE — last beat {_fmt_age(age)} ago (pid {b.pid}, beat #{b.counter})"
        return (
            f"STALE — last beat {_fmt_age(age)} ago, expected every "
            f"{_fmt_age(self.stale_after)}. The process is probably dead."
        )


def _fmt_age(seconds: float) -> str:
    if seconds == float("inf"):
        return "never"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"
