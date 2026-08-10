"""
tracker.py — bankroll ledger and heartbeat, priced the way Polymarket prices.

WHY PRICE AND SHARES, NOT ODDS AND STAKE
  On Polymarket you buy SHARES at a PRICE between 0 and 1, and each winning
  share pays exactly $1. The price IS the market's probability, which makes
  comparing it to a model probability trivial — no odds conversion, no vig to
  strip out.

  This also covers traditional bookmakers for free: a decimal-odds bet at
  odds O for stake S is the same position as shares = S*O bought at
  price = 1/O. Both live in one ledger with one set of arithmetic.

WHAT CHANGES VERSUS A SPORTSBOOK
  You can SELL before the match resolves. A position is therefore not just
  won/lost — it can be closed at a price, in part or in whole, and its open
  value moves with the market. Hence close() and mark_to_market().

DURABILITY
  The ledger is the only thing here you cannot regenerate. Every write goes to
  a temp file and is atomically renamed, so a crash mid-write leaves the
  previous ledger intact instead of truncated JSON.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
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


def price_to_odds(price: float) -> float:
    """Polymarket price -> decimal odds, for anyone who thinks in odds."""
    return float("inf") if price <= 0 else 1.0 / price


def odds_to_price(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    """
    One holding. `shares` of one outcome, bought at `price`.

      cost        = shares * price
      payout      = shares          (if the outcome resolves YES)
      profit      = shares * (1 - price)
      loss        = shares * price  (the whole cost)
    """

    id: str
    market: str               # human label, e.g. "Vitality vs Spirit"
    outcome: str              # what you actually bought, e.g. "Vitality"
    price: float              # entry price, 0-1
    shares: float
    model_prob: float         # model probability for THIS outcome
    opened_at: str

    token_id: str = ""        # Polymarket CLOB token; empty for manual bets
    venue: str = "polymarket"
    market_kind: str = "match"   # "match" | "map"
    map_name: str = ""
    best_of: int = 3
    event: str = ""
    note: str = ""
    dry_run: bool = True

    status: str = "open"      # open | won | lost | void | closed
    settled_at: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def cost(self) -> float:
        return self.shares * self.price

    @property
    def implied_prob(self) -> float:
        """On Polymarket the price is the market's probability, directly."""
        return self.price

    @property
    def edge(self) -> float:
        return self.model_prob - self.price

    @property
    def to_win(self) -> float:
        return self.shares * (1.0 - self.price)

    @property
    def odds(self) -> float:
        return price_to_odds(self.price)

    def value_at(self, mark: float) -> float:
        return self.shares * mark

    def unrealised_at(self, mark: float) -> float:
        return self.shares * (mark - self.price)


def kelly_fraction(prob: float, price: float, cap: float = 0.25) -> float:
    """
    Fraction of bankroll Kelly would put on a share bought at `price`.

    Buying at price p pays 1 per share, so b = (1 - p) / p and the Kelly
    fraction reduces to (prob - p) / (1 - p).

    Full Kelly is optimal only if your probability is exactly right. It is not.
    A model that is even slightly over-confident will size its way to zero, so
    the cap (quarter Kelly by default) is the entire reason this is safe to
    put in front of an automated system.
    """
    if not 0.0 < price < 1.0:
        return 0.0
    f = (prob - price) / (1.0 - price)
    return max(0.0, min(f, 1.0)) * cap


# ─────────────────────────────────────────────────────────────────────────────
# LEDGER
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Stats:
    starting_capital: float
    capital: float              # realised only
    equity: float               # realised + open positions marked to market
    realised_pnl: float
    unrealised_pnl: float
    open_count: int
    at_risk: float
    exposure_pct: float
    wins: int
    losses: int
    voids: int
    closed_early: int
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
    pnl_today: float


class Ledger:
    def __init__(self, path: str = "data/ledger.json", starting_capital: float = 0.0):
        self.path = path
        self.starting_capital = starting_capital
        self.positions: List[Position] = []
        self.created_at = _now().isoformat()
        self.marks: Dict[str, float] = {}   # position id -> latest market price
        if os.path.exists(path):
            self.load()

    # ── persistence ──────────────────────────────────────────────────────────

    def load(self) -> "Ledger":
        with open(self.path) as f:
            d = json.load(f)
        self.starting_capital = d.get("starting_capital", self.starting_capital)
        self.created_at = d.get("created_at", self.created_at)
        self.marks = d.get("marks", {})
        known = {f.name for f in Position.__dataclass_fields__.values()}
        self.positions = [
            Position(**{k: v for k, v in p.items() if k in known})
            for p in d.get("positions", [])
        ]
        return self

    def save(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "starting_capital": self.starting_capital,
                "created_at": self.created_at,
                "updated_at": _now().isoformat(),
                "marks": self.marks,
                "positions": [asdict(p) for p in self.positions],
            },
        )

    # ── mutation ─────────────────────────────────────────────────────────────

    def open_position(
        self,
        market: str,
        outcome: str,
        model_prob: float,
        price: float = 0.0,
        shares: float = 0.0,
        odds: float = 0.0,
        stake: float = 0.0,
        token_id: str = "",
        venue: str = "polymarket",
        market_kind: str = "match",
        map_name: str = "",
        best_of: int = 3,
        event: str = "",
        note: str = "",
        dry_run: bool = True,
    ) -> Position:
        """
        Accepts either Polymarket terms (price + shares) or sportsbook terms
        (odds + stake), and stores both the same way.
        """
        if odds and stake:
            price = odds_to_price(odds)
            shares = stake * odds
        if not 0.0 < price < 1.0:
            raise ValueError("price must be strictly between 0 and 1")
        if shares <= 0:
            raise ValueError("shares must be positive")
        if not 0.0 < model_prob < 1.0:
            raise ValueError("model_prob must be strictly between 0 and 1")

        cost = shares * price
        available = self.stats().capital - self.stats().at_risk
        if cost > available + 1e-9:
            raise ValueError(
                f"cost {cost:.2f} exceeds available capital {available:.2f} "
                f"(capital minus what is already at risk)"
            )

        p = Position(
            id=uuid.uuid4().hex[:8],
            market=market,
            outcome=outcome,
            price=price,
            shares=shares,
            model_prob=model_prob,
            opened_at=_now().isoformat(),
            token_id=token_id,
            venue=venue,
            market_kind=market_kind,
            map_name=map_name,
            best_of=best_of,
            event=event,
            note=note,
            dry_run=dry_run,
        )
        self.positions.append(p)
        self.save()
        return p

    def settle(self, position_id: str, result: str) -> Position:
        """Resolution at $1 or $0. result: 'won' | 'lost' | 'void'."""
        if result not in ("won", "lost", "void"):
            raise ValueError("result must be won, lost or void")
        p = self.get(position_id)
        if p is None:
            raise KeyError(f"no position with id {position_id!r}")
        if not p.is_open:
            raise ValueError(f"position {position_id} is already {p.status}")

        p.status = result
        p.settled_at = _now().isoformat()
        p.pnl = {"won": p.to_win, "lost": -p.cost, "void": 0.0}[result]
        self.marks.pop(p.id, None)
        self.save()
        return p

    def close(
        self, position_id: str, exit_price: float, shares: Optional[float] = None
    ) -> Position:
        """
        Sell before resolution — the thing a sportsbook cannot do and the
        reason this ledger is not just won/lost/void.

        A partial close splits the position: the sold portion becomes its own
        closed record with realised P&L, and the remainder stays open. Without
        the split, a partial exit would either overstate realised profit or
        silently lose the remaining exposure.
        """
        if not 0.0 <= exit_price <= 1.0:
            raise ValueError("exit_price must be between 0 and 1")
        p = self.get(position_id)
        if p is None:
            raise KeyError(f"no position with id {position_id!r}")
        if not p.is_open:
            raise ValueError(f"position {position_id} is already {p.status}")

        qty = p.shares if shares is None else float(shares)
        if qty <= 0:
            raise ValueError("shares to close must be positive")
        if qty > p.shares + 1e-9:
            raise ValueError(
                f"cannot close {qty} shares; position only holds {p.shares}"
            )

        realised = qty * (exit_price - p.price)

        if qty >= p.shares - 1e-9:
            p.status = "closed"
            p.exit_price = exit_price
            p.settled_at = _now().isoformat()
            p.pnl = realised
            self.marks.pop(p.id, None)
            self.save()
            return p

        p.shares -= qty
        sold = Position(
            id=uuid.uuid4().hex[:8],
            market=p.market,
            outcome=p.outcome,
            price=p.price,
            shares=qty,
            model_prob=p.model_prob,
            opened_at=p.opened_at,
            token_id=p.token_id,
            venue=p.venue,
            market_kind=p.market_kind,
            map_name=p.map_name,
            best_of=p.best_of,
            event=p.event,
            note=f"partial close of {p.id}",
            dry_run=p.dry_run,
            status="closed",
            settled_at=_now().isoformat(),
            exit_price=exit_price,
            pnl=realised,
        )
        self.positions.append(sold)
        self.save()
        return sold

    def mark(self, position_id: str, price: float) -> None:
        """Record the latest market price for an open position."""
        self.marks[position_id] = float(price)

    def mark_to_market(self, prices: Dict[str, float], save: bool = True) -> None:
        """prices: position id -> current market price."""
        for pid, pr in prices.items():
            self.marks[pid] = float(pr)
        if save:
            self.save()

    def get(self, position_id: str) -> Optional[Position]:
        for p in self.positions:
            if p.id == position_id:
                return p
        return None

    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.is_open]

    def settled_positions(self) -> List[Position]:
        return [p for p in self.positions if not p.is_open]

    def mark_for(self, p: Position) -> float:
        """Latest known price, falling back to entry (i.e. no move)."""
        return self.marks.get(p.id, p.price)

    def has_open_on(self, token_id: str) -> bool:
        return any(p.is_open and p.token_id == token_id for p in self.positions if token_id)

    # ── analysis ─────────────────────────────────────────────────────────────

    def equity_curve(self) -> List[float]:
        settled = sorted(self.settled_positions(), key=lambda p: p.settled_at or "")
        cap = self.starting_capital
        curve = [cap]
        for p in settled:
            cap += p.pnl
            curve.append(cap)
        return curve

    def pnl_since(self, since: datetime) -> float:
        total = 0.0
        for p in self.settled_positions():
            if not p.settled_at:
                continue
            try:
                t = datetime.fromisoformat(p.settled_at)
            except ValueError:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            if t >= since:
                total += p.pnl
        return total

    def stats(self) -> Stats:
        settled = self.settled_positions()
        opens = self.open_positions()

        realised = sum(p.pnl for p in settled)
        capital = self.starting_capital + realised
        at_risk = sum(p.cost for p in opens)
        unrealised = sum(p.unrealised_at(self.mark_for(p)) for p in opens)

        wins = sum(1 for p in settled if p.status == "won")
        losses = sum(1 for p in settled if p.status == "lost")
        voids = sum(1 for p in settled if p.status == "void")
        closed = sum(1 for p in settled if p.status == "closed")
        decided = wins + losses

        staked = sum(p.cost for p in settled if p.status != "void")

        curve = self.equity_curve()
        peak = max(curve) if curve else self.starting_capital
        drawdown = peak - capital

        # Was the model right about how sure it was? Only resolved positions
        # count — one closed early never revealed its answer.
        decided_ps = [p for p in settled if p.status in ("won", "lost")]
        if decided_ps:
            mean_pred = sum(p.model_prob for p in decided_ps) / len(decided_ps)
            calib_gap = mean_pred - (wins / len(decided_ps))
            brier = sum(
                (p.model_prob - (1.0 if p.status == "won" else 0.0)) ** 2
                for p in decided_ps
            ) / len(decided_ps)
        else:
            calib_gap = 0.0
            brier = 0.0

        pnls = [p.pnl for p in settled] or [0.0]
        midnight = _now().replace(hour=0, minute=0, second=0, microsecond=0)

        return Stats(
            starting_capital=self.starting_capital,
            capital=capital,
            equity=capital + unrealised,
            realised_pnl=realised,
            unrealised_pnl=unrealised,
            open_count=len(opens),
            at_risk=at_risk,
            exposure_pct=(at_risk / capital * 100.0) if capital > 0 else 0.0,
            wins=wins,
            losses=losses,
            voids=voids,
            closed_early=closed,
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
            pnl_today=self.pnl_since(midnight),
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

    A log tells you what happened; a heartbeat tells you what is happening NOW,
    in one cheap read with no parsing. For an unattended bot holding money,
    "is it still alive" is the first question and it deserves its own answer.
    """

    def __init__(self, path: str = "data/heartbeat.json", stale_after: float = 180.0):
        self.path = path
        self.stale_after = stale_after
        self._counter = 0

    def beat(self, status: str = "ok", note: str = "") -> Beat:
        self._counter += 1
        b = Beat(_now().isoformat(), os.getpid(), status, note, self._counter)
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
