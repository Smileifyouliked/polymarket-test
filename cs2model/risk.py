"""
risk.py — the limits that stand between a bug and your bankroll.

An unattended bot is a loop that spends money while you sleep. The forecaster
decides what looks good; this file decides what is ALLOWED, and it gets the
final word. Every rule here exists because of a specific way automated betting
goes wrong:

  edge/confidence floors   stop it trading noise it has no view on
  price bounds             stop it paying 0.97 for a "certainty"
  per-position cap         stop one bad call mattering
  exposure cap             stop twelve correlated calls mattering
  daily loss limit         stop a bad day becoming a bad week
  spread + depth floors    stop the order book eating the edge
  duplicate guard          stop it buying the same market every tick
  kill switch              stop everything, right now, from any SSH session

Defaults are deliberately timid. A bot that trades too little costs you
opportunity; a bot that trades too much costs you the account.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, Tuple

UTC = timezone.utc


@dataclass
class RiskLimits:
    # --- what is worth betting at all ---
    min_edge: float = 0.05           # model prob must beat price by this much
    min_confidence: float = 0.60     # and be at least this sure of the outcome
    min_price: float = 0.10          # ignore lottery tickets
    max_price: float = 0.90          # and near-certainties: no room, all risk

    # --- how much ---
    kelly_cap: float = 0.25          # fraction of full Kelly
    max_position_pct: float = 0.05   # hard cap per position, as % of capital
    min_position_size: float = 1.0   # skip dust; fees and slippage eat it

    # --- how much at once ---
    max_total_exposure_pct: float = 0.30
    max_open_positions: int = 10
    max_positions_per_market: int = 1

    # --- when to stop ---
    daily_loss_limit_pct: float = 0.10
    max_drawdown_pct: float = 0.25

    # --- market quality ---
    max_spread: float = 0.04         # wider than this and the spread is the trade
    min_book_depth: float = 50.0     # USDC available at the price we want

    # --- the big red button ---
    kill_switch_path: str = "data/STOP"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    allowed: bool
    reason: str
    size: float = 0.0        # approved position cost in USDC

    def __bool__(self) -> bool:
        return self.allowed


class RiskManager:
    """Stateless with respect to the ledger — it reads, it never writes."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

    # ── global halts ─────────────────────────────────────────────────────────

    def kill_switch_engaged(self) -> bool:
        return os.path.exists(self.limits.kill_switch_path)

    def halted(self, ledger) -> Tuple[bool, str]:
        """
        Reasons to stop trading entirely, checked before anything else.
        Returns (halted, why).
        """
        L = self.limits
        if self.kill_switch_engaged():
            return True, f"KILL SWITCH engaged ({L.kill_switch_path} exists)"

        s = ledger.stats()
        if s.capital <= 0:
            return True, "capital exhausted"

        if s.starting_capital > 0:
            day_limit = -abs(L.daily_loss_limit_pct) * s.starting_capital
            if s.pnl_today <= day_limit:
                return True, (
                    f"daily loss limit hit: {s.pnl_today:+,.2f} today vs limit "
                    f"{day_limit:,.2f}. Resumes tomorrow (UTC)."
                )

        if s.drawdown_pct >= L.max_drawdown_pct * 100.0:
            return True, (
                f"max drawdown hit: {s.drawdown_pct:.1f}% off peak vs limit "
                f"{L.max_drawdown_pct * 100:.0f}%. Manual review required."
            )
        return False, ""

    # ── per-opportunity ──────────────────────────────────────────────────────

    def evaluate(
        self,
        ledger,
        model_prob: float,
        price: float,
        token_id: str = "",
        spread: Optional[float] = None,
        depth: Optional[float] = None,
    ) -> Verdict:
        """
        Should we take this, and for how much? `price` is the ASK — what you
        would actually pay, not the midpoint. Sizing off the midpoint is how
        backtests beat live trading.
        """
        L = self.limits
        s = ledger.stats()

        stop, why = self.halted(ledger)
        if stop:
            return Verdict(False, why)

        if not 0.0 < price < 1.0:
            return Verdict(False, f"price {price} outside (0,1)")
        if price < L.min_price or price > L.max_price:
            return Verdict(False,
                           f"price {price:.2f} outside [{L.min_price}, {L.max_price}]")
        if model_prob < L.min_confidence:
            return Verdict(False,
                           f"confidence {model_prob:.1%} below {L.min_confidence:.0%}")

        edge = model_prob - price
        if edge < L.min_edge:
            return Verdict(False, f"edge {edge:+.1%} below {L.min_edge:.0%}")

        if spread is not None and spread > L.max_spread:
            return Verdict(False,
                           f"spread {spread:.3f} wider than {L.max_spread:.3f}")
        if depth is not None and depth < L.min_book_depth:
            return Verdict(False,
                           f"book depth {depth:,.0f} below {L.min_book_depth:,.0f}")

        if token_id and ledger.has_open_on(token_id):
            return Verdict(False, "already holding this market")
        if s.open_count >= L.max_open_positions:
            return Verdict(False,
                           f"at position limit ({L.max_open_positions})")

        # --- sizing ---
        from .tracker import kelly_fraction

        capital = s.capital
        frac = kelly_fraction(model_prob, price, cap=L.kelly_cap)
        size = capital * frac
        size = min(size, capital * L.max_position_pct)

        # Never let a new position push total exposure past the cap.
        room = capital * L.max_total_exposure_pct - s.at_risk
        if room <= 0:
            return Verdict(False,
                           f"exposure cap reached ({s.exposure_pct:.0f}% at risk)")
        size = min(size, room)

        # And never spend money that is already committed.
        size = min(size, capital - s.at_risk)

        if size < L.min_position_size:
            return Verdict(False,
                           f"size {size:,.2f} below minimum {L.min_position_size:,.2f}")

        return Verdict(True, f"edge {edge:+.1%}", size=round(size, 2))


def engage_kill_switch(path: str = "data/STOP", reason: str = "") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{datetime.now(UTC).isoformat()}\n{reason}\n")


def release_kill_switch(path: str = "data/STOP") -> bool:
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
