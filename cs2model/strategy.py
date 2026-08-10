"""
strategy.py — the loop that turns a forecast into a position.

For each live market:
    1. Identify the teams and match them to teams the model knows.
    2. Ask the model for a probability.
    3. Read the ASK — the price you would actually pay.
    4. edge = model probability - ask.
    5. Hand it to the risk manager, which has the final word.
    6. Buy, or record precisely why not.

Step 6 is why every skip carries a reason string. An automated system that
silently does nothing is indistinguishable from one that is broken, and you
will not find out which until you check the P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .data import ACTIVE_DUTY, Match
from .model import CS2Model, build_features
from .polymarket import BookTop, MarketInfo, PolymarketVenue, normalise_team
from .ratings import RatingBook
from .risk import RiskManager, Verdict
from .tracker import Ledger

UTC = timezone.utc


@dataclass
class Decision:
    market: MarketInfo
    outcome_name: str = ""
    token_id: str = ""
    model_prob: float = 0.0
    ask: float = 0.0
    edge: float = 0.0
    size_usd: float = 0.0
    shares: float = 0.0
    taken: bool = False
    reason: str = ""

    def line(self) -> str:
        head = f"{self.market.question[:52]:<52}"
        if not self.outcome_name:
            return f"  skip  {head} {self.reason}"
        tag = "BUY " if self.taken else "skip"
        return (
            f"  {tag}  {head} {self.outcome_name[:14]:<14} "
            f"model {self.model_prob:.2f} ask {self.ask:.2f} "
            f"edge {self.edge:+.1%}  {self.reason}"
        )


class Strategy:
    def __init__(
        self,
        book: RatingBook,
        model: CS2Model,
        ledger: Ledger,
        venue: PolymarketVenue,
        risk: RiskManager,
    ):
        self.book = book
        self.model = model
        self.ledger = ledger
        self.venue = venue
        self.risk = risk
        self._known = {normalise_team(t): t for t in book.teams}

    # ── team resolution ──────────────────────────────────────────────────────

    def resolve_team(self, name: str) -> Optional[str]:
        """
        Map a Polymarket team name onto a team the ratings know.

        Exact normalised match only. Fuzzy matching is tempting and dangerous:
        "Spirit" and "Spirit Academy" are different teams with very different
        strengths, and a near-miss here means confidently betting the wrong
        side. Unmatched markets are skipped and reported.
        """
        if not name:
            return None
        return self._known.get(normalise_team(name))

    # ── probability ──────────────────────────────────────────────────────────

    def probability(
        self, team_a: str, team_b: str, best_of: int = 3,
        lan: bool = False, map_name: str = "",
    ) -> float:
        """P(team_a wins). For a map market, the per-map rating answers directly."""
        now = datetime.now(UTC)
        if map_name:
            return self.book.map_win_prob(team_a, team_b, map_name, now)

        fixture = Match(
            team_a=team_a, team_b=team_b, date=now, best_of=best_of, lan=lan,
            lineup_a=self.book.team(team_a).lineup,
            lineup_b=self.book.team(team_b).lineup,
        )
        X = build_features(self.book, fixture).reshape(1, -1)
        return float(self.model.predict_proba(X)[0])

    # ── one market ───────────────────────────────────────────────────────────

    def evaluate_market(self, m: MarketInfo) -> Decision:
        d = Decision(market=m)

        if m.closed or not m.active:
            d.reason = "market closed"
            return d
        if not m.team_a or not m.team_b:
            d.reason = "could not parse teams from the question"
            return d
        if m.market_kind == "map" and m.map_name not in ACTIVE_DUTY:
            d.reason = f"map {m.map_name or '?'} not in the active pool"
            return d

        ta, tb = self.resolve_team(m.team_a), self.resolve_team(m.team_b)
        if ta is None or tb is None:
            missing = m.team_a if ta is None else m.team_b
            d.reason = f"unknown team {missing!r} — not in the match history"
            return d

        p_a = self.probability(ta, tb, map_name=m.map_name if m.market_kind == "map" else "")

        # Consider both sides; the value may be on the underdog.
        best: Optional[Tuple[float, float, str, str, BookTop]] = None
        for name, prob in ((m.team_a, p_a), (m.team_b, 1.0 - p_a)):
            o = m.outcome_for(name)
            if o is None or not o.token_id:
                continue
            try:
                top = self.venue.get_book_top(o.token_id)
            except Exception as e:
                d.reason = f"order book unavailable: {e}"
                continue
            if top.best_ask <= 0 or top.best_ask >= 1:
                continue
            edge = prob - top.best_ask
            if best is None or edge > best[0]:
                best = (edge, prob, name, o.token_id, top)

        if best is None:
            d.reason = d.reason or "no tradeable outcome"
            return d

        edge, prob, name, token_id, top = best
        d.outcome_name, d.token_id = name, token_id
        d.model_prob, d.ask, d.edge = prob, top.best_ask, edge

        verdict = self.risk.evaluate(
            self.ledger, model_prob=prob, price=top.best_ask,
            token_id=token_id, spread=top.spread, depth=top.ask_depth_usd,
        )
        if not verdict:
            d.reason = verdict.reason
            return d

        shares = verdict.size / top.best_ask
        # Never ask for more than is actually resting at the ask.
        if top.ask_size > 0:
            shares = min(shares, top.ask_size)
        cost = shares * top.best_ask
        if cost < self.risk.limits.min_position_size:
            d.reason = f"only {top.ask_size:,.0f} shares at the ask — too thin"
            return d

        d.size_usd, d.shares = cost, shares
        d.reason = verdict.reason
        d.taken = True
        return d

    # ── the pass ─────────────────────────────────────────────────────────────

    def run_once(self, markets: Sequence[MarketInfo]) -> List[Decision]:
        """Evaluate every market and act on the ones that clear risk."""
        decisions: List[Decision] = []

        stop, why = self.risk.halted(self.ledger)
        if stop:
            return [Decision(market=MarketInfo("", "", ""), reason=f"HALTED: {why}")]

        for m in markets:
            d = self.evaluate_market(m)
            if d.taken:
                try:
                    self.venue.buy(d.token_id, d.ask, d.shares)
                except Exception as e:
                    d.taken = False
                    d.reason = f"order failed: {e}"
                    decisions.append(d)
                    continue

                self.ledger.open_position(
                    market=m.question,
                    outcome=d.outcome_name,
                    model_prob=d.model_prob,
                    price=d.ask,
                    shares=d.shares,
                    token_id=d.token_id,
                    market_kind=m.market_kind,
                    map_name=m.map_name,
                    event=m.slug,
                    dry_run=self.venue.dry_run,
                    note=d.reason,
                )
            decisions.append(d)

            # Re-check halts between orders: a position opened this pass can
            # push exposure over the cap, and the next order must see that.
            stop, why = self.risk.halted(self.ledger)
            if stop:
                decisions.append(Decision(market=MarketInfo("", "", ""),
                                          reason=f"HALTED mid-pass: {why}"))
                break

        return decisions

    # ── marking and settlement ───────────────────────────────────────────────

    def mark_open_positions(self) -> int:
        """Refresh the market value of everything we hold."""
        marked = 0
        for p in self.ledger.open_positions():
            if not p.token_id:
                continue
            try:
                top = self.venue.get_book_top(p.token_id)
            except Exception:
                continue
            if top.best_bid > 0:
                # Mark at the BID — what you could actually sell for, not the
                # midpoint. Marking at mid flatters every open position.
                self.ledger.mark(p.id, top.best_bid)
                marked += 1
        if marked:
            self.ledger.save()
        return marked

    def settle_resolved(self, markets: Sequence[MarketInfo]) -> List[str]:
        """
        Settle positions whose market has closed.

        A resolved Polymarket outcome trades at essentially 1 or 0, so the
        final price is the result. Anything ambiguous is left open for a human
        rather than guessed at.
        """
        by_token: Dict[str, Tuple[MarketInfo, float]] = {}
        for m in markets:
            for o in m.outcomes:
                by_token[o.token_id] = (m, o.price)

        settled: List[str] = []
        for p in self.ledger.open_positions():
            entry = by_token.get(p.token_id)
            if entry is None:
                continue
            m, final_price = entry
            if not m.closed:
                continue
            if final_price >= 0.99:
                self.ledger.settle(p.id, "won")
                settled.append(f"{p.id} {p.outcome} WON  {p.to_win:+,.2f}")
            elif final_price <= 0.01:
                self.ledger.settle(p.id, "lost")
                settled.append(f"{p.id} {p.outcome} LOST {-p.cost:+,.2f}")
            # Anything in between is unresolved or disputed — leave it alone.
        return settled
