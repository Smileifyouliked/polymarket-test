"""
polymarket.py — the venue. Discovery, order books, and placing orders.

READ THIS BEFORE GOING LIVE
  I could not run any of this against Polymarket. The sandbox it was written
  in blocks the hosts. The method signatures come from inspecting the real
  installed py-clob-client, so the shapes are right, but no request here has
  ever been sent.

  Therefore: DRY RUN IS THE DEFAULT AND LIVE TRADING REQUIRES AN EXPLICIT
  FLAG. Dry run does everything real trading does — discovery, book reads,
  sizing, risk checks, ledger writes — and stops at the final step of signing
  an order. Run it that way for days before you consider anything else.

KEYS
  Signing orders needs your Polygon private key on the server. Consequences:
    * Use a DEDICATED wallet holding only what you are prepared to lose.
      Not your main wallet. There is no undo on a compromised key.
    * The key goes in an environment variable, never a file in this repo.
    * Anyone with root on this box can read it.
  These are properties of self-custody, not of this code.

PRICES ARE PROBABILITIES
  A share costs between $0 and $1 and pays $1 if it resolves YES. The price is
  the market's probability. No odds conversion, no vig to strip — you compare
  it to the model's number directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import requests

GAMMA_API = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
CLOB_HOST = os.getenv("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
POLYGON_CHAIN_ID = 137


class PolymarketError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# MARKET SHAPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Outcome:
    name: str
    token_id: str
    price: float = 0.0


@dataclass
class MarketInfo:
    """One tradeable market: usually two outcomes that sum to about 1."""

    question: str
    slug: str
    condition_id: str
    outcomes: List[Outcome] = field(default_factory=list)
    end_date: str = ""
    active: bool = True
    closed: bool = False
    volume: float = 0.0
    liquidity: float = 0.0

    # Filled in by parsing the question text.
    team_a: str = ""
    team_b: str = ""
    market_kind: str = "match"   # "match" | "map"
    map_name: str = ""

    def outcome_for(self, name: str) -> Optional[Outcome]:
        low = name.lower().strip()
        for o in self.outcomes:
            if o.name.lower().strip() == low:
                return o
        return None


@dataclass
class BookTop:
    """The top of the order book — what you would actually pay or receive."""

    best_bid: float = 0.0
    best_ask: float = 1.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def spread(self) -> float:
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def ask_depth_usd(self) -> float:
        """Money available at the ask — what you can actually buy."""
        return self.best_ask * self.ask_size


# ─────────────────────────────────────────────────────────────────────────────
# TEAM / MARKET PARSING
# ─────────────────────────────────────────────────────────────────────────────

_MAP_NAMES = ("Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Train")


def parse_question(question: str) -> Tuple[str, str, str, str]:
    """
    Pull (team_a, team_b, kind, map_name) out of a market question.

    Polymarket phrasing varies ("Vitality vs. Spirit", "Will Vitality beat
    Spirit?", "Vitality vs Spirit - Map 2: Nuke"), so this is best-effort and
    returns empty strings when unsure. The bot SKIPS anything it cannot parse
    confidently rather than guessing — a mis-parsed market means betting on
    the wrong team, which is worse than not betting at all.
    """
    q = question.strip()
    kind, map_name = "match", ""

    low = q.lower()
    for m in _MAP_NAMES:
        if m.lower() in low:
            map_name, kind = m, "map"
            break
    if "map " in low and kind == "match":
        kind = "map"

    head = q.split(" - ")[0].split(":")[0]
    head = head.replace("Will ", "").replace("?", "").strip()

    for sep in (" vs. ", " vs ", " v. ", " beat ", " defeat "):
        if sep in head:
            a, b = head.split(sep, 1)
            return a.strip(), b.strip(), kind, map_name
    return "", "", kind, map_name


def normalise_team(name: str) -> str:
    """Loose key for matching Polymarket names against Liquipedia names."""
    n = name.lower().strip()
    for junk in (" esports", " esport", " gaming", " team ", "team ", ".", "-", "_"):
        n = n.replace(junk, " ")
    return " ".join(n.split())


# ─────────────────────────────────────────────────────────────────────────────
# VENUE
# ─────────────────────────────────────────────────────────────────────────────


class PolymarketVenue:
    """
    Read side works without any credentials. Order placement needs a key and
    is refused unless dry_run is explicitly turned off.
    """

    def __init__(
        self,
        dry_run: bool = True,
        private_key: Optional[str] = None,
        funder: Optional[str] = None,
        host: str = CLOB_HOST,
        request_pause: float = 0.25,
    ):
        self.dry_run = dry_run
        self.host = host
        self.request_pause = request_pause
        self._client = None
        self._private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self._funder = funder or os.getenv("POLYMARKET_FUNDER_ADDRESS")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "cs2model/0.1"})

    # ── discovery (public, no key needed) ────────────────────────────────────

    def fetch_markets(
        self, slug_contains: str = "cs2", limit: int = 200, closed: bool = False
    ) -> List[MarketInfo]:
        """
        Discover CS2 markets via the public Gamma API.

        Never verified against the live service — if the shape differs, fix the
        parsing in _parse_gamma_market(). `cs2model.cli markets --raw` dumps
        exactly what came back so you can see it.
        """
        url = f"{GAMMA_API}/markets"
        params = {
            "closed": str(closed).lower(),
            "limit": limit,
            "order": "volume",
            "ascending": "false",
        }
        if slug_contains:
            params["slug"] = slug_contains
        try:
            r = self.session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            raise PolymarketError(f"could not reach Gamma API: {e}") from e
        if r.status_code != 200:
            raise PolymarketError(f"Gamma API HTTP {r.status_code}: {r.text[:300]}")

        payload = r.json()
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        out: List[MarketInfo] = []
        for row in rows:
            m = self._parse_gamma_market(row)
            if m is not None:
                out.append(m)
        return out

    @staticmethod
    def _parse_gamma_market(row: dict) -> Optional[MarketInfo]:
        import json as _json

        def _maybe_list(v):
            if isinstance(v, list):
                return v
            if isinstance(v, str):
                try:
                    parsed = _json.loads(v)
                    return parsed if isinstance(parsed, list) else []
                except _json.JSONDecodeError:
                    return []
            return []

        question = (row.get("question") or row.get("title") or "").strip()
        if not question:
            return None

        names = _maybe_list(row.get("outcomes"))
        tokens = _maybe_list(row.get("clobTokenIds"))
        prices = _maybe_list(row.get("outcomePrices"))
        if len(names) < 2 or len(tokens) < 2:
            return None

        outcomes = []
        for i, nm in enumerate(names):
            try:
                px = float(prices[i]) if i < len(prices) else 0.0
            except (TypeError, ValueError):
                px = 0.0
            outcomes.append(Outcome(name=str(nm), token_id=str(tokens[i]), price=px))

        a, b, kind, map_name = parse_question(question)

        def _f(key) -> float:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return MarketInfo(
            question=question,
            slug=row.get("slug", ""),
            condition_id=row.get("conditionId", ""),
            outcomes=outcomes,
            end_date=row.get("endDate", "") or "",
            active=bool(row.get("active", True)),
            closed=bool(row.get("closed", False)),
            volume=_f("volume"),
            liquidity=_f("liquidity"),
            team_a=a,
            team_b=b,
            market_kind=kind,
            map_name=map_name,
        )

    # ── order book ───────────────────────────────────────────────────────────

    def get_book_top(self, token_id: str) -> BookTop:
        """
        Best bid/ask and the size sitting there. Read-only, no key required.

        The bot sizes off the ASK, not the midpoint: the midpoint is a price
        nobody is offering, and sizing off it is how a backtest quietly beats
        the live account.
        """
        try:
            book = self.client.get_order_book(token_id)
        except Exception as e:  # the client raises a variety of types
            raise PolymarketError(f"order book fetch failed: {e}") from e

        def _best(levels, reverse: bool):
            best_p, best_s = (0.0, 0.0) if reverse else (1.0, 0.0)
            rows = []
            for lv in levels or []:
                try:
                    rows.append((float(lv.price), float(lv.size)))
                except (AttributeError, TypeError, ValueError):
                    continue
            if not rows:
                return best_p, best_s
            rows.sort(key=lambda x: x[0], reverse=reverse)
            return rows[0]

        bid_p, bid_s = _best(getattr(book, "bids", None), reverse=True)
        ask_p, ask_s = _best(getattr(book, "asks", None), reverse=False)
        return BookTop(best_bid=bid_p, best_ask=ask_p, bid_size=bid_s, ask_size=ask_s)

    # ── client ───────────────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is not None:
            return self._client
        try:
            from py_clob_client.client import ClobClient
        except ImportError as e:
            raise PolymarketError(
                "py-clob-client is not installed. pip install py-clob-client"
            ) from e

        if self._private_key:
            self._client = ClobClient(
                self.host,
                chain_id=POLYGON_CHAIN_ID,
                key=self._private_key,
                funder=self._funder,
                signature_type=1 if self._funder else None,
            )
            try:
                self._client.set_api_creds(self._client.create_or_derive_api_creds())
            except Exception as e:
                raise PolymarketError(
                    f"could not derive API credentials from the private key: {e}"
                ) from e
        else:
            # Read-only: books and prices work, orders do not.
            self._client = ClobClient(self.host, chain_id=POLYGON_CHAIN_ID)
        return self._client

    # ── orders ───────────────────────────────────────────────────────────────

    def buy(self, token_id: str, price: float, shares: float) -> dict:
        """
        Buy `shares` at a limit of `price`.

        A LIMIT order, deliberately — a market order on a thin CS2 book can
        fill many cents worse than the screen price, and that slippage comes
        straight out of the edge the model spent all its effort finding.
        """
        if shares <= 0:
            raise ValueError("shares must be positive")
        if not 0.0 < price < 1.0:
            raise ValueError("price must be between 0 and 1")

        if self.dry_run:
            return {
                "dry_run": True,
                "status": "simulated",
                "token_id": token_id,
                "price": price,
                "size": shares,
                "cost": round(price * shares, 4),
            }

        if not self._private_key:
            raise PolymarketError(
                "live trading requires POLYMARKET_PRIVATE_KEY. Use a dedicated "
                "wallet funded with only what you can afford to lose."
            )

        from py_clob_client.clob_types import OrderArgs, OrderType

        args = OrderArgs(
            token_id=token_id, price=round(float(price), 3),
            size=float(shares), side="BUY",
        )
        try:
            resp = self.client.create_and_post_order(args)
        except Exception as e:
            raise PolymarketError(f"order rejected: {e}") from e
        time.sleep(self.request_pause)
        return {"dry_run": False, "status": "submitted", "response": resp}

    def sell(self, token_id: str, price: float, shares: float) -> dict:
        if self.dry_run:
            return {"dry_run": True, "status": "simulated", "token_id": token_id,
                    "price": price, "size": shares}
        if not self._private_key:
            raise PolymarketError("live trading requires POLYMARKET_PRIVATE_KEY")

        from py_clob_client.clob_types import OrderArgs

        args = OrderArgs(token_id=token_id, price=round(float(price), 3),
                         size=float(shares), side="SELL")
        try:
            resp = self.client.create_and_post_order(args)
        except Exception as e:
            raise PolymarketError(f"order rejected: {e}") from e
        return {"dry_run": False, "status": "submitted", "response": resp}

    def describe_mode(self) -> str:
        if self.dry_run:
            return "DRY RUN — no orders will be sent"
        who = "with funder" if self._funder else "direct key"
        return f"LIVE — real money, real orders ({who})"
