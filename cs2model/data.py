"""
data.py — the shapes everything else speaks in, plus a synthetic league.

THE HONEST FRAMING:
  A CS2 "match" is not one event. It is a veto followed by 1, 3, or 5 separate
  games of Counter-Strike. Modelling the match as a single coin flip throws
  away the single most predictive thing on the table: WHICH MAPS GET PLAYED.
  So the atom in this codebase is a MAP, not a match.

The synthetic league exists for one reason: with real data you never know the
true probability of anything, so you cannot tell a well-calibrated model from
a lucky one. Here we GENERATE from known truth, which means the test suite can
assert that the pipeline recovers it. If the math is wrong, tests fail loudly
instead of quietly producing a confident, wrong model.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

# Valve rotates the active duty pool. This is a CONFIG, not a constant — when
# a map is added or dropped, historical data on the dropped map becomes partly
# dead weight and the model should stop quoting it.
ACTIVE_DUTY: Tuple[str, ...] = (
    "Ancient",
    "Anubis",
    "Dust2",
    "Inferno",
    "Mirage",
    "Nuke",
    "Train",
)

# The pool actually in play. ACTIVE_DUTY above is today's default, but a
# historical dataset has its own pool (2015 Counter-Strike had Cobblestone and
# Cache and no Ancient/Anubis), so the loader sets this from the data. Read it
# through active_pool() rather than importing ACTIVE_DUTY directly, otherwise
# a module captures the default at import time and never sees the change.
_ACTIVE_POOL: List[str] = list(ACTIVE_DUTY)


def active_pool() -> Tuple[str, ...]:
    return tuple(_ACTIVE_POOL)


def set_active_pool(maps: Sequence[str], allow_degenerate: bool = False) -> Tuple[str, ...]:
    """
    Replace the pool in play. A real veto needs at least 7 maps, because the
    scripts ban six and play what is left.

    `allow_degenerate` exists for datasets that record only who won the match
    and never which maps were played. There is no veto to simulate there, so
    the model falls back to rating teams overall — see has_map_detail().
    """
    maps = [str(m) for m in maps]
    if len(maps) < 7 and not allow_degenerate:
        raise ValueError(
            f"a veto needs at least 7 maps, got {len(maps)}: {maps}. "
            f"Pass allow_degenerate=True for match-level-only data."
        )
    _ACTIVE_POOL[:] = maps
    return tuple(_ACTIVE_POOL)


def has_map_detail() -> bool:
    """
    False when the loaded data has no per-map results.

    This is not a detail — it decides whether the central idea of this project
    (rate teams per map, simulate the veto) is usable at all. With match-level
    data only, the model degrades to plain Glicko and the veto layer is
    switched off rather than fed nonsense.
    """
    return len(_ACTIVE_POOL) >= 7


# Veto scripts. Index = whose turn, action = ban/pick. The team listed first in
# the match is "A". Standard competitive formats.
VETO_SCRIPTS: Dict[int, Sequence[Tuple[str, str]]] = {
    1: (("A", "ban"), ("B", "ban"), ("A", "ban"), ("B", "ban"), ("A", "ban"), ("B", "ban")),
    3: (("A", "ban"), ("B", "ban"), ("A", "pick"), ("B", "pick"), ("B", "ban"), ("A", "ban")),
    5: (("A", "ban"), ("B", "ban"), ("A", "pick"), ("B", "pick"), ("A", "pick"), ("B", "pick")),
}


@dataclass
class MapResult:
    """One map of Counter-Strike. The unit of learning."""

    map_name: str
    winner: str  # team id
    loser: str
    rounds_winner: int = 13
    rounds_loser: int = 0
    # Who chose this map: "A", "B", or "decider". Pick advantage is real and
    # the model gets to see it.
    picked_by: str = "decider"


@dataclass
class Match:
    team_a: str
    team_b: str
    date: datetime
    best_of: int
    maps: List[MapResult] = field(default_factory=list)
    winner: Optional[str] = None

    # False when the source could not tell us every map that was played.
    # Partial map records are not merely incomplete, they are BIASED: a
    # winner_map/loser_map/decider_map layout records all three maps of a 2-1
    # but only one map of a 2-0, and the ones it drops are the winner's. Per-map
    # ratings fed that diet systematically under-credit strong teams on their
    # best maps. Such matches still teach the overall rating.
    maps_complete: bool = True

    # Context that moves the needle in CS specifically.
    lan: bool = False
    tier: int = 2  # 1 = top international, 3 = regional filler
    event: str = ""

    # Lineups matter more than team names — orgs swap players constantly and a
    # team-level rating silently goes stale the moment a star leaves.
    lineup_a: Tuple[str, ...] = ()
    lineup_b: Tuple[str, ...] = ()

    # Only present in synthetic data. The thing you can never observe in real
    # life, and the reason the test suite can prove the pipeline works.
    true_p_a: Optional[float] = None

    # Optional market price (decimal odds implied prob) if you have it. The
    # closing line is the real scoreboard for any forecasting model.
    closing_prob_a: Optional[float] = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d

    @staticmethod
    def from_json(d: dict) -> "Match":
        maps = [MapResult(**m) for m in d.get("maps", [])]
        return Match(
            team_a=d["team_a"],
            team_b=d["team_b"],
            date=datetime.fromisoformat(d["date"]),
            best_of=d["best_of"],
            maps=maps,
            winner=d.get("winner"),
            maps_complete=d.get("maps_complete", True),
            lan=d.get("lan", False),
            tier=d.get("tier", 2),
            event=d.get("event", ""),
            lineup_a=tuple(d.get("lineup_a", ())),
            lineup_b=tuple(d.get("lineup_b", ())),
            true_p_a=d.get("true_p_a"),
            closing_prob_a=d.get("closing_prob_a"),
        )


def save_matches(matches: Sequence[Match], path: str) -> None:
    with open(path, "w") as f:
        json.dump([m.to_json() for m in matches], f)


def load_matches(path: str) -> List[Match]:
    with open(path) as f:
        return [Match.from_json(d) for d in json.load(f)]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC LEAGUE
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _TrueTeam:
    """Ground truth. The model never sees any of this."""

    tid: str
    skill: float  # overall latent strength, logit scale
    map_delta: Dict[str, float]  # per-map deviation from overall
    lan_delta: float  # some rosters are LAN monsters, some online merchants
    roster: Tuple[str, ...]

    def map_strength(self, map_name: str, lan: bool) -> float:
        s = self.skill + self.map_delta.get(map_name, 0.0)
        if lan:
            s += self.lan_delta
        return s


def _true_map_prob(a: _TrueTeam, b: _TrueTeam, map_name: str, lan: bool) -> float:
    """Bradley-Terry on the latent scale. This IS the answer, by construction."""
    diff = a.map_strength(map_name, lan) - b.map_strength(map_name, lan)
    return 1.0 / (1.0 + math.exp(-diff))


def _true_veto(
    a: _TrueTeam, b: _TrueTeam, best_of: int, lan: bool, rng: random.Random
) -> List[Tuple[str, str]]:
    """
    Teams ban where they are weakest and pick where they are strongest,
    relative to the opponent. Not perfectly rational — a temperature keeps it
    stochastic, because real vetoes are not deterministic either.
    """
    pool = list(ACTIVE_DUTY)
    chosen: List[Tuple[str, str]] = []  # (map, picked_by)
    temp = 0.6

    for actor, action in VETO_SCRIPTS[best_of]:
        me, them = (a, b) if actor == "A" else (b, a)
        # Edge on each remaining map from the acting team's point of view, on
        # the latent strength scale (log-odds). See the note in veto.py — using
        # probability units here would make the veto almost random.
        edges = [
            me.map_strength(m, lan) - them.map_strength(m, lan) for m in pool
        ]
        # Ban the worst, pick the best. Softmax over the appropriate direction.
        signs = [-e if action == "ban" else e for e in edges]
        mx = max(signs)
        weights = [math.exp((s - mx) / temp) for s in signs]
        total = sum(weights)
        r = rng.random() * total
        idx = 0
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                idx = i
                break
        m = pool.pop(idx)
        if action == "pick":
            chosen.append((m, actor))

    # Whatever survives the bans is the decider.
    for m in pool:
        chosen.append((m, "decider"))
    return chosen[:best_of]


def generate_synthetic_league(
    n_teams: int = 40,
    n_matches: int = 4000,
    seed: int = 7,
    roster_change_rate: float = 0.02,
    start: Optional[datetime] = None,
    skill_sd: float = 0.75,
    map_delta_sd: float = 0.40,
    pair_width: float = 1.5,
) -> List[Match]:
    """
    A fake but structurally honest CS2 scene: teams with per-map identities,
    LAN/online splits, roster churn that genuinely changes strength, and a
    veto that responds to all of it.

    Every Match carries `true_p_a`, so evaluation can compare the model against
    the real answer instead of against noise.
    """
    rng = random.Random(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Skill spread is deliberately modest. A wide spread makes the synthetic
    # scene far more predictable than real Counter-Strike and produces
    # flattering accuracy numbers that mean nothing.
    teams: Dict[str, _TrueTeam] = {}
    for i in range(n_teams):
        tid = f"team_{i:02d}"
        teams[tid] = _TrueTeam(
            tid=tid,
            skill=rng.gauss(0.0, skill_sd),
            map_delta={m: rng.gauss(0.0, map_delta_sd) for m in ACTIVE_DUTY},
            lan_delta=rng.gauss(0.0, 0.25),
            roster=tuple(f"p_{i:02d}_{j}" for j in range(5)),
        )

    matches: List[Match] = []
    player_counter = n_teams * 5

    def maybe_churn(t: _TrueTeam) -> None:
        """
        A swap moves true strength — this is exactly the event that a naive
        team-level rating fails to notice.

        `roster_change_rate` is PER MATCH THE TEAM PLAYS, not per tick of the
        global clock. That distinction is not pedantic: rolling it for every
        team on every match in the league gave each team a roster change every
        ~2.8 matches, which turns true skill into a random walk and makes the
        entire history uninformative. No model can beat a coinflip in a scene
        like that, and it is nothing like real Counter-Strike, where teams
        change one or two players a year across ~60-80 matches.
        """
        if rng.random() >= roster_change_rate:
            return
        nonlocal player_counter
        out = rng.randrange(5)
        new_roster = list(t.roster)
        new_roster[out] = f"p_new_{player_counter}"
        player_counter += 1
        t.roster = tuple(new_roster)
        t.skill += rng.gauss(0.0, 0.25)
        for m in ACTIVE_DUTY:
            t.map_delta[m] += rng.gauss(0.0, 0.10)

    for n in range(n_matches):
        date = start + timedelta(hours=6 * n)

        # Seeded pairing, not random pairing. Real tournaments put similar
        # teams in the same groups and bracket, so the slate you actually get
        # to predict is much closer to even than a uniform draw would be.
        # Random pairing across the whole field manufactures blowouts and
        # inflates accuracy for free.
        a_id = rng.choice(list(teams))
        a = teams[a_id]
        others = [t for t in teams if t != a_id]
        weights = [math.exp(-abs(a.skill - teams[t].skill) / pair_width) for t in others]
        b_id = rng.choices(others, weights=weights)[0]
        b = teams[b_id]

        maybe_churn(a)
        maybe_churn(b)

        lan = rng.random() < 0.35
        best_of = rng.choices([1, 3, 5], weights=[0.45, 0.5, 0.05])[0]
        tier = rng.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]

        veto = _true_veto(a, b, best_of, lan, rng)

        # Play it out, map by map, stopping when the series is decided.
        need = best_of // 2 + 1
        wins_a = wins_b = 0
        map_results: List[MapResult] = []
        map_probs: List[float] = []

        for map_name, picked_by in veto:
            p = _true_map_prob(a, b, map_name, lan)
            map_probs.append(p)
            if wins_a >= need or wins_b >= need:
                continue
            a_wins = rng.random() < p
            w, l = (a_id, b_id) if a_wins else (b_id, a_id)
            # Round score, loosely tied to how lopsided the map was.
            edge = abs(p - 0.5)
            loser_rounds = max(0, min(11, int(rng.gauss(9 - 12 * edge, 2.6))))
            map_results.append(
                MapResult(
                    map_name=map_name,
                    winner=w,
                    loser=l,
                    rounds_winner=13,
                    rounds_loser=loser_rounds,
                    picked_by=picked_by,
                )
            )
            if a_wins:
                wins_a += 1
            else:
                wins_b += 1

        winner = a_id if wins_a > wins_b else b_id
        matches.append(
            Match(
                team_a=a_id,
                team_b=b_id,
                date=date,
                best_of=best_of,
                maps=map_results,
                winner=winner,
                lan=lan,
                tier=tier,
                event=f"synthetic_t{tier}",
                lineup_a=a.roster,
                lineup_b=b.roster,
                true_p_a=series_prob(map_probs, best_of),
            )
        )

    return matches


def series_prob(map_probs: Sequence[float], best_of: int) -> float:
    """
    P(A wins the series) given P(A wins each scheduled map, in order).

    Exact, not simulated. For a Bo3 with map probabilities p1, p2, p3 the team
    wins by taking the first two, or by splitting and taking the decider:

        p1*p2 + p1*(1-p2)*p3 + (1-p1)*p2*p3

    Generalised below to any best_of via a small dynamic program over
    (maps played, maps won).
    """
    need = best_of // 2 + 1
    probs = list(map_probs[:best_of])
    while len(probs) < best_of:
        probs.append(0.5)

    # state[w] = P(A has exactly w wins) among decided-so-far series
    state = {0: 1.0}
    finished = 0.0
    for i, p in enumerate(probs):
        nxt: Dict[int, float] = {}
        for w, prob in state.items():
            losses = i - w
            if w >= need or losses >= need:
                continue  # already decided; folded into `finished` below
            nxt[w + 1] = nxt.get(w + 1, 0.0) + prob * p
            nxt[w] = nxt.get(w, 0.0) + prob * (1.0 - p)
        # harvest completed branches
        for w, prob in list(nxt.items()):
            if w >= need:
                finished += prob
                del nxt[w]
        state = nxt
    return finished
