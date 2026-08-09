"""
dashboard.py — one screen that answers "is it running, and am I up or down?"

Designed for a terminal over SSH: no ports opened, no web server, no browser.
Plain ANSI, degrades to no-colour when piped to a file.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional

from .tracker import Heartbeat, Ledger, Position, Stats, _fmt_age

_USE_COLOUR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t):  return _c(t, "32")
def red(t):    return _c(t, "31")
def yellow(t): return _c(t, "33")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


def money(v: float, width: int = 0, sign: bool = False) -> str:
    s = f"{v:+,.2f}" if sign else f"{v:,.2f}"
    s = s.rjust(width)
    if not sign:
        return s
    return green(s) if v > 0 else red(s) if v < 0 else s


def _bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + dim("░" * (width - filled))


def _rule(width: int, title: str = "") -> str:
    if not title:
        return dim("─" * width)
    head = f"── {title} "
    return dim(head + "─" * max(0, width - len(head)))


def render(
    ledger: Ledger,
    heartbeat: Optional[Heartbeat] = None,
    width: Optional[int] = None,
) -> str:
    w = width or min(shutil.get_terminal_size((88, 24)).columns, 100)
    s = ledger.stats()
    out: List[str] = []

    out.append("")
    out.append(bold("  CS2 MODEL — BANKROLL & STATUS"))
    out.append("  " + _rule(w - 2))

    # ── is it alive ──────────────────────────────────────────────────────────
    if heartbeat is not None:
        beat = heartbeat.read()
        if beat is None:
            state = yellow("○ NOT RUNNING")
            detail = dim("no heartbeat file — start it with: cs2model.cli run")
        elif heartbeat.is_alive():
            state = green("● ALIVE")
            detail = dim(
                f"last beat {_fmt_age(beat.age_seconds)} ago · pid {beat.pid} · "
                f"beat #{beat.counter}"
                + (f" · {beat.note}" if beat.note else "")
            )
        else:
            state = red("✕ STALE")
            detail = red(
                f"last beat {_fmt_age(beat.age_seconds)} ago — expected every "
                f"{_fmt_age(heartbeat.stale_after)}. Process likely dead."
            )
        out.append(f"  {state}  {detail}")
        out.append("  " + _rule(w - 2))

    # ── capital ──────────────────────────────────────────────────────────────
    out.append("  " + _rule(w - 2, "CAPITAL"))
    out.append(f"    starting        {money(s.starting_capital, 12)}")
    out.append(
        f"    current         {money(s.capital, 12)}   "
        + (green("▲") if s.capital > s.starting_capital
           else red("▼") if s.capital < s.starting_capital else dim("="))
    )
    out.append(f"    realised P&L    {money(s.realised_pnl, 12, sign=True)}")
    out.append(f"    peak            {money(s.peak_capital, 12)}")
    dd = f"    drawdown        {money(-s.drawdown, 12, sign=True)}"
    if s.drawdown > 0:
        dd += f"   ({s.drawdown_pct:.1f}% off peak)"
    out.append(dd)
    if s.starting_capital > 0:
        growth = (s.capital / s.starting_capital - 1.0) * 100.0
        out.append(f"    growth          {(green if growth>=0 else red)(f'{growth:+.1f}%'):>12}")
    out.append("")

    # ── holds ────────────────────────────────────────────────────────────────
    out.append("  " + _rule(w - 2, f"OPEN POSITIONS ({s.open_count})"))
    if s.open_count == 0:
        out.append(dim("    nothing at risk right now"))
    else:
        out.append(
            f"    at risk         {money(s.at_risk, 12)}   "
            f"({s.exposure_pct:.1f}% of capital)"
        )
        out.append("")
        out.append(dim(f"    {'id':<9}{'matchup':<34}{'pick':<16}{'stake':>9}{'to win':>10}"))
        for p in ledger.open_positions():
            matchup = f"{p.team_a} vs {p.team_b}"
            if len(matchup) > 32:
                matchup = matchup[:31] + "…"
            pick = p.pick if len(p.pick) <= 14 else p.pick[:13] + "…"
            out.append(
                f"    {p.id:<9}{matchup:<34}{pick:<16}"
                f"{p.stake:>9,.2f}{p.to_win:>10,.2f}"
            )
            out.append(
                dim(f"    {'':<9}Bo{p.best_of}  model {p.model_prob:.0%} vs "
                    f"market {p.implied_prob:.0%}  edge {p.edge:+.1%}  @ {p.odds:.2f}")
            )
    out.append("")

    # ── record ───────────────────────────────────────────────────────────────
    out.append("  " + _rule(w - 2, "RECORD"))
    if s.settled == 0:
        out.append(dim("    no settled positions yet"))
    else:
        out.append(
            f"    {green(str(s.wins) + 'W')} · {red(str(s.losses) + 'L')}"
            + (f" · {dim(str(s.voids) + ' void')}" if s.voids else "")
            + f"    win rate {s.win_rate:.1%}"
        )
        out.append(f"    {_bar(s.win_rate)}  {s.wins}/{s.wins + s.losses}")
        out.append(f"    ROI             {(green if s.roi>=0 else red)(f'{s.roi:+.1f}%'):>12}   "
                   + dim("(profit per unit staked)"))
        out.append(f"    best / worst    {money(s.biggest_win, 12, sign=True)}"
                   f" / {money(s.biggest_loss, 0, sign=True)}")
    out.append("")

    # ── is the model still honest ────────────────────────────────────────────
    out.append("  " + _rule(w - 2, "MODEL HEALTH"))
    decided = s.wins + s.losses
    # A calibration verdict off three bets is noise wearing a lab coat. Show
    # the number, but refuse to interpret it until the sample can carry one.
    MIN_FOR_VERDICT = 20
    if s.settled == 0:
        out.append(dim("    needs settled positions before this means anything"))
    else:
        gap = s.calibration_gap
        if decided < MIN_FOR_VERDICT:
            verdict = dim(f"too few bets to judge ({decided}/{MIN_FOR_VERDICT})")
        elif abs(gap) < 0.05:
            verdict = green("calibrated")
        elif gap > 0:
            verdict = red("OVER-CONFIDENT")
        else:
            verdict = yellow("under-confident")
        out.append(
            f"    calibration     {gap:+.1%} {verdict}   "
            + dim("(predicted minus actual win rate)")
        )
        out.append(f"    Brier score     {s.brier:>12.4f}   " + dim("(lower is better; 0.25 = coinflip)"))
        out.append(f"    average edge    {s.avg_edge:>+11.1%}   " + dim("(model prob minus market prob)"))
        if gap > 0.10 and decided >= MIN_FOR_VERDICT:
            out.append("")
            out.append(red("    ⚠ The model is claiming more certainty than it delivers."))
            out.append(red("      Raise your confidence threshold or re-run evaluate."))
    out.append("")
    return "\n".join(out)


def render_compact(ledger: Ledger, heartbeat: Optional[Heartbeat] = None) -> str:
    """One line, for a cron log or a status prompt."""
    s = ledger.stats()
    alive = "?" if heartbeat is None else ("up" if heartbeat.is_alive() else "DOWN")
    return (
        f"[{alive}] capital {s.capital:,.2f} "
        f"pnl {s.realised_pnl:+,.2f} "
        f"{s.wins}W-{s.losses}L ({s.win_rate:.0%}) "
        f"open {s.open_count} (risk {s.at_risk:,.2f})"
    )
