# CLAUDE.md

Read this first. It exists so a new session does not re-derive what has already
been tested, and does not re-propose ideas that have already failed.

## What this repo is

Two unrelated projects sharing a directory:

- **`quote_sim.py`** — a paper simulator for two-sided quoting on Polymarket's
  BTC up/down markets. Standalone, predates everything else, untouched.
- **`cs2model/`** — a CS2 match forecaster and a Polymarket trading bot.
  Everything below is about this.

## Current verdict: the model does not work

Tested on two independent real datasets. This is settled, not a hunch.

On 7,033 real CS2 matches (Kaggle HLTV, May 2024 – Oct 2025), out-of-sample:

```
always-pick-favourite   acc 0.618
glicko-only             acc 0.618   auc 0.653   <- best component
veto-structural         acc 0.597   auc 0.632
FULL MODEL              acc 0.564   auc 0.578   <- worst
confident-call calibration 0.055    BROKEN
```

Three facts that follow, all reproducible with `evaluate`:

1. **Every layer added on top of plain Glicko makes it worse.** The per-map
   ratings + veto simulation — the core thesis of this project — lose to a
   simple team rating.
2. **Plain Glicko only ties "pick the favourite."** No edge over a one-liner.
3. **The abstention mechanism is broken.** Accuracy peaks at 65% then falls:
   44% at confidence 0.80, 17% at 0.90. Raising the bar makes it *worse*, so
   the "72-75% on the matches it calls" idea does not hold on real data.

An earlier 2015 CS:GO dataset (1,551 series) gave the same shape: 59.5% vs a
59.1% baseline.

**Do not present this as a working betting model.** If asked to improve it,
say what has already been ruled out before proposing anything.

## Already tried and ruled out

- **Biased map records.** Sweeps under-recorded the winner's maps. Fixed
  (`maps_complete=False` skips per-map updates). Model did not improve —
  hypothesis tested and rejected.
- **Veto as a penalised feature** vs **as an offset.** Offset wins on synthetic
  data, neither wins on real data.
- **Correction features** (form, streak, h2h, rest, LAN split) — net negative
  on real data.

## Never tried, and the only promising lead

The dataset carries **per-player stats** (`team1_player_N_RATING/ADR/KAST/KPR`)
and **100% lineup coverage**, and the model uses *none of it* — it reads only
who won. Rating 10 players instead of 2 teams is the largest untouched signal.

**Blocking question:** are those stats pre-match form or from the match itself?
If in-match, using them is leakage. Unresolved — the user was asked and has not
yet answered.

## Verified vs unverified

| Area | State |
|---|---|
| Model, ratings, veto, evaluation, tracker, risk | Tested. 99 tests. |
| `datasets.py` CSV loading | Verified against two real files |
| `liquipedia.py` | **Never run against the live API.** Dead end anyway — free tier is education-only |
| `polymarket.py` | **Never run against Polymarket.** Signatures taken from the installed `py-clob-client`, but no request has ever been sent |

Two code reviews found 15 defects. **They clustered almost entirely in the code
that talks to services the build sandbox cannot reach.** Treat those two files
as unproven.

## Traps that have already bitten

- **Outcome-labelled columns.** The Kaggle file has `winner_map`, `loser_map`,
  `winner_mirage`. Anything named `winner_*` encodes the result. Usable only to
  reconstruct labels, never as a feature.
- **ECE hides tail failures.** It is size-weighted; a healthy middle drowns a
  broken tail. Real run: ECE 0.0203 "healthy" while the 0.80-0.90 bucket
  predicted 83% and delivered 40%. Use `tail_calibration_error` — a bot only
  acts on the tails.
- **Leakage ordering.** `build_features` must run before `observe`. Guarded by
  `test_features_never_see_the_match_they_describe`.
- **Baselines are mandatory.** Every `evaluate` prints always-pick-favourite,
  glicko-only and veto-structural. A number without them is meaningless.

## Layout

```
cs2model/
  data.py       Match/MapResult, synthetic league, active_pool()
  glicko.py     Glicko-2 incl. the Illinois volatility solver
  ratings.py    RatingBook: per-map ratings, roster churn, idle decay
  veto.py       ban/pick simulation -> series probability
  model.py      features, offset logistic, calibration, abstention
  evaluate.py   walk-forward, baselines, coverage curve, calibration
  datasets.py   CSV loaders (per-map, match-level, HLTV wide)
  tracker.py    bankroll ledger (price/shares), heartbeat
  risk.py       limits + kill switch — final word over the model
  strategy.py   market -> probability -> risk -> order
  polymarket.py venue adapter (UNVERIFIED)
  liquipedia.py ingest (UNVERIFIED, dead end)
  dashboard.py  terminal view
  cli.py        all commands
```

## Commands

```bash
python -m pytest tests/ -q                      # 99 tests
python -m cs2model.cli demo                     # synthetic, no data needed
python -m cs2model.cli load-csv --csv X --out data/matches.json
python -m cs2model.cli evaluate --data data/matches.json
python -m cs2model.cli run --capital 500        # DRY RUN unless --live
python -m cs2model.cli dashboard --watch 30
python -m cs2model.cli stop                     # kill switch
```

## Conventions

- Virtualenv at `venv/`. `ModuleNotFoundError: numpy` always means it is not
  activated.
- Tests must pass before committing. Every bug fix gets a regression test.
- `--live` is never the default and must never become one.
- User is a beginner on an AWS EC2 box, connecting via the browser console.
  Give exact copy-pasteable commands and say what success looks like.
- User docs: `START_HERE.md` (beginner), `SETUP.md`, `README_CS2.md`.

## Honesty rules for this project

The evaluation harness exists to stop the model flattering itself, and it has
caught a real failure three times. Do not weaken it.

Report results straight. The model losing to "pick the favourite" is the
finding, not something to explain away. A negative result delivered clearly is
worth more here than an optimistic one, because the next step after this is
real money.
