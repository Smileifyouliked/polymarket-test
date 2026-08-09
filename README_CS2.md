# cs2model — a CS2 match forecaster that knows when to shut up

## The logic, in plain terms

**1. A match isn't one game. It's a veto plus 1-3 maps.**

Two teams ban and pick maps before they play. Team A might be great on Mirage
and terrible on Nuke. So "who wins this match?" is really "which maps end up
getting played, and who wins those?"

Most prediction models skip this and treat a match like a coin with a bias.
This one rates every team **on every map separately**, then simulates the ban/pick
sequence to work out which maps actually get played.

**2. Rosters change, so ratings must be allowed to get less sure.**

If a team swaps a star player, its old rating isn't wrong exactly — it's
*unreliable*. Glicko ratings carry a second number for "how confident am I in
this rating", and a roster change widens it. Swap all five players and the
model treats you as a new team, because you are.

**3. The output is a probability, and it has to be honest.**

If the model says 70%, teams like that should win about 70% of the time. That
property is called **calibration**, and it's checked on every run.

**4. The model is allowed to say "no call".**

This is the part that gets you your accuracy number. If the probabilities are
honest, then *only answering when the model is above 72% confident* gives you
roughly 72%+ accuracy on the ones it answers. You don't get that by making the
model smarter — you get it by letting it skip the coinflips.

**You buy accuracy with silence.** That's the whole trick.

## Does it actually work?

Everything below is measured by `python -m cs2model.cli demo`, walk-forward
(train on the past, predict the future, never peek), on a synthetic league
where the true probabilities are known.

```
BASELINES FIRST — the model has to beat these or it is decoration:

  always-pick-favourite  acc= 0.679
  glicko-only            acc= 0.679  logloss= 0.594  auc= 0.747
  veto-structural        acc= 0.690  logloss= 0.583  auc= 0.758
  FULL MODEL             acc= 0.688  logloss= 0.586  auc= 0.752

  calibration error (ECE): 0.0208
```

And the part you care about:

```
COVERAGE CURVE — this is where your accuracy target lives

  confidence   accuracy   coverage   matches
        0.50      0.687    100.0%      3000
        0.55      0.706     88.7%      2661
        0.60      0.734     73.3%      2198   <-- your 72-75%, calling 73% of matches
        0.65      0.766     57.6%      1729
        0.70      0.800     40.8%      1223
        0.75      0.842     29.0%       871
        0.80      0.892     19.2%       575
```

**72-75% accuracy while still calling ~3 matches in 4.** Push the bar to 0.70
and you're at 80%, but you only answer 4 matches in 10.

### Two honest caveats

- **This is synthetic data.** The league is built to look like CS2 (an oracle
  that knows the true probabilities scores 73-74%, matching the real-world
  ceiling), but real matches are messier. Expect the whole curve to shift down
  when you plug in Liquipedia data. The *shape* — accuracy rising as coverage
  falls — is the part that transfers.
- **The correction layer currently earns nothing.** `FULL MODEL` is a hair
  worse than `veto-structural` alone (0.586 vs 0.583 log loss). That's expected
  here: the synthetic generator has no "form" or "momentum" effects, so those
  features are pure noise in this test. Whether they help is a question only
  real data can answer. The architecture is built so they can't do much damage
  either way — see below.

## How it's put together

```
per-map Glicko ratings      "A wins 63% on Mirage, 31% on Nuke"
        |
   veto simulator           simulate bans/picks -> which maps get played
        |
  structural probability    exact series math over the played maps
        |
  correction layer          form, rest, LAN split, head-to-head
        |
   calibration              make 70% actually mean 70%
        |
   abstention threshold     decline anything below your confidence bar
```

The correction layer takes the structural probability as a **fixed offset**,
not as another input feature. That matters: the first version fed it in as one
predictor among twelve, and ridge regularisation split its weight with the
correlated rating gap — the combined model scored *worse than its own best
input* (0.626 AUC vs 0.638). As an offset it's unpenalised, so the model starts
at the structural baseline and only moves if the data pays for it.

## Running it

```bash
pip install numpy scikit-learn scipy pytest requests

python -m cs2model.cli demo      # end-to-end on synthetic data
python -m pytest tests/ -q       # 24 tests
```

With real data:

```bash
export LIQUIPEDIA_API_KEY=...            # https://liquipedia.net/api-terms-of-use
export LIQUIPEDIA_USER_AGENT="you@example.com"

python -m cs2model.cli explore --limit 3          # CHECK THE FIELD NAMES FIRST
python -m cs2model.cli ingest --out data/matches.json --pages 40
python -m cs2model.cli evaluate --data data/matches.json
python -m cs2model.cli predict --data data/matches.json \
    --team-a "Vitality" --team-b "Spirit" --bo 3 --lan --threshold 0.72
```

> **`liquipedia.py` is the one file that was never run against the live API** —
> the sandbox it was written in blocks outbound calls to `api.liquipedia.net`.
> Run `explore` first, look at the real field names, and fix `_to_match` if they
> differ. Everything downstream is covered by the test suite; this seam isn't.

`predict` prints per-map probabilities, how often each map survives the veto,
and either a verdict or `NO CALL`.

## What the tests actually protect

The test suite exists to stop the model quietly lying. The interesting ones:

- `test_features_never_see_the_match_they_describe` — rebuilds the rating book
  independently and asserts the features match. This is the anti-leakage guard;
  without it, a random train/test split hands you a fake 75%.
- `test_model_is_calibrated` — if 80% doesn't mean 80%, the abstention trick is
  broken and no threshold saves you.
- `test_coverage_curve_delivers_the_accuracy_it_advertises` — the direct test of
  the claim this whole project rests on.
- `test_a_narrow_map_pool_is_a_liability` — two teams with the *same average*
  map probability; the specialist is the worse side, because the opponent's
  two bans remove exactly the maps it needs. A map-blind model calls these
  identical. This is why the veto simulator exists.
- `test_model_beats_always_pick_the_favourite` — the baseline that most public
  CS2 models never report and sometimes lose to.

## Things worth doing next

1. **Player-level ratings.** Right now roster changes only widen uncertainty.
   Rating the five players and summing to a lineup would handle stand-ins and
   transfers properly.
2. **Demo file parsing.** `.dem` files are free from HLTV/FACEIT and parse with
   `awpy`. Round-level economy, opening duels, utility usage — no public CS2
   model uses this, and it's the deepest signal available.
3. **Score the closing line.** `Match.closing_prob_a` is already plumbed
   through; fill it in and the evaluation prints the market as a baseline. That
   is the real scoreboard.
4. **Tune `MAP_SHRINK_K`** (in `ratings.py`) on real data — it controls how fast
   the model trusts thin per-map samples, and the right value depends on how
   much history you ingest.
