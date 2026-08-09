"""
Tests that would catch a model quietly lying to you.

The important ones are at the bottom: on synthetic data where the true
probabilities are known, we assert that the model is CALIBRATED and that the
coverage curve delivers the accuracy it advertises. Those two properties are
the entire basis for "72-75% accuracy on the matches it chooses to call".
"""

import math
import random

import numpy as np
import pytest

from cs2model import glicko
from cs2model.data import (
    ACTIVE_DUTY,
    Match,
    generate_synthetic_league,
    series_prob,
)
from cs2model.evaluate import (
    accuracy,
    coverage_curve,
    expected_calibration_error,
    log_loss,
    walk_forward,
)
from cs2model.model import CS2Model, build_dataset, build_features, decide
from cs2model.ratings import RatingBook
from cs2model.veto import match_win_prob, simulate_veto


# ── series probability ───────────────────────────────────────────────────────


def test_bo1_is_just_the_map():
    assert series_prob([0.63], 1) == pytest.approx(0.63)


def test_bo3_matches_closed_form():
    p1, p2, p3 = 0.6, 0.45, 0.55
    expected = p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
    assert series_prob([p1, p2, p3], 3) == pytest.approx(expected)


def test_series_probability_is_complementary():
    for bo in (1, 3, 5):
        probs = [random.random() for _ in range(bo)]
        flipped = [1 - p for p in probs]
        assert series_prob(probs, bo) + series_prob(flipped, bo) == pytest.approx(1.0)


def test_longer_series_favours_the_better_team():
    """The classic result: more maps means less variance means the dog suffers."""
    p = 0.60
    assert series_prob([p], 1) < series_prob([p] * 3, 3) < series_prob([p] * 5, 5)


# ── glicko ───────────────────────────────────────────────────────────────────


def test_winning_raises_rating_and_losing_lowers_it():
    a, b = glicko.Rating(), glicko.Rating()
    a2 = glicko.update(a, b, 1.0)
    b2 = glicko.update(b, a, 0.0)
    assert a2.r > a.r
    assert b2.r < b.r


def test_rating_deviation_shrinks_with_games():
    a, b = glicko.Rating(), glicko.Rating()
    for _ in range(25):
        a = glicko.update(a, b, 1.0)
    assert a.rd < glicko.DEFAULT_RD * 0.6


def test_expected_score_is_symmetric():
    a = glicko.Rating(r=1700, rd=60)
    b = glicko.Rating(r=1500, rd=60)
    assert glicko.expected_score(a, b) + glicko.expected_score(b, a) == pytest.approx(1.0, abs=1e-3)


def test_roster_change_increases_uncertainty_and_regresses_rating():
    strong = glicko.Rating(r=1900, rd=50)
    after = glicko.inflate_for_roster_change(strong, n_changed=3)
    assert after.rd > strong.rd, "a roster swap must make us LESS certain"
    assert 1500 < after.r < strong.r, "and pull the rating back toward the mean"


def test_full_roster_swap_regresses_more_than_single_swap():
    r = glicko.Rating(r=1900, rd=50)
    one = glicko.inflate_for_roster_change(r, 1)
    five = glicko.inflate_for_roster_change(r, 5)
    assert five.r < one.r
    assert five.rd > one.rd


def test_idle_decay_widens_rd_but_not_rating():
    r = glicko.Rating(r=1800, rd=60)
    d = glicko.decay_idle(r, days_idle=90)
    assert d.r == r.r
    assert d.rd > r.rd


# ── veto ─────────────────────────────────────────────────────────────────────


def test_veto_produces_the_right_number_of_maps():
    rng = random.Random(1)
    for bo in (1, 3, 5):
        played = simulate_veto(lambda m: 0.5, bo, rng)
        assert len(played) == bo
        assert len({m for m, _ in played}) == bo, "no map may be played twice"


def test_bo3_has_one_pick_each_and_a_decider():
    rng = random.Random(2)
    played = simulate_veto(lambda m: 0.5, 3, rng)
    pickers = sorted(pb for _, pb in played)
    assert pickers == ["A", "B", "decider"]


def test_veto_bans_your_worst_maps():
    """A team dreadful on Nuke should rarely have to play Nuke."""
    probs = {m: 0.5 for m in ACTIVE_DUTY}
    probs["Nuke"] = 0.05  # A is hopeless here
    rng = random.Random(3)
    plays = 0
    for _ in range(500):
        played = simulate_veto(probs.__getitem__, 3, rng)
        plays += any(m == "Nuke" for m, _ in played)
    assert plays / 500 < 0.25, "A should be banning Nuke away most of the time"


def test_better_team_favoured_after_veto():
    strong = {m: 0.65 for m in ACTIVE_DUTY}
    assert match_win_prob(strong.__getitem__, 3, n_sims=300) > 0.65


def test_a_narrow_map_pool_is_a_liability():
    """
    THE WHOLE REASON THIS MODULE EXISTS.

    Two teams with the SAME average map probability (0.55). One is evenly
    good everywhere; the other is dominant on three maps and poor on four.

    The map-blind model calls these identical. They are not, and the
    specialist is the WORSE side: in a Bo3 the opponent gets two bans and
    spends them removing exactly the maps the specialist needs. A narrow map
    pool is a well-known liability in Counter-Strike, and it falls straight
    out of simulating the veto instead of averaging over it.
    """
    broad = {m: 0.55 for m in ACTIVE_DUTY}
    polarized = {
        "Ancient": 0.85, "Anubis": 0.85, "Dust2": 0.85,
        "Inferno": 0.325, "Mirage": 0.325, "Nuke": 0.325, "Train": 0.325,
    }
    avg_broad = sum(broad.values()) / 7
    avg_pol = sum(polarized.values()) / 7
    assert avg_broad == pytest.approx(avg_pol, abs=0.005), "same average by construction"

    p_broad = match_win_prob(broad.__getitem__, 3, n_sims=1200, seed=4)
    p_pol = match_win_prob(polarized.__getitem__, 3, n_sims=1200, seed=4)
    map_blind = series_prob([avg_pol] * 3, 3)

    assert p_pol < p_broad - 0.05, "the specialist should be punished by the bans"
    assert abs(p_pol - map_blind) > 0.05, (
        "if simulating the veto gave the same answer as averaging the maps, "
        "this whole module would be pointless"
    )


# ── leakage ──────────────────────────────────────────────────────────────────


def test_features_never_see_the_match_they_describe():
    """
    Rebuild the book independently up to match k and confirm the features
    match. If build_dataset ever observed a match before extracting its
    features, this fails.
    """
    matches = generate_synthetic_league(n_teams=12, n_matches=180, seed=11)
    ds = build_dataset(matches)

    ordered = sorted(matches, key=lambda m: m.date)
    k = 120
    book = RatingBook()
    for m in ordered[:k]:
        if m.winner and m.maps:
            book.observe(m)

    # Same default seed as build_dataset — the veto feature is Monte Carlo, so
    # it only reproduces bit-for-bit under the same random stream.
    expected = build_features(book, ordered[k])
    idx = sum(1 for m in ordered[:k] if m.winner and m.maps)
    np.testing.assert_allclose(ds.X[idx], expected, rtol=1e-9, atol=1e-9)


# ── the properties your accuracy target depends on ───────────────────────────


@pytest.fixture(scope="module")
def synthetic_run():
    matches = generate_synthetic_league(n_teams=36, n_matches=3000, seed=5)
    ds = build_dataset(matches, seed=5)
    res = walk_forward(ds, initial_frac=0.5, n_folds=8)
    return ds, res


def test_model_beats_always_pick_the_favourite(synthetic_run):
    _, res = synthetic_run
    fav_acc = accuracy(res.y, np.where(res.p_glicko >= 0.5, 0.999, 0.001))
    model_acc = accuracy(res.y, res.p_model)
    assert model_acc >= fav_acc - 0.01, (
        f"model {model_acc:.3f} vs favourite baseline {fav_acc:.3f} — if the "
        "model cannot match the naive baseline it is decoration"
    )


def test_model_beats_baselines_on_log_loss(synthetic_run):
    _, res = synthetic_run
    assert log_loss(res.y, res.p_model) < log_loss(res.y, res.p_glicko)


def test_model_is_calibrated(synthetic_run):
    """
    Without this, thresholding is meaningless. A model that says 80% and wins
    60% of those will hand you a confident, wrong 'verdict' all season.
    """
    _, res = synthetic_run
    ece = expected_calibration_error(res.y, res.p_model)
    assert ece < 0.05, f"expected calibration error too high: {ece:.4f}"


def test_coverage_curve_delivers_the_accuracy_it_advertises(synthetic_run):
    """
    THE CLAIM UNDER TEST: 'only answer when confidence > t' yields roughly t
    accuracy on the answered subset. This is what buys the 72-75%.
    """
    _, res = synthetic_run
    for row in coverage_curve(res.y, res.p_model, thresholds=(0.60, 0.65, 0.70, 0.75)):
        if row["n"] < 60:
            continue
        assert row["accuracy"] >= row["threshold"] - 0.06, (
            f"at confidence {row['threshold']:.2f} the model only hit "
            f"{row['accuracy']:.3f} over {row['n']} matches"
        )


def test_accuracy_rises_as_coverage_falls(synthetic_run):
    """The trade is real and monotone: fewer calls, better calls."""
    _, res = synthetic_run
    rows = [r for r in coverage_curve(res.y, res.p_model) if r["n"] >= 60]
    accs = [r["accuracy"] for r in rows]
    covs = [r["coverage"] for r in rows]
    assert covs == sorted(covs, reverse=True)
    assert accs[-1] > accs[0] + 0.05, "raising the bar must actually raise accuracy"


def test_full_slate_accuracy_is_realistic(synthetic_run):
    """
    Sanity check on the headline number. If unfiltered accuracy comes out at
    90% something is leaking, because CS2 is not that predictable.
    """
    _, res = synthetic_run
    acc = accuracy(res.y, res.p_model)
    assert 0.55 < acc < 0.85, f"suspicious full-slate accuracy: {acc:.3f}"


def test_model_tracks_the_true_probabilities(synthetic_run):
    """
    Only possible because the synthetic league knows the truth.

    Note the use of res.idx rather than 'the last N matches' — walk_forward
    skips undersized folds, so those are not the same set. Lining them up by
    tail slice silently compares each prediction to the wrong match.
    """
    ds, res = synthetic_run
    true_p = np.array([ds.matches[i].true_p_a for i in res.idx])
    rmse = float(np.sqrt(np.mean((res.p_model - true_p) ** 2)))
    assert rmse < 0.20, f"model probabilities are far from truth: rmse={rmse:.3f}"


def test_decide_abstains_in_the_middle():
    p = np.array([0.10, 0.45, 0.50, 0.55, 0.90])
    calls = decide(p, 0.75)
    assert list(calls) == [-1, 0, 0, 0, 1]


# ── regressions from the code review ─────────────────────────────────────────
#
# Every test below reproduces a bug that shipped and passed the original suite.


def test_walk_forward_explains_itself_when_there_is_too_little_data():
    """Used to die with 'need at least one array to concatenate'."""
    matches = generate_synthetic_league(n_teams=8, n_matches=100, seed=3)
    ds = build_dataset(matches)
    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward(ds, initial_frac=0.5, n_folds=8)


def test_decide_never_inverts_below_half():
    """
    decide([0.55, 0.59], 0.45) used to return [-1, -1]: both masks were
    assigned unconditionally and the second overwrote the first.
    """
    p = np.array([0.55, 0.59, 0.40])
    calls = decide(p, 0.45)
    assert list(calls) == [1, 1, -1]
    # And a 0.5 threshold means "call everything", not "call everything for B".
    assert list(decide(np.array([0.6, 0.4]), 0.5)) == [1, -1]


def test_overall_rating_counts_matches_not_maps():
    """
    The overall rating updated once per MAP at full Glicko weight, so a Bo3
    moved it three times and RD collapsed far too fast.
    """
    matches = [m for m in generate_synthetic_league(n_teams=6, n_matches=60, seed=4)
               if m.winner and m.maps]
    book = RatingBook()
    played = 0
    tid = matches[0].team_a
    for m in matches:
        book.observe(m)
        if tid in (m.team_a, m.team_b):
            played += 1
    assert book.team(tid).overall.games == played


def test_roster_change_affects_the_very_match_it_appears_in():
    """
    Roster inflation lived in observe(), which runs AFTER build_features, so
    the first match with a new lineup was still forecast at full confidence
    using the departed players' rating.
    """
    matches = [m for m in generate_synthetic_league(n_teams=8, n_matches=200, seed=6)
               if m.winner and m.maps]
    book = RatingBook()
    for m in matches[:150]:
        book.observe(m)

    nxt = matches[150]
    before = book.team(nxt.team_a).overall.rd
    swapped = Match(
        team_a=nxt.team_a, team_b=nxt.team_b, date=nxt.date, best_of=3,
        lineup_a=("new1", "new2", "new3", "new4", "new5"),
        lineup_b=book.team(nxt.team_b).lineup,
    )
    build_features(book, swapped)
    assert book.team(nxt.team_a).overall.rd > before, (
        "a brand-new five should make the model LESS sure before this match, "
        "not after it"
    )


def test_partial_lineup_does_not_look_like_a_roster_swap():
    """
    Liquipedia does not always report five players. A 3-name record used to
    read as a two-player swap and permanently regressed the rating.
    """
    book = RatingBook()
    five = ("a", "b", "c", "d", "e")
    book.apply_lineup("T", five)
    book.team("T").overall = glicko.Rating(r=1850, rd=55)

    assert book.apply_lineup("T", ("a", "b", "c")) == 0
    assert book.team("T").overall.rd == 55
    assert book.team("T").overall.r == 1850
    assert book.team("T").lineup == five, "keep the last known good five"

    # A real, fully-reported swap still registers.
    assert book.apply_lineup("T", ("a", "b", "c", "d", "z")) == 1
    assert book.team("T").overall.rd > 55


def test_liquipedia_tier_survives_null_extradata():
    """dict.get('extradata', {}) returns None when the key holds an explicit
    null, and .get on that raised AttributeError mid-ingest."""
    from cs2model.liquipedia import _tier_from_liquipedia

    assert _tier_from_liquipedia({"extradata": None}) == 2
    assert _tier_from_liquipedia({"extradata": None, "liquipediatier": "1"}) == 1
    assert _tier_from_liquipedia({"extradata": {"liquipediatier": "3"}}) == 3


def test_missing_scores_do_not_give_the_winner_zero_rounds():
    """The 13/0 fallback was written from team 1's view, so a team-2 win with
    missing scores produced rounds_winner=0, rounds_loser=13."""
    from cs2model.liquipedia import _to_match

    rec = {
        "match2opponents": [{"name": "Alpha"}, {"name": "Bravo"}],
        "winner": "2",
        "date": "2025-01-01 12:00:00",
        "bestof": "1",
        "match2games": [{"map": "de_mirage", "winner": "2", "scores": []}],
    }
    m = _to_match(rec)
    assert m is not None
    assert m.winner == "Bravo"
    assert m.maps[0].winner == "Bravo"
    assert m.maps[0].rounds_winner > m.maps[0].rounds_loser

    # Real scores are still honoured, and still from the winner's side.
    rec["match2games"] = [{"map": "de_nuke", "winner": "2", "scores": ["7", "13"]}]
    m2 = _to_match(rec)
    assert (m2.maps[0].rounds_winner, m2.maps[0].rounds_loser) == (13, 7)
