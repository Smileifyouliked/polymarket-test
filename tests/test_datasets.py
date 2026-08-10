"""
CSV loading. Both shapes public CS2 dumps actually come in.
"""

import pytest

from cs2model.data import ACTIVE_DUTY, active_pool, has_map_detail, set_active_pool
from cs2model.datasets import load_csv_matches
from cs2model.model import build_features
from cs2model.ratings import RatingBook


@pytest.fixture(autouse=True)
def _restore_pool():
    """The pool is global state; put it back so test order cannot matter."""
    yield
    set_active_pool(ACTIVE_DUTY)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


PER_MAP = """Date,Map,Team,PTS,Team,PTS
2025-01-05,de_mirage,Vitality,16,Spirit,10
2025-01-05,de_nuke,Vitality,13,Spirit,16
2025-01-05,de_inferno,Vitality,16,Spirit,14
2025-01-06,de_dust2,FaZe,16,NAVI,9
2025-01-07,de_ancient,Vitality,10,NAVI,16
2025-01-08,de_anubis,Spirit,16,FaZe,12
2025-01-09,de_train,NAVI,16,Spirit,7
2025-01-10,de_mirage,FaZe,16,Vitality,11
"""

MATCH_LEVEL = """date,team_1,team_2,winner,event
2025-01-05,Vitality,Spirit,Vitality,IEM
2025-01-06,FaZe,NAVI,NAVI,IEM
2025-01-07,Vitality,NAVI,Vitality,IEM
"""


def test_per_map_rows_group_into_series(tmp_path):
    ms = load_csv_matches(_write(tmp_path, "m.csv", PER_MAP), verbose=False)
    bo3 = [m for m in ms if m.best_of == 3]
    assert bo3, "three maps on one day between two teams is a Bo3"
    assert len(bo3[0].maps) == 3
    assert bo3[0].winner == "Vitality"      # won maps 1 and 3
    assert has_map_detail()


def test_duplicate_score_headers_are_read_positionally(tmp_path):
    """
    Both score columns are named PTS. csv.DictReader keeps only the last, which
    silently records the winner with the loser's score.
    """
    ms = load_csv_matches(_write(tmp_path, "m.csv", PER_MAP), verbose=False)
    for m in ms:
        for r in m.maps:
            assert r.rounds_winner > r.rounds_loser, (
                "the winner must have more rounds than the loser"
            )


def test_map_pool_comes_from_the_data(tmp_path):
    ms = load_csv_matches(_write(tmp_path, "m.csv", PER_MAP), verbose=False)
    assert set(active_pool()) == {
        "Mirage", "Nuke", "Inferno", "Dust2", "Ancient", "Anubis", "Train"
    }


def test_match_level_rows_load_and_flag_themselves(tmp_path):
    ms = load_csv_matches(_write(tmp_path, "m.csv", MATCH_LEVEL), verbose=False)
    assert len(ms) == 3
    assert ms[0].winner == "Vitality"
    assert not has_map_detail(), (
        "a file with no map results must not pretend it has a map pool"
    )


def test_model_still_works_without_map_detail(tmp_path):
    """
    The veto simulator has nothing to simulate here. It must fall back to the
    overall rating rather than inventing a pool and quoting a confident number
    backed by nothing.
    """
    ms = load_csv_matches(_write(tmp_path, "m.csv", MATCH_LEVEL), verbose=False)
    book = RatingBook()
    book.observe(ms[0])
    feats = build_features(book, ms[1])
    assert len(feats) == 12
    assert all(f == f for f in feats), "no NaNs"


def test_unusable_file_says_what_is_missing(tmp_path):
    bad = _write(tmp_path, "bad.csv", "foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="could not identify columns"):
        load_csv_matches(bad, verbose=False)


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_csv_matches(_write(tmp_path, "e.csv", "date,team,team\n"), verbose=False)


def test_degenerate_pool_is_refused_unless_asked_for():
    with pytest.raises(ValueError, match="at least 7"):
        set_active_pool(["OnlyOne"])
    set_active_pool(["OnlyOne"], allow_degenerate=True)
    assert not has_map_detail()
