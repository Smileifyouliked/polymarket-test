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


# ── the HLTV wide format (one row per match, with lineups) ───────────────────

HLTV = """match_id,date,tournament,winner,score_team1,score_team2,winner_map,loser_map,decider_map,event_type,team1_name,team1_player_1_name,team1_player_2_name,team1_player_3_name,team1_player_4_name,team1_player_5_name,team2_name,team2_player_1_name,team2_player_2_name,team2_player_3_name,team2_player_4_name,team2_player_5_name
1,2025-01-05,IEM,Vitality,2,1,Mirage,Nuke,Inferno,LAN,Vitality,ZywOo,apEX,flameZ,mezii,ropz,Spirit,donk,zont1x,sh1ro,chopper,magixx
2,2025-01-06,IEM,NAVI,0,2,Dust2,,Ancient,Online,FaZe,karrigan,rain,frozen,ropz,broky,NAVI,w0nderful,b1t,iM,Aleksib,jL
3,2025-01-07,IEM,Vitality,2,1,Train,Anubis,Overpass,LAN,Vitality,ZywOo,apEX,flameZ,mezii,NEWGUY,NAVI,w0nderful,b1t,iM,Aleksib,jL
"""


def test_hltv_format_is_detected_and_parsed(tmp_path):
    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    assert len(ms) == 3
    assert ms[0].team_a == "Vitality" and ms[0].winner == "Vitality"
    assert ms[0].lan is True
    assert ms[1].lan is False


def test_hltv_lineups_are_captured(tmp_path):
    """Lineups drive the roster-change mechanism, dormant without them."""
    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    assert ms[0].lineup_a == ("ZywOo", "apEX", "flameZ", "mezii", "ropz")
    assert len(ms[0].lineup_b) == 5


def test_no_decider_map_is_invented_on_a_sweep(tmp_path):
    """
    A 2-0 has no decider. The column can still be populated, and counting it
    would invent a map that was never played and award it to the favourite.
    """
    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    sweep = [m for m in ms if m.team_a == "FaZe"][0]
    assert len(sweep.maps) == 1, [r.map_name for r in sweep.maps]
    assert "Ancient" not in [r.map_name for r in sweep.maps]

    split = [m for m in ms if m.team_b == "Spirit"][0]
    assert len(split.maps) == 3


def test_hltv_map_results_go_to_the_right_team(tmp_path):
    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    m = ms[0]
    by_map = {r.map_name: r.winner for r in m.maps}
    assert by_map["Mirage"] == "Vitality"   # winner_map
    assert by_map["Nuke"] == "Spirit"       # loser_map
    assert by_map["Inferno"] == "Vitality"  # decider goes to the series winner


def test_hltv_best_of_comes_from_the_series_score(tmp_path):
    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    assert all(m.best_of == 3 for m in ms)


def test_winner_comes_from_the_score_not_the_label(tmp_path):
    """
    The winner column holds a team name in one dataset, a side label in the
    next, an index in a third. The score needs no interpretation, so it wins.
    """
    csv = HLTV.replace(",Vitality,2,1,", ",team1,2,1,").replace(",NAVI,0,2,", ",GARBAGE,0,2,")
    ms = load_csv_matches(_write(tmp_path, "w.csv", csv), verbose=False)
    assert len(ms) == 3
    assert ms[0].winner == "Vitality"
    assert [m for m in ms if m.team_a == "FaZe"][0].winner == "NAVI"


def test_winner_label_forms_are_understood():
    from cs2model.datasets import _match_winner_name

    assert _match_winner_name("team1", "A", "B") == "A"
    assert _match_winner_name("Team 2", "A", "B") == "B"
    assert _match_winner_name("  vitality ", "Vitality", "Spirit") == "Vitality"
    assert _match_winner_name("junk", "A", "B") is None
    assert _match_winner_name("", "A", "B") is None


def test_failures_report_the_offending_values(tmp_path):
    """
    An error that hides the data forces a round trip to learn something the
    code already had. Every skip reason carries examples, not just dates.
    """
    bad = "date,winner,score_team1,score_team2,team1_name,team2_name\nNOT_A_DATE,???,1,1,A,B\n"
    with pytest.raises(ValueError) as e:
        load_csv_matches(_write(tmp_path, "b.csv", bad), verbose=False)
    assert "NOT_A_DATE" in str(e.value)


def test_partial_map_records_are_flagged(tmp_path):
    """
    A 2-0 names only one of the two maps played, and the one it drops is the
    winner's. Feeding that to per-map ratings biases every strong team down.
    """
    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    split = [m for m in ms if m.team_b == "Spirit"][0]     # 2-1, all 3 named
    sweep = [m for m in ms if m.team_a == "FaZe"][0]       # 0-2, only 1 named
    assert split.maps_complete is True
    assert sweep.maps_complete is False


def test_partial_records_skip_per_map_ratings_but_still_teach_overall(tmp_path):
    from cs2model.ratings import RatingBook

    ms = load_csv_matches(_write(tmp_path, "h.csv", HLTV), verbose=False)
    sweep = [m for m in ms if m.team_a == "FaZe"][0]
    book = RatingBook()
    book.observe(sweep)
    winner = book.team(sweep.winner)
    assert winner.per_map == {}, "a biased map record must not move per-map ratings"
    assert winner.overall.games == 1, "but the series still teaches the overall rating"
