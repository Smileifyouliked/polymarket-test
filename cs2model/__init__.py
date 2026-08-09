"""
cs2model — a calibrated CS2 match forecaster.

The thesis in four lines:
  * A match is a veto plus several maps, so the model works at map level.
  * Ratings are per-map and get MORE uncertain when a roster changes.
  * Probabilities are calibrated, so the numbers mean what they say.
  * The model is allowed to decline a match. That is how you buy accuracy.
"""

__version__ = "0.1.0"

from .data import Match, MapResult, generate_synthetic_league, series_prob  # noqa: F401
from .model import CS2Model, build_dataset, decide  # noqa: F401
from .ratings import RatingBook  # noqa: F401
from .veto import match_win_prob, veto_report  # noqa: F401
