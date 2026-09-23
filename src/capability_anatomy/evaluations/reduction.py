"""Pure versioned score semantics shared by online and offline verification."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from ..errors import InvalidEvidenceError

SCHEMA = 'capability-anatomy/score-reduction/v1'
_REASONS = {
    'score_list_empty': 'score reduction requires observations',
    'score_value_invalid': 'score reduction requires finite numeric values',
    'score_fraction_incomplete': 'every score fraction requires both numerator and denominator',
    'score_fraction_mixed': 'a metric cannot mix fractional and value-only observations',
    'score_denominator_invalid': 'score fractions require positive finite denominators',
    'score_fraction_inconsistent': 'score value differs from its declared fraction',
}


class ScoreReductionError(InvalidEvidenceError):
    def __init__(self, reason: str):
        super().__init__(_REASONS[reason])
        self.reason = reason


def _number(value: Any) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ScoreReductionError('score_value_invalid')
    return float(value)


def reduce_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce uniform fractions by ratio of sums, or uniform values by mean.

    Fraction presence determines the sealed contract only when it is uniform.
    No plugin identity, plugin import, or declared aggregate selects the reducer.
    """
    values = []
    fractions = []
    for score in scores:
        value = _number(score.get('value'))
        numerator, denominator = score.get('numerator'), score.get('denominator')
        if (numerator is None) != (denominator is None):
            raise ScoreReductionError('score_fraction_incomplete')
        values.append(value)
        if numerator is not None:
            numerator, denominator = _number(numerator), _number(denominator)
            if denominator <= 0:
                raise ScoreReductionError('score_denominator_invalid')
            if not math.isclose(value, numerator / denominator, rel_tol=1e-9, abs_tol=1e-12):
                raise ScoreReductionError('score_fraction_inconsistent')
            fractions.append((numerator, denominator))
    if not values:
        raise ScoreReductionError('score_list_empty')
    if fractions and len(fractions) != len(values):
        raise ScoreReductionError('score_fraction_mixed')
    if fractions:
        numerator = sum(item[0] for item in fractions)
        denominator = sum(item[1] for item in fractions)
    else:
        numerator, denominator = sum(values), len(values)
    value = _number(numerator / denominator)
    return {'value': value, 'numerator': _number(numerator), 'denominator': _number(denominator),
            'observations': len(values), 'reducer': 'ratio_of_sums' if fractions else 'mean_value'}
