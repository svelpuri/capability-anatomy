"""Regressions for the independent PR59 Phase 5 review."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from capability_anatomy.domain import EvaluationScore, NormalizedRecord, ScoreReason
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.evaluations import EvaluationExecutor
from capability_anatomy.evaluations.plugins.phase5 import Phase5EvaluationSuite
from capability_anatomy.serialization import serialize_observation
from capability_anatomy.telemetry import OperationTelemetry


@pytest.mark.parametrize('raw,reason,score', [
    ('[]', 'abstention_empty_call_list', 1),
    ('No applicable tool.', 'abstention_explicit_form', 1),
    ('I would prefer not to use tools.', 'abstention_unrecognized_or_malformed', 0),
    ('[{"name":"lookup","arguments":{}}]', 'abstention_tool_invocation', 0),
    ('', 'abstention_empty_or_invalid_type', 0),
])
def test_abstention_reason_survives_hash_only_observation_serialization(raw, reason, score):
    item = NormalizedRecord(id='a', partition='discovery', input={}, expected=[], metadata={'kind':'abstention','group_id':'a'})
    observation = EvaluationExecutor().evaluate(Phase5EvaluationSuite(), item, raw)
    row = serialize_observation(observation, retain_raw_output=False)
    assert row['raw_output'] is None and row['parsed']['value'] is None
    assert row['scores']['abstention'] == {'value':score, 'numerator':score, 'denominator':1, 'reason':reason}
    assert json.loads(json.dumps(row))['scores']['abstention']['reason'] == reason


def test_optional_score_reason_is_bounded_and_absent_for_other_scores():
    assert EvaluationScore(1).reason is None
    with pytest.raises(ValueError, match='bounded reason'):
        EvaluationScore(0, reason='arbitrary private output')


def test_metric_score_event_attributes_are_namespaced():
    exporter = InMemorySpanExporter(); provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    suite = Phase5EvaluationSuite(telemetry=OperationTelemetry.create('evaluation', tracer=provider.get_tracer('review')))
    item = NormalizedRecord(id='a', partition='discovery', input={}, expected=[], metadata={'kind':'abstention','group_id':'a'})
    EvaluationExecutor().evaluate(suite, item, '[]')
    events = [e for s in exporter.get_finished_spans() for e in s.events if e.name=='metric_scored']
    assert len(events)==1
    assert events[0].attributes == {'capability_anatomy.metric_id':'abstention', 'capability_anatomy.metric_value':1.0}
    provider.shutdown()




def test_live_phase5_suite_refuses_mixed_fraction_reducers():
    from dataclasses import replace
    from capability_anatomy.evaluations.reduction import ScoreReductionError
    item=NormalizedRecord(id='a',partition='discovery',input={},expected=[],metadata={'kind':'abstention','group_id':'a'})
    suite=Phase5EvaluationSuite();first=EvaluationExecutor().evaluate(suite,item,'[]')
    second=replace(first,example_id='b',scores={'abstention':EvaluationScore(.5)})
    with pytest.raises(ScoreReductionError) as error:suite.aggregate([first,second])
    assert error.value.reason=='score_fraction_mixed'
