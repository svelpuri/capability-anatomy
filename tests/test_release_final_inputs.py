"""Consumer regressions for honest generic policy and credential admission."""
import json
from pathlib import Path

import pytest

from capability_anatomy.cli import _example
from capability_anatomy.config import load_experiment_config
from capability_anatomy.credentials import CredentialReference
from capability_anatomy.errors import InvalidEvidenceError
from capability_anatomy.review_policy import validate_review_reference
from test_release_review_policy import reviewed_policy
from test_release_review_telemetry import cli, environment, consumer, attributes


@pytest.mark.parametrize('key', ['subscription_key', 'passphrase', 'privatekey', 'auth', 'signature',
    'session_key', 'master_key', 'serviceSubscriptionKey', 'service_passphrase', 'code'])
def test_credential_family_refuses_real_run_before_persistence(tmp_path, key):
    _example(tmp_path / 'example')
    path = tmp_path / 'example/experiment.json'
    value = json.loads(path.read_text())
    value['model']['parameters'][key] = 'FAKE-NONSECRET-FINAL-INPUT-CANARY'
    path.write_text(json.dumps(value))
    result = cli('run', '--config', path, env=environment())
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'CANARY' not in result.stdout + result.stderr
    assert not (path.parent / 'run/experiment-config.json').exists()


def test_legitimate_code_boolean_references_and_credential_environment_admit(tmp_path):
    _example(tmp_path / 'example')
    path = tmp_path / 'example/experiment.json'
    value = json.loads(path.read_text())
    value['model']['parameters'].update(trust_remote_code=False, authorization_file='approval.json',
        private_key_file='runtime-only.pem', credential_reference='service', tokenizer_revision='v1',
        max_new_tokens=3, authorization_policy='frozen')
    value['credentials'] = {'service': {'provider': 'env', 'name': 'CA_SERVICE_SECRET'}}
    path.write_text(json.dumps(value))
    assert load_experiment_config(path).credentials['service'] == CredentialReference('env', 'CA_SERVICE_SECRET')
    # Stock synthetic plugins do not consume credentials; exercise their
    # valid settings without falsely claiming that they support credential injection.
    value.pop('credentials'); path.write_text(json.dumps(value))
    result = cli('run', '--config', path, env=environment())
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli('verify', '--evidence', path.parent / 'run', env=environment()).returncode == 0


@pytest.mark.parametrize('url', ['https://example.invalid/?subscription_key=FAKE',
    'https://example.invalid/#passphrase=FAKE', 'https://example.invalid/#login?signature=FAKE'])
def test_credential_url_parameter_family_refuses(tmp_path, url):
    _example(tmp_path / 'example')
    path = tmp_path / 'example/experiment.json'
    value = json.loads(path.read_text())
    value['runtime']['parameters']['endpoint'] = url
    path.write_text(json.dumps(value))
    assert cli('run', '--config', path, env=environment()).returncode == 2
    assert not (path.parent / 'run/experiment-config.json').exists()


def test_generic_policy_is_not_silently_ignored_and_refusal_reaches_consumer(tmp_path):
    _example(tmp_path / 'example')
    path = tmp_path / 'example/experiment.json'
    value = json.loads(path.read_text())
    assert value.get('policy', {}) == {}
    with consumer() as (endpoint, spans, _):
        env = environment(endpoint); env['OTEL_EXPORTER_OTLP_TIMEOUT'] = '2'
        admitted = cli('run', '--config', path, env=env)
        assert admitted.returncode == 0, admitted.stderr
        assert any(attributes(span).get('capability_anatomy.reason') == 'schema_and_semantics_valid' for span in spans)
        value['output']['directory'] = 'refused'
        value['policy'] = {'file': 'does-not-exist-policy.json'}
        path.write_text(json.dumps(value)); spans.clear()
        refused = cli('run', '--config', path, env=env)
        assert refused.returncode == 2
        assert json.loads(refused.stderr.splitlines()[-1])['error'] == 'generic_policy_unsupported'
        assert not (path.parent / 'refused/experiment-config.json').exists()
        assert any(attributes(span).get('capability_anatomy.reason') == 'generic_policy_unsupported'
                   and span.status.code == 2 for span in spans)


@pytest.mark.parametrize('suffix', ['', '#review-29', '%252e%252e/17', '%252fother/17'])
def test_review_locator_requires_specific_canonical_review(suffix):
    protocol, approval = reviewed_policy()
    validate_review_reference(protocol, approval)
    approval['review_url'] = 'https://code.acme.test/research/anatomy/reviews/' + suffix
    with pytest.raises(InvalidEvidenceError):
        validate_review_reference(protocol, approval)
