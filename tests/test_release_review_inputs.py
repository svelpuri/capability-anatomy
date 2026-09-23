import json
from pathlib import Path
import subprocess
import sys

import pytest

from capability_anatomy.authored_inputs import load_mapping
from capability_anatomy.credentials import InlineCredentialError, reject_inline_credentials


@pytest.mark.parametrize('key', ['id_token','code','assertion','client_assertion','aws_session_token',
    'aws_secret_access_key','sas_token','account_key','connection_string','session_cookie',
    'proxy_authorization','db_credentials','awsSecretAccessKey','custom_service_access_token'])
def test_one_credential_predicate_covers_provider_prefixes(key):
    with pytest.raises(InlineCredentialError):
        reject_inline_credentials({'runtime':{'parameters':{key:'fake-sensitive-material'}}})
    reject_inline_credentials({'runtime':{'parameters':{'max_new_tokens':12,'tokenizer_revision':'frozen'}}})


@pytest.mark.parametrize('value', [
    'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYW5hcnkifQ.ZmFrZS1zaWduYXR1cmU',
    'ASIAABCDEFGHIJKLMNOP', 'IQoJb3JpZ2luX2Vj'+'A'*64,
])
def test_recognized_credential_values_refuse(value):
    with pytest.raises(InlineCredentialError):
        reject_inline_credentials({'runtime':{'parameters':{'opaque':value}}})


def test_valid_json_uses_json_grammar_and_actual_generated_configuration(tmp_path):
    path=tmp_path/'long.json';expected={'x'*2000: {'nested':[1,2,3]}}
    path.write_text(json.dumps(expected,indent='\t'));assert load_mapping(path)==expected
    created=subprocess.run([sys.executable,'-m','capability_anatomy.cli','example','--output',str(tmp_path/'example')],capture_output=True,text=True)
    assert created.returncode==0,created.stderr
    config=Path(json.loads(created.stdout)['configuration']);config.write_text(json.dumps(json.loads(config.read_text()),indent='\t'))
    result=subprocess.run([sys.executable,'-m','capability_anatomy.cli','run','--config',str(config)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_inline_provider_credentials_refuse_actual_public_run_before_persistence(tmp_path):
    created=subprocess.run([sys.executable,'-m','capability_anatomy.cli','example','--output',str(tmp_path/'example')],capture_output=True,text=True)
    assert created.returncode==0
    config=Path(json.loads(created.stdout)['configuration']);doc=json.loads(config.read_text())
    doc['runtime']['parameters']['aws_secret_access_key']='fake-aws-material'
    config.write_text(json.dumps(doc))
    result=subprocess.run([sys.executable,'-m','capability_anatomy.cli','run','--config',str(config)],capture_output=True,text=True)
    assert result.returncode==2
    assert 'fake-aws-material' not in result.stderr
    assert not (config.parent/'run/experiment-config.json').exists()


def test_shipped_document_limits_are_enforced_without_overrides(tmp_path):
    from capability_anatomy.authored_inputs import AuthoredInputError
    path = tmp_path / 'document.json'
    path.write_text('{"positive": [1, 2]}')
    assert load_mapping(path) == {"positive": [1, 2]}
    path.write_text('{"large":"' + 'x' * (8 * 1024 * 1024) + '"}')
    with pytest.raises(AuthoredInputError) as caught:
        load_mapping(path)
    assert caught.value.reason == 'document_byte_limit'
    path.write_text(json.dumps({'nodes': [0] * 100001}))
    with pytest.raises(AuthoredInputError) as caught:
        load_mapping(path)
    assert caught.value.reason == 'document_node_limit'
