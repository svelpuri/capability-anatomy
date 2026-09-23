"""Portable frozen review-location policy; source/scope admission is separate."""
import pytest

from capability_anatomy.review_policy import (
    AUTHORIZATION_FIELDS, AUTHORIZATION_V1, AUTHORIZATION_V2, PROTOCOL_V1, PROTOCOL_V2,
    REVIEW_POLICY_VERSION, validate_review_policy, validate_review_reference,
)
from capability_anatomy.serialization import canonical_sha256
from capability_anatomy.errors import InvalidEvidenceError


def reviewed_policy(origin='https://code.acme.test',prefix='/research/anatomy/reviews/'):
    policy={'schema_version':REVIEW_POLICY_VERSION,'policy_id':'acme-research',
            'allowed_locations':[{'origin':origin,'path_prefix':prefix}]}
    protocol={'schema_version':PROTOCOL_V2,'review_policy':policy}
    approval={key:False for key in AUTHORIZATION_FIELDS}
    approval.update(schema_version=AUTHORIZATION_V2,status='approved',conformance_authorized=True,
        full_scan_authorized=False,protocol_sha256='1'*64,approved_commit='2'*40,
        approved_source_sha256='3'*64,review_url=origin+prefix+'17#review-29',review_policy_sha256=canonical_sha256(policy))
    return protocol,approval


@pytest.mark.parametrize('origin,prefix',[
    ('https://code.acme.test','/research/anatomy/reviews/'),
    ('https://gitlab.example.org','/another/team/-/merge_requests/'),
    ('https://forge.other.test:8443','/changes/'),
])
def test_review_location_policy_is_provider_and_author_independent(origin,prefix):
    protocol,approval=reviewed_policy(origin,prefix)
    validate_review_policy(protocol['review_policy'])
    validate_review_reference(protocol,approval)


@pytest.mark.parametrize('change',[
    'different_origin','different_project','encoded_parent','credentials','query','policy_hash','empty_locations','legacy',
])
def test_review_policy_changes_do_not_grant_admission(change):
    protocol,approval=reviewed_policy()
    if change=='different_origin':approval['review_url']='https://evil.example/research/anatomy/reviews/17'
    elif change=='different_project':approval['review_url']='https://code.acme.test/another/reviews/17'
    elif change=='encoded_parent':approval['review_url']='https://code.acme.test/research/anatomy/reviews/%2e%2e/17'
    elif change=='credentials':approval['review_url']='https://secret@code.acme.test/research/anatomy/reviews/17'
    elif change=='query':approval['review_url']='https://code.acme.test/research/anatomy/reviews/17?access_token=secret'
    elif change=='policy_hash':approval['review_policy_sha256']='0'*64
    elif change=='empty_locations':protocol['review_policy']['allowed_locations']=[]
    elif change=='legacy':
        protocol={'schema_version':PROTOCOL_V1};approval.pop('review_policy_sha256');approval['schema_version']=AUTHORIZATION_V1
    with pytest.raises(InvalidEvidenceError,match='review policy'):
        validate_review_reference(protocol,approval)


def test_legacy_review_locator_is_read_only_and_never_new_execution_authority():
    _protocol,approval=reviewed_policy();approval.pop('review_policy_sha256');approval['schema_version']=AUTHORIZATION_V1
    protocol={'schema_version':PROTOCOL_V1}
    validate_review_reference(protocol,approval,allow_legacy=True)
    with pytest.raises(InvalidEvidenceError):validate_review_reference(protocol,approval)
