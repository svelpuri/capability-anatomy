"""Operator-configured review locations, sealed by the governed source protocol.

A locator is an audit reference, not proof of reviewer identity. Only separately
validated source/protocol identity and explicit operation scope grant admission.
"""
from __future__ import annotations

import posixpath
import re
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .errors import InvalidConfigurationError, InvalidEvidenceError
from .serialization import canonical_sha256

PROTOCOL_V1 = 'capability-anatomy/phase5-protocol/v1'
PROTOCOL_V2 = 'capability-anatomy/phase5-protocol/v2'
AUTHORIZATION_V1 = 'capability-anatomy/gate5a-authorization/v1'
AUTHORIZATION_V2 = 'capability-anatomy/gate5a-authorization/v2'
REVIEW_POLICY_VERSION = 'capability-anatomy/review-policy/v1'
AUTHORIZATION_FIELDS = frozenset({'schema_version','status','conformance_authorized','full_scan_authorized',
                                'protocol_sha256','approved_commit','approved_source_sha256','review_url'})


def _https_location(value: Any):
    if not isinstance(value,str) or not value or len(value)>4096 or any(c.isspace() or ord(c)<32 for c in value):
        raise ValueError('invalid review location')
    parts=urlsplit(value)
    if parts.scheme!='https' or not parts.hostname or parts.username is not None or parts.password is not None or parts.query:
        raise ValueError('invalid review location')
    port=parts.port
    origin='https://'+parts.hostname.lower()+(f':{port}' if port not in (None,443) else '')
    path=unquote(parts.path,errors='strict')
    if '%' in path or '\\' in path or any(ord(c)<32 for c in path) or not path.startswith('/'):
        raise ValueError('invalid review path')
    if posixpath.normpath(path)!=path.rstrip('/') or path.startswith('//'):
        raise ValueError('noncanonical review path')
    if parts.fragment and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,255}',parts.fragment):
        raise ValueError('invalid review anchor')
    return origin,path,parts.fragment


def validate_review_policy(value: Any) -> None:
    try:
        if not isinstance(value,dict) or set(value)!={'schema_version','policy_id','allowed_locations'}:
            raise ValueError('policy fields')
        if value['schema_version']!=REVIEW_POLICY_VERSION or not isinstance(value['policy_id'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',value['policy_id']):
            raise ValueError('policy identity')
        locations=value['allowed_locations']
        if not isinstance(locations,list) or not 1<=len(locations)<=32:
            raise ValueError('policy locations')
        seen=set()
        for location in locations:
            if not isinstance(location,dict) or set(location)!={'origin','path_prefix'}:
                raise ValueError('location fields')
            origin,prefix=location['origin'],location['path_prefix']
            if not isinstance(origin,str) or not isinstance(prefix,str) or not prefix.endswith('/') or prefix=='/':
                raise ValueError('location prefix')
            normalized,path,anchor=_https_location(origin+prefix)
            if origin!=normalized or path!=prefix or anchor or (origin,prefix) in seen:
                raise ValueError('noncanonical or duplicate location')
            seen.add((origin,prefix))
    except (TypeError,ValueError,UnicodeError):
        raise InvalidConfigurationError('Phase 5 review policy is invalid') from None


def validate_review_reference(protocol: Mapping[str,Any], authorization: Mapping[str,Any], *, allow_legacy: bool=False) -> None:
    """Validate the reviewed location policy; callers also enforce source/scope."""
    try:
        version=protocol.get('schema_version')
        legacy=version==PROTOCOL_V1 and authorization.get('schema_version')==AUTHORIZATION_V1
        if legacy:
            if not allow_legacy or set(authorization)!=AUTHORIZATION_FIELDS:
                raise ValueError('legacy approval is reconstruction only')
            _https_location(authorization.get('review_url'))
            return
        if version!=PROTOCOL_V2 or authorization.get('schema_version')!=AUTHORIZATION_V2 or set(authorization)!=AUTHORIZATION_FIELDS|{'review_policy_sha256'}:
            raise ValueError('versioned review policy required')
        policy=protocol.get('review_policy')
        validate_review_policy(policy)
        if authorization.get('review_policy_sha256')!=canonical_sha256(policy):
            raise ValueError('review policy identity changed')
        origin,path,_anchor=_https_location(authorization.get('review_url'))
        if not any(origin==location['origin'] and path.startswith(location['path_prefix']) and len(path)>len(location['path_prefix']) for location in policy['allowed_locations']):
            raise ValueError('reference outside reviewed locations')
    except (InvalidConfigurationError,TypeError,ValueError,UnicodeError):
        raise InvalidEvidenceError('Gate 5A review policy or reference is not authorized') from None
