"""The operator-delivery marker: package-local BEHAVIOR tests.

The marker is a pinned copy of a brokkr TypeScript constant (see
docs/brokkr-integration.md). The cross-repo byte pin is owned by brokkr's CI — this suite
never reaches outside the package. What is asserted here is the package-local literal (so an
accidental in-package edit is caught locally) and the matching behavior: position-exact
prefix matching with the bounded header shape and the ECMAScript trimStart alphabet.
"""
from conftest import cq

# The package-local pin. Changing these bytes is a brokkr-coordinated change by contract.
MARKER = "message from the operator via the Control Room"


def header(delivery_id="d-123", cls="task"):
    return f"[{MARKER} | delivery {delivery_id} | class: {cls}]"


def test_package_local_literal():
    assert cq.OPERATOR_DELIVERY_MARKER == MARKER
    assert cq.OPERATOR_PROMPT_PREFIX == "[" + MARKER + " | delivery "


def test_full_header_matches_as_prefix():
    assert cq.OPERATOR_PROMPT_RE.match(header() + "\nplease do the thing")


def test_substring_occurrence_does_not_match():
    body = "quoting the framing: " + header()
    assert cq.OPERATOR_PROMPT_RE.match(body) is None


def test_half_written_header_does_not_match():
    # Bounded on BOTH sides: the opening prefix alone (no closing ` |` after a
    # non-empty id) must not attribute the turn.
    assert cq.OPERATOR_PROMPT_RE.match("[" + MARKER + " | delivery ") is None
    assert cq.OPERATOR_PROMPT_RE.match("[" + MARKER + " | delivery  |]") is None  # empty id


def test_leading_js_whitespace_is_tolerated():
    # Mirrors trimStart(): ordinary whitespace and U+FEFF are trimmed by the verifier,
    # so the composition must attribute the turn too.
    for ws in (" ", "\t", "\n", " ", "﻿"):
        assert cq.OPERATOR_PROMPT_RE.match(ws + header()), repr(ws)


def test_python_only_whitespace_is_not_tolerated():
    # \s alphabets differ: U+001C..U+001F match python's \s but are NOT trimmed by
    # ECMAScript trimStart, so the verifier rejects such a turn and this side must too.
    for ws in ("\x1c", "\x1d", "\x1e", "\x1f", "\x85"):
        assert cq.OPERATOR_PROMPT_RE.match(ws + header()) is None, repr(ws)


def test_bucket_via_the_real_entry_point():
    rec = {"uuid": "00000000-0000-4000-8000-0000000000aa", "type": "user",
           "message": {"role": "user", "content": header() + "\nship it"}}
    assert cq._human_prompt_bucket(rec, "user") == cq.OPERATOR_PROMPT_BUCKET
    typed = {"uuid": "00000000-0000-4000-8000-0000000000ab", "type": "user",
             "promptSource": "typed",
             "message": {"role": "user", "content": "ship it"}}
    assert cq._human_prompt_bucket(typed, "user") == "typed"
