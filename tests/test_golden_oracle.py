"""The oracle pin: assert bin/testdata/golden-session.expected.json over the fixture.

This is one half of a cross-language pair — the brokkr integration's TypeScript suite
asserts the SAME values from the SAME two files against its mirror module. A failure here is
never fixed by regenerating the expected file from program output: decide which language is
right, fix both, then restate the expectation (the expected file's own _comment carries the
rule and the hand-computed provenance).

Everything goes through the real entry points (read_jsonl, effective_input,
scoreable_assistants, detect_window, score_session, cache_efficiency_signal) so the
assertions travel with the definitions.

The `known_divergences` section of the oracle records where the two implementations
deliberately disagree; this suite asserts the PYTHON half of each recorded case, so a change
to this side's behavior lands on the record instead of silently widening the gap.
"""
import json
import os

import pytest

from conftest import EXPECTED, FIXTURE, cq, load_expected, load_records

RECORDS = load_records()
E = load_expected()


def view_items():
    return sorted(E["views"].items())


def prefix(view):
    """The records a view scores: 0 .. last_record_index inclusive."""
    return RECORDS[: view["last_record_index"] + 1]


# -- the fixture files themselves -------------------------------------------

def test_fixture_files_are_ascii_and_nul_free():
    # A NUL byte hides a file from common grep configurations silently; ASCII keeps
    # the two languages' loaders from disagreeing over decoding.
    for path in (FIXTURE, EXPECTED):
        with open(path, "rb") as f:
            raw = f.read()
        assert b"\x00" not in raw, f"{path} contains a NUL byte"
        raw.decode("ascii")


def test_record_count_and_unique_uuids():
    assert len(RECORDS) == E["record_count"]
    uuids = [r["uuid"] for r in RECORDS]
    assert len(set(uuids)) == len(uuids)


# -- effective_input per record, independent of the scoreable predicate ------

def test_effective_input_of_every_record_carrying_usage():
    exp = E["effective_input_by_index"]
    have_usage = {i for i, r in enumerate(RECORDS) if cq._usage(r) is not None}
    assert have_usage == {int(k) for k in exp}, "the set of usage-carrying records moved"
    for key, want in exp.items():
        i = int(key)
        assert cq.effective_input(cq._usage(RECORDS[i])) == want, f"record {i}"


# -- the model -> window map, read directly -----------------------------------

def test_window_map_probe():
    for model, want in E["window_map_probe"].items():
        if model == "why":
            continue
        assert cq.detect_window(model) == want, model


# -- per-view assertions -------------------------------------------------------

@pytest.mark.parametrize("name", [n for n, _ in view_items()])
def test_scoreable_verdicts_per_view(name):
    view = E["views"][name]
    sub = prefix(view)
    asst = cq.scoreable_assistants(sub)
    got_uuids = [r["uuid"] for r in asst]
    selected = set(got_uuids)
    got_indices = [i for i, r in enumerate(sub) if r["uuid"] in selected]
    assert got_indices == view["scoreable_indices"]
    assert got_uuids == view["scoreable_uuids"]
    assert len(asst) == view["scoreable_turn_count"]


@pytest.mark.parametrize("name", [n for n, _ in view_items()])
def test_final_turn_model_window_occupancy_per_view(name):
    view = E["views"][name]
    res = cq.score_session(prefix(view))
    raw = res["signals"]["occupancy"]["raw"]
    assert res["model"] == view["final_model"]
    assert res["window"] == view["window"]
    assert raw["effective_input"] == view["final_effective_input"]
    # Exact, not approximate: the fixture's numbers are exact at 4 decimal places.
    assert raw["occupancy_fraction"] == view["occupancy_fraction"]


@pytest.mark.parametrize("name", [n for n, _ in view_items()])
def test_cache_totals_and_ratio_per_view(name):
    view = E["views"][name]
    _score, raw = cq.cache_efficiency_signal(prefix(view))
    assert raw["total_cache_read"] == view["sum_cache_read"]
    assert raw["total_effective"] == view["sum_effective_input"]
    assert raw["ratio"] == view["cache_ratio"]
    assert raw["assistant_turns"] == view["scoreable_turn_count"]


# -- the recorded divergences: assert the PYTHON half of each case -------------

def test_divergence_effective_input_coercion():
    cases = E["known_divergences"]["effective_input_coercion"]
    assert cases
    for case in cases:
        assert cq.effective_input(case["usage"]) == case["python"], case["usage"]


def test_divergence_reported_fraction_rounding():
    div = E["known_divergences"]["reported_fraction_rounding"]
    assert div["occupancy_cases"] and div["cache_ratio_cases"]
    for case in div["occupancy_cases"]:
        assert cq.detect_window(case["model"]) == case["window"]
        rec = {"type": "assistant",
               "message": {"model": case["model"],
                           "usage": {"input_tokens": case["input_tokens"]}}}
        _s, raw = cq.occupancy_signal([rec], case["window"])
        assert raw["occupancy_fraction"] == case["python_reported"], case
    for case in div["cache_ratio_cases"]:
        rec = {"type": "assistant",
               "message": {"model": case["model"],
                           "usage": {"input_tokens": case["input_tokens"],
                                     "cache_read_input_tokens": case["cache_read_input_tokens"]}}}
        _s, raw = cq.cache_efficiency_signal([rec])
        assert raw["ratio"] == case["python_reported"], case


def test_divergence_usage_shape():
    cases = E["known_divergences"]["scoreable_predicate_usage_shape"]
    assert cases
    for case in cases:
        rec = {"type": "assistant",
               "message": {"model": case["model"], "usage": case["usage"]}}
        assert cq._is_real_assistant(rec) is case["python_scoreable"], case


def test_divergence_is_sidechain_truthiness():
    cases = E["known_divergences"]["record_shape_is_sidechain_truthiness"]
    assert cases
    for case in cases:
        assert cq._is_real_assistant(case["record"]) is case["python_scoreable"], case


def test_divergence_absent_model_window():
    cases = E["known_divergences"]["absent_model_window"]
    assert cases
    for case in cases:
        rec = case["record"]
        assert cq._is_real_assistant(rec) is case["python_scoreable"], case
        res = cq.score_session([rec])
        raw = res["signals"]["occupancy"]["raw"]
        assert res["window"] == case["python_window"], case
        assert raw["occupancy_fraction"] == case["python_occupancy_fraction"], case
    # An ABSENT model key behaves identically to explicit null on this side.
    null_case = cases[0]
    absent = {"uuid": "00000000-0000-4000-8000-0000000000ff", "type": "assistant",
              "message": {"role": "assistant",
                          "usage": dict(null_case["record"]["message"]["usage"])}}
    res = cq.score_session([absent])
    assert res["window"] == null_case["python_window"]


def test_divergence_no_scoreable_turn():
    cases = E["known_divergences"]["no_scoreable_turn"]
    assert cases
    for case in cases:
        records = case["records"]
        assert len(cq.scoreable_assistants(records)) == case["scoreable_count"]
        res = cq.score_session(records)
        occ_raw = res["signals"]["occupancy"]["raw"]
        assert res["window"] == case["python_window"]
        assert occ_raw["occupancy_fraction"] == case["python_occupancy_fraction"]
        _s, cache_raw = cq.cache_efficiency_signal(records)
        assert cache_raw["ratio"] is None if case["python_cache_ratio"] is None \
            else cache_raw["ratio"] == case["python_cache_ratio"]


def test_divergence_loader_keeps_nonstandard_json_tokens(tmp_path):
    # Divergence #7, prose-only in the oracle: python's json.loads accepts bare
    # NaN/Infinity tokens, so read_jsonl KEEPS such a record (the TS loader skips it).
    p = tmp_path / "nonstandard.jsonl"
    p.write_text('{"uuid": "u1", "type": "assistant", "x": NaN}\n'
                 '{"uuid": "u2", "type": "user"}\n')
    records = cq.read_jsonl(str(p))
    assert [r["uuid"] for r in records] == ["u1", "u2"]
