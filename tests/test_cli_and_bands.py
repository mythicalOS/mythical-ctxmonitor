"""CLI surface + the band/emit gate, against the golden fixture."""
import json
import os
import subprocess
import sys

from conftest import BIN, FIXTURE, cq

SCORER = os.path.join(BIN, "context_quality.py")


def run_cli(*args, **kw):
    return subprocess.run([sys.executable, SCORER, *args],
                          capture_output=True, text=True, timeout=120, **kw)


def test_score_json_on_the_fixture():
    p = run_cli("score", FIXTURE, "--json")
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    # Structural contract the hooks and statusline rely on.
    assert isinstance(res["score"], (int, float)) and 0 <= res["score"] <= 100
    assert res["grade"]
    for sig in ("occupancy", "degradation", "cache_efficiency"):
        assert sig in res["signals"], sig
    # The whole-session view of the oracle, through the CLI.
    assert res["model"] == "claude-opus-4-10-20260401"
    assert res["window"] == 1000000
    assert res["signals"]["occupancy"]["raw"]["occupancy_fraction"] == 0.1875


def test_score_human_output_runs():
    p = run_cli("score", FIXTURE)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip()


def test_compose_and_cache_views_run_on_the_fixture():
    for view in ("compose", "cache"):
        p = run_cli(view, FIXTURE, "--json")
        assert p.returncode == 0, f"{view}: {p.stderr}"
        json.loads(p.stdout)


def test_missing_transcript_is_a_loud_error():
    p = run_cli("score", "/nonexistent/transcript.jsonl", "--json")
    assert p.returncode != 0


def test_band_gate_is_grade_first():
    # Fire only when score < 80 AND occupancy >= 0.65; A-grade is never nagged.
    assert cq.band_of(85, 0.80) == "ok"       # high grade wins even at high occupancy
    assert cq.band_of(60, 0.20) == "ok"       # frayed but low fill => silent
    assert cq.band_of(79, 0.649) == "ok"      # just under the occupancy knee
    assert cq.band_of(75, 0.65) == "advisory"  # both conditions met
    assert cq.band_of(78, 0.75) == "strong"
    assert cq.band_of(60, 0.90) == "strong"


def test_should_emit_is_edge_triggered():
    assert cq.should_emit(None, "ok")            # first observation always emits
    assert cq.should_emit("ok", "advisory")      # worsening
    assert cq.should_emit("strong", "ok")        # recovery
    assert not cq.should_emit("advisory", "advisory")
