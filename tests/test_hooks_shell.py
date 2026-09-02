"""The shell payload, end to end: the signal hook against a scratch HOME.

context-signal.sh is a Claude Code UserPromptSubmit hook: JSON on stdin, fail-safe by
contract (any error => exit 0, never blocks a prompt). These tests run the real script.
"""
import json
import os
import subprocess

from conftest import BIN, FIXTURE

HOOK = os.path.join(BIN, "context-signal.sh")
STATUSLINE = os.path.join(BIN, "statusline-command.sh")
SID = "ctxmon-pkg-test"


def run_hook(home, stdin, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k != "CTX_MONITOR_DIR"}
    env["HOME"] = home
    env.update(extra_env or {})
    return subprocess.run(["bash", HOOK], input=stdin, text=True,
                          env=env, capture_output=True, timeout=120)


def cache_path(base):
    return os.path.join(base, f"ctxmonitor-{SID}.json")


def hook_stdin():
    return json.dumps({"session_id": SID, "transcript_path": FIXTURE})


def test_scripts_parse(tmp_path):
    for script in (HOOK, STATUSLINE):
        subprocess.run(["bash", "-n", script], check=True)


def test_hook_writes_cache_under_the_default_home(tmp_path):
    home = str(tmp_path)
    p = run_hook(home, hook_stdin())
    assert p.returncode == 0, p.stderr
    cache = cache_path(os.path.join(home, ".ctxmonitor", "cache"))
    assert os.path.exists(cache), "score cache must land at ~/.ctxmonitor/cache"
    res = json.loads(open(cache).read())
    assert 0 <= res["score"] <= 100
    assert res["window"] == 1000000  # the fixture's whole-session view


def test_ctx_monitor_dir_override_wins(tmp_path):
    home = str(tmp_path / "home")
    drop = str(tmp_path / "drop")
    os.makedirs(home)
    p = run_hook(home, hook_stdin(), {"CTX_MONITOR_DIR": drop})
    assert p.returncode == 0, p.stderr
    assert os.path.exists(cache_path(drop)), "override must redirect the write"
    assert not os.path.exists(os.path.join(home, ".ctxmonitor")), \
        "the default home must be untouched when the override is set"


def test_relative_override_falls_back_to_home(tmp_path):
    home = str(tmp_path)
    p = run_hook(home, hook_stdin(), {"CTX_MONITOR_DIR": "relative/path"})
    assert p.returncode == 0, p.stderr
    assert os.path.exists(cache_path(os.path.join(home, ".ctxmonitor", "cache"))), \
        "a non-absolute override is ignored, not resolved against cwd"


def test_hook_is_fail_safe_on_garbage_stdin(tmp_path):
    home = str(tmp_path)
    p = run_hook(home, "this is not json")
    assert p.returncode == 0, "fail-safe contract: any error exits 0"


def test_hook_is_fail_safe_on_missing_transcript(tmp_path):
    home = str(tmp_path)
    p = run_hook(home, json.dumps({"session_id": SID,
                                   "transcript_path": "/nonexistent/x.jsonl"}))
    assert p.returncode == 0, p.stderr


def test_healthy_fixture_emits_no_nudge(tmp_path):
    # The golden session is far below the occupancy knee, so no additionalContext
    # note may be injected (the nudge fires only when degraded AND >= 65% full).
    p = run_hook(str(tmp_path), hook_stdin())
    assert p.returncode == 0
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        assert "additionalContext" not in json.dumps(payload) or \
            "[ctxmonitor]" not in payload.get("hookSpecificOutput", {}).get("additionalContext", ""), \
            "healthy session must not be nagged"
