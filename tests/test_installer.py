"""The settings transaction engine + the install.sh driver, hermetically.

Engine cases import lib/settings_txn.py directly and drive the race points through the
_between_read_and_publish test seam (and one monkeypatch of os.replace for the
post-re-read swap). Driver cases run the real install.sh with HOME, CTXMONITOR_HOME and
CTXMONITOR_SETTINGS redirected into a temp tree. Nothing here touches the real machine.
"""
import json
import os
import pathlib
import stat
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "lib"))
import settings_txn as txn  # noqa: E402

HOOK = "bash /x/.ctxmonitor/bin/context-signal.sh"
SL = "bash /x/.ctxmonitor/bin/statusline-command.sh"


def install(path, **kw):
    return txn.run(path, "install", HOOK, SL, **kw)


def uninstall(path, **kw):
    return txn.run(path, "uninstall", HOOK, SL, **kw)


def read(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Creation / idempotency / preservation

def test_absent_file_created_0600_with_only_our_entries(tmp_path):
    p = str(tmp_path / "settings.json")
    r = install(p)
    assert r["wrote"] and r["created"] and r["backup"] is None
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    obj = read(p)
    assert obj["hooks"]["UserPromptSubmit"] == [
        {"hooks": [{"type": "command", "command": HOOK, "timeout": txn.HOOK_TIMEOUT}]}]
    assert obj["statusLine"] == {"type": "command", "command": SL}
    assert set(obj) == {"hooks", "statusLine"}


def test_empty_object_gains_both_entries(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    r = install(p)
    assert r["wrote"] and not r["created"]
    obj = read(p)
    assert obj["statusLine"]["command"] == SL
    assert txn._hook_present(obj, HOOK)


def test_foreign_hooks_survive_semantically(tmp_path):
    p = str(tmp_path / "settings.json")
    foreign = {
        "hooks": {"UserPromptSubmit": [
            {"matcher": "x", "hooks": [{"type": "command", "command": "their-thing"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "their-stop"}]}]},
        "permissions": {"allow": ["Read"]},
    }
    with open(p, "w") as f:
        json.dump(foreign, f)
    install(p)
    obj = read(p)
    assert obj["hooks"]["Stop"] == foreign["hooks"]["Stop"]
    assert obj["hooks"]["UserPromptSubmit"][0] == foreign["hooks"]["UserPromptSubmit"][0]
    assert obj["permissions"] == foreign["permissions"]
    assert txn._hook_present(obj, HOOK)


def test_existing_statusline_is_never_touched(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        json.dump({"statusLine": {"type": "command", "command": "their-statusline"}}, f)
    r = install(p)
    assert r["statusline"] == "kept-existing"
    assert read(p)["statusLine"]["command"] == "their-statusline"


def test_our_own_statusline_reports_already_ours_not_kept_existing(tmp_path):
    # A re-run where the present statusLine is OURS is "already-ours" (surfaced as
    # "on-newest"), not "kept-existing" (which means a FOREIGN statusline).
    p = str(tmp_path / "settings.json")
    install(p)                      # sets ours
    r = install(p)                  # re-run: ours is present
    assert r["statusline"] == "already-ours"
    assert read(p)["statusLine"]["command"] == SL


def test_double_install_is_a_noop(tmp_path):
    p = str(tmp_path / "settings.json")
    install(p)
    before = open(p, "rb").read()
    r = install(p)
    assert r.get("noop") and not r["wrote"]
    assert open(p, "rb").read() == before


def test_mode_preserved_across_rewrite(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    os.chmod(p, 0o644)
    # Exact preservation must not be filtered through the ambient umask.
    old_umask = os.umask(0o077)
    try:
        install(p)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o644


def test_chmod_landing_mid_transaction_is_honored(tmp_path):
    # A chmod between the initial read and the final re-read passes the
    # hash/identity checks (same bytes, same inode) — the published file must
    # carry the NEW mode, not revert to the stale initial one.
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    os.chmod(p, 0o644)

    def chmod_race():
        os.chmod(p, 0o640)

    install(p, _between_read_and_publish=chmod_race)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o640


def test_short_writes_never_publish_torn_content(tmp_path, monkeypatch):
    # write(2) may return short; every write path — the create publish, the
    # replace publish, AND the disaster-recovery backup — must loop to completion.
    p = str(tmp_path / "settings.json")
    real_write = os.write

    def one_byte_writes(fd, data):
        return real_write(fd, bytes(data[:1]))

    monkeypatch.setattr(txn.os, "write", one_byte_writes)
    r = install(p)  # create path
    assert r["wrote"] and r["created"]
    obj = read(p)  # parses ⇒ not torn
    assert txn._hook_present(obj, HOOK)
    r2 = uninstall(p)  # replace path + backup path, still under 1-byte writes
    assert r2["wrote"] and r2["backup"]
    assert json.load(open(r2["backup"])) == obj, "the backup must be complete, not torn"
    assert not txn._hook_present(read(p), HOOK)


# --------------------------------------------------------------------------- #
# Malformed shapes: fail WITHOUT writing

@pytest.mark.parametrize("content", [
    '{"hooks": {"UserPromptSubmit": {"not": "an array"}}}',
    '{"hooks": {"UserPromptSubmit": ["not-an-object"]}}',
    '{"hooks": {"UserPromptSubmit": [{"hooks": "not-an-array"}]}}',
    '{"hooks": {"UserPromptSubmit": [{"hooks": ["not-an-object"]}]}}',
    '{"statusLine": "not-an-object"}',
    '{"hooks": "not-an-object"}',
    '["top-level-array"]',
    'not json at all',
    # An explicit null is a malformed shape, not an absence.
    '{"statusLine": null}',
    '{"hooks": null}',
    '{"hooks": {"UserPromptSubmit": null}}',
    '{"hooks": {"UserPromptSubmit": [{"hooks": null}]}}',
])
def test_malformed_shapes_fail_without_write(tmp_path, content):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write(content)
    with pytest.raises(txn.TxnError):
        install(p)
    assert open(p).read() == content, "a refusal must write nothing"


# --------------------------------------------------------------------------- #
# Symlinks

def test_preexisting_symlink_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    p = str(tmp_path / "settings.json")
    os.symlink(str(real), p)
    with pytest.raises(txn.TxnError) as e:
        install(p)
    assert e.value.reason == "symlink-refused"
    assert real.read_text() == "{}", "the transaction must never write through a symlink"


def test_swap_to_symlink_before_the_final_reread_is_refused(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    real = tmp_path / "victim.json"
    real.write_text("{}")

    def swap():
        os.unlink(p)
        os.symlink(str(real), p)

    with pytest.raises(txn.TxnError) as e:
        install(p, _between_read_and_publish=swap)
    assert e.value.reason == "symlink-refused"
    assert real.read_text() == "{}"
    # A refusal leaves the tree untouched — no backup sidecar, no leaked temp.
    leftovers = [f for f in os.listdir(tmp_path) if "ctxmonitor" in f]
    assert leftovers == [], leftovers


def test_swap_landing_after_the_final_reread_is_safe_not_refused(tmp_path, monkeypatch):
    # The narrowest window: a symlink lands between the pre-publish re-read and the
    # rename. rename(2) replaces the final path component WITHOUT following it, so the
    # racing symlink itself is replaced and the file it pointed at is untouched — a safe
    # outcome by design, asserted as such (NOT a refusal).
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    victim = tmp_path / "victim.json"
    victim.write_text('{"their": "data"}')

    real_replace = os.replace

    def swapping_replace(src, dst):
        if dst == p:
            os.unlink(p)
            os.symlink(str(victim), p)
        return real_replace(src, dst)

    monkeypatch.setattr(txn.os, "replace", swapping_replace)
    r = install(p)
    assert r["wrote"]
    assert not os.path.islink(p), "the racing symlink was replaced by the real file"
    assert victim.read_text() == '{"their": "data"}', "the symlink's target is untouched"
    assert txn._hook_present(read(p), HOOK)


# --------------------------------------------------------------------------- #
# Concurrency

def test_concurrent_creator_on_the_enoent_branch(tmp_path):
    # The exclusive link-publish loses to a creator that appeared mid-transaction;
    # the transaction re-enters the read path and MERGES rather than renaming over it.
    p = str(tmp_path / "settings.json")
    fired = {"n": 0}

    def create_concurrently():
        if fired["n"] == 0:
            with open(p, "w") as f:
                json.dump({"permissions": {"allow": ["Read"]}}, f)
        fired["n"] += 1

    r = install(p, _between_read_and_publish=create_concurrently)
    assert r["wrote"]
    obj = read(p)
    assert obj["permissions"] == {"allow": ["Read"]}, "the concurrent creator's content survives"
    assert txn._hook_present(obj, HOOK)
    assert fired["n"] >= 2, "the exclusive publish must have re-entered the read path"


def test_interleaved_external_edit_detected_and_retried(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    fired = {"n": 0}

    def external_edit():
        if fired["n"] == 0:
            with open(p, "w") as f:
                json.dump({"env": {"THEIRS": "1"}}, f)
        fired["n"] += 1

    r = install(p, _between_read_and_publish=external_edit)
    assert r["wrote"]
    obj = read(p)
    assert obj["env"] == {"THEIRS": "1"}, "the external edit survives the retry"
    assert txn._hook_present(obj, HOOK)


def test_aba_restore_is_undetectable_and_harmless(tmp_path):
    # An external writer that changes the file and restores the EXACT original bytes
    # before our publish is undetectable by construction — and harmless: our merge
    # re-applies onto content identical to what it validated. Asserted harmless, not
    # detected.
    p = str(tmp_path / "settings.json")
    original = '{\n  "permissions": {}\n}'
    with open(p, "w") as f:
        f.write(original)

    def a_b_a():
        with open(p, "w") as f:
            f.write('{"intermediate": true}')
        with open(p, "w") as f:
            f.write(original)

    r = install(p, _between_read_and_publish=a_b_a)
    assert r["wrote"]
    obj = read(p)
    assert obj["permissions"] == {}
    assert "intermediate" not in obj, "the intermediate state was the external writer's to discard"
    assert txn._hook_present(obj, HOOK)


def test_retries_exhaust_loudly(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write("{}")
    n = {"i": 0}

    def always_edit():
        n["i"] += 1
        with open(p, "w") as f:
            json.dump({"tick": n["i"]}, f)

    with pytest.raises(txn.TxnError) as e:
        install(p, _between_read_and_publish=always_edit)
    assert e.value.reason == "retries-exhausted"


# --------------------------------------------------------------------------- #
# Uninstall / dry-run / backup

def test_uninstall_is_surgical(tmp_path):
    p = str(tmp_path / "settings.json")
    foreign = {"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": "their-thing"}]}]},
        "env": {"K": "v"}}
    with open(p, "w") as f:
        json.dump(foreign, f)
    install(p)
    r = uninstall(p)
    assert r["hook"] == "removed" and r["statusline"] == "removed"
    obj = read(p)
    assert obj["hooks"]["UserPromptSubmit"] == foreign["hooks"]["UserPromptSubmit"]
    assert obj["env"] == {"K": "v"}
    assert "statusLine" not in obj
    assert not txn._hook_present(obj, HOOK)


def test_uninstall_keeps_foreign_empty_and_matcher_groups(tmp_path):
    # A group that was ALREADY empty, and a matcher group our removal empties,
    # are foreign intent: only a group OUR removal emptied — and that carries no
    # foreign keys — may be dropped.
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        json.dump({"hooks": {"UserPromptSubmit": [
            {"hooks": []},                                    # foreign, already empty
            {"matcher": "x", "hooks": [
                {"type": "command", "command": HOOK, "timeout": 15}]},  # ours inside a matcher group
        ]}}, f)
    r = uninstall(p)
    assert r["hook"] == "removed"
    ups = read(p)["hooks"]["UserPromptSubmit"]
    assert {"hooks": []} in ups, "the pre-existing empty foreign group survives"
    assert {"matcher": "x", "hooks": []} in ups, "the matcher group's foreign key survives"


def test_uninstall_leaves_foreign_statusline(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        json.dump({"statusLine": {"type": "command", "command": "their-statusline"}}, f)
    r = uninstall(p)
    assert r["statusline"] == "absent"
    assert read(p)["statusLine"]["command"] == "their-statusline"


def test_dry_run_writes_nothing_and_shows_the_full_diff(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write('{"permissions": {}}')
    before = open(p, "rb").read()
    r = install(p, dry_run=True)
    assert r["dry_run"] and not r["wrote"]
    assert open(p, "rb").read() == before
    assert "context-signal.sh" in r["diff"]
    assert "+" in r["diff"] and "---" in r["diff"], "a full unified diff, normalization included"


def test_dry_run_on_absent_file_writes_nothing(tmp_path):
    p = str(tmp_path / "settings.json")
    r = install(p, dry_run=True)
    assert not os.path.exists(p)
    assert r["dry_run"]


def test_backup_taken_before_every_replace(tmp_path):
    p = str(tmp_path / "settings.json")
    with open(p, "w") as f:
        f.write('{"env": {"A": "1"}}')
    r = install(p)
    assert r["backup"] and os.path.exists(r["backup"])
    assert json.load(open(r["backup"])) == {"env": {"A": "1"}}, \
        "the backup is the pre-write bytes (disaster recovery only)"


# --------------------------------------------------------------------------- #
# The install.sh driver, end to end (hermetic tree)

INSTALL_SH = os.path.join(REPO, "install.sh")


def driver(tmp_path, *args, path_override=None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    dest = home / ".ctxmonitor"
    settings = home / ".claude" / "settings.json"
    env = dict(os.environ)
    env.update(HOME=str(home), CTXMONITOR_HOME=str(dest),
               CTXMONITOR_SETTINGS=str(settings))
    if path_override is not None:
        env["PATH"] = path_override
    p = subprocess.run(["bash", INSTALL_SH, *args], env=env,
                       capture_output=True, text=True, timeout=120)
    return p, dest, settings


def test_driver_fresh_install_then_idempotent(tmp_path):
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0, p.stderr
    for f in ("bin/context_quality.py", "bin/context-signal.sh",
              "bin/statusline-command.sh", "bin/testdata/golden-session.jsonl",
              "bin/testdata/golden-session.expected.json", "lib/settings_txn.py"):
        assert (dest / f).exists(), f
    assert os.access(dest / "bin/context-signal.sh", os.X_OK), "modes preserved"
    obj = json.load(open(settings))
    assert stat.S_IMODE(os.stat(settings).st_mode) == 0o600
    assert obj["statusLine"]["command"].endswith("statusline-command.sh'"), \
        "the registered command is the shell-quoted absolute form"
    p2, _, _ = driver(tmp_path, "install")
    assert p2.returncode == 0
    assert json.load(open(settings)) == obj, "second install is a settings no-op"


def test_driver_keeps_existing_statusline_and_prints_recipe(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(
        '{"statusLine": {"type": "command", "command": "their-statusline"}}')
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0, p.stderr
    assert json.load(open(settings))["statusLine"]["command"] == "their-statusline"
    assert "statusline-command.sh" in p.stdout, "the manual wrapping recipe is printed"


def test_driver_dry_run_copies_and_writes_nothing(tmp_path):
    p, dest, settings = driver(tmp_path, "install", "--dry-run")
    assert p.returncode == 0, p.stderr
    assert not dest.exists(), "dry run must not copy program files"
    assert not settings.exists(), "dry run must not create settings"
    assert "context-signal.sh" in p.stdout, "the diff is shown"


def test_driver_uninstall_round_trip(tmp_path):
    driver(tmp_path, "install")
    p, dest, settings = driver(tmp_path, "uninstall")
    assert p.returncode == 0, p.stderr
    assert not dest.exists()
    obj = json.load(open(settings))
    assert "statusLine" not in obj
    assert not (obj.get("hooks") or {}).get("UserPromptSubmit")


def test_driver_hard_fails_without_jq(tmp_path):
    # A PATH with python3+bash but no jq: the installer must refuse loudly and
    # copy nothing — never a half-install.
    shims = tmp_path / "shims"
    shims.mkdir()
    for tool in ("python3", "bash", "dirname", "mkdir", "cp", "cat", "rm", "uname"):
        real = subprocess.run(["which", tool], capture_output=True, text=True).stdout.strip()
        if real:
            os.symlink(real, shims / tool)
    p, dest, settings = driver(tmp_path, "install", path_override=str(shims))
    assert p.returncode == 1
    assert "jq" in p.stderr and "remediation" in p.stderr
    assert not dest.exists() and not settings.exists()


def test_driver_handles_a_home_with_spaces(tmp_path):
    # The registered command strings are shell-quoted, so a DEST with spaces
    # must round-trip: install → registered as ONE argument → uninstall finds
    # the exact same string.
    home = tmp_path / "ho me"
    home.mkdir()
    dest = home / ".ctx monitor"
    settings = home / ".claude" / "settings.json"
    env = dict(os.environ)
    env.update(HOME=str(home), CTXMONITOR_HOME=str(dest),
               CTXMONITOR_SETTINGS=str(settings))
    p = subprocess.run(["bash", INSTALL_SH, "install"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    obj = json.load(open(settings))
    cmd = obj["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert cmd == f"bash '{dest}/bin/context-signal.sh'", cmd
    # The quoted command must actually execute the hook (fail-safe exit 0 either way,
    # but the cache write only happens when bash resolved the path as one word).
    import shlex
    parts = shlex.split(cmd)
    assert parts == ["bash", f"{dest}/bin/context-signal.sh"]
    assert os.path.exists(parts[1])
    p2 = subprocess.run(["bash", INSTALL_SH, "uninstall"], env=env,
                        capture_output=True, text=True, timeout=120)
    assert p2.returncode == 0, p2.stderr
    obj = json.load(open(settings))
    assert not (obj.get("hooks") or {}).get("UserPromptSubmit")


def test_driver_refuses_relative_home(tmp_path):
    env = dict(os.environ)
    env.update(HOME=str(tmp_path), CTXMONITOR_HOME="relative/dir",
               CTXMONITOR_SETTINGS=str(tmp_path / "s.json"))
    p = subprocess.run(["bash", INSTALL_SH, "install"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 1
    assert "absolute" in p.stderr


def test_installed_copy_is_self_contained_for_uninstall_and_status(tmp_path):
    # The installer ships install.sh into $DEST, so uninstall/status run from the
    # installed copy with no re-download. This is what makes uninstall painless.
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0, p.stderr
    assert (dest / "install.sh").exists(), "install.sh must be shipped into the install home"

    env = dict(os.environ)
    env.update(HOME=str(tmp_path / "home"), CTXMONITOR_HOME=str(dest),
               CTXMONITOR_SETTINGS=str(settings))

    st = subprocess.run(["bash", str(dest / "install.sh"), "status"], env=env,
                        capture_output=True, text=True, timeout=120)
    assert st.returncode == 0, st.stderr
    assert "hook:       registered" in st.stdout
    assert "self-test: scorer runs" in st.stdout

    un = subprocess.run(["bash", str(dest / "install.sh"), "uninstall"], env=env,
                        capture_output=True, text=True, timeout=120)
    assert un.returncode == 0, un.stderr
    assert not dest.exists(), "uninstall from the installed copy removes the install home"
    obj = json.load(open(settings))
    assert not (obj.get("hooks") or {}).get("UserPromptSubmit")


def test_reinstall_from_installed_copy_is_safe(tmp_path):
    # Running the installed copy's install verb (HERE == DEST) must not error on
    # a self-copy and must be an idempotent no-op on settings.
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0
    before = open(settings).read()
    env = dict(os.environ)
    env.update(HOME=str(tmp_path / "home"), CTXMONITOR_HOME=str(dest),
               CTXMONITOR_SETTINGS=str(settings))
    again = subprocess.run(["bash", str(dest / "install.sh"), "install"], env=env,
                          capture_output=True, text=True, timeout=120)
    assert again.returncode == 0, again.stderr
    assert open(settings).read() == before, "re-install from $DEST is a settings no-op"
    assert (dest / "bin" / "context_quality.py").exists()


def _set_installed_version(dest, v):
    (dest / "VERSION").write_text(v + "\n")


def test_version_written_on_install(tmp_path):
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0, p.stderr
    self_v = (pathlib.Path(REPO) / "VERSION").read_text().strip()
    assert (dest / "VERSION").read_text().strip() == self_v


def test_reinstall_same_version_reports_on_newest(tmp_path):
    driver(tmp_path, "install")
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0, p.stderr
    assert "on-newest" in p.stdout
    assert "statusline: on-newest" in p.stdout, "our own statusline reads on-newest, not kept-existing"


def test_check_on_fresh_reports_not_installed_and_writes_nothing(tmp_path):
    home = tmp_path / "home"
    dest = home / ".ctxmonitor"
    p, dest, settings = driver(tmp_path, "install", "--check")
    assert p.returncode == 0, p.stderr
    assert "not installed" in p.stdout
    assert not dest.exists(), "--check must install nothing"
    assert not settings.exists()


def test_check_reports_upgrade_without_installing(tmp_path):
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0
    _set_installed_version(dest, "0.1.0")            # pretend an older version is installed
    before = (dest / "bin" / "context_quality.py").read_bytes()
    p2, _, _ = driver(tmp_path, "install", "--check")
    assert p2.returncode == 0, p2.stderr
    assert "upgrade available" in p2.stdout
    assert "0.1.0" in p2.stdout
    assert (dest / "VERSION").read_text().strip() == "0.1.0", "--check must not bump VERSION"
    assert (dest / "bin" / "context_quality.py").read_bytes() == before, "--check must not touch files"


def test_upgrade_with_yes_proceeds_and_bumps_version(tmp_path):
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0
    _set_installed_version(dest, "0.1.0")
    self_v = (pathlib.Path(REPO) / "VERSION").read_text().strip()
    p2, _, _ = driver(tmp_path, "install", "--yes")
    assert p2.returncode == 0, p2.stderr
    assert "upgrade" in p2.stdout.lower()
    assert f"0.1.0 -> v{self_v}" in p2.stdout or f"0.1.0 -> {self_v}" in p2.stdout
    assert (dest / "VERSION").read_text().strip() == self_v, "upgrade bumps the installed VERSION"


def test_downgrade_detected_and_gated(tmp_path):
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 0
    _set_installed_version(dest, "9.9.9")            # installed is newer than this installer
    self_v = (pathlib.Path(REPO) / "VERSION").read_text().strip()
    p2, _, _ = driver(tmp_path, "install", "--yes")  # --yes proceeds through the downgrade
    assert p2.returncode == 0, p2.stderr
    assert "DOWNGRADE" in p2.stdout
    assert (dest / "VERSION").read_text().strip() == self_v


def test_driver_status_reports_and_self_tests(tmp_path):
    driver(tmp_path, "install")
    p, _, _ = driver(tmp_path, "status")
    assert p.returncode == 0, p.stderr
    assert "hook:       registered" in p.stdout
    assert "self-test: scorer runs" in p.stdout


def test_driver_refuses_symlinked_settings(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    victim = home / "victim.json"
    victim.write_text("{}")
    os.symlink(str(victim), str(home / ".claude" / "settings.json"))
    p, dest, settings = driver(tmp_path, "install")
    assert p.returncode == 1
    assert victim.read_text() == "{}"
