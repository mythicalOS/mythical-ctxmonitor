"""The settings transaction engine: register/deregister ctxmonitor's entries in a Claude
Code settings.json under an honest concurrency contract.

Called by install.sh; stdlib-only, like the scorer. The contract this implements is spelled
out in the package README and enforced by tests/test_installer.py:

GUARANTEED
  - A torn file is impossible: every publish is an atomic same-directory operation on an
    fsync'd temp file — rename(2) when replacing an existing file, link(2) when creating one
    (a direct O_EXCL create would be observably empty/partial while being written).
  - The transaction never writes THROUGH a symlink: every read of the target — the initial
    read and the pre-publish re-read — uses O_NOFOLLOW, and the re-read re-verifies file
    identity (st_dev + st_ino). A swap detected at any read or at the identity re-check is
    refused with no write. rename(2) replaces the final path component without following it,
    so a symlink landing AFTER the final re-read can itself be replaced — the file it pointed
    to is untouched (a safe outcome, not a refusal).
  - ENOENT is the normal first-install state: create from an empty {} base, mode 0600,
    published by exclusive link — EEXIST from a concurrent creator re-enters the read path
    (re-read, merge, retry). The ENOENT branch never renames over a file it did not see.
  - Malformed structures (non-array hooks.UserPromptSubmit, non-object groups/entries,
    non-object statusLine) fail WITHOUT writing, naming the malformed key.

BEST-EFFORT, stated plainly
  - External-edit detection is by CONTENT HASH: bytes are hashed at read; immediately before
    publish the target is re-read (no-follow, identity-checked) and compared. A completed
    external write whose result differs from what we read aborts the attempt and the
    transaction retries from the read (bounded, default 3).
  - Two things are NOT claimed: an external write sequence that ends by restoring the exact
    bytes we read (A→B→A) is undetectable by construction — and harmless to OUR merge, which
    re-applies onto content identical to what it validated; and a non-cooperating write
    landing between the final re-read and the rename is silently overwritten. That window is
    microseconds, but it is not zero, and no lock can close it because the harness takes none.
  - Serialization: read → parse → modify → json.dumps(indent=2). Formatting normalization of
    the whole file is accepted and disclosed (--dry-run prints the full unified diff before
    any write). Semantic preservation of foreign content is promised and tested;
    byte-preservation is not.
  - The timestamped backup taken before every replace is DISASTER-RECOVERY only, never a
    restore mechanism (it will not reflect post-install user edits).

POLICY
  - Idempotent: if our exact hook command is present anywhere, the hook step is a no-op.
  - statusLine is set only if ABSENT; an existing statusLine is never touched (the caller
    prints a manual wrapping recipe instead).
  - Uninstall surgically removes exactly our entries via the same parse-modify engine.
"""
import difflib
import errno
import hashlib
import json
import os
import sys
import time

RETRIES = 3
HOOK_TIMEOUT = 15  # seconds; matches the hook's own fail-safe budget


class TxnError(Exception):
    """A refusal: nothing was written. .reason is stable for tests."""

    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# --------------------------------------------------------------------------- #
# Reading (no-follow, identity-capturing)

def _read_nofollow(path):
    """Read path without following a symlink. Returns (bytes, stat) or (None, None) on
    ENOENT. A symlink raises TxnError('symlink-refused')."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None, None
    except OSError as e:
        if e.errno in (errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)):
            raise TxnError("symlink-refused", path)
        raise
    try:
        st = os.fstat(fd)
        data = b""
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            data += chunk
        return data, st
    finally:
        os.close(fd)


def _parse(data, path):
    if data is None:
        return {}
    text = data.decode("utf-8")
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
    except ValueError as e:
        raise TxnError("unparseable-json", f"{path}: {e}")
    if not isinstance(obj, dict):
        raise TxnError("malformed-shape", f"{path}: top level is not an object")
    return obj


# --------------------------------------------------------------------------- #
# The pure modification (validate + merge/remove); no I/O

def _validate_shapes(obj):
    # An explicit null is a malformed shape, not an absence: writing "around" a
    # null would silently bless a state the harness itself may choke on.
    if "hooks" in obj and not isinstance(obj["hooks"], dict):
        raise TxnError("malformed-shape", "hooks is not an object")
    hooks = obj.get("hooks") or {}
    if "UserPromptSubmit" in hooks and not isinstance(hooks["UserPromptSubmit"], list):
        raise TxnError("malformed-shape", "hooks.UserPromptSubmit is not an array")
    for g in hooks.get("UserPromptSubmit") or []:
        if not isinstance(g, dict):
            raise TxnError("malformed-shape", "hooks.UserPromptSubmit group is not an object")
        if "hooks" in g and not isinstance(g["hooks"], list):
            raise TxnError("malformed-shape", "a matcher group's hooks is not an array")
        for e in g.get("hooks") or []:
            if not isinstance(e, dict):
                raise TxnError("malformed-shape", "a hook entry is not an object")
    if "statusLine" in obj and not isinstance(obj["statusLine"], dict):
        raise TxnError("malformed-shape", "statusLine is not an object")


def _hook_present(obj, command):
    for g in (obj.get("hooks") or {}).get("UserPromptSubmit") or []:
        for e in g.get("hooks") or []:
            if e.get("type") == "command" and e.get("command") == command:
                return True
    return False


def _is_our_statusline(sl, statusline_command):
    """ONE ownership predicate, shared by install-reporting and uninstall — a
    statusLine object is ours only if it is exactly our {type:command, command}.
    A foreign object that merely reuses our command string (e.g. type != command)
    is NOT ours: install keeps it (recipe printed) and uninstall leaves it."""
    return (isinstance(sl, dict)
            and sl.get("type") == "command"
            and sl.get("command") == statusline_command)


def apply_install(obj, hook_command, statusline_command):
    """Pure merge. Returns (new_obj, report) — report says what happened per entry."""
    _validate_shapes(obj)
    out = json.loads(json.dumps(obj))  # deep copy through JSON, our serialization domain
    report = {}
    if _hook_present(out, hook_command):
        report["hook"] = "already-present"
    else:
        group = {"hooks": [{"type": "command", "command": hook_command,
                            "timeout": HOOK_TIMEOUT}]}
        out.setdefault("hooks", {}).setdefault("UserPromptSubmit", []).append(group)
        report["hook"] = "added"
    if statusline_command is None:
        report["statusline"] = "skipped"
    elif "statusLine" in out:
        # Distinguish OUR own current statusLine (a re-run/upgrade — nothing to do,
        # "on-newest") from a genuinely foreign one (which we leave untouched and
        # tell the user how to wrap).
        cur = out.get("statusLine")
        if _is_our_statusline(cur, statusline_command):
            report["statusline"] = "already-ours"
        else:
            report["statusline"] = "kept-existing"
    else:
        out["statusLine"] = {"type": "command", "command": statusline_command}
        report["statusline"] = "added"
    return out, report


def apply_uninstall(obj, hook_command, statusline_command):
    """Pure removal of exactly our entries."""
    _validate_shapes(obj)
    out = json.loads(json.dumps(obj))
    report = {"hook": "absent", "statusline": "absent"}
    hooks = out.get("hooks")
    ups = (hooks or {}).get("UserPromptSubmit")
    if isinstance(ups, list):
        emptied_by_us = []
        for g in ups:
            entries = g.get("hooks")
            if isinstance(entries, list):
                kept = [e for e in entries
                        if not (e.get("type") == "command" and e.get("command") == hook_command)]
                if len(kept) != len(entries):
                    report["hook"] = "removed"
                    # Drop the group ONLY if OUR removal emptied it and nothing
                    # foreign remains in it. A group that was already empty — or
                    # carries foreign keys — is foreign content and survives.
                    if not kept and not any(k for k in g if k != "hooks"):
                        emptied_by_us.append(id(g))
                g["hooks"] = kept
        ups = [g for g in ups if id(g) not in emptied_by_us]
        if ups:
            hooks["UserPromptSubmit"] = ups
        else:
            hooks.pop("UserPromptSubmit", None)
        if not hooks:
            out.pop("hooks", None)
    if _is_our_statusline(out.get("statusLine"), statusline_command):
        out.pop("statusLine")
        report["statusline"] = "removed"
    return out, report


def _serialize(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Publishing (atomic; exclusive on the ENOENT branch)

def _write_all(fd, payload):
    # write(2) may return short — a short write published as-is would be a torn
    # file with a successful exit status, so loop to completion or fail loudly.
    view = memoryview(payload)
    off = 0
    while off < len(view):
        n = os.write(fd, view[off:])
        if n <= 0:
            raise OSError(f"short write ({off}/{len(view)} bytes)")
        off += n


def _fsync_write_temp(dirname, payload, mode):
    """Write payload to a same-dir temp with EXACT mode (fchmod, umask-independent),
    fsync'd. The temp is unlinked on any failure — never leaked."""
    tmp = os.path.join(dirname, f".ctxmonitor-txn.{os.getpid()}.{time.monotonic_ns()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(fd, mode)  # exact; the open() mode was filtered through umask
        _write_all(fd, payload)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.close(fd)
    return tmp


def _publish_create(path, payload):
    """ENOENT branch: link-based exclusive publish. Returns False if a concurrent creator
    won (EEXIST) — the caller re-enters the read path."""
    d = os.path.dirname(path) or "."
    tmp = _fsync_write_temp(d, payload, 0o600)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        os.unlink(tmp)


def _publish_replace(path, payload, expected_hash, expected_ident):
    """Replace branch: no-follow re-read, identity + content-hash checks, THEN the
    disaster-recovery backup, then rename(2). Ordering matters: every refusal path
    (symlink, identity, hash) must leave the tree untouched — no backup sidecar, no
    temp. The mode comes from the RE-READ's stat, so a chmod that landed since the
    initial read is honored, not reverted. Returns (ok, backup_path)."""
    data, st = _read_nofollow(path)
    if data is None:
        raise TxnError("target-vanished", path)
    if (st.st_dev, st.st_ino) != expected_ident:
        raise TxnError("identity-changed", path)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        return False, None  # external edit completed since our read: retry from the top
    backup = _backup(path, data)
    d = os.path.dirname(path) or "."
    tmp = _fsync_write_temp(d, payload, stat_mode(st))
    try:
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True, backup


def stat_mode(st):
    return (st.st_mode & 0o7777) or 0o600


def _backup(path, data):
    if data is None:
        return None
    b = f"{path}.ctxmonitor-backup.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    n, cand = 0, b
    while os.path.exists(cand):
        n += 1
        cand = f"{b}.{n}"
    fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, data)  # a torn backup reported as taken would be worse than none
    finally:
        os.close(fd)
    return cand


# --------------------------------------------------------------------------- #
# The transaction driver

def run(path, action, hook_command, statusline_command, dry_run=False,
        _between_read_and_publish=None):
    """One transaction. Returns a dict report. Raises TxnError on refusal.

    _between_read_and_publish is a TEST SEAM: called after the read+merge and before the
    publish attempt, so the hermetic suite can interleave concurrent writers/swaps at the
    exact race point. Production callers never pass it.
    """
    assert action in ("install", "uninstall")
    apply_fn = apply_install if action == "install" else apply_uninstall
    last_reason = None
    for _attempt in range(RETRIES):
        data, st = _read_nofollow(path)
        obj = _parse(data, path)
        new_obj, report = apply_fn(obj, hook_command, statusline_command)
        payload = _serialize(new_obj)
        changed = payload != (data if data is not None else b"")
        if not changed and data is not None:
            report.update(path=path, wrote=False, noop=True)
            return report
        if dry_run:
            old_text = (data or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
            new_text = payload.decode("utf-8").splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(old_text, new_text,
                                                fromfile=path, tofile=path + " (proposed)"))
            report.update(path=path, wrote=False, dry_run=True, diff=diff)
            return report
        if _between_read_and_publish is not None:
            _between_read_and_publish()
        if data is None:
            if _publish_create(path, payload):
                report.update(path=path, wrote=True, created=True, backup=None)
                return report
            last_reason = "concurrent-creator"
            continue  # EEXIST: someone created it mid-transaction — re-read and merge
        ok, backup = _publish_replace(path, payload,
                                      hashlib.sha256(data).hexdigest(),
                                      (st.st_dev, st.st_ino))
        if ok:
            report.update(path=path, wrote=True, created=False, backup=backup)
            return report
        last_reason = "external-edit"
    raise TxnError("retries-exhausted", last_reason or "unknown")


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="settings_txn")
    ap.add_argument("action", choices=["install", "uninstall"])
    ap.add_argument("--settings", required=True)
    ap.add_argument("--hook-command", required=True)
    ap.add_argument("--statusline-command", default=None,
                    help="omit to skip statusLine handling entirely")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        report = run(a.settings, a.action, a.hook_command, a.statusline_command,
                     dry_run=a.dry_run)
    except TxnError as e:
        print(json.dumps({"error": e.reason, "detail": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
