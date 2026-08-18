#!/usr/bin/env bash
# mythical-ctxmonitor installer — install / uninstall / status / --dry-run.
#
# Program files land in ~/.ctxmonitor/ (harness-neutral, tool-owned; the score cache lives
# in ~/.ctxmonitor/cache/). Claude Code wiring — the UserPromptSubmit hook (default-on) and
# the statusline (only if you have none) — is registered in ~/.claude/settings.json through
# lib/settings_txn.py, which owns the transaction contract (atomic publish, no-follow reads,
# content-hash retry; see its header). --dry-run prints the full settings diff and writes
# NOTHING (no files copied either).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
DEST="${CTXMONITOR_HOME:-$HOME/.ctxmonitor}"
SETTINGS="${CTXMONITOR_SETTINGS:-$HOME/.claude/settings.json}"

# Both paths must be absolute: the registered command strings and the settings
# target may not depend on whatever cwd a later shell happens to have.
case "$DEST" in /*) ;; *) echo "FATAL: CTXMONITOR_HOME must be absolute (got: $DEST)" >&2; exit 1 ;; esac
case "$SETTINGS" in /*) ;; *) echo "FATAL: CTXMONITOR_SETTINGS must be absolute (got: $SETTINGS)" >&2; exit 1 ;; esac

# POSIX single-quote escaping: the registered strings are executed by a shell
# later, so a path with spaces or metacharacters must arrive as ONE argument —
# raw interpolation would split or, worse, execute it.
shq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# The exact registered command strings — absolute, shell-quoted, and the
# idempotency/uninstall keys (byte-exact on both sides by construction).
HOOK_CMD="bash $(shq "$DEST/bin/context-signal.sh")"
SL_CMD="bash $(shq "$DEST/bin/statusline-command.sh")"

PAYLOAD=(bin/context_quality.py bin/statusline-command.sh bin/context-signal.sh
         bin/testdata/golden-session.jsonl bin/testdata/golden-session.expected.json)

usage() {
  cat <<EOF
usage: install.sh [install|uninstall|status] [--dry-run]

install    copy program files to $DEST, register the Claude Code hook
           (default-on) and the statusline (only if none is set)
uninstall  remove exactly our settings entries, then remove $DEST
status     report installed files, registration state, and a scorer self-test
--dry-run  (install/uninstall) print the full settings diff; write nothing
EOF
}

MODE="install"; DRY=""
for arg in "$@"; do
  case "$arg" in
    install|uninstall|status) MODE="$arg" ;;
    --dry-run) DRY="--dry-run" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

# --- prerequisites: hard-fail with remediation, never a half-install.
need() {
  command -v "$1" >/dev/null 2>&1 && return 0
  echo "FATAL: '$1' is required and not on PATH." >&2
  echo "  remediation: $2" >&2
  exit 1
}
need python3 "macOS: xcode-select --install · Debian/Ubuntu: apt install python3"
[ "$MODE" = "status" ] || need jq "macOS: brew install jq · Debian/Ubuntu: apt install jq"

txn() {  # txn <action> [--dry-run] — drives the settings transaction, prints its report
  python3 "$1" "$2" --settings "$SETTINGS" \
    --hook-command "$HOOK_CMD" --statusline-command "$SL_CMD" ${3:+"$3"}
}

case "$MODE" in
  install)
    for f in "${PAYLOAD[@]}" lib/settings_txn.py; do
      [ -f "$HERE/$f" ] || { echo "FATAL: payload file missing: $f (broken unpack?)" >&2; exit 1; }
    done
    if [ -n "$DRY" ]; then
      echo "-- dry run: no files copied, no settings written --"
      REPORT="$(txn "$HERE/lib/settings_txn.py" install --dry-run)" || exit 1
      printf '%s\n' "$REPORT" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("diff") or "(settings already exactly as installed — no change)")'
      exit 0
    fi
    mkdir -p "$DEST/bin/testdata" "$DEST/lib" "$DEST/cache"
    # Skip all copying when already running from $DEST (re-running the installed
    # copy) — the files are in place, and cp of a file onto itself errors. Ship
    # install.sh into $DEST too, so `uninstall`/`status` are self-contained.
    if [ "$HERE" != "$DEST" ]; then
      for f in "${PAYLOAD[@]}" lib/settings_txn.py install.sh; do
        cp -p "$HERE/$f" "$DEST/$f"
      done
    fi
    mkdir -p "$(dirname "$SETTINGS")"
    REPORT="$(txn "$HERE/lib/settings_txn.py" install)" || {
      echo "FATAL: settings registration refused — program files are in place, settings untouched." >&2
      exit 1
    }
    echo "installed: $DEST"
    # The report printer receives SL_CMD via the environment — command strings are
    # never spliced into interpreted source.
    printf '%s\n' "$REPORT" | CTXMON_SL_CMD="$SL_CMD" python3 -c '
import json, os, sys
r = json.load(sys.stdin)
print("hook:       " + r["hook"] + "  (" + ("registered" if r["hook"] != "already-present" else "was already registered") + ")")
print("statusline: " + r["statusline"])
if r["statusline"] == "kept-existing":
    print("")
    print("Your existing statusLine was left untouched. To render CtxMonitor through it,")
    print("have your statusline script delegate to (or append the output of):")
    print("    " + os.environ["CTXMON_SL_CMD"])
if r.get("backup"):
    print("settings backup (disaster recovery only, not a restore mechanism): " + r["backup"])
'
    ;;
  uninstall)
    if [ -n "$DRY" ]; then
      REPORT="$(txn "$HERE/lib/settings_txn.py" uninstall --dry-run)" || exit 1
      printf '%s\n' "$REPORT" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("diff") or "(nothing of ours is registered — no change)")'
      echo "-- dry run: $DEST would be removed --"
      exit 0
    fi
    # Same engine for the real run and --dry-run: the repo copy when present
    # (the tarball always ships it), the installed copy as the fallback — the
    # two runs must never diverge on engine version.
    ENGINE="$HERE/lib/settings_txn.py"; [ -f "$ENGINE" ] || ENGINE="$DEST/lib/settings_txn.py"
    if [ -f "$SETTINGS" ] || [ -L "$SETTINGS" ]; then
      txn "$ENGINE" uninstall >/dev/null || {
        echo "FATAL: settings deregistration refused — nothing removed." >&2
        exit 1
      }
    fi
    rm -rf "$DEST"
    echo "uninstalled: settings entries removed, $DEST removed"
    ;;
  status)
    ok=1
    for f in "${PAYLOAD[@]}"; do
      if [ -f "$DEST/$f" ]; then echo "present  $DEST/$f"; else echo "MISSING  $DEST/$f"; ok=0; fi
    done
    if [ -L "$SETTINGS" ]; then
      echo "settings: WARNING — $SETTINGS is a symlink; install/uninstall refuse to operate on it"
    fi
    if [ -f "$SETTINGS" ]; then
      python3 - "$SETTINGS" "$HOOK_CMD" "$SL_CMD" <<'EOF'
import json, sys
path, hook_cmd, sl_cmd = sys.argv[1:4]
try:
    obj = json.load(open(path))
except Exception as e:
    print(f"settings: UNREADABLE ({e})"); sys.exit(0)
hook = any(e.get("type") == "command" and e.get("command") == hook_cmd
           for g in (obj.get("hooks") or {}).get("UserPromptSubmit") or []
           for e in g.get("hooks") or [])
sl = obj.get("statusLine") or {}
print("hook:       " + ("registered" if hook else "NOT registered"))
print("statusline: " + ("ours" if sl.get("command") == sl_cmd
                        else ("other (untouched by us)" if sl else "none")))
EOF
    else
      echo "settings: $SETTINGS does not exist"
    fi
    if [ "$ok" = 1 ] && [ -f "$DEST/bin/context_quality.py" ]; then
      if python3 "$DEST/bin/context_quality.py" score \
           "$DEST/bin/testdata/golden-session.jsonl" --json >/dev/null 2>&1; then
        echo "self-test: scorer runs on the shipped fixture"
      else
        echo "self-test: FAILED (scorer errored on the shipped fixture)"; exit 1
      fi
    fi
    ;;
esac
