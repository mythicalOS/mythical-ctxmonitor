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

PAYLOAD=(VERSION bin/context_quality.py bin/statusline-command.sh bin/context-signal.sh
         bin/testdata/golden-session.jsonl bin/testdata/golden-session.expected.json)

usage() {
  cat <<EOF
usage: install.sh [install|uninstall|status] [--dry-run] [--check] [--yes]

install    copy program files to $DEST, register the Claude Code hook
           (default-on) and the statusline (only if none is set). Version-aware:
           on a re-run it reports "on-newest"; on a newer installer it shows the
           <from> -> <to> upgrade and asks to confirm (unless --yes).
uninstall  remove exactly our settings entries, then remove $DEST
status     report versions, installed files, registration state, self-test
--dry-run  (install/uninstall) print the full settings diff; write nothing
--check    (install) report the version delta / upgrade availability, then exit
           WITHOUT installing anything
--yes,-y   (install) proceed through an upgrade/downgrade without prompting
EOF
}

MODE="install"; DRY=""; CHECK=""; ASSUME_YES=""
for arg in "$@"; do
  case "$arg" in
    install|uninstall|status) MODE="$arg" ;;
    --dry-run) DRY="--dry-run" ;;
    --check) CHECK=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
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

# --- versioning ------------------------------------------------------------
SELF_VERSION="$( [ -f "$HERE/VERSION" ] && tr -d '[:space:]' < "$HERE/VERSION" || echo 'unknown' )"
installed_version() {  # echoes the installed version, or empty if not installed
  [ -f "$DEST/VERSION" ] && tr -d '[:space:]' < "$DEST/VERSION" || true
}

# ver_cmp A B -> "eq" | "lt" (A older than B) | "gt" (A newer than B).
# Uses version sort; falls back to string compare if sort -V is unavailable.
ver_cmp() {
  [ "$1" = "$2" ] && { echo eq; return; }
  local older
  older="$(printf '%s\n%s\n' "$1" "$2" | sort -V 2>/dev/null | head -1)" || older="$1"
  [ "$older" = "$1" ] && echo lt || echo gt
}

# confirm "<question>" -> 0 if the user says yes. --yes forces yes; otherwise it
# reads from the CONTROLLING TERMINAL (/dev/tty), so it still prompts under
# `curl … | bash` (where stdin is the script, not the keyboard). With no
# terminal available (a truly non-interactive/automated run) it proceeds — the
# caller pinned this version deliberately — and says so.
confirm() {
  [ -n "$ASSUME_YES" ] && return 0
  # Probe the controlling terminal by actually opening it — `[ -r /dev/tty ]` is
  # true even when there is NO controlling terminal (the device node exists), and
  # the later open would then fail. This distinguishes "a human can answer" from
  # "truly non-interactive".
  if { true > /dev/tty; } 2> /dev/null; then
    local ans=""
    printf '%s [y/N] ' "$1" > /dev/tty
    read -r ans < /dev/tty || ans=""
    case "$ans" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
  fi
  echo "  (non-interactive: no terminal to confirm on — proceeding; pass --yes to silence this)" >&2
  return 0
}

case "$MODE" in
  install)
    for f in "${PAYLOAD[@]}" lib/settings_txn.py; do
      [ -f "$HERE/$f" ] || { echo "FATAL: payload file missing: $f (broken unpack?)" >&2; exit 1; }
    done

    # --- version check: report the delta, gate an upgrade/downgrade on confirmation.
    INSTALLED_VERSION="$(installed_version)"
    if [ -z "$INSTALLED_VERSION" ]; then
      REL="fresh"
      echo "ctxmonitor: fresh install of v${SELF_VERSION}."
    else
      REL="$(ver_cmp "$INSTALLED_VERSION" "$SELF_VERSION")"
      case "$REL" in
        eq) echo "ctxmonitor: on-newest — v${INSTALLED_VERSION} is already installed (this installer is v${SELF_VERSION})." ;;
        lt) echo "ctxmonitor: upgrade available — v${INSTALLED_VERSION} -> v${SELF_VERSION}." ;;
        gt) echo "ctxmonitor: DOWNGRADE — installed v${INSTALLED_VERSION} is NEWER than this installer v${SELF_VERSION}." ;;
      esac
    fi
    # --check: report only, install nothing.
    if [ -n "$CHECK" ]; then
      case "$REL" in
        fresh) echo "  not installed; run without --check to install v${SELF_VERSION}." ;;
        eq)    echo "  nothing to do." ;;
        lt)    echo "  run without --check (or with --yes) to upgrade." ;;
        gt)    echo "  run without --check to reinstall v${SELF_VERSION} over it." ;;
      esac
      exit 0
    fi
    # Gate a version change on confirmation (fresh install + same-version reconverge
    # proceed silently). Never prompt under --dry-run — it writes nothing anyway.
    if [ -z "$DRY" ]; then
      case "$REL" in
        lt) confirm "Upgrade ctxmonitor v${INSTALLED_VERSION} -> v${SELF_VERSION}?" || {
              echo "ctxmonitor: upgrade declined — nothing changed."; exit 0; } ;;
        gt) confirm "Reinstall the OLDER v${SELF_VERSION} over installed v${INSTALLED_VERSION}?" || {
              echo "ctxmonitor: left the newer version in place — nothing changed."; exit 0; } ;;
      esac
    fi

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
    case "$REL" in
      lt) echo "upgraded: $DEST (v${INSTALLED_VERSION} -> v${SELF_VERSION})" ;;
      gt) echo "reinstalled v${SELF_VERSION}: $DEST" ;;
      eq) echo "reconverged v${SELF_VERSION}: $DEST" ;;
      *)  echo "installed: $DEST (v${SELF_VERSION})" ;;
    esac
    # The report printer receives SL_CMD via the environment — command strings are
    # never spliced into interpreted source.
    printf '%s\n' "$REPORT" | CTXMON_SL_CMD="$SL_CMD" python3 -c '
import json, os, sys
r = json.load(sys.stdin)
print("hook:       " + r["hook"] + "  (" + ("registered" if r["hook"] != "already-present" else "was already registered") + ")")
sl = r["statusline"]
if sl == "already-ours":
    print("statusline: on-newest  (ours, already current)")
else:
    print("statusline: " + sl)
if sl == "kept-existing":
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
    INSTALLED_VERSION="$(installed_version)"
    if [ -z "$INSTALLED_VERSION" ]; then
      echo "version:    not installed  (this installer is v${SELF_VERSION})"
    else
      case "$(ver_cmp "$INSTALLED_VERSION" "$SELF_VERSION")" in
        eq) echo "version:    v${INSTALLED_VERSION}  (on-newest)" ;;
        lt) echo "version:    v${INSTALLED_VERSION}  (upgrade available -> v${SELF_VERSION})" ;;
        gt) echo "version:    v${INSTALLED_VERSION}  (newer than this installer v${SELF_VERSION})" ;;
      esac
    fi
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
