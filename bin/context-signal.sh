#!/usr/bin/env bash
# context-signal.sh — the floor's UserPromptSubmit context-quality signal (CtxMonitor).
#
# WHAT: on every prompt, score THIS session's transcript with the first-party scorer
# bin/context_quality.py (occupancy + degradation + cache efficiency), write a
# per-session cache the statusline reads, and inject an in-context warning to the
# agent when quality is fraying. Replaces the old reader of the Token-Optimizer
# quality cache, which is never recomputed under the permission floor (that
# project's scorer is excluded for license/trust) and so was stale/inert — the
# whole reason the floor quality signal did not work.
#
# PRODUCES: $HOME/.ctxmonitor/cache/ctxmonitor-<sid>.json  (full score JSON;
#   both the floor and global statuslines render "CtxMonitor:X(NN)" from it).
#   CTX_MONITOR_DIR (absolute) overrides that directory -- the container lane sets
#   it per session so the daemon can read the cache back; see the write block below.
#   AND, beside it, TWO on-demand sidecars derived from the SAME parse:
#     ctxcompose-<sid>.json -- the BOUNDED byte-composition (compose_cache_payload);
#     ctxcache-<sid>.json   -- the BOUNDED prompt-cache report (cache_report_payload).
#   Deliberately SIBLING files rather than sections of the score cache: the score cache
#   is on the daemon's 2-second sessions poll and must not grow; these two are served on
#   demand only, and a malformed one can void neither the grade nor the other.
#
# CONTRACT: Claude Code UserPromptSubmit hook. Reads the hook JSON on stdin
# ({session_id, transcript_path, ...}); emits hookSpecificOutput.additionalContext
# (a one-line note) ONLY when degraded, and ALWAYS exits 0 — a non-zero/blocking
# UserPromptSubmit hook would BLOCK the prompt. FAIL-SAFE: any error => silent exit 0.
#
# OWNED command form (emitter/validator/provenance): `bash <abs>/context-signal.sh`.

set -uo pipefail
INPUT="$(cat)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"   # mythical/bin — sibling of context_quality.py

SIGNAL_PY=$(cat <<'PY'
import glob
import json
import os
import re
import sys


def done():
    # Fail-safe SILENT: emit nothing, never block the prompt.
    sys.exit(0)


try:
    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
except Exception:
    done()
if not isinstance(data, dict):
    done()

sid = data.get("session_id") or ""
safe_sid = re.sub(r"[^a-zA-Z0-9_-]", "", sid) if isinstance(sid, str) else ""
if not safe_sid:
    done()

here = os.environ.get("CQ_DIR", "") or ""
home = os.environ.get("HOME", "") or ""

# Resolve this session's transcript: prefer the payload's transcript_path, else
# glob by session id across the project stores (filename is "<session-id>.jsonl").
tpath = data.get("transcript_path") or ""
if not (isinstance(tpath, str) and tpath and os.path.isfile(tpath)):
    hits = glob.glob(os.path.join(home, ".claude", "projects", "*", safe_sid + ".jsonl"))
    tpath = hits[0] if hits else ""
if not tpath or not os.path.isfile(tpath):
    done()

# Latency guard: bound per-prompt parse cost. An oversized transcript is scored
# from its TAIL (read_jsonl_tail) instead of skipped -- skipping (the old behavior)
# silenced the warning exactly when a long session needed it most: huge transcript
# == long session == the sessions most likely to be running out of room. Occupancy,
# the only signal that gates the nudge, needs just the final assistant turn, so the
# tail is sufficient for the gate; degradation/cache become recent-window
# approximations and the cached JSON says so (partial_tail) rather than passing
# them off as whole-session truth.
try:
    oversize = os.path.getsize(tpath) > 25 * 1024 * 1024
except OSError:
    done()

# First-party scorer is this hook's sibling (bin/context_quality.py).
sys.path.insert(0, here)
try:
    import context_quality as cq
    if oversize:
        recs, tail_partial = cq.read_jsonl_tail(tpath)
    else:
        recs, tail_partial = cq.read_jsonl(tpath), False
    res = cq.score_session(recs)
    # The helper's OWN flag, not the size threshold: tail growth that ends
    # up parsing the whole file is honestly NOT partial.
    if tail_partial:
        res["partial_tail"] = True
except Exception:
    done()

# Cache the full result for the statuslines to render (atomic write; never blocks).
#
# WHERE. $HOME/.ctxmonitor/cache by default -- the harness-neutral, tool-owned host/CLI
# location the statuslines read. CTX_MONITOR_DIR overrides it, and exists for exactly one caller: the
# container daemon, which spawns each session with the var pointing at that session's
# own drop dir under an image-provisioned base. A session's HOME in that topology is
# 0700 and owned by the session uid, so a cache written there is one the daemon cannot
# traverse to -- the score would be computed every turn and read by nobody. The override
# is the whole mechanism by which an in-container session gets a grade at all.
#
# It must be ABSOLUTE. A relative value would resolve against the session's cwd, which is
# its own writable project tree -- i.e. a path the agent can move, delete or fill, and one
# nothing else knows how to find. Anything else falls back to HOME rather than guessing.
#
# MODE, on the override branch only: dir 0710 and file 0640, set explicitly rather than
# left to whatever umask the session inherited (the daemon's own umask is 002, which would
# make both group-WRITABLE). The daemon reads through the group; the session owns; nothing
# else has any bit. The HOME branch keeps its historical umask-derived modes untouched --
# there is exactly one reader there and it is the same user.
cm_dir = None
use_override = False
try:
    cm_override = (os.environ.get("CTX_MONITOR_DIR", "") or "").strip()
    use_override = bool(cm_override) and os.path.isabs(cm_override)
    cm_dir = cm_override if use_override else os.path.join(home, ".ctxmonitor", "cache")
    if use_override:
        os.makedirs(cm_dir, mode=0o710, exist_ok=True)
    else:
        os.makedirs(cm_dir, exist_ok=True)
except Exception:
    cm_dir = None


def drop(name, payload):
    """Atomically write ONE drop file into cm_dir. Never raises, never blocks.

    Extracted (WS-10) so the score cache and the composition sidecar cannot drift on
    the properties that matter -- same directory, same 0640-before-rename ordering on
    the override branch, same atomic os.replace. Each call is independently fenced:
    a failure to write one file must not cost the other, and neither may reach the
    caller (this is a UserPromptSubmit hook; a raised exception would still exit 0 via
    the `2>/dev/null` invocation, but the nudge below would be skipped).
    """
    if cm_dir is None:
        return
    try:
        target = os.path.join(cm_dir, name)
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        if use_override:
            # Before the rename, so the file is never briefly readable at a looser mode
            # under its final name. os.replace preserves the mode of the SOURCE inode.
            os.chmod(tmp, 0o640)
        os.replace(tmp, target)
    except Exception:
        pass


def undrop(name):
    """Remove a drop file this run can no longer stand behind. Never raises.

    Used for the composition ONLY, and only when the answer is persistently
    unavailable rather than transiently un-writable. The distinction is the
    point: a failed WRITE leaves the previous file, because a stale composition is
    still a true past observation carrying its own age; but a transcript that has
    grown past the composition's input bound will never compose again, and leaving a
    frozen `partial_tail: false` snapshot there would claim to be the whole picture of
    a window that no longer exists. Absence is a first-class answer on that route.
    """
    if cm_dir is None:
        return
    try:
        os.unlink(os.path.join(cm_dir, name))
    except Exception:
        pass


drop("ctxmonitor-" + safe_sid + ".json", res)

# Inject a warning ONLY when the gate fires. The gate itself is NOT defined here:
# context_quality.band_of is the single source (grade-first: fires iff
# score < 80 AND occupancy >= 65%; >= 75% = strong). The Stop-hook telemetry calls
# the same function, so the nudge and the bus band-change can never disagree --
# this hook only RENDERS the band. Rationale and thresholds live on band_of's
# comment block.
try:
    score = float(res.get("score") or 0)
except (TypeError, ValueError):
    score = 0.0
sig = res.get("signals") or {}


def sub(name):
    s = sig.get(name) if isinstance(sig.get(name), dict) else {}
    return s.get("score")


occ_raw = (sig.get("occupancy") or {}).get("raw") if isinstance(sig.get("occupancy"), dict) else {}
occ_frac = (occ_raw or {}).get("occupancy_fraction") or 0.0
pct = int(round(occ_frac * 100))
band = cq.band_of(score, occ_frac)
lead = None
if band == "strong":
    lead = "context ~{}% full -- FINISH or hand off NOW; little room left".format(pct)
elif band == "advisory":
    lead = "context ~{}% full -- wrap up the current unit while you still have room to finish cleanly".format(pct)
if lead is not None:
    note = ("CTX: {lead}. Grade {g}({s}) [occupancy {o} / degradation {d} / "
            "cache {c}]. Commit in-progress work or hand off before quality drops further. "
            "[ctxmonitor]").format(
        lead=lead, g=res.get("grade") or "?", s=int(round(score)),
        o=sub("occupancy"), d=sub("degradation"), c=sub("cache_efficiency"))
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": note}}))
# FLUSHED HERE, not left to interpreter exit. stdout is a PIPE, so python
# block-buffers it: anything printed above sits in the buffer until exit. The hook is
# armed with a 15s timeout (spawn/ctx-signal-hook.ts), and a killed process's buffer is
# lost -- so a nudge that is merely "printed" is not yet a nudge that was DELIVERED.
# Everything below this line is display-only work; nothing it can do may cost the
# session the one output that changes its behaviour.
sys.stdout.flush()

# COMPOSITION SIDECAR (WS-10) -- "what is IN the window", beside "how full it is".
#
# LAST, DELIBERATELY. It runs after the score cache is written AND after the
# nudge is flushed, because it is the only piece here nothing depends on: a session that
# loses its composition loses a panel, while a session that loses its nudge keeps working
# past the point it should have handed off.
#
# SECOND FOLD, SAME PARSE. compose_session walks the records ALREADY in memory -- the
# transcript is still read and parsed exactly once per prompt. The added work is a
# bounded MULTIPLE of work the hook already does, never a new unbounded term. Measured
# on this machine:
#   * 60 sampled real transcripts: p50 +0.34 ms / p90 +0.60 ms, on a p50 1.06 ms
#     parse+score -- ~0.3x.
#   * largest whole-file parse (25 MB, 25k records): +47 ms on 243 ms -- ~0.2x.
#   * the pathological shape a gate round raised -- an OVERSIZE transcript holding ONE
#     30 MB record, where read_jsonl_tail must grow past its budget to land a record:
#     +133 ms on a 62 ms parse -- 2.14x, and 195 ms total against a 15s timeout. The
#     growth and the parse are pre-existing and are paid by the SCORE regardless; what
#     this adds is proportional to the same bytes, so it cannot make the hook unbounded
#     in a way it was not already.
#
# A SEPARATE FILE, NOT A SECTION OF THE SCORE CACHE, and that is the load-bearing
# choice: ctxmonitor-<sid>.json is read by the daemon ONCE PER SESSION PER 2-SECOND
# POLL for the grade. Folding ~2 KB of composition into it would make every poll read
# and parse a payload nothing on that path consumes. The sidecar is read only by the
# on-demand composition route, so the poll path stays byte-identical -- and a
# malformed composition cannot void the grade, because they are not the same bytes.
#
# HARD INPUT BOUND. The 24 KiB cap bounds the OUTPUT; this bounds the
# PRODUCTION COST, which the output cap says nothing about -- compose_session
# serializes every record and every block, and retains one map entry per distinct
# tool id, so its cost tracks the bytes it is handed. The transcript is written into
# the session's own tree, so those bytes are influenceable; the blast radius is the
# session's own prompt latency, but "bounded by the attacker's patience" is not a
# bound.
#
# The ceiling is the transcript-size threshold the hooks already trust, and the branch
# is chosen from a size read AFTER the score's parse, not the one taken before it:
#   * <= 25 MiB BOTH before and after the parse: a transcript is append-only, so a
#     size that is under the threshold after the read proves it was never over it
#     during the read -- i.e. the score's parse really did consume at most 25 MiB, and
#     the composition can reuse those records with no read of its own. Measured at the
#     ceiling (25 MB, 25k records): +47 ms.
#   * anything else -- oversize at the start, OR grown past the threshold while the
#     score was reading (the `getsize`-then-read gap is a real race, and the
#     re-check above is what stops the composition inheriting it) -- takes a
#     NON-GROWING tail instead. read_jsonl_tail's x4 GROWTH LOOP is the unbounded
#     term: one record larger than the budget drags it up until a record lands, and on
#     a pathological transcript that is the whole file (measured: a 30 MB single
#     record => 62 ms parse, +133 ms compose; with the bound, 0.5 ms and no
#     composition). The SCORE has to accept that loop -- an empty parse would score as
#     a healthy empty session, a lie in the dangerous direction -- and a composition
#     does not. One bounded 2 MiB read on a rare path; if no complete record fits,
#     there is honestly no composition.
#
# RESIDUAL, stated rather than implied: the re-check rests on transcripts being
# APPEND-ONLY. A session that TRUNCATED and rewrote its own transcript mid-read could
# still show a small size afterwards, and the composition would then fold whatever the
# score's own (pre-existing, unbounded-by-the-same-race) read produced. That costs the
# attacking session its own prompt latency, is capped by the harness's 15s hook
# timeout, and is dominated by the score's read, which pays it first and cannot opt
# out. Closing it fully means the score reading through a single size-bounded fd --
# a change to the scoring path, not to this feature.
#
# TWO SIDECARS, ONE READ (WS-11). The cache report is a second FOLD over the very same
# records, not a second parse: the bounded read below happens once and both derivations
# consume its output. Their WRITES are fenced independently, though -- a failure to
# produce one must not cost the other, and neither may reach the caller.
_compose_name = "ctxcompose-" + safe_sid + ".json"
_cache_name = "ctxcache-" + safe_sid + ".json"
c_recs, c_partial = None, False
try:
    if oversize or os.path.getsize(tpath) > 25 * 1024 * 1024:
        c_recs, c_partial = cq.read_jsonl_tail(
            tpath, max_bytes=cq.TAIL_DEFAULT_BYTES, grow=False)
    else:
        c_recs, c_partial = recs, tail_partial
except Exception:
    # None (not []) so the two writers below can tell "read failed" from "read the whole
    # file and it was empty" -- the second is a true zero, the first is not an answer.
    c_recs = None

# A TRUE ZERO IS NOT AN ABSENCE. `not c_partial` means the WHOLE file was
# read, so zero records is a real observation about a real (empty) transcript --
# exactly the empty composition the reader validates and admits. Only a BOUNDED
# read that came up empty is an absence, because there the emptiness is a property
# of the budget rather than of the session.
try:
    if c_recs is not None and (c_recs or not c_partial):
        drop(_compose_name,
             cq.compose_cache_payload(cq.compose_session(c_recs, partial=c_partial)))
    else:
        # Nothing composable within the bound -- and that is now the STANDING answer
        # for this transcript, not a blip. Clear any earlier sidecar rather than let
        # the route keep serving a frozen snapshot of a window that is gone.
        undrop(_compose_name)
except Exception:
    undrop(_compose_name)

# The CACHE report, under the identical rule: same records, same true-zero test, same
# clear-rather-than-freeze policy. Its own try/except so a failure in either fold leaves
# the other file exactly as its own run left it.
try:
    if c_recs is not None and (c_recs or not c_partial):
        drop(_cache_name,
             cq.cache_report_payload(cq.cache_session(c_recs, partial=c_partial)))
    else:
        undrop(_cache_name)
except Exception:
    undrop(_cache_name)
sys.exit(0)
PY
)
printf '%s' "$INPUT" | CQ_DIR="$HERE" python3 -c "$SIGNAL_PY" 2>/dev/null
exit 0
