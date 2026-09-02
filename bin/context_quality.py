#!/usr/bin/env python3
"""context_quality.py — context-health / quality score for a Claude Code session.

Parses a session transcript JSONL and emits a 0-100 score from THREE orthogonal
signals. First-party, stdlib-only, no network, no LLM calls. Replaces a third-party
scorer that had a denominator bug (it divided cumulative input by the window, which
measures session LENGTH, not occupancy) and redundant signals.

Replacing the prior tool — quick reference
-----------------------------------------------
Signals (weights):
  1. occupancy (0.40)  -- the LAST assistant turn's effective input
                          (input + cache_read + cache_creation) / context window.
                          This is fill, NOT cumulative tokens across the session.
  2. degradation (0.35) -- cheap behavioral proxies for quality decay: stale-read
                          churn, duplicate large tool outputs, corrective human
                          turns, and auto-compaction events.
  3. cache efficiency (0.25) -- cache_read / (input + cache_read + cache_creation),
                          summed over the session. Higher = warmer reuse. NEUTRALIZED
                          (50.0, with a note) below 3 scoreable assistant turns: turn
                          one of ANY session has ~zero cache_read by construction, so
                          the raw ratio scored a pristine brand-new session ~77 = "B".
Run:
  context_quality.py score   <path-to.jsonl> [--json] [--explain] [--window N]
  context_quality.py latest  [--project <slug>] [--json] [--explain] [--window N]
  context_quality.py compose <path-to.jsonl>|latest [--project <slug>] [--json]
  context_quality.py cache   <path-to.jsonl>|latest [--project <slug>] [--json]

`compose` -- what FILLS the window (the story the occupancy scalar cannot tell)
------------------------------------------------------------------------------
`score` says how full the context is; `compose` says what is in it. It attributes the
SERIALIZED BYTES of every record to a category -- human_prompts (bucketed by HOW the
human reached the session: `typed` at a terminal, `control_room` through the operator
delivery lane) / assistant_text / tool_calls (per tool) / tool_results (per the tool
that PRODUCED them, joined through tool_use_id) / sidechain / other -- under one
accounting identity that holds for every transcript:
        sum(categories) + transcript_overhead == record_bytes_measured
Nothing is silently dropped: an unrecognized block or record type shows up as a
labeled `other` bucket instead of vanishing, and a tool_result whose tool_use_id
resolves to nothing lands in an explicit `unattributed_results` bucket. Three honesty
rules it keeps:
  - Shares are of the ATTRIBUTED total. `transcript_overhead` -- the per-record
    envelope and a top-level toolUseResult that MIRRORS a block in its own record --
    is pure bookkeeping that never rides in the window (45% of the bytes on a
    measured real transcript), so it is counted and printed but kept out of the
    denominator. `record_bytes_measured` is the sum over the records MEASURED
    (post-compaction, tail-only when partial) -- not the file size.
  - Post-compaction only. Records at or before the LAST compact_boundary no longer
    occupy the window, so they are EXCLUDED and the excluded count is reported.
  - The system prompt, tool schemas and CLAUDE.md imports are NEVER in the transcript,
    so they cannot be measured here. What is reported instead is an explicitly
    labeled APPROXIMATION of that remainder (effective_input * ~4 chars/token minus
    `approx_subtrahend_bytes`, floored at 0) -- never presented as a measurement.
Oversized transcripts (> OVERSIZE_TAIL_BYTES, the same threshold the floor hooks use)
are read through read_jsonl_tail and the output is marked partial_tail: the shares are
then tail-local, not whole-session.

`cache` -- WHEN reuse broke (the story the cache-efficiency scalar cannot tell)
------------------------------------------------------------------------------
The third signal is one number: cache_read over effective input. `cache` reconstructs
the per-turn history behind it from the only cache facts a transcript records -- the
three usage counters and the per-record timestamp. It reports totals and the read
share (the SAME ratio the signal scores, over the same turns), a per-turn series, the
BREAKS in reuse with an honest heuristic class each (`compaction`, which the transcript
records; `gap-consistent`, which says only that the idle gap was long enough to be
consistent with an expiry; and `prefix-change`, the explicit residual), what each break
cost in re-warm tokens, and how many turns the session took to reach steady reuse.
Unlike `compose`, it does NOT cut at the last compaction: a compaction is one of the
break causes it classifies, so the history is the subject rather than noise.

Calibration status (sourced vs guess)
--------------------------------------
  - SOURCED (shape): occupancy degrades gently then steeply — modeled on long-context
    retrieval benchmarks (MRCR-style multi-round co-reference / needle tests show recall
    near-flat early, sagging mid-context, dropping sharply past ~70-85% fill).
  - CALIBRATED against a private session corpus (via the maintainers' calibration
    tooling): the stale-read / duplicate-output / occupancy anchor
    knees sit on the observed distributions (e.g. repeated_reads med 1 / p75 5 / p90 11), so
    the healthy majority is barely penalized and only the p90+ tail is. Cross-checked vs
    Token-Optimizer ContextQ (Pearson ~0.60 -- correlated but independent; we deliberately
    diverge on long, low-final-occupancy sessions that ContextQ's cumulative-fill bug
    under-scores). Window is inferred up to 1M when a turn's effective input exceeds the
    detected window (fixed newer large-context models such as claude-fable-5). The corrective
    ratio is ignored below 3 human turns (one "no" was maxing it). Re-run the harness after
    collecting more data; use --explain to inspect raw inputs.
  - STILL A PREFERENCE KNOB (no ground-truth labels): the 0.40/0.35/0.25 signal weights, the
    degradation sub-weights, the corrective lexicon, and the agent-note GATE (grade-first:
    fires only when score < 80 AND occ >= 0.65; see band_of).
    Context windows: the map lists the LEGACY 200k tier (Haiku, the 3.x family, Sonnet/Opus
    <= 4.5); everything else -- including every model newer than the map -- defaults to 1M,
    the catalog norm since mid-2026. The inverse shape (list the 1M models, default 200k)
    rots: a newly shipped model misses the list and occupancy reads ~5x too full. The
    >window => 1M inference net bumps a listed-200k model that provably carries more. Authoritative live source remains the Models API max_input_tokens; kept
    static to preserve the no-network, per-turn-hook design.
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Reusable: piecewise-linear interpolation between (x, y) anchor points.
# NO step functions anywhere — every value->score mapping goes through here.
# --------------------------------------------------------------------------- #
def interpolate(value, points):
    """Linear-interpolate value across sorted (x, y) anchors; clamp outside range."""
    pts = sorted(points)
    if value <= pts[0][0]:
        return float(pts[0][1])
    if value >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return float(y1)
            t = (value - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(pts[-1][1])


# --------------------------------------------------------------------------- #
# Context-window detection from the model id (no window field exists in the
# transcript). HEURISTIC lookup; override with --window.
#
# The map lists the 200k LEGACY tier and defaults the REST to 1M. The inverse
# shape (list the 1M models, default 200k) rots: a newly shipped large-context
# model misses the list and is scored against 200k (~5x too small), so occupancy
# reads ~5x too full and the nudge fires on near-empty sessions. Since mid-2026
# every current Opus/Sonnet/Fable/Mythos ships 1M: NEW models are the common
# case and must be right BY DEFAULT; the finite, shrinking legacy set is what a
# static list can keep up with. The failure modes are asymmetric but both bounded: an unlisted LEGACY
# 200k model now under-reads occupancy (fewer nudges; the inference net cannot
# catch that direction), while a listed-200k model that really carries more
# still self-corrects via the >window => 1M net in score_session.
# --------------------------------------------------------------------------- #
# The 200k tier, matched as substrings of the model id. The dated forms are
# deliberate discrimination: "sonnet-4-2025" matches claude-sonnet-4-20250514
# (Sonnet 4, 200k) but NOT claude-sonnet-4-6-20251114 (Sonnet 4.6, 1M) -- the
# "-6-" breaks the substring; same construction for opus.
KNOWN_200K_TOKENS = (
    "haiku",                                      # every shipped Haiku (incl. 4.5)
    "claude-3", "sonnet-3", "opus-3",             # the whole 3.x family (incl. reversed forms)
    "sonnet-4-5", "sonnet-4-0", "sonnet-4-2025",  # Sonnet 4.5 / Sonnet 4 (alias + dated)
    "opus-4-5", "opus-4-1", "opus-4-0", "opus-4-2025",  # Opus 4.5 / 4.1 / 4 (alias + dated)
)
# Version-style tags carry a DIGIT BOUNDARY: a bare
# substring "opus-4-1" also matches a future "opus-4-10", dragging a new 1M
# model back to 200k -- the exact rot the inversion exists to end. "haiku" and
# the date-prefix tags stay plain substrings: a date deliberately CONTINUES in
# digits ("sonnet-4-2025" is a prefix of "...-20250514" by design).
_KNOWN_200K_RE = re.compile("|".join(
    re.escape(t) if t == "haiku" or t.endswith("2025") else re.escape(t) + r"(?!\d)"
    for t in KNOWN_200K_TOKENS))
WINDOW_1M = 1_000_000
# Unknown/new (and absent -- effectively unreachable when scoring: the model is
# read off the final scoreable turn, and with no such turn occupancy never
# consults the window) => the 1M tier.
DEFAULT_WINDOW = WINDOW_1M


def detect_window(model):
    # NON-STRING = ABSENT. `(model or "").lower()` looked total and was
    # not: a TRUTHY non-string model -- `{"id": "..."}`, a bare number, a list --
    # sails past the `or` and raises AttributeError on .lower(). Both hooks call the
    # scorer under a bare `except`, so that one AttributeError did not surface as an
    # error anywhere: it silently turned EVERY scoring pass into a no-op and took the
    # whole context signal offline with no trace (the same failure shape `_as_int`
    # already documents for malformed usage numbers). isinstance is the fence; the
    # PRIMARY one is at the extraction boundary (`_model`), which normalizes before a
    # value ever gets here -- this is defense-in-depth for the direct callers of a
    # public function (tests, other tools), not a second policy.
    m = model.lower() if isinstance(model, str) else ""
    if _KNOWN_200K_RE.search(m):
        return 200_000
    return DEFAULT_WINDOW


# --------------------------------------------------------------------------- #
# Transcript loading + record helpers. Fail-open: a corrupt line is skipped.
# --------------------------------------------------------------------------- #
def _parse_jsonl_lines(lines):
    """Fail-open line parser shared by read_jsonl / read_jsonl_tail."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue  # fail-open: never crash on one bad record
        if isinstance(rec, dict):
            out.append(rec)
    return out


def read_jsonl(path):
    """Return the list of valid JSON object records; skip blank/corrupt lines."""
    with open(path, encoding="utf-8") as f:
        return _parse_jsonl_lines(f)


TAIL_DEFAULT_BYTES = 2 * 1024 * 1024


def read_jsonl_tail(path, max_bytes=TAIL_DEFAULT_BYTES, grow=True):
    """Parse only the LAST max_bytes of a transcript; return (records, partial).

    The oversized-transcript fallback for the per-turn hooks: parsing a 100MB+
    JSONL before every prompt is a latency tax, but SKIPPING the score (the old
    behavior above 25MB) silenced the occupancy warning exactly when a long
    session needed it most. Occupancy -- the highest-weighted signal and the
    only one that gates the nudge (band_of fires solely on score+occupancy, and
    at occ >= 0.65 the composite is occupancy-capped below the score gate) -- is
    measured on the FINAL assistant turn, so a tail is sufficient for the gate.
    Degradation/cache over a tail are recent-window approximations; the caller
    must label the result (partial=True), not pass it off as whole-session truth.

    Seeks to size-max_bytes and drops everything up to the first newline (an
    almost-certainly partial line; decode uses errors="replace" so a mid-
    multibyte seek cannot raise). If the tail holds no newline there is no
    complete record to parse and the result is honestly empty. partial=False
    means the whole file fit and the read is identical to read_jsonl.

    `grow` (default True, so every pre-existing caller is byte-identical) controls
    the x4 growth loop below. It exists for ONE caller with the opposite risk
    profile: the composition sidecar, which is DISPLAY-ONLY. The growth
    loop is right for SCORING -- an empty parse scores as an empty session, i.e.
    HEALTHY, which is a lie in the dangerous direction -- but it is what makes the
    parse volume unbounded, since a single huge record drags the budget up x4 at a
    time. A composition simply does not exist when it cannot be produced within a
    fixed budget, and absence is already a first-class answer on that surface, so
    that caller passes grow=False and gets a HARD ceiling instead of a guarantee of
    at least one record. Do NOT pass grow=False from a scoring path.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], False
    # Floor the budget at 1 byte: a zero/negative max_bytes would otherwise pin the
    # growth loop below at 0 forever (0 * 4 == 0) -- an infinite loop in a fail-open
    # helper. A 1-byte start just grows x4 until a record lands.
    max_bytes = max(1, int(max_bytes))
    if size <= max_bytes:
        return read_jsonl(path), False
    budget = max_bytes
    while True:
        with open(path, "rb") as f:
            f.seek(size - budget)
            data = f.read(budget)
        if budget < size:
            nl = data.find(b"\n")
            data = data[nl + 1:] if nl >= 0 else b""
        out = _parse_jsonl_lines(data.decode("utf-8", "replace").splitlines())
        if out or budget >= size or not grow:
            return out, budget < size
        # Known edge: a single record LARGER than the budget (one
        # huge tool_result line) left the tail without a complete record -- and an
        # empty parse scores as an EMPTY session, i.e. HEALTHY: a lie in the
        # dangerous direction (silent nudge, false band recovery). Grow bounded
        # (x4) until at least one record lands or the whole file is read; the
        # common case still pays only max_bytes.
        budget = min(size, budget * 4)


def _message(rec):
    m = rec.get("message")
    return m if isinstance(m, dict) else {}


def _usage(rec):
    u = _message(rec).get("usage")
    return u if isinstance(u, dict) else None


def _model(rec):
    """The turn's model id, or None when the record carries no USABLE one.

    THE SINGLE NORMALIZATION BOUNDARY for `model`. A transcript record
    is whatever the JSON held, so `message.model` can be a dict, a number, a list --
    anything. Everything downstream treats it as a string-or-absent: detect_window
    lowercases it, and score_session PUBLISHES it in the cached result JSON the
    statuslines and the daemon read. Normalizing here means one isinstance check
    protects all of them, instead of one guard per consumer that the next consumer
    forgets.

    NON-STRING => None, and that is EXACTLY the pre-existing absent-model behaviour,
    not a new state: the two other readers of this value compare it against the
    literal "<synthetic>" (_is_real_assistant, _is_real_turn), and None fails that
    comparison identically to a dict -- so the scoreable-turn predicate is the same
    for every input. detect_window is never handed a dict, and the emitted "model"
    field is never an object.

    PARITY, asserted (session-metrics.ts `scoreableAssistant`): the TypeScript mirror
    already ends with `typeof model === "string" ? model : undefined`, i.e. it has
    always treated a non-string model as absent. This closes a REAL asymmetry -- the
    TS lane returned undefined where python raised -- so the two lanes now agree that
    a non-string model is absent, and then diverge only in the documented, deliberate
    way recorded under `known_divergences.absent_model_window` in
    testdata/golden-session.expected.json (python defaults the window to 1M and
    reports occupancy against it; resolveWindow returns undefined and publishes
    nothing). Do not "fix" that second half here.
    """
    m = _message(rec).get("model")
    return m if isinstance(m, str) else None


def _content_blocks(rec):
    c = _message(rec).get("content")
    return c if isinstance(c, list) else []


def _as_int(v):
    """Fail-open int coercion for transcript numbers.

    A usage field is whatever the JSON held. A non-numeric one (`"input_tokens":
    "n/a"`) made int() raise ValueError, and every caller sits under a bare
    `except Exception` in the hooks -- so ONE malformed number silently took the
    whole context signal offline rather than degrading it. Fail open: unusable
    value counts as 0, matching this module's stated posture on corrupt records.

    OverflowError is in the net too: json.loads ACCEPTS the bare tokens
    `Infinity` / `-Infinity` and yields a float, and int(float("inf")) raises
    OverflowError, not ValueError -- so the fail-open promise had a hole exactly
    where a JSON parser hands you a legal-but-unusable number. (NaN already landed
    in ValueError.)
    """
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return 0


def effective_input(usage):
    """Occupancy numerator: tokens the model actually carried as context on a call.
    input + cache_read + cache_creation. Output tokens are NOT occupancy."""
    return (_as_int(usage.get("input_tokens") or 0)
            + _as_int(usage.get("cache_read_input_tokens") or 0)
            + _as_int(usage.get("cache_creation_input_tokens") or 0))


def _is_real_assistant(r):
    """A genuine assistant API turn (not sidechain, not a <synthetic> error placeholder)."""
    return (r.get("type") == "assistant" and not r.get("isSidechain")
            and _model(r) != "<synthetic>" and _usage(r) is not None)


def scoreable_assistants(records):
    """Real assistant API turns (see _is_real_assistant)."""
    return [r for r in records if _is_real_assistant(r)]


# --------------------------------------------------------------------------- #
# CLI-lane central telemetry — turns/cost/minutes, derived HONESTLY from
# the on-disk CLI transcript JSONL (NOT the SDK message stream the daemon taps
# for the SDK lane; the daemon captures num_turns/total_cost_usd off the SDK
# 'result' message, which the CLI transcript never carries).
# --------------------------------------------------------------------------- #
def _parse_iso_ts(ts):
    """Parse a transcript ISO-8601 timestamp (e.g. '2026-07-06T12:34:56.789Z') to
    epoch seconds (float). Fail-open: None on any parse failure/absence."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # datetime.fromisoformat() only accepted a bare 'Z' suffix from Python
        # 3.11 -- normalize to an explicit UTC offset so this parses on older
        # stdlib too (this repo targets no-network, stdlib-only, best-effort).
        s = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _is_real_turn(r):
    """A genuine conversational turn for num_turns: a real (non-sidechain,
    non-<synthetic>) assistant API turn, OR a genuine typed human prompt (NOT a
    tool_result echoed back as a user-role record). Mirrors the same isSidechain/
    model/promptSource markers _is_real_assistant and the corrective-turn scan
    already use elsewhere in this module -- no new heuristic invented."""
    if r.get("isSidechain"):
        return False
    t = r.get("type")
    if t == "assistant":
        return _model(r) != "<synthetic>"
    if t == "user":
        return r.get("promptSource") == "typed"
    return False


def session_totals(records):
    """turns/cost/minutes for the central telemetry payload (both POSTed by
    the Stop hook every turn AND consumed by the SessionEnd hook's caller
    indirectly via the same transcript). VERIFIED against real CLI transcripts
    (a private CLI-transcript corpus): there is NO per-message cost field
    anywhere in this format -- no 'costUSD', no 'total_cost_usd', no
    type=='result' record at all (that SDK-only message type is never written to
    the CLI transcript). total_cost_usd is therefore honestly None for this lane
    unless a message actually carries a 'costUSD' field (summed if so, e.g. if a
    future transcript format grows one) -- this function never invents a number
    either way.

    Returns {"num_turns", "total_cost_usd", "duration_ms", "started_at",
    "updated_at"}; the latter two are epoch SECONDS (int), matching the central
    nowSeconds convention. duration_ms is wall-clock: max-timestamp minus
    min-timestamp across every record carrying a parseable 'timestamp' field.
    """
    n_turns = 0
    total_cost = 0.0
    have_cost = False
    first_ts = None
    last_ts = None
    for r in records:
        if _is_real_turn(r):
            n_turns += 1
        msg = _message(r)
        c = msg.get("costUSD") if isinstance(msg, dict) else None
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            total_cost += float(c)
            have_cost = True
        ts = _parse_iso_ts(r.get("timestamp"))
        if ts is not None:
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts
    duration_ms = None
    if first_ts is not None and last_ts is not None:
        duration_ms = max(0, round((last_ts - first_ts) * 1000))
    return {
        "num_turns": n_turns,
        "total_cost_usd": (round(total_cost, 6) if have_cost else None),
        "duration_ms": duration_ms,
        "started_at": (int(first_ts) if first_ts is not None else None),
        "updated_at": (int(last_ts) if last_ts is not None else None),
    }


# --------------------------------------------------------------------------- #
# Signal 1 — occupancy at session end (weight 0.40)
# --------------------------------------------------------------------------- #
# fraction -> health(0..100). SHAPE is sourced (MRCR-style long-context retrieval:
# near-flat early, sag mid, sharp drop late); the y-VALUES are heuristic, tune later.
# EARLY-WARNING curve: a worker needs headroom to FINISH, so the score
# degrades by mid-fill -- ~50% = "wrap up soon" (room to finish, not to start new work),
# ~70%+ = "finish/hand off now". Steeper than ContextQ. SHAPE (gentle then steep) still
# tracks long-context retrieval (MRCR-style) decay; the y-values are an early-bias choice.
# CORROBORATED by a compaction-anchor harvest over a transcript corpus: well-sampled
# models auto-compact (~ "window effectively full") at p50 ~0.80-0.88 / p90 ~0.97 fill
# (opus-4-8 0.80/0.97 over 1M; haiku-4-5 0.88/1.04 over 200k). The 0.70->0.90 steep drop
# (38->24->12) brackets that real fill-at-limit, so the knees are VALIDATED, not moved.
# Low-n cohorts that compact at ~0.17 are user-threshold /compact, not window-fill, excluded.
# Re-derivable any time via the maintainers' calibration tooling (its empirical
# occupancy-anchor pass).
OCCUPANCY_ANCHORS = [
    (0.00, 98.0), (0.25, 90.0), (0.40, 78.0), (0.50, 66.0), (0.60, 52.0),
    (0.70, 38.0), (0.80, 24.0), (0.90, 12.0), (1.00, 4.0), (1.20, 0.0),
]


def occupancy_signal(records, window):
    asst = scoreable_assistants(records)
    if not asst or not window:
        # No assistant usage -> near-empty context -> healthy.
        return 98.0, {"effective_input": 0, "window": window,
                      "occupancy_fraction": 0.0,
                      "note": "no assistant usage; treated as near-empty"}
    usage = _usage(asst[-1])          # the FINAL real assistant API call
    eff = effective_input(usage)
    frac = eff / float(window)
    return interpolate(frac, OCCUPANCY_ANCHORS), {
        "effective_input": eff, "window": window,
        "occupancy_fraction": round(frac, 4)}


# --------------------------------------------------------------------------- #
# Signal 2 — degradation proxy from session behavior (weight 0.35)
# Four sub-components, each mapped to a 0..1 health, then weighted. All HEURISTIC.
# --------------------------------------------------------------------------- #
# Repeated Read of an already-read path = thrashing.
# Calibrated (post edit-aware tightening): repeated_reads now counts only TRUE
# redundant re-reads (re-read with no edit between) -> corpus med 1, p75 4, p90 7, max 45.
# Penalized a bit sooner than the loose proxy since each counted re-read is genuine churn.
STALE_ANCHORS = [(0, 1.0), (3, 0.85), (8, 0.5), (25, 0.0)]
# Identical large tool outputs returned again = wasted context.
# Recalibrated after the dedup fix (one payload/record): genuine duplicate
# outputs are RARE -- corpus med/p75/p90 = 0, max 17 -> register a real duplicate cluster.
DUP_ANCHORS = [(0, 1.0), (2, 0.75), (6, 0.45), (15, 0.0)]
# Fraction of human turns that read as corrections.
CORR_ANCHORS = [(0.0, 1.0), (0.15, 0.7), (0.35, 0.3), (0.6, 0.0)]
# Even ONE auto-compaction means context was lost — penalize immediately.
COMPACT_ANCHORS = [(0, 1.0), (1, 0.55), (2, 0.30), (4, 0.0)]
# Sub-weights (sum 1.0): compaction + corrective rated highest as the truest
# quality proxies; stale/dup are cheaper churn signals. Heuristic.
DEG_WEIGHTS = {"stale": 0.30, "dup": 0.20, "corrective": 0.25, "compaction": 0.25}
DUP_MIN_CHARS = 500  # only "large" outputs count as worth-flagging duplicates (heuristic)
MODIFY_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}  # a Read after one of these is a legit re-read
# A Bash command may modify files with no visible file_path (redirection, sed -i, cp/mv,
# generators). On a command matching these write indicators we conservatively invalidate ALL
# read state -- safer to under-count stale (false negatives) than to flag legit re-reads.
BASH_WRITE_RE = re.compile(
    r">"                                              # any redirection: > >> >| &> 2>
    r"|\btee\b"
    r"|\b(?:cp|mv|dd|install|rsync|truncate|ln)\b"
    r"|\b(?:sed|perl)\b[^|;&]*\s-\S*i"            # in-place sed/perl
    r"|\b(?:python3?|node|ruby|bash|sh|make|go|cargo|npm|pnpm|yarn)\b"  # interpreters/build
)


def _bash_may_write(cmd):
    return bool(cmd and isinstance(cmd, str) and BASH_WRITE_RE.search(cmd))
# Small, explicit, commented-as-heuristic lexicon for human corrective turns.
CORRECTIVE_LEXICON = (
    "no,", "no.", "nope", "actually", "that's wrong", "thats wrong", "incorrect",
    "undo", "revert", "wrong", "not what i", "instead", "do not", "don't",
)


def degradation_signal(records):
    # (a) redundant re-read churn: a Read is "stale" only if the file was already read AND
    # not modified since -- re-reading after an Edit/Write is legitimate, not thrashing.
    # (A plain "any re-read" proxy over-counts.)
    last_event = {}  # path -> "read" | "modified"
    repeated, total_reads = 0, 0
    for r in records:
        if r.get("type") != "assistant" or r.get("isSidechain"):
            continue
        for b in _content_blocks(r):
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            name = b.get("name")
            inp = b.get("input") or {}
            if name == "Bash":
                # Bash may write files we cannot see -> invalidate read state on write-ish cmds.
                if _bash_may_write(inp.get("command")):
                    last_event.clear()
                continue
            path = inp.get("file_path") or inp.get("notebook_path")
            if not path:
                continue
            if name == "Read":
                total_reads += 1
                if last_event.get(path) == "read":
                    repeated += 1  # consecutive re-read, no edit/Bash-write between => stale
                last_event[path] = "read"
            elif name in MODIFY_TOOLS:
                last_event[path] = "modified"
    # (b) duplicate large tool outputs (hash normalized payloads)
    counts = {}
    for r in records:
        if r.get("isSidechain"):
            continue
        # One payload per tool-output EVENT. Each tool_result block is one event; the
        # top-level toolUseResult is a redundant MIRROR of a block, so blocks are the source
        # -> avoids double-counting the mirror (fix A) AND catches dups in the 2nd+ block of a
        # parallel-tool record (fix B). toolUseResult is used only when there is no block.
        payloads = [json.dumps(b.get("content"), sort_keys=True, default=str)
                    for b in _content_blocks(r)
                    if isinstance(b, dict) and b.get("type") == "tool_result"]
        if not payloads and r.get("toolUseResult") is not None:
            payloads = [json.dumps(r.get("toolUseResult"), sort_keys=True, default=str)]
        for payload in payloads:
            if len(payload) >= DUP_MIN_CHARS:
                h = hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()
                counts[h] = counts.get(h, 0) + 1
    dup_count = sum(c - 1 for c in counts.values() if c > 1)
    # (c) corrective-turn ratio over GENUINE human prompts (promptSource == "typed")
    # Only typed human turns that come AFTER an assistant turn count -- a correction replies
    # to the assistant's work; the opening prompt's task constraints ("Do NOT commit") are
    # instructions, not corrections.
    humans = []  # eligible typed prompts (post-assistant), as text
    corr = 0
    seen_assistant = False
    for r in records:
        if r.get("isSidechain"):
            continue
        if _is_real_assistant(r):
            seen_assistant = True
            continue
        if r.get("type") == "user" and r.get("promptSource") == "typed" and seen_assistant:
            c = _message(r).get("content")
            text = c if isinstance(c, str) else " ".join(
                b.get("text", "") for b in (c or []) if isinstance(b, dict))
            humans.append(text or "")
            low = (text or "").lower().strip()
            if any(tok in low for tok in CORRECTIVE_LEXICON):
                corr += 1
    # Min-sample guard: below 3 eligible turns the ratio is too noisy (one "no" -> 1.0).
    corr_ratio = (corr / len(humans)) if len(humans) >= 3 else 0.0
    # (d) auto-compaction events
    compactions = sum(1 for r in records
                      if r.get("type") == "system" and r.get("subtype") == "compact_boundary")

    comp = {
        "stale": interpolate(repeated, STALE_ANCHORS),
        "dup": interpolate(dup_count, DUP_ANCHORS),
        "corrective": interpolate(corr_ratio, CORR_ANCHORS),
        "compaction": interpolate(compactions, COMPACT_ANCHORS),
    }
    health = sum(comp[k] * DEG_WEIGHTS[k] for k in DEG_WEIGHTS)
    return health * 100.0, {
        "repeated_reads": repeated, "total_reads": total_reads,
        "duplicate_outputs": dup_count,
        "corrective_turns": corr, "human_turns": len(humans),
        "corrective_ratio": round(corr_ratio, 3),
        "compactions": compactions,
        "component_health": {k: round(v, 3) for k, v in comp.items()},
    }


# --------------------------------------------------------------------------- #
# Signal 3 — cache efficiency (weight 0.25)
# --------------------------------------------------------------------------- #
# cache-read share -> health. Direction (more reuse = better) is the principle;
# y-values heuristic. Deliberately the ONLY token-ratio signal (no output/input
# "productivity" ratio — with cache_read in the denominator it just re-measures this).
CACHE_ANCHORS = [(0.0, 10.0), (0.5, 50.0), (0.8, 80.0), (0.95, 95.0), (1.0, 100.0)]
# COLD-START MIN-SAMPLE GUARD, the same idiom the corrective ratio already
# uses: turn ONE of any session has ~zero cache_read BY CONSTRUCTION (there is nothing
# warm to read yet), so the raw ratio scored a pristine brand-new session ~10/100 on
# this signal and composited to ~77 = grade B -- a visible lie the moment a grade is
# rendered anywhere. Below CACHE_MIN_TURNS scoreable assistant turns the signal is
# therefore NEUTRAL (50.0) and says so in its raw dict; the measured ratio is still
# reported (the statusline's CacheU reads raw.ratio and must keep telling the truth),
# it just does not move the score. At/above the threshold the real ratio scores.
CACHE_MIN_TURNS = 3


def cache_efficiency_signal(records):
    asst = scoreable_assistants(records)
    total_cache, total_eff = 0, 0
    for r in asst:
        u = _usage(r)
        total_cache += _as_int(u.get("cache_read_input_tokens") or 0)
        total_eff += effective_input(u)
    if total_eff == 0:
        return 50.0, {"total_cache_read": 0, "total_effective": 0,
                      "ratio": None, "note": "no token data; neutral"}
    ratio = total_cache / float(total_eff)
    if len(asst) < CACHE_MIN_TURNS:
        return 50.0, {
            "total_cache_read": total_cache, "total_effective": total_eff,
            "ratio": round(ratio, 4), "assistant_turns": len(asst),
            "min_turns": CACHE_MIN_TURNS,
            "note": ("cold start: fewer than {} scoreable assistant turns; ratio "
                     "measured but NOT scored (neutral 50)".format(CACHE_MIN_TURNS))}
    return interpolate(ratio, CACHE_ANCHORS), {
        "total_cache_read": total_cache, "total_effective": total_eff,
        "assistant_turns": len(asst),
        "ratio": round(ratio, 4)}


# --------------------------------------------------------------------------- #
# Composite score + grade
# --------------------------------------------------------------------------- #
WEIGHTS = {"occupancy": 0.40, "degradation": 0.35, "cache_efficiency": 0.25}


def grade_for(score):
    # Heuristic S/A-F ladder + a human-readable band.
    # Ladder aligned to Token-Optimizer ContextQ for cross-comparability:
    for cut, g, band in [(90, "S", "pristine"), (80, "A", "healthy"),
                         (70, "B", "good"), (55, "C", "fraying"),
                         (40, "D", "degraded"), (0, "F", "critical")]:
        if score >= cut:
            return g, band
    return "F", "critical"


def score_session(records, window=None):
    records = list(records or [])
    # Model = the FINAL scoreable assistant turn's model. Occupancy is measured on that
    # final turn, so a mixed-model transcript must window against the model that produced
    # the last turn, not the first.
    _asst = scoreable_assistants(records)
    model = _model(_asst[-1]) if _asst else None
    if window is None:
        window = detect_window(model)
        # Inference safety net: a turn cannot carry more context than the window; if any
        # turn's effective input exceeds the detected window, the guess was too small
        # (a newer large-context model not in the map) -> use the 1M tier. Skipped when
        # --window is supplied explicitly.
        if _asst:
            # Occupancy is the FINAL turn, so test the FINAL turn's effective input against
            # the final model's window -- NOT the max across the transcript (an earlier
            # large-window turn must not force a smaller final model's window up). (review fix)
            _final_eff = effective_input(_usage(_asst[-1]))
            if _final_eff > window:
                window = 1_000_000
    occ_s, occ_raw = occupancy_signal(records, window)
    deg_s, deg_raw = degradation_signal(records)
    cache_s, cache_raw = cache_efficiency_signal(records)
    overall = (WEIGHTS["occupancy"] * occ_s
               + WEIGHTS["degradation"] * deg_s
               + WEIGHTS["cache_efficiency"] * cache_s)
    grade, band = grade_for(overall)
    return {
        "score": round(overall, 1), "grade": grade, "band": band,
        "model": model, "window": window,
        "signals": {
            "occupancy": {"score": round(occ_s, 1), "weight": WEIGHTS["occupancy"], "raw": occ_raw},
            "degradation": {"score": round(deg_s, 1), "weight": WEIGHTS["degradation"], "raw": deg_raw},
            "cache_efficiency": {"score": round(cache_s, 1), "weight": WEIGHTS["cache_efficiency"], "raw": cache_raw},
        },
    }


# --- Band model (telemetry signal/noise contract) -------------------------
# THE nudge gate, single source: context-signal.sh (the in-context
# nudge) and session-telemetry.sh stop (bus band-change events) both CALL this --
# neither re-implements the thresholds, so they cannot disagree by construction
# (an inline copy "mirroring" this would be a mirror enforced by nothing, which
# is how definitions drift). GRADE-FIRST GATE ("trust the grade"): a band fires ONLY when score < 80 (below grade A) AND occ >= 0.65.
#   score >= 80  OR  occ < 0.65  -> "ok"        (grade A+ or low fill => silent)
#   occ >= 0.75                  -> "strong"
#   0.65 <= occ < 0.75           -> "advisory"
# There is no low-occupancy "quality" band: occ < 0.65 suppresses every band.
def band_of(score, occ):
    try:
        occ = float(occ); score = float(score)
    except (TypeError, ValueError):
        return "ok"
    # Grade-first gate: trust the composite. Grade A+ (>=80) OR sub-65% fill => never nudge.
    if score >= 80 or occ < 0.65:
        return "ok"
    if occ >= 0.75:
        return "strong"
    return "advisory"   # 0.65 <= occ < 0.75

def should_emit(prev_band, cur_band):
    # Edge-triggered: emit only when the band CHANGES (worsening OR recovery),
    # and on the first observation (prev is None). Same band => noise => no emit.
    return prev_band != cur_band


# --------------------------------------------------------------------------- #
# `compose` — WHAT fills the window, by category (the occupancy scalar says how
# full; this says what is in it). Byte-attribution, not token-counting: the
# transcript records no per-piece token counts, so serialized bytes are the only
# thing actually MEASURABLE here. Bytes are a proxy for tokens, not tokens.
#
# THE INVARIANT: every byte of every attributed record lands in exactly one
# place, and
#     sum(category buckets) + transcript_overhead == record_bytes_measured
# holds for every transcript. That is why `other` exists and why it is bucketed
# rather than opaque: an unrecognized block type or record type must become a
# visible labeled row, never a silent drop. The walk sums the whole-record
# serializations independently and appends an integrity note if the identity ever
# fails (it cannot, by the substring property below).
#
# WHY `transcript_overhead` IS ITS OWN LINE and not an `other` bucket: two byte
# classes are pure transcript bookkeeping that NEVER ride in the context window --
# the per-record envelope (uuid/timestamp/parentUuid/cwd/... and JSON scaffolding)
# and the top-level `toolUseResult` MIRROR of a tool_result block. On a real
# 431-record transcript they were 45% of the file. Folding them into the shares
# would answer "what is in this FILE", and would say "half your context is
# envelope" -- false, and false in the misleading direction, since the question
# this subcommand exists for is what is in the WINDOW. They are still measured,
# still bucketed, still printed, and still in the sum identity above; they are
# just not in the denominator of the shares. (Measured corroboration on that same
# transcript: content total 748,427 B vs effective_input*4 = 772,628 B -- the two
# agree to ~3%; the file total, 1,369,282 B, exceeds the window by 77%.)
#
# Substring property: _blob_bytes serializes a piece with the SAME json.dumps
# arguments used for the whole record, so a nested piece's serialization is a
# verbatim substring of the record's -- per-record piece bytes can never exceed
# the record's own bytes, and the remainder (uuids, timestamps, keys, JSON
# scaffolding) is a non-negative "envelope" bucket.
#
# COMPACTION POLICY: records at or before the LAST
# compact_boundary are EXCLUDED. After a compaction those turns no longer occupy
# the window -- the carried-over summary does -- so attributing them would answer
# "what did this session ever contain", not "what is in the window now", which is
# the question this subcommand exists for. The count of excluded records and the
# number of boundaries are reported, so the exclusion is visible, never silent.
# --------------------------------------------------------------------------- #
# Same latency threshold the transcript-reading hooks apply before falling back to
# read_jsonl_tail (bin/context-signal.sh, bin/session-telemetry.sh stop mode).
# Pinned by a test so the three copies cannot drift apart unnoticed.
OVERSIZE_TAIL_BYTES = 25 * 1024 * 1024
# Chars-per-token divisor for the LABELED approximation of the unattributed
# remainder (system prompt + tool schemas + imports -- none of which the transcript
# records, so none of which can be measured here). Rough English/code average.
CHARS_PER_TOKEN = 4
COMPOSE_CATEGORIES = ("human_prompts", "assistant_text", "tool_calls",
                      "tool_results", "sidechain", "other")
UNATTRIBUTED_RESULTS = "unattributed_results"
_TEXTLIKE_BLOCKS = ("text", "thinking", "redacted_thinking")

# --------------------------------------------------------------------------- #
# The OPERATOR delivery marker.
#
# SOURCE OF TRUTH: core/src/backend/delivery-framing.ts, `OPERATOR_DELIVERY_MARKER`
# (in the brokkr integration; the vendor contract is documented in the
# mythical-ctxmonitor package's docs/brokkr-integration.md). This is a COPY, and
# it is a copy only because a python scorer cannot import a TypeScript module.
# Brokkr's own CI pins these bytes from its side -- its test suite reads its
# vendored copy of this file and asserts the marker against the TS constant --
# so the copy cannot drift silently past a vendor bump: change the TS constant
# and that suite fails until the copy follows.
#
# WHY A HEADER AND NOT A FIELD. A prompt an operator types into the Control Room
# reaches the harness as a delivery envelope the daemon injects into the session's
# input queue. The CLI writes that turn to Store-2 as an ordinary `user` record with
# NO `promptSource` -- the field only marks input the CLI itself read from a
# terminal -- so structurally the record is indistinguishable from a compaction
# summary or an injected reminder, and it landed in `other`/`user_text`. The one
# thing that IS distinguishable is the framing the daemon writes at position 0 of
# the message, and it is the same string the delivery engine's own release proof
# scans transcripts for (`operatorDeliveryToken`). Keying on it here means the
# composition and the delivery proof agree, byte for byte, about which turns came
# from the human.
#
# POSITION-EXACT, mirroring the TS verifier (`scanTranscript`): the header is
# matched as a PREFIX of the user text, never as a substring. A body that merely
# quotes the header does not become a human prompt.
#
# NOT A SECURITY BOUNDARY, said plainly: this is a display attribution, and a human
# who pastes the header verbatim as the first line of a turn will have that turn
# counted as a human prompt. It was one either way.
OPERATOR_DELIVERY_MARKER = "message from the operator via the Control Room"
# The full header prefix the framer emits: `[<marker> | delivery <id> | class: <c>]`.
# Bounded by ` | delivery ` so a turn that merely opens with the marker sentence in
# some other framing is not swept in.
OPERATOR_PROMPT_PREFIX = "[" + OPERATOR_DELIVERY_MARKER + " | delivery "
# The verifier's token is bounded on BOTH sides -- `operatorDeliveryToken` closes with
# ` |` after the id -- so matching the opening prefix alone was laxer than the contract
# it claims to mirror: a turn beginning with the marker plus a
# half-written header would have been attributed to the operator. This pattern is the
# same shape as the token: a non-empty id that cannot itself contain the delimiter, and
# the closing ` |`.
#
# LEADING WHITESPACE IS TOLERATED, deliberately, and it is not laxness: the TS verifier
# reads `userEntryText(entry).trimStart()` (transcript-verifier.ts `scanTranscript`), so
# a whitespace-shifted header still satisfies the release proof and the turn IS a
# delivered Control Room prompt. Refusing it here would have made the composition
# undercount turns the delivery engine itself vouched for -- mirroring means mirroring
# the whole rule, including this one.
#
# ...and mirroring it means mirroring ECMAScript's ALPHABET, not python's.
# `\s` and `trimStart()` are not the same set, and they disagree in BOTH directions:
# python's `\s` matches U+001C..U+001F and U+0085, which JS does NOT trim (a turn behind
# one would be attributed to the operator though the verifier rejects it), while JS trims
# U+FEFF, which python's `\s` does not (a turn the verifier accepted would be missed).
# So the class is spelled out as ECMAScript's WhiteSpace + LineTerminator productions,
# and `_JS_TRIMSTART_CLASS` is what the drift pin in the test suite checks.
_JS_TRIMSTART_CLASS = (
    "\t\n\v\f\r   "
    "           "
    "    　﻿"
)
OPERATOR_PROMPT_RE = re.compile(
    r"^[" + re.escape(_JS_TRIMSTART_CLASS) + r"]*\["
    + re.escape(OPERATOR_DELIVERY_MARKER)
    + r" \| delivery [^|\]\s]+ \|"
)
# The bucket operator-originated prompts land in, kept SEPARATE from `typed` so the
# popover can still tell "the human sat at a terminal" from "the human used the
# Control Room" -- both are human prompts, and they arrived by different roads.
OPERATOR_PROMPT_BUCKET = "control_room"


def _blob_bytes(obj):
    """Serialized byte length of one piece, measured exactly as whole records are."""
    try:
        return len(json.dumps(obj, default=str).encode("utf-8"))
    except Exception:            # fail-open: never crash the walk on one weird piece
        return len(repr(obj).encode("utf-8"))


def _tool_bucket(name):
    """Bucket label for a tool. MCP tools (mcp__<server>__<tool>) group under their
    SERVER -- one row per server beats fifty rows of near-identical tool names."""
    if isinstance(name, str) and name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) >= 2 and parts[1]:
            return "mcp:" + parts[1]
    if isinstance(name, str) and name:
        return name
    return "<unnamed>"


def _index_one_call(live, block):
    """Fold ONE `tool_use` block into the live `tool_use_id -> bucket` map."""
    tid = block.get("id")
    if isinstance(tid, str) and tid:
        live[tid] = _tool_bucket(block.get("name"))


def index_tool_use(live, record):
    """Fold a whole record's `tool_use` blocks into the live map, at record
    granularity. Used for records that are NOT dissected block-by-block -- i.e. the
    ones the compaction cut excluded, whose calls must still resolve a result that
    survived. Attributed records index INLINE instead (see compose_session), because
    only per-block placement makes the rule below exact within a single content list.

    THE RULE: the map is folded forward in FILE ORDER, so a `tool_result` resolves
    against the MOST RECENT PRECEDING call carrying its id -- the only rule that is
    right under id reuse whichever direction the reuse runs. (The first cut chose
    main-chain-first + first-writer-wins; that pins the STALE producer when
    a pre-compaction call and a live one share an id, and "collision impossible"
    was false. Record-granular indexing let a result resolve to a call
    that FOLLOWS it, and deferring to end-of-record hid a call that PRECEDES
    it in the same list.) A result whose call sits in the un-parsed head of a partial
    tail resolves to nothing and is reported as unattributed.

    SIDECHAIN CALLS ARE DELIBERATELY NOT INDEXED: sidechain records are attributed
    whole and never dissected, so no sidechain result is ever resolved, and a
    sidechain call therefore has nothing legitimate to resolve -- indexing it could
    only ever shadow a main-chain producer.
    """
    if record.get("isSidechain"):
        return
    for b in _content_blocks(record):
        if isinstance(b, dict) and b.get("type") == "tool_use":
            _index_one_call(live, b)


def split_at_last_compaction(records):
    """(kept, excluded_count, boundary_count) per the COMPACTION POLICY above."""
    last, boundaries = -1, 0
    for i, r in enumerate(records):
        if r.get("type") == "system" and r.get("subtype") == "compact_boundary":
            boundaries += 1
            last = i
    if last < 0:
        return list(records), 0, 0
    return list(records[last + 1:]), last + 1, boundaries


def _add(cats, category, bucket, nbytes):
    if nbytes:
        b = cats[category]
        b[bucket] = b.get(bucket, 0) + nbytes


def _user_entry_text(record):
    """The concatenated user text of one record -- a byte-for-byte mirror of the TS
    verifier's `userEntryText` (core/src/backend/transcript-verifier.ts), so the two
    languages agree on what "the text of this turn" means. String content passes
    through; a content LIST joins each block's `text` (missing -> empty) with a
    newline, exactly as the verifier does."""
    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            text = block.get("text") if isinstance(block, dict) else None
            parts.append(text if isinstance(text, str) else "")
        return "\n".join(parts)
    return ""


def _human_prompt_bucket(record, rtype):
    """Which `human_prompts` bucket this record belongs to, or None if it is not a
    human prompt at all.

    TWO ROADS, ONE HUMAN:
      · `typed`        -- the CLI's own `promptSource` marker: somebody typed it at a
                          terminal. Structured metadata, preferred wherever present.
      · `control_room` -- the operator framing at position 0 of a delivery the daemon
                          injected. There is no structured marker for this (see
                          OPERATOR_DELIVERY_MARKER above), so the header IS the key.

    A PEER delivery is deliberately NOT a human prompt. Its header is a different
    string entirely (`[mythical delivery <id> | from: <slug> | ...]`) and it fails
    this test, so an agent's message to another agent keeps landing in
    `other`/`user_text` exactly as before. A message from a sibling session is real
    window content, but it is not the human speaking."""
    if rtype != "user":
        return None
    if record.get("promptSource") == "typed":
        return "typed"
    if OPERATOR_PROMPT_RE.match(_user_entry_text(record)):
        return OPERATOR_PROMPT_BUCKET
    return None


def _add_text(cats, rtype, human_bucket, kind, nbytes):
    """Route one text-ish piece. Assistant prose/thinking is assistant_text; a turn
    `_human_prompt_bucket` recognized as coming from the human is a human prompt,
    under the bucket naming HOW it arrived; every other user text (compaction
    summaries, local-command caveats, injected reminders, peer-agent deliveries) is
    real window content but NOT a human prompt, so it is a labeled `other` bucket
    rather than an inflated one."""
    if rtype == "assistant":
        _add(cats, "assistant_text", kind, nbytes)
    elif human_bucket:
        _add(cats, "human_prompts", human_bucket, nbytes)
    elif rtype == "user":
        _add(cats, "other", "user_text", nbytes)
    else:
        _add(cats, "other", "unrecognized_blocks", nbytes)


def compose_session(records, partial=False):
    """Attribute the serialized bytes of a transcript to context categories.

    `partial` is read_jsonl_tail's own flag (NOT the size threshold): it says the
    parse skipped the head, so every share is tail-local.
    """
    records = list(records or [])
    kept, excluded, boundaries = split_at_last_compaction(records)
    cats = {name: {} for name in COMPOSE_CATEGORIES}
    overhead = {}          # transcript bookkeeping; measured, reported, NOT in the shares
    record_bytes = 0
    # Walk ALL records in file order to keep the tool_use_id map live (see
    # index_tool_use), but ATTRIBUTE only from `cut` on -- the post-compaction slice.
    cut = len(records) - len(kept)
    live = {}

    for i, r in enumerate(records):
        # Attributed records index their calls INLINE, block by block, as the content
        # list is walked (see the tool_use branch below); excluded ones are folded in
        # whole here. Either way the map only ever holds calls that PRECEDE the result
        # being resolved.
        if i < cut:
            index_tool_use(live, r)   # excluded from the shares; still resolves results
            continue
        rb = _blob_bytes(r)
        record_bytes += rb
        if r.get("isSidechain"):
            # Subagent traffic, one bucket: it never occupied THIS window, but it is
            # in the file and the sums must account for it. Kept whole (undissected).
            # index_tool_use is a no-op here by design -- see its docstring.
            _add(cats, "sidechain", "sidechain", rb)
            continue
        rtype = r.get("type")
        human_bucket = _human_prompt_bucket(r, rtype)
        used, pieces, result_blocks = 0, 0, 0
        content = _message(r).get("content")

        if isinstance(content, str):
            n = _blob_bytes(content)
            used += n
            pieces += 1
            _add_text(cats, rtype, human_bucket, "text", n)
        elif isinstance(content, list):
            for b in content:
                n = _blob_bytes(b)
                used += n
                pieces += 1
                if not isinstance(b, dict):
                    # A non-dict entry in a content list: skipping it
                    # dropped its bytes into the ENVELOPE, which is documented as never
                    # riding in the window -- a silent misattribution the byte identity
                    # cannot catch. It is unrecognized CONTENT; label it as such.
                    _add(cats, "other", "unrecognized_blocks", n)
                    continue
                btype = b.get("type")
                if btype == "tool_use":
                    _add(cats, "tool_calls", _tool_bucket(b.get("name")), n)
                    # Index INLINE, at this block's position: indexing the
                    # whole record up-front let a result resolve to a call that
                    # FOLLOWS it (r3), and moving it to the end of the record hid a
                    # call that PRECEDES it in the same list. Folding it in here is
                    # the only placement under which "most recent PRECEDING call" is
                    # true in both directions, within a record and across records.
                    _index_one_call(live, b)
                elif btype == "tool_result":
                    result_blocks += 1
                    tid = b.get("tool_use_id")
                    who = live.get(tid) if isinstance(tid, str) else None
                    # No resolvable id -> an EXPLICIT bucket, never a silent drop.
                    _add(cats, "tool_results", who or UNATTRIBUTED_RESULTS, n)
                elif btype in _TEXTLIKE_BLOCKS:
                    _add_text(cats, rtype, human_bucket, btype, n)
                else:
                    _add(cats, "other", "unrecognized_blocks", n)
        elif content is not None:
            # message.content that is neither a string nor a list: a future
            # single-object shape. Its bytes are counted here as unrecognized CONTENT,
            # never as ENVELOPE (which is documented as never riding in the window).
            n = _blob_bytes(content)
            used += n
            pieces += 1
            _add(cats, "other", "unrecognized_blocks", n)

        tur = r.get("toolUseResult")
        if tur is not None:
            n = _blob_bytes(tur)
            used += n
            pieces += 1
            if result_blocks == 1:
                # A MIRROR of this record's single tool_result block: transcript
                # bookkeeping that does NOT ride in the window. Counted separately so
                # it neither inflates tool_results nor hides inside the envelope.
                #
                # STRUCTURAL test, and MEASURED rather than assumed (40
                # real transcripts, 10KB-6MB, most recently modified): 3080 records
                # carried a toolUseResult and ALL 3080 sat beside EXACTLY ONE
                # tool_result block -- zero with none, zero with two or more. The
                # 1:1 shape is what the CLI writes.
                # AN IDENTITY TEST WAS TRIED AND REJECTED ON THE EVIDENCE:
                # the mirror is written in a DIFFERENT envelope from the block (the
                # block often carries a RENDERED view -- "The file ... has been
                # updated successfully" -- while the mirror carries structured data)
                # and it has no tool_use_id, so the only candidate test is textual
                # containment. Measured on the same corpus, containment FAILS on
                # 1521 of 2921 string-content mirrors and 150 of 159 object ones --
                # it would misclassify over half of all REAL mirrors as content.
                # A structural test with zero observed counterexamples beats a
                # content test that is wrong half the time.
                # THE RESIDUAL, stated plainly and undecidable by construction: a
                # top-level payload that is NOT this record's mirror would be counted
                # here and kept out of the shares. Nothing in the record can
                # distinguish that case -- see above -- so it is surfaced as a runtime
                # caveat on transcript_overhead["note"] rather than silently assumed.
                # The bias below is one-directional and applies OUTSIDE this shape
                # only: there the payload counts as CONTENT, because over-counting a
                # duplicate in a labeled bucket is a visible error while hiding real
                # tool output from the shares is a silent one.
                overhead["tool_result_mirror"] = overhead.get("tool_result_mirror", 0) + n
            else:
                # 0 blocks: degradation_signal already treats this shape as the real
                # payload ("toolUseResult is used only when there is no block"), so
                # calling it overhead would drop a genuine tool output from the
                # shares. 2+ blocks: one top-level value cannot mirror them all, so
                # it is not established as a mirror. Either way the record
                # carries no tool_use_id, so the producer is unresolvable.
                _add(cats, "tool_results", UNATTRIBUTED_RESULTS, n)

        rest = rb - used
        if rest > 0:
            if pieces:
                # Leftover of a record we DID dissect: uuids, timestamps, keys, JSON
                # scaffolding. Transcript-only -- never in the window.
                overhead["envelope"] = overhead.get("envelope", 0) + rest
            else:
                # Nothing recognized in this record (attachments, system records,
                # snapshots): counted WHOLE and labeled by type -- conservative, since
                # some of these (attachments) really are window content and we must
                # not drop them. The reader can see the type and judge.
                _add(cats, "other", "record:{}".format(rtype), rest)

    total = sum(sum(bk.values()) for bk in cats.values())
    overhead_bytes = sum(overhead.values())
    notes = []
    if partial:
        notes.append("partial_tail: only the transcript TAIL was parsed; every share "
                     "below is tail-local, not whole-session. The ~unattributed "
                     "overhead approximation is MEANINGLESS here -- the un-parsed "
                     "head would be counted into it.")
        notes.append("partial_tail: the compaction counts below are TAIL-LOCAL too -- a "
                     "compact_boundary in the un-parsed head cannot be seen, so "
                     "boundaries=0 here means 'none in the tail', NOT 'none in the "
                     "session'.")
    if boundaries:
        notes.append("compaction: {} boundary/boundaries; {} record(s) at or before the "
                     "LAST one are EXCLUDED -- after a compaction they no longer occupy "
                     "the window.".format(boundaries, excluded))
    if cats["sidechain"]:
        notes.append("sidechain bytes are subagent traffic: they occupied the SUBAGENT's "
                     "window, not this one. Counted here because they are in the file.")
    if total + overhead_bytes != record_bytes:
        notes.append("integrity: categories {} + overhead {} != measured record bytes {} "
                     "-- report this, the walk is meant to make them equal.".format(
                         total, overhead_bytes, record_bytes))

    def _rows(bucketmap, denom=None):
        d = total if denom is None else denom
        return [{"name": k, "bytes": v,
                 "share": (round(v / float(d), 4) if d else 0.0)}
                for k, v in sorted(bucketmap.items(), key=lambda kv: (-kv[1], kv[0]))]

    categories = [{"category": name,
                   "bytes": sum(cats[name].values()),
                   "share": (round(sum(cats[name].values()) / float(total), 4) if total else 0.0),
                   "buckets": _rows(cats[name])}
                  for name in COMPOSE_CATEGORIES]
    categories.sort(key=lambda c: (-c["bytes"], c["category"]))

    # Occupancy numerator: the final scoreable assistant turn AMONG THE KEPT RECORDS.
    # Reading it off the full list contradicted the post-compaction
    # policy -- a transcript ending "...assistant(900k), compact_boundary, summary"
    # attributed ~nothing but quoted a 900k pre-compaction numerator and derived the
    # approximation from it. In the ordinary case the final turn IS post-boundary, so
    # this still agrees with `score` exactly; when it does not, the fallback to the
    # full list is taken ONLY to avoid reporting 0, and it says so in the notes.
    kept_asst = scoreable_assistants(kept)
    all_asst = scoreable_assistants(records)
    if kept_asst:
        eff = effective_input(_usage(kept_asst[-1]))
    elif all_asst:
        eff = effective_input(_usage(all_asst[-1]))
        notes.append("effective_input is PRE-COMPACTION: no scoreable assistant turn "
                     "survives the boundary cut, so the numerator comes from the last "
                     "turn before it and does not describe the current window.")
    else:
        eff = 0
    # The approximation's subtrahend EXCLUDES sidechain bytes:
    # subagent traffic never rode in this window, so leaving it in biased the
    # remainder downward -- a big sidechain could drive it to 0 and imply "no system
    # prompt". Sidechain stays a reported category with a share, per the contract.
    subtrahend = total - sum(cats["sidechain"].values())
    return {
        "records": len(records),
        "records_attributed": len(kept),
        "compaction": {"boundaries": boundaries, "excluded_records": excluded,
                       "policy": "attribute only records AFTER the last compact_boundary"},
        "partial_tail": bool(partial),
        "total_bytes": total,
        "record_bytes_measured": record_bytes,
        "transcript_overhead": {
            "bytes": overhead_bytes,
            "buckets": _rows(overhead, denom=overhead_bytes),
            "note": ("record envelope + a toolUseResult that mirrors this record's single "
                     "tool_result block: transcript bookkeeping that does not ride in the "
                     "context window. Measured and reported, but excluded from the shares "
                     "above. CAVEAT: the mirror test is STRUCTURAL -- the mirror is written "
                     "in a different envelope from the block and carries no tool_use_id, so "
                     "identity cannot be tested (textual containment was measured and "
                     "rejected: it misses over half of real mirrors). A top-level payload "
                     "that is NOT this record's mirror would be counted here and would not "
                     "appear in the shares. Measured over a private transcript corpus: "
                     "3080/3080 toolUseResult records sat beside exactly one block, zero "
                     "counterexamples."),
        },
        "categories": categories,
        "effective_input": eff,
        "approx_subtrahend_bytes": subtrahend,
        "approx_unattributed_overhead_bytes": max(0, eff * CHARS_PER_TOKEN - subtrahend),
        "approx_basis": (
            "APPROXIMATION, NOT MEASURED: effective_input * {} chars/token minus "
            "approx_subtrahend_bytes (= total_bytes less the sidechain category, which never "
            "rode in this window), floored at 0. Stands for the system prompt, tool schemas "
            "and imported files, none of which the transcript records.".format(CHARS_PER_TOKEN)),
        "notes": notes,
    }


def load_for_compose(path, oversize_bytes=OVERSIZE_TAIL_BYTES,
                     tail_bytes=TAIL_DEFAULT_BYTES):
    """(records, partial) — the whole file, or only its TAIL above the size threshold.

    Exactly the hooks' two-step: the threshold decides WHETHER to fall back, and
    read_jsonl_tail's own `partial` flag (never the threshold) decides whether the
    result is honestly partial -- tail growth that ends up reading the whole file is
    NOT partial. Both budgets are parameters so a test can exercise the tail path
    without writing a 25MB fixture.
    """
    try:
        oversize = os.path.getsize(path) > oversize_bytes
    except OSError:
        oversize = False
    if oversize:
        return read_jsonl_tail(path, max_bytes=tail_bytes)
    return read_jsonl(path), False


# --------------------------------------------------------------------------- #
# `cache` — the per-turn story behind the cache-efficiency SCALAR (the third
# signal's detail surface, the way `compose` is occupancy's).
#
# `score` reports one number for prompt-cache reuse: cache_read over effective
# input, summed across the session. That scalar cannot say WHEN the reuse broke,
# how often, or what it cost — and those are the only actionable parts. This walk
# reconstructs them from the ONLY cache facts a transcript records: the three usage
# counters on each assistant turn (`cache_read_input_tokens`,
# `cache_creation_input_tokens`, `input_tokens`) and the per-record timestamp.
#
# THE HONESTY RULE THAT SHAPES EVERYTHING HERE: a transcript records OUTCOMES, not
# CAUSES. It says a turn read 0 cached tokens and created 180k; it never says why.
# The cache TTL is not recorded, the system prompt and tool schemas are not
# recorded, and no field anywhere says "the cache expired". So this module names
# what it can prove and refuses to name what it cannot:
#
#   * a BREAK is defined by measurement, not by cause — a turn that reused less
#     than {CACHE_BREAK_RETAINED} of the prefix that was warm when the previous
#     turn ended. That is an observation about two numbers.
#   * its CLASS is evidence-ranked, and two of the three names say out loud that
#     they are circumstantial:
#       - `compaction`   — a `compact_boundary` record sits between the two turns.
#                          RECORDED, not inferred: the transcript states it, and a
#                          compaction demonstrably rewrites the prefix. Strongest
#                          evidence, so it wins when it applies.
#       - `gap-consistent` — the idle gap before the turn is >= CACHE_TTL_GAP_SEC.
#                          NOT "expired": the TTL is not in the transcript and this
#                          walk cannot observe one. The name states exactly what
#                          was measured — a gap CONSISTENT with an expiry — and a
#                          long think between turns produces the same reading.
#       - `prefix-change` — the explicit RESIDUAL. Neither of the above held, so
#                          something the transcript does not record changed the
#                          cached prefix (a settings/tool-schema change, a model
#                          switch, an edited system prompt, a harness restart). It
#                          is named for the only thing that must be true — the
#                          prefix stopped matching — and claims nothing further.
#   * TURN ONE IS NEVER A BREAK. The first scoreable turn of any session reads ~0
#     by construction (nothing is warm yet), which is the same cold start
#     `cache_efficiency_signal`'s min-sample guard exists for. Counting it would
#     report every healthy session as opening with a break.
#
# COMPACTION POLICY, DELIBERATELY THE OPPOSITE OF `compose`'s. The composition
# EXCLUDES records at or before the last `compact_boundary`, because they no longer
# occupy the window. This walk KEEPS them: a compaction is one of the break causes
# it classifies, so cutting the history would delete exactly the evidence the
# surface exists to show, and the totals would stop reconciling with the score's
# own cache ratio (which is also whole-transcript). The two surfaces answer
# different questions — "what is in the window NOW" vs "how did reuse behave over
# this session" — so they cut differently and say so.
#
# READ SHARE IS THE SCORER'S OWN RATIO -- ON THE SAME RECORDS. `totals.read_share`
# is computed over the same turn set (`_is_real_assistant`) and the same
# denominator as `cache_efficiency_signal`, so given the same records it equals
# `signals.cache_efficiency.raw.ratio` EXACTLY. That is deliberate: a detail
# surface that disagreed with the scalar it explains would be worse than none.
#
# THE ONE CASE WHERE THE TWO CAN DIFFER, stated rather than implied: the
# hook feeds the score a GROWING tail (read_jsonl_tail with grow=True, because an
# empty parse would score as a healthy empty session) and feeds this walk a
# NON-growing one, so on an oversized transcript the score can cover more of the
# file than the report does. Whenever that is possible the report is flagged
# `partial_tail` and says so in its own notes -- the identity is claimed only for
# a whole-file read, never asserted over a span this walk did not measure.
# --------------------------------------------------------------------------- #
# The idle gap at/above which a break is CONSISTENT with a cache expiry. 300 s is
# the documented default prompt-cache TTL, and it is the only number here with an
# outside source — but the TTL a given call actually ran under is NOT recorded, so
# this is a threshold for a NAME, never a claim that a cache expired. A longer TTL
# (or a break at a shorter gap) is invisible to the transcript either way.
CACHE_TTL_GAP_SEC = 300
# The retention knee. A turn reusing less than half of the prefix that was warm at
# the end of the previous turn counts as a break. HEURISTIC: reuse is normally
# monotone (this turn's read ~= the previous turn's read + creation, since the old
# prefix is a prefix of the new one), so a real break collapses read toward 0 and
# anything near 1.0 is intact. Half is far from both, which is what keeps ordinary
# jitter out of the list; the exact value is a preference knob, not a measurement.
CACHE_BREAK_RETAINED = 0.5
# "Steady reuse" for the warm-up count: the first turn whose OWN read share reaches
# this. Heuristic, and the same direction as CACHE_ANCHORS (more reuse = warmer).
CACHE_WARM_READ_SHARE = 0.5


# An ISO timestamp is a measurable INSTANT here only when it carries an explicit
# zone. `_parse_iso_ts` hands a zone-less string to
# `datetime.fromisoformat().timestamp()`, which resolves it in the HOST's local
# timezone -- so the same transcript would publish different `at` values, and could
# classify a break differently across a DST boundary, depending on where the hook
# ran. Every timestamp Claude Code writes is a `...Z` form, so this rejects nothing
# real; it just refuses to turn an ambiguous string into an environment-dependent
# number. The score path is deliberately left alone: `session_totals` uses these
# timestamps only as a DIFFERENCE within one host, where the offset cancels.
_ZONED_ISO_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def _zoned_iso_ts(value):
    """Epoch seconds for an ISO timestamp carrying an EXPLICIT zone; else None."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not _ZONED_ISO_RE.search(s):
        return None
    return _parse_iso_ts(s)


def _cache_break_class(boundaries, gap):
    """The break's class, evidence-ranked. See the header for what each name claims.

    `boundaries` = compact_boundary records between the two turns; `gap` = the idle
    gap in seconds, or None when either timestamp was unparseable. An unmeasurable
    gap CANNOT be called gap-consistent, so it falls to the residual — which the
    payload makes visible by omitting `gap_sec` on that row rather than printing a 0.
    """
    if boundaries:
        return "compaction"
    if gap is not None and gap >= CACHE_TTL_GAP_SEC:
        return "gap-consistent"
    return "prefix-change"


def cache_session(records, partial=False):
    """Per-turn prompt-cache behaviour: totals, breaks, re-warm cost, warm-up.

    `partial` is read_jsonl_tail's own flag (NOT the size threshold), exactly as in
    `compose_session`: it says the parse skipped the head, so every count below is
    tail-local and turn 0 is the first turn IN THE TAIL, not in the session.
    """
    records = list(records or [])
    rows, breaks, notes = [], [], []
    total_read = total_creation = total_uncached = 0
    rewarm_total = lost_total = 0
    boundaries_since = 0        # compact_boundary records since the last scoreable turn
    prev_warm = None            # the prefix warm at the end of the previous scoreable turn
    prev_ts = None
    unknown_gap = 0             # breaks whose idle gap could not be measured
    for r in records:
        if r.get("type") == "system" and r.get("subtype") == "compact_boundary":
            boundaries_since += 1
            continue
        if not _is_real_assistant(r):
            continue
        u = _usage(r)
        read = _as_int(u.get("cache_read_input_tokens") or 0)
        creation = _as_int(u.get("cache_creation_input_tokens") or 0)
        uncached = _as_int(u.get("input_tokens") or 0)
        ts = _zoned_iso_ts(r.get("timestamp"))
        turn = len(rows)
        total_read += read
        total_creation += creation
        total_uncached += uncached
        row = {"turn": turn, "at": (int(ts) if ts is not None else None),
               "cache_read_tokens": read, "cache_creation_tokens": creation,
               "uncached_input_tokens": uncached}
        # A break needs a PREVIOUS warm prefix to have lost. Turn one has none (see
        # the header), and neither does a turn following one that cached nothing.
        if prev_warm:
            if read < prev_warm * CACHE_BREAK_RETAINED:
                gap = None
                if ts is not None and prev_ts is not None:
                    # Clamped at 0: an out-of-order or skewed pair of timestamps is
                    # not evidence of a NEGATIVE idle period, and a negative gap
                    # would silently fail the >= test either way.
                    gap = max(0.0, ts - prev_ts)
                cls = _cache_break_class(boundaries_since, gap)
                if gap is None:
                    unknown_gap += 1
                lost = max(0, prev_warm - read)
                rewarm_total += creation
                lost_total += lost
                row["break"] = True
                breaks.append({
                    "turn": turn,
                    "at": (int(ts) if ts is not None else None),
                    "gap_sec": (round(gap, 1) if gap is not None else None),
                    "class": cls,
                    # The creation-token spike attributable to the re-warm, and an
                    # UPPER BOUND on it rather than a measurement of it:
                    # this is ALL the cache-creation on the break turn, and a turn
                    # writes both the prefix it had to re-warm AND whatever content
                    # is genuinely new. The transcript reports one number for both,
                    # so the split is not observable -- what IS measured is that a
                    # break happened here and that this many tokens were written.
                    "rewarm_tokens": creation,
                    # The warm prefix this turn did not reuse.
                    "lost_tokens": lost,
                })
        rows.append(row)
        prev_warm = read + creation
        prev_ts = ts
        boundaries_since = 0

    effective = total_read + total_creation + total_uncached
    # Warm-up: the index of the first turn whose OWN read share reaches steady
    # reuse == the number of turns spent getting there. None when it never did,
    # which is an honest absence and not a 0 (0 means "warm from turn one").
    # "Steady reuse" is AT LEAST half of that turn's input coming from the cache --
    # stated as `>=` here and everywhere it is described (an exact 50/50
    # split is a tie, so calling it "dominant" would over-read the threshold).
    warmup = None
    for row in rows:
        eff = (row["cache_read_tokens"] + row["cache_creation_tokens"]
               + row["uncached_input_tokens"])
        if eff and row["cache_read_tokens"] / float(eff) >= CACHE_WARM_READ_SHARE:
            warmup = row["turn"]
            break

    if partial:
        notes.append("partial_tail: only the transcript TAIL was parsed. Every count below "
                     "is tail-local, turn 0 is the first turn IN THE TAIL, and a break in "
                     "the un-parsed head cannot be seen.")
        notes.append("partial_tail: the read share below is therefore NOT guaranteed to equal "
                     "the score's cache ratio -- the score reads a GROWING tail and this walk "
                     "a bounded one, so on an oversized transcript the two can describe "
                     "different spans of the same session.")
    if not rows:
        notes.append("no scoreable assistant turn in the measured records: every total below "
                     "is 0 because nothing was MEASURED, not because nothing was cached.")
    elif len(rows) < CACHE_MIN_TURNS:
        notes.append("cold start: fewer than {} scoreable turns, so the share below is "
                     "dominated by turn one's by-construction zero -- the same reason the "
                     "score NEUTRALIZES its cache signal at this length.".format(CACHE_MIN_TURNS))
    if unknown_gap:
        notes.append("{} break(s) carry no measurable idle gap (unparseable or absent "
                     "timestamps), so they could not be tested for gap-consistency and fall "
                     "in the residual class.".format(unknown_gap))
    return {
        "turns": len(rows),
        "partial_tail": bool(partial),
        "totals": {
            "cache_read_tokens": total_read,
            "cache_creation_tokens": total_creation,
            "uncached_input_tokens": total_uncached,
            "effective_input_tokens": effective,
            # 0.0 when nothing was measured -- `turns`/`effective_input_tokens` are
            # what distinguish that from a real zero-reuse session, and the notes
            # above say so in prose. Same convention as compose_session's shares.
            "read_share": (round(total_read / float(effective), 4) if effective else 0.0),
        },
        "breaks": breaks,
        "break_count": len(breaks),
        "rewarm_tokens": rewarm_total,
        "lost_tokens": lost_total,
        "warmup_turns": warmup,
        "series": rows,
        "break_basis": (
            "MEASURED, NOT CAUSAL: a break is a turn that reused less than {:.0%} of the "
            "prefix that was warm when the previous turn ended (this turn's "
            "cache_read vs the previous turn's cache_read + cache_creation). The first "
            "turn is never a break -- nothing is warm yet. rewarm_tokens is ALL the "
            "cache-creation on a break turn, so it is an UPPER BOUND on the re-warm: a "
            "turn writes both the prefix it lost and any genuinely new content, and the "
            "transcript reports one number for both.".format(CACHE_BREAK_RETAINED)),
        "class_basis": (
            "A transcript records outcomes, never causes. `compaction` is RECORDED (a "
            "compact_boundary sits between the two turns). `gap-consistent` means only "
            "that the idle gap was >= {}s, which is consistent with a cache expiry -- the "
            "TTL is not recorded and no expiry is observable, and a long think reads the "
            "same. `prefix-change` is the explicit RESIDUAL: the warm prefix was not "
            "reused and neither recorded explanation applies. Common shapes it covers, "
            "none of them distinguishable here: a settings or tool-schema change, a model "
            "switch, and turns issued so close together that the previous turn's cache "
            "write had not landed -- which is why breaks often arrive in "
            "runs.".format(CACHE_TTL_GAP_SEC)),
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# `compose_cache_payload` — the BOUNDED projection the per-prompt hook drops for
# the daemon to read back (bin/context-signal.sh -> ctxcompose-<sid>.json;
# core/src/telemetry/ctxmonitor-cache.ts readCtxComposition).
#
# WHY A PROJECTION AND NOT `compose_session` VERBATIM. The drop file is written by
# a session and read by the daemon, so its size is an input the READER cannot
# choose: whatever the reader's cap is, it is the bound on what one session can
# make the daemon allocate and parse. `compose_session`'s own output is unbounded
# in exactly one dimension -- the per-bucket tables, whose row COUNT and whose row
# NAMES both come from the transcript (a tool name is echoed verbatim, an MCP
# server groups but is still transcript-supplied). A session that calls 5,000
# distinct tool names, or one tool whose name is 30 KB, produces a file no fixed
# cap can hold, and the honest outcome would be "no composition" for exactly the
# session that has the most to say. So the payload is bounded HERE, by
# construction, and the reader's cap is a backstop rather than the mechanism:
#
#   * rows: the top {COMPOSE_CACHE_TOP_BUCKETS} buckets by bytes per category. The
#     remainder is NOT dropped -- it is aggregated into an explicitly labeled
#     `buckets_omitted` sibling carrying its count, bytes and share, so the
#     category's own total still accounts for every byte under it. That is the
#     same rule `other` and `unattributed_results` already keep in the walk: an
#     unmodelled remainder becomes a visible labeled row, never a silent drop.
#   * names: truncated to {COMPOSE_CACHE_NAME_MAX} chars (measured: the longest
#     bucket name across 86 real transcripts was 28) and flagged when truncated.
#   * charset: printable ASCII only. A tool name is transcript-supplied text that
#     ends up on an HTTP wire and in a UI, so control characters, terminal escapes
#     and newlines are replaced with `?` here rather than left for every downstream
#     consumer to remember. This bounds the BYTES and the SHAPE; HTML escaping
#     remains the renderer's job, exactly as it is for every other string field.
#   * prose: the scorer's own `notes`, `approx_basis` and the overhead caveat ride
#     along, length-capped. They are carried rather than restated on the reader's
#     side ON PURPOSE -- they describe how these particular numbers were derived,
#     and a hand-written copy in another language is the mirror-drift class this
#     whole delivery exists to avoid.
#
# NOT INCLUDED, deliberately: the overhead's own bucket split (envelope vs
# tool_result_mirror). `overhead_bytes` + `overhead_note` answer the question the
# shares raise ("what is excluded from the denominator, and on what basis"), and a
# second attacker-shaped array would have to be fenced for a split the CLI can
# still print. Adding it later is additive.
# --------------------------------------------------------------------------- #
COMPOSE_CACHE_VERSION = 1
# Rows kept per category before the remainder is aggregated. MEASURED, not
# guessed: across 86 real transcripts (random sample + the six largest) the
# largest single category held 15 buckets and the p99 was 11 -- but those are
# categories, and the ones that reach double digits are `other`'s record-type
# rows, not tools. 12 keeps every realistic table whole while making the row
# count a constant rather than a function of the transcript.
COMPOSE_CACHE_TOP_BUCKETS = 12
# Categories emitted. compose_session always produces exactly len(COMPOSE_CATEGORIES)
# = 6, so this never binds on the hook's path. It exists because this function is
# PUBLIC and its contract is "bounded by construction", which has to hold for whatever
# it is handed and not only for the one caller that behaves. Matches the
# reader's own COMPOSE_MAX_CATEGORIES, so anything this emits is structurally
# admissible there.
COMPOSE_CACHE_MAX_CATEGORIES = 8
COMPOSE_CACHE_NAME_MAX = 64
# The two long prose fields (`overhead_note` 696 chars, `approx_basis` 283 --
# measured, not guessed). 1024 is above both with room for a re-wording.
COMPOSE_CACHE_TEXT_MAX = 1024
# One `notes` entry. Measured maximum across every note the walk can emit: 211.
COMPOSE_CACHE_NOTE_MAX = 320
# compose_session emits at most 5 notes today (partial x2, compaction, sidechain,
# integrity, pre-compaction numerator). 8 leaves room without letting the array
# become a size lever: notes are bounded at NOTES_MAX * NOTE_MAX bytes, full stop.
COMPOSE_CACHE_NOTES_MAX = 8
# Printable ASCII. Everything else -- control characters, ANSI escapes, newlines,
# non-ASCII -- becomes `?`. Deliberately applied to NAMES and PROSE alike so there
# is one rule, and the reader can enforce the identical one.
_CACHE_TEXT_RE = re.compile(r"[^\x20-\x7e]")
# Ceiling for every count/byte field. 2**53-1 == Number.MAX_SAFE_INTEGER, the exact
# domain the TypeScript reader validates against (`Number.isSafeInteger`). See
# _cache_int for why a ceiling exists at all.
CACHE_INT_MAX = 2 ** 53 - 1


def _cache_text(value, limit=COMPOSE_CACHE_TEXT_MAX):
    """Sanitize + length-cap one string for the drop file. Non-string => ""."""
    if not isinstance(value, str):
        return ""
    return _CACHE_TEXT_RE.sub("?", value)[:limit]


def _cache_int(value):
    """Fail-open non-negative int for the drop file, CLAMPED to the reader's domain.

    Two reasons for the ceiling, and neither is theoretical:

      * SIZE. Python integers are arbitrary-precision, so a single field could serialize
        to hundreds of digits and, across every numeric field and row, push the payload
        past the byte cap the reader enforces -- i.e. defeat the bound this projection
        exists to guarantee. 2**53-1 is 16 digits, so the numeric contribution to the
        payload is a constant.
      * AGREEMENT. The reader validates each of these with `Number.isSafeInteger`
        (core/src/telemetry/ctxmonitor-cache.ts isCtxCount), which is exactly this
        domain. Emitting a value outside it would produce a file the writer considers
        well-formed and the reader refuses whole -- the worst kind of disagreement,
        because it looks like corruption.

    Clamping rather than refusing: these are byte counts, and a count this large is
    already nonsense, so the ceiling is a saturating bound on a value no real transcript
    can reach (2**53 bytes is 9 petabytes) -- not a silent alteration of a real number.
    """
    n = _as_int(value)
    if n <= 0:
        return 0
    return min(n, CACHE_INT_MAX)


def _cache_share(value):
    """A share is a fraction of a total: finite, 0..1. Anything else reads as 0.0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):   # NaN / +-Inf
        return 0.0
    return round(min(1.0, max(0.0, f)), 4)


def _cache_bucket(bucket):
    if not isinstance(bucket, dict):
        return {"name": "", "bytes": 0, "share": 0.0}
    raw = bucket.get("name") if isinstance(bucket.get("name"), str) else ""
    clean = _CACHE_TEXT_RE.sub("?", raw)
    row = {"name": clean[:COMPOSE_CACHE_NAME_MAX],
           "bytes": _cache_int(bucket.get("bytes")),
           "share": _cache_share(bucket.get("share"))}
    if len(clean) > COMPOSE_CACHE_NAME_MAX:
        row["name_truncated"] = True
    return row


def compose_cache_payload(res, top=COMPOSE_CACHE_TOP_BUCKETS):
    """Bounded, drop-file shape of one `compose_session` result. PURE.

    Every field is copied from `res`; nothing is recomputed and nothing is
    invented. The only transformations are the bounding rules described above.
    """
    res = res if isinstance(res, dict) else {}
    # CLAMPED, not merely defaulted: a caller passing a large `top` would
    # emit every transcript-created bucket and could push the payload past the cap the
    # reader enforces -- i.e. the parameter could switch off the very property this
    # function's contract asserts. It stays a parameter so a test can ask for FEWER
    # rows; it can no longer ask for more than the bound.
    top = max(0, min(int(top), COMPOSE_CACHE_TOP_BUCKETS))
    categories = []
    for c in (res.get("categories") or [])[:COMPOSE_CACHE_MAX_CATEGORIES]:
        if not isinstance(c, dict):
            continue
        buckets = c.get("buckets") if isinstance(c.get("buckets"), list) else []
        row = {"category": _cache_text(c.get("category"), COMPOSE_CACHE_NAME_MAX),
               "bytes": _cache_int(c.get("bytes")),
               "share": _cache_share(c.get("share")),
               "buckets": [_cache_bucket(b) for b in buckets[:top]]}
        rest = buckets[top:]
        if rest:
            # The remainder, LABELED. Aggregated from the same rows that were
            # dropped, so `bytes` still accounts for the category's whole total.
            # Both aggregates go BACK through the clamps: a sum of clamped
            # terms is not itself clamped, and this is the one place the projection
            # computes rather than copies.
            row["buckets_omitted"] = {
                "count": _cache_int(len(rest)),
                "bytes": _cache_int(sum(_cache_int(b.get("bytes"))
                                        for b in rest if isinstance(b, dict))),
                "share": _cache_share(sum(_cache_share(b.get("share"))
                                          for b in rest if isinstance(b, dict))),
            }
        categories.append(row)
    ovh = res.get("transcript_overhead") if isinstance(res.get("transcript_overhead"), dict) else {}
    comp = res.get("compaction") if isinstance(res.get("compaction"), dict) else {}
    notes = res.get("notes") if isinstance(res.get("notes"), list) else []
    return {
        "v": COMPOSE_CACHE_VERSION,
        "records": _cache_int(res.get("records")),
        "records_attributed": _cache_int(res.get("records_attributed")),
        "compaction": {"boundaries": _cache_int(comp.get("boundaries")),
                       "excluded_records": _cache_int(comp.get("excluded_records"))},
        "partial_tail": bool(res.get("partial_tail")),
        "total_bytes": _cache_int(res.get("total_bytes")),
        "record_bytes_measured": _cache_int(res.get("record_bytes_measured")),
        "overhead_bytes": _cache_int(ovh.get("bytes")),
        "overhead_note": _cache_text(ovh.get("note")),
        "effective_input": _cache_int(res.get("effective_input")),
        "approx_subtrahend_bytes": _cache_int(res.get("approx_subtrahend_bytes")),
        "approx_unattributed_overhead_bytes": _cache_int(
            res.get("approx_unattributed_overhead_bytes")),
        "approx_basis": _cache_text(res.get("approx_basis")),
        "categories": categories,
        "notes": [_cache_text(n, COMPOSE_CACHE_NOTE_MAX)
                  for n in notes[:COMPOSE_CACHE_NOTES_MAX]],
    }


# --------------------------------------------------------------------------- #
# `cache_report_payload` — the BOUNDED projection of one `cache_session` result,
# dropped beside the other two files (bin/context-signal.sh ->
# ctxcache-<sid>.json; core/src/telemetry/ctxmonitor-cache.ts readCtxCacheReport).
#
# SAME REASONING AS `compose_cache_payload`, and deliberately the same shape of
# solution rather than a second invention: the file is written by a session and
# read by the daemon, so its size is an input the READER cannot choose, and the
# analysis above is unbounded in exactly two dimensions — one series row per
# assistant turn and one row per break, both of which grow with the session. A
# 40-hour session must not produce a file no fixed cap can hold, because the
# honest outcome would then be "no cache report" for exactly the session that has
# the most to say. So the payload is bounded HERE, by construction, and the
# reader's byte cap is a backstop rather than the mechanism.
#
#   * rows: the most RECENT {CACHE_REPORT_MAX_TURNS} turns and
#     {CACHE_REPORT_MAX_BREAKS} breaks. Recent rather than largest, for both, so
#     the two tables describe the same stretch of the session — a "biggest breaks"
#     list next to a recent series would invite reading one off the other.
#   * the remainder is NOT dropped: `turns_omitted` / `breaks_omitted` carry the
#     count AND the aggregated tokens of the rows that were cut, so the session
#     totals still account for every turn. Same rule the composition keeps for
#     `buckets_omitted`, and the same rule the walk itself keeps with its residual
#     `prefix-change` class: an unmodelled remainder becomes a visible labeled row,
#     never a silent drop.
#   * the class token is charset- and length-capped like every other
#     transcript-adjacent string on a drop file.
#   * prose (`break_basis`, `class_basis`, `notes`) rides along, length-capped, for
#     the same reason the composition carries its own: those sentences describe how
#     THESE numbers were derived, and a hand-written copy on the reader's side in
#     another language is the mirror drift this whole delivery avoids.
#
# ABSENT RATHER THAN ZERO, in three places, because a 0 would be a claim:
# `warmup_turns` (absent = steady reuse was never reached; 0 = warm from turn one),
# `at` (absent = no USABLE timestamp -- unparseable, or outside the epoch range
# this payload can carry; see _cache_epoch) and `gap_sec` (absent = the gap could
# not be measured, so the break could not be tested for gap-consistency at all).
# --------------------------------------------------------------------------- #
CACHE_REPORT_VERSION = 1
# Series rows kept. A turn is one row of ~5 small integers, so this is a size knob
# rather than a fidelity one: 64 turns is a long stretch of recent history (the
# median real session in the local corpus is far shorter), and the omitted head is
# still accounted for in `turns_omitted` and in the whole-session totals.
CACHE_REPORT_MAX_TURNS = 64
# Break rows kept. Breaks are rare in a healthy session and clustered in an
# unhealthy one; 16 recent ones is more than enough to see the pattern, and the
# rest are aggregated rather than dropped.
CACHE_REPORT_MAX_BREAKS = 16
# The class token (`gap-consistent` is 14). Bounded, not enumerated — the reader
# applies the same rule, so a FOURTH class is a legitimate upstream change that
# does not need a code change on the other side of the wire.
CACHE_REPORT_CLASS_MAX = 32
# The drop-file PROSE caps are SHARED with the composition on purpose: they are a
# property of "text written by a session that reaches an HTTP wire", not of either
# payload, and one rule is what lets the reader enforce identical numbers for both
# files instead of two sets that drift.
CACHE_REPORT_TEXT_MAX = COMPOSE_CACHE_TEXT_MAX
CACHE_REPORT_NOTE_MAX = COMPOSE_CACHE_NOTE_MAX
CACHE_REPORT_NOTES_MAX = COMPOSE_CACHE_NOTES_MAX


def _cache_seconds(value):
    """A non-negative finite duration for the drop file, clamped to the reader's
    domain. Non-numeric / NaN / +-Inf => None (i.e. the field is OMITTED, never
    printed as a 0 that would read as "no gap"). Same ceiling as _cache_int, for
    the same two reasons: bounded serialization, and agreement with the reader's
    `Number.isFinite` + safe-integer range check.

    OverflowError is in the net for the MIRROR of the reason `_as_int` records: a
    python int is arbitrary-precision, so `float(10**500)` RAISES rather than
    returning inf -- and this function sits under the drop-file writer, whose whole
    contract is that no input can make it throw (own adversarial test, first run).
    """
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(min(float(CACHE_INT_MAX), max(0.0, f)), 1)


def _cache_epoch(value):
    """An epoch SECOND for the drop file, or None when the value cannot carry one.

    A separate helper from _cache_seconds, and the difference is the whole
    point. A DURATION is legitimately clamped at 0 -- a negative idle gap is
    clock skew, and 0 is the truthful floor. A TIMESTAMP is not: clamping a
    pre-epoch value to 0 would publish `1970-01-01` as an observation, when this
    payload's own contract is that an unusable timestamp is OMITTED. So a negative
    epoch reads as absent, exactly like an unparseable one.
    """
    sec = _cache_seconds(value)
    if sec is None:
        return None
    try:
        if float(value) < 0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return _cache_int(sec)


def _cache_tail(rows, keep):
    """The most recent `keep` rows, and the older ones that were cut. PURE.

    `rows[-keep:]` is WRONG for keep == 0 — it is `rows[0:]`, i.e. everything, the
    exact opposite of what a caller asking for no rows means. Written out so that
    the bound cannot be switched off by the one value most likely to be passed by a
    test or a defensive caller.
    """
    if keep <= 0:
        return [], list(rows)
    cut = max(0, len(rows) - keep)
    return list(rows[cut:]), list(rows[:cut])


def _cache_turn_row(row):
    if not isinstance(row, dict):
        return {"turn": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "uncached_input_tokens": 0}
    out = {"turn": _cache_int(row.get("turn")),
           "cache_read_tokens": _cache_int(row.get("cache_read_tokens")),
           "cache_creation_tokens": _cache_int(row.get("cache_creation_tokens")),
           "uncached_input_tokens": _cache_int(row.get("uncached_input_tokens"))}
    at = _cache_epoch(row.get("at"))
    if at is not None:
        out["at"] = at
    if row.get("break"):
        out["break"] = True
    return out


def _cache_break_row(row):
    if not isinstance(row, dict):
        return {"turn": 0, "class": "", "rewarm_tokens": 0, "lost_tokens": 0}
    cls = row.get("class") if isinstance(row.get("class"), str) else ""
    out = {"turn": _cache_int(row.get("turn")),
           "class": _CACHE_TEXT_RE.sub("?", cls)[:CACHE_REPORT_CLASS_MAX],
           "rewarm_tokens": _cache_int(row.get("rewarm_tokens")),
           "lost_tokens": _cache_int(row.get("lost_tokens"))}
    at = _cache_epoch(row.get("at"))
    if at is not None:
        out["at"] = at
    gap = _cache_seconds(row.get("gap_sec"))
    if gap is not None:
        out["gap_sec"] = gap
    return out


def cache_report_payload(res, top_turns=CACHE_REPORT_MAX_TURNS,
                         top_breaks=CACHE_REPORT_MAX_BREAKS):
    """Bounded, drop-file shape of one `cache_session` result. PURE.

    Every field is copied from `res`; nothing is recomputed and nothing is invented
    — except the two `*_omitted` aggregates, which are summed from the very rows
    they account for and then re-clamped (a sum of clamped terms is not itself
    clamped).

    Both bounds are CLAMPED, not merely defaulted: a caller passing a larger value
    could emit one row per turn of an arbitrarily long session and push the payload
    past the cap the reader enforces — i.e. the parameter could switch off the
    property this function's contract asserts. They stay parameters so a test can
    ask for FEWER rows; they cannot ask for more.
    """
    res = res if isinstance(res, dict) else {}
    top_turns = max(0, min(int(top_turns), CACHE_REPORT_MAX_TURNS))
    top_breaks = max(0, min(int(top_breaks), CACHE_REPORT_MAX_BREAKS))
    totals = res.get("totals") if isinstance(res.get("totals"), dict) else {}
    series_raw = res.get("series") if isinstance(res.get("series"), list) else []
    breaks_raw = res.get("breaks") if isinstance(res.get("breaks"), list) else []
    kept_turns, cut_turns = _cache_tail(series_raw, top_turns)
    kept_breaks, cut_breaks = _cache_tail(breaks_raw, top_breaks)
    notes = res.get("notes") if isinstance(res.get("notes"), list) else []
    out = {
        "v": CACHE_REPORT_VERSION,
        "turns": _cache_int(res.get("turns")),
        "partial_tail": bool(res.get("partial_tail")),
        "totals": {
            "cache_read_tokens": _cache_int(totals.get("cache_read_tokens")),
            "cache_creation_tokens": _cache_int(totals.get("cache_creation_tokens")),
            "uncached_input_tokens": _cache_int(totals.get("uncached_input_tokens")),
            "effective_input_tokens": _cache_int(totals.get("effective_input_tokens")),
            "read_share": _cache_share(totals.get("read_share")),
        },
        "break_count": _cache_int(res.get("break_count")),
        "rewarm_tokens": _cache_int(res.get("rewarm_tokens")),
        "lost_tokens": _cache_int(res.get("lost_tokens")),
        "breaks": [_cache_break_row(b) for b in kept_breaks],
        "series": [_cache_turn_row(t) for t in kept_turns],
        "break_basis": _cache_text(res.get("break_basis"), CACHE_REPORT_TEXT_MAX),
        "class_basis": _cache_text(res.get("class_basis"), CACHE_REPORT_TEXT_MAX),
        "notes": [_cache_text(n, CACHE_REPORT_NOTE_MAX)
                  for n in notes[:CACHE_REPORT_NOTES_MAX]],
    }
    # ABSENT, not 0: 0 means "warm from turn one", which is a different claim from
    # "steady reuse was never reached". Anything non-integral reads as absent too.
    warmup = res.get("warmup_turns")
    if isinstance(warmup, int) and not isinstance(warmup, bool) and warmup >= 0:
        out["warmup_turns"] = _cache_int(warmup)
    if cut_turns:
        out["turns_omitted"] = {
            "count": _cache_int(len(cut_turns)),
            "cache_read_tokens": _cache_int(sum(
                _cache_int(t.get("cache_read_tokens")) for t in cut_turns if isinstance(t, dict))),
            "cache_creation_tokens": _cache_int(sum(
                _cache_int(t.get("cache_creation_tokens")) for t in cut_turns if isinstance(t, dict))),
            "uncached_input_tokens": _cache_int(sum(
                _cache_int(t.get("uncached_input_tokens")) for t in cut_turns if isinstance(t, dict))),
        }
    if cut_breaks:
        out["breaks_omitted"] = {
            "count": _cache_int(len(cut_breaks)),
            "rewarm_tokens": _cache_int(sum(
                _cache_int(b.get("rewarm_tokens")) for b in cut_breaks if isinstance(b, dict))),
            "lost_tokens": _cache_int(sum(
                _cache_int(b.get("lost_tokens")) for b in cut_breaks if isinstance(b, dict))),
        }
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def find_latest(project=None):
    base = os.path.join(os.path.expanduser("~"), ".claude", "projects")
    try:
        all_dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    except OSError:
        all_dirs = []
    if project:
        # Exact slug, else substring match. Project slugs start with "-" (e.g.
        # "-home-user-projects-example"), which argparse rejects as a flag value in the
        # bare "--project <slug>" form; a substring lets you pass just "mythical".
        if project in all_dirs:
            dirs = [os.path.join(base, project)]
        else:
            dirs = [os.path.join(base, d) for d in all_dirs if project in d]
    else:
        dirs = [os.path.join(base, d) for d in all_dirs]
    newest, newest_mt = None, -1.0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if not n.endswith(".jsonl"):
                continue
            p = os.path.join(d, n)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt > newest_mt:
                newest, newest_mt = p, mt
    return newest


def _print_human(res, explain=False):
    win = res["window"] or 0
    print("context-quality: {}/100  grade {} ({})   [model={} window={:,}]".format(
        res["score"], res["grade"], res["band"], res["model"], win))
    for name in ("occupancy", "degradation", "cache_efficiency"):
        s = res["signals"][name]
        print("  - {:16s} {:5.1f}  (weight {:.2f})".format(name, s["score"], s["weight"]))
    if explain:
        print("  raw inputs:")
        for name in ("occupancy", "degradation", "cache_efficiency"):
            print("    {}: {}".format(name, json.dumps(res["signals"][name]["raw"])))
        print("  anchors: occupancy curve = MRCR-shaped (heuristic y-values); "
              "degradation sub-weights + cache anchors = heuristic, tune after calibration.")


def _fmt_bytes(n):
    """Compact human size next to the exact byte count (both, never only the rounded)."""
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or unit == "GiB":
            return "{:.0f}{}".format(v, unit) if unit == "B" else "{:.1f}{}".format(v, unit)
        v /= 1024.0
    return "{:.1f}GiB".format(v)


def _print_compose(res, path):
    print("compose: {}".format(path))
    comp = res["compaction"]
    tail = "  [PARTIAL TAIL]" if res["partial_tail"] else ""
    print("  records: {} loaded, {} attributed ({} excluded at/before the last of {} "
          "compact_boundary){}".format(res["records"], res["records_attributed"],
                                       comp["excluded_records"], comp["boundaries"], tail))
    ovh = res["transcript_overhead"]
    print("  attributed: {:,} bytes ({}) — every share below is of THIS".format(
        res["total_bytes"], _fmt_bytes(res["total_bytes"])))
    # NOT "file total": this is the sum over the records actually
    # MEASURED -- post-compaction, and tail-only when partial. On a compacted or
    # truncated transcript it is materially smaller than the file.
    print("  measured records: {:,} bytes = attributed + {:,} transcript overhead "
          "(envelope + result mirror; never in the window)".format(
              res["record_bytes_measured"], ovh["bytes"]))
    print("")
    print("  {:<34s} {:>14s} {:>10s}  {:>9s}".format("category / bucket", "bytes", "human", "share"))
    for cat in res["categories"]:
        print("  {:<34s} {:>14,d} {:>10s}  {:>8.1f}%".format(
            cat["category"], cat["bytes"], _fmt_bytes(cat["bytes"]), cat["share"] * 100.0))
        for b in cat["buckets"]:
            print("    {:<32s} {:>14,d} {:>10s}  {:>8.1f}%".format(
                b["name"][:32], b["bytes"], _fmt_bytes(b["bytes"]), b["share"] * 100.0))
    print("  {:<34s} {:>14,d} {:>10s}  {:>9s}".format(
        "(transcript overhead)", ovh["bytes"], _fmt_bytes(ovh["bytes"]), "excluded"))
    for b in ovh["buckets"]:
        print("    {:<32s} {:>14,d} {:>10s}  {:>9s}".format(
            b["name"][:32], b["bytes"], _fmt_bytes(b["bytes"]), "-"))
    if ovh["bytes"]:
        # The overhead caveat is RUNTIME output, not just a source comment:
        # what this bucket excludes from the shares rests on a structural heuristic,
        # and a reader of the table has to be able to see that.
        print("    ^ {}".format(ovh["note"]))
    print("")
    print("  final turn effective_input: {:,} tokens (the occupancy numerator)".format(
        res["effective_input"]))
    print("  ~unattributed overhead:     {:,} bytes  <- {}".format(
        res["approx_unattributed_overhead_bytes"], res["approx_basis"]))
    for n in res["notes"]:
        print("  note: {}".format(n))


# Per-turn rows printed by `cache` on a HUMAN run. The JSON output carries the
# whole series; the terminal gets the recent end plus an explicit count of what is
# not shown, because a 500-turn table scrolls the useful part off the screen.
CACHE_CLI_SERIES_ROWS = 20


def _fmt_epoch(sec):
    """An epoch second as a UTC ISO stamp, or '-' when the turn carried no
    parseable timestamp. Display only — the payload keeps the number."""
    if sec is None:
        return "-"
    try:
        return datetime.fromtimestamp(int(sec), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OSError, OverflowError):
        return "-"


def _print_cache(res, path):
    print("cache: {}".format(path))
    tail = "  [PARTIAL TAIL]" if res["partial_tail"] else ""
    t = res["totals"]
    print("  turns measured: {}{}".format(res["turns"], tail))
    print("  read {:,} + creation {:,} + uncached {:,} = effective {:,} tokens".format(
        t["cache_read_tokens"], t["cache_creation_tokens"],
        t["uncached_input_tokens"], t["effective_input_tokens"]))
    # The identity worth stating on the surface itself: on a whole-file read this IS
    # the number the score's cache signal reads, over the same turns and the same
    # denominator. On a partial tail it is the same computation over a possibly
    # smaller span, so the claim is qualified rather than dropped.
    print("  read share: {:.1%} of effective input ({})".format(
        t["read_share"],
        "tail-local -- may differ from the score's cache ratio" if res["partial_tail"]
        else "the cache-efficiency signal's own ratio"))
    if res["warmup_turns"] is None:
        print("  warm-up: steady reuse (>= {:.0%} read share) never reached in these turns".format(
            CACHE_WARM_READ_SHARE))
    else:
        print("  warm-up: {} turn(s) before steady reuse (>= {:.0%} read share)".format(
            res["warmup_turns"], CACHE_WARM_READ_SHARE))
    print("")
    print("  breaks: {}  — re-warm {:,} tokens, unreused {:,} tokens".format(
        res["break_count"], res["rewarm_tokens"], res["lost_tokens"]))
    if res["breaks"]:
        print("  {:>5s} {:>21s} {:>10s}  {:<16s} {:>12s} {:>12s}".format(
            "turn", "at (UTC)", "idle gap", "class", "re-warm", "unreused"))
        for b in res["breaks"]:
            print("  {:>5d} {:>21s} {:>10s}  {:<16s} {:>12,d} {:>12,d}".format(
                b["turn"], _fmt_epoch(b["at"]),
                ("-" if b["gap_sec"] is None else "{:.0f}s".format(b["gap_sec"])),
                b["class"], b["rewarm_tokens"], b["lost_tokens"]))
    print("  ^ {}".format(res["break_basis"]))
    print("  ^ {}".format(res["class_basis"]))
    print("")
    shown = res["series"][-CACHE_CLI_SERIES_ROWS:] if CACHE_CLI_SERIES_ROWS else []
    hidden = len(res["series"]) - len(shown)
    print("  per-turn series — showing the last {} of {}{}".format(
        len(shown), len(res["series"]),
        (" ({} earlier turn(s) not shown; use --json for all)".format(hidden) if hidden else "")))
    print("  {:>5s} {:>21s} {:>12s} {:>12s} {:>12s}  {}".format(
        "turn", "at (UTC)", "read", "creation", "uncached", "break"))
    for row in shown:
        print("  {:>5d} {:>21s} {:>12,d} {:>12,d} {:>12,d}  {}".format(
            row["turn"], _fmt_epoch(row["at"]), row["cache_read_tokens"],
            row["cache_creation_tokens"], row["uncached_input_tokens"],
            "BREAK" if row.get("break") else ""))
    for n in res["notes"]:
        print("  note: {}".format(n))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="context_quality.py",
        description="Context-health / quality score for a Claude Code session transcript.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # `compose` and `cache` share one surface on purpose: same positional, same
    # `latest` sugar, same --project, same --json. They are the two detail views
    # behind two of the three score signals, and a caller who learned one has
    # learned the other.
    for name, helptext in (("compose", "what FILLS the window, by category"),
                           ("cache", "how prompt-cache reuse behaved, turn by turn")):
        pc = sub.add_parser(name, help=helptext)
        pc.add_argument("path", help="path to a session .jsonl, or the word 'latest'")
        pc.add_argument("--project", default=None,
                        help="with path='latest': project slug OR substring under ~/.claude/projects")
        pc.add_argument("--json", action="store_true", help="machine-readable output")
    for name in ("score", "latest"):
        p = sub.add_parser(name)
        if name == "score":
            p.add_argument("path", help="path to a session .jsonl")
        else:
            p.add_argument("--project", default=None, help="project slug OR substring (e.g. 'mythical') under ~/.claude/projects")
        p.add_argument("--json", action="store_true", help="machine-readable output")
        p.add_argument("--explain", action="store_true", help="print raw signal inputs")
        p.add_argument("--window", type=int, default=None, help="override context window (tokens)")
    args = ap.parse_args(argv)

    # `compose latest` / `cache latest` resolve the newest transcript, mirroring
    # the `latest` subcommand -- but a REAL file named "latest" still wins, so the
    # sugar can never shadow an actual path the caller passed.
    resolve_latest = (args.cmd == "latest"
                      or (args.cmd in ("compose", "cache") and args.path == "latest"
                          and not os.path.isfile(args.path)))
    if resolve_latest:
        path = find_latest(args.project)
        if not path:
            print("no transcripts found under ~/.claude/projects", file=sys.stderr)
            return 0
        print("# latest: {}".format(path), file=sys.stderr)
    else:
        path = args.path

    if not os.path.exists(path):
        ap.error("file not found: {}".format(path))  # bad invocation -> exit 2

    if args.cmd in ("compose", "cache"):
        # ONE loader for both detail views (it is named for its first caller): one
        # oversize threshold and one partial-flag rule, so the two surfaces can
        # never disagree about whether a transcript was read whole.
        recs, partial = load_for_compose(path)
        if args.cmd == "compose":
            res = compose_session(recs, partial=partial)
        else:
            res = cache_session(recs, partial=partial)
        if args.json:
            print(json.dumps(res, indent=2))
        elif args.cmd == "compose":
            _print_compose(res, path)
        else:
            _print_cache(res, path)
        return 0

    res = score_session(read_jsonl(path), window=args.window)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        _print_human(res, explain=args.explain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
