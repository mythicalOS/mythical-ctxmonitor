# Where the harness seams are

A design note for anyone (including a future us) adding support for a second coding-agent
harness. v1 ships Claude Code support only; this records what is generic, what is
Claude-Code-shaped, and where the cut lines run — so an adapter is added by construction, not
by archaeology.

## Three layers

1. **The scoring core** — `context_quality.py`'s signal functions (`occupancy_signal`,
   `degradation_signal`, `cache_efficiency_signal`, `score_session`, `band_of`,
   `should_emit`) plus the grade/band vocabulary. Harness-generic in behavior: they consume a
   parsed record list and a model→window map. What is *not* generic is the record **shape**
   they expect (see layer 2) — the functions and the parser are in one file today because a
   single stdlib-only file is the whole point of the payload.

2. **The transcript adapter** — everything that knows what a session log looks like:
   `read_jsonl` / `read_jsonl_tail`, the record accessors (`_message`, `_usage`, `_model`),
   the scoreable-turn predicate, `detect_window`'s model-name heuristics, and `find_latest`'s
   knowledge of where Claude Code keeps transcripts. This is the layer a Codex/opencode/pi
   adapter replaces: parse *that* harness's log into records the core understands (or map its
   fields into the same accessor contract), and supply a window map for its models.

3. **The delivery plugin** — how score output reaches eyes: `statusline-command.sh` speaks
   Claude Code's statusline stdin contract; `context-signal.sh` speaks its `UserPromptSubmit`
   hook contract (`hookSpecificOutput.additionalContext`). Each harness gets its own thin
   delivery scripts; the cache directory (`~/.ctxmonitor/cache/`, `CTX_MONITOR_DIR` to
   override) is the shared interchange point — score once, render anywhere.

## What stays shared

- The **cache file format** (`ctxmonitor-<session>.json`, the full score JSON) and its home.
- The **golden oracle** (`bin/testdata/`) for the core arithmetic.
- The **install home** `~/.ctxmonitor/` — one payload, per-harness wiring in each harness's
  own config. The installer writes Claude Code's wiring today; a second harness adds its own
  wiring step without moving any file.

## One deliberate non-goal

The scorer does not abstract the record shape behind a plugin interface today. Two harnesses
is the point at which that abstraction earns its complexity; until then the adapter seam is
documented here rather than engineered ahead of need.
