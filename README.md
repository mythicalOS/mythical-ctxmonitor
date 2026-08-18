# mythical-ctxmonitor

Context-quality monitoring for coding-agent sessions: a transcript scorer, a terminal
statusline, and an in-context early-warning signal. When a session's context window fills up
and its quality frays, the agent itself is the last to know — this tool measures it and says
so, both to you (statusline) and to the agent (a `UserPromptSubmit` nudge injected before
quality drops off the cliff).

Part of the [mythicalOS](https://github.com/mythicalOS) family; used standalone or as the
vendored context engine of the brokkr agent container.

## What it does

- **`bin/context_quality.py`** — the scorer. Stdlib-only Python 3, read-only: it parses a
  Claude Code session transcript (JSONL) and produces a 0–100 quality score from three
  weighted signals — **occupancy** (how full the window is, against the model's real window
  size), **degradation** (stale re-reads, duplicated output, error loops), and **cache
  efficiency**. Also a CLI: `score`, `compose` (what fills the window, by category), `cache`
  (cache-break analysis).
- **`bin/statusline-command.sh`** — a Claude Code statusline that renders the current
  session's grade inline (`CtxMonitor:A(92)`), cached and throttled so it costs nothing per
  keystroke.
- **`bin/context-signal.sh`** — a `UserPromptSubmit` hook. On every prompt it re-scores the
  transcript and, when quality is degraded *and* the window is meaningfully full, injects a
  short `[ctxmonitor]` note telling the agent to wrap up or hand off. Fail-safe by contract:
  any error exits 0 and never blocks a prompt.

## Install

Prerequisites: `python3` and `jq` on `PATH`.

Program files live in **`~/.ctxmonitor/`** — a harness-neutral, tool-owned home (installing
this tool never creates another tool's dotfiles). The score cache lives beside them in
`~/.ctxmonitor/cache/`; the `CTX_MONITOR_DIR` environment variable (absolute path) overrides
the cache location per session.

```sh
# one-liner (verifies the release tarball's sha256 before running anything):
curl -fsSL https://get.mythicalos.ai/ctxmonitor | bash

# or from a release tarball you fetched yourself (sha256 published beside every release):
tar -xzf mythical-ctxmonitor-<version>.tar.gz
cd mythical-ctxmonitor && ./install.sh
```

The installer registers the hook in `~/.claude/settings.json` (default-on) and sets the
statusline **only if you have none** — an existing statusline is never touched; the installer
prints a manual wrapping recipe instead. Preview before writing with `--dry-run` (it prints
the full settings diff and writes nothing).

### Uninstall / status

The installer ships a copy of itself into the install home, so these are self-contained —
no re-download:

```sh
~/.ctxmonitor/install.sh uninstall   # surgically remove our entries, then ~/.ctxmonitor
~/.ctxmonitor/install.sh status      # what's installed, registration state, a scorer self-test
```

The one-liner works too (it fetches the release to run):
`curl -fsSL https://get.mythicalos.ai/ctxmonitor | bash -s -- uninstall`.

Uninstall removes exactly the entries the installer added — foreign hooks and any existing
statusline are untouched. If you manually wrapped an existing statusline to delegate to
`~/.ctxmonitor/bin/statusline-command.sh`, revert that delegate line before uninstalling so
it doesn't point at a removed path.

## Harness support

v1 wires up **Claude Code** (hook + statusline contracts and the transcript format it
parses). The payload home is harness-neutral by design so further harness adapters can share
one installed payload; see `docs/adapter-seam.md` for where the seams are.

## Scoring model

The score is grade-first: **A+ sessions are never nagged**, whatever their occupancy. The
signal fires only when the score is under 80 *and* occupancy is at or past 65% — measured
knees, biased early because an agent needs headroom to *finish* work, not just to start it.
The calibration provenance is documented in the scorer's own header comments; the golden
fixture under `bin/testdata/` pins the arithmetic cross-language (see
`docs/brokkr-integration.md`).

## Brokkr integration

The brokkr agent container vendors these files byte-exact and builds its permission-floor
wiring, daemon-side cache readers, and telemetry on top of them. The vendor contract — and
the one deliberate brokkr-ism in the scorer (an inert-when-standalone attribution marker for
Control-Room-delivered prompts) — is documented in `docs/brokkr-integration.md`.

## License

Apache-2.0.
