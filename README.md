<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/logo-dark.svg">
    <img src=".github/assets/logo-light.svg" alt="mythicalOS" width="84" height="84">
  </picture>
</p>

<h1 align="center">mythical-ctxmonitor</h1>

<p align="center">
  <strong>Know when your coding agent is running out of context — and make sure the agent knows too.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3-3776AB.svg?logo=python&logoColor=white" alt="Python 3">
  <img src="https://img.shields.io/badge/deps-python3_+_jq-555.svg" alt="Dependencies: python3 + jq">
  <a href="https://mythicalos.ai"><img src="https://img.shields.io/badge/part_of-mythicalOS-0F6B66.svg" alt="Part of mythicalOS"></a>
</p>

---

As a session's context window fills, its answers quietly get worse — and the agent is the last to
notice. **ctxmonitor** measures that and says so out loud: to **you**, on the terminal statusline,
and to the **agent**, as a short nudge injected the moment quality starts to slip.

Runs standalone, or as the vendored context engine inside the brokkr agent container.

## What's in the box

| Tool | What it does |
|------|--------------|
| **Scorer** — `bin/context_quality.py` | Stdlib-only Python 3, read-only. Turns a Claude Code JSONL transcript into a **0–100 quality grade** from three signals: occupancy, degradation (stale re-reads, duplicated output, error loops), and cache efficiency. CLI: `score`, `compose`, `cache`. |
| **Statusline** — `bin/statusline-command.sh` | Renders the current grade inline (`CtxMonitor:A(92)`), cached and throttled so it costs nothing per keystroke. |
| **Signal hook** — `bin/context-signal.sh` | A `UserPromptSubmit` hook that injects a `[ctxmonitor]` note when quality is degraded *and* the window is meaningfully full. Fail-safe: any error exits 0 and never blocks your prompt. |

## Install

Needs `python3` and `jq` on your `PATH`.

```sh
curl -fsSL https://get.mythicalos.ai/ctxmonitor | bash
```

The installer verifies the release tarball's sha256 before running anything, registers the hook in
`~/.claude/settings.json`, and sets the statusline **only if you don't already have one** (an existing
one is never touched). Preview every change first with `--dry-run`. Program files live in
`~/.ctxmonitor/`.

```sh
~/.ctxmonitor/install.sh install --check   # report the version delta, change nothing
~/.ctxmonitor/install.sh install           # upgrade (shows <from> -> <to>, asks to confirm)
~/.ctxmonitor/install.sh status            # version, files, registration, self-test
~/.ctxmonitor/install.sh uninstall         # remove only what we added, then ~/.ctxmonitor
```

Version-aware upgrades, non-interactive behaviour, and the uninstall contract are in
**[docs/install.md](docs/install.md)**.

## How the grade works

Grade-first: **an A+ session is never nagged**, whatever its occupancy. The signal fires only when the
score drops under 80 *and* occupancy is at or past 65% — knees measured from real sessions, biased
early because an agent needs headroom to *finish* work, not just to start it. Calibration provenance
lives in the scorer's header comments; a golden fixture pins the arithmetic cross-language.

## Harness support & integration

v1 wires up **Claude Code** (hook + statusline contracts, and the transcript format it parses). The
install home is harness-neutral so future adapters can share one payload — see
**[docs/adapter-seam.md](docs/adapter-seam.md)**. The brokkr container vendors these files byte-exact
and builds its floor wiring, cache readers, and telemetry on top — see
**[docs/brokkr-integration.md](docs/brokkr-integration.md)**.

## License

**Apache-2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The licence covers the code, not the
mythicalOS name and marks — see [TRADEMARK.md](TRADEMARK.md). Contributions welcome under a DCO
sign-off, no CLA: [CONTRIBUTING.md](CONTRIBUTING.md).
