# AGENTS.md — mythical-ctxmonitor

Context-quality monitor for coding-agent sessions: a stdlib-only Python scorer
(`bin/context_quality.py`), a statusline renderer and a `UserPromptSubmit` signal hook (both
bash), their shared golden fixture, and an installer. No build step; no runtime dependencies
beyond `python3` and `jq`.

## Authority & precedence

Repository orientation, not a role contract. If a role, playbook, or system prompt governs
your session, that contract is authoritative and supersedes anything here. This file grants
no edit, run, commit, push, or release permission.

## Commands

Run only if your active role permits command execution.

- Tests: `python3 -m pytest tests/` (or `python3 -m unittest discover -s tests`)
- Shell syntax: `bash -n bin/*.sh install.sh`
- Score the fixture: `python3 bin/context_quality.py score bin/testdata/golden-session.jsonl --json`

## Boundaries & gotchas

- **`bin/testdata/golden-session.expected.json` is a hand-computed cross-language oracle**,
  asserted here and by the brokkr integration's TypeScript suite. Never regenerate it from
  program output — a failing assertion means one implementation moved; decide which side is
  right, fix both, then restate the expectation (its own `_comment` carries the rule).
- **This repo is vendored byte-exact (bytes AND file modes) by brokkr.** Land and push here
  first; the consumer re-vendors against a pinned release. Keep the three scripts executable
  (755) — the executable bit is part of the vendor contract.
- **The operator-delivery marker in the scorer is a pinned copy** of a brokkr constant
  (inert standalone; see `docs/brokkr-integration.md`). Do not edit it here — the change
  starts at brokkr's TypeScript source of truth.
- CI runs a `docs-bar` content gate on this repository — keep all content and commit
  messages free of internal project vocabulary.
- Fixture files must stay ASCII and NUL-free (tested) — a NUL byte hides a file from
  common grep configurations silently.
