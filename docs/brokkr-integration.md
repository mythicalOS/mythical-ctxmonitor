# The brokkr vendor contract, stated from the package side

[brokkr](https://github.com/mythicalOS/mythical-brokkr) consumes this package as a
**byte-exact, mode-preserving vendored copy** of five files, at the same relative paths they
have here:

| File | Mode |
|---|---|
| `bin/context_quality.py` | 755 |
| `bin/statusline-command.sh` | 755 |
| `bin/context-signal.sh` | 755 |
| `bin/testdata/golden-session.jsonl` | 644 |
| `bin/testdata/golden-session.expected.json` | 644 |

**This repository is canonical.** Brokkr's copy is tool-managed: edits land here first and
are re-vendored; brokkr CI verifies its copy against the pinned release's per-file manifest
(sha256 **and** mode — the executable bit is load-bearing for its sealed permission floors)
and refuses drift in either direction.

Every release publishes two artifacts, for two different purposes:

1. **the per-file manifest** (`path → sha256 + mode` for the five files) — what the brokkr
   vendor gate pins;
2. **the release tarball's own sha256** — what the install bootstrap verifies before running
   anything (a per-file manifest cannot authenticate an archive).

## The golden fixture is a cross-language oracle

`bin/testdata/golden-session.expected.json` is **hand-computed** (see its own provenance
comment) and asserted by two independent suites: this repo's Python tests and brokkr's
TypeScript tests against its `session-metrics` mirror. It is *not* the CLI's output — never
regenerate it from what the code prints today. A failure means one language moved; decide
which side is right, fix both, then restate the expectation. Its `known_divergences` section
records the places the two implementations deliberately disagree; each suite asserts its own
language's half.

## The operator-delivery marker

`bin/context_quality.py` carries a literal copy of brokkr's operator-delivery marker (the
framing brokkr's daemon writes on prompts a human sends through its Control Room), used by
the `compose` view to attribute those turns to human prompts. Standalone, the feature is
**inert**: nothing outside a brokkr container writes that framing, and the position-exact
prefix match never fires on ordinary transcripts.

The marker's source of truth is brokkr's TypeScript constant, and the cross-repo byte pin is
**owned by brokkr CI**: brokkr's test suite asserts the marker bytes in its vendored copy of
the scorer on every run and at every vendor bump. This package's tests assert only the
marker's *behavior* (prefix construction, position-exact matching) against a package-local
literal — CI here never checks out brokkr.

Consequence, stated honestly: between a change to brokkr's constant and the next vendor bump,
a release of this package can carry the previous marker. For standalone users that window is
meaningless (their transcripts never contain either form); for brokkr it is caught
deterministically at bump time by its own CI.
