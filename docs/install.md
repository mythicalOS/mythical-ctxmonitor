# Install, upgrade, and uninstall

The reference detail behind the [README](../README.md) install section. Program files live in
`~/.ctxmonitor/` — a harness-neutral, tool-owned home (installing this tool never creates another
tool's dotfiles). The score cache lives in `~/.ctxmonitor/cache/`; `CTX_MONITOR_DIR` (absolute path)
overrides the cache location per session.

## Installing

```sh
# one-liner (verifies the release tarball's sha256 before running anything):
curl -fsSL https://get.mythicalos.ai/ctxmonitor | bash

# or from a release tarball you fetched yourself (sha256 published beside every release):
tar -xzf mythical-ctxmonitor-<version>.tar.gz
cd mythical-ctxmonitor && ./install.sh
```

The installer registers the hook in `~/.claude/settings.json` (default-on) and sets the statusline
**only if you have none** — an existing statusline is never touched; the installer prints a manual
wrapping recipe instead. Preview before writing with `--dry-run` (it prints the full settings diff and
writes nothing).

## Version-aware upgrades

The installer knows what version is installed and what version it is:

```sh
~/.ctxmonitor/install.sh install --check   # report the version delta; install nothing
~/.ctxmonitor/install.sh install           # upgrade: shows <from> -> <to> and asks to confirm
~/.ctxmonitor/install.sh install --yes      # ...or proceed without prompting
```

- A re-run of the same version reports **on-newest** and re-converges (repairs any drift), no prompt.
- A newer installer shows `v<from> -> v<to>` and asks before changing anything — the prompt reads your
  terminal even under `curl … | bash`. Decline and nothing changes.
- Truly non-interactive runs (no terminal) proceed with a printed notice — the version was pinned
  deliberately by whatever invoked it; pass `--yes` to silence the notice.

## Uninstall and status

The installer ships a copy of itself into the install home, so these are self-contained — no
re-download:

```sh
~/.ctxmonitor/install.sh uninstall   # surgically remove our entries, then ~/.ctxmonitor
~/.ctxmonitor/install.sh status      # version, installed files, registration state, self-test
```

The one-liner works too (it fetches the release to run):

```sh
curl -fsSL https://get.mythicalos.ai/ctxmonitor | bash -s -- uninstall
```

Uninstall removes exactly the entries the installer added — foreign hooks and any existing statusline
are untouched. If you manually wrapped an existing statusline to delegate to
`~/.ctxmonitor/bin/statusline-command.sh`, revert that delegate line before uninstalling so it doesn't
point at a removed path.
