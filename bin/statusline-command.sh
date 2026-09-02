#!/usr/bin/env bash
# Status line: fully inline single-line layout
#   [Model] | ████░░░░░░ 42% | 📁 dirname | 🌿 branch [git-state] | CTX:S(90) | CacheU:85% | ⚡ 35% | session [| $X.XX] [| Xm Ys]
#
# Features:
#   - ANSI color thresholds (green/yellow/orange/red) on context bar, CTX quality, rate limit
#   - OSC 8 hyperlinks (clickable in iTerm2/Kitty/WezTerm; renders plain text elsewhere)
#       * directory  -> file:///absolute/path
#       * branch     -> git remote URL (HTTPS) when discoverable
#   - Sub-agent indicator when JSON payload contains .agent.name
#   - Rate limit segment from .rate_limits.five_hour.used_percentage (Claude.ai Pro/Max)
#   - Git working-tree state (ahead/behind, modified, untracked, merge/rebase) via porcelain=v2

input=$(cat)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
CYAN="\033[0;36m"
BOLD_GREEN="\033[1;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
ORANGE="\033[38;5;208m"
DIM="\033[2m"
SOFT_GREEN="\033[92m"      # bright green for healthy indicators (replaces former DIM_GREEN)
RESET="\033[0m"
BLINK_RED="\033[5;31m"
BRIGHT_WHITE="\033[97m"    # session ID/name
PURPLE="\033[38;5;141m"    # sub-agent indicator
ORANGE_209="\033[38;5;209m"  # 256-color orange 209 for weekly rate limit

# ---------------------------------------------------------------------------
# Parse stdin JSON once into variables
# ---------------------------------------------------------------------------
model=$(echo "$input"        | jq -r '.model.display_name // "Claude"')
session_name=$(echo "$input" | jq -r '.session_name // empty')
session_id=$(echo "$input"   | jq -r '.session_id // ""')
current_dir=$(echo "$input"  | jq -r '.workspace.current_dir // ""')
used_pct_raw=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
agent_name=$(echo "$input"   | jq -r '.agent.name // empty')
rate_5h=$(echo "$input"      | jq -r '.rate_limits.five_hour.used_percentage // empty')
rate_7d=$(echo "$input"      | jq -r '.rate_limits.seven_day.used_percentage // empty')

# Cost / duration: documented under .cost in current statusLine payload
cost_raw=$(echo "$input"     | jq -r '.cost.total_cost_usd // empty')
duration_ms=$(echo "$input"  | jq -r '.cost.total_duration_ms // empty')

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
cwd=$(pwd)
abs_dir="${current_dir:-$cwd}"

# Use workspace.current_dir basename for display if available, else pwd basename
if [ -n "$current_dir" ]; then
  dirname="${current_dir##*/}"
else
  dirname="${cwd##*/}"
fi

# Session label: prefer human-readable name, fall back to short ID
if [ -n "$session_name" ]; then
  session_label="$session_name"
else
  session_label="${session_id:0:8}"
fi

# ---------------------------------------------------------------------------
# OSC 8 hyperlink helper
#   Format per official docs: \e]8;;URL\a TEXT \e]8;;\a
#   Terminals without OSC 8 support strip the escapes and show TEXT plainly.
# ---------------------------------------------------------------------------
osc8() {
  local url="$1" text="$2"
  printf '\033]8;;%s\a%s\033]8;;\a' "$url" "$text"
}

# Directory display: full path with ~ substitution so iTerm2 Semantic History
# (Cmd+click) can resolve it. OSC 8 hyperlinks are avoided because file:// URL
# handling is system-dependent; plain-text paths work via NSWorkspace.openFile.
dir_display="${abs_dir/#$HOME/~}"

# ---------------------------------------------------------------------------
# Git: branch + working-tree state via single porcelain=v2 call
#   Counts: ahead/behind, modified (staged+unstaged), untracked, merge/rebase
# ---------------------------------------------------------------------------
git_segment=""
if git -C "$cwd" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  porc=$(git --no-optional-locks -C "$cwd" status --porcelain=v2 --branch 2>/dev/null)
  git_dir=$(git -C "$cwd" rev-parse --git-dir 2>/dev/null)

  # Branch from porcelain header (# branch.head <name>)
  branch=$(printf '%s\n' "$porc" | awk '/^# branch.head /{print $3; exit}')
  # Ahead/behind from "# branch.ab +A -B"
  read -r ahead behind <<<"$(printf '%s\n' "$porc" \
    | awk '/^# branch.ab /{sub(/^\+/,"",$3); sub(/^-/,"",$4); print $3, $4; exit}')"
  ahead=${ahead:-0}; behind=${behind:-0}

  # Tracked changes: lines starting with 1, 2 (staged/unstaged), u (unmerged)
  tracked=$(printf '%s\n' "$porc" | awk '/^[12u] /{n++} END{print n+0}')
  untracked=$(printf '%s\n' "$porc" | awk '/^\? /{n++} END{print n+0}')
  unmerged=$(printf '%s\n' "$porc" | awk '/^u /{n++} END{print n+0}')

  # In-progress merge/rebase detection
  in_progress=""
  if [ -n "$git_dir" ]; then
    if   [ -d "$git_dir/rebase-merge" ] || [ -d "$git_dir/rebase-apply" ]; then in_progress="rebase"
    elif [ -f "$git_dir/MERGE_HEAD" ]; then                                     in_progress="merge"
    elif [ -f "$git_dir/CHERRY_PICK_HEAD" ]; then                               in_progress="cherry-pick"
    fi
  fi

  if [ -n "$branch" ] || [ "$branch" = "(detached)" ]; then
    # Build the state suffix
    state=""
    [ "$ahead" -gt 0 ]    && state="${state} ↑${ahead}"
    [ "$behind" -gt 0 ]   && state="${state} ↓${behind}"
    [ "$tracked" -gt 0 ]  && state="${state} *${tracked}"
    [ "$untracked" -gt 0 ] && state="${state} ?${untracked}"
    [ -n "$in_progress" ] && state="${state} (${in_progress})"

    # Color: clean=green, dirty/diverged=yellow, in-progress/unmerged=red
    if   [ -n "$in_progress" ] || [ "$unmerged" -gt 0 ]; then git_color="$RED"
    elif [ -n "$state" ];                                then git_color="$YELLOW"
    else                                                       git_color="$BOLD_GREEN"
    fi

    # Branch hyperlink: try to derive HTTPS URL from origin remote
    remote_url=$(git -C "$cwd" config --get remote.origin.url 2>/dev/null)
    if [ -n "$remote_url" ]; then
      # Normalise SSH -> HTTPS (github.com, gitlab.com, bitbucket.org, generic)
      https_url=$(printf '%s' "$remote_url" \
        | sed -E 's#^git@([^:]+):#https://\1/#; s#\.git$##')
      branch_text=$(osc8 "$https_url" "$branch")
    else
      branch_text="$branch"
    fi

    git_segment="${git_color}🌿 ${branch_text}${state}${RESET}"
  fi
fi

# ---------------------------------------------------------------------------
# Context bar (10-char, thresholds: <50% bold-green, 50-69% yellow, 70-89% orange, >=90% blink-red)
# ---------------------------------------------------------------------------
build_bar() {
  local pct="$1"
  local filled=$(echo "$pct" | awk '{printf "%d", ($1/100)*10}')
  local empty=$((10 - filled))
  local bar=""
  local i
  for i in $(seq 1 $filled); do bar="${bar}█"; done
  for i in $(seq 1 $empty);  do bar="${bar}░"; done
  echo "$bar"
}

if [ -n "$used_pct_raw" ]; then
  pct_int=$(echo "$used_pct_raw" | awk '{printf "%d", $1}')
  bar=$(build_bar "$pct_int")

  if   [ "$pct_int" -ge 90 ]; then bar_color="$BLINK_RED"
  elif [ "$pct_int" -ge 70 ]; then bar_color="$ORANGE"
  elif [ "$pct_int" -ge 50 ]; then bar_color="$YELLOW"
  else                              bar_color="$BOLD_GREEN"
  fi

  ctx_colored="${bar_color}${bar} ${pct_int}%${RESET}"
else
  ctx_colored="${DIM}ctx:--${RESET}"
fi

# Cost segment
cost_colored=""
if [ -n "$cost_raw" ]; then
  # awk uses C-style number parsing regardless of LC_NUMERIC; bash printf does not.
  cost_fmt=$(echo "$cost_raw" | awk '{printf "$%.2f", $1}')
  cost_colored=" | ${YELLOW}${cost_fmt}${RESET}"
fi

# Duration segment
dur_colored=""
if [ -n "$duration_ms" ] && [ "$duration_ms" != "0" ]; then
  mins=$(echo "$duration_ms" | awk '{printf "%d", $1/60000}')
  secs=$(echo "$duration_ms" | awk '{printf "%d", ($1%60000)/1000}')
  dur_colored=" | ${CYAN}${mins}m ${secs}s${RESET}"
fi

# Rate-limit segment (Claude.ai Pro/Max only — both windows may be absent)
rate_colored=""
if [ -n "$rate_5h" ]; then
  rate_int=$(echo "$rate_5h" | awk '{printf "%d", $1}')
  if   [ "$rate_int" -ge 90 ]; then rate_color="$RED"
  elif [ "$rate_int" -ge 70 ]; then rate_color="$YELLOW"
  elif [ "$rate_int" -ge 50 ]; then rate_color="$ORANGE"
  else                               rate_color="$SOFT_GREEN"
  fi
  rate_colored=" | ${rate_color}⚡ ${rate_int}%${RESET}"
  # If 7-day window is also present, append it compactly
  if [ -n "$rate_7d" ]; then
    rate_7d_int=$(echo "$rate_7d" | awk '{printf "%d", $1}')
    rate_colored="${rate_colored} ${ORANGE_209}${rate_7d_int}%w${RESET}"
  fi
fi

# Sub-agent indicator (.agent.name appears only with --agent flag/settings)
agent_colored=""
if [ -n "$agent_name" ]; then
  agent_colored="${PURPLE}🤖 ${agent_name}${RESET} ${DIM}|${RESET} "
fi

# ---------------------------------------------------------------------------
# Session id (sanitized) -- used by the CtxMonitor cache lookup below.
# ---------------------------------------------------------------------------
safe_sid=$(echo "$session_id" | tr -cd 'a-zA-Z0-9_-')

# ---------------------------------------------------------------------------
# Assemble all segments
# Order: [agent] [Model] | ctx | dir | branch+state | CTX | CacheU | rate | session | [cost] | [duration]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CtxMonitor: first-party context-quality (context_quality.py), cached + throttled.
# ---------------------------------------------------------------------------
ctxmon_colored=""
CQ_BIN="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd -P)/context_quality.py"
if [ -n "$safe_sid" ] && [ -f "$CQ_BIN" ]; then
  ctxmon_dir="$HOME/.ctxmonitor/cache"   # CtxMonitor-owned; independent of token-optimizer
  mkdir -p "$ctxmon_dir" 2>/dev/null
  ctxmon_cache="${ctxmon_dir}/ctxmonitor-${safe_sid}.json"
  cm_recompute=1
  if [ -f "$ctxmon_cache" ]; then
    cm_mtime=$(stat -c %Y "$ctxmon_cache" 2>/dev/null || stat -f %m "$ctxmon_cache" 2>/dev/null || echo 0)
    cm_age=$(( $(date +%s) - ${cm_mtime:-0} ))
    [ "$cm_age" -lt 20 ] && cm_recompute=0   # throttle: <20s old -> reuse (no transcript parse)
  fi
  if [ "$cm_recompute" = 1 ]; then
    cm_tx=""
    for _f in "$HOME/.claude/projects"/*/"${session_id}.jsonl"; do
      [ -f "$_f" ] && cm_tx="$_f" && break
    done
    if [ -n "$cm_tx" ]; then
      cm_out=$(python3 "$CQ_BIN" score "$cm_tx" --json 2>/dev/null)
      [ -n "$cm_out" ] && printf %s "$cm_out" > "${ctxmon_cache}.tmp" 2>/dev/null \
        && mv "${ctxmon_cache}.tmp" "$ctxmon_cache" 2>/dev/null || true
    fi
  fi
  if [ -f "$ctxmon_cache" ]; then
    cm_score=$(jq -r '.score // empty' "$ctxmon_cache" 2>/dev/null)
    cm_grade=$(jq -r '.grade // empty' "$ctxmon_cache" 2>/dev/null)
    if [ -n "$cm_score" ]; then
      cm_int=$(echo "$cm_score" | awk '{printf "%d", $1}')
      if   [ "$cm_int" -ge 85 ]; then cm_color="$SOFT_GREEN"
      elif [ "$cm_int" -ge 75 ]; then cm_color="$YELLOW"
      elif [ "$cm_int" -ge 50 ]; then cm_color="$ORANGE"
      else                            cm_color="$RED"; fi
      ctxmon_colored=" | ${cm_color}CTX:${cm_grade}(${cm_int})${RESET}"
    fi
  fi
  [ -z "$ctxmon_colored" ] && ctxmon_colored=" | ${DIM}CTX:--${RESET}"
fi

# ---------------------------------------------------------------------------
# CacheU: cache-reuse % (cache_read / effective) from the same CtxMonitor cache.
# ---------------------------------------------------------------------------
cacheu_colored=""
if [ -n "${ctxmon_cache:-}" ] && [ -f "$ctxmon_cache" ]; then
  cu_ratio=$(jq -r '.signals.cache_efficiency.raw.ratio // empty' "$ctxmon_cache" 2>/dev/null)
  if [ -n "$cu_ratio" ] && [ "$cu_ratio" != "null" ]; then
    cu_pct=$(echo "$cu_ratio" | awk '{printf "%d", $1*100}')
    if   [ "$cu_pct" -ge 80 ]; then cu_color="$SOFT_GREEN"
    elif [ "$cu_pct" -ge 50 ]; then cu_color="$YELLOW"
    else                            cu_color="$ORANGE"; fi
    cacheu_colored=" | ${cu_color}CacheU:${cu_pct}%${RESET}"
  fi
fi

SEP="${DIM} | ${RESET}"

line="${agent_colored}${CYAN}[${model}]${RESET}"
line="${line}${SEP}${ctx_colored}"
line="${line}${SEP}📁 ${dir_display}"

if [ -n "$git_segment" ]; then
  line="${line}${SEP}${git_segment}"
fi

line="${line}${ctxmon_colored}${cacheu_colored}${rate_colored}"

if [ -n "$session_label" ]; then
  line="${line}${SEP}${BRIGHT_WHITE}${session_label}${RESET}"
fi

line="${line}${cost_colored}${dur_colored}"

# %b interprets ANSI + OSC 8 escapes reliably across shells (per official docs)
printf '%b\n' "$line"
