---
quick_id: 261028-th2
slug: redesign-guardrails-dashboard-ui
status: complete
date: 2026-10-28
commit: 3881dfb
files:
  - src/dashboard/templates/index.html
---

# Summary: guardrails dashboard remake

`src/dashboard/templates/index.html` is rebuilt as a reference-monitor console. `app.py` and every
JSON route are unchanged.

## What changed

- **Decisions ledger:** shows the total of mediated actions, one stacked ALLOW/FLAG/BLOCK bar, and
  a count and share for each decision. Each decision has a colour, an icon and a word.
- **Where actions were stopped:** the path runs agent → tool broker → policy engine → namespaces →
  seccomp-BPF → host kernel. Each stage shows its block count and its share of all blocks.
  Blocks are counted client-side from `/api/logs`, because `stats.layers` also counts ALLOW rows.
  A layer name the page does not know gets its own stage. The kernel stages say "Needs Linux or
  WSL2" when `/api/info` reports a non-Linux host.
- **Event stream:** filters by decision (with counts) and by a text search over target, reason,
  action and layer. Rows expand on click, Enter or Space; an expanded cell is capped at 14em and
  scrolls. Newly arrived rows get a short highlight, and a screen-reader announcement counts the
  new decisions and blocks. The table renders at most 400 rows. Below 720px, rows reflow into
  stacked records.
- **States:** skeleton rows while loading; an empty state that lists the three commands that
  populate the log; a "no events match" state; and "Disconnected, retrying" in the live indicator.
- **Clear:** takes two steps (Clear log, then Confirm clear within 4 s). The button is disabled
  while the log is empty.
- **Theme:** light and dark tokens follow `prefers-color-scheme`, with a toggle remembered in
  localStorage. Motion runs only under `prefers-reduced-motion: no-preference`.
- **Dependencies:** IBM Plex Sans/Mono from `@fontsource` 5.2.5 and Phosphor icons from
  `@phosphor-icons/web` 2.1.2, both from jsDelivr with exact pinned versions.
- **Test hooks kept:** `<tbody id="tbody"></tbody>`, `id="btn-clear"` and `id="empty"`.

## Not possible with the current API

Per-model stats: `/api/stats` and the log entries carry no model field. The header shows the
active model from `/api/info`. A per-model view would need a `model` field written by
`PolicyEngine._log_decision` and an aggregation in `/api/stats`.

## Verification

- In WSL, `pytest tests/test_dashboard_clear.py tests/test_dashboard_bind.py tests/test_review_fix_dashboard.py`
  gave 83 passed.
- Checked in the browser at 1440×900 (dark and light) and at 375×812 against three logs: the real
  log (45 entries), an empty log, and a 52-entry demo log with sandbox, seccomp, tool-broker and
  FLAG rows. The demo and empty logs were served from scratch copies. There was no horizontal
  scroll at 375px, no console errors, and the two-step clear worked on the demo instance.
- No em-dashes or en-dashes in the template.
