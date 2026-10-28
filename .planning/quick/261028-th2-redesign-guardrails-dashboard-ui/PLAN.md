---
quick_id: 261028-th2
slug: redesign-guardrails-dashboard-ui
date: 2026-10-28
mode: quick (inline, no subagents)
files:
  - src/dashboard/templates/index.html
---

# Quick task: remake the real-time guardrails dashboard UI

Rebuild `src/dashboard/templates/index.html` as a reference-monitor console for the
OS-security review panel. The Flask routes and their JSON stay unchanged; `app.py` is not touched
(it carries the user's uncommitted work-in-progress hunks).

## Audit of the previous UI

- Generic dark SaaS template: four equal glowing metric cards, a purple logo tile, and Fira Sans.
- The "defense layer" column read "Policy Engine" on every row, and `stats.layers` was never shown.
  The panel could not see *where* a block happened, which is the point of defense in depth.
- The reason column wrapped into tall rows. The model name appeared only in the footer. The title
  used an em-dash.

## Direction

The design read: a real-time security-observability console for an OS-security review panel,
with a precise, instrument-like (audit-log) language, built with native CSS and IBM Plex Sans/Mono
(the Carbon lineage, which the taste skill recommends for dashboards). The dials are VARIANCE 4,
MOTION 3 and DENSITY 7.

- The palette is monochrome graphite. The only colours are the semantic decision colours
  (ALLOW green, FLAG amber, BLOCK red), each paired with an icon. Corners are 2px everywhere.
  Light and dark themes follow the system setting, with a manual toggle.
- The decision ledger shows the total of mediated actions and one stacked ALLOW/FLAG/BLOCK bar.
- The mediation path is a pipeline: agent → policy engine → namespaces → seccomp-BPF → kernel.
  Each layer shows how many blocks it recorded, counted client-side from `/api/logs`
  (`stats.layers` also counts ALLOW rows). Unknown layers such as `tool_broker` get their own stage.
- The event stream is a table that filters by decision and by text search. Rows expand to show
  the full target and reason, and newly arrived rows are highlighted. On a phone, rows reflow into
  stacked records.
- States: skeleton rows while loading, an empty state that names the commands that populate the
  log, and a "disconnected, retrying" indicator. Clearing the log takes two steps (click Clear,
  then Confirm within 4 s).
- Kept for tests: `<tbody id="tbody"></tbody>`, `id="btn-clear"` and `id="empty"`.
- Per-model stats are not available: `/api/stats` and the log entries carry no model field. The
  console shows the active model from `/api/info` instead.

## Verify

- `pytest tests/test_dashboard_clear.py tests/test_dashboard_bind.py tests/test_review_fix_dashboard.py` in WSL.
- Visual check against an empty log and a populated log, at 375px and at desktop width, in both themes.
- No em-dash or en-dash in the user-visible strings.
