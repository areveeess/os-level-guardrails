---
quick_id: 261029-qmt
slug: remake-dashboard-with-results-view
status: complete
date: 2026-10-29
commit: cebff4b
---

# Summary: dashboard rebuilt, with a results view

The dashboard is rebuilt from scratch with a new direction: a light-first lab instrument with
Geist type, a cobalt accent, soft panels and a left rail. Nothing is reused from 261028-th2.
It now has two views: the live monitor at `/` and the frozen Phase 5 comparison at `/results`.

## Files

- `src/dashboard/templates/_shell.html`: the shared rail and the theme toggle.
- `src/dashboard/templates/index.html`: the live monitor, which keeps `<tbody id="tbody"></tbody>`,
  `#btn-clear` and `#empty`.
- `src/dashboard/templates/results.html`: the results view, rendered client-side from `/api/results`.
- `src/dashboard/static/dashboard.css`, `monitor.js`, `results.js`: the CSS and the two view scripts.
- `src/dashboard/app.py`: adds `/results`, `/api/results` and `/api/results/files/<name>`, with
  an allow list and `nosniff`. Only this task's hunks are committed. The user's WIP hunks
  (the extra `request` import, `RESULTS_DIR`, the `sys.path` insert) stay uncommitted in the
  working tree.
- `tests/test_dashboard_results.py`: 28 tests.

## Notes

- Blocks per layer on the monitor are counted from `/api/logs`, because `stats.layers` also
  counts ALLOW rows.
- The results view reads `data_state`. While the data is a snapshot, it shows the snapshot and
  freeze times with a countdown. When the build writes `data_state: final`, it switches to
  "Final data" with no code change.
- Per-model live stats are still not possible, because log entries carry no model field.
- Only complete runs appear in the headline table and the arrow chart. Partial hosted runs are
  shown separately and labelled "not final and not ranked".

## Verification

- The dashboard tests (`test_dashboard_clear`, `test_dashboard_bind`,
  `test_review_fix_dashboard`, `test_dashboard_results`) pass: 111 passed in WSL, on both the
  working tree and a HEAD-plus-these-hunks checkout.
- Checked in the browser:
  - both views in light and dark at 1440×900 and 375×812, with no horizontal scroll at 375;
  - no console errors;
  - the chart tooltips;
  - the clear popover against a scratch log;
  - the empty and populated logs.
- No em-dashes or en-dashes in the templates or static files.
