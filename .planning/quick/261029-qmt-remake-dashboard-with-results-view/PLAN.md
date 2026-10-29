---
quick_id: 261029-qmt
slug: remake-dashboard-with-results-view
date: 2026-10-29
mode: quick (inline, no subagents)
supersedes: 261028-th2 (the reference-monitor console design)
files:
  - src/dashboard/app.py (this task's hunks only; the user's WIP hunks stay uncommitted)
  - src/dashboard/templates/_shell.html
  - src/dashboard/templates/index.html
  - src/dashboard/templates/results.html
  - src/dashboard/static/dashboard.css
  - src/dashboard/static/monitor.js
  - src/dashboard/static/results.js
  - tests/test_dashboard_results.py
---

# Quick task: rebuild the dashboard from scratch and add a results view

The user asked: "remake the front end from scratch using impeccable and taste skill disregarding
previous design". The 261028-th2 console (graphite palette, IBM Plex, 2px corners, top bar,
horizontal layer pipeline, table stream) is discarded completely.

## Audit of 261028-th2

- The design read as a terminal: dark-first and monochrome. It suited a log reader but not a
  panel that also has to read a results comparison.
- The page had one view only, so the Phase 5 numbers lived in `summary.md` and the PNGs, outside
  the dashboard.
- The horizontal pipeline needed a different layout at every breakpoint. The layers' nesting
  (defense in depth) was described in words, not shown.

## Direction

The design read: a two-view security research instrument for an OS-security review panel, with a
calm, light-first "lab instrument" language, built with native CSS, Geist Sans and Geist Mono
(a taste-skill pairing), and native SVG charts.

- Dials: VARIANCE 5, MOTION 3, DENSITY 6.
- Cool paper surfaces in light mode and deep navy in dark mode. One cobalt accent marks the
  guardrails and every interactive element. Panels have soft 12px corners and buttons are
  pill-shaped.
- A persistent left rail holds the two views, the session facts and the theme toggle. At phone
  width it becomes a top bar with two tabs.
- **Live monitor (`/`):**
  - a "blocked of mediated" hero with the allow, flag and block split;
  - concentric defense rings (tool broker → policy engine → namespaces → seccomp-BPF → kernel),
    where each arc's length is that layer's share of the blocks;
  - a decision timeline with filter chips and search;
  - a clear confirmation popover.
  - The same JSON routes as before; the test hooks `#tbody`, `#btn-clear` and `#empty` are kept.
- **Results (`/results`):**
  - a snapshot vs final banner;
  - key figures;
  - two native charts drawn from numbers.json (attack success with Wilson intervals; benign task
    done vs attack success as arrows);
  - the headline, partial and legacy tables;
  - blocks by layer, pooled and per model;
  - the scripted control;
  - the three build PNGs;
  - threats, confounds and sources.
- Chart palette: orange = no guardrails and blue = guardrails (#eb6834/#2a78d6 in light,
  #d95926/#3987e5 in dark). It passes the dataviz validator against both panel surfaces.

## Backend

- `GET /results` renders the view.
- `GET /api/results` returns `results/phase5-final/numbers.json` unchanged: a JSON 404 when the
  file is missing and a JSON 500 when it is unreadable.
- `GET /api/results/files/<name>` serves an allow list only: `numbers.json`, `freeze.json`,
  `summary.md`, `summary.csv`, `layer_blocks.csv` and `charts/<name>.png`. `..`, encoded
  separators, absolute paths, `sources/` and `inputs/` never match. Responses carry `nosniff`.
- `RESULTS_FINAL_DIR` is built from `PROJECT_ROOT`, not from the user's uncommitted `RESULTS_DIR`,
  so the committed file stands on its own.
- The code is read-only: it never calls `src.deliverables build`.

## Verify

- `tests/test_dashboard_results.py` covers traversal, the allow list, read-only methods, missing
  and corrupt data, the page render and the Host guard.
- Run all the dashboard tests, against both the working tree and HEAD plus this task's hunks.
- Check both views in light and dark at 1440 and 375, against an empty log, a populated log and
  the real numbers.json.
