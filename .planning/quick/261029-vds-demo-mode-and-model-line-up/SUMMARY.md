---
quick_id: 261029-vds
slug: demo-mode-and-model-line-up
status: complete
date: 2026-10-29
commit: c072341
---

# Summary: demo page and model line-up

- New `/demo` page with:
  - recorded-episode replay (four data-picked stories, plus any model, case and run);
  - the live scripted policy simulation;
  - a 5-minute "how to demo" script;
  - a presenter view.
- New routes:
  - `/api/lineup`;
  - `/api/demo/replay`, `/api/demo/cases`, `/api/demo/episode`;
  - `/api/demo/scenarios` and `/api/demo/simulate` (GET for status, same-origin POST to start).
- The rail no longer shows a single "agent model". It shows the line-up ("8 LLMs + control"),
  and Results has a grouped line-up panel. config/models.yaml is unchanged.
- Option (c), a live local-model run from the browser, is skipped. The page explains why.
- Only this task's app.py hunks are committed; the user's WIP hunks stay in the working tree.

## Verification

- 161 dashboard tests pass in WSL (50 of them new in `tests/test_dashboard_demo.py`), on both the
  working tree and a HEAD-plus-these-hunks checkout. The new tests cover:
  - path traversal and unknown models;
  - case and run patterns;
  - field allow-listing;
  - the simulation's origin, Host, query, body and method checks;
  - the concurrency lock, the deadline, and the run executing nothing;
  - log-only writes.
- Checked in the browser on the final post-freeze data (`data_state: final`):
  - Demo in presenter view at 1440×900 in light mode, and at 375×812 in dark mode, with no
    horizontal scroll;
  - the replay of the Qwen LoRA and control stories;
  - the live simulation streaming 8 decisions into a scratch log, which the monitor counted;
  - the Results line-up panel.
