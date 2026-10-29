---
quick_id: 261029-vds
slug: demo-mode-and-model-line-up
date: 2026-10-29
mode: quick (inline, no subagents)
files:
  - src/dashboard/demo.py
  - src/dashboard/app.py (this task's hunks only; the user's WIP hunks stay uncommitted)
  - src/dashboard/templates/demo.html
  - src/dashboard/templates/_shell.html
  - src/dashboard/templates/index.html
  - src/dashboard/static/demo.js
  - src/dashboard/static/dashboard.css
  - src/dashboard/static/monitor.js
  - src/dashboard/static/results.js
  - tests/test_dashboard_demo.py
---

# Quick task: a demo page for the review, and the real model line-up

The user said: "how am i supposed to present my project i mean demo???? add some ways in frontend
webpage. also the website states agent model gemini whereas we are using many llms including locally
built one".

## Model label

- The rail used to show `info.model`, which is `default:` in config/models.yaml (gemini-flash-lite).
  The study compares many models, so the rail now shows the line-up count and links to it.
- `/api/lineup` lists every model from the frozen `numbers.json`. Each entry carries only the
  provider and the served model id from `inputs/config/models.yaml`, never `base_url` or
  `api_key_env`. Models are grouped into: locally made, local pretrained, hosted, legacy and the
  scripted control.
- The live log records no model, and the monitor now says so instead of naming one. config is
  unchanged.

## Demo page (`/demo`)

- **(a) Replay.**
  - A recorded episode from `results/phase5-final/sources/<run>/raw.jsonl`, both arms side by side,
    with Play, Step and Reset, the Space/P/R keys, and ←/→ to move between stories.
  - The case text is rebuilt from the case id with `src.benchmark.injecagent`, and the injected
    instruction is highlighted. Canary tokens are marked as planted test data.
  - Per call, the decision comes from the record's `tool_calls[].allowed`, which is the tool broker.
  - OS actions are re-checked with the policy engine, logging to `/dev/null`, and only when the
    policy sha256 equals the run's. Otherwise the decision is shown as "checked, per-episode count
    only".
  - Four stories are picked from the data, not by hand:
    - Qwen LoRA hijacked, then blocked (a `ds` case);
    - ScratchGPT the same;
    - a benign task done in both arms;
    - the control, where both the broker and the policy engine act.
  - Models are looked up from the frozen source list and never joined into a path. Case and run
    must match strict patterns. Only an allow list of record fields is returned.
- **(b) Live simulation.**
  - `POST /api/demo/simulate` with `{"scenario": "all" | "s1".."s8"}` and nothing else (no query
    string, no other keys).
  - It applies the same-origin check as `/api/clear` and sits behind the Host guard.
  - The scenarios are `run_attack_simulation.ATTACKS`, unchanged. Each action goes through
    `PolicyEngine.check_detailed` only and is never executed (Round 1 of that script really runs
    the commands, so it is not used).
  - One run at a time (409 while busy), paced at 0.9 s, with a 30 s deadline. It writes only to
    the dashboard's `LOG_FILE`.
- **(c) Live local-model run: skipped.** It would mean loading torch or calling Ollama from a web
  request through the runner and harness, and a browser button must never reach the hosted models.
  The page says so and points to the terminal.
- **(d) How to demo:** a 5-minute script with what to click and what to say, tied to the viva story.
- **Presenter view:** larger type, no rail, remembered per browser.
