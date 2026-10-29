---
schema_version: 1
open_count: 2
waived_count: 0
fixed_count: 2
total_count: 4
last_updated: 2026-10-29T13:03:49.833Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 2 | stub | src/scratch_llm/dataset.py | 676 | generate_episodes raises NotImplementedError for split other than train (dh/ds classes, dev splits and synthetic tools land in plan 02-02) | fixed |  | 2026-10-21T04:21:03.303Z | 2026-10-21T04:54:27.652Z |
| 2 | 4 | deviation | docs/viva/04-namespaces-seccomp-and-local-inference.md |  | Part C Groq rows quote the 2026-10-29T03:06Z snapshot; regenerate results/phase4-report after the 15:30 UTC freeze and update Part C with the final Groq n (CMP-03) | open |  | 2026-10-29T03:21:00.341Z |  |
| 3 | 4 | stub | docs/viva/04-namespaces-seccomp-and-local-inference.md | 431 | Part C section 'What the code review found and what we fixed' is a placeholder until the Phase 4 code review runs | fixed |  | 2026-10-29T03:21:01.256Z | 2026-10-29T13:03:49.833Z |
| 4 | 4 | todo | src/benchmark/outcomes.py |  | Phase 4 review findings deferred until after the 2026-10-29 15:30 UTC freeze (run-affecting paths; the Groq loop stamps them): WR-01 mid-episode context overflow classed format_failure biases asr_valid upward (src/benchmark/outcomes.py); IN-04 overflow marker matched as a substring of any provider error (src/benchmark/outcomes.py); IN-05 priority_kinds kept in plan extra and overwritten on resume, stale _work_queue docstring (src/benchmark/runner.py, run_benchmark.py). See 04-REVIEW-FIX.md. | open |  | 2026-10-29T03:51:26.188Z |  |

````json
[
  {
    "id": 1,
    "kind": "stub",
    "phase": "2",
    "file": "src/scratch_llm/dataset.py",
    "line": 676,
    "description": "generate_episodes raises NotImplementedError for split other than train (dh/ds classes, dev splits and synthetic tools land in plan 02-02)",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-21T04:21:03.303Z",
    "resolved_at": "2026-10-21T04:54:27.652Z",
    "milestone": null
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "4",
    "file": "docs/viva/04-namespaces-seccomp-and-local-inference.md",
    "line": null,
    "description": "Part C Groq rows quote the 2026-10-29T03:06Z snapshot; regenerate results/phase4-report after the 15:30 UTC freeze and update Part C with the final Groq n (CMP-03)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T03:21:00.341Z",
    "resolved_at": null,
    "milestone": null
  },
  {
    "id": 3,
    "kind": "stub",
    "phase": "4",
    "file": "docs/viva/04-namespaces-seccomp-and-local-inference.md",
    "line": 431,
    "description": "Part C section 'What the code review found and what we fixed' is a placeholder until the Phase 4 code review runs",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-10-29T03:21:01.256Z",
    "resolved_at": "2026-10-29T13:03:49.833Z",
    "milestone": null
  },
  {
    "id": 4,
    "kind": "todo",
    "phase": "4",
    "file": "src/benchmark/outcomes.py",
    "line": null,
    "description": "Phase 4 review findings deferred until after the 2026-10-29 15:30 UTC freeze (run-affecting paths; the Groq loop stamps them): WR-01 mid-episode context overflow classed format_failure biases asr_valid upward (src/benchmark/outcomes.py); IN-04 overflow marker matched as a substring of any provider error (src/benchmark/outcomes.py); IN-05 priority_kinds kept in plan extra and overwritten on resume, stale _work_queue docstring (src/benchmark/runner.py, run_benchmark.py). See 04-REVIEW-FIX.md.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-29T03:51:26.188Z",
    "resolved_at": null,
    "milestone": null
  }
]
````
