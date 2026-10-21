---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-10-21T04:21:03.303Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 2 | stub | src/scratch_llm/dataset.py | 676 | generate_episodes raises NotImplementedError for split other than train (dh/ds classes, dev splits and synthetic tools land in plan 02-02) | open |  | 2026-10-21T04:21:03.303Z |  |

````json
[
  {
    "id": 1,
    "kind": "stub",
    "phase": "2",
    "file": "src/scratch_llm/dataset.py",
    "line": 676,
    "description": "generate_episodes raises NotImplementedError for split other than train (dh/ds classes, dev splits and synthetic tools land in plan 02-02)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-10-21T04:21:03.303Z",
    "resolved_at": null,
    "milestone": null
  }
]
````
