"""Unit tests for Phase 10 Computer-Tool Safety, Read-Only Defaults, and Confirmation Enforcement."""

from pathlib import Path
import tempfile
import pytest

from robbie.computer.tools import (
    PermissionLevel,
    SafetyViolation,
    SafetyPolicy,
    FileToolResult,
    FileOps,
    CommandToolResult,
    CommandRunner,
    ComputerToolManager,
)


class TestSafetyPolicy:
    """Test policy validation, read-only defaults, path sandboxing, and blacklist checks."""

    def test_read_only_default(self):
        policy = SafetyPolicy(read_only=True)
        assert policy.read_only is True
        assert PermissionLevel.READ_ONLY in policy.granted_permissions
        assert PermissionLevel.WORKSPACE_WRITE not in policy.granted_permissions

        # Mutation should be rejected
        violation = policy.validate_file_mutation("write_file", "test.txt", confirmed=True)
        assert violation is not None
        assert "read-only mode by default" in violation.reason

    def test_sensitive_path_blocking(self):
        policy = SafetyPolicy(read_only=False)
        policy.grant(PermissionLevel.WORKSPACE_WRITE)

        # Sensitive credential files
        assert policy.is_sensitive_path(".env") is True
        assert policy.is_sensitive_path("C:/Users/name/.ssh/id_rsa") is True
        assert policy.is_sensitive_path("secrets.json") is True

        violation = policy.validate_file_read(".env")
        assert violation is not None
        assert "sensitive" in violation.reason.lower()

    def test_dangerous_command_blacklist(self):
        policy = SafetyPolicy(read_only=False)
        policy.grant(PermissionLevel.COMMAND_EXECUTE)

        dangerous_cmds = [
            "rm -rf /",
            "rm -rf ./data",
            "format c:",
            "del /f /s /q c:\\",
            "shutdown /s /t 0",
            ":(){ :|:& };:",
        ]
        for cmd in dangerous_cmds:
            is_bad, _ = policy.is_dangerous_command(cmd)
            assert is_bad is True, f"Failed to detect dangerous command: {cmd}"
            violation = policy.validate_command_execution(cmd, confirmed=True)
            assert violation is not None
            assert "blacklisted" in violation.reason.lower()

    def test_confirmation_callback_enforcement(self):
        confirmed_flag = False

        def mock_confirm(msg, details):
            return confirmed_flag

        policy = SafetyPolicy(read_only=False, confirm_callback=mock_confirm)
        policy.grant(PermissionLevel.WORKSPACE_WRITE)

        # 1. User denies confirmation
        confirmed_flag = False
        v_denied = policy.validate_file_mutation("write_file", Path.cwd() / "test.txt", confirmed=False)
        assert v_denied is not None
        assert "cancelled by user" in v_denied.reason.lower()

        # 2. User approves confirmation
        confirmed_flag = True
        v_approved = policy.validate_file_mutation("write_file", Path.cwd() / "test.txt", confirmed=False)
        assert v_approved is None


class TestFileOps:
    """Test safe file operations in an isolated temporary directory."""

    def test_read_operations_allowed_by_default(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_file = tmp_path / "sample.txt"
            sample_file.write_text("Hello Robbie safe tools!", encoding="utf-8")

            # Default manager is read-only
            manager = ComputerToolManager(read_only=True, allowed_roots=[tmp_path])

            # List dir succeeds
            res_list = manager.list_dir(tmp_path)
            assert res_list.success is True
            assert any(e["name"] == "sample.txt" for e in res_list.entries)

            # Read file succeeds
            res_read = manager.read_file(sample_file)
            assert res_read.success is True
            assert res_read.content == "Hello Robbie safe tools!"

    def test_write_blocked_when_read_only(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_file = tmp_path / "new_file.txt"

            manager = ComputerToolManager(read_only=True, allowed_roots=[tmp_path])
            res_write = manager.write_file(target_file, "Unauthorized write attempt", confirm=True)

            assert res_write.success is False
            assert "read-only" in res_write.error.lower()
            assert not target_file.exists()

    def test_write_and_delete_with_permissions_and_confirmation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_file = tmp_path / "valid_file.txt"

            manager = ComputerToolManager(read_only=False, allowed_roots=[tmp_path])
            manager.grant_permission(PermissionLevel.WORKSPACE_WRITE)

            # Write with explicit confirm=True
            res_write = manager.write_file(target_file, "Authorized content", confirm=True)
            assert res_write.success is True
            assert target_file.exists()
            assert target_file.read_text(encoding="utf-8") == "Authorized content"

            # Delete with explicit confirm=True
            res_del = manager.delete_file(target_file, confirm=True)
            assert res_del.success is True
            assert not target_file.exists()


class TestCommandRunner:
    """Test command runner safety guards."""

    def test_command_blocked_when_read_only(self):
        manager = ComputerToolManager(read_only=True)
        res = manager.run_command("python --version", confirm=True)
        assert res.success is False
        assert "read-only" in res.error.lower()

    def test_safe_command_execution_when_authorized(self):
        manager = ComputerToolManager(read_only=False)
        manager.grant_permission(PermissionLevel.COMMAND_EXECUTE)

        # Run benign echo command
        res = manager.run_command("echo Hello Robbie Safe Execution", confirm=True)
        assert res.success is True
        assert "Hello Robbie Safe Execution" in res.stdout
        assert res.exit_code == 0

    def test_audit_log_tracking(self):
        manager = ComputerToolManager(read_only=True)
        manager.list_dir(Path.cwd())
        manager.run_command("dir", confirm=True)

        audit = manager.get_audit_log()
        assert len(audit) == 2
        assert audit[0]["action"] == "list_dir"
        assert audit[1]["action"] == "run_command"
        assert audit[1]["success"] is False  # Blocked because read-only
