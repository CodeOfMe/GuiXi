"""
Comprehensive test suite for GuiXi following python-project-developer standards.

Includes all required test classes:
- TestToolResult
- TestGuiXiAPI
- TestToolsSchema
- TestToolsDispatch
- TestCLIFlags
- TestPackageExports
"""

import subprocess
import sys
from pathlib import Path

import pytest


# ============================================================================
# TestToolResult
# ============================================================================


class TestToolResult:
    """Test ToolResult dataclass behavior."""

    def test_success_result(self):
        """Test creating a success result."""
        from guixi.api import ToolResult

        r = ToolResult(success=True, data={"key": "value"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        from guixi.api import ToolResult

        r = ToolResult(success=False, error="failed")
        assert r.success is False
        assert r.error == "failed"

    def test_to_dict_success(self):
        """Test to_dict method for success result."""
        from guixi.api import ToolResult

        r = ToolResult(success=True, data=[1, 2])
        d = r.to_dict()
        assert set(d.keys()) == {"success", "data", "error", "metadata"}
        assert d["success"] is True
        assert d["data"] == [1, 2]

    def test_to_dict_failure(self):
        """Test to_dict method for failure result."""
        from guixi.api import ToolResult

        r = ToolResult(success=False, error="error message")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "error message"

    def test_default_metadata_isolation(self):
        """Test that default metadata dicts don't share state."""
        from guixi.api import ToolResult

        r1 = ToolResult(success=True)
        r2 = ToolResult(success=True)
        r1.metadata["a"] = 1
        assert "a" not in r2.metadata

    def test_metadata_with_version(self):
        """Test metadata includes version."""
        from guixi.api import ToolResult

        r = ToolResult(success=True, metadata={"version": "0.1.0"})
        assert r.metadata["version"] == "0.1.0"


# ============================================================================
# TestGuiXiAPI
# ============================================================================


class TestGuiXiAPI:
    """Test API functions."""

    def test_api_infer_signature(self):
        """Test api_infer has correct signature with keyword-only args."""
        import inspect
        from guixi.api import api_infer

        sig = inspect.signature(api_infer)
        params = list(sig.parameters.values())

        # Check that all params after * are KEYWORD_ONLY
        # The * itself may not appear in params, but following params should be KEYWORD_ONLY
        param_kinds = [p.kind for p in params]
        assert inspect.Parameter.KEYWORD_ONLY in param_kinds, (
            "Function should have keyword-only args after *"
        )

    @pytest.mark.asyncio
    async def test_api_infer_basic(self):
        """Test basic api_infer call."""
        from guixi.api import api_infer

        result = await api_infer(prompt="test prompt")
        assert hasattr(result, "success")
        assert hasattr(result, "to_dict")

    def test_api_compress_signature(self):
        """Test api_compress has correct signature with keyword-only args."""
        import inspect
        from guixi.api import api_compress

        sig = inspect.signature(api_compress)
        params = list(sig.parameters.values())

        # Check that all params after * are KEYWORD_ONLY
        param_kinds = [p.kind for p in params]
        assert inspect.Parameter.KEYWORD_ONLY in param_kinds, (
            "Function should have keyword-only args after *"
        )

    @pytest.mark.asyncio
    async def test_api_compress_basic(self):
        """Test basic api_compress call."""
        from guixi.api import api_compress

        result = await api_compress(data=[1, 2, 3, 4, 5])
        assert hasattr(result, "success")

    def test_api_cache_stats_signature(self):
        """Test api_cache_stats has correct signature."""
        import inspect
        from guixi.api import api_cache_stats

        sig = inspect.signature(api_cache_stats)
        # No required parameters
        assert len(sig.parameters) == 0

    @pytest.mark.asyncio
    async def test_api_cache_stats_basic(self):
        """Test basic api_cache_stats call."""
        from guixi.api import api_cache_stats

        result = await api_cache_stats()
        assert hasattr(result, "success")


# ============================================================================
# TestToolsSchema
# ============================================================================


class TestToolsSchema:
    """Test OpenAI function-calling tool schemas."""

    def test_tool_structure(self):
        """Test each tool has correct structure."""
        from guixi.tools import TOOLS

        for tool in TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_required_fields_in_properties(self):
        """Test required fields exist in properties."""
        from guixi.tools import TOOLS

        for tool in TOOLS:
            func = tool["function"]
            props = func["parameters"]["properties"]
            for req in func["parameters"]["required"]:
                assert req in props, f"Required field '{req}' not in properties"

    def test_tool_names_unique(self):
        """Test all tool names are unique."""
        from guixi.tools import TOOLS

        names = [t["function"]["name"] for t in TOOLS]
        assert len(names) == len(set(names))

    def test_guixi_infer_tool(self):
        """Test guixi_infer tool schema."""
        from guixi.tools import get_tool

        tool = get_tool("guixi_infer")
        assert tool is not None
        assert tool["function"]["name"] == "guixi_infer"
        assert "prompt" in tool["function"]["parameters"]["required"]

    def test_guixi_batch_infer_tool(self):
        """Test guixi_batch_infer tool schema."""
        from guixi.tools import get_tool

        tool = get_tool("guixi_batch_infer")
        assert tool is not None
        assert "prompts" in tool["function"]["parameters"]["required"]


# ============================================================================
# TestToolsDispatch
# ============================================================================


class TestToolsDispatch:
    """Test dispatch function."""

    def test_dispatch_known_tool(self):
        """Test dispatching to known tool."""
        from guixi.tools import dispatch

        result = dispatch("guixi_cache_stats", {})
        assert isinstance(result, dict)
        assert "success" in result

    def test_dispatch_unknown_tool(self):
        """Test dispatching to unknown tool raises error."""
        from guixi.tools import dispatch

        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("unknown_tool", {})

    def test_dispatch_with_string_args(self):
        """Test dispatch with JSON string arguments."""
        from guixi.tools import dispatch

        result = dispatch("guixi_cache_stats", "{}")
        assert isinstance(result, dict)

    def test_dispatch_guixi_infer(self):
        """Test dispatching guixi_infer."""
        from guixi.tools import dispatch

        result = dispatch("guixi_infer", {"prompt": "test", "max_tokens": 10})
        assert isinstance(result, dict)
        assert "success" in result


# ============================================================================
# TestCLIFlags
# ============================================================================


class TestCLIFlags:
    """Test CLI flags and behavior."""

    def _run_cli(self, *args):
        """Run CLI with given arguments."""
        return subprocess.run(
            [sys.executable, "-m", "guixi"] + list(args),
            capture_output=True,
            text=True,
            timeout=15,
            cwd=Path(__file__).parent.parent,
        )

    def test_version_flag(self):
        """Test -V flag outputs version."""
        r = self._run_cli("-V")
        assert r.returncode == 0
        assert "guixi" in r.stdout.lower() or "0.1" in r.stdout

    def test_version_long_flag(self):
        """Test --version flag."""
        r = self._run_cli("--version")
        assert r.returncode == 0

    def test_help_has_unified_flags(self):
        """Test --help shows unified flags."""
        r = self._run_cli("--help")
        assert "--json" in r.stdout
        assert "-q" in r.stdout or "--quiet" in r.stdout
        assert "-v" in r.stdout or "--verbose" in r.stdout
        assert "-o" in r.stdout or "--output" in r.stdout

    def test_help_flag(self):
        """Test -h flag shows help."""
        r = self._run_cli("-h")
        assert r.returncode == 0
        assert "GuiXi" in r.stdout or "usage" in r.stdout.lower()


# ============================================================================
# TestPackageExports
# ============================================================================


class TestPackageExports:
    """Test package __init__.py exports."""

    def test_version_export(self):
        """Test __version__ is exported."""
        import guixi

        assert hasattr(guixi, "__version__")
        assert guixi.__version__ == "0.1.0"

    def test_toolresult_export(self):
        """Test ToolResult is exported."""
        import guixi

        assert hasattr(guixi, "ToolResult")
        from guixi.api import ToolResult

        assert guixi.ToolResult is ToolResult

    def test_tools_export(self):
        """Test TOOLS is exported."""
        import guixi

        assert hasattr(guixi, "TOOLS")
        from guixi.tools import TOOLS

        assert guixi.TOOLS is TOOLS

    def test_dispatch_export(self):
        """Test dispatch is exported."""
        import guixi

        assert hasattr(guixi, "dispatch")
        from guixi.tools import dispatch

        assert guixi.dispatch is dispatch

    def test_list_tool_names_export(self):
        """Test list_tool_names is exported."""
        import guixi

        assert hasattr(guixi, "list_tool_names")

    def test_all_contains_exports(self):
        """Test __all__ contains expected exports."""
        import guixi

        assert "__version__" in guixi.__all__
        assert "ToolResult" in guixi.__all__
        assert "TOOLS" in guixi.__all__
        assert "dispatch" in guixi.__all__

    def test_import_api_functions(self):
        """Test API functions can be imported from package."""
        from guixi import api_infer, api_batch_infer, api_cache_stats

        assert callable(api_infer)
        assert callable(api_batch_infer)
        assert callable(api_cache_stats)
