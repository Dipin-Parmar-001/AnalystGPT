"""
Agent Tools Module - AnalystGPT

Re-exports tool primitives and ToolManager from sandbox_execution for backwards compatibility.
"""

from sandbox_execution.tool_manager import (
    ToolManager,
    execute_python,
    EXECUTE_PYTHON_TOOL_SPEC,
)

try:
    from sandbox_execution.utils import reset_global_sandbox_manager
except ImportError:
    reset_global_sandbox_manager = None

__all__ = [
    "ToolManager",
    "execute_python",
    "EXECUTE_PYTHON_TOOL_SPEC",
    "reset_global_sandbox_manager",
]
