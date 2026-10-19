"""
Agent Tools Module - AnalystGPT

Re-exports tool primitives and ToolManager from sandbox_execution for backwards compatibility.
"""

from sandbox_execution.tool_manager import (
    ToolManager,
    execute_python,
    EXECUTE_PYTHON_TOOL_SPEC,
)
from agents.fallback_parser import (
    extract_code_from_content,
    extract_action_code_from_content,
    create_synthetic_execute_tool_call,
    unwrap_envelope,
    SyntheticToolCall,
    SyntheticFunction,
    repair_json_escapes,
    parse_and_clean_tool_args,
    validate_or_retry_tool_calls,
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
    "extract_code_from_content",
    "extract_action_code_from_content",
    "create_synthetic_execute_tool_call",
    "unwrap_envelope",
    "SyntheticToolCall",
    "SyntheticFunction",
    "repair_json_escapes",
    "parse_and_clean_tool_args",
    "validate_or_retry_tool_calls",
]

