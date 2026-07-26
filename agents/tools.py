"""
Agent Tools Module - AnalystGPT

This module defines tools for AI agents, specifically the `execute_python` tool.
The `execute_python` tool uses Modal cloud sandboxing to build or use container images,
run Python code with provided files, capture execution metrics (stdout, stderr, execution time),
and return generated/returned file names and execution details back to the agent.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

try:
    import logfire
    from agents.logger import setup_logfire
    setup_logfire()
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False

# Import Modal Sandbox utilities
try:
    from sandbox_execution.utils import (
        ModalSandboxManager,
        get_global_sandbox_manager,
        reset_global_sandbox_manager,
        prepare_input_files,
        HAS_MODAL,
    )
except ImportError as err:
    print(f"[Tools Warning] Failed to import sandbox_execution.utils: {err}")
    ModalSandboxManager = None
    get_global_sandbox_manager = None
    reset_global_sandbox_manager = None
    prepare_input_files = None
    HAS_MODAL = False



def execute_python(
    code: str,
    files: Optional[Union[List[str], Dict[str, Union[str, bytes, Path]]]] = None,
    output_dir: str = "outputs",
    timeout: int = 60,
    extra_packages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes Python code with input files statefully inside a Modal cloud sandbox environment.
    Retains workspace state, generated files, and in-memory variables across tool calls.
    Captures stdout, stderr, execution time, and returned output file names.
    """
    if get_global_sandbox_manager is None:
        raise RuntimeError("Sandbox execution utilities are not available.")


    def _run():
        # 1. Prepare input files into raw bytes map
        input_files_map: Dict[str, bytes] = {}
        if files:
            if isinstance(files, str):
                files_list = [files]
            else:
                files_list = files
            input_files_map = prepare_input_files(files_list)

        if HAS_LOGFIRE:
            logfire.info(
                "Tool execute_python prepared input files",
                file_count=len(input_files_map),
                filenames=list(input_files_map.keys()),
            )
        else:
            print(f"[Tool execute_python] Prepared {len(input_files_map)} file(s) for execution: {list(input_files_map.keys())}", flush=True)

        # 2. Get Global Stateful Modal Sandbox Manager
        manager = get_global_sandbox_manager(
            app_name="analystGPT-agent-sandbox",
            extra_packages=extra_packages,
            timeout=timeout,
        )

        # 3. Run code statefully inside Modal container sandbox
        result = manager.run_code(code=code, files=input_files_map)


        # 4. Save returned files locally to host disk
        saved_file_paths = manager.save_outputs(result, output_dir=output_dir)

        returned_files_dict = result.get("files", {})
        returned_file_names = list(returned_files_dict.keys())

        if HAS_LOGFIRE:
            logfire.info(
                "Tool execute_python completed",
                success=result.get("status", False),
                execution_time=result.get("execution_time", 0.0),
                returned_files=returned_file_names,
                saved_files=saved_file_paths,
            )
        else:
            if returned_file_names:
                print(f"[Tool execute_python] Saved {len(returned_file_names)} generated output file(s): {returned_file_names} -> {output_dir}", flush=True)

        stdout_str = result.get("stdout", "")
        stderr_str = result.get("stderr", "")
        if len(stdout_str) > 2000:
            stdout_str = stdout_str[:2000] + "\n... [stdout truncated at 2000 characters limit]"
        if len(stderr_str) > 2000:
            stderr_str = "... [stderr truncated from start, showing last 2000 characters limit]\n" + stderr_str[-2000:]

        # 5. Format and return execution details to agent
        return {
            "success": result.get("status", False),
            "execution_time": result.get("execution_time", 0.0),
            "stdout": stdout_str,
            "stderr": stderr_str,
            "returned_files": returned_file_names,
            "saved_files": saved_file_paths,
            "error": stderr_str if not result.get("status", False) else "",
        }

    if HAS_LOGFIRE:
        with logfire.span(
            "Tool Execution: execute_python",
            code_length=len(code),
            code=code,
            output_dir=output_dir,
            has_input_files=bool(files),
        ):
            return _run()
    else:
        return _run()




# ==========================================
# OpenAI Function Schema Definition for Agent
# ==========================================

EXECUTE_PYTHON_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Executes Python code inside a Modal cloud sandbox container with input files. "
            "Captures stdout, stderr, execution time, and returned output file names (plots, CSVs, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code snippet to execute.",
                },
                "files": {
                    "type": "object",
                    "description": "Map of filename to file content or file paths provided to the container.",
                    "additionalProperties": {"type": "string"},
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save generated output files locally.",
                    "default": "outputs",
                },
            },
            "required": ["code"],
        },
    },
}
