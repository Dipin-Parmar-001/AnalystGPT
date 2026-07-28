"""
Tool Manager & Execution Workflow Module - AnalystGPT

Manages tool registration, argument parsing, exception handling,
and stateful sandbox execution workflow for AI agents.
"""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

try:
    import logfire
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
    print(f"[ToolManager Warning] Failed to import sandbox_execution.utils: {err}")
    ModalSandboxManager = None
    get_global_sandbox_manager = None
    reset_global_sandbox_manager = None
    prepare_input_files = None
    HAS_MODAL = False


# ==========================================
# 1. Default Python Sandbox Tool & Spec
# ==========================================

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


# ==========================================
# 2. ToolManager Class
# ==========================================

class ToolManager:
    """
    Decoupled tool workflow manager.
    Handles tool registration, schema management, JSON argument parsing,
    tool execution, exception handling, and Logfire telemetry.
    """
    def __init__(self, initial_tools: Optional[Dict[str, Callable]] = None):
        self.tools: Dict[str, Callable] = {
            "execute_python": execute_python
        }
        if initial_tools:
            self.tools.update(initial_tools)

        self.specs: Dict[str, Dict[str, Any]] = {
            "execute_python": EXECUTE_PYTHON_TOOL_SPEC
        }

    def register_tool(self, name: str, fn: Callable, spec: Optional[Dict[str, Any]] = None):
        """Registers a new tool function and optional OpenAI JSON schema spec."""
        self.tools[name] = fn
        if spec:
            self.specs[name] = spec

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Returns list of OpenAI function tool specifications."""
        return list(self.specs.values())

    def execute_tool_call(
        self,
        tool_call: Any,
        files: Optional[Union[List[str], Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Parses and executes a single LLM tool call object.
        Handles argument parsing, file context injection, tool execution,
        exception recovery, and telemetry.
        """
        tool_call_id = getattr(tool_call, "id", None) or (tool_call.get("id") if isinstance(tool_call, dict) else "unknown")
        
        # Function name extraction
        if hasattr(tool_call, "function"):
            fn_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
        elif isinstance(tool_call, dict) and "function" in tool_call:
            fn_name = tool_call["function"].get("name", "")
            raw_args = tool_call["function"].get("arguments", "{}")
        else:
            fn_name = "unknown"
            raw_args = "{}"

        # JSON parsing of arguments
        parse_error = None
        try:
            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception as parse_err:
            fn_args = {}
            parse_error = f"{type(parse_err).__name__}: {str(parse_err)}"
            if HAS_LOGFIRE:
                logfire.error(
                    "LLM Tool Call JSON Argument Parsing Failed",
                    tool_call_id=tool_call_id,
                    fn_name=fn_name,
                    raw_arguments=raw_args,
                    parse_error=parse_error,
                    traceback=traceback.format_exc(),
                )
            else:
                print(f"[Tool Parse Error] Failed to parse JSON for '{fn_name}' (ID: {tool_call_id}). Raw: {raw_args}. Error: {parse_error}")

        def _do_execute() -> Dict[str, Any]:
            if fn_name in self.tools:
                if files and "files" not in fn_args:
                    fn_args["files"] = files

                try:
                    tool_res = self.tools[fn_name](**fn_args)
                except Exception as err:
                    tool_res = {
                        "success": False,
                        "error": f"Tool execution raised exception: {str(err)}",
                        "stderr": traceback.format_exc(),
                    }
                    if HAS_LOGFIRE:
                        logfire.error(
                            "Tool Execution Raised Exception",
                            fn_name=fn_name,
                            tool_call_id=tool_call_id,
                            error=str(err),
                            traceback=traceback.format_exc(),
                        )
            else:
                tool_res = {
                    "success": False,
                    "error": f"Unknown tool requested by model: '{fn_name}'",
                }
                if HAS_LOGFIRE:
                    logfire.warn(
                        "Model requested unknown tool",
                        requested_tool=fn_name,
                        available_tools=list(self.tools.keys()),
                    )

            is_success = tool_res.get("success", False)
            exec_time = tool_res.get("execution_time", 0.0)
            returned_files = tool_res.get("returned_files", [])
            stdout_str = str(tool_res.get("stdout", ""))
            stderr_str = str(tool_res.get("stderr", ""))

            if HAS_LOGFIRE:
                log_func = logfire.info if is_success else logfire.error
                log_func(
                    f"Tool Execution {'SUCCESS' if is_success else 'FAILED'}: {fn_name}",
                    fn_name=fn_name,
                    tool_call_id=tool_call_id,
                    success=is_success,
                    execution_time=exec_time,
                    code=fn_args.get("code", "") if isinstance(fn_args, dict) and "code" in fn_args else None,
                    returned_files=returned_files,
                    stdout_snippet=stdout_str[:500],
                    stderr_snippet=stderr_str[-500:] if len(stderr_str) > 500 else stderr_str,
                    error=tool_res.get("error"),
                    parse_error=parse_error,
                )
            else:
                print(f"[Tool Result]: status={'SUCCESS' if is_success else 'FAILED'} (time: {exec_time}s)", flush=True)

            return tool_res

        if HAS_LOGFIRE:
            with logfire.span(
                f"Tool Call: {fn_name}",
                fn_name=fn_name,
                tool_call_id=tool_call_id,
                raw_arguments=raw_args,
                parsed_arguments=fn_args,
                parse_error=parse_error,
            ):
                return _do_execute()
        else:
            print(f"\n[Tool Execution Request]: '{fn_name}' (ID: {tool_call_id})", flush=True)
            return _do_execute()
