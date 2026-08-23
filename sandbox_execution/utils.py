"""
Utility functions and manager classes for Modal cloud sandbox execution environment.

This module manages:
1. Modal App initialization and container image building with data science libraries.
2. Input file packaging and sending (Host -> Cloud Sandbox).
3. Execution output capture (stdout, stderr, execution time, tracebacks).
4. Output file extraction and saving (Cloud Sandbox -> Host).
5. High-level manager interface for agent integration.
"""

import os
import io
import json
import time
import mimetypes
import tempfile
import contextlib
import traceback
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    import modal
    HAS_MODAL = True
except ImportError:
    modal = None
    HAS_MODAL = False

try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False




# Default data science libraries pre-installed in the Modal cloud sandbox image
DEFAULT_SANDBOX_PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "imbalanced-learn",
    "joblib",
    "openpyxl",
]


# ==========================================
# 1. Image & App Initialization Utilities
# ==========================================

def get_modal_app(app_name: str = "analystGPT-sandbox") -> Any:
    """
    Initializes and returns a Modal App instance.
    """
    if not HAS_MODAL:
        raise ImportError(
            "The 'modal' package is not installed. Please install it using `pip install modal`."
        )
    return modal.App(app_name)


def build_sandbox_image(
    extra_pip_packages: Optional[List[str]] = None,
    debian_packages: Optional[List[str]] = None,
) -> Any:
    """
    Builds and configures a Modal container image with required Python packages.

    Args:
        extra_pip_packages: Additional pip libraries to install alongside default ones.
        debian_packages: System apt packages to install in the debian container.

    Returns:
        modal.Image: Configured Modal container image.
    """
    if not HAS_MODAL:
        raise ImportError(
            "The 'modal' package is not installed. Please install it using `pip install modal`."
        )

    pip_packages = list(DEFAULT_SANDBOX_PACKAGES)
    if extra_pip_packages:
        for pkg in extra_pip_packages:
            if pkg not in pip_packages:
                pip_packages.append(pkg)

    image = modal.Image.debian_slim()

    if debian_packages:
        image = image.apt_install(*debian_packages)

    if pip_packages:
        image = image.pip_install(*pip_packages)

    # Mount local python source into remote container so deserialization of sandbox_execution succeeds
    image = image.add_local_python_source("sandbox_execution")

    return image


# ==========================================
# 2. File Registry & File Input Utilities
# ==========================================

def resolve_file_path(file_path: Union[str, Path]) -> str:
    """
    Robustly resolves a file path across:
    1. Direct absolute/relative path
    2. Project workspace root
    3. uploads/ folder
    4. outputs/ folder
    5. Case-insensitive filename match in uploads/
    """
    if not file_path:
        raise FileNotFoundError("Empty file path provided.")

    p_str = str(file_path).strip()
    if os.path.exists(p_str) and not os.path.isdir(p_str):
        return os.path.abspath(p_str)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_name = os.path.basename(p_str)

    candidates = [
        os.path.join(workspace_root, p_str),
        os.path.join(workspace_root, "uploads", base_name),
        os.path.join(workspace_root, "uploads", p_str),
        os.path.join(workspace_root, "outputs", base_name),
        os.path.join(workspace_root, base_name),
        os.path.join(os.getcwd(), "uploads", base_name),
        os.path.join(os.getcwd(), "outputs", base_name),
        os.path.join(os.getcwd(), "..", "uploads", base_name),
        os.path.join(os.getcwd(), "..", "outputs", base_name),
    ]

    for cand in candidates:
        if os.path.exists(cand) and not os.path.isdir(cand):
            return os.path.abspath(cand)

    # Case-insensitive fallback in uploads/
    uploads_dir = os.path.join(workspace_root, "uploads")
    if os.path.exists(uploads_dir):
        for existing in os.listdir(uploads_dir):
            if existing.lower() == base_name.lower():
                cand_upload = os.path.join(uploads_dir, existing)
                if os.path.exists(cand_upload) and not os.path.isdir(cand_upload):
                    return os.path.abspath(cand_upload)

    raise FileNotFoundError(f"File not found: '{file_path}'. Checked workspace and uploads folder.")


def build_file_registry(
    files: Optional[Union[List[str], Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs a structured FILE_REGISTRY dictionary mapping filename / relative path
    to file metadata and concrete path inside the sandbox environment.
    """
    registry: Dict[str, Dict[str, Any]] = {}
    if not files:
        return registry

    if isinstance(files, list):
        for f in files:
            if not f:
                continue
            base = os.path.basename(str(f))
            ext = os.path.splitext(base)[1].lstrip(".").lower() or "unknown"
            size = 0
            try:
                resolved_p = resolve_file_path(f)
                size = os.path.getsize(resolved_p)
            except Exception:
                pass
            entry = {
                "filename": base,
                "path": base,
                "type": ext,
                "size_bytes": size,
            }
            registry[base] = entry
            # Also register normalized relative path if different
            rel_norm = str(f).replace("\\", "/")
            if rel_norm != base:
                registry[rel_norm] = entry
    elif isinstance(files, dict):
        for fname, content in files.items():
            base = os.path.basename(str(fname))
            ext = os.path.splitext(base)[1].lstrip(".").lower() or "unknown"
            size = len(content) if isinstance(content, (bytes, str)) else 0
            entry = {
                "filename": base,
                "path": base,
                "type": ext,
                "size_bytes": size,
            }
            registry[base] = entry
            if fname != base:
                registry[fname] = entry

    return registry


def format_file_registry_prompt(files: Optional[Union[List[str], Dict[str, Any]]]) -> str:
    """
    Formats the file registry as a readable Python definition and guidelines block for LLM prompts.
    """
    if not files:
        return "=== FILE REGISTRY ===\nNo uploaded files available.\n"
    registry = build_file_registry(files)
    # Simplify dictionary for prompt clarity (show unique basenames)
    prompt_registry = {v["filename"]: {"path": v["path"], "type": v["type"]} for v in registry.values()}
    lines = [
        "=== FILE REGISTRY & DATASET PATHS ===",
        "The following dataset file(s) are registered and available in the Python sandbox:",
        "```python",
        f"FILE_REGISTRY = {json.dumps(prompt_registry, indent=2)}",
        "```",
        "Loading Instructions:",
        "- Load directly by registry filename: e.g. pd.read_csv('filename.csv')",
        "- Or use the available helper function: e.g. pd.read_csv(get_file_path('filename.csv'))",
        "- Do NOT use directory prefixes like 'uploads/' or 'data/' unless specified in the registry.",
    ]
    return "\n".join(lines) + "\n"


def read_file_as_bytes(file_path: str) -> bytes:
    """
    Reads a local file and returns its content as binary bytes.
    Automatically resolves file location across project root, uploads/, and outputs/ directories.
    """
    resolved_path = resolve_file_path(file_path)
    with open(resolved_path, "rb") as f:
        return f.read()


def prepare_input_files(
    files: Union[List[str], Dict[str, Union[str, bytes, Path]]]
) -> Dict[str, bytes]:
    """
    Prepares input files into a key-value byte map (`filename -> bytes`) for sending to the sandbox.
    Also ensures both base filename and any relative subpaths are registered.

    Args:
        files: Either a list of file paths OR a dict mapping filename -> path/bytes.

    Returns:
        Dict[str, bytes]: Filename to raw file binary contents.
    """
    input_files_map: Dict[str, bytes] = {}

    if isinstance(files, list):
        for file_path in files:
            if not file_path:
                continue
            file_name = os.path.basename(str(file_path))
            try:
                content = read_file_as_bytes(file_path)
                input_files_map[file_name] = content
                # If path has a subfolder like uploads/file.csv, provide subpath too
                rel_path = str(file_path).replace("\\", "/")
                if rel_path != file_name:
                    input_files_map[rel_path] = content
            except Exception as e:
                print(f"[prepare_input_files Warning] Could not read file '{file_path}': {e}")
    elif isinstance(files, dict):
        for file_name, content in files.items():
            if isinstance(content, bytes):
                input_files_map[file_name] = content
            elif isinstance(content, str):
                try:
                    input_files_map[file_name] = read_file_as_bytes(content)
                except Exception:
                    # Assume text string content
                    input_files_map[file_name] = content.encode("utf-8")
            else:
                raise TypeError(f"Unsupported content type for file {file_name}: {type(content)}")

    return input_files_map


# ==========================================
# 3. File Output / Receiving Utilities
# ==========================================

def extract_created_files(
    tempdir: str, initial_filenames: set
) -> Dict[str, Dict[str, Any]]:
    """
    Scans the temporary sandbox execution directory for newly created or modified files.

    Args:
        tempdir: Temporary workspace directory path inside the sandbox container.
        initial_filenames: Set of input file relative paths to exclude.

    Returns:
        Dict mapping relative filepath -> file info metadata dict.
    """
    returned_files: Dict[str, Dict[str, Any]] = {}

    for root, _, filenames in os.walk(tempdir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, tempdir).replace("\\", "/")

            if relative_path in initial_filenames:
                continue

            try:
                with open(filepath, "rb") as f:
                    data = f.read()

                mime, _ = mimetypes.guess_type(relative_path)

                returned_files[relative_path] = {
                    "content": data,
                    "size": len(data),
                    "extension": os.path.splitext(relative_path)[1],
                    "mime": mime or "application/octet-stream",
                }
            except Exception as err:
                print(f"[Sandbox File Warning] Error reading {relative_path}: {err}")

    return returned_files


def save_returned_files(
    returned_files: Dict[str, Dict[str, Any]], target_dir: str = "outputs"
) -> List[str]:
    """
    Saves files returned from the cloud sandbox to the local disk.

    Args:
        returned_files: Dict of returned file objects containing 'content'.
        target_dir: Local directory path where files should be saved.

    Returns:
        List[str]: List of local file paths created.
    """
    saved_paths: List[str] = []
    
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    effective_target_dir = target_dir if (target_dir and os.path.isabs(target_dir)) else os.path.join(workspace_root, target_dir or "outputs")

    for filename, info in returned_files.items():
        local_path = os.path.join(effective_target_dir, filename)
        dir_name = os.path.dirname(local_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        content = info.get("content")
        if isinstance(content, bytes):
            with open(local_path, "wb") as f:
                f.write(content)
            saved_paths.append(local_path)

    return saved_paths


# ==========================================
# 4. Stdout, Stderr & Result Management
# ==========================================


def clean_output_stream(stream_text: str, max_chars: int = 2000, truncate_from_start: bool = False) -> str:
    """
    Cleans and formats stdout or stderr output streams, limiting output to max_chars.
    If truncate_from_start is True (default for stderr), truncates from the starting (head) and keeps the end (tail).
    """
    if not stream_text:
        return ""
    text = stream_text.strip()
    if len(text) > max_chars:
        if truncate_from_start:
            return f"... [Output truncated from start, showing last {max_chars} characters limit]\n" + text[-max_chars:]
        return text[:max_chars] + f"\n... [Output truncated at {max_chars} characters limit]"
    return text


def format_execution_result(
    status: bool,
    stdout: str,
    stderr: str,
    execution_time: float,
    files: Optional[Dict[str, Dict[str, Any]]] = None,
    error: str = "",
    max_chars: int = 2000,
) -> Dict[str, Any]:
    """
    Formats raw sandbox execution results into a clean dictionary, slicing stdout/stderr to max 2000 chars.
    Truncates stderr from the start so that final traceback/exception errors at the end are preserved.
    """
    formatted_stdout = clean_output_stream(stdout, max_chars=max_chars, truncate_from_start=False)
    formatted_stderr = clean_output_stream(stderr, max_chars=max_chars, truncate_from_start=True)

    if error:
        cleaned_error = clean_output_stream(error, max_chars=max_chars, truncate_from_start=True)
        if formatted_stderr:
            formatted_stderr = clean_output_stream(f"{formatted_stderr}\n{cleaned_error}", max_chars=max_chars, truncate_from_start=True)
        else:
            formatted_stderr = cleaned_error

    return {
        "status": status,
        "stdout": formatted_stdout,
        "stderr": formatted_stderr,
        "execution_time": round(execution_time, 4),
        "files": files or {},
    }



# ==========================================
# ==========================================
# 5. Sandbox Execution Function Factory
# ==========================================

def _raw_execute_python(code: str, files: Dict[str, bytes]) -> Dict[str, Any]:
    start_time = time.time()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        
        # Write incoming input files into container workspace
        for filename, content in files.items():
            dir_path = os.path.dirname(filename)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filename, "wb") as f:
                f.write(content)

        file_registry = build_file_registry(files)

        def get_file_path(fname: str) -> str:
            if not fname:
                raise ValueError("Filename cannot be empty")
            if fname in file_registry:
                return file_registry[fname]["path"]
            base = os.path.basename(fname)
            if base in file_registry:
                return file_registry[base]["path"]
            for k, v in file_registry.items():
                if k.lower() == fname.lower() or k.lower() == base.lower():
                    return v["path"]
            if os.path.exists(fname):
                return fname
            if os.path.exists(base):
                return base
            raise FileNotFoundError(f"File '{fname}' not found in FILE_REGISTRY: {list(file_registry.keys())}")

        local_globals = {
            "__name__": "__main__",
            "FILE_REGISTRY": file_registry,
            "get_file_path": get_file_path,
        }

        initial_filenames = set(files.keys())
        success = False
        error_str = ""

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, local_globals)
            success = True
        except Exception:
            success = False
            error_str = traceback.format_exc()
        finally:
            os.chdir(cwd)

        # Extract generated files from workspace
        returned_files = extract_created_files(tempdir, initial_filenames)

    execution_time = time.time() - start_time

    return format_execution_result(
        status=success,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        execution_time=execution_time,
        files=returned_files,
        error=error_str,
    )


if HAS_MODAL:
    _default_app = modal.App("analystGPT-sandbox")
    _default_image = (
        modal.Image.debian_slim()
        .pip_install(*DEFAULT_SANDBOX_PACKAGES)
        .add_local_python_source("sandbox_execution")
    )

    @_default_app.function(
        image=_default_image,
        timeout=500,
        memory=2048,
        cpu=2.0,
        serialized=True,
    )
    def global_execute_python_in_sandbox(code: str, files: Dict[str, bytes]) -> Dict[str, Any]:
        return _raw_execute_python(code, files)
else:
    _default_app = None
    _default_image = None
    global_execute_python_in_sandbox = None


def create_sandbox_executor(
    app: Optional[Any] = None,
    image: Optional[Any] = None,
    timeout: int = 60,
    memory: int = 1024,
    cpu: float = 1.0,
):
    """
    Creates and returns a Modal registered function for running Python code in the cloud sandbox.
    """
    if not HAS_MODAL:
        raise ImportError(
            "The 'modal' package is not installed. Please install it using `pip install modal`."
        )

    if app is None and image is None and timeout == 60 and memory == 1024 and cpu == 1.0:
        return global_execute_python_in_sandbox

    if app is None:
        app = get_modal_app()
    if image is None:
        image = build_sandbox_image()

    @app.function(
        image=image,
        timeout=timeout,
        memory=memory,
        cpu=cpu,
        serialized=True,
    )
    def execute_python_in_sandbox(code: str, files: Dict[str, bytes]) -> Dict[str, Any]:
        return _raw_execute_python(code, files)

    return execute_python_in_sandbox


# ==========================================
# 6. Global Singleton & Stateful Sandbox Manager Class
# ==========================================

_GLOBAL_SANDBOX_MANAGER: Optional["ModalSandboxManager"] = None


def get_global_sandbox_manager(
    app_name: str = "analystGPT-sandbox",
    extra_packages: Optional[List[str]] = None,
    timeout: int = 60,
    memory: int = 1024,
    cpu: float = 1.0,
    reset: bool = False,
) -> "ModalSandboxManager":
    """
    Returns the global stateful ModalSandboxManager singleton instance.
    Creates and initializes a new instance if one does not exist or if reset=True.
    """
    global _GLOBAL_SANDBOX_MANAGER
    if reset and _GLOBAL_SANDBOX_MANAGER is not None:
        _GLOBAL_SANDBOX_MANAGER.terminate()
        _GLOBAL_SANDBOX_MANAGER = None

    if _GLOBAL_SANDBOX_MANAGER is None:
        _GLOBAL_SANDBOX_MANAGER = ModalSandboxManager(
            app_name=app_name,
            extra_packages=extra_packages,
            timeout=timeout,
            memory=memory,
            cpu=cpu,
        )
        _GLOBAL_SANDBOX_MANAGER.initialize()

    return _GLOBAL_SANDBOX_MANAGER


def reset_global_sandbox_manager():
    """
    Resets and terminates the active global ModalSandboxManager instance.
    """
    global _GLOBAL_SANDBOX_MANAGER
    if _GLOBAL_SANDBOX_MANAGER is not None:
        try:
            _GLOBAL_SANDBOX_MANAGER.terminate()
        except Exception as e:
            print(f"[Sandbox Warning] Error during global sandbox termination: {e}")
        _GLOBAL_SANDBOX_MANAGER = None


class ModalSandboxManager:
    """
    Stateful high-level manager class for Modal Cloud Sandbox & Python execution.
    Preserves container workspace files, generated charts/CSVs, and Python execution state across calls.
    """

    def __init__(
        self,
        app_name: str = "analystGPT-sandbox",
        extra_packages: Optional[List[str]] = None,
        timeout: int = 60,
        memory: int = 1024,
        cpu: float = 1.0,
    ):
        self.app_name = app_name
        self.timeout = timeout
        self.memory = memory
        self.cpu = cpu
        self.extra_packages = extra_packages or []
        self._app = None
        self._image = None
        self._executor = None
        self._sandbox_dir: Optional[str] = None
        self._known_files: set = set()
        self.file_registry: Dict[str, Dict[str, Any]] = {}
        self._globals: Dict[str, Any] = {"__name__": "__main__"}

    def initialize(self):
        """
        Initializes persistent sandbox workspace directory, Modal App, Image, and Executor.
        """
        if not self._sandbox_dir or not os.path.exists(self._sandbox_dir):
            self._sandbox_dir = tempfile.mkdtemp(prefix="analystgpt_sandbox_")
            self._known_files = set()
            self.file_registry = {}

        if HAS_MODAL:
            try:
                self._app = get_modal_app(self.app_name)
                self._image = build_sandbox_image(extra_pip_packages=self.extra_packages)
                self._executor = create_sandbox_executor(
                    app=self._app,
                    image=self._image,
                    timeout=self.timeout,
                    memory=self.memory,
                    cpu=self.cpu,
                )
            except Exception as err:
                if HAS_LOGFIRE:
                    logfire.warn("Modal app initialization notice", error=str(err))

        if HAS_LOGFIRE:
            logfire.info("Initialized Stateful Sandbox Manager", app_name=self.app_name, workspace=self._sandbox_dir)

    def prepare_files(
        self, files: Union[List[str], Dict[str, Union[str, bytes]]]
    ) -> Dict[str, bytes]:
        """
        Wrapper to package local files or raw data into input bytes for the sandbox.
        """
        prepared = prepare_input_files(files)
        if HAS_LOGFIRE:
            logfire.info("Prepared input files for sandbox", count=len(prepared), filenames=list(prepared.keys()))
        return prepared

    def get_file_path(self, filename: str) -> str:
        """
        Returns the resolved path for a registered file in the sandbox.
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        if filename in self.file_registry:
            return self.file_registry[filename]["path"]
        base = os.path.basename(filename)
        if base in self.file_registry:
            return self.file_registry[base]["path"]
        for k, v in self.file_registry.items():
            if k.lower() == filename.lower() or k.lower() == base.lower():
                return v["path"]
        if self._sandbox_dir and os.path.exists(os.path.join(self._sandbox_dir, filename)):
            return filename
        if self._sandbox_dir and os.path.exists(os.path.join(self._sandbox_dir, base)):
            return base
        raise FileNotFoundError(f"File '{filename}' not found in FILE_REGISTRY. Available files: {list(self.file_registry.keys())}")

    def run_code(
        self, code: str, files: Optional[Union[List[str], Dict[str, Union[str, bytes]]]] = None
    ) -> Dict[str, Any]:
        """
        Executes Python code inside the persistent sandbox environment.
        Retains generated files, uploaded datasets, and executed variable state across consecutive calls.
        """
        if not self._sandbox_dir or not os.path.exists(self._sandbox_dir):
            self.initialize()

        if isinstance(files, list):
            input_files = prepare_input_files(files)
        elif isinstance(files, dict):
            input_files = files
        else:
            input_files = {}

        # 1. Update file registry
        new_registry = build_file_registry(files if files else input_files)
        self.file_registry.update(new_registry)

        # 2. Write any incoming input files into persistent sandbox workspace
        for filename, content in input_files.items():
            file_path = os.path.join(self._sandbox_dir, filename)
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as f:
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
                    f.write(content)

        # 3. Inject FILE_REGISTRY and get_file_path into global state
        self._globals["FILE_REGISTRY"] = self.file_registry
        self._globals["get_file_path"] = self.get_file_path

        def _execute():
            start_t = time.time()
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            cwd = os.getcwd()

            # Record files in directory before execution
            initial_files = set()
            for root, _, filenames in os.walk(self._sandbox_dir):
                for fname in filenames:
                    rel = os.path.relpath(os.path.join(root, fname), self._sandbox_dir).replace("\\", "/")
                    initial_files.add(rel)

            success = False
            error_str = ""

            try:
                os.chdir(self._sandbox_dir)
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code, self._globals)
                success = True
            except Exception:
                success = False
                error_str = traceback.format_exc()
            finally:
                os.chdir(cwd)

            execution_time = time.time() - start_t

            # Extract newly generated or modified files
            returned_files = extract_created_files(self._sandbox_dir, self._known_files.union(set(input_files.keys())))

            # Update set of known files in sandbox
            for root, _, filenames in os.walk(self._sandbox_dir):
                for fname in filenames:
                    rel = os.path.relpath(os.path.join(root, fname), self._sandbox_dir).replace("\\", "/")
                    self._known_files.add(rel)

            result = format_execution_result(
                status=success,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                execution_time=execution_time,
                files=returned_files,
                error=error_str,
            )

            status_str = "SUCCESS" if success else "FAILED"
            if HAS_LOGFIRE:
                logfire.info(
                    f"Stateful Sandbox Execution completed: {status_str}",
                    status=success,
                    elapsed_seconds=round(execution_time, 2),
                    stdout_length=len(result.get("stdout", "")),
                    stderr_length=len(result.get("stderr", "")),
                    returned_files=list(returned_files.keys()),
                )
            else:
                print(f"[Stateful Sandbox] Execution completed in {round(execution_time, 2)}s. Status: {status_str}", flush=True)

            return result

        if HAS_LOGFIRE:
            with logfire.span("Stateful Sandbox execution", app_name=self.app_name, input_files=list(input_files.keys())):
                return _execute()
        else:
            return _execute()

    def save_outputs(
        self, execution_result: Dict[str, Any], output_dir: str = "outputs"
    ) -> List[str]:
        """
        Saves returned sandbox files locally to the specified output directory.
        """
        returned_files = execution_result.get("files", {})
        saved_paths = save_returned_files(returned_files, target_dir=output_dir)
        if HAS_LOGFIRE:
            logfire.info("Saved sandbox output files", count=len(saved_paths), output_dir=output_dir, saved_paths=saved_paths)
        return saved_paths

    def terminate(self):
        """
        Terminates and cleans up the persistent sandbox workspace.
        """
        if self._sandbox_dir and os.path.exists(self._sandbox_dir):
            try:
                import shutil
                shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            except Exception as e:
                print(f"[Sandbox Warning] Error removing temp workspace {self._sandbox_dir}: {e}")
        self._sandbox_dir = None
        self._known_files = set()
        self._globals = {"__name__": "__main__"}
        if HAS_LOGFIRE:
            logfire.info("Terminated Stateful Sandbox Manager")


