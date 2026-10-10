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
import sys
import json
import time
import ctypes
import threading
import mimetypes
import tempfile
import contextlib
import traceback
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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

# Hard execution cap for any single sandboxed code call (seconds). Matches the
# Modal @app.function(timeout=...) on the remote executor; long-running work
# (e.g. large hyperparameter searches) is cut off at the container layer.
SANDBOX_TIMEOUT_S = 500


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
    try:
        image = image.add_local_python_source("sandbox_execution")
    except Exception:
        pass

    return image


# ==========================================
# 2. File Registry & File Input Utilities
# ==========================================

def resolve_file_path(file_path: Union[str, Path], extra_dirs: Optional[List[str]] = None) -> str:
    """
    Robustly resolves a file path across:
    1. Direct absolute/relative path
    2. Optional extra directories (active output_dir, sandbox_dir, etc.)
    3. Project workspace root
    4. uploads/ folder
    5. outputs/ folder and run_* subfolders
    6. Case-insensitive filename match in uploads/
    """
    if not file_path:
        raise FileNotFoundError("Empty file path provided.")

    p_str = str(file_path).strip()
    if os.path.exists(p_str) and not os.path.isdir(p_str):
        return os.path.abspath(p_str)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_name = os.path.basename(p_str)

    candidates = []
    if extra_dirs:
        for ed in extra_dirs:
            if ed and os.path.exists(ed):
                candidates.append(os.path.join(ed, p_str))
                candidates.append(os.path.join(ed, base_name))

    # Also check active sandbox dir if global manager is initialized
    try:
        if _GLOBAL_SANDBOX_MANAGER and _GLOBAL_SANDBOX_MANAGER._sandbox_dir:
            sb = _GLOBAL_SANDBOX_MANAGER._sandbox_dir
            if os.path.exists(sb):
                candidates.append(os.path.join(sb, p_str))
                candidates.append(os.path.join(sb, base_name))
    except Exception:
        pass

    candidates.extend([
        os.path.join(workspace_root, p_str),
        os.path.join(workspace_root, "uploads", base_name),
        os.path.join(workspace_root, "uploads", p_str),
        os.path.join(workspace_root, "outputs", base_name),
        os.path.join(workspace_root, base_name),
        os.path.join(os.getcwd(), "uploads", base_name),
        os.path.join(os.getcwd(), "outputs", base_name),
        os.path.join(os.getcwd(), "..", "uploads", base_name),
        os.path.join(os.getcwd(), "..", "outputs", base_name),
    ])

    # Check subfolders in outputs/ (e.g. outputs/run_*/base_name) - most recent first
    outputs_root = os.path.join(workspace_root, "outputs")
    if os.path.exists(outputs_root):
        try:
            subdirs = sorted([d for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))], reverse=True)
            for sub in subdirs:
                candidates.append(os.path.join(outputs_root, sub, base_name))
        except Exception:
            pass

    for cand in candidates:
        if os.path.exists(cand) and not os.path.isdir(cand) and os.path.getsize(cand) > 0:
            return os.path.abspath(cand)

    # Secondary check for candidates that exist even if size is 0 (as fallback)
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

    raise FileNotFoundError(f"File not found: '{file_path}'. Checked workspace, outputs subdirectories, and uploads folder.")


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
        "- Datasets in FILE_REGISTRY are already present in the workspace: in execute_python, supply only the 'code' parameter (do not pass 'files').",
    ]
    return "\n".join(lines) + "\n"


def read_file_as_bytes(file_path: str, extra_dirs: Optional[List[str]] = None) -> bytes:
    """
    Reads a local file and returns its content as binary bytes.
    Automatically resolves file location across project root, uploads/, outputs/, and extra directories.
    """
    resolved_path = resolve_file_path(file_path, extra_dirs=extra_dirs)
    with open(resolved_path, "rb") as f:
        return f.read()


def prepare_input_files(
    files: Union[List[str], Dict[str, Union[str, bytes, Path]]],
    output_dir: Optional[str] = None,
    sandbox_dir: Optional[str] = None,
) -> Dict[str, bytes]:
    """
    Prepares input files into a key-value byte map (`filename -> bytes`) for sending to the sandbox.
    Also ensures both base filename and any relative subpaths are registered.

    Args:
        files: Either a list of file paths OR a dict mapping filename -> path/bytes.
        output_dir: Optional active output directory to look up existing files.
        sandbox_dir: Optional active sandbox directory to check for already-existing files.

    Returns:
        Dict[str, bytes]: Filename to raw file binary contents.
    """
    input_files_map: Dict[str, bytes] = {}
    extra_dirs = [d for d in [output_dir, sandbox_dir] if d and os.path.exists(d)]

    if isinstance(files, list):
        for file_path in files:
            if not file_path:
                continue
            file_name = os.path.basename(str(file_path))
            try:
                content = read_file_as_bytes(file_path, extra_dirs=extra_dirs)
                input_files_map[file_name] = content
                # If path has a subfolder like uploads/file.csv, provide subpath too
                rel_path = str(file_path).replace("\\", "/")
                if rel_path != file_name:
                    input_files_map[rel_path] = content
            except Exception as e:
                print(f"[prepare_input_files Warning] Could not read file '{file_path}': {e}")
    elif isinstance(files, dict):
        for file_name, content in files.items():
            if not file_name:
                continue
            base_fname = os.path.basename(str(file_name))
            if isinstance(content, bytes):
                input_files_map[file_name] = content
            elif content is None or (isinstance(content, str) and not content.strip()):
                # Placeholder or empty string from model (e.g. {"creditcard.csv": None} or {"cleaned_dataset.csv": ""})
                # DO NOT encode as b"". Attempt to resolve the real file on disk.
                try:
                    resolved_content = read_file_as_bytes(file_name, extra_dirs=extra_dirs)
                    if resolved_content:
                        input_files_map[file_name] = resolved_content
                except Exception:
                    # If the file already exists in sandbox_dir, do NOT supply empty bytes to truncate it!
                    if sandbox_dir and os.path.exists(os.path.join(sandbox_dir, base_fname)):
                        continue
                    print(f"[prepare_input_files Warning] Skipping empty/unresolvable file placeholder: '{file_name}'")
            elif isinstance(content, str):
                # Could be a path, filename, or actual text content
                resolved_content = None
                try:
                    resolved_content = read_file_as_bytes(content, extra_dirs=extra_dirs)
                except Exception:
                    try:
                        resolved_content = read_file_as_bytes(file_name, extra_dirs=extra_dirs)
                    except Exception:
                        pass

                if resolved_content is not None and len(resolved_content) > 0:
                    input_files_map[file_name] = resolved_content
                else:
                    # Check if string looks like an unresolvable path or placeholder
                    str_clean = content.strip()
                    is_placeholder = (
                        str_clean == file_name.strip()
                        or "content would be loaded" in str_clean.lower()
                        or str_clean.endswith(".csv")
                        or str_clean.endswith(".json")
                        or str_clean.endswith(".parquet")
                    )
                    if is_placeholder:
                        # If file already in sandbox, preserve it
                        if sandbox_dir and os.path.exists(os.path.join(sandbox_dir, base_fname)):
                            continue
                        print(f"[prepare_input_files Warning] Unresolved path/placeholder string for '{file_name}': '{content}'. Skipping.")
                        continue

                    # Treat as legitimate text string content
                    input_files_map[file_name] = content.encode("utf-8")
            else:
                print(f"[prepare_input_files Warning] Unsupported content type for file {file_name}: {type(content)}. Skipping.")

    return input_files_map


# ==========================================
# 3. File Output / Receiving Utilities
# ==========================================

def snapshot_directory_files(directory: str) -> Dict[str, Tuple[int, float, str]]:
    """
    Snapshots file metadata (size, mtime, md5 hash) for all files in a directory.
    Enables detecting modified files in addition to newly created files.
    """
    snapshot: Dict[str, Tuple[int, float, str]] = {}
    if not directory or not os.path.exists(directory):
        return snapshot

    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if fname == "__sandbox_session_state__.pkl" or fname.endswith(".pyc") or "__pycache__" in root:
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, directory).replace("\\", "/")
            try:
                stat = os.stat(fpath)
                if stat.st_size < 25 * 1024 * 1024:
                    with open(fpath, "rb") as f:
                        fhash = hashlib.md5(f.read()).hexdigest()
                else:
                    fhash = f"{stat.st_size}_{stat.st_mtime}"
                snapshot[rel] = (stat.st_size, stat.st_mtime, fhash)
            except Exception:
                pass
    return snapshot


def extract_created_files(
    tempdir: str,
    initial_filenames: Union[set, Dict[str, Any]],
    initial_snapshots: Optional[Dict[str, Tuple[int, float, str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Scans the temporary sandbox execution directory for newly created or modified files.

    Args:
        tempdir: Temporary workspace directory path inside the sandbox container.
        initial_filenames: Set of input file relative paths (or snapshot dict).
        initial_snapshots: Optional snapshot dict mapping rel_path -> (size, mtime, hash).

    Returns:
        Dict mapping relative filepath -> file info metadata dict.
    """
    returned_files: Dict[str, Dict[str, Any]] = {}

    snapshots = initial_snapshots
    if snapshots is None and isinstance(initial_filenames, dict):
        snapshots = initial_filenames
        initial_names = set(initial_filenames.keys())
    elif isinstance(initial_filenames, (set, list, tuple)):
        initial_names = set(initial_filenames)
    else:
        initial_names = set()

    for root, _, filenames in os.walk(tempdir):
        for filename in filenames:
            if filename == "__sandbox_session_state__.pkl" or filename.endswith(".pyc") or "__pycache__" in root:
                continue
            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, tempdir).replace("\\", "/")

            is_initial = relative_path in initial_names

            try:
                stat = os.stat(filepath)
                # Check if an initial file was modified
                if is_initial:
                    if snapshots and relative_path in snapshots:
                        old_size, old_mtime, old_hash = snapshots[relative_path]
                        if stat.st_size < 25 * 1024 * 1024:
                            with open(filepath, "rb") as f:
                                current_data = f.read()
                            current_hash = hashlib.md5(current_data).hexdigest()
                        else:
                            current_data = None
                            current_hash = f"{stat.st_size}_{stat.st_mtime}"

                        # Skip if UNMODIFIED
                        if stat.st_size == old_size and current_hash == old_hash:
                            continue

                        # If modified, retain data
                        if current_data is None:
                            with open(filepath, "rb") as f:
                                current_data = f.read()
                        data = current_data
                    else:
                        continue
                else:
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
        if filename == "__sandbox_session_state__.pkl":
            continue
        local_path = os.path.join(effective_target_dir, filename)
        dir_name = os.path.dirname(local_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        content = info.get("content")
        if isinstance(content, bytes):
            with open(local_path, "wb") as f:
                f.write(content)
            saved_paths.append(local_path)

            # Files created inside subdirectories (e.g. tests/test.csv) are also
            # saved flat into the main output directory so every artifact is
            # retrievable from one location.
            if os.sep in filename or "/" in filename:
                flat_name = filename.replace("\\", "/").split("/")[-1]
                flat_path = os.path.join(effective_target_dir, flat_name)
                if os.path.abspath(flat_path) != os.path.abspath(local_path):
                    try:
                        with open(flat_path, "wb") as f:
                            f.write(content)
                        saved_paths.append(flat_path)
                    except Exception as err:
                        print(f"[Sandbox File Warning] Error saving flat copy of {filename}: {err}", flush=True)

    return saved_paths


# ==========================================
# 4. Stdout, Stderr & Result Management
# ==========================================


def clean_output_stream(stream_text: str, max_chars: int = 4000, truncate_from_start: bool = False) -> str:
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
    max_stdout_chars: int = 4000,
    max_stderr_chars: int = 2000,
) -> Dict[str, Any]:
    """
    Formats raw sandbox execution results into a clean dictionary, slicing stdout to max 4000 chars and stderr to max 2000 chars.
    Truncates stderr from the start so that final traceback/exception errors at the end are preserved.
    """
    formatted_stdout = clean_output_stream(stdout, max_chars=max_stdout_chars, truncate_from_start=False)
    formatted_stderr = clean_output_stream(stderr, max_chars=max_stderr_chars, truncate_from_start=True)

    if error:
        cleaned_error = clean_output_stream(error, max_chars=max_stderr_chars, truncate_from_start=True)
        if formatted_stderr:
            formatted_stderr = clean_output_stream(f"{formatted_stderr}\n{cleaned_error}", max_chars=max_stderr_chars, truncate_from_start=True)
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
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
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

        # Restore persistent session state if available
        session_file = "__sandbox_session_state__.pkl"
        if os.path.exists(session_file):
            try:
                import joblib
                with open(session_file, "rb") as f:
                    saved_globals = joblib.load(f)
                if isinstance(saved_globals, dict):
                    local_globals.update(saved_globals)
            except Exception:
                try:
                    import pickle
                    with open(session_file, "rb") as f:
                        saved_globals = pickle.load(f)
                    if isinstance(saved_globals, dict):
                        local_globals.update(saved_globals)
                except Exception:
                    pass

        initial_snapshots = snapshot_directory_files(tempdir)
        initial_filenames = set(files.keys()).union(set(initial_snapshots.keys()))
        success = False
        error_str = ""

        # CRITICAL FIX: Set matplotlib to use 'Agg' backend (non-interactive) BEFORE executing user code.
        # This prevents Tkinter threading errors ("main thread is not in main loop") that occur when
        # matplotlib tries to use a GUI backend in a background thread (Modal sandbox).
        # We inject this setup code at the beginning of the user's code execution context.
        matplotlib_setup_code = (
            "import os\n"
            "os.environ['MPLBACKEND'] = 'Agg'\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
        )

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                # First execute matplotlib backend setup
                exec(matplotlib_setup_code, local_globals)
                # Then execute user code
                exec(code, local_globals)
            success = True
        except Exception:
            success = False
            error_str = traceback.format_exc()
        finally:
            os.chdir(cwd)

        # Persist session state across consecutive calls
        if success:
            try:
                import joblib
                persisted = {}
                skip_keys = {
                    "__name__", "__doc__", "__package__", "__loader__", "__spec__",
                    "__builtins__", "FILE_REGISTRY", "get_file_path"
                }
                for k, v in local_globals.items():
                    if k in skip_keys or k.startswith("__"):
                        continue
                    try:
                        joblib.dumps(v)
                        persisted[k] = v
                    except Exception:
                        pass
                if persisted:
                    with open(os.path.join(tempdir, session_file), "wb") as f:
                        joblib.dump(persisted, f)
            except Exception:
                pass

        # Extract generated or modified files from workspace
        returned_files = extract_created_files(tempdir, initial_filenames, initial_snapshots=initial_snapshots)

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
        # Per-invocation cap: kills only this function call, never the caller's
        # persistent sandbox state.
        timeout=SANDBOX_TIMEOUT_S,
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
    timeout: int = SANDBOX_TIMEOUT_S,
    memory: int = 2048,
    cpu: float = 2.0,
):
    """
    Creates and returns a Modal registered function for running Python code in the cloud sandbox.
    """
    if not HAS_MODAL:
        raise ImportError(
            "The 'modal' package is not installed. Please install it using `pip install modal`."
        )

    if app is None and image is None and timeout == SANDBOX_TIMEOUT_S and memory == 2048 and cpu == 2.0:
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
    timeout: int = SANDBOX_TIMEOUT_S,
    memory: int = 2048,
    cpu: float = 2.0,
    backend: Optional[str] = None,
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
            backend=backend,
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
    Supports running directly on Modal Cloud (`backend='modal'`) with automatic local fallback.
    """

    def __init__(
        self,
        app_name: str = "analystGPT-sandbox",
        extra_packages: Optional[List[str]] = None,
        timeout: int = SANDBOX_TIMEOUT_S,
        memory: int = 2048,
        cpu: float = 2.0,
        backend: Optional[str] = None,
    ):
        self.app_name = app_name
        self.timeout = timeout
        self.memory = memory
        self.cpu = cpu
        self.extra_packages = extra_packages or []
        self.backend = (backend or os.environ.get("SANDBOX_BACKEND", "modal")).strip().lower()
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

        if self.backend == "modal" and HAS_MODAL:
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
                print(f"[Modal Sandbox Notice] Initialization error: {err}", flush=True)
                if HAS_LOGFIRE:
                    logfire.warn("Modal app initialization notice", error=str(err))

        if HAS_LOGFIRE:
            logfire.info(
                "Initialized Stateful Sandbox Manager",
                app_name=self.app_name,
                backend=self.backend,
                workspace=self._sandbox_dir,
            )

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
        When backend='modal', routes execution to Modal Cloud with cloud containers.
        Falls back to isolated local execution if configured or if remote execution is unavailable.
        """
        if not self._sandbox_dir or not os.path.exists(self._sandbox_dir):
            self.initialize()

        if files:
            input_files = prepare_input_files(files, sandbox_dir=self._sandbox_dir)
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

            # SAFETY GUARD: Never write empty/zero-byte content into sandbox workspace!
            content_len = len(content) if isinstance(content, (bytes, str)) else 0
            if content_len == 0:
                print(f"[Sandbox Safety] Skipping write of 0-byte input content for '{filename}'.", flush=True)
                continue

            with open(file_path, "wb") as f:
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
                    f.write(content)

        # 2b. Sync existing non-empty files from registry/host if missing or 0 bytes in sandbox
        for reg_name, reg_meta in list(self.file_registry.items()):
            sb_path = os.path.join(self._sandbox_dir, reg_name)
            if not os.path.exists(sb_path) or os.path.getsize(sb_path) == 0:
                cand_paths = []
                if isinstance(reg_meta, dict) and reg_meta.get("path"):
                    cand_paths.append(reg_meta["path"])
                try:
                    cand_paths.append(resolve_file_path(reg_name))
                except Exception:
                    pass

                for cp in cand_paths:
                    if cp and os.path.exists(cp) and not os.path.isdir(cp) and os.path.getsize(cp) > 0:
                        try:
                            import shutil
                            os.makedirs(os.path.dirname(sb_path), exist_ok=True)
                            shutil.copy2(cp, sb_path)
                            print(f"[Sandbox Sync] Restored '{reg_name}' from host ({os.path.getsize(cp)} bytes) into sandbox.", flush=True)
                            break
                        except Exception:
                            pass

        # 3. Modal Cloud Execution
        if self.backend == "modal":
            if not HAS_MODAL:
                print("[Modal Sandbox] 'modal' package not installed. Falling back to local execution...", flush=True)
                return self._execute_local(code, input_files)

            if self._executor is None or self._app is None:
                self.initialize()

            if self._executor is None or self._app is None:
                print("[Modal Sandbox] Modal executor not initialized. Falling back to local execution...", flush=True)
                return self._execute_local(code, input_files)

            # Gather all workspace files to sync with Modal Cloud container
            all_files: Dict[str, bytes] = {}
            for root, _, filenames in os.walk(self._sandbox_dir):
                for fname in filenames:
                    rel = os.path.relpath(os.path.join(root, fname), self._sandbox_dir).replace("\\", "/")
                    full_p = os.path.join(root, fname)
                    try:
                        with open(full_p, "rb") as f:
                            all_files[rel] = f.read()
                    except Exception:
                        pass

            try:
                print(f"[Modal Sandbox] Executing code remotely on Modal Cloud (app: {self.app_name}, files: {len(all_files)})...", flush=True)
                start_modal_t = time.time()
                with self._app.run():
                    result = self._executor.remote(code, all_files)
                elapsed_modal = time.time() - start_modal_t
                status_str = "SUCCESS" if result.get("status") else "FAILED"
                print(f"[Modal Sandbox] Remote execution finished in {round(elapsed_modal, 2)}s. Status: {status_str}", flush=True)

                # Persist any returned files from remote container into host sandbox workspace
                returned_files = result.get("files", {})
                for filename, file_info in returned_files.items():
                    content = file_info.get("content")
                    if content is not None:
                        target_path = os.path.join(self._sandbox_dir, filename)
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with open(target_path, "wb") as f:
                            f.write(content if isinstance(content, bytes) else content.encode("utf-8"))
                        self._known_files.add(filename)

                # Update registry with returned files
                new_files_reg = build_file_registry(list(returned_files.keys()))
                self.file_registry.update(new_files_reg)

                if HAS_LOGFIRE:
                    logfire.info(
                        "Modal Cloud Sandbox execution completed",
                        status=result.get("status", False),
                        elapsed=elapsed_modal,
                        returned_files=list(returned_files.keys()),
                    )
                return result

            except Exception as modal_err:
                print(f"[Modal Sandbox Error] Remote execution on Modal Cloud failed: {modal_err}", flush=True)
                if HAS_LOGFIRE:
                    logfire.warn("Modal Cloud execution error", error=str(modal_err))
                if os.environ.get("SANDBOX_FALLBACK_TO_LOCAL", "true").lower() in ("true", "1", "yes"):
                    print("[Modal Sandbox] Falling back to local sandbox execution...", flush=True)
                    return self._execute_local(code, input_files)
                else:
                    return format_execution_result(
                        status=False,
                        stdout="",
                        stderr=f"ModalCloudExecutionError: {modal_err}",
                        execution_time=0.0,
                        error=str(modal_err),
                    )

        # 4. Local Execution
        return self._execute_local(code, input_files)

    def _execute_local(self, code: str, input_files: Dict[str, bytes]) -> Dict[str, Any]:
        """
        Executes Python code locally in an isolated thread worker.
        """
        # Inject FILE_REGISTRY and get_file_path into global state
        self._globals["FILE_REGISTRY"] = self.file_registry
        self._globals["get_file_path"] = self.get_file_path

        def _execute():
            start_t = time.time()
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            cwd = os.getcwd()

            # Record files in directory before execution
            initial_snapshots = snapshot_directory_files(self._sandbox_dir)
            initial_files = set(initial_snapshots.keys())

            success = False
            error_str = ""
            timed_out = False

            matplotlib_setup_code = (
                "import os\n"
                "os.environ['MPLBACKEND'] = 'Agg'\n"
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
            )

            exec_holder: Dict[str, Any] = {"error": None}

            def _run_user_code() -> None:
                try:
                    exec(matplotlib_setup_code, self._globals)
                    exec(code, self._globals)
                except BaseException:
                    exec_holder["error"] = traceback.format_exc()

            real_stdout, real_stderr = sys.stdout, sys.stderr
            try:
                os.chdir(self._sandbox_dir)
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                worker = threading.Thread(target=_run_user_code, daemon=True)
                worker.start()
                worker.join(self.timeout)
                if worker.is_alive():
                    timed_out = True
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(worker.ident), ctypes.py_object(SystemExit)
                    )
                    worker.join(min(30, self.timeout))
                    error_str = (
                        f"SandboxExecutionTimeout: code execution exceeded the "
                        f"{self.timeout}s limit and was terminated. Reduce the workload "
                        f"(fewer hyperparameter combinations, smaller n_iter/cv, fewer "
                        f"estimators) and split long computations across tool calls."
                    )
                    if HAS_LOGFIRE:
                        logfire.warn(
                            "Stateful Sandbox execution TIMED OUT (call interrupted, sandbox kept alive)",
                            timeout_s=self.timeout,
                            app_name=self.app_name,
                        )
                elif exec_holder["error"] is not None:
                    error_str = exec_holder["error"]
                else:
                    success = True
            finally:
                sys.stdout, sys.stderr = real_stdout, real_stderr
                os.chdir(cwd)

            execution_time = time.time() - start_t

            # Extract newly generated or modified files
            all_known = self._known_files.union(set(input_files.keys())).union(initial_files)
            returned_files = extract_created_files(
                self._sandbox_dir,
                initial_filenames=all_known,
                initial_snapshots=initial_snapshots,
            )

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


