"""
Unit tests for sandbox file preservation and zero-byte truncation prevention.
Verifies that dummy/empty files dictionaries from LLMs (e.g. nemotron-3.5-lightning-30b)
do not wipe existing artifacts in the sandbox workspace.
"""

import os
import tempfile
import pytest
from sandbox_execution.utils import (
    prepare_input_files,
    ModalSandboxManager,
    build_file_registry,
)
from sandbox_execution.tool_manager import ToolManager, EXECUTE_PYTHON_TOOL_SPEC


def test_prepare_input_files_skips_empty_string_when_in_sandbox():
    """Empty string content for a file already in the sandbox should be skipped, not encoded as b''."""
    with tempfile.TemporaryDirectory() as temp_sandbox:
        target_file = os.path.join(temp_sandbox, "cleaned_dataset.csv")
        with open(target_file, "wb") as f:
            f.write(b"col1,col2\n1,2\n3,4\n")

        # LLM passes empty placeholder
        files_dict = {"cleaned_dataset.csv": ""}
        result = prepare_input_files(files_dict, sandbox_dir=temp_sandbox)

        # Should NOT map "cleaned_dataset.csv" to b""
        assert "cleaned_dataset.csv" not in result or result["cleaned_dataset.csv"] != b""


def test_prepare_input_files_resolves_valid_file_from_extra_dirs():
    """If an empty placeholder is given but the real file exists in output_dir, it should read it."""
    with tempfile.TemporaryDirectory() as temp_out:
        real_file = os.path.join(temp_out, "cleaned_dataset.csv")
        content = b"header1,header2\nval1,val2\n"
        with open(real_file, "wb") as f:
            f.write(content)

        files_dict = {"cleaned_dataset.csv": ""}
        result = prepare_input_files(files_dict, output_dir=temp_out)

        assert "cleaned_dataset.csv" in result
        assert result["cleaned_dataset.csv"] == content


def test_tool_manager_execute_tool_call_strips_empty_files():
    """ToolManager should strip empty placeholder 'files' so execute_python is called safely."""
    executed_args = {}

    def mock_execute_python(**kwargs):
        executed_args.update(kwargs)
        return {"success": True, "stdout": "done"}

    tm = ToolManager(initial_tools={"execute_python": mock_execute_python})

    tool_call = {
        "id": "call_123",
        "function": {
            "name": "execute_python",
            "arguments": '{"code": "import pandas as pd", "files": {"cleaned_dataset.csv": ""}}'
        }
    }

    result = tm.execute_tool_call(tool_call)
    assert result["success"] is True
    # 'files' should have been popped from kwargs because it only contained an empty placeholder
    assert "files" not in executed_args


def test_modal_sandbox_manager_preserves_existing_file():
    """ModalSandboxManager.run_code should never overwrite a non-empty sandbox file with 0 bytes."""
    with tempfile.TemporaryDirectory() as temp_sb:
        # Pre-create non-empty file in sandbox
        existing_csv = os.path.join(temp_sb, "cleaned_dataset.csv")
        original_data = b"id,name,value\n1,alpha,100\n"
        with open(existing_csv, "wb") as f:
            f.write(original_data)

        mgr = ModalSandboxManager()
        mgr._sandbox_dir = temp_sb
        mgr.backend = "local"
        mgr._initialized = True

        # Run code with empty placeholder in files
        code = "print('hello from sandbox')"
        mgr.run_code(code, files={"cleaned_dataset.csv": ""})

        # File must NOT be 0 bytes
        assert os.path.exists(existing_csv)
        assert os.path.getsize(existing_csv) == len(original_data)
        with open(existing_csv, "rb") as f:
            assert f.read() == original_data
