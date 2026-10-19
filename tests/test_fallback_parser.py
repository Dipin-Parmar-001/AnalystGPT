import os
import sys
import json

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from agents.fallback_parser import (
    extract_code_from_content,
    extract_action_code_from_content,
    unwrap_envelope,
    create_synthetic_execute_tool_call,
    SyntheticToolCall
)

from sandbox_execution.tool_manager import ToolManager

def test_fallback_parser():
    # 1. Test exact user's sample payload
    user_sample = {
      "status": 200,
      "headers": {
        "x-portkey-provider": "google"
      },
      "body": {
        "id": "chatcmpl-gHZk92I8ctUcx5ETQHMoI5CEaFzRw",
        "object": "chat.completion",
        "choices": [
          {
            "message": {
              "role": "assistant",
              "content": "Malformed function call: import pandas as pd\nimport numpy as np\n\nprint('Testing sample extraction')\n"
            },
            "index": 0,
            "finish_reason": "stop"
          }
        ]
      }
    }

    # Unwrap envelope
    unwrapped = unwrap_envelope(user_sample)
    assert "Malformed function call:" in unwrapped, f"Unwrap failed: {unwrapped}"

    # Extract code from envelope directly or string
    code1 = extract_code_from_content(user_sample)
    assert code1 is not None, "Failed to extract code from envelope dict"
    assert "import pandas as pd" in code1, f"Extracted code missing expected content: {code1}"
    assert not code1.startswith("Malformed function call"), "Failed to strip prefix"

    # 2. Test string format with Malformed function call:
    raw_str = "Malformed function call: import pandas as pd\nimport sklearn\nprint('hello')"
    code2 = extract_code_from_content(raw_str)
    assert code2 == "import pandas as pd\nimport sklearn\nprint('hello')", f"Got: {code2}"

    # 3. Plain content with an explicitly python-fenced block IS action code
    #    (even with a short lead-in and even if the snippet is small)
    md_str = "Here is the code:\n```python\nimport pandas as pd\ndf = pd.DataFrame()\n```"
    code3 = extract_action_code_from_content(md_str)
    assert code3 == "import pandas as pd\ndf = pd.DataFrame()", f"Got: {code3}"

    # 3a. Small snippet embedded in prose is still executed
    snippet = "Let me verify the output quickly:\n```python\nimport pandas as pd\ndf = pd.read_csv('x.csv')\nprint(df.shape)\n```\nThat check confirms the dataset."
    code3a = extract_action_code_from_content(snippet)
    assert code3a is not None and "pd.read_csv" in code3a, f"Small snippet not extracted: {code3a}"

    # 3b. A markdown REPORT without a python fence is a non-execution response
    report = (
        "## Data Inspection & Cleaning - Task Complete\n\n### Summary of Findings\n\n"
        "**Dataset Overview:**\n- **Source:** `Telco_customer_churn.csv`\n- Initial Shape: 7,043 rows\n\n"
        "| Column | Original | Cleaned |\n|--------|----------|---------|\n| Total Charges | object | float |\n\n"
        "### Artifacts Generated\n1. **`cleaned_dataset.csv`**\n"
    )
    assert extract_action_code_from_content(report) is None, "Markdown report must not be executed as code"

    # 3c. JSON summary content is never executed as code
    json_content = '{"status": "completed", "summary": "task.status = \\"completed\\"", "code_sample": "print(1)"}'
    assert extract_action_code_from_content(json_content) is None, "JSON summary must not be executed as code"

    # 3d. Raw python script in content (no fence) is executed
    raw_script = "import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('data.csv')\ndf.to_csv('cleaned_dataset.csv', index=False)"
    code3d = extract_action_code_from_content(raw_script)
    assert code3d == raw_script, f"Raw script not extracted: {code3d}"

    # 3e. Markdown report that merely QUOTES code in a fence still executes the
    #     fence (python-labelled block is always an execution candidate)
    report_with_code = report + "\nFor reference, the cleaning logic was:\n```python\ndf = pd.read_csv('raw.csv')\ndf.to_csv('cleaned_dataset.csv')\n```"
    code3e = extract_action_code_from_content(report_with_code)
    assert code3e is not None and "to_csv" in code3e, f"Fenced block in report not extracted: {code3e}"

    # 3f. Provider-flagged malformed function call with a fenced block is recovered
    malformed_md = "Malformed function call: ```python\nimport pandas as pd\ndf = pd.DataFrame()\n```"
    code3f = extract_code_from_content(malformed_md)
    assert code3f is not None and "import pandas as pd" in code3f, f"Malformed call not recovered: {code3f}"

    # 3d. Structural classification via interpret_response
    from agents.fallback_parser import interpret_response
    stop_response = unwrap_envelope(user_sample)
    assert "Malformed function call:" in stop_response, f"Unwrap failed: {stop_response}"

    class FakeTC:
        class function:
            name = "execute_python"
            arguments = '{"code": "print(1)"}'
        id = "call-1"

    class FakeMsg:
        content = None
        tool_calls = [FakeTC()]

    interp = interpret_response(FakeMsg())
    assert interp["mode"] == "tool_calls" and interp["tool_calls"], "tool_calls response misclassified"

    class FakeMsgText:
        content = "## Task Complete\ntask.status = 'completed'"
        tool_calls = None

    interp2 = interpret_response(FakeMsgText())
    assert interp2["mode"] == "final_text", "plain summary misclassified as execution"

    # 4. Test SyntheticToolCall creation and execution in ToolManager
    synthetic_tc = create_synthetic_execute_tool_call(code2, "test_call_1")
    assert synthetic_tc.id == "test_call_1"
    assert synthetic_tc.function.name == "execute_python"
    assert "import pandas as pd" in synthetic_tc.function.arguments

    # ToolManager execution with synthetic tool call
    tm = ToolManager()
    # Register mock execute_python
    executed = []
    tm.register_tool("execute_python", lambda code, **kwargs: executed.append(code) or {"success": True, "stdout": "ok"})
    res = tm.execute_tool_call(synthetic_tc)
    assert res.get("success") is True, f"Tool execution failed: {res}"
    assert len(executed) == 1 and "import pandas as pd" in executed[0]

    # 5. Test ToolManager with malformed raw_args in tool_call
    malformed_tc = {
        "id": "bad_args_call",
        "type": "function",
        "function": {
            "name": "execute_python",
            "arguments": "Malformed function call: import pandas as pd\nprint(123)"
        }
    }
    res_malformed = tm.execute_tool_call(malformed_tc)
    assert res_malformed.get("success") is True, f"Malformed args execution failed: {res_malformed}"
    assert len(executed) == 2

    # 6. Test repair_json_escapes and parse_and_clean_tool_args with user's exact invalid escape pattern:
    # 'f1': xgb_f1,\        'roc_auc': xgb_roc_auc
    from agents.fallback_parser import repair_json_escapes, parse_and_clean_tool_args, validate_or_retry_tool_calls
    malformed_json_args = '{"code": "import pandas as pd\\nxgb_metrics = {\\n    \'f1\': xgb_f1,\\        \'roc_auc\': xgb_roc_auc\\n}"}'
    try:
        json.loads(malformed_json_args)
        assert False, "Should have failed standard json.loads"
    except json.JSONDecodeError:
        pass

    ok, parsed_dict, err = parse_and_clean_tool_args(malformed_json_args)
    assert ok is True, f"parse_and_clean_tool_args failed: {err}"
    assert "code" in parsed_dict
    assert "import pandas as pd" in parsed_dict["code"]

    # 7. Test validate_or_retry_tool_calls when tool arguments fail completely:
    # Verifies that message history in memory_manager is NOT corrupted,
    # and retry prompt is sent to the LLM directly without corrupting history.
    class MockMemoryManager:
        def __init__(self):
            self.messages = [{"role": "system", "content": "You are ML agent."}]

    mock_mm = MockMemoryManager()

    broken_tc = {
        "id": "broken_call_1",
        "type": "function",
        "function": {
            "name": "execute_python",
            "arguments": '{"code": import pandas as pd, unquoted: true'
        }
    }

    class MockAssistantMsg:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    initial_msg = MockAssistantMsg("I will run code", [broken_tc])

    llm_calls = []
    def mock_call_llm(messages, tools=None):
        llm_calls.append(list(messages))
        fixed_tc = {
            "id": "fixed_call_1",
            "type": "function",
            "function": {
                "name": "execute_python",
                "arguments": '{"code": "import pandas as pd\\nprint(1)"}'
            }
        }
        return MockAssistantMsg("Here is the corrected call", [fixed_tc])

    final_msg, clean_tcs = validate_or_retry_tool_calls(
        agent_name="MLAgent",
        assistant_msg=initial_msg,
        memory_manager=mock_mm,
        call_llm_fn=mock_call_llm,
        tool_specs=[],
        max_retries=2
    )

    # Verify: memory_manager.messages was NEVER corrupted with broken_tc!
    assert len(mock_mm.messages) == 1
    assert mock_mm.messages[0]["role"] == "system"

    # Verify: retry was called with error message
    assert len(llm_calls) == 1
    retry_prompt = llm_calls[0][-1]["content"]
    assert "Tool Call Argument Parsing Failed" in retry_prompt
    assert "execute_python" in retry_prompt

    # Verify: clean_tcs contains the fixed tool call with valid JSON
    assert len(clean_tcs) == 1
    assert json.loads(clean_tcs[0]["function"]["arguments"])["code"] == "import pandas as pd\nprint(1)"

    # 8. Test sanitize_messages defends against corrupted tool calls
    from agents.memory_manager import sanitize_messages
    corrupted_history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "run task"},
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": "corrupted_call",
                    "type": "function",
                    "function": {
                        "name": "execute_python",
                        "arguments": "{bad_json: true"
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "corrupted_call",
            "content": "failed"
        }
    ]

    sanitized = sanitize_messages(corrupted_history)
    for m in sanitized:
        if m.get("role") == "assistant":
            assert not m.get("tool_calls"), "Unparseable tool_calls should have been dropped/converted"

    print("ALL FALLBACK PARSER TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_fallback_parser()
