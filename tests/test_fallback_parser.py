import os
import sys
import json

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from agents.fallback_parser import (
    extract_code_from_content,
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

    # 3. Test markdown block
    md_str = "Here is the code:\n```python\nimport pandas as pd\ndf = pd.DataFrame()\n```"
    code3 = extract_code_from_content(md_str)
    assert code3 == "import pandas as pd\ndf = pd.DataFrame()", f"Got: {code3}"

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

    print("ALL 5 FALLBACK PARSER TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_fallback_parser()
