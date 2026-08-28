"""
Fallback Parser & Code Extraction Utilities - AnalystGPT

Handles non-standard, malformed, or envelope-wrapped LLM responses across providers
(e.g., Gemini / OpenRouter / Portkey returning 'Malformed function call:', raw markdown code blocks,
or HTTP response envelopes) and extracts executable Python code or JSON payloads seamlessly.
"""

import re
import json
from typing import Optional, Dict, Any, Union, List


class SyntheticFunction:
    def __init__(self, name: str, arguments: Union[str, Dict[str, Any]]):
        self.name = name
        if isinstance(arguments, dict):
            self.arguments = json.dumps(arguments)
        else:
            self.arguments = str(arguments)

    def __getitem__(self, item):
        if item == "name":
            return self.name
        elif item == "arguments":
            return self.arguments
        raise KeyError(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default


class SyntheticToolCall:
    """
    Object compatible with both OpenAI SDK ChatCompletionMessageToolCall
    (attribute access: tc.id, tc.function.name, tc.function.arguments)
    and dictionary access (tc['id'], tc['function']['name']).
    """
    def __init__(self, id: str, name: str, arguments: Union[str, Dict[str, Any]]):
        self.id = id
        self.type = "function"
        self.function = SyntheticFunction(name, arguments)

    def __getitem__(self, item):
        if item == "id":
            return self.id
        elif item == "type":
            return self.type
        elif item == "function":
            return self.function
        raise KeyError(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments
            }
        }


def unwrap_envelope(response_or_text: Any) -> Any:
    """
    Unwraps Portkey/OpenAI HTTP REST envelopes like:
    {
      "status": 200,
      "body": {
        "choices": [
          { "message": { "content": "..." } }
        ]
      }
    }
    or returns the original object/string.
    """
    if isinstance(response_or_text, str):
        text = response_or_text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                return unwrap_envelope(parsed)
            except Exception:
                return response_or_text
        return response_or_text

    if isinstance(response_or_text, dict):
        # Case: { "status": 200, "body": { "choices": ... } }
        if "body" in response_or_text and isinstance(response_or_text["body"], dict):
            return unwrap_envelope(response_or_text["body"])
        
        # Case: { "choices": [ { "message": { ... } } ] }
        if "choices" in response_or_text and isinstance(response_or_text["choices"], list) and len(response_or_text["choices"]) > 0:
            first_choice = response_or_text["choices"][0]
            if isinstance(first_choice, dict):
                msg = first_choice.get("message", {})
                if isinstance(msg, dict):
                    return msg.get("content", "")
                return str(first_choice)

        # Case: { "message": { "content": "..." } }
        if "message" in response_or_text and isinstance(response_or_text["message"], dict):
            return response_or_text["message"].get("content", "")

    return response_or_text


def extract_code_from_content(content: Optional[str]) -> Optional[str]:
    """
    Extracts executable Python code from an LLM response string when standard tool_calls
    are empty or when the model returned non-standard formats (e.g. Gemini's 'Malformed function call:').
    
    Handles:
    1. 'Malformed function call: import pandas ...'
    2. 'Malformed function call: execute_python(code=...)'
    3. 'Malformed function call: {"name": "execute_python", "arguments": {"code": "..."}}'
    4. Markdown code blocks ```python ... ``` or ``` ... ```
    5. Direct raw Python script text containing common Python keywords.
    """
    if not content:
        return None

    # First unwrap any JSON envelope string if present
    unwrapped = unwrap_envelope(content)
    if isinstance(unwrapped, str):
        raw_text = unwrapped.strip()
    else:
        raw_text = str(unwrapped).strip()

    if not raw_text:
        return None

    # Check for 'Malformed function call:' prefix (case-insensitive)
    malformed_prefix_match = re.match(r"^Malformed function call:?\s*", raw_text, re.IGNORECASE)
    if malformed_prefix_match:
        stripped_body = raw_text[malformed_prefix_match.end():].strip()
        # 1a. Check if the remainder is a JSON string
        if stripped_body.startswith("{") and stripped_body.endswith("}"):
            try:
                parsed_json = json.loads(stripped_body)
                if isinstance(parsed_json, dict):
                    # Check arguments.code
                    if "arguments" in parsed_json and isinstance(parsed_json["arguments"], dict):
                        if "code" in parsed_json["arguments"]:
                            return parsed_json["arguments"]["code"].strip()
                    if "code" in parsed_json:
                        return str(parsed_json["code"]).strip()
            except Exception:
                pass

        # 1b. Check if the remainder is function call syntax execute_python(...)
        fn_syntax_match = re.match(r"^execute_python\s*\(\s*(?:code\s*=\s*)?([\"']{1,3})(.*?)\1\s*\)$", stripped_body, re.DOTALL)
        if fn_syntax_match:
            return fn_syntax_match.group(2).strip()

        # 1c. Check if there are markdown blocks inside
        code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", stripped_body, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            return code_block_match.group(1).strip()

        # 1d. The remainder itself is the raw Python code (e.g. 'import pandas as pd\n...')
        if looks_like_python_code(stripped_body):
            return stripped_body

    # Check for markdown code blocks ```python ... ``` or ``` ... ```
    code_block_matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if code_block_matches:
        for block in code_block_matches:
            block_clean = block.strip()
            if looks_like_python_code(block_clean):
                return block_clean
        # If any block was found, return the largest one
        longest_block = max(code_block_matches, key=len).strip()
        if len(longest_block) > 10:
            return longest_block

    # Check if raw text is directly Python code
    if looks_like_python_code(raw_text):
        return raw_text

    return None


def looks_like_python_code(text: str) -> bool:
    """
    Heuristics to determine if a string contains executable Python code.
    """
    if not text:
        return False

    code_indicators = [
        r"^\s*import\s+[a-zA-Z0-9_]+",
        r"^\s*from\s+[a-zA-Z0-9_.]+\s+import\s+",
        r"pd\.read_csv",
        r"pd\.DataFrame",
        r"np\.",
        r"plt\.",
        r"sns\.",
        r"sklearn",
        r"train_test_split",
        r"\.fit\(",
        r"\.predict\(",
        r"\.to_csv\(",
        r"joblib\.dump",
        r"json\.dump",
        r"def\s+[a-zA-Z0-9_]+\s*\(",
        r"print\s*\(",
    ]

    matches = 0
    for pattern in code_indicators:
        if re.search(pattern, text, re.MULTILINE):
            matches += 1
            if matches >= 2:
                return True

    # If it starts with import/from and has several lines
    lines = text.strip().splitlines()
    if len(lines) >= 3:
        first_line = lines[0].strip()
        if first_line.startswith("import ") or first_line.startswith("from ") or first_line.startswith("#"):
            return True

    return False


def create_synthetic_execute_tool_call(code: str, call_id: Optional[str] = None) -> SyntheticToolCall:
    """
    Creates a SyntheticToolCall for execute_python that can be processed
    identically to an OpenAI function tool call.
    """
    tid = call_id or f"fallback_call_{abs(hash(code)) % 1000000}"
    return SyntheticToolCall(
        id=tid,
        name="execute_python",
        arguments={"code": code}
    )
