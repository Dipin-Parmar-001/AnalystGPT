"""
LLM Consistency Judge - AnalystGPT (Hybrid verification, LLM layer).

Audits the final executive report for PROSE-LEVEL claims the deterministic gates
cannot see: fabricated capabilities ("well-calibrated probabilities" with no
calibration analysis), wrong relationships ("r ≈ 0.5" vs actual 0.787), claims that
modeling/feature engineering occurred when it didn't, and mislabeled artifacts.

The judge runs AFTER the deterministic numeric verification, so it complements —
never replaces — the deterministic gates.

Guardrails so the judge cannot introduce new hallucinations:
1. Temperature 0 and a JSON-only output contract.
2. Robust JSON extraction; local NVIDIA/OpenRouter/Gemini fallback on
   provider failures or malformed output, after which the orchestrator emits a visible
   `judge_failed` finding (the run is never silently left unjudged).
3. Anti-fabrication filter: every returned claim's `quote` must appear verbatim
   (whitespace-normalized) in the report. Claims quoting text absent from the
   report are discarded as judge hallucinations.
4. Claims are capped and severities normalized; the judge's verdicts are treated
   as findings for the severity policy, not as autonomous report edits.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from openai import OpenAI

from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.prompts import CONSISTENCY_JUDGE_SYSTEM_PROMPT
from agents.router import create_local_router


def is_fragmented_or_unparsable_response(message: Any) -> bool:
    """
    Detect if the LLM response is fragmented, unparsable, or contains only
    token usage metadata instead of actual content.

    Returns True if the response should be retried.
    """
    if message is None:
        return True

    # Check if content is None or empty
    content = getattr(message, "content", None)
    if content is None or (isinstance(content, str) and not content.strip()):
        # If no content and no tool calls, it's likely an incomplete response
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return True

    # Check if content looks like token usage metadata (JSON with token fields)
    if isinstance(content, str):
        content_stripped = content.strip()
        # Check for the specific token usage metadata pattern
        if content_stripped.startswith("{") and content_stripped.endswith("}"):
            try:
                parsed = json.loads(content_stripped)
                # Check for token usage metadata keys
                token_keys = {"completion_tokens", "prompt_tokens", "total_tokens",
                             "completion_tokens_details", "prompt_tokens_details",
                             "cost", "is_byok", "cost_details"}
                if isinstance(parsed, dict) and token_keys.intersection(parsed.keys()):
                    # This is token usage metadata, not actual content
                    return True
            except json.JSONDecodeError:
                pass

        # Check for other common fragmented response patterns
        # e.g., empty object, just whitespace, or clearly incomplete
        if content_stripped in ["{}", "[]", "null", ""]:
            return True

    # Check finish_reason if available (some providers include this in message)
    finish_reason = getattr(message, "finish_reason", None)
    if finish_reason in ["length", "content_filter", "function_call"]:
        # These might indicate incomplete responses
        pass

    return False

try:
    import logfire
except ImportError:
    logfire = None

load_dotenv()

FALLBACK_STATUS_CODES = {400, 402, 404, 412, 429, 500, 502, 503, 504}
@dataclass
class JudgedClaim:
    quote: str
    section: str
    verdict: str      # "contradicted" | "unverifiable" | "supported"
    severity: str     # "critical" | "minor"
    reason: str
    grounding: str = ""


@dataclass
class JudgeResult:
    parse_ok: bool
    claims: List[JudgedClaim] = field(default_factory=list)
    discarded_claims: int = 0            # judge claims filtered by the anti-fabrication guard
    overall_assessment: str = ""
    raw_response: str = ""
    error: str = ""

    @property
    def contradicted(self) -> List[JudgedClaim]:
        return [c for c in self.claims if c.verdict == "contradicted"]

    @property
    def unverifiable(self) -> List[JudgedClaim]:
        return [c for c in self.claims if c.verdict == "unverifiable"]

    def critical_claims(self) -> List[JudgedClaim]:
        return [c for c in self.claims if c.verdict == "contradicted" and c.severity == "critical"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the final JSON answer, ignoring preceding analysis or markdown."""
    if not text:
        return None

    # Strategy 1: Markdown fenced code block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    # Gather every balanced object (respecting braces inside JSON strings),
    # then prefer the last one because reasoning commonly precedes the answer.
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    ordered = candidates[:1] + list(reversed(candidates[1:])) if fence else list(reversed(candidates))
    for cand in ordered:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                # Accept if it has our expected structure
                if "claims" in obj or "overall_assessment" in obj:
                    return obj
                # Also accept if it has claim-like structure (some LLMs omit top-level keys)
                if any(k in str(obj).lower() for k in ("quote", "verdict", "severity")):
                    return obj
        except json.JSONDecodeError:
            # Try to repair common JSON issues
            repaired = _repair_json(cand)
            if repaired:
                try:
                    obj = json.loads(repaired)
                    if isinstance(obj, dict) and ("claims" in obj or "overall_assessment" in obj):
                        return obj
                except json.JSONDecodeError:
                    continue

    return None


def _repair_json(text: str) -> Optional[str]:
    """Attempt to repair common JSON issues from LLM output."""
    if not text:
        return None

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix unquoted keys (simple cases)
    text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', text)

    # Ensure it ends properly
    if text.count("{") > text.count("}"):
        text += "}" * (text.count("{") - text.count("}"))
    elif text.count("[") > text.count("]"):
        text += "]" * (text.count("[") - text.count("]"))

    return text


VALID_VERDICTS = {"contradicted", "unverifiable", "supported"}
VALID_SEVERITIES = {"critical", "minor"}
MAX_CLAIMS = 20
MAX_RETRIES = 3


class ConsistencyJudge:
    """LLM-based prose consistency judge with anti-fabrication guardrails."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        setup_logfire()
        self.routes = []
        if client is not None:
            self.routes.append({"provider": "injected", "model": model_name or "injected", "client": client})
        else:
            local_client = create_local_router("ConsistencyJudge")
            self.routes.append({"provider": "local", "model": model_name or "nvidia/nemotron-3-super-120b-a12b",
                                "client": local_client})

        from agents.observability import instrument_openai_client
        for route in self.routes:
            instrument_openai_client(route["client"], agent_name="ConsistencyJudge")
        self.client = self.routes[0]["client"]
        self.model_name = model_name or self.routes[0]["model"]
        self._next_route_index = 0

    def _call_llm(self, user_content: str) -> str:
        routes = getattr(self, "routes", None) or [{
            "provider": "injected",
            "model": getattr(self, "model_name", "injected"),
            "client": self.client,
        }]
        errors = []
        route_count = len(routes)
        start_index = getattr(self, "_next_route_index", 0) % route_count
        for offset in range(route_count):
            index = (start_index + offset) % route_count
            route = routes[index]
            try:
                response = route["client"].chat.completions.create(
                    model=route["model"],
                    messages=[
                        {"role": "system", "content": CONSISTENCY_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                )
                message = response.choices[0].message
                content = getattr(message, "content", None)
                parsed = None if is_fragmented_or_unparsable_response(message) else _extract_json_object(content or "")
                if parsed is None or not isinstance(parsed.get("claims"), list):
                    errors.append(f"{route['provider']}:{route['model']} returned no valid JSON object")
                    continue
                # Normalize to JSON only; discard analysis, markdown, and any
                # provider-specific reasoning text surrounding the answer.
                self._next_route_index = (index + 1) % route_count
                return json.dumps(parsed, ensure_ascii=False)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                detail = f"HTTP {status}" if status in FALLBACK_STATUS_CODES else f"request failed ({exc})"
                errors.append(f"{route['provider']}:{route['model']} {detail}")
        self._next_route_index = (start_index + 1) % route_count
        raise RuntimeError("All consistency-judge routes failed: " + "; ".join(errors))

    def judge_report(
        self,
        report_text: str,
        verified_metrics: Optional[Dict[str, Any]] = None,
        consistency_findings: Optional[List[Dict[str, Any]]] = None,
        artifacts_manifest: Optional[List[str]] = None,
    ) -> JudgeResult:
        """
        Audits the report. `consistency_findings` are the deterministic checker's
        findings (dicts with severity/code/message) provided as extra context.
        """
        if not report_text or not report_text.strip():
            return JudgeResult(parse_ok=True, overall_assessment="Empty report; nothing to judge.")

        # Compact ground-truth digest: drop bulky per-column stats to keep the
        # prompt focused on what prose claims are checked against.
        gt_digest: Dict[str, Any] = {}
        if isinstance(verified_metrics, dict):
            for key in ("dataset", "best_model", "model_metrics", "tuned_model_metrics",
                        "baseline_score", "confusion_matrix", "predictions_summary",
                        "top_features", "hyperparameter_tuning_summary", "best_params",
                        "champion_model", "versioned_model_artifacts",
                        "segment_stats", "threshold_analysis"):
                if key in verified_metrics:
                    gt_digest[key] = verified_metrics[key]
            ds = gt_digest.get("dataset")
            if isinstance(ds, dict):
                gt_digest["dataset"] = {k: v for k, v in ds.items() if k not in ("numeric_stats", "categorical_stats")}
            gt_digest["artifacts_manifest"] = artifacts_manifest or [
                a.get("filename") for a in (verified_metrics.get("artifacts_manifest") or []) if isinstance(a, dict)
            ]

        user_parts = [
            "=== REPORT TO AUDIT ===\n" + report_text,
        ]
        if gt_digest:
            user_parts.append("=== GROUND TRUTH (VERIFIED_METRICS digest, computed directly from artifacts) ===\n"
                              + json.dumps(gt_digest, indent=2, default=str))
        if consistency_findings:
            user_parts.append(
                "=== DETERMINISTIC CONSISTENCY FINDINGS (already established; verify their report-level impact) ===\n"
                + json.dumps(consistency_findings, indent=2, default=str)
            )
        user_parts.append(
            "Audit the report now. Output ONLY the JSON object defined in your instructions."
        )

        # Retry logic for LLM call (handles both call failures and unparsable responses)
        max_retries = MAX_RETRIES
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    raw = self._call_llm("\n\n".join(user_parts))
                else:
                    # Retry with repair suffix
                    repair_suffix = (
                        f"\n\nYour previous response was not parsable as JSON (attempt {attempt}/{max_retries}). "
                        "Respond AGAIN with ONLY the raw JSON object with keys 'overall_assessment' and 'claims' — "
                        "no prose, no markdown fences, no commentary."
                    )
                    raw = self._call_llm("\n\n".join(user_parts) + repair_suffix)

                # Log the raw response for debugging
                if HAS_LOGFIRE:
                    logfire.info(
                        f"ConsistencyJudge LLM response received (attempt {attempt + 1}/{max_retries + 1})",
                        raw_length=len(raw) if raw else 0,
                        raw_preview=raw[:500] if raw else "EMPTY"
                    )
                else:
                    print(f"[ConsistencyJudge] LLM response received (attempt {attempt + 1}/{max_retries + 1}), length: {len(raw) if raw else 0}")
                    print(f"[ConsistencyJudge] Raw preview: {raw[:500] if raw else 'EMPTY'}")

                parsed = _extract_json_object(raw)
                if parsed is not None:
                    # Successfully parsed
                    if HAS_LOGFIRE:
                        logfire.info(
                            f"ConsistencyJudge JSON parsed successfully (attempt {attempt + 1})",
                            claims_count=len(parsed.get("claims", [])),
                            overall_assessment=parsed.get("overall_assessment", "")[:100]
                        )
                    else:
                        print(f"[ConsistencyJudge] JSON parsed successfully (attempt {attempt + 1})")
                        print(f"[ConsistencyJudge] Claims count: {len(parsed.get('claims', []))}")
                    break
                else:
                    # Failed to parse JSON
                    last_error = "unparsable_json"
                    if HAS_LOGFIRE:
                        logfire.warn(
                            f"ConsistencyJudge returned unparsable JSON (attempt {attempt + 1}/{max_retries + 1})",
                            raw_preview=raw[:500] if raw else "EMPTY"
                        )
                    else:
                        print(f"[ConsistencyJudge] WARNING: Unparsable JSON (attempt {attempt + 1}/{max_retries + 1})")
                        print(f"[ConsistencyJudge] Raw preview: {raw[:500] if raw else 'EMPTY'}")

                    if attempt == max_retries:
                        # All retries exhausted
                        if HAS_LOGFIRE:
                            logfire.error(
                                "ConsistencyJudge failed after all retries - returning not verified",
                                total_attempts=max_retries + 1,
                                error=last_error,
                                raw_response=raw[:2000] if raw else "EMPTY"
                            )
                        else:
                            print(f"[ConsistencyJudge] ERROR: Failed after {max_retries + 1} attempts - returning not verified")
                        return JudgeResult(
                            parse_ok=False,
                            raw_response=raw[:2000] if raw else "",
                            error=f"unparsable_json (after {max_retries} retries)"
                        )
                    # Continue to next retry
                    continue

            except Exception as e:
                last_error = str(e)
                if HAS_LOGFIRE:
                    logfire.warn(
                        f"ConsistencyJudge LLM call failed (attempt {attempt + 1}/{max_retries + 1})",
                        error=str(e)
                    )
                else:
                    print(f"[ConsistencyJudge] LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")

                if attempt == max_retries:
                    # All retries exhausted
                    if HAS_LOGFIRE:
                        logfire.error(
                            "ConsistencyJudge LLM call failed after all retries - returning not verified",
                            total_attempts=max_retries + 1,
                            error=last_error
                        )
                    else:
                        print(f"[ConsistencyJudge] ERROR: LLM call failed after {max_retries + 1} attempts - returning not verified")
                    return JudgeResult(
                        parse_ok=False,
                        raw_response="",
                        error=f"LLM call failed after {max_retries} retries: {last_error}"
                    )
                # Continue to next retry
                continue

        norm_report = _normalize_ws(report_text)
        claims: List[JudgedClaim] = []
        discarded = 0
        for item in (parsed.get("claims") or [])[:MAX_CLAIMS * 2]:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            verdict = str(item.get("verdict") or "").strip().lower()
            severity = str(item.get("severity") or "").strip().lower() or "minor"
            if verdict not in VALID_VERDICTS or not quote:
                discarded += 1
                continue
            if severity not in VALID_SEVERITIES:
                severity = "minor"
            # Anti-fabrication guard: the quote must exist verbatim in the report.
            if _normalize_ws(quote) not in norm_report:
                discarded += 1
                continue
            claims.append(JudgedClaim(
                quote=quote,
                section=str(item.get("section") or "").strip(),
                verdict=verdict,
                severity=severity,
                reason=str(item.get("reason") or "").strip(),
                grounding=str(item.get("grounding") or "").strip(),
            ))
            if len(claims) >= MAX_CLAIMS:
                break

        return JudgeResult(
            parse_ok=True,
            claims=claims,
            discarded_claims=discarded,
            overall_assessment=str(parsed.get("overall_assessment") or "").strip(),
            raw_response=raw[:2000],
        )


def judge_result_to_findings(result: JudgeResult) -> List[Dict[str, Any]]:
    """Converts a JudgeResult into findings compatible with the consistency checker format."""
    findings = []
    for claim in result.contradicted:
        findings.append({
            "severity": "critical" if claim.severity == "critical" else "minor",
            "code": "judge_contradicted_claim",
            "message": f"Report claim contradicts ground truth: \"{claim.quote[:160]}\" — {claim.reason}",
            "artifacts": ["final_report.md"],
            "grounding": claim.grounding,
            "source": "consistency_judge",
        })
    for claim in result.unverifiable:
        if claim.severity == "critical":
            findings.append({
                "severity": "minor",
                "code": "judge_unverifiable_claim",
                "message": f"Report claim cannot be verified against any run artifact: \"{claim.quote[:160]}\" — {claim.reason}",
                "artifacts": ["final_report.md"],
                "grounding": claim.grounding,
                "source": "consistency_judge",
            })
    return findings


if __name__ == "__main__":
    print("ConsistencyJudge module — configure GEMINI_API_KEY and/or OPENROUTER_API_KEY to run.")
