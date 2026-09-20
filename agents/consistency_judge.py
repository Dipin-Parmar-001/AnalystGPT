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
2. Robust JSON extraction + strict schema validation; one strict retry on an
   unparsable response, after which the orchestrator emits a visible
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

try:
    import logfire
except ImportError:
    logfire = None

load_dotenv()

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")


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
    """Extracts the first JSON object from an LLM response (fenced or bare)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


VALID_VERDICTS = {"contradicted", "unverifiable", "supported"}
VALID_SEVERITIES = {"critical", "minor"}
MAX_CLAIMS = 20


class ConsistencyJudge:
    """LLM-based prose consistency judge with anti-fabrication guardrails."""

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
        client: Optional[OpenAI] = None,
    ):
        if not PORTKEY_API_KEY:
            raise ValueError("PORTKEY_API_KEY is not set in environment variables.")
        setup_logfire()

        if client is not None:
            self.client = client
        else:
            headers = {"x-portkey-api-key": PORTKEY_API_KEY}
            if PORTKEY_CONFIG_ID:
                headers["x-portkey-config"] = PORTKEY_CONFIG_ID
            else:
                headers["x-portkey-config"] = json.dumps({
                    "strategy": {"mode": "fallback", "on_status_codes": [429, 500, 502, 503, 504]},
                    "targets": [
                        {"virtual_key": "analystgpt-openrouter",
                         "override_params": {"model": "nemotron-3-ultra-550b-a55b", "temperature": 0, "max_tokens": 2000}},
                        {"virtual_key": "analystgpt-gemini",
                         "override_params": {"model": "gemini-2.5-flash", "temperature": 0, "max_tokens": 2000}},
                    ],
                })
            self.client = OpenAI(
                base_url="https://api.portkey.ai/v1",
                api_key=PORTKEY_API_KEY,
                default_headers=headers,
            )

        self.model_name = model_name

        # Local observability: per-call latency, served model, token metrics
        from agents.observability import instrument_openai_client
        instrument_openai_client(self.client, agent_name="ConsistencyJudge")

    def _call_llm(self, user_content: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": CONSISTENCY_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""

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

        try:
            raw = self._call_llm("\n\n".join(user_parts))
        except Exception as e:
            if HAS_LOGFIRE:
                logfire.warn("ConsistencyJudge LLM call failed; proceeding without judge findings", error=str(e))
            return JudgeResult(parse_ok=False, error=f"LLM call failed: {e}")

        parsed = _extract_json_object(raw)
        if parsed is None:
            # One strict retry: an unparsable judge response must not silently
            # zero out the entire prose-verification layer.
            repair_suffix = (
                "\n\nYour previous response was not parsable as JSON. Respond AGAIN with "
                "ONLY the raw JSON object with keys 'overall_assessment' and 'claims' — "
                "no prose, no markdown fences, no commentary."
            )
            try:
                raw = self._call_llm("\n\n".join(user_parts) + repair_suffix)
            except Exception as e:
                if HAS_LOGFIRE:
                    logfire.warn("ConsistencyJudge retry LLM call failed", error=str(e))
                return JudgeResult(parse_ok=False, raw_response=raw[:2000],
                                   error=f"unparsable_json (retry LLM call failed: {e})")
            parsed = _extract_json_object(raw)
            if parsed is None:
                if HAS_LOGFIRE:
                    logfire.warn("ConsistencyJudge returned unparsable output after one retry; ignoring judge findings")
                return JudgeResult(parse_ok=False, raw_response=raw[:2000],
                                   error="unparsable_json (after 1 retry)")

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
    print("ConsistencyJudge module — requires PORTKEY_API_KEY to run.")
