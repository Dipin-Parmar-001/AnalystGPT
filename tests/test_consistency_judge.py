"""Tests for the consistency judge's anti-fabrication guardrails (no network needed)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.consistency_judge import (
    ConsistencyJudge,
    JudgeResult,
    judge_result_to_findings,
    _extract_json_object,
)

REPORT = (
    "# Analysis Report\n\n"
    "## Executive Summary\n"
    "The model is well-calibrated and achieved ROC-AUC of 0.8453.\n\n"
    "## Data Overview\n"
    "The dataset has 7,043 rows and 50 columns.\n"
)


class FakeClient:
    """Stand-in for the OpenAI client returning a canned completion."""
    def __init__(self, content):
        self._content = content
        class _Msg:
            content = self._content
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        self._resp = _Resp()

    class chat:
        pass


def _make_judge_with_response(content):
    judge = object.__new__(ConsistencyJudge)  # skip __init__ (no API key needed)
    judge.model_name = "fake"
    fake = FakeClient(content)
    fake.chat.completions = None

    class _Completions:
        def create(self, **kwargs):
            return fake._resp
    fake.chat.completions = _Completions()
    judge.client = fake
    return judge


class TestJudgeGuardrails(unittest.TestCase):
    def test_extract_json_object_from_fenced(self):
        text = 'Blah\n```json\n{"claims": []}\n```\n'
        self.assertEqual(_extract_json_object(text), {"claims": []})

    def test_extract_json_object_from_bare(self):
        text = 'Here it is: {"claims": [], "overall_assessment": "ok"}'
        self.assertEqual(_extract_json_object(text)["claims"], [])

    def test_unparsable_output_fails_open(self):
        judge = _make_judge_with_response("I cannot produce JSON, sorry!")
        res = judge.judge_report(REPORT, verified_metrics={})
        self.assertFalse(res.parse_ok)
        self.assertEqual(res.claims, [])

    def test_fabricated_quotes_discarded(self):
        # Judge invents a quote that is NOT in the report -> must be discarded
        raw = '''{"claims": [
            {"quote": "The model achieved 99.9% accuracy", "verdict": "contradicted",
             "severity": "critical", "reason": "wrong", "grounding": ""},
            {"quote": "The model is well-calibrated", "verdict": "contradicted",
             "severity": "critical", "reason": "no calibration analysis exists", "grounding": ""}
        ], "overall_assessment": "unsafe"}'''
        judge = _make_judge_with_response(raw)
        res = judge.judge_report(REPORT, verified_metrics={})
        self.assertTrue(res.parse_ok)
        self.assertEqual(res.discarded_claims, 1)
        self.assertEqual(len(res.claims), 1)
        self.assertEqual(res.claims[0].quote, "The model is well-calibrated")

    def test_invalid_verdict_discarded_and_claims_capped(self):
        raw = '''{"claims": [''' + ",".join([
            f'{{"quote": "claim {i} xyz", "verdict": "made_up_verdict", "severity": "critical", "reason": "r"}}'
            for i in range(5)
        ]) + '''], "overall_assessment": ""}'''
        judge = _make_judge_with_response(raw)
        res = judge.judge_report(REPORT, verified_metrics={})
        self.assertEqual(res.claims, [])
        self.assertEqual(res.discarded_claims, 5)

    def test_judge_findings_conversion(self):
        from agents.consistency_judge import JudgedClaim
        res = JudgeResult(parse_ok=True, claims=[
            JudgedClaim(quote="q1", section="", verdict="contradicted", severity="critical", reason="r"),
            JudgedClaim(quote="q2", section="", verdict="contradicted", severity="minor", reason="r"),
            JudgedClaim(quote="q3", section="", verdict="unverifiable", severity="critical", reason="r"),
            JudgedClaim(quote="q4", section="", verdict="supported", severity="minor", reason="r"),
        ])
        findings = judge_result_to_findings(res)
        codes = [f["code"] for f in findings]
        self.assertEqual(codes, ["judge_contradicted_claim", "judge_contradicted_claim", "judge_unverifiable_claim"])
        # unverifiable critical is downgraded to minor finding severity
        self.assertEqual(findings[-1]["severity"], "minor")


if __name__ == "__main__":
    unittest.main()
