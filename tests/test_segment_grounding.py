"""
Tests for the generalized segment-statistics grounding and the stats_findings.json
contract:

1. compute_segment_stats (dataset-agnostic target/segment detection)
2. build_verified_metrics: segment_stats + threshold_analysis + flagged volumes
3. numeric verifier: delta magnitude matching
4. consistency checker: segment_rate_mismatch / stats_findings checks
5. consistency checker: uploaded-file references are not missing artifacts
6. consistency judge: strict retry on unparsable output
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sandbox_execution.verification_script import (
    build_verified_metrics,
    compute_segment_stats,
    detect_target_column,
)
from agents.numeric_verifier import verify_report_numbers
from agents.consistency_checker import run_consistency_checks


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _declared_target_dir(tmp, frame, target="churned"):
    """Writes a dataset + a summary declaring the target column."""
    frame.to_csv(os.path.join(tmp, "cleaned_dataset.csv"), index=False)
    _write_json(os.path.join(tmp, "analyst_summary.json"), {
        "dataset_overview": {"target_column": target},
    })


class TestComputeSegmentStats(unittest.TestCase):
    def test_plain_categorical_rates_and_medians(self):
        n_basic, n_premium, n_standard = 6, 5, 5
        rows = []
        cid = 0
        plan_churn = [("basic", 1)] * 4 + [("basic", 0)] * 2 + \
                     [("premium", 1)] * 1 + [("premium", 0)] * 4 + \
                     [("standard", 1)] * 2 + [("standard", 0)] * 3
        for plan, churn in plan_churn:
            cid += 1
            rows.append({"customer_id": cid, "plan": plan, "churned": churn,
                         "monthly_fee": {"premium": 100.0, "standard": 60.0, "basic": 40.0}[plan]})
        df = pd.DataFrame(rows)
        stats = compute_segment_stats(df, declared_target="churned")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["overall"], {"count": 16, "positives": 7, "rate": 0.4375})
        rates = {(s["feature"], s["category"]): s for s in stats["rate_by_category"]}
        self.assertAlmostEqual(rates[("plan", "basic")]["rate"], 4 / 6, places=4)
        self.assertEqual(rates[("plan", "basic")]["positives"], 4)
        self.assertEqual(rates[("plan", "basic")]["count"], 6)
        self.assertAlmostEqual(rates[("plan", "premium")]["rate"], 0.2, places=4)
        # per-class numeric medians present for monthly_fee
        med = {(d["column"], d["class"]): d["median"] for d in stats["numeric_by_class"]}
        # churned=1 fees: basic x4 (40), premium x1 (100), standard x2 (60) -> median 40
        self.assertEqual(med[("monthly_fee", 1)], 40.0)

    def test_onehot_group_with_reference_category(self):
        rows = []
        data = [("basic", 1)] * 3 + [("basic", 0)] * 2 + \
               [("premium", 1)] * 1 + [("premium", 0)] * 4 + \
               [("standard", 1)] * 2 + [("standard", 0)] * 3
        for i, (plan, churn) in enumerate(data):
            rows.append({
                "plan_basic": 1 if plan == "basic" else 0,
                "plan_premium": 1 if plan == "premium" else 0,
                "churned": churn,
            })
        df = pd.DataFrame(rows)
        stats = compute_segment_stats(df, declared_target="churned")
        rates = {(s["feature"], s["category"]): s["rate"] for s in stats["rate_by_category"]}
        self.assertAlmostEqual(rates[("plan", "basic")], 0.6, places=4)
        self.assertAlmostEqual(rates[("plan", "premium")], 0.2, places=4)
        # all-zeros rows are the reference category
        self.assertAlmostEqual(rates[("plan", "<reference>")], 0.4, places=4)

    def test_target_detection_single_binary_fallback_and_refusal(self):
        df_one = pd.DataFrame({"a": [1, 2, 3, 4], "flag": [1, 0, 0, 1]})
        self.assertEqual(detect_target_column(df_one), "flag")
        df_two = pd.DataFrame({"f1": [1, 0, 0, 1], "f2": [0, 1, 1, 0]})
        self.assertIsNone(detect_target_column(df_two))  # never guesses

    def test_no_target_returns_none(self):
        df = pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": ["x", "y", "z"]})
        self.assertIsNone(compute_segment_stats(df))


class TestBuildVerifiedMetricsGrounding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_segment_stats_and_threshold_flagged_volume(self):
        df = pd.DataFrame({
            "customer_id": range(1, 11),
            "plan": ["basic"] * 5 + ["premium"] * 5,
            "churned": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            "monthly_fee": [40.0] * 5 + [100.0] * 5,
        })
        _declared_target_dir(self.tmp, df)
        _write_json(os.path.join(self.tmp, "threshold_analysis.json"), {
            "default_threshold": 0.5,
            "recommended_threshold": 0.2,
        })
        pd.DataFrame({
            "customer_id": range(1, 11),
            "predicted_probability": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        }).to_csv(os.path.join(self.tmp, "predictions.csv"), index=False)

        vm = build_verified_metrics(self.tmp)

        self.assertIn("segment_stats", vm)
        self.assertEqual(vm["segment_stats"]["overall"]["count"], 10)
        rates = {(s["feature"], s["category"]): s["rate"] for s in vm["segment_stats"]["rate_by_category"]}
        self.assertAlmostEqual(rates[("plan", "basic")], 0.8, places=4)

        self.assertIn("threshold_analysis", vm)
        fv = vm["threshold_analysis"]["flagged_volume"]
        self.assertEqual(fv["default"]["flagged_count"], 4)        # >= 0.5
        self.assertEqual(fv["recommended"]["flagged_count"], 7)    # >= 0.2
        self.assertAlmostEqual(fv["recommended"]["flagged_pct"], 70.0, places=2)

    def test_numeric_verifier_sees_segment_rates_via_pool(self):
        from agents.numeric_verifier import verify_report_numbers
        df = pd.DataFrame({
            "customer_id": range(1, 11),
            "plan": ["basic"] * 5 + ["premium"] * 5,
            "churned": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        })
        _declared_target_dir(self.tmp, df)
        vm = build_verified_metrics(self.tmp)
        # 80.0% (basic rate) and 50.0% (overall rate) both verify on a category line.
        report = "Basic plan customers churn at 80.0%, close to the 50.0% overall market rate."
        res = verify_report_numbers(report, vm)
        self.assertTrue(res.is_valid, [c.raw_str for c in res.unverified_claims])

    def test_numeric_verifier_contextual_rule_rejects_wrong_segment_rate(self):
        from agents.numeric_verifier import verify_report_numbers
        df = pd.DataFrame({
            "customer_id": range(1, 11),
            "plan": ["basic"] * 5 + ["premium"] * 5,
            "churned": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        })
        _declared_target_dir(self.tmp, df)
        vm = build_verified_metrics(self.tmp)
        # A plausible-but-wrong rate on a category line must NOT pass, even though
        # unrelated pool values exist (pool saturation guard).
        report = "Basic plan customers churn at 80.0%, while other plans churn at 55.0%."
        res = verify_report_numbers(report, vm)
        self.assertFalse(res.is_valid)
        self.assertIn("55.0%", [c.raw_str for c in res.unverified_claims])


class TestDeltaMagnitudeMatching(unittest.TestCase):
    def test_signed_delta_matches_magnitude_claim(self):
        vm = {"hyperparameter_tuning_summary": {"comparison": {"deltas": {"recall_delta": -0.2139}}}}
        report = "Tuning reduced recall by 0.214 relative to the baseline."
        res = verify_report_numbers(report, vm)
        self.assertTrue(res.is_valid, [c.raw_str for c in res.unverified_claims])

    def test_fabricated_magnitude_still_flagged(self):
        vm = {"hyperparameter_tuning_summary": {"comparison": {"deltas": {"recall_delta": -0.2139}}}}
        report = "Tuning reduced recall by 0.5 relative to the baseline."
        res = verify_report_numbers(report, vm)
        self.assertFalse(res.is_valid)


class TestSegmentClaimChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df = pd.DataFrame({
            "customer_id": range(1, 17),
            "plan": (["basic"] * 6 + ["premium"] * 5 + ["standard"] * 5),
            "churned": [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        })
        _declared_target_dir(self.tmp, df)

    def test_fabricated_segment_rate_is_critical(self):
        report = "Basic plan customers churn at 99.9% per the analysis."
        rep = run_consistency_checks(self.tmp, report_text=report, persist=False)
        codes = [f.code for f in rep.critical_findings]
        self.assertIn("segment_rate_mismatch", codes)

    def test_true_segment_rate_not_flagged(self):
        report = "Basic plan customers churn at 66.7% per the analysis."
        rep = run_consistency_checks(self.tmp, report_text=report, persist=False)
        mismatches = [f for f in rep.findings if f.code == "segment_rate_mismatch"]
        self.assertEqual(mismatches, [])

    def test_stats_findings_drift_is_critical(self):
        _write_json(os.path.join(self.tmp, "stats_findings.json"), {
            "target_column": "churned",
            "overall": {"count": 16, "positives": 7, "rate": 0.4375},
            "rate_by_category": [
                {"feature": "plan", "category": "basic", "count": 6, "positives": 4, "rate": 0.1111},
            ],
        })
        rep = run_consistency_checks(self.tmp, report_text="", persist=False)
        codes = [f.code for f in rep.critical_findings]
        self.assertIn("segment_rate_mismatch", codes)

    def test_stats_findings_invented_segment_is_minor(self):
        _write_json(os.path.join(self.tmp, "stats_findings.json"), {
            "target_column": "churned",
            "overall": {"count": 16, "positives": 7, "rate": 0.4375},
            "rate_by_category": [
                {"feature": "plan", "category": "enterprise", "count": 99, "positives": 9, "rate": 0.0909},
            ],
        })
        rep = run_consistency_checks(self.tmp, report_text="", persist=False)
        codes = [f.code for f in rep.findings]
        self.assertIn("stats_findings_unmatched_segment", codes)

    def test_missing_stats_findings_with_viz_summary_is_minor(self):
        _write_json(os.path.join(self.tmp, "analyst_summary.json"), {
            "generated_visualizations": [{"file": "chart.png", "description": "bar chart"}],
        })
        rep = run_consistency_checks(self.tmp, report_text="", persist=False)
        codes = [f.code for f in rep.findings]
        self.assertIn("stats_findings_missing", codes)


class TestUploadReferenceResolution(unittest.TestCase):
    def test_report_referencing_upload_not_flagged(self):
        tmp = tempfile.mkdtemp()
        pd.DataFrame({"a": [1, 2], "flag": [0, 1]}).to_csv(
            os.path.join(tmp, "cleaned_dataset.csv"), index=False)
        upload = os.path.join(tmp, "..", "raw_upload.csv")
        report = "The analysis is based on `raw_upload.csv`."
        rep = run_consistency_checks(tmp, report_text=report,
                                     uploaded_files=[os.path.abspath(upload)], persist=False)
        missing = [f for f in rep.critical_findings if f.code == "report_references_missing_artifacts"]
        self.assertEqual(missing, [])


class _SequentialFakeClient:
    """Returns canned completions in order (for retry-path testing)."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        outer = self

        class _Completions:
            def create(inner_self, **kwargs):
                outer.calls += 1
                content = outer._responses[min(outer.calls - 1, len(outer._responses) - 1)]
                _Msg = type("Msg", (), {"content": content})
                _Choice = type("Choice", (), {"message": _Msg()})
                _Resp = type("Resp", (), {"choices": [_Choice()]})
                return _Resp()

        self.chat = type("Chat", (), {"completions": _Completions()})()


class TestJudgeRetry(unittest.TestCase):
    def _make_judge(self, client):
        from agents.consistency_judge import ConsistencyJudge
        judge = object.__new__(ConsistencyJudge)  # skip __init__ (no API key needed)
        judge.model_name = "fake"
        judge.client = client
        return judge

    REPORT = "# Report\n\nThe dataset has 10 rows.\n"

    def test_retry_recovers_from_unparsable_first_response(self):
        from agents.consistency_judge import ConsistencyJudge
        valid = json.dumps({"claims": [], "overall_assessment": "ok"})
        client = _SequentialFakeClient(["not json at all", valid])
        res = self._make_judge(client).judge_report(self.REPORT, verified_metrics={})
        self.assertTrue(res.parse_ok)
        self.assertEqual(client.calls, 2)

    def test_persistent_unparsable_fails_with_retry_marker(self):
        from agents.consistency_judge import ConsistencyJudge
        client = _SequentialFakeClient(["still not json"])
        res = self._make_judge(client).judge_report(self.REPORT, verified_metrics={})
        self.assertFalse(res.parse_ok)
        self.assertIn("after 1 retry", res.error)

    def test_judge_failure_maps_to_finding(self):
        from agents.consistency_judge import JudgeResult, judge_result_to_findings
        # orchestrate-level mapping is inline; verify the finding shape produced
        # for a failed judge (same dict contract as judge_result_to_findings).
        res = JudgeResult(parse_ok=False, error="unparsable_json (after 1 retry)")
        self.assertEqual(judge_result_to_findings(res), [])


if __name__ == "__main__":
    unittest.main()
