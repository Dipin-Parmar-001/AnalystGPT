"""Tests for the numeric verifier's provenance and noise-masking behavior."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.numeric_verifier import verify_report_numbers, extract_numeric_claims


VERIFIED = {
    "dataset": {"rows": 7043, "cols": 50},
    "model_metrics": {"LogisticRegression": {"ROC_AUC": 0.8453, "F1": 0.62}},
    "top_features": [{"feature": "tenure", "importance": -0.3739}],
}


class TestNumericVerifier(unittest.TestCase):
    def test_accurate_numbers_verified_with_provenance(self):
        report = "The dataset has 7,043 rows and the model reached ROC-AUC of 0.8453."
        res = verify_report_numbers(report, VERIFIED)
        self.assertTrue(res.is_valid)
        self.assertIn("dataset.rows", res.provenance_map.get("7,043", []))
        # 0.8453 claim should trace to the model metrics path
        prov = res.provenance_map.get("0.8453", [])
        self.assertTrue(any("ROC_AUC" in p for p in prov), prov)

    def test_fabricated_number_flagged(self):
        report = "The dataset has 7,043 rows and 54 columns."
        res = verify_report_numbers(report, VERIFIED)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.unverified_claims[0].raw_str, "54")

    def test_bucket_ranges_masked(self):
        report = "The `tenure_bucket` bins are 0-12, 13-24, 25-48 and 49+ months."
        res = verify_report_numbers(report, VERIFIED)
        self.assertTrue(res.is_valid, [c.raw_str for c in res.unverified_claims])

    def test_split_fraction_and_file_size_masked(self):
        report = "Split was 80/20 and the file size was 1.35 MB."
        res = verify_report_numbers(report, VERIFIED)
        self.assertTrue(res.is_valid, [c.raw_str for c in res.unverified_claims])

    def test_section_attribution(self):
        report = "## Machine Learning & Model Evaluation\nAccuracy was 0.8453.\n"
        claims = extract_numeric_claims(report)
        self.assertTrue(claims)
        self.assertIn("Machine Learning", claims[0].section)

    def test_correction_prompt_lists_offenders(self):
        res = verify_report_numbers("Value 12345.678 is claimed.", VERIFIED)
        self.assertFalse(res.is_valid)
        self.assertIn("AUTOMATED NUMERIC VERIFICATION FAILED", res.correction_prompt)
        self.assertIn("12345.678", res.correction_prompt)


if __name__ == "__main__":
    unittest.main()
