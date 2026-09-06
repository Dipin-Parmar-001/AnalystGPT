"""Tests for agents.run_context — authoritative dataset profile."""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.run_context import (
    build_dataset_profile,
    render_dataset_profile_block,
    compare_profile_claims,
)


def _write_csv(path, header, n_rows=5):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(n_rows):
            f.write(",".join(str(i % 3) for _ in header) + "\n")


class TestDatasetProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"),
                   ["customerID", "tenure", "MonthlyCharges", "Churn Value"], n_rows=10)
        _write_csv(os.path.join(self.tmp, "feature_engineered_dataset.csv"),
                   ["customerID", "tenure", "MonthlyCharges", "Churn Value",
                    "tenure_bucket_ordinal", "charges_ratio"], n_rows=10)

    def test_profile_shapes_are_accurate(self):
        profile = build_dataset_profile(self.tmp, persist=False)
        cleaned = profile["datasets"]["cleaned"]
        fe = profile["datasets"]["feature_engineered"]
        self.assertEqual(cleaned["rows"], 10)
        self.assertEqual(cleaned["cols"], 4)
        self.assertEqual(fe["rows"], 10)
        self.assertEqual(fe["cols"], 6)

    def test_engineered_features_are_the_delta(self):
        profile = build_dataset_profile(self.tmp, persist=False)
        self.assertEqual(sorted(profile["engineered_features"]),
                         ["charges_ratio", "tenure_bucket_ordinal"])

    def test_current_dataset_prefers_feature_engineered(self):
        profile = build_dataset_profile(self.tmp, persist=False)
        self.assertEqual(profile["current_dataset"], "feature_engineered_dataset.csv")

    def test_persist_writes_dataset_profile_json(self):
        build_dataset_profile(self.tmp, persist=True)
        path = os.path.join(self.tmp, "dataset_profile.json")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("datasets", data)

    def test_render_block_contains_ground_rules(self):
        profile = build_dataset_profile(self.tmp, persist=False)
        block = render_dataset_profile_block(profile)
        self.assertIn("AUTHORITATIVE DATASET PROFILE", block)
        self.assertIn("GROUND RULES", block)
        self.assertIn("tenure_bucket_ordinal", block)

    def test_compare_profile_claims_flags_wrong_cols(self):
        profile = build_dataset_profile(self.tmp, persist=False)
        issues = compare_profile_claims(profile, claimed_cols=54, dataset_key="feature_engineered")
        self.assertEqual(len(issues), 1)
        self.assertIn("54", issues[0])
        self.assertEqual(compare_profile_claims(profile, claimed_cols=6, dataset_key="feature_engineered"), [])


if __name__ == "__main__":
    unittest.main()
