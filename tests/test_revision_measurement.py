"""Scientific invariants for paired measurement uncertainty."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("measurement_reliability", Path(__file__).resolve().parents[1] / "scripts/revisions/measurement_reliability.py")
measurement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measurement)


class MeasurementTests(unittest.TestCase):
    def test_shared_scale_noise_cancels_in_ratio_and_delta_covariance(self):
        values = np.array([[1., 2., 1.], [2., 4., 2.], [4., 8., 4.]])
        self.assertLess(measurement.delta_ratio_se(values), 1e-8)
        data = pd.DataFrame(values, columns=["p", "b1", "b2"])
        data["variant"] = "v"
        data["condition"] = "c"
        data["replicate"] = [1, 2, 3]
        result = measurement.analyze_replicates(data, ["p", "b1", "b2"], "variant", "condition", "replicate", {}, bootstrap=100)
        self.assertGreater(result.iloc[0].product_cv, 0)
        self.assertEqual(result.iloc[0].replicate_ratio_sd, 0)
        self.assertEqual(result.iloc[0].ratio_of_means_ci_low, .25)
        self.assertEqual(result.iloc[0].ratio_of_means_ci_high, .25)

    def test_exclusion_preserves_complete_pair_and_half_lod_is_explicit(self):
        values = np.array([[.01, 1.], [1., 2.], [np.nan, 2.]])
        limits = {"p": {"lod": .1, "loq": .3}}
        actual = measurement.apply_censoring(values, ["p", "b"], limits, "exclude_below_loq")
        np.testing.assert_array_equal(actual, [[1., 2.]])
        actual = measurement.apply_censoring(values, ["p", "b"], limits, "half_lod_sensitivity")
        np.testing.assert_array_equal(actual, [[.05, 1.], [1., 2.]])

    def test_duplicate_replicates_rejected(self):
        data = pd.DataFrame({"variant": ["v", "v"], "condition": ["c", "c"], "replicate": [1, 1], "p": [1., 2.], "b": [1., 1.]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            measurement.analyze_replicates(data, ["p", "b"], "variant", "condition", "replicate", {}, bootstrap=100)

    def test_summary_audit_does_not_infer_replicates(self):
        data = pd.DataFrame({"Enzyme": ["v"], "0_DVA": [2.], "0_DVA_std": [.5], "0_DVA/(DVA+DVO)": [.5], "0_DVA/(DVA+DVO)_std": [.1]})
        report, result = measurement.summary_audit(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].product_cv, .25)
        self.assertIn("not independent replicates", report["numbered_prefixes_are"])
        self.assertTrue(np.isnan(measurement.coefficient_of_variation(0., 1.)))


if __name__ == "__main__":
    unittest.main()
