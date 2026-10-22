import unittest

from shrine_classifier.analysis import classification_metrics, paired_holm_tests
from shrine_classifier.data import CLASSES


class AnalysisContractTests(unittest.TestCase):
    def test_four_class_macro_metrics_keep_fixed_labels(self):
        metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(list(metrics["per_class"]), CLASSES)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def test_paired_results_join_by_seed_and_holm_adjusts_family(self):
        rows = []
        for seed, val in ((123, .7), (42, .8), (456, .9), (789, .75)):
            rows.append({"architecture": "ref", "seed": seed, "status": "completed", "test_acc": val, "test_macro_f1": val})
        for seed, val in ((456, .5), (789, .6), (42, .6), (123, .55)):
            rows.append({"architecture": "other", "seed": seed, "status": "completed", "test_acc": val, "test_macro_f1": val})
        result = paired_holm_tests(rows, "ref", architecture_order=["ref", "other"])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r["paired_n"] == 4 and r["status"] == "ok" for r in result))
        self.assertAlmostEqual(result[0]["mean_difference"], .225)
        self.assertGreaterEqual(result[0]["p_holm"], result[0]["p_value"])

    def test_missing_or_zero_variance_pairs_are_not_claimed_significant(self):
        rows = []
        for seed in (42, 123, 456):
            rows.extend([{ "architecture": "ref", "seed": seed, "status": "completed", "test_acc": .5, "test_macro_f1": .5},
                         { "architecture": "other", "seed": seed, "status": "completed", "test_acc": .5, "test_macro_f1": .5}])
        result = paired_holm_tests(rows, "ref", architecture_order=["ref", "other"])
        self.assertTrue(all(r["p_value"] is None and r["significant_holm"] is None for r in result))
        rows = [r for r in rows if not (r["architecture"] == "other" and r["seed"] == 123)]
        result = paired_holm_tests(rows, "ref", architecture_order=["ref", "other"])
        self.assertTrue(all(r["status"] == "undefined_zero_variance_or_nonfinite" for r in result))


if __name__ == "__main__":
    unittest.main()
