import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from train_model import (
    add_symptom_dropout,
    compare_models,
    few_symptom_metrics,
    fit_on_dropout_data,
    keep_few_symptoms,
    random_symptom_subset,
    top_k_accuracy,
)


class TestSymptomDropout(unittest.TestCase):
    def setUp(self):
        self.values = np.array(
            [[1, 1, 1, 1, 1, 0],
             [1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1]],
            dtype=np.uint8,
        )
        self.X = pd.DataFrame(self.values, columns=list("abcdef"))
        self.y = pd.Series(["w", "x", "y", "z"])

    def test_keeps_k_symptoms_or_all_if_fewer(self):
        result = random_symptom_subset(self.values, 2, np.random.default_rng(0)).toarray()
        self.assertEqual(result.sum(axis=1).tolist(), [2, 1, 0, 2])

    def test_only_keeps_symptoms_the_row_had(self):
        result = random_symptom_subset(self.values, 3, np.random.default_rng(0)).toarray()
        self.assertTrue(np.all(result <= self.values))

    def test_per_row_keep_counts(self):
        result = random_symptom_subset(self.values, np.array([4, 1, 1, 3]), np.random.default_rng(0)).toarray()
        self.assertEqual(result.sum(axis=1).tolist(), [4, 1, 0, 3])

    def test_choice_varies(self):
        rng = np.random.default_rng(0)
        seen = {tuple(random_symptom_subset(self.values, 2, rng).toarray()[0]) for _ in range(30)}
        self.assertGreater(len(seen), 1)

    def test_augmentation_shape_labels_and_original_rows(self):
        X_aug, y_aug = add_symptom_dropout(self.X, self.y, 2, 0)
        self.assertEqual(X_aug.shape, (12, 6))
        self.assertEqual(y_aug.tolist(), ["w", "x", "y", "z"] * 3)
        self.assertTrue(np.array_equal(X_aug.toarray()[:4], self.values))

    def test_labels_stay_aligned_with_their_rows(self):
        dense = add_symptom_dropout(self.X, self.y, 5, 0)[0].toarray()
        for start in range(0, len(dense), 4):
            self.assertTrue(np.all(dense[start:start + 4] <= self.values))

    def test_keep_few_symptoms_preserves_frame(self):
        reduced = keep_few_symptoms(self.X, 2, 0)
        self.assertEqual(list(reduced.columns), list("abcdef"))
        self.assertEqual(reduced.sum(axis=1).tolist(), [2, 1, 0, 2])


class TestModelComparison(unittest.TestCase):
    def test_report_compares_both_models_at_every_symptom_count(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.integers(0, 2, size=(300, 8)), columns=[f"s{i}" for i in range(8)], dtype=np.uint8)
        y = (X["s0"] * 2 + X["s1"]).astype(str)
        X_train, X_test, y_train, y_test = X[:240], X[240:], y[:240], y[240:]
        tree = fit_on_dropout_data(DecisionTreeClassifier(random_state=0), *add_symptom_dropout(X_train, y_train, 1, 0),
                                   list(X.columns))
        evaluation = {
            "test_metrics": {"macro_f1": 0.5},
            "top_k": top_k_accuracy(tree, X_test, y_test),
            "few_symptoms": few_symptom_metrics(tree, X_test, y_test, 0),
        }
        with tempfile.TemporaryDirectory() as folder:
            rows = compare_models(tree, {"final_fit_seconds": 1.0}, evaluation, X_train, y_train, X_test, y_test,
                                  1, 0, Path(folder))
            table = pd.read_csv(Path(folder) / "model_comparison.csv")
        self.assertEqual(len(rows), 10)
        self.assertEqual(set(table["model"]), {"DecisionTreeClassifier", "BernoulliNB"})
        self.assertEqual(sorted(set(table["symptoms_given"].astype(str))), ["1", "2", "3", "4", "all"])
        self.assertTrue(table[table["model"] == "DecisionTreeClassifier"]["explainable"].all())
        self.assertFalse(table[table["model"] == "BernoulliNB"]["explainable"].any())
        self.assertTrue(table["top_1_accuracy"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
