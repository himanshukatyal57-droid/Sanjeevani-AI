import unittest

import numpy as np
import pandas as pd

from train_model import add_symptom_dropout, keep_few_symptoms, random_symptom_subset


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


if __name__ == "__main__":
    unittest.main()
