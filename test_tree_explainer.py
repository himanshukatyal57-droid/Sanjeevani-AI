import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from tree_explainer import choose_follow_up, explain_tree_prediction

FEATURES = ["fever", "cough", "rash", "headache", "nausea"]


class TestExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = pd.DataFrame(rng.integers(0, 2, size=(400, len(FEATURES))), columns=FEATURES)
        y = (cls.X["fever"] * 4 + cls.X["cough"] * 2 + cls.X["rash"]).astype(str)
        y[rng.random(400) < 0.1] = "noise"
        cls.y = y
        cls.model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(cls.X, y)

    def explain(self, row):
        return explain_tree_prediction(self.model, FEATURES, list(row))

    def test_reaches_the_same_leaf_as_scikit_learn_for_every_input(self):
        for row in self.X.drop_duplicates().itertuples(index=False):
            frame = pd.DataFrame([row], columns=FEATURES)
            self.assertEqual(self.explain(row)["leaf_node"], int(self.model.apply(frame)[0]), row)

    def test_answers_match_the_input(self):
        row = [1, 0, 1, 0, 0]
        for step in self.explain(row)["steps"]:
            self.assertEqual(step["present"], row[FEATURES.index(step["symptom"])] == 1, step)

    def test_samples_only_shrink_going_down_the_tree(self):
        result = self.explain([1, 1, 0, 0, 0])
        samples = [step["samples"] for step in result["steps"]] + [result["leaf_samples"]]
        self.assertEqual(samples, sorted(samples, reverse=True))
        self.assertEqual(samples[0], 400)

    def test_values_are_plain_json_types(self):
        result = self.explain([0, 0, 0, 0, 0])
        self.assertIs(type(result["leaf_node"]), int)
        self.assertIs(type(result["steps"][0]["present"]), bool)
        self.assertIs(type(result["steps"][0]["samples"]), int)

    def test_tree_with_a_single_node_has_no_steps(self):
        model = DecisionTreeClassifier().fit(self.X, ["same"] * len(self.X))
        result = explain_tree_prediction(model, FEATURES, [0, 0, 0, 0, 0])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["leaf_samples"], 400)

    def test_other_model_types_have_no_path(self):
        self.assertIsNone(explain_tree_prediction(object(), FEATURES, [0, 0, 0, 0, 0]))


def path_of(*steps):
    return {"steps": [{"symptom": name, "present": present, "samples": 1} for name, present in steps]}


class TestFollowUp(unittest.TestCase):
    def test_asks_the_first_split_after_the_last_yes(self):
        path = path_of(("rash", False), ("cough", True), ("fever", False), ("nausea", False))
        self.assertEqual(choose_follow_up(path, set()), "fever")

    def test_skips_what_was_already_answered(self):
        path = path_of(("cough", True), ("fever", False), ("nausea", False))
        self.assertEqual(choose_follow_up(path, {"fever"}), "nausea")

    def test_path_ending_on_yes_uses_the_nearest_earlier_split(self):
        path = path_of(("rash", False), ("fever", False), ("cough", True))
        self.assertEqual(choose_follow_up(path, set()), "fever")

    def test_no_yes_on_the_path_starts_at_the_top(self):
        path = path_of(("rash", False), ("fever", False))
        self.assertEqual(choose_follow_up(path, set()), "rash")

    def test_never_asks_unsuitable_symptoms(self):
        path = path_of(("cough", True), ("regurgitation.1", False), ("symptoms_of_eye", False),
                       ("problems_with_orgasm", False), ("irritable_infant", False), ("vaginal_discharge", False),
                       ("penis_pain", False), ("painful_menstruation", False), ("fever", False))
        self.assertEqual(choose_follow_up(path, set()), "fever")

    def test_nothing_left_to_ask(self):
        self.assertIsNone(choose_follow_up(path_of(("cough", True), ("fever", False)), {"fever"}))
        self.assertIsNone(choose_follow_up(path_of(("cough", True)), set()))


if __name__ == "__main__":
    unittest.main()
