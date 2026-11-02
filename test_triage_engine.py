import csv
import time
import unittest
from pathlib import Path

from nlp_extractor import CLINICAL_FLAG_PHRASES, SymptomExtractor
from triage_engine import (
    CARE_PRIORITY,
    EMERGENCY,
    ROUTINE,
    RULES,
    UNKNOWN,
    URGENT,
    Rule,
    forward_chain,
    triage,
)

DATA_PATH = Path(__file__).parent / "data" / "cleaned_disease_symptom.csv"


def load_features():
    with open(DATA_PATH, encoding="utf-8") as f:
        return next(csv.reader(f))[1:]


class TestRuleBase(unittest.TestCase):
    def test_every_condition_can_become_true(self):
        # A misspelled condition would stop a rule firing, with no error anywhere
        derived = {fact for rule in RULES for fact in rule.then}
        known = set(load_features()) | set(CLINICAL_FLAG_PHRASES) | derived
        for rule in RULES:
            for fact in rule.all_of + rule.any_of:
                self.assertIn(fact, known, f"{rule.name} uses unknown fact '{fact}'")

    def test_every_clinical_flag_is_used_by_a_rule(self):
        used = {fact for rule in RULES for fact in rule.all_of + rule.any_of}
        self.assertEqual(set(CLINICAL_FLAG_PHRASES) - used, set())

    def test_rule_names_are_unique(self):
        names = [rule.name for rule in RULES]
        self.assertEqual(len(names), len(set(names)))

    def test_every_care_conclusion_is_in_the_priority_list(self):
        for rule in RULES:
            for fact in rule.then:
                if fact.startswith("needs_"):
                    self.assertIn(fact[len("needs_"):-len("_care")], CARE_PRIORITY)

    def test_rule_without_conditions_never_matches(self):
        self.assertFalse(Rule("X", "empty", ("y",)).matches({"anything"}))


class TestForwardChaining(unittest.TestCase):
    def test_chains_through_intermediate_facts(self):
        facts, trace = forward_chain({"sharp_chest_pain", "sweating"})
        fired = [step["rule"] for step in trace]
        self.assertLessEqual({"chest_pain_present", "cardiac_warning_sign", "cardiac_emergency",
                              "emergency", "needs_cardiac_care"}, facts)
        self.assertLess(fired.index("S1"), fired.index("E2"))
        self.assertLess(fired.index("E2"), fired.index("L1"))

    def test_each_rule_fires_once(self):
        _, trace = forward_chain({"sharp_chest_pain", "chest_tightness", "sweating", "nausea"})
        fired = [step["rule"] for step in trace]
        self.assertEqual(len(fired), len(set(fired)))

    def test_no_facts_fires_nothing(self):
        self.assertEqual(forward_chain(set()), (set(), []))

    def test_trace_explains_the_firing(self):
        _, trace = forward_chain({"seizures"})
        step = next(s for s in trace if s["rule"] == "E6")
        self.assertEqual(step["because"], ["seizures"])
        self.assertEqual(step["new_facts"], ["neuro_emergency"])

    def test_input_is_not_mutated(self):
        original = {"seizures"}
        forward_chain(original)
        self.assertEqual(original, {"seizures"})


class TestLevels(unittest.TestCase):
    def test_emergency_outranks_urgent(self):
        result = triage(["sharp_chest_pain", "sweating", "shortness_of_breath"])
        self.assertEqual(result["level"], EMERGENCY)
        self.assertTrue(result["emergency"])

    def test_chest_pain_alone_is_urgent(self):
        result = triage(["sharp_chest_pain"])
        self.assertEqual(result["level"], URGENT)
        self.assertFalse(result["emergency"])
        self.assertEqual(result["primary_care"], "cardiac")

    def test_recognized_minor_symptoms_are_routine(self):
        result = triage(["cough", "sore_throat"])
        self.assertEqual(result["level"], ROUTINE)
        self.assertIsNone(result["primary_care"])
        self.assertEqual(result["rules_fired"], [])

    def test_nothing_recognized_is_unknown_not_routine(self):
        self.assertEqual(triage([])["level"], UNKNOWN)

    def test_primary_care_follows_priority(self):
        result = triage(["sharp_chest_pain", "difficulty_breathing"])
        self.assertEqual(result["care_needed"][:2], ["cardiac", "respiratory"])
        self.assertEqual(result["primary_care"], "cardiac")

    def test_possible_meningitis(self):
        self.assertEqual(triage(["fever", "headache"])["level"], ROUTINE)
        self.assertEqual(triage(["fever", "neck_stiffness_or_tightness"])["level"], EMERGENCY)
        self.assertEqual(triage(["fever", "neck_pain"])["level"], EMERGENCY)
        self.assertEqual(triage(["headache", "neck_pain"])["level"], URGENT)
        self.assertEqual(triage(["neck_pain"])["level"], ROUTINE)


class TestFromPlainEnglish(unittest.TestCase):
    """The whole path a user's sentence takes: extraction, then triage."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = SymptomExtractor(load_features())

    def run_text(self, text):
        found = self.extractor.extract(text)
        return triage(found["symptoms"], found["clinical_flags"])

    def test_emergencies(self):
        cases = {
            "I have crushing chest pain and I cant breathe": "cardiac",
            "my chest hurts and I feel short of breath": "cardiac",
            "pressure on my chest and sweating": "cardiac",
            "I think I am having a heart attack": "cardiac",
            "I can barely breathe": "respiratory",
            "my father is not breathing": "respiratory",
            "he is unconscious": "neuro",
            "my baby has a high fever and is not waking up": "neuro",
            "one side of my body feels numb and I cant speak properly": "neuro",
            "I had a seizure this morning": "neuro",
            "I am bleeding heavily": "trauma",
            "I am coughing up blood": "trauma",
            "I got hit by a car and my leg is broken": "trauma",
            "she took an overdose": "toxicology",
            "I want to kill myself": "mental_health",
        }
        for text, care in cases.items():
            result = self.run_text(text)
            self.assertEqual(result["level"], EMERGENCY, text)
            self.assertEqual(result["primary_care"], care, text)

    def test_not_emergencies(self):
        cases = {
            "I have a fever and a headache": ROUTINE,
            "dry cough and sore throat since yesterday": ROUTINE,
            "no chest pain, just a mild cough": ROUTINE,
            "I have chest pain": URGENT,
            "I fainted yesterday": URGENT,
            "hello there": UNKNOWN,
        }
        for text, level in cases.items():
            self.assertEqual(self.run_text(text)["level"], level, text)

    def test_denied_emergency_does_not_fire(self):
        self.assertNotEqual(self.run_text("no heart attack, I have a cough")["level"], EMERGENCY)

    def test_triage_is_under_a_second(self):
        # The project promises instant triage
        start = time.perf_counter()
        self.run_text("I have crushing chest pain and I cant breathe")
        self.assertLess(time.perf_counter() - start, 1.0)


if __name__ == "__main__":
    unittest.main()
