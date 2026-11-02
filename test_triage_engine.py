import json
import time
import unittest
from pathlib import Path

from nlp_extractor import CLINICAL_FLAG_PHRASES, SymptomExtractor
from triage_engine import (
    CARE_PRIORITY,
    EMERGENCY,
    LONG_STANDING_FACT,
    ROUTINE,
    RULES,
    UNKNOWN,
    URGENT,
    Rule,
    forward_chain,
    triage,
)

METADATA_PATH = Path(__file__).parent / "models" / "sanjeevani_model_metadata.json"


def load_features():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)["feature_names"]


class TestRuleBase(unittest.TestCase):
    def test_every_condition_can_become_true(self):
        # A misspelled condition would stop a rule firing, with no error anywhere
        derived = {fact for rule in RULES for fact in rule.then}
        known = set(load_features()) | set(CLINICAL_FLAG_PHRASES) | derived | {LONG_STANDING_FACT}
        for rule in RULES:
            for fact in rule.all_of + rule.any_of + rule.none_of:
                self.assertIn(fact, known, f"{rule.name} uses unknown fact '{fact}'")

    def test_absence_conditions_only_use_input_facts(self):
        # If a rule could conclude one of these, the result would depend on firing order
        derived = {fact for rule in RULES for fact in rule.then}
        for rule in RULES:
            self.assertEqual(set(rule.none_of) & derived, set(), rule.name)

    def test_every_clinical_flag_is_used_by_a_rule(self):
        used = {fact for rule in RULES for fact in rule.all_of + rule.any_of}
        self.assertEqual(set(CLINICAL_FLAG_PHRASES) - used, set())

    def test_descriptions_do_not_claim_an_absence_the_rule_never_checks(self):
        # Debug Mode shows these beside the facts, so a wrong claim is visible to evaluators
        for rule in RULES:
            if not rule.none_of:
                self.assertNotRegex(rule.description.lower(), r"\b(without|no|not reported)\b",
                                    f"{rule.name} describes an absence but has no none_of condition")

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


class TestDuration(unittest.TestCase):
    LONG = "More than a week"

    def test_long_standing_stroke_sign_is_urgent_not_routine(self):
        for symptom in ["slurring_words", "difficulty_speaking", "focal_weakness"]:
            result = triage([symptom], duration=self.LONG)
            self.assertEqual(result["level"], URGENT, symptom)
            self.assertEqual(result["primary_care"], "neuro", symptom)

    def test_every_other_answer_stays_an_emergency(self):
        for duration in ["Less than 24 hours", "1–3 days", "4–7 days", "", None, "not sure", "a week"]:
            self.assertEqual(triage(["slurring_words"], duration=duration)["level"], EMERGENCY, duration)

    def test_answer_matching_ignores_case_and_spaces(self):
        self.assertEqual(triage(["slurring_words"], duration="  more than a WEEK ")["level"], URGENT)

    def test_sudden_onset_overrides_the_form(self):
        result = triage(["slurring_words"], ["sudden_onset"], self.LONG)
        self.assertEqual(result["level"], EMERGENCY)
        self.assertIn("E13", [step["rule"] for step in result["rules_fired"]])

    def test_explicit_stroke_overrides_the_form(self):
        self.assertEqual(triage([], ["suspected_stroke"], self.LONG)["level"], EMERGENCY)

    def test_other_emergencies_ignore_duration(self):
        cases = [(["sharp_chest_pain", "sweating"], []), (["difficulty_breathing"], []), (["seizures"], []),
                 (["vomiting_blood"], []), (["fever", "neck_pain"], []), ([], ["severe_bleeding"]),
                 ([], ["major_trauma"]), ([], ["self_harm_risk"]), ([], ["unconscious"])]
        for symptoms, flags in cases:
            self.assertEqual(triage(symptoms, flags, self.LONG)["level"], EMERGENCY, symptoms + flags)

    def test_a_second_emergency_still_counts(self):
        self.assertEqual(triage(["slurring_words", "seizures"], duration=self.LONG)["level"], EMERGENCY)

    def test_duration_alone_is_not_a_symptom(self):
        self.assertEqual(triage([], duration=self.LONG)["level"], UNKNOWN)
        self.assertEqual(triage(["cough"], duration=self.LONG)["level"], ROUTINE)

    def test_long_standing_trace_never_says_sudden(self):
        for step in triage(["slurring_words"], duration=self.LONG)["rules_fired"]:
            self.assertNotIn("sudden", step["description"].lower(), step["rule"])

    def test_result_lists_the_starting_facts(self):
        result = triage(["slurring_words"], ["sudden_onset"], self.LONG)
        self.assertEqual(result["initial_facts"], sorted(["slurring_words", "sudden_onset", LONG_STANDING_FACT]))
        self.assertEqual(result["rules_total"], len(RULES))

    def test_trace_shows_what_had_to_be_absent(self):
        step = next(s for s in triage(["slurring_words"])["rules_fired"] if s["rule"] == "E5")
        self.assertEqual(step["required_absent"], [LONG_STANDING_FACT])


class TestFromPlainEnglish(unittest.TestCase):
    """The whole path a user's sentence takes: extraction, then triage."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = SymptomExtractor(load_features())

    def run_text(self, text, duration=None):
        found = self.extractor.extract(text)
        return triage(found["symptoms"], found["clinical_flags"], duration)

    def test_words_about_timing_outrank_the_duration_answer(self):
        long = "More than a week"
        cases = {
            "my speech has been slurred": URGENT,
            "I have trouble speaking": URGENT,
            "my speech suddenly became slurred": EMERGENCY,
            "I have had a cough for ages and today I am slurring": EMERGENCY,
            "I started slurring my words two hours ago": EMERGENCY,
            "I started slurring this morning": EMERGENCY,
            "my speech is slurred, it was not sudden": URGENT,
            "I think I had a stroke": EMERGENCY,
        }
        for text, level in cases.items():
            self.assertEqual(self.run_text(text, long)["level"], level, text)

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
            "I am coughing up blood": "respiratory",
            "I am vomiting blood": "trauma",
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

    def test_everyday_emergency_phrasing(self):
        for text in [
            "he isn't breathing", "she is not breathing", "my baby stopped breathing", "I can't breathe properly",
            "I am choking", "someone is choking", "my lips are turning blue", "my heart is paining and I am sweating",
            "severe pain in my chest and left arm", "sharp pain in chest spreading to my left arm",
            "I am having a stroke", "my face is drooping and my arm is weak", "I can't move my arm suddenly",
            "worst headache of my life", "sudden severe headache", "he collapsed",
            "she has fainted and won't wake up", "having fits", "I am having convulsions", "bleeding won't stop",
            "there is blood everywhere", "I was stabbed", "I got shot", "severe burns all over my body",
            "bitten by a snake", "snake bite", "electric shock", "he is drowning", "fell from the roof",
            "bad road accident", "I am vomiting blood", "coughing up blood", "pregnant and bleeding heavily",
            "my baby is not moving", "I swallowed poison", "I took too many pills", "I feel like ending it all",
            "I want to commit suicide", "I am going to die", "throat is swelling after a bee sting",
            "severe allergic reaction",
            # Still reported, on purpose: a worried person may really be having one
            "worried about a heart attack but no chest pain",
            # Without a past time it could be happening now
            "my father had a stroke", "my father had a stroke this morning", "he isn't breathing since 2 minutes",
        ]:
            self.assertEqual(self.run_text(text)["level"], EMERGENCY, text)

    def test_not_emergencies_despite_alarming_words(self):
        for text in [
            "food poisoning", "I think I have food poisoning from last night",
            "I had an accident last year but I am fine now", "my father had a heart attack in 2010",
            "I have a family history of stroke", "no history of chest pain or seizures",
            "I had a seizure two years ago", "I am fit and healthy", "I got a flu shot",
            "I don't have chest pain or shortness of breath", "no chest pain, no fever, no cough",
            "I do not have any difficulty breathing", "this shirt fits well",
        ]:
            self.assertNotEqual(self.run_text(text)["level"], EMERGENCY, text)

    def test_past_cue_needs_the_same_clause(self):
        self.assertEqual(self.run_text("I had a stroke last year and now I am slurring")["level"], EMERGENCY)
        self.assertEqual(self.run_text("chest pain for the last year")["level"], URGENT)
        self.assertEqual(self.run_text("chest pain since last year")["level"], URGENT)
        self.assertEqual(self.run_text("chest pain two years ago")["level"], UNKNOWN)

    def test_a_year_number_is_not_a_clause_break(self):
        found = self.extractor.extract("heart attack in 2010")
        self.assertEqual(found["clinical_flags"], [])
        self.assertEqual(found["negated_flags"], ["suspected_heart_attack"])

    def test_denied_emergency_does_not_fire(self):
        self.assertNotEqual(self.run_text("no heart attack, I have a cough")["level"], EMERGENCY)

    def test_triage_is_under_a_second(self):
        # The project promises instant triage
        start = time.perf_counter()
        self.run_text("I have crushing chest pain and I cant breathe")
        self.assertLess(time.perf_counter() - start, 1.0)


if __name__ == "__main__":
    unittest.main()
