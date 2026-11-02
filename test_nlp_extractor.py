"""Run with: python -m unittest -v"""
import json
import re
import time
import unittest
from pathlib import Path

from nlp_extractor import (
    BODY_PART_ALIASES,
    SYMPTOM_PHRASES,
    SymptomExtractor,
)

ROOT = Path(__file__).parent
METADATA_PATH = ROOT / "models" / "sanjeevani_model_metadata.json"


def load_features():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)["feature_names"]


class ExtractorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = load_features()
        cls.extractor = SymptomExtractor(cls.features)

    def symptoms(self, text):
        return set(self.extractor.extract(text)["symptoms"])

    def flags(self, text):
        return set(self.extractor.extract(text)["clinical_flags"])


class TestLexicon(ExtractorTestCase):
    def test_every_lexicon_target_is_a_real_feature(self):
        # A typo here would silently disable the synonyms for that symptom
        self.assertEqual(set(SYMPTOM_PHRASES) - set(self.features), set())
        for part in BODY_PART_ALIASES:
            self.assertTrue(any(f.startswith(part) for f in self.features), part)

    def test_every_feature_matches_its_own_name(self):
        for feature in self.features:
            readable = feature.replace("_", " ").replace(",", "").replace("(s)", "").replace(".1", "")
            self.assertIn(feature, self.symptoms(readable), readable)

    def test_target_column_is_not_a_symptom(self):
        self.assertEqual(self.symptoms("I have read about many diseases"), set())

    def test_duplicate_regurgitation_columns_both_set(self):
        self.assertLessEqual({"regurgitation", "regurgitation.1"}, self.symptoms("I have regurgitation"))


class TestEmergencyPhrasing(ExtractorTestCase):
    """Sentences the original substring matcher returned nothing for."""

    def test_crushing_chest_pain_and_cant_breathe(self):
        text = "I have crushing chest pain and I cant breathe"
        self.assertLessEqual({"sharp_chest_pain", "difficulty_breathing"}, self.symptoms(text))
        self.assertIn("crushing_chest_pain", self.flags(text))

    def test_chest_hurts_short_of_breath(self):
        text = "my chest hurts and I feel short of breath"
        self.assertLessEqual({"sharp_chest_pain", "shortness_of_breath"}, self.symptoms(text))

    def test_passed_out_and_bleeding(self):
        text = "I passed out and am bleeding heavily"
        self.assertIn("fainting", self.symptoms(text))
        self.assertIn("severe_bleeding", self.flags(text))

    def test_cant_with_curly_apostrophe(self):
        self.assertIn("difficulty_breathing", self.symptoms("I can’t breathe"))

    def test_not_breathing_is_an_emergency_not_a_denial(self):
        text = "my father is not breathing"
        self.assertIn("not_breathing", self.flags(text))
        self.assertIn("difficulty_breathing", self.symptoms(text))

    def test_flags_without_a_dataset_column(self):
        cases = {
            "he is unconscious": "unconscious",
            "I think I am having a heart attack": "suspected_heart_attack",
            "her face is drooping": "suspected_stroke",
            "I want to kill myself": "self_harm_risk",
            "she took an overdose": "overdose_or_poisoning",
        }
        for text, flag in cases.items():
            self.assertIn(flag, self.flags(text), text)


class TestRealWorldPhrasing(ExtractorTestCase):
    """Sentences written separately from the lexicon; each one exposed a gap."""

    def test_symptoms(self):
        cases = {
            "there is a tightness in my chest": "chest_tightness",
            "difficulty in breathing since morning": "difficulty_breathing",
            "I can barely breathe": "difficulty_breathing",
            "my chest is paining a lot": "sharp_chest_pain",
            "severe pain in my stomach": "sharp_abdominal_pain",
            "pain in my head for two days": "headache",
            "pressure on my chest and sweating": "chest_tightness",
            "I wasn't able to breathe last night": "difficulty_breathing",
            "not feeling well": "feeling_ill",
            "blood in my vomit": "vomiting_blood",
            "blood when I cough": "hemoptysis",
            "burning sensation while urinating": "painful_urination",
            "my heart is beating very fast": "increased_heart_rate",
            "I feel like I am going to pass out": "fainting",
        }
        for text, feature in cases.items():
            self.assertIn(feature, self.symptoms(text), text)

    def test_flags(self):
        cases = {
            "my baby has a high fever and is not waking up": "unconscious",
            "he collapsed and is not responding": "unconscious",
            "one side of my body feels numb and I cant speak properly": "suspected_stroke",
            "I got hit by a car and my leg is broken": "major_trauma",
            "I cut my hand and there is a lot of blood": "severe_bleeding",
            "my sugar is low and I am shaking": "low_blood_sugar",
        }
        for text, flag in cases.items():
            self.assertIn(flag, self.flags(text), text)


class TestNegation(ExtractorTestCase):
    def test_simple_negation(self):
        result = self.extractor.extract("no fever")
        self.assertEqual(result["symptoms"], [])
        self.assertEqual(result["negated_symptoms"], ["fever"])

    def test_negation_forms(self):
        for text in ["I don't have a fever", "without fever", "denies fever",
                     "I haven't had a fever", "I do not have fever", "never had a fever"]:
            self.assertNotIn("fever", self.symptoms(text), text)

    def test_but_ends_the_scope(self):
        result = self.extractor.extract("no fever but I threw up twice and my head is killing me")
        self.assertEqual(set(result["symptoms"]), {"vomiting", "headache"})
        self.assertEqual(result["negated_symptoms"], ["fever"])

    def test_or_continues_the_scope(self):
        result = self.extractor.extract("no fever or cough")
        self.assertEqual(set(result["negated_symptoms"]), {"fever", "cough"})

    def test_long_denied_lists_stay_denied(self):
        result = self.extractor.extract("no fever or cough or headache or nausea or chest pain")
        self.assertEqual(result["symptoms"], [])

    def test_very_long_input_stays_fast(self):
        start = time.perf_counter()
        self.extractor.extract("cough " * 20000)
        self.assertLess(time.perf_counter() - start, 2.0)

    def test_ambiguous_lists_keep_the_symptom(self):
        self.assertIn("sharp_chest_pain", self.symptoms("no fever, chest pain"))
        self.assertIn("sharp_chest_pain", self.symptoms("no fever and chest pain"))

    def test_uncertainty_is_not_denial(self):
        self.assertIn("sharp_chest_pain", self.symptoms("I'm not sure if this is chest pain"))
        self.assertIn("sharp_chest_pain", self.symptoms("no relief from chest pain"))

    def test_negator_inside_a_phrase_does_not_leak(self):
        found = self.symptoms("no energy chest pain")
        self.assertLessEqual({"fatigue", "sharp_chest_pain"}, found)

    def test_negated_flag_is_not_active(self):
        result = self.extractor.extract("no heart attack")
        self.assertEqual(result["clinical_flags"], [])
        self.assertEqual(result["negated_flags"], ["suspected_heart_attack"])

    def test_present_wins_over_negated(self):
        result = self.extractor.extract("no fever yesterday, but I have a fever now")
        self.assertEqual(result["symptoms"], ["fever"])
        self.assertEqual(result["negated_symptoms"], [])


class TestMatching(ExtractorTestCase):
    def test_word_forms(self):
        self.assertIn("cough", self.symptoms("I keep coughing"))
        self.assertIn("vomiting", self.symptoms("I vomited"))
        self.assertIn("dizziness", self.symptoms("I feel dizzy"))

    def test_longest_phrase_wins(self):
        self.assertEqual(self.symptoms("frontal headache"), {"frontal_headache"})
        self.assertEqual(self.symptoms("coughing up blood"), {"hemoptysis"})
        self.assertEqual(self.symptoms("vomiting blood"), {"vomiting_blood"})

    def test_body_part_templates(self):
        self.assertIn("knee_pain", self.symptoms("my left knee really hurts"))
        self.assertIn("foot_or_toe_swelling", self.symptoms("swollen feet"))
        self.assertIn("low_back_pain", self.symptoms("pain in my lower back"))
        self.assertIn("arm_weakness", self.symptoms("my arm feels weak"))
        self.assertIn("arm_weakness", self.symptoms("weak arm"))

    def test_no_substring_false_positives(self):
        self.assertEqual(self.symptoms("I went back to the hospital yesterday"), set())
        self.assertEqual(self.symptoms("the weather is hot and my coffee is cold"), set())

    def test_empty_and_nonsense_input(self):
        self.assertEqual(self.symptoms(""), set())
        self.assertEqual(self.symptoms("!!! ??? 123"), set())

    def test_frontend_chips_are_understood(self):
        # Every suggestion in Frontend/js/index.js, comma-separated as the form sends them
        chips = {
            "Fever": "fever", "Headache": "headache", "Cough": "cough", "Fatigue": "fatigue",
            "Body pain": "ache_all_over", "Nausea": "nausea", "Dizziness": "dizziness",
            "Sore throat": "sore_throat", "Throbbing pain": "headache",
            "Pain around forehead": "frontal_headache", "Chills": "chills", "Sweating": "sweating",
            "Feeling unusually hot": "feeling_hot", "Weakness": "weakness", "Body aches": "ache_all_over",
            "Dry cough": "cough", "Wet cough": "coughing_up_sputum", "Chest discomfort": "chest_tightness",
            "Mucus while coughing": "coughing_up_sputum", "Low energy": "fatigue", "Feeling exhausted": "fatigue",
            "Muscle weakness": "muscle_weakness", "Feeling like vomiting": "nausea",
            "Loss of appetite": "decreased_appetite", "Stomach discomfort": "sharp_abdominal_pain",
            "Feeling lightheaded": "dizziness", "Loss of balance": "dizziness", "Spinning sensation": "dizziness",
            "Blurred vision": "diminished_vision", "Muscle aches": "muscle_pain", "Joint pain": "joint_pain",
            "Back pain": "back_pain", "Neck pain": "neck_pain", "Pain while swallowing": "difficulty_in_swallowing",
            "Throat irritation": "throat_irritation", "Dry throat": "throat_irritation", "Hoarse voice": "hoarse_voice",
        }
        for chip, feature in chips.items():
            self.assertIn(feature, self.symptoms(chip), chip)
        self.assertLessEqual({"fever", "headache", "cough"}, self.symptoms("Fever, Headache, Cough"))

    def test_every_chip_in_the_frontend_files_is_recognized(self):
        # Reads the real files, so a new chip the AI can't understand fails here
        html = (ROOT / "Frontend" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "Frontend" / "js" / "index.js").read_text(encoding="utf-8")
        suggestions = re.search(r"const symptomSuggestions = \{(.*?)\n\};", script, re.DOTALL).group(1)
        chips = re.findall(r'data-symptom="([^"]+)"', html) + re.findall(r'"([^"]+)"', suggestions)
        self.assertGreater(len(chips), 30)
        for chip in chips:
            self.assertTrue(self.symptoms(chip), f"The AI does not recognize the chip '{chip}'")


if __name__ == "__main__":
    unittest.main()
