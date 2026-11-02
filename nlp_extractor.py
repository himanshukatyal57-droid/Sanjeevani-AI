"""
Rule-based symptom extraction from plain English (Unit 5 & Lab 9).

Pipeline: tokenize -> drop filler words -> Porter stem -> longest-phrase matching
against a lexicon -> negation scoping.

Results come in two groups:
  * symptoms       - model features, used to build the Decision Tree input
  * clinical_flags - facts the dataset has no column for (e.g. heavy bleeding, sudden
                     onset), for the triage rules only

Known limits: it does not tell who has the symptom ("my father had a stroke" still
flags stroke) or when ("my fever is gone" still reports fever). Both err towards
reporting a symptom, which is the safer mistake for triage.
"""
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

# Dropped before matching, so "my chest really hurts" matches the phrase "chest hurts".
# Lexicon phrases go through the same filter, so both sides stay consistent.
FILLER_WORDS = {
    "a", "an", "the", "my", "me", "mine", "i", "im", "ive", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "some", "this",
    "that", "it", "its", "any", "really", "very", "so", "quite", "too", "extremely", "pretty",
    "super", "left", "right", "both",
}

# Feature -> extra ways people describe it. The feature's own name is always matched.
SYMPTOM_PHRASES = {
    "sharp_chest_pain": ["chest pain", "pain in chest", "chest hurts", "chest ache", "chest sore"],
    "burning_chest_pain": ["burning in chest"],
    "chest_tightness": ["tight chest", "chest tight", "chest feels tight", "chest pressure",
                        "pressure in chest", "pressure on chest", "tightness in chest", "heavy chest",
                        "chest heaviness", "heaviness in chest", "chest discomfort"],
    "congestion_in_chest": ["chest congestion"],
    "difficulty_breathing": ["cant breathe", "cannot breathe", "can not breathe", "not breathing",
                             "stopped breathing", "trouble breathing", "trouble in breathing",
                             "difficulty in breathing", "problem in breathing", "hard to breathe",
                             "barely breathe", "struggling to breathe", "unable to breathe",
                             "not able to breathe", "wasnt able to breathe", "gasping for air",
                             "breathing difficulty", "breathing problem", "breathing trouble"],
    "shortness_of_breath": ["short of breath", "out of breath", "breathless", "breathlessness",
                            "cant catch breath", "winded"],
    "headache": ["head hurts", "head ache", "head pain", "pain in head", "head pounding", "pounding head",
                 "head killing me", "migraine", "throbbing pain"],
    "feeling_ill": ["not feeling well", "not feel well", "unwell", "feeling ill", "feel ill"],
    "frontal_headache": ["pain around forehead", "forehead pain", "forehead hurts"],
    "fever": ["feverish", "high temperature", "running temperature", "high temp"],
    "feeling_hot": ["feeling hot", "feel hot", "unusually hot", "burning up"],
    "feeling_cold": ["feel cold"],
    "chills": ["shivering", "rigors"],
    "sweating": ["sweats", "sweaty"],
    "weakness": ["weak"],
    "fatigue": ["tired", "exhausted", "exhaustion", "low energy", "no energy", "worn out", "drained"],
    "ache_all_over": ["body pain", "body ache", "aching all over", "hurts all over"],
    "muscle_pain": ["muscle ache", "sore muscles", "muscles hurt"],
    "muscle_cramps,_contractures,_or_spasms": ["muscle cramp", "muscle spasm"],
    "cramps_and_spasms": ["cramp", "spasm"],
    "bones_are_painful": ["bone pain", "bones hurt"],
    "nausea": ["nauseous", "nauseated", "queasy", "feel sick", "feeling sick", "sick to stomach",
               "feeling like vomiting", "feel like vomiting"],
    "vomiting": ["vomit", "threw up", "throw up", "throwing up", "thrown up", "puke", "puking", "puked"],
    "vomiting_blood": ["threw up blood", "throw up blood", "throwing up blood", "blood in vomit"],
    "dizziness": ["dizzy", "lightheaded", "light headed", "lightheadedness", "spinning sensation",
                  "room spinning", "head spinning", "vertigo", "loss of balance", "losing balance",
                  "off balance"],
    "fainting": ["pass out", "passed out", "passing out", "blacked out", "black out", "collapsed"],
    "seizures": ["seizure", "convulsion", "convulsing"],
    "slurring_words": ["slurred speech", "slurring", "slurred"],
    "difficulty_speaking": ["cant speak", "cannot speak", "cant talk", "trouble speaking"],
    "focal_weakness": ["weak on one side", "weakness on one side", "one side weak", "one side of body feels weak"],
    "cough": ["dry cough"],
    "coughing_up_sputum": ["coughing up phlegm", "coughing up mucus", "phlegm", "mucus", "sputum",
                           "wet cough", "productive cough", "mucus while coughing"],
    "hemoptysis": ["coughing up blood", "cough up blood", "coughing blood", "blood when cough", "blood in cough"],
    "sore_throat": ["throat pain", "throat hurts", "throat sore"],
    "throat_irritation": ["scratchy throat", "itchy throat", "dry throat"],
    "throat_feels_tight": ["throat tight", "tight throat", "throat closing"],
    "difficulty_in_swallowing": ["pain while swallowing", "difficulty swallowing", "trouble swallowing",
                                 "hard to swallow", "painful swallowing", "hurts to swallow"],
    "hoarse_voice": ["hoarse", "lost voice"],
    "nasal_congestion": ["stuffy nose", "blocked nose", "stuffed nose", "congested"],
    "coryza": ["runny nose", "running nose"],
    "nosebleed": ["nose bleeding", "bleeding nose", "nose bleed"],
    "decreased_appetite": ["loss of appetite", "lost appetite", "no appetite", "poor appetite", "not hungry"],
    "sharp_abdominal_pain": ["abdominal pain", "pain in abdomen", "stomach pain", "pain in stomach", "stomach ache",
                             "stomachache", "stomach hurts", "stomach cramp", "stomach discomfort", "belly pain",
                             "pain in belly", "belly hurts", "tummy pain", "pain in tummy", "tummy ache", "tummy hurts"],
    "burning_abdominal_pain": ["burning stomach", "burning in stomach"],
    "stomach_bloating": ["bloated", "bloating"],
    "diarrhea": ["diarrhoea", "loose motion", "loose stool", "watery stool"],
    "constipation": ["constipated"],
    "heartburn": ["acid reflux"],
    "blood_in_stool": ["bloody stool"],
    "blood_in_urine": ["bloody urine", "peeing blood"],
    "painful_urination": ["burning when urinating", "burning while urinating", "burning sensation while urinating",
                          "burning when peeing", "pain when urinating", "pain while urinating", "hurts to pee"],
    "frequent_urination": ["peeing a lot", "urinating frequently"],
    "palpitations": ["heart racing", "racing heart", "heart pounding", "heart fluttering"],
    "increased_heart_rate": ["rapid heartbeat", "fast heartbeat", "fast heart rate", "high heart rate",
                             "heart beating fast", "heart beating rapidly"],
    "diminished_vision": ["blurred vision", "blurry vision", "vision blurred", "loss of vision",
                          "poor vision", "cant see"],
    "skin_rash": ["rash", "hives"],
    "itching_of_skin": ["itchy skin", "itching", "itchy"],
    "jaundice": ["yellow skin", "yellow eyes"],
    "insomnia": ["cant sleep", "trouble sleeping", "sleeplessness"],
    "sleepiness": ["drowsy", "drowsiness", "sleepy"],
    "depression": ["depressed"],
    "anxiety_and_nervousness": ["anxiety", "anxious", "nervous", "panic attack"],
    "ear_pain": ["earache"],
    "toothache": ["tooth pain", "tooth hurts", "teeth hurt"],
    "facial_pain": ["face pain", "face hurts"],
    "painful_sinuses": ["sinus pain"],
    "recent_weight_loss": ["weight loss", "lost weight", "losing weight"],
    "weight_gain": ["gained weight"],
    "thirst": ["thirsty"],
    "mouth_dryness": ["dry mouth"],
    "paresthesia": ["tingling", "pins and needles"],
    "loss_of_sensation": ["numb", "numbness"],
}

# Triage facts the dataset has no column for. Never fed to the model.
CLINICAL_FLAG_PHRASES = {
    # Timing in the user's own words outranks the single duration answer on the form
    "sudden_onset": ["sudden", "suddenly", "abruptly", "just now", "just started", "this morning", "today",
                     "tonight", "last night", "yesterday", "minutes ago", "hours ago"],
    "unconscious": ["unconscious", "lost consciousness", "loss of consciousness", "unresponsive",
                    "not responding", "wont wake up", "not waking up"],
    "not_breathing": ["not breathing", "stopped breathing"],
    "severe_bleeding": ["bleeding heavily", "heavy bleeding", "severe bleeding", "bleeding badly",
                        "bleeding a lot", "lot of blood", "losing blood", "wont stop bleeding",
                        "bleeding wont stop", "internal bleeding"],
    "crushing_chest_pain": ["crushing chest pain", "crushing pain in chest", "chest feels crushed"],
    "suspected_heart_attack": ["heart attack", "myocardial infarction"],
    "suspected_stroke": ["stroke", "face drooping", "drooping face", "face droop", "facial droop",
                         "one side of body feels numb", "one side of body feels weak", "numb on one side"],
    "major_trauma": ["hit by car", "hit by bike", "car accident", "road accident", "accident", "fracture",
                     "fractured", "broken bone", "broken leg", "leg broken", "broken arm", "arm broken",
                     "head injury"],
    "low_blood_sugar": ["low blood sugar", "low sugar", "sugar low", "hypoglycemia"],
    "anaphylaxis": ["anaphylaxis", "anaphylactic", "throat closing"],
    "overdose_or_poisoning": ["overdose", "overdosed", "poisoned", "poisoning"],
    "self_harm_risk": ["suicidal", "kill myself", "want to die", "hurt myself", "end my life"],
}

# Body-part features follow a pattern, so their phrases are generated instead of listed.
BODY_PART_ALIASES = {
    "foot_or_toe": ["foot", "feet", "toe"],
    "hand_or_finger": ["hand", "finger"],
    "low_back": ["low back", "lower back"],
}
BODY_PARTS = [
    "ankle", "arm", "back", "elbow", "foot_or_toe", "gum", "groin", "hand_or_finger", "hip",
    "jaw", "joint", "knee", "leg", "low_back", "mouth", "muscle", "neck", "rib", "shoulder",
    "side", "tongue", "wrist",
]
BODY_PART_TEMPLATES = {
    "pain": ["{x} pain", "{x} hurts", "{x} ache", "{x} sore", "{x} feels sore", "sore {x}", "pain in {x}", "aching {x}"],
    "swelling": ["{x} swelling", "swollen {x}", "{x} swollen", "{x} feels swollen", "swelling in {x}", "swelling of {x}"],
    "weakness": ["{x} weakness", "weak {x}", "{x} weak", "{x} feels weak", "weakness in {x}"],
    "stiffness_or_tightness": ["{x} stiffness", "stiff {x}", "{x} stiff", "{x} feels stiff", "stiffness in {x}", "tight {x}", "{x} tight"],
    "cramps_or_spasms": ["{x} cramp", "{x} spasm", "cramp in {x}", "spasm in {x}"],
}

# A negator applies to symptoms after it, up to one of these or NEGATION_WINDOW tokens.
# "and" and commas end the scope on purpose: in "no fever, chest pain" it is safer to
# keep the chest pain than to assume it was denied. "or" continues it ("no fever or cough").
NEGATION_WINDOW = 4
NEGATORS = ["no", "not", "without", "never", "deny", "denies", "denied", "dont", "doesnt", "didnt",
            "havent", "hasnt", "hadnt", "isnt", "arent", "wasnt", "werent", "nor", "neither", "none"]
SCOPE_ENDERS = ["but", "however", "although", "though", "except", "yet", "and", "because", "since"]
# "not sure if this is chest pain" and "no relief from chest pain" do not deny anything.
PSEUDO_NEGATION_FOLLOWERS = ["sure", "certain", "know", "only", "just", "relief", "better", "improving", "doubt"]

TOKEN_PATTERN = re.compile(r"[a-z]+|[,.;:!?\n]")
Phrase = Tuple[Optional[str], ...]


@lru_cache(maxsize=4096)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def _tokenize(text: str) -> List[Optional[str]]:
    """Lowercase, drop filler words, and stem. Punctuation becomes None (a clause break)."""
    text = text.lower().replace("'", "").replace("’", "")
    tokens: List[Optional[str]] = []
    for raw in TOKEN_PATTERN.findall(text):
        if not raw.isalpha():
            tokens.append(None)
        elif raw not in FILLER_WORDS:
            tokens.append(_stem(raw))
    return tokens


_NEGATOR_STEMS = {_stem(w) for w in NEGATORS}
_SCOPE_ENDER_STEMS = {_stem(w) for w in SCOPE_ENDERS}
_PSEUDO_STEMS = {_stem(w) for w in PSEUDO_NEGATION_FOLLOWERS}


def _feature_to_phrase(feature: str) -> str:
    """'itchy_ear(s)' -> 'itchy ear', 'regurgitation.1' -> 'regurgitation'."""
    readable = re.sub(r"\(s\)|\.\d+$", "", feature)
    return readable.replace("_", " ").replace(",", "")


def _index_phrase(index: Dict[Phrase, Set[str]], phrase: str, target: str) -> None:
    key = tuple(_tokenize(phrase))
    if key:
        index.setdefault(key, set()).add(target)


def _build_symptom_index(features: Iterable[str]) -> Dict[Phrase, Set[str]]:
    known = set(features)
    index: Dict[Phrase, Set[str]] = {}
    for feature in known:
        _index_phrase(index, _feature_to_phrase(feature), feature)
    for feature, phrases in SYMPTOM_PHRASES.items():
        if feature in known:
            for phrase in phrases:
                _index_phrase(index, phrase, feature)
    for part in BODY_PARTS:
        for suffix, templates in BODY_PART_TEMPLATES.items():
            feature = f"{part}_{suffix}"
            if feature in known:
                for alias in BODY_PART_ALIASES.get(part, [part]):
                    for template in templates:
                        _index_phrase(index, template.format(x=alias), feature)
    return index


def _find_matches(tokens: List[Optional[str]], index: Dict[Phrase, Set[str]]) -> List[Tuple[int, int, Set[str]]]:
    """Longest phrase wins, so 'frontal headache' does not also report 'headache'."""
    lengths = sorted({len(key) for key in index}, reverse=True)
    taken: Set[int] = set()
    matches = []
    for length in lengths:
        for start in range(len(tokens) - length + 1):
            span = range(start, start + length)
            key = tuple(tokens[start:start + length])
            if key in index and not taken.intersection(span):
                taken.update(span)
                matches.append((start, length, index[key]))
    return matches


def _is_negated(tokens: List[Optional[str]], start: int, matched_positions: Set[int]) -> bool:
    for j in range(start - 1, max(start - 1 - NEGATION_WINDOW, -1), -1):
        token = tokens[j]
        if token is None or token in _SCOPE_ENDER_STEMS:
            return False
        # A negator inside another matched phrase ("no energy") belongs to that phrase
        if j in matched_positions or token not in _NEGATOR_STEMS:
            continue
        follower = tokens[j + 1] if j + 1 < len(tokens) else None
        if follower not in _PSEUDO_STEMS:
            return True
    return False


class SymptomExtractor:
    def __init__(self, feature_names: Iterable[str]):
        self.feature_names = list(feature_names)
        self._symptom_index = _build_symptom_index(self.feature_names)
        self._flag_index: Dict[Phrase, Set[str]] = {}
        for flag, phrases in CLINICAL_FLAG_PHRASES.items():
            for phrase in phrases:
                _index_phrase(self._flag_index, phrase, flag)

    def extract(self, text: str) -> dict:
        tokens = _tokenize(text)
        symptom_matches = _find_matches(tokens, self._symptom_index)
        flag_matches = _find_matches(tokens, self._flag_index)

        matched_positions: Set[int] = set()
        for start, length, _ in symptom_matches + flag_matches:
            matched_positions.update(range(start, start + length))

        symptoms, negated_symptoms = self._split_by_negation(tokens, symptom_matches, matched_positions)
        flags, negated_flags = self._split_by_negation(tokens, flag_matches, matched_positions)
        return {
            "symptoms": [f for f in self.feature_names if f in symptoms],
            "negated_symptoms": [f for f in self.feature_names if f in negated_symptoms],
            "clinical_flags": sorted(flags),
            "negated_flags": sorted(negated_flags),
        }

    @staticmethod
    def _split_by_negation(tokens, matches, matched_positions) -> Tuple[Set[str], Set[str]]:
        present: Set[str] = set()
        negated: Set[str] = set()
        for start, _, targets in matches:
            (negated if _is_negated(tokens, start, matched_positions) else present).update(targets)
        # "no fever yesterday, but I have a fever now" counts as present
        return present, negated - present
