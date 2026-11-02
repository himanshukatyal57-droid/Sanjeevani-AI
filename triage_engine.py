"""
Forward chaining triage engine (IF-THEN production rules).

Starts from the facts the NLP module extracted and keeps firing every rule whose
conditions hold, adding its conclusions as new facts, until no rule can add anything.
Rules chain in three stages: symptoms -> intermediate signs -> emergency conditions ->
triage level and required care. The trace records each firing for Debug Mode.

The rules are illustrative, written for an academic demonstration, and have not been
clinically validated. They lean towards false alarms over missed emergencies.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

EMERGENCY = "EMERGENCY"
URGENT = "URGENT"
ROUTINE = "ROUTINE"
# Nothing was recognized. That is not evidence the person is fine, so it is not ROUTINE.
UNKNOWN = "UNKNOWN"

# Highest priority first; the first one needed is the primary destination for routing
CARE_PRIORITY = ["cardiac", "neuro", "respiratory", "trauma", "toxicology", "mental_health", "general"]

# The longest option on the symptom form. Anything else, including no answer, is treated
# as possibly recent, so an unclear duration can never lower a triage level.
LONG_STANDING_DURATION = "more than a week"
LONG_STANDING_FACT = "reported_long_standing"


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    then: Tuple[str, ...]
    all_of: Tuple[str, ...] = ()
    any_of: Tuple[str, ...] = ()
    # Input facts only, never concluded by a rule, so the firing order can't change the result
    none_of: Tuple[str, ...] = ()

    def matches(self, facts: Set[str]) -> bool:
        if not self.all_of and not self.any_of:
            return False
        return (all(f in facts for f in self.all_of)
                and (not self.any_of or any(f in facts for f in self.any_of))
                and not any(f in facts for f in self.none_of))

    def matched_facts(self, facts: Set[str]) -> List[str]:
        return [f for f in self.all_of + self.any_of if f in facts]


RULES = [
    # Stage 1: group symptoms into intermediate signs
    Rule("S1", "Any form of chest pain", ("chest_pain_present",),
         any_of=("sharp_chest_pain", "burning_chest_pain", "chest_tightness")),
    Rule("S2", "Symptom that commonly accompanies a cardiac event", ("cardiac_warning_sign",),
         any_of=("shortness_of_breath", "difficulty_breathing", "sweating", "nausea", "fainting",
                 "dizziness", "arm_pain", "jaw_pain", "shoulder_pain", "palpitations")),
    Rule("S3", "Sudden speech or one-sided weakness problem (FAST signs)", ("stroke_sign",),
         any_of=("suspected_stroke", "slurring_words", "difficulty_speaking", "focal_weakness")),
    Rule("S4", "Major blood loss", ("major_bleeding",),
         any_of=("severe_bleeding", "vomiting_blood", "hemoptysis")),
    Rule("S5", "Bleeding that needs prompt review", ("moderate_bleeding",),
         any_of=("blood_in_stool", "melena", "rectal_bleeding", "blood_in_urine",
                 "spotting_or_bleeding_during_pregnancy")),

    # Stage 2: identify emergency conditions
    Rule("E1", "Reported crushing chest pain or heart attack", ("cardiac_emergency",),
         any_of=("crushing_chest_pain", "suspected_heart_attack")),
    Rule("E2", "Chest pain with a cardiac warning sign", ("cardiac_emergency",),
         all_of=("chest_pain_present", "cardiac_warning_sign")),
    Rule("E3", "Not breathing, struggling to breathe, or anaphylaxis", ("respiratory_emergency",),
         any_of=("not_breathing", "difficulty_breathing", "anaphylaxis")),
    Rule("E4", "Allergic reaction affecting the throat", ("respiratory_emergency",),
         all_of=("allergic_reaction",), any_of=("throat_feels_tight", "throat_swelling", "swollen_tongue")),
    Rule("E5", "Possible stroke, not reported as long-standing", ("neuro_emergency",),
         all_of=("stroke_sign",), none_of=(LONG_STANDING_FACT,)),
    Rule("E12", "Stroke explicitly reported", ("neuro_emergency",), all_of=("suspected_stroke",)),
    Rule("E13", "Stroke sign described as sudden or recent", ("neuro_emergency",),
         all_of=("stroke_sign", "sudden_onset")),
    Rule("E6", "Unconscious or having a seizure", ("neuro_emergency",), any_of=("unconscious", "seizures")),
    Rule("E7", "Fever with a stiff or painful neck (possible meningitis)", ("neuro_emergency",),
         all_of=("fever",), any_of=("neck_stiffness_or_tightness", "neck_pain")),
    Rule("E8", "Major blood loss", ("bleeding_emergency",), all_of=("major_bleeding",)),
    Rule("E9", "Major trauma", ("trauma_emergency",), all_of=("major_trauma",)),
    Rule("E10", "Overdose or poisoning", ("poisoning_emergency",), all_of=("overdose_or_poisoning",)),
    Rule("E11", "Risk of self-harm", ("mental_health_emergency",), all_of=("self_harm_risk",)),

    # Stage 3: emergency conditions -> level and required care
    Rule("L1", "Cardiac emergency", ("emergency", "needs_cardiac_care"), all_of=("cardiac_emergency",)),
    Rule("L2", "Respiratory emergency", ("emergency", "needs_respiratory_care"), all_of=("respiratory_emergency",)),
    Rule("L3", "Neurological emergency", ("emergency", "needs_neuro_care"), all_of=("neuro_emergency",)),
    Rule("L4", "Bleeding emergency", ("emergency", "needs_trauma_care"), all_of=("bleeding_emergency",)),
    Rule("L5", "Trauma emergency", ("emergency", "needs_trauma_care"), all_of=("trauma_emergency",)),
    Rule("L6", "Poisoning emergency", ("emergency", "needs_toxicology_care"), all_of=("poisoning_emergency",)),
    Rule("L7", "Mental health emergency", ("emergency", "needs_mental_health_care"),
         all_of=("mental_health_emergency",)),

    # Not immediately life-threatening on their own, but should not wait
    Rule("U1", "Chest pain is at least urgent", ("urgent", "needs_cardiac_care"),
         all_of=("chest_pain_present",)),
    Rule("U2", "Shortness of breath", ("urgent", "needs_respiratory_care"), all_of=("shortness_of_breath",)),
    Rule("U3", "Fainting", ("urgent", "needs_neuro_care"), all_of=("fainting",)),
    Rule("U4", "Bleeding that needs prompt review", ("urgent", "needs_general_care"),
         all_of=("moderate_bleeding",)),
    Rule("U5", "Low blood sugar", ("urgent", "needs_general_care"), all_of=("low_blood_sugar",)),
    Rule("U6", "Headache with a stiff or painful neck", ("urgent", "needs_neuro_care"),
         all_of=("headache",), any_of=("neck_stiffness_or_tightness", "neck_pain")),
    Rule("U7", "Speech problem or one-sided weakness reported as long-standing", ("urgent", "needs_neuro_care"),
         all_of=("stroke_sign", LONG_STANDING_FACT)),
]


def forward_chain(initial_facts: Iterable[str]) -> Tuple[Set[str], List[dict]]:
    """Fire rules until nothing new can be concluded. Each rule fires at most once."""
    facts = set(initial_facts)
    fired: Set[str] = set()
    trace: List[dict] = []
    changed = True
    while changed:
        changed = False
        for rule in RULES:
            if rule.name not in fired and rule.matches(facts):
                fired.add(rule.name)
                added = [f for f in rule.then if f not in facts]
                facts.update(rule.then)
                changed = True
                trace.append({
                    "rule": rule.name,
                    "description": rule.description,
                    "because": rule.matched_facts(facts),
                    "required_absent": list(rule.none_of),
                    "concluded": list(rule.then),
                    "new_facts": added,
                })
    return facts, trace


def triage(symptoms: Iterable[str], clinical_flags: Iterable[str] = (), duration: Optional[str] = None) -> dict:
    observed = set(symptoms) | set(clinical_flags)
    initial_facts = set(observed)
    if duration and duration.strip().lower() == LONG_STANDING_DURATION:
        initial_facts.add(LONG_STANDING_FACT)
    facts, trace = forward_chain(initial_facts)

    if "emergency" in facts:
        level = EMERGENCY
    elif "urgent" in facts:
        level = URGENT
    elif observed:
        level = ROUTINE
    else:
        level = UNKNOWN

    care_needed = [care for care in CARE_PRIORITY if f"needs_{care}_care" in facts]
    return {
        "level": level,
        "emergency": level == EMERGENCY,
        "primary_care": care_needed[0] if care_needed else None,
        "care_needed": care_needed,
        "initial_facts": sorted(initial_facts),
        "rules_fired": trace,
        "rules_total": len(RULES),
    }
