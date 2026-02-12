"""
Patient simulator for the CaseSim Training GUI.

Generates patient responses based on case facts only.
"""

from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

from utils import safe_get, sanitize_text, is_placeholder, is_clean


# System prompt for patient responses - critical for non-leading behavior
PATIENT_SYSTEM_PROMPT = """You are a patient in a clinical training scenario. You must follow these rules EXACTLY:

CRITICAL RULES:
1. Base your responses primarily on the AVAILABLE FACTS below
2. Be conversational and helpful - patients want to help their doctor understand their condition
3. For general lifestyle questions (smoking, alcohol, diet, exercise, work, family) that aren't in the facts, MAKE UP reasonable plausible answers that fit the patient profile
4. For demographic details not in facts (age, occupation), infer something plausible from context or make up something reasonable
5. NEVER say "I don't know" or "I'm not sure" - instead either:
   - Make up a plausible detail that doesn't contradict the case, OR
   - Gently redirect: "That's not really relevant to why I'm here today" or "I'd rather focus on what's been bothering me"
6. Respond naturally as a patient would - conversational, not clinical
7. Be brief - usually 1-3 sentences

Do NOT:
- Hint at what the doctor should do
- Use legal terms (negligence, malpractice, consent, should have, failed to)
- Volunteer information not asked about
- Make up specific medical test results, lab values, or diagnosis dates
- Say "I don't know", "I'm not certain", or similar uncertainty phrases

AVAILABLE FACTS ABOUT ME:
{facts}

The doctor has asked: {question}

Respond as the patient in a natural, conversational way. If the question is about something not in the facts, either make up something plausible or redirect to your main concerns."""


class PatientSimulator:
    """Simulates patient responses constrained to case facts."""

    def __init__(self, case: dict, use_llm: bool = True):
        """
        Initialize the patient simulator.

        Args:
            case: The full case dict
            use_llm: Whether to use LLM for responses (default True)
        """
        self.case = case
        self.use_llm = use_llm
        self.fact_index = self._build_fact_index()

        # Dynamic state tracking
        self.elapsed_minutes: int = 0
        self.tests_ordered: set[str] = set()
        self.treatments_given: list[str] = []
        self.condition_trajectory: str = "stable"  # stable, worsening, improving

        # Initialize OpenAI client if using LLM
        if use_llm:
            api_key = os.environ.get("OPENAI_API_KEY")
            # Also check Streamlit secrets
            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get("OPENAI_API_KEY")
                except Exception:
                    pass
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = None
                self.use_llm = False
        else:
            self.client = None

    def _build_fact_index(self) -> dict[str, list[str]]:
        """
        Build an index of facts organized by topic.

        Returns:
            Dict mapping topics to lists of fact strings
        """
        facts: dict[str, list[str]] = {
            "demographics": [],
            "symptoms": [],
            "history": [],
            "medications": [],
            "allergies": [],
            "examination": [],
            "tests": [],
            "general": [],
        }

        initial_state = safe_get(self.case, "simulation.initial_state", {})

        # Demographics
        demo = initial_state.get("patient_demographics", {})
        if demo:
            age = demo.get("age_at_presentation", "")
            if not is_placeholder(age):
                facts["demographics"].append(f"I am {age} old")
            sex = demo.get("sex", "")
            if not is_placeholder(sex) and sex != "unknown":
                facts["demographics"].append(f"I am {sex}")
            social = demo.get("relevant_social_history", "")
            if not is_placeholder(social):
                facts["history"].append(social)

        # Chief complaint and HPI
        chief = initial_state.get("chief_complaint", "")
        if not is_placeholder(chief):
            facts["symptoms"].append(sanitize_text(chief))

        hpi = initial_state.get("history_of_present_illness", "")
        if not is_placeholder(hpi):
            facts["symptoms"].append(sanitize_text(hpi))

        # Past medical history
        pmh = initial_state.get("past_medical_history", [])
        if pmh and isinstance(pmh, list):
            for item in pmh:
                if item and not is_placeholder(item):
                    facts["history"].append(sanitize_text(item))

        # Medications
        meds = initial_state.get("medications", [])
        if meds and isinstance(meds, list):
            for med in meds:
                if med and not is_placeholder(med):
                    facts["medications"].append(sanitize_text(med))

        # Allergies
        allergies = initial_state.get("allergies", [])
        if allergies and isinstance(allergies, list):
            for allergy in allergies:
                if allergy and not is_placeholder(allergy):
                    facts["allergies"].append(sanitize_text(allergy))

        # Physical examination
        exam = initial_state.get("physical_examination", {}) or {}
        if exam:
            vital_signs = exam.get("vital_signs", {}) or {}
            if isinstance(vital_signs, dict):
                for key, value in vital_signs.items():
                    if value and not is_placeholder(str(value)):
                        facts["examination"].append(f"{key}: {value}")

            focused = exam.get("focused_exam", {}) or {}
            if isinstance(focused, dict):
                for key, value in focused.items():
                    if value and not is_placeholder(str(value)):
                        facts["examination"].append(f"{key}: {value}")

        # Evidence index - extract patient-relevant facts
        evidence = self.case.get("evidence_index", [])
        for item in evidence:
            text = item.get("text", "")
            if text and item.get("type") in ["FACTUAL_FINDING", "WITNESS_STATEMENT"]:
                # Only include patient-relevant facts, sanitized
                clean_text = sanitize_text(text)
                if clean_text and is_clean(clean_text):
                    facts["general"].append(clean_text)

        return facts

    def _get_relevant_facts(self, question: str, revealed_info: set[str]) -> list[str]:
        """
        Get facts relevant to the question.

        Args:
            question: The doctor's question
            revealed_info: Set of revealed request_ids

        Returns:
            List of relevant fact strings
        """
        question_lower = question.lower()
        relevant = []

        # Map question keywords to fact categories
        keyword_map = {
            "demographics": ["old", "age", "gender", "sex", "male", "female", "years"],
            "symptoms": ["pain", "hurt", "symptom", "feel", "complaint", "problem", "bother", "wrong",
                        "worse", "better", "start", "began", "onset", "duration", "long", "when",
                        "where", "location", "describe", "character", "intensity", "severe", "mild",
                        "sharp", "dull", "throb", "ache", "constant", "intermittent"],
            "history": ["medical history", "past medical", "pmh", "condition", "previous", "before", "diagnosed",
                       "surgery", "operation", "hospital", "admission", "illness", "disease", "chronic"],
            "medications": ["medication", "medicine", "taking", "prescription", "pills", "dose", "drug"],
            "allergies": ["allergy", "allergic", "react", "sensitive"],
            "examination": ["exam", "vital", "blood pressure", "heart rate", "temperature", "pulse",
                          "respiratory", "breathing", "oxygen", "saturation"],
        }

        # Family history should use generic response, not pull from history
        if any(kw in question_lower for kw in ["family", "relative", "parent", "mother", "father", "sibling", "hereditary"]):
            return []  # Let generic response handle this

        # Check which categories are relevant
        matched_any = False
        for category, keywords in keyword_map.items():
            if any(kw in question_lower for kw in keywords):
                relevant.extend(self.fact_index.get(category, []))
                matched_any = True

        # Check for test/imaging requests
        if any(kw in question_lower for kw in ["test", "result", "scan", "blood", "imaging", "lab", "x-ray", "ct", "mri", "ultrasound"]):
            relevant.extend(self._get_test_results(revealed_info))
            matched_any = True

        # If no specific category matched, provide ALL available facts
        # This is better than just repeating symptoms
        if not matched_any:
            # Collect all facts from all categories
            for category in ["symptoms", "history", "demographics", "medications", "allergies", "examination", "general"]:
                facts = self.fact_index.get(category, [])
                relevant.extend(facts)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for fact in relevant:
            if fact not in seen:
                seen.add(fact)
                unique.append(fact)

        return unique[:15]  # Limit to 15 facts

    def _get_test_results(self, revealed_info: set[str]) -> list[str]:
        """
        Get test results that have been revealed.

        Args:
            revealed_info: Set of revealed request_ids

        Returns:
            List of test result strings
        """
        results = []
        requestables = safe_get(self.case, "simulation.requestables", [])

        for req in requestables:
            req_id = req.get("request_id", "")
            if req_id in revealed_info:
                reveal = req.get("reveal", {})
                summary = reveal.get("result_summary", "")
                if summary and not is_placeholder(summary):
                    name = req.get("name", "Test")
                    results.append(f"{name}: {sanitize_text(summary)}")

        return results

    def respond_to_question(self, question: str, revealed_info: set[str]) -> str:
        """
        Generate a patient response to the doctor's question.

        Args:
            question: The doctor's question
            revealed_info: Set of revealed request_ids

        Returns:
            Patient's response string
        """
        # Get relevant facts
        facts = self._get_relevant_facts(question, revealed_info)

        # Use LLM for natural response if available (even without facts)
        if self.use_llm and self.client:
            return self._llm_response(question, facts)
        else:
            return self._template_response(question, facts)

    def _llm_response(self, question: str, facts: list[str]) -> str:
        """
        Generate response using LLM.

        Args:
            question: The doctor's question
            facts: List of relevant facts

        Returns:
            LLM-generated patient response
        """
        facts_str = "\n".join(f"- {fact}" for fact in facts)
        prompt = PATIENT_SYSTEM_PROMPT.format(facts=facts_str, question=question)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            content = response.choices[0].message.content

            # Safety check - ensure response is clean
            if not is_clean(content):
                return self._template_response(question, facts)

            return content
        except Exception:
            return self._template_response(question, facts)

    def _template_response(self, question: str, facts: list[str]) -> str:
        """
        Generate response using templates (fallback).

        Args:
            question: The doctor's question
            facts: List of relevant facts

        Returns:
            Template-based patient response
        """
        question_lower = question.lower()

        # Try to match specific question types
        if any(kw in question_lower for kw in ["old", "age", "years old", "how old"]):
            demo_facts = self.fact_index.get("demographics", [])
            # Look specifically for age fact
            for fact in demo_facts:
                if "old" in fact.lower() or "year" in fact.lower():
                    return fact
            # If no age found, make up something plausible
            return "I'm in my mid-40s."

        if any(kw in question_lower for kw in ["medication", "drug", "taking", "medicine", "prescription"]):
            med_facts = self.fact_index.get("medications", [])
            if med_facts:
                return f"I'm currently taking {', '.join(med_facts)}."
            return "I'm not on any medications that I know of."

        if any(kw in question_lower for kw in ["allergy", "allergic"]):
            allergy_facts = self.fact_index.get("allergies", [])
            if allergy_facts:
                return f"I'm allergic to {', '.join(allergy_facts)}."
            return "I don't have any allergies that I know of."

        if any(kw in question_lower for kw in ["history", "past", "condition", "medical history"]):
            history_facts = self.fact_index.get("history", [])
            if history_facts:
                return f"Yes, I have a history of {history_facts[0]}."
            return "Nothing major in my medical history."

        if any(kw in question_lower for kw in ["family", "relative", "parent", "sibling"]):
            return "I'm not aware of any major family medical history."

        if any(kw in question_lower for kw in ["smoke", "smoking", "tobacco", "cigarette"]):
            return "I don't smoke."

        if any(kw in question_lower for kw in ["alcohol", "drink", "drinking"]):
            return "I drink occasionally, nothing excessive."

        if any(kw in question_lower for kw in ["when", "start", "began", "onset", "how long"]):
            symptoms = self.fact_index.get("symptoms", [])
            if symptoms:
                return f"It started recently. {symptoms[0]}"
            return "It started a few days ago."

        if any(kw in question_lower for kw in ["worse", "better", "change", "improved"]):
            return "It's been about the same, maybe slightly worse."

        if any(kw in question_lower for kw in ["scale", "rate", "1 to 10", "one to ten"]):
            return "I'd say about 7 or 8 out of 10."

        if any(kw in question_lower for kw in ["nausea", "vomit", "sick to stomach"]):
            return "I've felt a bit nauseous but haven't vomited."

        if any(kw in question_lower for kw in ["fever", "temperature", "chills"]):
            return "I haven't noticed a fever, but I've felt unwell."

        if any(kw in question_lower for kw in ["sleep", "rest", "tired", "fatigue"]):
            return "I haven't been sleeping well because of the symptoms."

        # Try generic response first for lifestyle/unrelated questions
        generic = self._generic_response(question)
        if generic:
            return generic

        # If we have facts and it's a symptom-related question, use the first relevant one
        if facts:
            # Filter out very short facts that are just symptom names
            good_facts = [f for f in facts if len(f) > 20]
            if good_facts:
                return good_facts[0]
            return facts[0]

        # Final fallback - redirect to main concern
        symptoms = self.fact_index.get("symptoms", [])
        if symptoms:
            return f"I'd rather focus on what's been bothering me - {symptoms[0]}"
        return "That's not really relevant to why I'm here today."

    def _generic_response(self, question: str) -> str | None:
        """
        Generate a generic response for questions not covered by case facts.

        Args:
            question: The doctor's question

        Returns:
            A reasonable generic patient response, or None if no match
        """
        question_lower = question.lower()

        # Common lifestyle questions - provide reasonable generic answers
        if any(kw in question_lower for kw in ["smoke", "smoking", "cigarette", "tobacco"]):
            return "No, I don't smoke."

        if any(kw in question_lower for kw in ["alcohol", "drink", "drinking"]):
            return "I drink occasionally, maybe socially, but nothing excessive."

        if any(kw in question_lower for kw in ["drug", "recreational", "substance", "marijuana", "cocaine"]):
            return "No, I don't use any recreational drugs."

        if any(kw in question_lower for kw in ["exercise", "physical activity", "active"]):
            return "I try to stay moderately active, though I could probably do more."

        if any(kw in question_lower for kw in ["diet", "eat", "nutrition"]):
            return "I try to eat reasonably well, nothing too unusual."

        if any(kw in question_lower for kw in ["family", "relative", "parent", "mother", "father", "sibling"]):
            return "I'm not aware of any major illnesses running in my family."

        if any(kw in question_lower for kw in ["work", "job", "occupation", "employed"]):
            return "I work a regular job, nothing too physically demanding."

        if any(kw in question_lower for kw in ["stress", "anxiety", "worried", "mental"]):
            return "I've been feeling more stressed lately with everything going on."

        if any(kw in question_lower for kw in ["sleep", "rest", "tired", "fatigue", "exhausted"]):
            return "My sleep has been affected by my symptoms. I haven't been resting well."

        if any(kw in question_lower for kw in ["travel", "abroad", "foreign", "holiday"]):
            return "I haven't traveled anywhere unusual recently."

        # Symptom-related fallbacks
        if any(kw in question_lower for kw in ["pain", "hurt", "ache", "sore"]):
            # Try to relate to available symptoms
            symptoms = self.fact_index.get("symptoms", [])
            if symptoms:
                return f"Well, mainly what's been bothering me is this: {symptoms[0]}"
            return "It's been quite uncomfortable. That's part of why I'm here today."

        if any(kw in question_lower for kw in ["when", "start", "began", "how long", "duration"]):
            return "It started recently - over the past few days or so. Things have been getting worse."

        if any(kw in question_lower for kw in ["better", "worse", "change", "progress"]):
            return "If anything, it seems to be getting worse. That's why I came in."

        # Add more generic responses for common unrelated questions
        if any(kw in question_lower for kw in ["pet", "dog", "cat", "animal"]):
            return "I have a dog at home. But that's not really related to why I'm here."

        if any(kw in question_lower for kw in ["married", "spouse", "partner", "relationship"]):
            return "Yes, I'm married. My spouse is quite worried about me."

        if any(kw in question_lower for kw in ["children", "kids", "son", "daughter"]):
            return "I have kids, yes. They're concerned about me too."

        if any(kw in question_lower for kw in ["live", "living", "home", "house", "apartment"]):
            return "I live at home with my family. Nothing unusual about my living situation."

        if any(kw in question_lower for kw in ["insurance", "coverage", "pay"]):
            return "I have insurance. That shouldn't be an issue."

        # Return None to signal no match - caller will handle fallback
        return None

    def check_requestable(self, question: str) -> tuple[bool, str | None, dict | None]:
        """
        Check if the question is asking for a requestable (test, imaging, etc).

        Args:
            question: The doctor's question

        Returns:
            Tuple of (is_requestable, request_type, requestable_dict)
        """
        question_lower = question.lower()
        requestables = safe_get(self.case, "simulation.requestables", [])

        # Map question keywords to requestable types
        type_keywords = {
            "LAB": ["blood test", "lab", "blood work", "laboratory"],
            "IMAGING": ["scan", "ct", "mri", "x-ray", "xray", "ultrasound", "imaging", "radiograph"],
            "PATHOLOGY": ["pathology", "biopsy", "histology"],
            "CONSULT_NOTE": ["consult", "specialist", "referral", "opinion"],
            "OP_NOTE": ["operative", "surgery note", "procedure note"],
        }

        for req_type, keywords in type_keywords.items():
            if any(kw in question_lower for kw in keywords):
                # Find matching requestable
                for req in requestables:
                    if req.get("type") == req_type:
                        return True, req_type, req

        return False, None, None

    def get_requestable_info(self, request_id: str) -> str:
        """
        Get the reveal information for a requestable.

        Args:
            request_id: The request ID

        Returns:
            The result summary or a message if not available
        """
        requestables = safe_get(self.case, "simulation.requestables", [])

        for req in requestables:
            if req.get("request_id") == request_id:
                reveal = req.get("reveal", {})
                summary = reveal.get("result_summary", "")
                if summary and not is_placeholder(summary):
                    return sanitize_text(summary)
                return "The results are not available."

        return "That information isn't available."

    def get_available_requestables_list(self) -> list[str]:
        """
        Get a list of available requestable types (without revealing content).

        Returns:
            List of requestable names/types
        """
        requestables = safe_get(self.case, "simulation.requestables", [])
        return [
            f"{req.get('name', 'Unknown')} ({req.get('type', 'Unknown')})"
            for req in requestables
        ]

    def advance_time(self, minutes: int = 30):
        """
        Advance simulation time and update condition if needed.

        For urgent cases, if appropriate tests/treatments haven't been ordered
        within a reasonable time, the condition may worsen.

        Args:
            minutes: Minutes to advance (default 30)
        """
        self.elapsed_minutes += minutes

        # Get urgency level from case
        urgency = safe_get(
            self.case, "simulation.initial_state.urgency_level", "ROUTINE"
        )

        # If urgent/emergent and no imaging ordered after 2 hours, condition worsens
        if urgency in ["URGENT", "EMERGENT"]:
            if self.elapsed_minutes > 120 and not self._imaging_ordered():
                self.condition_trajectory = "worsening"
            elif self._appropriate_tests_ordered():
                self.condition_trajectory = "stable"

    def _imaging_ordered(self) -> bool:
        """Check if any imaging has been ordered."""
        requestables = safe_get(self.case, "simulation.requestables", [])
        for req in requestables:
            if req.get("type") == "IMAGING" and req.get("request_id") in self.tests_ordered:
                return True
        return False

    def _appropriate_tests_ordered(self) -> bool:
        """Check if appropriate tests have been ordered for the case."""
        # Check if court-expected tests were ordered
        decision_points = safe_get(self.case, "simulation.decision_points", [])
        for dp in decision_points:
            if dp.get("action_type") == "ORDER_TEST" and dp.get("is_malpractice_point"):
                expected = dp.get("expected_action_court", {})
                expected_desc = expected.get("description", "").lower()
                # Check if any ordered test matches expected
                for test_id in self.tests_ordered:
                    requestables = safe_get(self.case, "simulation.requestables", [])
                    for req in requestables:
                        if req.get("request_id") == test_id:
                            if req.get("name", "").lower() in expected_desc:
                                return True
        return len(self.tests_ordered) > 0

    def get_time_evolved_response(self, question: str) -> str | None:
        """
        Get response reflecting condition change over time.

        Args:
            question: The doctor's question

        Returns:
            Time-evolved response if condition has changed, None otherwise
        """
        question_lower = question.lower()

        # Only return time-evolved response for symptom questions when worsening
        if self.condition_trajectory == "worsening":
            symptom_keywords = ["pain", "feel", "symptom", "worse", "better", "how are you"]
            if any(kw in question_lower for kw in symptom_keywords):
                if self.elapsed_minutes > 60:
                    return "Actually, it's gotten worse. The pain is more severe now and I feel more unwell than before."
                elif self.elapsed_minutes > 120:
                    return "I'm feeling much worse now. The pain has really intensified and I'm starting to feel very sick."

        return None

    def record_test_order(self, request_id: str):
        """Record that a test was ordered."""
        self.tests_ordered.add(request_id)

    def record_treatment(self, treatment: str):
        """Record that a treatment was given."""
        self.treatments_given.append(treatment)

    def get_state_summary(self) -> dict:
        """Get summary of current simulation state."""
        return {
            "elapsed_minutes": self.elapsed_minutes,
            "tests_ordered": list(self.tests_ordered),
            "treatments_given": self.treatments_given,
            "condition_trajectory": self.condition_trajectory,
        }
