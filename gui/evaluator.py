"""
Response evaluator for the Medical Consultation Simulation.

Scores clinician responses against case criteria.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from utils import safe_get, sanitize_text


@dataclass
class ChecklistItem:
    """A single item in the evaluation checklist."""
    criterion: str
    met: bool = False
    completeness_score: float = 0.0  # 0.0-1.0 for partial credit
    reason: str = ""
    weight: int = 1
    source: str = ""
    is_primary: bool = False  # True for expected_action_court (the critical action)


@dataclass
class ReasoningQuality:
    """Assessment of clinical reasoning quality in response."""
    considers_differential: bool = False
    integrates_evidence: bool = False
    acknowledges_uncertainty: bool = False
    considers_urgency: bool = False
    quality_score: float = 0.0  # 0.0-1.0


@dataclass
class EvaluationResult:
    """Result of evaluating a clinician's response."""
    score: int  # 0, 1, or 2
    risk_flag: bool  # True if response resembles suboptimal approach
    checklist: list[ChecklistItem] = field(default_factory=list)
    feedback: str = ""
    court_outcome: dict = field(default_factory=dict)
    defendant_action: str = ""
    expected_action: str = ""
    what_should_have_happened: str = ""
    legal_standard: str = ""
    reasoning_quality: ReasoningQuality | None = None
    cognitive_error_avoided: str | None = None  # The cognitive error the defendant made
    score_valid: bool = True  # False if decision point not reached (score should be NA)
    deferral_reason: str = ""  # Why the score is not valid (if applicable)


# Prompt for LLM-based evaluation
EVALUATION_SYSTEM_PROMPT = """You are a medical-legal expert evaluating whether a clinician's response would meet the standard of care in a malpractice context.

CONTEXT: This is from a real malpractice case. The checklist items represent what the court determined should have been done.

CHECKLIST ITEMS (court-endorsed standard of care):
{checklist}

DEFENDANT'S ERROR (what the original doctor did wrong):
{defendant_action}

CLINICIAN'S RESPONSE TO EVALUATE:
{response}

EVALUATION RULES:
1. A criterion is MET if the response demonstrates equivalent clinical reasoning OR takes appropriate action that would achieve the same outcome
2. An emergency room/hospital referral MEETS criteria for tests and treatments that would be performed there
3. Using different terminology for the same concept is acceptable (e.g., "CT angiogram" = "CTPA")
4. Implicit understanding counts - if they recommend the right action, they've met the criterion even without stating why
5. DO NOT require exact keyword matches - evaluate the clinical substance

For each checklist item, output JSON:
[
  {{"criterion": "exact criterion text", "met": true/false, "reason": "brief explanation of clinical reasoning"}}
]

Be clinically reasonable. The question is: Would this response have avoided the malpractice outcome?"""


class ResponseEvaluator:
    """Evaluates clinician responses against case standards."""

    def __init__(self, case: dict, use_llm: bool = True):
        """
        Initialize the evaluator.

        Args:
            case: The full case dict
            use_llm: Whether to use LLM for nuanced evaluation
        """
        self.case = case
        self.use_llm = use_llm
        self.malpractice_points = self._extract_malpractice_points()

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

    def _extract_malpractice_points(self) -> list[dict]:
        """
        Extract decision points where is_malpractice_point is True.

        Returns:
            List of malpractice decision point dicts
        """
        decision_points = safe_get(self.case, "simulation.decision_points", [])
        return [
            dp for dp in decision_points
            if dp.get("is_malpractice_point", False)
        ]

    def build_checklist(self) -> list[ChecklistItem]:
        """
        Build evaluation checklist from malpractice decision points.

        The checklist extracts what the clinician should do from:
        1. expected_action_court.description
        2. explanation.what_should_have_happened
        3. court-endorsed options

        Returns:
            List of ChecklistItem objects
        """
        checklist = []
        seen_criteria = set()

        for dp in self.malpractice_points:
            # From expected_action_court - this is the PRIMARY criterion
            expected = dp.get("expected_action_court", {})
            if expected.get("description"):
                criterion = expected["description"]
                if criterion not in seen_criteria:
                    seen_criteria.add(criterion)
                    checklist.append(ChecklistItem(
                        criterion=criterion,
                        weight=2,
                        source="court_expected",
                        is_primary=True  # Mark as primary action
                    ))

            # From explanation
            explanation = dp.get("explanation", {})
            what_should = explanation.get("what_should_have_happened", "")
            if what_should and what_should not in seen_criteria:
                seen_criteria.add(what_should)
                checklist.append(ChecklistItem(
                    criterion=what_should,
                    weight=1,
                    source="explanation"
                ))

            # From court-endorsed options
            for opt in dp.get("options", []):
                if opt.get("is_court_endorsed"):
                    desc = opt.get("description", "")
                    if desc and desc not in seen_criteria:
                        seen_criteria.add(desc)
                        checklist.append(ChecklistItem(
                            criterion=desc,
                            weight=1,
                            source="endorsed_option"
                        ))

        return checklist

    def build_checklist_from_rubric(self) -> list[ChecklistItem]:
        """
        Build checklist from case-extracted scoring rubric.

        This uses the rubric criteria that were extracted from the court judgment,
        providing more precise evaluation criteria than the generic build_checklist.

        Returns:
            List of ChecklistItem objects from rubric, or fallback to build_checklist
        """
        checklist = []
        seen_criteria = set()

        for dp in self.malpractice_points:
            rubric = dp.get("scoring_rubric", {}) or {}
            criteria = rubric.get("criteria", []) or []

            for crit in criteria:
                # Handle both dict format {"criterion": "...", "points": 1}
                # and string format "criterion text"
                if isinstance(crit, dict):
                    criterion = crit.get("criterion", "")
                    weight = crit.get("points", 1)
                elif isinstance(crit, str):
                    criterion = crit
                    weight = 1
                else:
                    continue

                if criterion and criterion not in seen_criteria:
                    seen_criteria.add(criterion)
                    checklist.append(ChecklistItem(
                        criterion=criterion,
                        weight=weight,
                        source="rubric"
                    ))

        # Fall back to existing method if no rubric criteria found
        if not checklist:
            return self.build_checklist()

        return checklist

    def assess_reasoning_quality(self, response: str) -> ReasoningQuality:
        """
        Assess quality of clinical reasoning in the response.

        Evaluates whether the clinician demonstrates good clinical thinking:
        - Considering differential diagnoses
        - Integrating available evidence
        - Acknowledging uncertainty appropriately
        - Recognizing urgency when relevant

        Args:
            response: The clinician's response

        Returns:
            ReasoningQuality assessment
        """
        response_lower = response.lower()

        considers_differential = any(term in response_lower for term in [
            "differential", "could be", "alternatively", "rule out", "consider",
            "possibility", "possibilities", "other causes", "exclude"
        ])

        integrates_evidence = any(term in response_lower for term in [
            "based on", "given the", "labs show", "exam shows", "history of",
            "findings suggest", "results indicate", "symptoms suggest",
            "presentation suggests", "consistent with"
        ])

        acknowledges_uncertainty = any(term in response_lower for term in [
            "uncertain", "possible", "likely", "may be", "should confirm",
            "cannot rule out", "would need", "further testing", "to clarify"
        ])

        considers_urgency = any(term in response_lower for term in [
            "urgent", "immediate", "emergent", "time-sensitive", "cannot wait",
            "promptly", "as soon as", "without delay", "emergency", "stat"
        ])

        scores = [considers_differential, integrates_evidence,
                  acknowledges_uncertainty, considers_urgency]
        quality_score = sum(scores) / len(scores)

        return ReasoningQuality(
            considers_differential=considers_differential,
            integrates_evidence=integrates_evidence,
            acknowledges_uncertainty=acknowledges_uncertainty,
            considers_urgency=considers_urgency,
            quality_score=quality_score
        )

    def _get_cognitive_error(self) -> str | None:
        """
        Get the cognitive error type the defendant made.

        Returns:
            Cognitive error type string or None
        """
        for dp in self.malpractice_points:
            error_type = dp.get("reasoning_error_type")
            if error_type:
                return error_type
        return None

    def _get_defendant_action(self) -> str:
        """
        Get description of what the defendant did wrong.

        Returns:
            Description of defendant's action
        """
        for dp in self.malpractice_points:
            actual = dp.get("actual_action_defendant", {})
            if actual.get("description"):
                return actual["description"]
        return ""

    def _get_expected_action(self) -> str:
        """
        Get description of what the court expected.

        Returns:
            Description of expected action
        """
        for dp in self.malpractice_points:
            expected = dp.get("expected_action_court", {})
            if expected.get("description"):
                return expected["description"]
        return ""

    def _get_what_should_have_happened(self) -> str:
        """
        Get explanation of what should have happened.

        Returns:
            Explanation string
        """
        for dp in self.malpractice_points:
            explanation = dp.get("explanation", {})
            if explanation.get("what_should_have_happened"):
                return explanation["what_should_have_happened"]
        return ""

    def _get_legal_standard(self) -> str:
        """
        Get the legal standard applied.

        Returns:
            Legal standard string
        """
        for dp in self.malpractice_points:
            explanation = dp.get("explanation", {})
            if explanation.get("legal_standard_applied"):
                return explanation["legal_standard_applied"]
        return ""

    def _check_decision_point_reached(self, response: str) -> tuple[bool, str]:
        """
        Check if the clinician's response reached the decision point where malpractice occurred.

        Some cases have malpractice at a later phase (e.g., surgical consent, intra-operative
        decisions). If the AI appropriately defers to that phase (e.g., "after MRI results",
        "will discuss surgical options once..."), the current response cannot be fairly
        evaluated against those criteria.

        Returns:
            Tuple of (decision_point_reached: bool, deferral_reason: str)
            - (True, "") means the response can be evaluated
            - (False, reason) means the response appropriately deferred
        """
        response_lower = response.lower()

        # Check what kind of decision point this case involves
        decision_requires_surgical = False
        decision_requires_immediate = False
        decision_requires_treatment = False

        for dp in self.malpractice_points:
            prompt = dp.get("prompt", "").lower()
            context = dp.get("clinical_context", "").lower()
            expected = dp.get("expected_action_court", {})
            expected_desc = expected.get("description", "").lower() if isinstance(expected, dict) else ""

            # Surgical/treatment decision points
            if any(x in prompt for x in ["surgical option", "surgical technique", "surgery should",
                                          "discussed with the patient", "what treatment"]):
                decision_requires_surgical = True

            # Treatment consent/alternative decision points
            if any(x in expected_desc for x in ["consent", "alternative", "both technique",
                                                  "treatment option", "hearing aid"]):
                decision_requires_treatment = True

            # Immediate action decision points
            if any(x in expected_desc for x in ["immediate", "urgently", "emergency", "stat"]):
                decision_requires_immediate = True

        # Check if response shows appropriate deferral patterns

        # Deferral pattern 1: Ordering tests before making treatment decision
        orders_tests = any(x in response_lower for x in [
            "order an mri", "order mri", "get an mri", "mri scan", "mri of",
            "order a ct", "ct scan", "imaging", "blood work", "lab tests",
            "further testing", "diagnostic workup", "blood test", "x-ray",
            "ultrasound", "ecg", "ekg", "echocardiogram", "order some tests",
            "run some tests", "like to order", "recommend ordering"
        ])

        defers_to_results = any(x in response_lower for x in [
            "after the results", "once we have the results", "after mri",
            "pending results", "based on the results", "results will help",
            "help determine", "will guide", "will inform our", "before deciding",
            "once the results are", "when the results", "results are available",
            "will communicate the results"
        ])

        # Deferral pattern 2: Appropriately staging care
        defers_decision = any(x in response_lower for x in [
            "we'll discuss", "will discuss", "can discuss",
            "consider surgical options later", "if conservative treatment fails",
            "if symptoms persist", "reassess", "follow-up to discuss",
            "before considering surgery", "discuss treatment options",
            "further recommendations", "next steps", "follow-up appointment"
        ])

        # Deferral pattern 3: Starting conservative before surgical
        starts_conservative = any(x in response_lower for x in [
            "start with conservative", "try conservative", "physical therapy first",
            "begin with", "let's start with medication", "non-surgical approach first",
            "pain management", "over-the-counter", "nsaids"
        ])

        # Does the response actually make a definitive treatment decision?
        makes_definitive_decision = any(x in response_lower for x in [
            "i recommend surgery", "recommend surgical", "you need surgery",
            "surgery is indicated", "should have surgery", "surgical intervention",
            "i would recommend proceeding with", "the surgery i recommend",
            "i'm recommending", "my recommendation is to proceed",
            "we should proceed with", "you will need"
        ])

        # Does the response recognize an emergency?
        recognizes_emergency = any(x in response_lower for x in [
            "emergency", "immediately", "right now", "urgent", "stat",
            "call 911", "go to the er", "emergency room", "admitted",
            "this is an emergency", "life-threatening", "critical"
        ])

        # Determine if decision point was reached

        # For immediate action cases
        if decision_requires_immediate:
            if recognizes_emergency:
                return (True, "")  # AI recognized emergency, evaluate normally
            if orders_tests and not recognizes_emergency:
                return (False, "AI in diagnostic workup phase; case required immediate action recognition")

        # For surgical/treatment decision cases
        if decision_requires_surgical or decision_requires_treatment:
            # If AI makes a definitive decision: evaluable
            if makes_definitive_decision:
                return (True, "")

            if orders_tests and (defers_to_results or defers_decision):
                return (False, "AI appropriately ordered diagnostic workup before discussing treatment options")

            if starts_conservative and defers_decision:
                return (False, "AI appropriately started conservative management before considering definitive treatment")

            # If AI just orders tests without reaching treatment discussion
            if orders_tests and not makes_definitive_decision:
                return (False, "AI in diagnostic workup phase; has not reached treatment decision point")

        # Default: decision point reached, evaluate normally
        return (True, "")

    def _check_matches_defendant(self, response: str, checklist: list[ChecklistItem] | None = None) -> bool:
        """
        Check if the response resembles the defendant's conduct.

        IMPORTANT: This check is now secondary to checklist scoring. If the checklist
        shows the response met the key criteria, we should NOT flag it as matching
        defendant conduct just because of keyword overlap.

        Args:
            response: The clinician's response
            checklist: Optional scored checklist - if provided and criteria are met,
                      we skip the keyword matching to avoid false positives

        Returns:
            True if response matches defendant conduct
        """
        # If checklist is provided and shows good performance, don't flag as defendant match
        # This prevents false positives from naive keyword matching
        if checklist:
            met_count = sum(1 for item in checklist if item.met)
            total_count = len(checklist)
            if total_count > 0 and met_count / total_count >= 0.5:
                # Response met at least half the criteria - clearly not matching defendant
                return False

        response_lower = response.lower()

        for dp in self.malpractice_points:
            actual = dp.get("actual_action_defendant", {})

            # Check for defendant's description keywords
            desc = actual.get("description", "").lower()
            if desc:
                # Look for explicit phrases that indicate matching defendant conduct
                # These are more specific than single keywords
                defendant_phrases = [
                    "reassure without",
                    "proceed without informing",
                    "not discuss risk",
                    "skip the",
                    "omit the",
                    "without warning about",
                    "without informing about",
                    "no need for",  # dismissive of tests/referrals
                ]
                for phrase in defendant_phrases:
                    if phrase in desc and phrase in response_lower:
                        return True

        # Removed the overly aggressive keyword matching that caused false positives
        # The old logic matched if 3+ words from defendant option appeared anywhere
        # in the response, even in completely different contexts

        return False

    def _score_checklist_llm(self, response: str, checklist: list[ChecklistItem]) -> list[ChecklistItem]:
        """
        Score checklist items using LLM.

        Args:
            response: The clinician's response
            checklist: List of checklist items

        Returns:
            Updated checklist with met/reason fields
        """
        if not self.client or not checklist:
            return self._score_checklist_rule(response, checklist)

        checklist_str = "\n".join(
            f"- {item.criterion}" for item in checklist
        )

        prompt = EVALUATION_SYSTEM_PROMPT.format(
            checklist=checklist_str,
            defendant_action=self._get_defendant_action(),
            response=response
        )

        try:
            # Use gpt-4o for better accuracy (gpt-4o-mini was too unreliable)
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for consistency
                max_tokens=1000,
            )

            content = completion.choices[0].message.content

            # Parse JSON response
            import json

            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', content)
            if match:
                results = json.loads(match.group())

                # Update checklist items - match by criterion text for robustness
                result_by_criterion = {}
                for r in results:
                    crit = r.get("criterion", "").lower().strip()
                    result_by_criterion[crit] = r

                for item in checklist:
                    item_crit = item.criterion.lower().strip()
                    # Try exact match first
                    if item_crit in result_by_criterion:
                        result = result_by_criterion[item_crit]
                        item.met = result.get("met", False)
                        item.reason = result.get("reason", "")
                    else:
                        # Fallback to positional match
                        for r in results:
                            if r.get("criterion", "").lower().strip().startswith(item_crit[:30]):
                                item.met = r.get("met", False)
                                item.reason = r.get("reason", "")
                                break

                return checklist

        except Exception as e:
            # Log error for debugging but continue
            import sys
            print(f"LLM evaluation error: {e}", file=sys.stderr)

        # Fallback to rule-based
        return self._score_checklist_rule(response, checklist)

    def _score_checklist_rule(self, response: str, checklist: list[ChecklistItem]) -> list[ChecklistItem]:
        """
        Score checklist items using rule-based matching.

        Args:
            response: The clinician's response
            checklist: List of checklist items

        Returns:
            Updated checklist with met/reason fields
        """
        response_lower = response.lower()

        # Check if response includes appropriate hospital/ER referral
        # This is important because many malpractice cases involve failure to refer
        # and an ER referral would result in appropriate workup being done
        hospital_referral_patterns = [
            "emergency room", "emergency department", "go to the er", "go to er",
            "hospital immediately", "admit to hospital", "hospital for evaluation",
            "a&e", "emergency medical", "call 911", "call ambulance", "seek emergency",
            "urgent evaluation", "urgent care", "go to hospital"
        ]
        has_appropriate_referral = any(p in response_lower for p in hospital_referral_patterns)

        # Meta-criteria that should be evaluated differently
        # These are process criteria, not clinical action criteria
        meta_criteria_keywords = ["document", "clinical reasoning", "record", "note"]

        for item in checklist:
            criterion_lower = item.criterion.lower()

            # Check if this is a meta-criterion (documentation/process)
            is_meta_criterion = any(kw in criterion_lower for kw in meta_criteria_keywords)

            if is_meta_criterion:
                # For documentation criteria, check if response shows clear reasoning
                reasoning_indicators = [
                    "based on", "given", "therefore", "because", "considering",
                    "assessment", "differential", "plan", "recommend", "suspect",
                    "concern", "rule out", "indicates", "suggests", "likely"
                ]
                matches = sum(1 for ind in reasoning_indicators if ind in response_lower)
                if matches >= 3:
                    item.met = True
                    item.reason = f"Response demonstrates clinical reasoning ({matches} reasoning indicators)"
                else:
                    item.met = False
                    item.reason = f"Limited clinical reasoning shown ({matches} reasoning indicators)"
                continue

            # Extract key concepts from criterion
            # Remove common words and look for key terms
            stop_words = {"the", "a", "an", "to", "of", "and", "or", "for", "in", "with", "should", "have", "been", "was", "were", "that", "this", "would", "could"}
            key_words = [w for w in criterion_lower.split() if w not in stop_words and len(w) > 3]

            # Also look for medical terms (CT, MRI, etc. that are short but important)
            medical_short_terms = ["ct", "mri", "ecg", "ekg", "cbc", "bmp", "cmp", "abg", "esr", "crp"]
            for term in medical_short_terms:
                if term in criterion_lower and term not in key_words:
                    key_words.append(term)

            # Check how many key words appear in response
            matches = sum(1 for w in key_words if w in response_lower)

            # Check if criterion involves ordering tests/workup OR emergency treatments
            # If so, an appropriate hospital referral should count as meeting the criterion
            # because the hospital would perform the necessary tests or administer treatments
            is_test_ordering_criterion = any(kw in criterion_lower for kw in [
                "order", "test", "imaging", "d-dimer", "ctpa", "scan", "x-ray",
                "bloodwork", "labs", "workup", "investigation"
            ])

            # Check if criterion involves emergency treatments that would be done at hospital
            # A primary care physician referring to ER is the correct action for these
            is_emergency_treatment_criterion = any(kw in criterion_lower for kw in [
                "thrombolytic", "angioplasty", "catheterization", "catheter",
                "aspirin", "antibiotic", "antidote", "transfusion", "intubat",
                "defibrillat", "resuscitat", "surgery", "surgical", "operative",
                "laparoscop", "debride", "drain", "stent"
            ])

            # Check if criterion involves specialist referral - ER can provide or arrange these
            is_referral_criterion = any(kw in criterion_lower for kw in [
                "refer", "consult", "specialist", "cardiology", "neurology",
                "neurosurg", "hospitali", "admit"
            ])

            # Consider criterion met if:
            # 1. >40% of key words present (direct mention), OR
            # 2. Appropriate hospital referral for test-ordering or emergency treatment criteria
            if key_words and matches >= len(key_words) * 0.4:
                item.met = True
                item.reason = f"Response addresses key concepts ({matches}/{len(key_words)} key terms)"
            elif has_appropriate_referral and is_test_ordering_criterion:
                # Hospital/ER referral would result in appropriate testing
                item.met = True
                item.reason = "Appropriate hospital referral would result in necessary workup"
            elif has_appropriate_referral and is_emergency_treatment_criterion:
                # Hospital/ER referral would result in appropriate emergency treatment
                item.met = True
                item.reason = "Appropriate hospital referral would result in necessary treatment"
            elif has_appropriate_referral and is_referral_criterion:
                # Hospital/ER referral fulfills specialist referral criteria
                item.met = True
                item.reason = "Hospital referral provides access to required specialist care"
            else:
                item.met = False
                item.reason = f"Key concepts not adequately addressed ({matches}/{len(key_words)} key terms)"

        return checklist

    def evaluate_response(self, response: str, chat_history: list[dict] | None = None) -> EvaluationResult:
        """
        Evaluate the clinician's response.

        This evaluation determines: Would this response have avoided the malpractice outcome?

        Args:
            response: The clinician's substantive response
            chat_history: Optional chat history for context

        Returns:
            EvaluationResult with score, checklist, and court outcome
        """
        # Check if the decision point was reached (or if AI appropriately deferred)
        decision_reached, deferral_reason = self._check_decision_point_reached(response)

        # Build checklist - prefer rubric-based if available
        checklist = self.build_checklist_from_rubric()

        # Score checklist items FIRST (before defendant match check)
        if self.use_llm and self.client:
            checklist = self._score_checklist_llm(response, checklist)
        else:
            checklist = self._score_checklist_rule(response, checklist)

        # Check if response matches defendant conduct
        # Pass the scored checklist to avoid false positives when criteria are met
        defendant_match = self._check_matches_defendant(response, checklist)

        # Assess reasoning quality
        reasoning_quality = self.assess_reasoning_quality(response)

        # Calculate score with partial credit
        # Use completeness_score if available, otherwise binary met
        weighted_score = 0.0
        weighted_total = 0.0
        for item in checklist:
            if item.completeness_score > 0:
                weighted_score += item.weight * item.completeness_score
            elif item.met:
                weighted_score += item.weight
            weighted_total += item.weight

        if weighted_total == 0:
            weighted_total = 1

        score_ratio = weighted_score / weighted_total

        # Factor in reasoning quality (10% of final score)
        adjusted_ratio = score_ratio * 0.9 + reasoning_quality.quality_score * 0.1

        # Determine final score
        if defendant_match or adjusted_ratio < 0.25:
            score = 0
            risk_flag = True
        elif adjusted_ratio < 0.5:
            score = 1
            risk_flag = False
        else:
            score = 2
            risk_flag = defendant_match

        # Generate feedback
        feedback = self._generate_feedback(score, checklist, defendant_match, reasoning_quality)

        # If decision point not reached, add note to feedback
        if not decision_reached:
            feedback = f"**Note:** {deferral_reason}\n\n{feedback}"

        # Get court outcome
        court_outcome = safe_get(self.case, "simulation.end_state.legal_outcome", {})

        return EvaluationResult(
            score=score,
            risk_flag=risk_flag,
            checklist=checklist,
            feedback=feedback,
            court_outcome=court_outcome,
            defendant_action=self._get_defendant_action(),
            expected_action=self._get_expected_action(),
            what_should_have_happened=self._get_what_should_have_happened(),
            legal_standard=self._get_legal_standard(),
            reasoning_quality=reasoning_quality,
            cognitive_error_avoided=self._get_cognitive_error(),
            score_valid=decision_reached,
            deferral_reason=deferral_reason,
        )

    def _generate_feedback(
        self,
        score: int,
        checklist: list[ChecklistItem],
        defendant_match: bool,
        reasoning_quality: ReasoningQuality | None = None
    ) -> str:
        """
        Generate explanatory feedback.

        Args:
            score: The score (0, 1, or 2)
            checklist: The evaluated checklist
            defendant_match: Whether response matched defendant conduct
            reasoning_quality: Assessment of clinical reasoning quality

        Returns:
            Feedback string
        """
        lines = []

        if score == 2:
            lines.append("**Legally defensible response.** You addressed the key clinical issues that the court identified as standard of care.")
        elif score == 1:
            lines.append("**Partially defensible.** Some court-endorsed elements were addressed, but key issues identified in the judgment were missed.")
        else:
            lines.append("**Would likely not avoid malpractice finding.** Important clinical considerations identified by the court were not addressed.")

        if defendant_match:
            lines.append("\n**Warning:** Your response follows a similar approach to the defendant's conduct in this case.")

        # Summarize checklist
        met_items = [item for item in checklist if item.met or item.completeness_score > 0.5]
        unmet_items = [item for item in checklist if not item.met and item.completeness_score <= 0.5]

        if met_items:
            lines.append("\n**Addressed (court-endorsed):**")
            for item in met_items[:3]:  # Limit display
                lines.append(f"- {item.criterion[:100]}...")

        if unmet_items:
            lines.append("\n**Missing (per court standard):**")
            for item in unmet_items[:3]:  # Limit display
                lines.append(f"- {item.criterion[:100]}...")

        # Reasoning quality feedback
        if reasoning_quality:
            reasoning_items = []
            if reasoning_quality.considers_differential:
                reasoning_items.append("differential diagnosis")
            if reasoning_quality.integrates_evidence:
                reasoning_items.append("evidence integration")
            if reasoning_quality.acknowledges_uncertainty:
                reasoning_items.append("appropriate uncertainty")
            if reasoning_quality.considers_urgency:
                reasoning_items.append("urgency recognition")

            if reasoning_items:
                lines.append(f"\n**Clinical reasoning strengths:** {', '.join(reasoning_items)}")
            elif reasoning_quality.quality_score < 0.5:
                lines.append("\n**Reasoning note:** Consider explicitly discussing differential diagnosis and acknowledging clinical uncertainty.")

        return "\n".join(lines)

    def evaluate_primary_action(
        self, response: str, chat_history: list[dict] | None = None
    ) -> EvaluationResult:
        """
        Evaluate the clinician's response using primary-action scoring.

        This scoring method requires the PRIMARY criterion (expected_action_court)
        to be met. Missing the primary action = automatic score 0, regardless of
        other criteria met. This better matches how courts reason about liability.

        Scoring:
        - Score 0: Primary action NOT met (regardless of secondary criteria)
        - Score 1: Primary action MET, <50% of secondary criteria met
        - Score 2: Primary action MET, ≥50% of secondary criteria met

        Args:
            response: The clinician's substantive response
            chat_history: Optional chat history for context

        Returns:
            EvaluationResult with score, checklist, and court outcome
        """
        # Check if the decision point was reached
        decision_reached, deferral_reason = self._check_decision_point_reached(response)

        # Build checklist (with is_primary marked)
        checklist = self.build_checklist_from_rubric()

        # Score checklist items
        if self.use_llm and self.client:
            checklist = self._score_checklist_llm(response, checklist)
        else:
            checklist = self._score_checklist_rule(response, checklist)

        # Check if response matches defendant conduct
        defendant_match = self._check_matches_defendant(response, checklist)

        # Assess reasoning quality
        reasoning_quality = self.assess_reasoning_quality(response)

        # --- PRIMARY ACTION SCORING LOGIC ---
        # Separate primary and secondary criteria
        primary_items = [item for item in checklist if item.is_primary]
        secondary_items = [item for item in checklist if not item.is_primary]

        # Check if ALL primary actions are met (threshold: completeness >= 0.5 or met=True)
        primary_met = all(
            item.met or item.completeness_score >= 0.5
            for item in primary_items
        ) if primary_items else True  # If no primary items, consider met

        if not primary_met or defendant_match:
            # Primary action not addressed = automatic score 0
            score = 0
            risk_flag = True
        else:
            # Primary met, calculate secondary ratio
            if secondary_items:
                secondary_score = sum(
                    item.completeness_score if item.completeness_score > 0
                    else (1.0 if item.met else 0.0)
                    for item in secondary_items
                )
                secondary_ratio = secondary_score / len(secondary_items)
            else:
                secondary_ratio = 1.0  # No secondary = full credit

            # Factor in reasoning quality (10% of secondary score)
            adjusted_ratio = secondary_ratio * 0.9 + reasoning_quality.quality_score * 0.1

            if adjusted_ratio < 0.5:
                score = 1  # Primary met, but secondary incomplete
                risk_flag = False
            else:
                score = 2  # Primary met + majority of secondary
                risk_flag = defendant_match

        # Generate feedback for primary-action scoring
        feedback = self._generate_primary_feedback(
            score, checklist, primary_items, secondary_items, defendant_match, reasoning_quality
        )

        if not decision_reached:
            feedback = f"**Note:** {deferral_reason}\n\n{feedback}"

        court_outcome = safe_get(self.case, "simulation.end_state.legal_outcome", {})

        return EvaluationResult(
            score=score,
            risk_flag=risk_flag,
            checklist=checklist,
            feedback=feedback,
            court_outcome=court_outcome,
            defendant_action=self._get_defendant_action(),
            expected_action=self._get_expected_action(),
            what_should_have_happened=self._get_what_should_have_happened(),
            legal_standard=self._get_legal_standard(),
            reasoning_quality=reasoning_quality,
            cognitive_error_avoided=self._get_cognitive_error(),
            score_valid=decision_reached,
            deferral_reason=deferral_reason,
        )

    def _generate_primary_feedback(
        self,
        score: int,
        checklist: list[ChecklistItem],
        primary_items: list[ChecklistItem],
        secondary_items: list[ChecklistItem],
        defendant_match: bool,
        reasoning_quality: ReasoningQuality | None = None
    ) -> str:
        """
        Generate feedback for primary-action scoring.

        Args:
            score: The score (0, 1, or 2)
            checklist: The full evaluated checklist
            primary_items: Primary criterion items
            secondary_items: Secondary criterion items
            defendant_match: Whether response matched defendant conduct
            reasoning_quality: Assessment of clinical reasoning quality

        Returns:
            Feedback string
        """
        lines = []

        # Check if primary criteria were met
        primary_met = all(
            item.met or item.completeness_score >= 0.5
            for item in primary_items
        ) if primary_items else True

        if score == 2:
            lines.append("**Court-aligned response.** You addressed the primary action required by the court AND supporting criteria.")
        elif score == 1:
            lines.append("**Partially aligned.** You addressed the primary court-required action, but some supporting criteria were missed.")
        else:
            if not primary_met:
                lines.append("**Primary criterion not met.** The court identified a specific action that was required, which was not addressed.")
            else:
                lines.append("**Would not meet standard of care.** Important clinical considerations were not addressed.")

        if defendant_match:
            lines.append("\n**Warning:** Your response follows a similar approach to the defendant's conduct in this case.")

        # Show primary criterion status
        if primary_items:
            lines.append("\n**Primary Action (court-required):**")
            for item in primary_items:
                status = "✓ Addressed" if (item.met or item.completeness_score >= 0.5) else "✗ NOT addressed"
                lines.append(f"- {status}: {item.criterion[:100]}...")

        # Summarize secondary checklist
        met_secondary = [item for item in secondary_items if item.met or item.completeness_score > 0.5]
        unmet_secondary = [item for item in secondary_items if not item.met and item.completeness_score <= 0.5]

        if met_secondary:
            lines.append(f"\n**Supporting criteria met ({len(met_secondary)}/{len(secondary_items)}):**")
            for item in met_secondary[:3]:
                lines.append(f"- {item.criterion[:100]}...")

        if unmet_secondary:
            lines.append(f"\n**Supporting criteria missed ({len(unmet_secondary)}/{len(secondary_items)}):**")
            for item in unmet_secondary[:3]:
                lines.append(f"- {item.criterion[:100]}...")

        # Reasoning quality feedback
        if reasoning_quality:
            reasoning_items = []
            if reasoning_quality.considers_differential:
                reasoning_items.append("differential diagnosis")
            if reasoning_quality.integrates_evidence:
                reasoning_items.append("evidence integration")
            if reasoning_quality.acknowledges_uncertainty:
                reasoning_items.append("appropriate uncertainty")
            if reasoning_quality.considers_urgency:
                reasoning_items.append("urgency recognition")

            if reasoning_items:
                lines.append(f"\n**Clinical reasoning strengths:** {', '.join(reasoning_items)}")

        return "\n".join(lines)

    def get_court_summary(self) -> dict[str, Any]:
        """
        Get a summary of the court outcome for display.

        Returns:
            Dict with verdict, damages, key_findings, etc.
        """
        end_state = safe_get(self.case, "simulation.end_state", {})
        legal = end_state.get("legal_outcome", {})
        malpractice = end_state.get("malpractice_determination", {})

        return {
            "verdict": legal.get("verdict", "Unknown"),
            "damages_awarded": legal.get("damages_awarded", "Not documented"),
            "key_findings": legal.get("key_findings", []),
            "breach_found": malpractice.get("breach_found", None),
            "causation_established": malpractice.get("causation_established", None),
            "point_of_failure": malpractice.get("point_of_failure", ""),
            "counterfactual": malpractice.get("counterfactual", ""),
        }
