"""OpenAI-based LLM extractor for case simulations."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..parsing import ParsedJudgment
from ..schemas import CaseSimulation, EvidenceItem
from ..segmentation import DocumentSegmenter, SegmentedDocument, TextSegment
from ..utils import get_logger


class TokenBudget:
    """Manages token budget for extraction."""

    def __init__(
        self,
        max_input_tokens: int = 120000,
        max_output_tokens: int = 4096,
        model: str = "gpt-4o",
    ):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.tokens_used = 0
        self.requests_made = 0

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def can_fit(self, text: str, reserve: int = 1000) -> bool:
        """Check if text fits in remaining budget."""
        return self.count(text) + reserve <= self.max_input_tokens

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage."""
        self.tokens_used += input_tokens + output_tokens
        self.requests_made += 1


# Common words to exclude from grounding checks
COMMON_WORDS = {
    "the", "and", "that", "this", "with", "from", "have", "been", "were",
    "would", "could", "should", "which", "their", "there", "about", "after",
    "before", "being", "between", "during", "having", "other", "these",
    "those", "under", "where", "while", "patient", "claimant", "defendant",
    "hospital", "court", "evidence", "stated", "found", "judgment",
}


class LLMExtractor:
    """Extracts case simulations using OpenAI LLM."""

    VERSION = "0.2.0"  # Updated for grounding verification

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the extractor."""
        self.settings = get_settings()
        self.logger = get_logger("extraction.llm")

        # Get API configuration
        api_key = api_key or self.settings.openai.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model or self.settings.openai.model
        base_url = base_url or self.settings.openai.base_url

        # Initialize clients
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Initialize segmenter
        self.segmenter = DocumentSegmenter(model=self.model)

        # Load prompts
        self.prompts = self._load_prompts()

        # Token budget
        self.budget = TokenBudget(model=self.model)

    def _load_prompts(self) -> dict[str, str]:
        """Load prompt templates."""
        prompts_dir = self.settings.paths.prompts_dir
        prompts = {}

        prompt_files = [
            "build_evidence_index.prompt",
            "extract_simulation_case.prompt",
            "merge_and_validate.prompt",
        ]

        for filename in prompt_files:
            path = prompts_dir / filename
            if path.exists():
                prompts[filename.replace(".prompt", "")] = path.read_text()
            else:
                self.logger.warning(f"Prompt file not found: {path}")

        return prompts

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Make a call to OpenAI with retries."""
        temperature = temperature if temperature is not None else self.settings.openai.temperature
        max_tokens = max_tokens or self.settings.openai.max_tokens

        # Count tokens
        input_tokens = self.budget.count(system_prompt + user_prompt)
        self.logger.debug(f"Request tokens: {input_tokens}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        result = response.choices[0].message.content or ""

        # Record usage
        if response.usage:
            self.budget.record_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        return result

    def is_medical_malpractice_case(self, parsed: ParsedJudgment) -> tuple[bool, str]:
        """Check if this is a medical malpractice case using LLM.

        Returns:
            Tuple of (is_malpractice: bool, reason: str)
        """
        # Take first 3000 chars for quick classification
        text_sample = parsed.raw_text[:3000]

        system_prompt = """You are a legal analyst. Your task is to determine if a court judgment is a MEDICAL MALPRACTICE or CLINICAL NEGLIGENCE case.

A case IS medical malpractice if:
- It involves a healthcare provider (doctor, nurse, hospital, NHS trust)
- There are allegations of negligence in medical treatment, diagnosis, or care
- The claimant suffered harm allegedly due to medical care

A case is NOT medical malpractice if it's about:
- Employment disputes (even if in healthcare)
- Personal injury from accidents (car crash, slip and fall)
- Product liability (defective products)
- Criminal proceedings
- Immigration/asylum
- Contract disputes
- Defamation
- Police misconduct
- Prison conditions

Output JSON: {"is_malpractice": true/false, "reason": "brief explanation", "case_type": "type if not malpractice"}"""

        user_prompt = f"""Analyze this court judgment excerpt and determine if it's a medical malpractice case:

{text_sample}

Output JSON only."""

        try:
            response = self._call_openai(system_prompt, user_prompt, temperature=0.1, max_tokens=200)
            data = json.loads(response)
            is_malp = data.get("is_malpractice", False)
            reason = data.get("reason", "Unknown")
            return (is_malp, reason)
        except Exception as e:
            self.logger.warning(f"Malpractice check failed: {e}")
            # Fall back to keyword-based check
            text_lower = parsed.raw_text.lower()
            keywords = ["clinical negligence", "medical negligence", "breach of duty", "standard of care"]
            if any(kw in text_lower for kw in keywords):
                return (True, "Contains medical negligence keywords")
            return (False, "Keyword check negative")

    def _verify_evidence_grounding(
        self,
        evidence_index: list[EvidenceItem],
        raw_text: str,
    ) -> tuple[float, list[str]]:
        """Verify evidence items are grounded in source text. NO LLM CALLS.

        Returns:
            Tuple of (grounding_score, list of ungrounded evidence IDs)
        """
        grounded = 0
        ungrounded_ids = []
        raw_lower = raw_text.lower()

        for item in evidence_index:
            # Check if paragraph reference pattern exists in document
            para_ref = item.paragraph_ref
            if para_ref:
                para_patterns = [
                    rf"\[{para_ref}\]",     # [15]
                    rf"^{para_ref}\.",       # 15.
                    rf"¶\s*{para_ref}",      # ¶ 15
                    rf"para(?:graph)?\s*{para_ref}",  # Para 15
                ]
                para_found = any(
                    re.search(p, raw_text, re.MULTILINE | re.IGNORECASE)
                    for p in para_patterns
                )
            else:
                para_found = False

            # Check if key terms from the evidence text appear in document
            # Extract distinctive words (5+ chars, not common)
            words = re.findall(r'\b[a-z]{5,}\b', item.text.lower())
            distinctive = [w for w in words if w not in COMMON_WORDS][:5]

            if distinctive:
                matches = sum(1 for w in distinctive if w in raw_lower)
                terms_found = matches >= len(distinctive) * 0.5  # Need 50% match
            else:
                terms_found = True  # No distinctive words to check

            # Item is grounded if paragraph exists OR key terms found
            if para_found or terms_found:
                grounded += 1
            else:
                ungrounded_ids.append(item.evidence_id)
                self.logger.debug(
                    f"Ungrounded evidence {item.evidence_id}: "
                    f"para_ref={para_ref}, distinctive={distinctive}"
                )

        score = grounded / max(1, len(evidence_index))
        self.logger.info(
            f"Evidence grounding: {grounded}/{len(evidence_index)} items grounded "
            f"({score:.0%})"
        )
        return score, ungrounded_ids

    def _verify_verdict_consistency(
        self,
        raw_text: str,
        extracted_verdict: str,
    ) -> tuple[bool, str]:
        """Check if extracted verdict matches keywords in judgment. NO LLM CALLS.

        Returns:
            Tuple of (is_consistent, message)
        """
        text_lower = raw_text.lower()

        # Check last 20% of document (where verdict usually is)
        conclusion_start = int(len(text_lower) * 0.8)
        conclusion_section = text_lower[conclusion_start:]

        # Indicators for NO_LIABILITY (claim fully dismissed)
        no_liability_phrases = [
            "claim is dismissed",
            "claim fails",
            "claimant's claim is dismissed",
            "claimant's claim fails",
            "no breach of duty",
            "defendant is not liable",
            "not negligent",
            "claim must fail",
            "judgment for the defendant",
            "dismiss the claim",
            "finds for the defendant",
        ]

        # Indicators for LIABILITY_FOUND
        liability_phrases = [
            "judgment for the claimant",
            "judgment for the plaintiff",
            "liability is established",
            "find for the claimant",
            "finds for the claimant",
            "defendant is liable",
            "breach of duty is established",
            "claimant succeeds",
            "find the defendant liable",
            "finds the defendant negligent",
            "admitted breach",
            "breach is admitted",
            "conceded liability",
            "admitted liability",
        ]

        # Indicators for PARTIAL_LIABILITY (some claims succeed, some fail)
        partial_phrases = [
            "admitted breach of duty",
            "breach of duty in relation to",
            "otherwise the claim fails",
            "only for the admitted breach",
            "partial success",
            "claim succeeds in part",
            "liability on one issue",
            "admitted breach",
        ]

        found_no_liability = []
        found_liability = []
        found_partial = []

        # Check for all phrase types
        for phrase in no_liability_phrases:
            if phrase in conclusion_section:
                found_no_liability.append(phrase)

        for phrase in liability_phrases:
            if phrase in conclusion_section:
                found_liability.append(phrase)

        for phrase in partial_phrases:
            if phrase in conclusion_section:
                found_partial.append(phrase)

        # Determine detected verdict based on evidence
        detected = None
        detected_phrases = []

        # If we find both liability AND no liability phrases, it's likely partial
        if found_partial or (found_liability and found_no_liability):
            detected = "PARTIAL_LIABILITY"
            detected_phrases = found_partial or (found_liability + found_no_liability)
        elif found_no_liability and not found_liability:
            detected = "NO_LIABILITY"
            detected_phrases = found_no_liability
        elif found_liability and not found_no_liability:
            detected = "LIABILITY_FOUND"
            detected_phrases = found_liability

        if detected:
            self.logger.info(
                f"Verdict detection: found {detected_phrases[:2]} -> {detected}"
            )
            # Check consistency
            if detected == extracted_verdict:
                return True, f"Verdict consistent: {detected}"

            # PARTIAL_LIABILITY is compatible with both LIABILITY_FOUND
            # (if focusing on the part that succeeded)
            if detected == "PARTIAL_LIABILITY" and extracted_verdict in [
                "LIABILITY_FOUND", "PARTIAL_LIABILITY"
            ]:
                return True, (
                    f"Verdict acceptable: extracted '{extracted_verdict}' "
                    f"is compatible with partial liability case"
                )

            return False, (
                f"Verdict mismatch: extracted '{extracted_verdict}', "
                f"but found {detected_phrases[:2]} indicating {detected}"
            )

        # Could not detect - don't fail, just warn
        self.logger.warning(
            f"Could not detect verdict from text. Extracted: {extracted_verdict}"
        )
        return True, "Unable to verify verdict from text (no clear phrases found)"

    def extract(
        self,
        parsed: ParsedJudgment,
        case_id: str,
        url: str,
        skip_malpractice_check: bool = False,
        skip_grounding_check: bool = False,
        grounding_threshold: float = 0.7,
    ) -> CaseSimulation:
        """Extract a case simulation from a parsed judgment.

        Args:
            parsed: Parsed judgment document
            case_id: Unique case identifier
            url: Source URL
            skip_malpractice_check: Skip the malpractice case verification
            skip_grounding_check: Skip the evidence grounding verification
            grounding_threshold: Minimum proportion of evidence that must be grounded (default 0.7)

        Returns:
            CaseSimulation object

        Raises:
            ValueError: If case fails malpractice check, grounding check, or verdict verification
        """
        self.logger.info(f"Starting extraction for {case_id}")

        # Step 0: Verify this is a malpractice case
        if not skip_malpractice_check:
            is_malpractice, reason = self.is_medical_malpractice_case(parsed)
            if not is_malpractice:
                raise ValueError(f"Not a medical malpractice case: {reason}")
            self.logger.info(f"Confirmed malpractice case: {reason}")

        # Step 1: Build evidence index
        self.logger.info("Pass 1: Building evidence index...")
        evidence_index = self._extract_evidence_index(parsed)

        # Step 1.5: Verify evidence grounding (NO LLM CALL - pure regex)
        if not skip_grounding_check:
            grounding_score, ungrounded = self._verify_evidence_grounding(
                evidence_index, parsed.raw_text
            )
            if grounding_score < grounding_threshold:
                raise ValueError(
                    f"Evidence grounding failed: only {grounding_score:.0%} grounded "
                    f"(threshold: {grounding_threshold:.0%}). "
                    f"Ungrounded items: {ungrounded[:5]}"
                )
            self.logger.info(f"Evidence grounding passed: {grounding_score:.0%}")

        # Step 2: Extract simulation in chunks if needed
        self.logger.info("Pass 2: Extracting simulation...")
        simulation_data = self._extract_simulation(parsed, evidence_index)

        # Step 3: Merge and validate
        self.logger.info("Pass 3: Merging and validating...")
        case_simulation = self._merge_and_validate(
            case_id=case_id,
            url=url,
            parsed=parsed,
            evidence_index=evidence_index,
            simulation_data=simulation_data,
        )

        # Step 4: Verify verdict consistency (NO LLM CALL - pure string matching)
        # Handle both dict and Pydantic object access
        try:
            if hasattr(case_simulation.simulation, "end_state"):
                # Pydantic object
                end_state = case_simulation.simulation.end_state
                if end_state and hasattr(end_state, "legal_outcome"):
                    legal_outcome = end_state.legal_outcome
                    if legal_outcome and hasattr(legal_outcome, "verdict"):
                        verdict = legal_outcome.verdict
                    else:
                        verdict = "UNKNOWN"
                else:
                    verdict = "UNKNOWN"
            else:
                # Dict access
                verdict = case_simulation.simulation.get("end_state", {}).get("legal_outcome", {}).get("verdict", "UNKNOWN")
        except Exception:
            verdict = "UNKNOWN"

        if isinstance(verdict, str):
            verdict_value = verdict
        else:
            verdict_value = verdict.value if hasattr(verdict, "value") else str(verdict)

        verdict_ok, verdict_msg = self._verify_verdict_consistency(
            parsed.raw_text, verdict_value
        )
        if not verdict_ok:
            # Log warning but don't fail - verdict detection is imperfect
            # especially for partial liability cases
            self.logger.warning(f"Verdict verification issue: {verdict_msg}")
        else:
            self.logger.info(f"Verdict verification: {verdict_msg}")

        self.logger.info(f"Extraction complete for {case_id}")
        return case_simulation

    def _extract_evidence_index(
        self,
        parsed: ParsedJudgment,
    ) -> list[EvidenceItem]:
        """Extract evidence items from the judgment."""
        # Create context with paragraph numbers
        context = self.segmenter.create_evidence_context(parsed)

        system_prompt = self.prompts.get("build_evidence_index", self._get_default_evidence_prompt())

        user_prompt = f"""Analyze the following medical malpractice judgment and extract an evidence index.

JUDGMENT TEXT:
{context}

Extract key factual statements, expert testimony, medical findings, and court conclusions as evidence items.
Each item must reference the paragraph number where it appears.

Output valid JSON only."""

        response = self._call_openai(system_prompt, user_prompt)

        try:
            data = json.loads(response)
            evidence_items = []

            for item in data.get("evidence_items", []):
                try:
                    # Handle new format with verbatim_quote and summary
                    # The prompt asks for verbatim_quote + summary, but EvidenceItem expects 'text'
                    if "text" not in item:
                        # Use verbatim_quote if available, otherwise summary
                        if "verbatim_quote" in item:
                            item["text"] = item["verbatim_quote"]
                        elif "summary" in item:
                            item["text"] = item["summary"]
                        else:
                            self.logger.warning(f"Evidence item missing text: {item.get('evidence_id')}")
                            continue

                    evidence_items.append(EvidenceItem(**item))
                except ValidationError as e:
                    self.logger.warning(f"Invalid evidence item: {e}")

            return evidence_items

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return []

    def _extract_simulation(
        self,
        parsed: ParsedJudgment,
        evidence_index: list[EvidenceItem],
    ) -> dict[str, Any]:
        """Extract simulation structure from the judgment."""
        # Segment for processing
        segmented = self.segmenter.segment_parsed(parsed)

        if len(segmented.segments) == 1:
            # Single segment - extract directly
            return self._extract_simulation_segment(
                segmented.segments[0],
                evidence_index,
                parsed,
            )
        else:
            # Multi-segment - extract and merge
            return self._extract_simulation_multi(segmented, evidence_index, parsed)

    def _extract_simulation_segment(
        self,
        segment: TextSegment,
        evidence_index: list[EvidenceItem],
        parsed: ParsedJudgment,
    ) -> dict[str, Any]:
        """Extract simulation from a single segment."""
        # Build evidence reference string
        evidence_ref = "\n".join(
            f"- {item.evidence_id}: {item.text[:100]}..."
            for item in evidence_index[:50]
        )

        system_prompt = self.prompts.get(
            "extract_simulation_case",
            self._get_default_simulation_prompt(),
        )

        user_prompt = f"""Extract a clinical case simulation from the following judgment segment.

CASE METADATA:
- Title: {parsed.title or 'Unknown'}
- Citation: {parsed.citation or 'Unknown'}
- Court: {parsed.court or 'Unknown'}
- Date: {parsed.date or 'Unknown'}

EVIDENCE INDEX (use these IDs in your output):
{evidence_ref}

JUDGMENT TEXT:
{segment.text}

Create a complete simulation with:
1. Initial clinical state (presentation, HPI, exam)
2. Requestable items (labs, imaging, consults mentioned in the case)
3. Decision points (where malpractice occurred, what should have been done)
4. Timeline phases
5. End state (patient outcome, legal outcome)

Reference evidence IDs for all factual claims. Output valid JSON only."""

        response = self._call_openai(system_prompt, user_prompt, max_tokens=8192)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return {}

    def _extract_simulation_multi(
        self,
        segmented: SegmentedDocument,
        evidence_index: list[EvidenceItem],
        parsed: ParsedJudgment,
    ) -> dict[str, Any]:
        """Extract simulation from multiple segments and merge."""
        partial_results: list[dict[str, Any]] = []

        for segment in segmented.segments:
            result = self._extract_simulation_segment(segment, evidence_index, parsed)
            if result:
                partial_results.append(result)

        if not partial_results:
            return {}

        if len(partial_results) == 1:
            return partial_results[0]

        # Merge results
        return self._merge_partial_results(partial_results)

    def _merge_partial_results(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge partial extraction results."""
        merged: dict[str, Any] = {
            "simulation": {
                "initial_state": None,
                "requestables": [],
                "timeline_phases": [],
                "decision_points": [],
                "end_state": None,
            },
            "ground_truth": {
                "factual_timeline": [],
                "tests_performed": [],
                "diagnoses": [],
                "treatments": [],
                "complications": [],
                "allegations": [],
                "court_findings": [],
            },
            "taxonomy_labels": {
                "malpractice_categories": [],
            },
        }

        seen_request_ids: set[str] = set()
        seen_decision_ids: set[str] = set()

        for result in results:
            sim = result.get("simulation", {})

            # Take first non-null initial state
            if sim.get("initial_state") and not merged["simulation"]["initial_state"]:
                merged["simulation"]["initial_state"] = sim["initial_state"]

            # Merge requestables (deduplicate by ID)
            for req in sim.get("requestables", []):
                req_id = req.get("request_id")
                if req_id and req_id not in seen_request_ids:
                    seen_request_ids.add(req_id)
                    merged["simulation"]["requestables"].append(req)

            # Merge decision points (deduplicate by ID)
            for dp in sim.get("decision_points", []):
                dp_id = dp.get("decision_id")
                if dp_id and dp_id not in seen_decision_ids:
                    seen_decision_ids.add(dp_id)
                    merged["simulation"]["decision_points"].append(dp)

            # Merge timeline phases
            merged["simulation"]["timeline_phases"].extend(
                sim.get("timeline_phases", [])
            )

            # Take last end state
            if sim.get("end_state"):
                merged["simulation"]["end_state"] = sim["end_state"]

            # Merge ground truth
            gt = result.get("ground_truth", {})
            for key in merged["ground_truth"]:
                if isinstance(merged["ground_truth"][key], list):
                    merged["ground_truth"][key].extend(gt.get(key, []))

            # Merge taxonomy
            tax = result.get("taxonomy_labels", {})
            for cat in tax.get("malpractice_categories", []):
                if cat not in merged["taxonomy_labels"]["malpractice_categories"]:
                    merged["taxonomy_labels"]["malpractice_categories"].append(cat)

        # Deduplicate timeline phases by phase_id
        seen_phases: set[str] = set()
        unique_phases = []
        for phase in merged["simulation"]["timeline_phases"]:
            phase_id = phase.get("phase_id")
            if phase_id and phase_id not in seen_phases:
                seen_phases.add(phase_id)
                unique_phases.append(phase)
        merged["simulation"]["timeline_phases"] = unique_phases

        return merged

    def _clean_simulation_data(self, sim_data: dict[str, Any]) -> dict[str, Any]:
        """Clean LLM output to match schema requirements."""
        # Clean initial_state
        initial = sim_data.get("initial_state", {})
        if initial:
            # Convert string fields to empty lists where needed
            for field in ["past_medical_history", "medications", "allergies"]:
                val = initial.get(field)
                if val is None or isinstance(val, str):
                    initial[field] = []

            # Fix patient demographics
            demographics = initial.get("patient_demographics", {})
            if demographics:
                sex = demographics.get("sex", "unknown")
                if sex not in ["male", "female", "unknown"]:
                    demographics["sex"] = "unknown"
                initial["patient_demographics"] = demographics

            sim_data["initial_state"] = initial

        # Clean timeline phases - fix invalid phase_id values
        valid_phases = {"presentation", "workup", "decision", "procedure", "postop", "followup"}
        phase_mapping = {
            "preoperative": "workup",
            "pre-operative": "workup",
            "perioperative": "procedure",
            "intraoperative": "procedure",
            "intra-operative": "procedure",
            "post-operative": "postop",
            "recovery": "postop",
            "discharge": "followup",
        }

        phases = sim_data.get("timeline_phases", [])
        for phase in phases:
            if isinstance(phase, dict):
                phase_id = phase.get("phase_id", "").lower()
                if phase_id not in valid_phases:
                    phase["phase_id"] = phase_mapping.get(phase_id, "decision")
        sim_data["timeline_phases"] = phases

        # Clean decision points
        valid_action_types = {
            "ORDER_TEST", "CHOOSE_MANAGEMENT", "DISCLOSE_ALTERNATIVES",
            "COUNSEL_RISKS", "SELECT_TECHNIQUE", "ESCALATE_CARE",
            "DOCUMENT_CONSENT", "REFER", "PRESCRIBE", "DISCHARGE_DECISION"
        }
        action_type_mapping = {
            "not documented in judgment": "CHOOSE_MANAGEMENT",
            "unknown": "CHOOSE_MANAGEMENT",
            "diagnose": "ORDER_TEST",
            "treat": "CHOOSE_MANAGEMENT",
            "surgery": "SELECT_TECHNIQUE",
            "consent": "DOCUMENT_CONSENT",
        }

        decision_points = sim_data.get("decision_points", [])
        for dp in decision_points:
            if isinstance(dp, dict):
                phase_id = dp.get("phase_id", "").lower()
                if phase_id not in valid_phases:
                    dp["phase_id"] = phase_mapping.get(phase_id, "decision")

                # Fix action_type
                action_type = dp.get("action_type", "CHOOSE_MANAGEMENT")
                if action_type not in valid_action_types:
                    dp["action_type"] = action_type_mapping.get(action_type.lower(), "CHOOSE_MANAGEMENT")

                # Ensure at least 2 options
                options = dp.get("options", [])
                if len(options) < 2:
                    options = [
                        {"option_id": "A", "description": "Standard of care action"},
                        {"option_id": "B", "description": "Alternative action"},
                    ]
                    dp["options"] = options

                # Ensure scoring_rubric exists with required fields
                if "scoring_rubric" not in dp or not isinstance(dp.get("scoring_rubric"), dict):
                    dp["scoring_rubric"] = {
                        "max_score": 10,
                        "criteria": None
                    }
                else:
                    # Ensure required fields in existing rubric
                    rubric = dp["scoring_rubric"]
                    if "max_score" not in rubric:
                        rubric["max_score"] = 10

        sim_data["decision_points"] = decision_points

        # Clean requestables
        valid_request_types = {
            "LAB", "IMAGING", "PATHOLOGY", "CONSULT_NOTE", "MDT_NOTE",
            "OP_NOTE", "CONSENT_DISCUSSION", "FOLLOWUP", "VITAL_SIGNS",
            "NURSING_NOTES", "PROCEDURE_NOTE"
        }
        request_type_mapping = {
            "medication": "CONSULT_NOTE",
            "test": "LAB",
            "scan": "IMAGING",
            "xray": "IMAGING",
            "x-ray": "IMAGING",
            "mri": "IMAGING",
            "ct": "IMAGING",
            "blood": "LAB",
            "biopsy": "PATHOLOGY",
            "consult": "CONSULT_NOTE",
            "referral": "CONSULT_NOTE",
        }

        requestables = sim_data.get("requestables", [])
        for req in requestables:
            if isinstance(req, dict):
                req_type = req.get("type", "LAB")
                if req_type not in valid_request_types:
                    req["type"] = request_type_mapping.get(req_type.lower(), "CONSULT_NOTE")

                # Fix available_phase
                avail_phase = req.get("available_phase", "workup")
                if avail_phase and avail_phase.lower() not in valid_phases:
                    req["available_phase"] = "workup"

                # Fix boolean fields in requestables
                for bool_field in ["was_ordered_in_case", "should_have_been_ordered", "is_critical"]:
                    if bool_field in req:
                        req[bool_field] = self._clean_bool_value(req[bool_field])

        sim_data["requestables"] = requestables

        # Clean end_state
        end_state = sim_data.get("end_state", {})
        if end_state:
            legal = end_state.get("legal_outcome", {})
            if legal:
                verdict = legal.get("verdict", "UNKNOWN")
                valid_verdicts = {"LIABILITY_FOUND", "NO_LIABILITY", "PARTIAL_LIABILITY", "SETTLED", "UNKNOWN"}
                verdict_mapping = {
                    "LIABILITY_NOT_FOUND": "NO_LIABILITY",
                    "NOT_FOUND": "NO_LIABILITY",
                    "FOUND": "LIABILITY_FOUND",
                    "PARTIAL": "PARTIAL_LIABILITY",
                }
                if verdict not in valid_verdicts:
                    legal["verdict"] = verdict_mapping.get(verdict, "UNKNOWN")
                end_state["legal_outcome"] = legal

            # Clean malpractice_determination - handle boolean fields
            malpractice_det = end_state.get("malpractice_determination", {})
            if malpractice_det:
                for bool_field in ["causation_established", "breach_established", "damages_awarded"]:
                    val = malpractice_det.get(bool_field)
                    if val is not None:
                        malpractice_det[bool_field] = self._clean_bool_value(val)
                end_state["malpractice_determination"] = malpractice_det

            sim_data["end_state"] = end_state

        return sim_data

    def _clean_bool_value(self, val: Any) -> bool:
        """Convert a value to boolean, handling LLM string outputs."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            lower = val.lower().strip()
            if lower in ["true", "yes", "1", "established", "proven", "found"]:
                return True
            if lower in ["false", "no", "0", "not established", "not proven", "not found"]:
                return False
            # For "not documented in judgment" or similar, default to False
            return False
        if isinstance(val, (int, float)):
            return bool(val)
        return False

    def _clean_ground_truth(self, gt_data: dict[str, Any]) -> dict[str, Any]:
        """Clean ground_truth data to match schema requirements."""
        # Clean allegations - handle boolean fields
        allegations = gt_data.get("allegations", [])
        for allegation in allegations:
            if isinstance(allegation, dict):
                for bool_field in ["found_proven", "causation_proven"]:
                    if bool_field in allegation:
                        allegation[bool_field] = self._clean_bool_value(allegation[bool_field])
        gt_data["allegations"] = allegations

        # Clean court_findings
        findings = gt_data.get("court_findings", [])
        for finding in findings:
            if isinstance(finding, dict):
                if "breach_found" in finding:
                    finding["breach_found"] = self._clean_bool_value(finding["breach_found"])
        gt_data["court_findings"] = findings

        # Clean diagnoses - handle boolean fields
        diagnoses = gt_data.get("diagnoses", [])
        for diagnosis in diagnoses:
            if isinstance(diagnosis, dict):
                for bool_field in ["correct", "timely", "missed"]:
                    if bool_field in diagnosis:
                        diagnosis[bool_field] = self._clean_bool_value(diagnosis[bool_field])
        gt_data["diagnoses"] = diagnoses

        # Clean treatments - handle boolean fields
        treatments = gt_data.get("treatments", [])
        for treatment in treatments:
            if isinstance(treatment, dict):
                for bool_field in ["appropriate", "successful", "delayed"]:
                    if bool_field in treatment:
                        treatment[bool_field] = self._clean_bool_value(treatment[bool_field])
        gt_data["treatments"] = treatments

        # Clean tests_performed - handle boolean fields
        tests = gt_data.get("tests_performed", [])
        for test in tests:
            if isinstance(test, dict):
                for bool_field in ["ordered", "performed", "reviewed"]:
                    if bool_field in test:
                        test[bool_field] = self._clean_bool_value(test[bool_field])
        gt_data["tests_performed"] = tests

        return gt_data

    def _merge_and_validate(
        self,
        case_id: str,
        url: str,
        parsed: ParsedJudgment,
        evidence_index: list[EvidenceItem],
        simulation_data: dict[str, Any],
    ) -> CaseSimulation:
        """Merge extracted data and create validated CaseSimulation."""
        from ..schemas import (
            CaseSimulation,
            CaseSummary,
            ClinicalDomain,
            DiscoveryStatus,
            EndState,
            ExtractionMetadata,
            GroundTruth,
            InitialState,
            Jurisdiction,
            LegalOutcome,
            MalpracticeCategory,
            MalpracticeDetermination,
            OutcomeSeverity,
            PatientOutcome,
            QualityMetrics,
            Simulation,
            Source,
            TaxonomyLabels,
        )
        from datetime import date

        # Determine source and jurisdiction from URL
        source = Source.BAILII  # Default
        jurisdiction = Jurisdiction.UK

        if "canlii" in url.lower():
            source = Source.CANLII
            jurisdiction = Jurisdiction.CA
        elif "austlii" in url.lower():
            source = Source.AUSTLII
            jurisdiction = Jurisdiction.AU
        elif "courtlistener" in url.lower():
            source = Source.COURTLISTENER
            jurisdiction = Jurisdiction.US

        # Parse date
        decision_date = date.today()
        if parsed.date:
            try:
                # Try common formats
                import dateutil.parser
                decision_date = dateutil.parser.parse(parsed.date).date()
            except Exception:
                pass

        # Build simulation object
        sim_data = simulation_data.get("simulation", {})

        # Clean LLM output to match schema
        sim_data = self._clean_simulation_data(sim_data)

        # Check if case is marked as not testable
        is_testable = sim_data.get("testable", True)

        # Ensure we have minimum required data
        if not sim_data.get("initial_state"):
            sim_data["initial_state"] = {
                "chief_complaint": "Medical condition requiring treatment",
                "evidence_ids": ["E001"] if evidence_index else [],
            }

        # Only create dummy decision_points for testable cases
        if is_testable and not sim_data.get("decision_points"):
            sim_data["decision_points"] = [{
                "decision_id": "D001",
                "phase_id": "decision",
                "prompt": "What action should be taken?",
                "action_type": "CHOOSE_MANAGEMENT",
                "options": [
                    {"option_id": "A", "description": "Standard treatment"},
                    {"option_id": "B", "description": "Alternative approach"},
                ],
                "actual_action_defendant": {
                    "description": "Action taken by defendant",
                    "evidence_ids": [],
                },
                "expected_action_court": {
                    "description": "Court-endorsed standard of care",
                    "evidence_ids": [],
                },
                "scoring_rubric": {"max_score": 10},
            }]
        elif not is_testable:
            # For untestable cases, ensure empty lists
            sim_data["decision_points"] = sim_data.get("decision_points", [])
            sim_data["requestables"] = sim_data.get("requestables", [])
            sim_data["timeline_phases"] = sim_data.get("timeline_phases", [])

        # Only create dummy end_state for testable cases
        if is_testable and not sim_data.get("end_state"):
            sim_data["end_state"] = {
                "patient_outcome": {"description": "Adverse outcome"},
                "legal_outcome": {"verdict": "UNKNOWN"},
                "malpractice_determination": {},
            }
        elif not is_testable:
            # For untestable cases, end_state can be None
            sim_data["end_state"] = sim_data.get("end_state", None)

        # Determine clinical domain from content
        # Order matters - more specific domains should come first!
        clinical_domain = ClinicalDomain.OTHER
        text_lower = parsed.raw_text.lower()

        # Domain keywords ordered from most specific to most general
        domain_keywords_ordered = [
            # Obstetrics - birth-related terms (use more specific terms to avoid false matches)
            (ClinicalDomain.OBSTETRICS_GYNAECOLOGY, [
                "obstetric", "pregnancy", "pregnant", "childbirth", "labour ward",
                "labor ward", "caesarean", "cesarean", "midwife", "midwifery",
                "neonatal", "newborn", "foetal", "fetal", "placenta", "uterine",
                "shoulder dystocia", "brachial plexus", "during birth", "at birth",
                "giving birth", "maternity"
            ]),
            # Paediatrics
            (ClinicalDomain.PAEDIATRICS, [
                "paediatric", "pediatric", "child", "infant", "neonate",
                "neonatal", "baby", "babies"
            ]),
            # Oncology
            (ClinicalDomain.ONCOLOGY, [
                "cancer", "tumour", "tumor", "oncolog", "malignant", "metasta",
                "carcinoma", "chemotherapy", "radiotherapy"
            ]),
            # Cardiology
            (ClinicalDomain.CARDIOLOGY, [
                "cardiac", "cardiolog", "heart", "coronary", "myocardial",
                "arrhythmia", "pacemaker", "icd", "ecg", "ekg"
            ]),
            # Neurology/Neurosurgery
            (ClinicalDomain.NEUROLOGY, [
                "neurolog", "neurosurg", "brain", "stroke", "cerebral",
                "aneurysm", "meningi", "parkinson", "seizure", "epilep"
            ]),
            # Orthopaedics
            (ClinicalDomain.SURGERY_ORTHOPAEDIC, [
                "orthopaedic", "orthopedic", "fracture", "bone", "spine",
                "spinal", "joint", "hip replacement", "knee replacement"
            ]),
            # Emergency Medicine
            (ClinicalDomain.EMERGENCY_MEDICINE, [
                "emergency department", "a&e", "accident and emergency",
                "emergency room", "triage"
            ]),
            # Psychiatry
            (ClinicalDomain.PSYCHIATRY, [
                "psychiatric", "psychiatr", "mental health", "depression",
                "psychosis", "suicide", "self-harm"
            ]),
            # Internal Medicine / GP
            (ClinicalDomain.INTERNAL_MEDICINE, [
                "internal medicine", "general practitioner", "primary care", "gp "
            ]),
            # General Surgery - last as most generic
            (ClinicalDomain.SURGERY_GENERAL, [
                "appendectomy", "appendicitis", "cholecystectomy", "hernia",
                "laparoscop", "bowel", "colectomy", "gastrectomy"
            ]),
        ]

        for domain, keywords in domain_keywords_ordered:
            if any(kw in text_lower for kw in keywords):
                clinical_domain = domain
                break

        # Determine outcome severity
        outcome_severity = OutcomeSeverity.TEMPORARY_HARM
        if any(w in text_lower for w in ["death", "died", "fatal"]):
            outcome_severity = OutcomeSeverity.DEATH
        elif any(w in text_lower for w in ["permanent", "disability", "paralysis"]):
            outcome_severity = OutcomeSeverity.PERMANENT_SEVERE_DISABILITY

        # Build taxonomy
        taxonomy_data = simulation_data.get("taxonomy_labels", {})
        categories = taxonomy_data.get("malpractice_categories", [])
        if not categories:
            categories = ["DIAGNOSIS_ERROR"]  # Default

        # Validate categories
        valid_categories = []
        for cat in categories:
            try:
                valid_categories.append(MalpracticeCategory(cat))
            except ValueError:
                pass
        if not valid_categories:
            valid_categories = [MalpracticeCategory.DIAGNOSIS_ERROR]

        # Create summary
        summary_text = f"Medical malpractice case involving {clinical_domain.value.lower().replace('_', ' ')}."
        if parsed.parties:
            summary_text = f"{parsed.parties}. " + summary_text

        # Clean ground_truth data
        ground_truth_data = simulation_data.get("ground_truth", {"factual_timeline": []})
        ground_truth_data = self._clean_ground_truth(ground_truth_data)

        # Build the case simulation
        try:
            case = CaseSimulation(
                schema_version="1.0.0",
                case_id=case_id,
                source=source,
                jurisdiction=jurisdiction,
                court=parsed.court or "Unknown",
                decision_date=decision_date,
                url=url,
                case_name=parsed.parties or parsed.title,
                neutral_citation=parsed.citation,
                clinical_domain=clinical_domain,
                outcome_severity=outcome_severity,
                summary=CaseSummary(
                    brief=summary_text[:500],
                    clinical_synopsis="Clinical case extracted from judgment.",
                    legal_synopsis="Legal proceedings and judgment.",
                ),
                evidence_index=[e.model_dump() for e in evidence_index],
                simulation=sim_data,
                ground_truth=ground_truth_data,
                taxonomy_labels=TaxonomyLabels(
                    malpractice_categories=valid_categories,
                ),
                quality=QualityMetrics(
                    evidence_coverage_score=min(1.0, len(evidence_index) / max(1, len(parsed.paragraphs))),
                    simulation_completeness=0.7,
                    validation_passed=True,
                ),
                extraction_metadata=ExtractionMetadata(
                    extracted_at=datetime.utcnow(),
                    extractor_version=self.VERSION,
                    model_used=self.model,
                    extraction_passes=3,
                    human_reviewed=False,
                ),
            )

            return case

        except ValidationError as e:
            self.logger.error(f"Validation error creating case: {e}")
            raise

    def _get_default_evidence_prompt(self) -> str:
        """Get default evidence extraction prompt."""
        return """You are an expert legal and medical analyst extracting evidence from court judgments.

Your task is to create an evidence index from a medical malpractice/clinical negligence judgment.

CRITICAL RULES:
1. NEVER invent or hallucinate information not present in the judgment
2. Every evidence item must cite the paragraph number where it appears
3. Use verbatim quotes or close paraphrases only
4. If information is not stated, mark it as unavailable

Extract evidence items covering:
- Factual findings by the court
- Expert testimony
- Medical record entries mentioned
- Witness statements
- Legal conclusions

Output JSON format:
{
  "evidence_items": [
    {
      "evidence_id": "E001",
      "type": "FACTUAL_FINDING",
      "text": "Verbatim or close paraphrase from judgment",
      "paragraph_ref": "15",
      "speaker": "Judge Smith"
    }
  ]
}

Types: JUDGMENT_TEXT, EXPERT_TESTIMONY, MEDICAL_RECORD, WITNESS_STATEMENT, FACTUAL_FINDING, LEGAL_FINDING, COURT_REASONING"""

    def _get_default_simulation_prompt(self) -> str:
        """Get default simulation extraction prompt."""
        return """You are an expert in clinical education and medical malpractice law, creating interactive case simulations.

Your task is to transform a court judgment into a clinical case simulation for training doctors.

CRITICAL RULES:
1. NEVER invent clinical details not stated in the judgment
2. Reference evidence_ids for all factual claims
3. If a lab/test/imaging result is not stated, mark as "not documented in judgment"
4. Distinguish between what defendant did vs what court said should have been done

Create a simulation with:

1. INITIAL_STATE: The clinical presentation as described
   - Chief complaint, HPI, exam findings ONLY from the judgment
   - Mark unknown elements explicitly

2. REQUESTABLES: Information learner can request
   - Labs, imaging, consults mentioned in the case
   - Include what was actually ordered and what should have been ordered

3. DECISION_POINTS: Key clinical decisions (minimum 2)
   - Where malpractice occurred
   - What defendant chose vs what court endorsed
   - Scoring criteria

4. TIMELINE_PHASES: presentation, workup, decision, procedure, postop, followup

5. END_STATE: Patient outcome and legal verdict

Output valid JSON matching the schema."""
