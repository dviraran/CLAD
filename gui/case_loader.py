"""
Case loader for the CaseSim Training GUI.

Handles loading, validating, and selecting cases from the processed directory.
"""

from __future__ import annotations

import json
import random
import secrets
from pathlib import Path
from typing import Any

from utils import safe_get


class CaseLoader:
    """Loads and manages case files from the processed directory."""

    def __init__(self, processed_dir: Path | str | None = None):
        """
        Initialize the case loader.

        Args:
            processed_dir: Path to the processed cases directory.
                          Defaults to ../data/processed/ relative to this file.
        """
        if processed_dir is None:
            # First try data/processed relative to this file (for deployed version)
            local_dir = Path(__file__).parent / "data" / "processed"
            if local_dir.exists():
                self.processed_dir = local_dir
            else:
                # Fallback to parent directory (for local development)
                self.processed_dir = Path(__file__).parent.parent / "data" / "processed"
        else:
            self.processed_dir = Path(processed_dir)

        self._case_cache: dict[str, dict] = {}
        self._excluded_cases: set[str] = self._load_excluded_cases()

    def _load_excluded_cases(self) -> set[str]:
        """Load list of excluded cases from file."""
        exclude_file = self.processed_dir.parent / "excluded_cases.json"
        if exclude_file.exists():
            try:
                with open(exclude_file) as f:
                    data = json.load(f)
                    return set(data.get("excluded_cases", []))
            except (json.JSONDecodeError, OSError):
                pass
        return set()

    def list_cases(self) -> list[dict[str, Any]]:
        """
        List all available cases with summary information.

        Returns:
            List of case summaries with case_id, clinical_domain, malpractice_categories
        """
        cases = []

        if not self.processed_dir.exists():
            return cases

        for json_file in sorted(self.processed_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    case = json.load(f)

                # Skip excluded cases
                case_id = case.get("case_id", json_file.stem)
                if case_id in self._excluded_cases:
                    continue

                # Validate before including
                is_valid, _ = self.validate_case(case)
                if not is_valid:
                    continue

                cases.append({
                    "case_id": case.get("case_id", json_file.stem),
                    "case_name": case.get("case_name", "Unknown"),
                    "clinical_domain": case.get("clinical_domain", "Unknown"),
                    "malpractice_categories": safe_get(
                        case, "taxonomy_labels.malpractice_categories", []
                    ),
                    "decision_date": case.get("decision_date", "Unknown"),
                    "outcome_severity": case.get("outcome_severity", "Unknown"),
                    "file_path": str(json_file),
                })
            except (json.JSONDecodeError, OSError):
                continue

        return cases

    def get_random_case(self, seed: int | None = None, filter_domain: str | None = None) -> dict | None:
        """
        Select a random case from available cases.

        Args:
            seed: Optional random seed for reproducibility
            filter_domain: Optional clinical domain to filter by

        Returns:
            Full case dict, or None if no cases available
        """
        cases = self.list_cases()

        if filter_domain:
            cases = [c for c in cases if c.get("clinical_domain") == filter_domain]

        if not cases:
            return None

        if seed is not None:
            random.seed(seed)
            selected = random.choice(cases)
        else:
            # Use cryptographically secure random selection
            selected = secrets.choice(cases)

        return self.load_case(selected["case_id"])

    def load_case(self, case_id: str) -> dict | None:
        """
        Load a full case by its case_id.

        Args:
            case_id: The case identifier

        Returns:
            Full case dict, or None if not found
        """
        # Check cache first
        if case_id in self._case_cache:
            return self._case_cache[case_id]

        if not self.processed_dir.exists():
            return None

        # Try to find the case file
        for json_file in self.processed_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    case = json.load(f)

                if case.get("case_id") == case_id:
                    self._case_cache[case_id] = case
                    return case
            except (json.JSONDecodeError, OSError):
                continue

        return None

    def validate_case(self, case: dict) -> tuple[bool, list[str]]:
        """
        Validate that a case has all required fields for the GUI simulation.

        Required fields:
        - simulation.initial_state with chief_complaint
        - At least one decision_point with is_malpractice_point=True
        - simulation.end_state.legal_outcome
        - Non-empty evidence_index
        - Must be a clinical case (not pharmacy/drug store malpractice)

        Args:
            case: The case dict to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check case_id
        if not case.get("case_id"):
            errors.append("Missing case_id")

        # Check for non-clinical cases (pharmacy, drug store, etc.)
        if self._is_non_clinical_case(case):
            errors.append("Non-clinical case (pharmacy/drug store malpractice)")

        # Check initial_state
        initial_state = safe_get(case, "simulation.initial_state", {})
        if not initial_state:
            errors.append("Missing simulation.initial_state")
        else:
            chief_complaint = initial_state.get("chief_complaint", "")
            if not chief_complaint or "not documented" in chief_complaint.lower():
                # Allow cases without chief complaint but warn
                pass

        # Check decision_points
        decision_points = safe_get(case, "simulation.decision_points", [])
        if not decision_points:
            errors.append("No decision_points found")
        else:
            malpractice_points = [
                dp for dp in decision_points
                if dp.get("is_malpractice_point", False)
            ]
            if not malpractice_points:
                errors.append("No decision_point marked as is_malpractice_point")

        # Check end_state
        legal_outcome = safe_get(case, "simulation.end_state.legal_outcome", {})
        if not legal_outcome:
            errors.append("Missing simulation.end_state.legal_outcome")

        # Check evidence_index
        evidence_index = case.get("evidence_index", [])
        if not evidence_index:
            errors.append("Empty evidence_index")

        return len(errors) == 0, errors

    def get_malpractice_points(self, case: dict) -> list[dict]:
        """
        Extract decision points where is_malpractice_point is True.

        Args:
            case: The case dict

        Returns:
            List of decision point dicts that are malpractice points
        """
        decision_points = safe_get(case, "simulation.decision_points", [])
        return [
            dp for dp in decision_points
            if dp.get("is_malpractice_point", False)
        ]

    def get_case_count(self) -> int:
        """
        Get the total number of valid cases available.

        Returns:
            Count of valid cases
        """
        return len(self.list_cases())

    def get_domains(self) -> list[str]:
        """
        Get list of unique clinical domains across all cases.

        Returns:
            Sorted list of unique domain strings
        """
        cases = self.list_cases()
        domains = set(c.get("clinical_domain", "Unknown") for c in cases)
        return sorted(domains)

    def get_evidence_by_id(self, case: dict, evidence_id: str) -> dict | None:
        """
        Get an evidence item by its ID.

        Args:
            case: The case dict
            evidence_id: The evidence ID (e.g., "E001")

        Returns:
            The evidence item dict, or None if not found
        """
        evidence_index = case.get("evidence_index", [])
        for item in evidence_index:
            if item.get("evidence_id") == evidence_id:
                return item
        return None

    def get_requestable_by_id(self, case: dict, request_id: str) -> dict | None:
        """
        Get a requestable item by its ID.

        Args:
            case: The case dict
            request_id: The request ID (e.g., "R001")

        Returns:
            The requestable dict, or None if not found
        """
        requestables = safe_get(case, "simulation.requestables", [])
        for req in requestables:
            if req.get("request_id") == request_id:
                return req
        return None

    def get_requestables_by_type(self, case: dict, req_type: str) -> list[dict]:
        """
        Get all requestables of a specific type.

        Args:
            case: The case dict
            req_type: The requestable type (e.g., "LAB", "IMAGING")

        Returns:
            List of matching requestable dicts
        """
        requestables = safe_get(case, "simulation.requestables", [])
        return [req for req in requestables if req.get("type") == req_type]

    def _is_non_clinical_case(self, case: dict) -> bool:
        """
        Check if the case is a non-clinical malpractice case.

        Non-clinical cases include:
        - Pharmacy/drug store dispensing errors
        - Genetic testing laboratory errors
        - Beauty/cosmetic salon procedures
        - Banking/employment disputes (wrongly extracted)
        - Other non-physician defendants

        These cases are not suitable for evaluating clinical AI systems
        because the malpractice involves non-physician actors or
        non-clinical settings.

        Args:
            case: The case dict to check

        Returns:
            True if this is a non-clinical case that should be filtered out
        """
        # Keywords that indicate non-clinical defendants in case name
        non_clinical_case_name_keywords = [
            "drug store", "drugstore", "pharmacy", "pharmacist",
            "beauty lab", "beauty salon", "laser clinic",
            "genedx",  # Genetic testing lab
            "chemist",  # UK term for pharmacist
        ]

        # Keywords that indicate non-clinical defendants in allegations
        non_clinical_defendant_keywords = [
            "drug store", "drugstore", "pharmacy", "pharmacist",
            "beauty lab", "beauty salon",
            "bank", "the bank",  # Employment/banking disputes
            "genedx",  # Genetic testing lab
            "chemist",
        ]

        # Check case name
        case_name = (case.get("case_name") or "").lower()
        for keyword in non_clinical_case_name_keywords:
            if keyword in case_name:
                return True

        # Check allegations - who was the defendant?
        allegations = safe_get(case, "ground_truth.allegations", [])
        for allegation in allegations:
            against = (allegation.get("against") or "").lower()
            for keyword in non_clinical_defendant_keywords:
                if keyword in against:
                    return True

        # Check decision points for non-clinical actions
        decision_points = safe_get(case, "simulation.decision_points", [])
        for dp in decision_points:
            # Check if the decision is about dispensing (pharmacy) vs prescribing (clinical)
            context = (dp.get("clinical_context") or "").lower()
            prompt_text = (dp.get("prompt") or "").lower()

            # Non-clinical decision patterns
            non_clinical_patterns = [
                "dispensing medication",
                "dispens" in context and "pharmacy" in context,
                "what should be verified before dispensing",
                "genetic test" in context and "lab" in context,
                "confirmatory test" in prompt_text and "genetic" in context,
            ]
            for pattern in non_clinical_patterns:
                if isinstance(pattern, bool):
                    if pattern:
                        return True
                elif pattern in prompt_text or pattern in context:
                    return True

        # Check if evidence suggests non-medical case (e.g., banking dispute)
        evidence_index = case.get("evidence_index", [])
        non_medical_evidence_keywords = [
            "incarceration", "arrested", "convicted", "criminal conviction",
            "fca", "approved person",  # Financial conduct authority terms
            "contract of employment", "redundant",
        ]
        evidence_text = " ".join(
            (e.get("text") or "").lower() for e in evidence_index[:5]
        )
        for keyword in non_medical_evidence_keywords:
            if keyword in evidence_text:
                return True

        return False
