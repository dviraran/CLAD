"""Rule-based classification for specialty and malpractice type."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .models import MalpracticeType


class RuleClassifier:
    """Classifies specialty and malpractice type using YAML-configured regex rules."""

    def __init__(self, config_dir: Path | str | None = None):
        """
        Initialize classifier with rules from YAML files.

        Args:
            config_dir: Path to directory containing specialty_rules.yaml and malpractice_rules.yaml
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"
        else:
            config_dir = Path(config_dir)

        self.specialty_rules: list[dict[str, str]] = []
        self.specialty_default: str = "unknown"
        self.malpractice_rules: list[dict[str, str]] = []
        self.malpractice_default: str = "other"
        self.malpractice_priority: list[str] = []

        self._load_specialty_rules(config_dir / "specialty_rules.yaml")
        self._load_malpractice_rules(config_dir / "malpractice_rules.yaml")

    def _load_specialty_rules(self, path: Path) -> None:
        """Load specialty classification rules from YAML."""
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.specialty_rules = data.get("rules", [])
        self.specialty_default = data.get("default", "unknown")

        # Pre-compile regex patterns
        for rule in self.specialty_rules:
            rule["_compiled"] = re.compile(rule["pattern"], re.IGNORECASE)

    def _load_malpractice_rules(self, path: Path) -> None:
        """Load malpractice type classification rules from YAML."""
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.malpractice_rules = data.get("rules", [])
        self.malpractice_default = data.get("default", "other")
        self.malpractice_priority = data.get("priority", [])

        # Pre-compile regex patterns
        for rule in self.malpractice_rules:
            rule["_compiled"] = re.compile(rule["pattern"], re.IGNORECASE)

    def classify_specialty(self, text: str) -> str:
        """
        Classify specialty from text using regex rules.

        Args:
            text: Combined text from system prompt and evaluation fields

        Returns:
            Specialty name or default if no match
        """
        if not text:
            return self.specialty_default

        for rule in self.specialty_rules:
            compiled = rule.get("_compiled")
            if compiled and compiled.search(text):
                return rule["specialty"]

        return self.specialty_default

    def classify_malpractice_type(self, text: str) -> MalpracticeType:
        """
        Classify malpractice type from text using regex rules.

        Args:
            text: Combined text from evaluation fields

        Returns:
            MalpracticeType enum value
        """
        if not text:
            return MalpracticeType(self.malpractice_default)

        # Find all matching types
        matches: set[str] = set()

        for rule in self.malpractice_rules:
            compiled = rule.get("_compiled")
            if compiled and compiled.search(text):
                matches.add(rule["type"])

        if not matches:
            return MalpracticeType(self.malpractice_default)

        # If multiple matches, use priority order
        if len(matches) > 1 and self.malpractice_priority:
            for priority_type in self.malpractice_priority:
                if priority_type in matches:
                    return MalpracticeType(priority_type)

        # Return first match
        return MalpracticeType(matches.pop())

    def build_specialty_text(
        self,
        system_prompt: str,
        defendant_action: str | None = None,
        expected_action: str | None = None,
        feedback: str | None = None,
    ) -> str:
        """
        Build combined text for specialty classification.

        Args:
            system_prompt: System prompt from conversation
            defendant_action: From evaluation
            expected_action: From evaluation
            feedback: From evaluation

        Returns:
            Combined text string
        """
        parts = [system_prompt or ""]
        if defendant_action:
            parts.append(defendant_action)
        if expected_action:
            parts.append(expected_action)
        if feedback:
            parts.append(feedback)
        return " ".join(parts)

    def build_malpractice_text(
        self,
        defendant_action: str | None = None,
        expected_action: str | None = None,
        feedback: str | None = None,
        checklist_criteria: list[str] | None = None,
    ) -> str:
        """
        Build combined text for malpractice type classification.

        Args:
            defendant_action: From evaluation
            expected_action: From evaluation
            feedback: From evaluation
            checklist_criteria: List of criterion strings from checklist

        Returns:
            Combined text string
        """
        parts = []
        if defendant_action:
            parts.append(defendant_action)
        if expected_action:
            parts.append(expected_action)
        if feedback:
            parts.append(feedback)
        if checklist_criteria:
            parts.extend(checklist_criteria)
        return " ".join(parts)


# Singleton instance for convenience
_classifier: RuleClassifier | None = None


def get_classifier(config_dir: Path | str | None = None) -> RuleClassifier:
    """Get or create singleton classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RuleClassifier(config_dir)
    return _classifier
