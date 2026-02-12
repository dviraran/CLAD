"""Filters for evaluating discovered cases."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, RejectionReason, StructuralHeuristicsConfig
from ..utils import get_logger


@dataclass
class FilterResult:
    """Result of applying a filter."""

    passed: bool
    reason: RejectionReason | None = None
    score_adjustment: float = 0.0
    details: str | None = None


class CaseFilter(ABC):
    """Base class for case filters."""

    name: str

    def __init__(self):
        self.logger = get_logger(f"filters.{self.name}")

    @abstractmethod
    def apply(
        self,
        record: DiscoveryRecord,
        content: str | None = None,
    ) -> FilterResult:
        """Apply the filter to a case."""
        ...


class StructuralFilter(CaseFilter):
    """Filter based on structural heuristics."""

    name = "structural"

    def __init__(self, config: StructuralHeuristicsConfig | None = None):
        super().__init__()
        self.config = config or StructuralHeuristicsConfig()

    def apply(
        self,
        record: DiscoveryRecord,
        content: str | None = None,
    ) -> FilterResult:
        """Apply structural heuristics."""
        if not content:
            # Can only check URL-based heuristics
            return self._check_url(record)

        # Check length
        if len(content) < self.config.min_length:
            return FilterResult(
                passed=False,
                reason=RejectionReason.TOO_SHORT,
                details=f"Length {len(content)} < {self.config.min_length}",
            )

        # Check for required headings
        heading_score = self._check_headings(content)

        # Check for court type
        court_score = self._check_court(record)

        # Calculate overall score adjustment
        score_adjustment = heading_score + court_score

        # Must have at least some headings
        if heading_score <= 0:
            return FilterResult(
                passed=False,
                reason=RejectionReason.PROCEDURAL_ONLY,
                details="No relevant section headings found",
            )

        return FilterResult(
            passed=True,
            score_adjustment=score_adjustment,
        )

    def _check_url(self, record: DiscoveryRecord) -> FilterResult:
        """Check URL-based heuristics."""
        url_str = str(record.url).lower()

        # Boost for trial-level courts
        for court in self.config.preferred_courts:
            if court.lower() in url_str:
                return FilterResult(passed=True, score_adjustment=0.1)

        return FilterResult(passed=True)

    def _check_headings(self, content: str) -> float:
        """Check for relevant section headings."""
        content_lower = content.lower()
        matches = 0

        for heading in self.config.required_headings:
            # Check various formats
            patterns = [
                heading.lower(),
                heading.lower().replace(" ", ""),
                heading.upper(),
            ]
            for pattern in patterns:
                if pattern in content_lower:
                    matches += 1
                    break

        if matches >= self.config.min_heading_matches:
            return min(0.2, matches * 0.05)

        return 0.0

    def _check_court(self, record: DiscoveryRecord) -> float:
        """Check court type."""
        if record.court:
            for court in self.config.preferred_courts:
                if court in record.court:
                    return 0.1
        return 0.0


class ContentFilter(CaseFilter):
    """Filter based on content analysis.

    This filter determines whether a case is likely to be a medical
    malpractice case suitable for clinical simulation. It checks for:
    1. Strong medical/clinical indicators (must have enough)
    2. Disqualifying non-medical indicators (instant rejection)
    3. Weak negative indicators (reduce score)
    4. Clinical fact patterns (patient encounters, treatments, etc.)
    """

    name = "content"

    # Strong medical negligence indicators - case must have several of these
    POSITIVE_INDICATORS = [
        # Medical negligence legal terms (strong)
        r"clinical negligence",
        r"medical negligence",
        r"medical malpractice",
        r"breach of duty.{0,30}(doctor|hospital|surgeon|clinician|nurse)",
        r"standard of care",
        r"informed consent",
        r"bolam",
        r"bolitho",

        # Clinical encounter terms (strong)
        r"the patient",
        r"the claimant.{0,20}(presented|attended|admitted|underwent)",
        r"admitted to hospital",
        r"presented to.{0,20}(hospital|a&e|emergency|gp|doctor)",
        r"underwent.{0,20}(surgery|procedure|operation|treatment)",

        # Healthcare provider terms
        r"consultant",
        r"surgeon",
        r"registrar",
        r"general practitioner",
        r"\bgp\b",
        r"obstetrician",
        r"anaesthetist",
        r"midwife",
        r"radiologist",

        # Clinical terms
        r"diagnosis",
        r"prognosis",
        r"symptoms",
        r"examination",
        r"blood test",
        r"scan",
        r"\bmri\b",
        r"\bct scan",
        r"x-ray",
        r"biopsy",

        # Treatment terms
        r"prescription",
        r"medication",
        r"antibiotics",
        r"chemotherapy",
        r"radiotherapy",
        r"surgery",
        r"operation",
        r"procedure",

        # Clinical outcome terms
        r"complications",
        r"haemorrhage",
        r"hemorrhage",
        r"sepsis",
        r"infection",
        r"delayed diagnosis",
        r"misdiagnosis",
        r"failure to diagnose",
    ]

    # Disqualifying indicators - if these are prominent, reject the case
    # These indicate the case is NOT about medical consultation
    DISQUALIFYING_INDICATORS = [
        # Social services / child protection
        r"social services",
        r"local authority.{0,30}(child|care order|protection)",
        r"child protection",
        r"care order",
        r"care proceedings",
        r"foster care",
        r"safeguarding",

        # Criminal matters (not civil medical negligence)
        r"criminal (conviction|proceedings|trial|appeal)",
        r"crown court.{0,30}(convicted|sentenced|guilty)",
        r"false imprisonment",
        r"sexual (assault|abuse|offence)",
        r"\brape\b",
        r"\bbuggery\b",
        r"indecent assault",

        # Fraud / dishonesty cases
        r"fundamental dishonesty",
        r"contempt of court",
        r"perjury",
        r"fraudulent (claim|conduct|misrepresentation)",
        r"struck out.{0,30}(dishonesty|fraud|abuse)",

        # Travel / premises liability
        r"tour operator",
        r"package holiday",
        r"(hotel|resort).{0,20}accident",
        r"premises liability",
        r"occupier.{0,10}liability",

        # Employment / workplace
        r"employment tribunal",
        r"unfair dismissal",
        r"workplace (accident|injury)",
        r"industrial (accident|injury|disease)",
        r"health and safety.{0,20}(work|employ)",

        # Other non-medical
        r"road traffic accident",
        r"rta claim",
        r"motor (accident|collision)",
        r"immigration (tribunal|appeal)",
        r"planning permission",
        r"judicial review.{0,30}(secretary|minister|council)",
    ]

    # Weak negative indicators - reduce score but don't reject
    WEAK_NEGATIVE_INDICATORS = [
        r"personal injury(?!.{0,20}clinical)",
        r"limitation (period|act)",
        r"costs (assessment|order)",
        r"summary judgment",
        r"strike out",
        r"procedural",
    ]

    # Minimum thresholds
    MIN_POSITIVE_INDICATORS = 5  # Increased from 3
    MIN_STRONG_MEDICAL_TERMS = 2  # Must have at least 2 strong medical terms

    # Strong medical terms that confirm this is a medical case
    STRONG_MEDICAL_TERMS = [
        r"clinical negligence",
        r"medical negligence",
        r"the patient",
        r"(doctor|surgeon|consultant|gp).{0,30}(failed|negligent|breach)",
        r"hospital.{0,30}(trust|nhs)",
        r"admitted to hospital",
        r"underwent.{0,20}(surgery|procedure|operation)",
        r"diagnosis",
        r"treatment",
    ]

    def apply(
        self,
        record: DiscoveryRecord,
        content: str | None = None,
    ) -> FilterResult:
        """Apply content analysis."""
        if not content:
            return FilterResult(passed=True)

        content_lower = content.lower()

        # First check for disqualifying indicators
        disqualifying_matches = []
        for pattern in self.DISQUALIFYING_INDICATORS:
            matches = re.findall(pattern, content_lower)
            if matches:
                disqualifying_matches.append((pattern, len(matches)))

        # If we have strong disqualifying indicators, check if medical context overrides
        if disqualifying_matches:
            # Count strong medical terms
            strong_medical_count = sum(
                1 for p in self.STRONG_MEDICAL_TERMS
                if re.search(p, content_lower)
            )

            # Calculate disqualification strength
            disqualifying_strength = sum(count for _, count in disqualifying_matches)

            # If disqualifying indicators are strong and medical terms are weak, reject
            if disqualifying_strength >= 3 and strong_medical_count < 3:
                top_disqualifier = max(disqualifying_matches, key=lambda x: x[1])
                return FilterResult(
                    passed=False,
                    reason=RejectionReason.NOT_MEDICAL,
                    details=f"Disqualifying content: '{top_disqualifier[0]}' ({top_disqualifier[1]} matches)",
                )

            # If roughly equal, still reject - better to be conservative
            if disqualifying_strength > strong_medical_count:
                return FilterResult(
                    passed=False,
                    reason=RejectionReason.NOT_MEDICAL,
                    details=f"Non-medical indicators ({disqualifying_strength}) outweigh medical ({strong_medical_count})",
                )

        # Count positive indicators
        positive_count = 0
        matched_indicators = []
        for pattern in self.POSITIVE_INDICATORS:
            if re.search(pattern, content_lower):
                positive_count += 1
                matched_indicators.append(pattern)

        # Count strong medical terms
        strong_medical_count = sum(
            1 for p in self.STRONG_MEDICAL_TERMS
            if re.search(p, content_lower)
        )

        # Must have minimum positive indicators
        if positive_count < self.MIN_POSITIVE_INDICATORS:
            return FilterResult(
                passed=False,
                reason=RejectionReason.NOT_MEDICAL,
                details=f"Only {positive_count} medical indicators (need {self.MIN_POSITIVE_INDICATORS})",
            )

        # Must have minimum strong medical terms
        if strong_medical_count < self.MIN_STRONG_MEDICAL_TERMS:
            return FilterResult(
                passed=False,
                reason=RejectionReason.NOT_MEDICAL,
                details=f"Only {strong_medical_count} strong medical terms (need {self.MIN_STRONG_MEDICAL_TERMS})",
            )

        # Count weak negative indicators (reduce score but don't reject)
        weak_negative_count = sum(
            1 for p in self.WEAK_NEGATIVE_INDICATORS
            if re.search(p, content_lower)
        )

        # Check for clinical facts section
        has_clinical_facts = self._has_clinical_facts(content)

        # Calculate score adjustment
        score_adjustment = min(0.3, positive_count * 0.015)
        score_adjustment -= weak_negative_count * 0.03

        if not has_clinical_facts:
            score_adjustment -= 0.1

        return FilterResult(
            passed=True,
            score_adjustment=score_adjustment,
            details=f"Found {positive_count} medical indicators, {strong_medical_count} strong terms",
        )

    def _has_clinical_facts(self, content: str) -> bool:
        """Check if content has clinical facts describing a patient encounter."""
        clinical_fact_patterns = [
            # Date patterns with medical context
            r"on \d{1,2}(?:st|nd|rd|th)? (?:january|february|march|april|may|june|july|august|september|october|november|december).{0,50}(hospital|doctor|gp|surgery|admitted)",
            # Patient encounter patterns
            r"(the patient|the claimant|mrs?\.?\s+\w+).{0,30}(presented|attended|admitted|seen by)",
            r"admitted to.{0,20}(hospital|ward|unit)",
            r"underwent.{0,20}(surgery|procedure|operation|examination)",
            r"was diagnosed with",
            r"complained of.{0,30}(pain|symptoms|discomfort)",
            r"examination (revealed|showed|demonstrated)",
            r"blood tests? (showed|revealed|demonstrated)",
            r"(ct|mri|x-ray|scan).{0,30}(showed|revealed|demonstrated)",
            # Treatment patterns
            r"was (prescribed|given|administered)",
            r"received (treatment|antibiotics|medication)",
            r"discharged (from|with)",
        ]

        content_lower = content.lower()
        matches = sum(1 for p in clinical_fact_patterns if re.search(p, content_lower))
        return matches >= 3  # Increased from 2


class MedicalDomainFilter(CaseFilter):
    """Filter to identify medical domain/specialty."""

    name = "domain"

    DOMAIN_PATTERNS = {
        "SURGERY_GENERAL": [r"general surgery", r"laparoscop", r"appendix", r"hernia"],
        "SURGERY_ORTHOPAEDIC": [r"orthopaedic", r"orthopedic", r"hip replacement", r"knee", r"fracture", r"bone"],
        "SURGERY_CARDIAC": [r"cardiac surgery", r"heart surgery", r"bypass", r"valve replacement"],
        "SURGERY_NEURO": [r"neurosurgery", r"brain surgery", r"spinal surgery", r"craniotomy"],
        "SURGERY_VASCULAR": [r"vascular surgery", r"aneurysm", r"arterial"],
        "SURGERY_OPHTHALMIC": [r"eye surgery", r"ophthalmic", r"cataract", r"laser eye"],
        "OBSTETRICS_GYNAECOLOGY": [r"obstetric", r"gynaecolog", r"pregnancy", r"labour", r"cesarean", r"birth"],
        "ONCOLOGY": [r"oncolog", r"cancer", r"tumour", r"chemotherapy", r"radiotherapy"],
        "CARDIOLOGY": [r"cardiolog", r"heart attack", r"myocardial", r"cardiac arrest"],
        "NEUROLOGY": [r"neurolog", r"stroke", r"brain", r"epilepsy"],
        "EMERGENCY_MEDICINE": [r"emergency department", r"a&e", r"accident and emergency"],
        "ANAESTHESIA": [r"anaesthe", r"anesthe", r"intubation"],
        "RADIOLOGY": [r"radiolog", r"x-ray", r"ct scan", r"mri", r"ultrasound"],
        "PRIMARY_CARE": [r"general practitioner", r"gp", r"family doctor", r"primary care"],
    }

    def apply(
        self,
        record: DiscoveryRecord,
        content: str | None = None,
    ) -> FilterResult:
        """Identify medical domain."""
        if not content:
            return FilterResult(passed=True)

        content_lower = content.lower()
        detected_domains: list[str] = []

        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    detected_domains.append(domain)
                    break

        if detected_domains:
            # Primary domain is the first detected
            return FilterResult(
                passed=True,
                score_adjustment=0.05,
                details=f"Detected domains: {', '.join(detected_domains)}",
            )

        return FilterResult(passed=True, details="Domain not identified")
