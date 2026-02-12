"""Rule-based extractor for metadata and structural elements."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional, List

from ..parsing import ParsedJudgment, Paragraph
from ..schemas import (
    ClinicalDomain,
    EvidenceItem,
    EvidenceType,
    Jurisdiction,
    MalpracticeCategory,
    OutcomeSeverity,
    Source,
)
from ..utils import get_logger


@dataclass
class ExtractedMetadata:
    """Metadata extracted using rules."""

    title: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    judge: Optional[str] = None
    date: Optional[date] = None
    parties: Optional[str] = None
    source: Optional[Source] = None
    jurisdiction: Optional[Jurisdiction] = None
    clinical_domain: Optional[ClinicalDomain] = None
    outcome_severity: Optional[OutcomeSeverity] = None
    malpractice_categories: List[MalpracticeCategory] = field(default_factory=list)


@dataclass
class ExtractedSection:
    """A section identified by rules."""

    name: str
    start_para: int
    end_para: int
    paragraphs: list[Paragraph] = field(default_factory=list)


class RuleBasedExtractor:
    """Extracts structured information using pattern matching rules."""

    # Citation patterns by jurisdiction
    CITATION_PATTERNS = {
        "UK": [
            r"\[(\d{4})\]\s+(EWHC|EWCA|UKSC|UKHL|EWCOP)\s+(\d+)(?:\s*\(([A-Za-z]+)\))?",
        ],
        "CA": [
            r"(\d{4})\s+(ONSC|ONCA|BCSC|BCCA|ABQB|ABCA|SCC|QCCS|QCCA)\s+(\d+)",
            r"(\d{4})\s+CanLII\s+(\d+)",
        ],
        "AU": [
            r"\[(\d{4})\]\s+(HCA|NSWSC|NSWCA|VSC|VSCA|QSC|QCA|FCA)\s+(\d+)",
        ],
        "US": [
            r"(\d+)\s+U\.S\.\s+(\d+)",
            r"(\d+)\s+F\.\s*(?:2d|3d)?\s+(\d+)",
        ],
    }

    # Section heading patterns
    SECTION_PATTERNS = {
        "facts": [
            r"^(?:the\s+)?facts?\s*$",
            r"^factual\s+background\s*$",
            r"^background\s*$",
        ],
        "clinical": [
            r"^clinical\s+(?:background|history|chronology)\s*$",
            r"^medical\s+(?:history|background)\s*$",
            r"^the\s+medical\s+evidence\s*$",
        ],
        "expert_evidence": [
            r"^expert\s+(?:evidence|testimony|witnesses?)\s*$",
            r"^the\s+experts?\s*$",
        ],
        "consent": [
            r"^(?:informed\s+)?consent\s*$",
            r"^the\s+consent\s+(?:issue|process)\s*$",
        ],
        "breach": [
            r"^breach(?:\s+of\s+duty)?\s*$",
            r"^standard\s+of\s+care\s*$",
            r"^negligence\s*$",
        ],
        "causation": [
            r"^causation\s*$",
            r"^cause\s+and\s+effect\s*$",
        ],
        "damages": [
            r"^damages?\s*$",
            r"^quantum\s*$",
            r"^compensation\s*$",
        ],
        "conclusion": [
            r"^conclusion\s*$",
            r"^judgment\s*$",
            r"^disposition\s*$",
            r"^decision\s*$",
        ],
    }

    # Clinical domain indicators
    DOMAIN_INDICATORS = {
        ClinicalDomain.SURGERY_GENERAL: [
            "general surg", "laparoscop", "appendectomy", "cholecystectomy",
            "hernia repair", "bowel", "abdominal surgery",
        ],
        ClinicalDomain.SURGERY_ORTHOPAEDIC: [
            "orthopaedic", "orthopedic", "hip replacement", "knee replacement",
            "fracture", "joint replacement", "arthroscop",
        ],
        ClinicalDomain.SURGERY_CARDIAC: [
            "cardiac surgery", "heart surgery", "bypass", "valve replacement",
            "cabg", "cardiothoracic",
        ],
        ClinicalDomain.SURGERY_NEURO: [
            "neurosurg", "brain surgery", "spinal surgery", "craniotomy",
            "laminectomy", "discectomy",
        ],
        ClinicalDomain.OBSTETRICS_GYNAECOLOGY: [
            "obstetric", "gynaecolog", "gynecolog", "pregnancy", "labour",
            "cesarean", "c-section", "birth", "delivery", "midwi",
        ],
        ClinicalDomain.ONCOLOGY: [
            "oncolog", "cancer", "tumour", "tumor", "chemotherapy",
            "radiotherapy", "malignant", "metastas",
        ],
        ClinicalDomain.CARDIOLOGY: [
            "cardiolog", "heart attack", "myocardial infarction", "mi ",
            "cardiac arrest", "echocardiog", "angiogra",
        ],
        ClinicalDomain.NEUROLOGY: [
            "neurolog", "stroke", "cva", "epilepsy", "seizure", "multiple sclerosis",
        ],
        ClinicalDomain.EMERGENCY_MEDICINE: [
            "emergency department", "a&e", "accident and emergency",
            "emergency room", " ed ", "casualty",
        ],
        ClinicalDomain.ANAESTHESIA: [
            "anaesthe", "anesthe", "intubation", "general anaesthetic",
        ],
        ClinicalDomain.RADIOLOGY: [
            "radiolog", "x-ray", "ct scan", "mri scan", "ultrasound",
            "imaging department",
        ],
        ClinicalDomain.PRIMARY_CARE: [
            "general practitioner", " gp ", "family doctor", "primary care",
            "surgery hours",
        ],
        ClinicalDomain.PSYCHIATRY: [
            "psychiatr", "mental health", "depression", "psychosis",
            "sectioned", "mental health act",
        ],
    }

    # Outcome severity indicators
    SEVERITY_INDICATORS = {
        OutcomeSeverity.DEATH: [
            "died", "death", "fatal", "deceased", "passed away", "mortality",
        ],
        OutcomeSeverity.PERMANENT_SEVERE_DISABILITY: [
            "permanent disability", "paralysis", "brain damage", "amputation",
            "blindness", "permanent vegetative", "severe brain injury",
        ],
        OutcomeSeverity.PERMANENT_MODERATE_DISABILITY: [
            "chronic pain", "permanent", "ongoing disability", "lasting",
            "residual deficit",
        ],
        OutcomeSeverity.TEMPORARY_HARM: [
            "recovered", "temporary", "resolved", "healed",
        ],
    }

    # Malpractice category indicators
    CATEGORY_INDICATORS = {
        MalpracticeCategory.DIAGNOSIS_ERROR: [
            "failed to diagnose", "missed diagnosis", "misdiagnos", "delayed diagnosis",
            "failure to diagnose", "diagnostic error", "wrong diagnosis",
        ],
        MalpracticeCategory.TREATMENT_PROCEDURE_ERROR: [
            "surgical error", "wrong site", "operative error", "procedural error",
            "retained foreign", "wrong patient", "technique",
        ],
        MalpracticeCategory.INFORMED_CONSENT: [
            "informed consent", "consent", "failed to warn", "did not disclose",
            "material risk", "montgomery", "alternatives",
        ],
        MalpracticeCategory.MONITORING_FOLLOWUP: [
            "failed to monitor", "follow-up", "monitoring", "observation",
            "post-operative care", "aftercare",
        ],
        MalpracticeCategory.MEDICATION_ERROR: [
            "wrong dose", "medication error", "drug error", "prescription error",
            "adverse drug", "allergic reaction", "drug interaction",
        ],
        MalpracticeCategory.SYSTEM_COMMUNICATION: [
            "communication failure", "handover", "referral", "failed to communicate",
            "system failure", "coordination",
        ],
        MalpracticeCategory.DOCUMENTATION: [
            "documentation", "record keeping", "notes", "failed to document",
        ],
    }

    def __init__(self):
        self.logger = get_logger("extraction.rule_based")

    def extract_metadata(self, parsed: ParsedJudgment, url: str = "") -> ExtractedMetadata:
        """Extract metadata from parsed judgment."""
        metadata = ExtractedMetadata()

        # Extract from parsed data
        metadata.title = parsed.title
        metadata.citation = parsed.citation
        metadata.court = parsed.court
        metadata.judge = parsed.judge
        metadata.parties = parsed.parties

        # Parse date
        if parsed.date:
            metadata.date = self._parse_date(parsed.date)

        # Determine source and jurisdiction from URL
        metadata.source, metadata.jurisdiction = self._detect_source(url)

        # Extract citation if not found
        if not metadata.citation:
            metadata.citation = self._extract_citation(
                parsed.raw_text,
                metadata.jurisdiction,
            )

        # Detect clinical domain
        metadata.clinical_domain = self._detect_domain(parsed.raw_text)

        # Detect outcome severity
        metadata.outcome_severity = self._detect_severity(parsed.raw_text)

        # Detect malpractice categories
        metadata.malpractice_categories = self._detect_categories(parsed.raw_text)

        return metadata

    def extract_sections(self, parsed: ParsedJudgment) -> list[ExtractedSection]:
        """Extract document sections using heading patterns."""
        sections: list[ExtractedSection] = []
        current_section: ExtractedSection | None = None

        for i, para in enumerate(parsed.paragraphs):
            if para.is_heading:
                # Check if this matches a known section
                section_type = self._match_section(para.text)

                if section_type:
                    # Close previous section
                    if current_section:
                        current_section.end_para = i - 1
                        sections.append(current_section)

                    # Start new section
                    current_section = ExtractedSection(
                        name=section_type,
                        start_para=i,
                        end_para=i,  # Will be updated
                    )

            # Add paragraph to current section
            if current_section:
                current_section.paragraphs.append(para)

        # Close final section
        if current_section:
            current_section.end_para = len(parsed.paragraphs) - 1
            sections.append(current_section)

        return sections

    def extract_evidence_items(
        self,
        parsed: ParsedJudgment,
        sections: list[ExtractedSection],
    ) -> list[EvidenceItem]:
        """Extract evidence items using rule-based patterns."""
        evidence: list[EvidenceItem] = []
        evidence_id = 1

        # Extract from expert evidence sections
        for section in sections:
            if section.name == "expert_evidence":
                for para in section.paragraphs:
                    if self._is_expert_statement(para.text):
                        evidence.append(EvidenceItem(
                            evidence_id=f"E{evidence_id:03d}",
                            type=EvidenceType.EXPERT_TESTIMONY,
                            text=para.text[:500],
                            paragraph_ref=para.number or str(evidence_id),
                        ))
                        evidence_id += 1

        # Extract factual findings
        for section in sections:
            if section.name in ["facts", "clinical"]:
                for para in section.paragraphs:
                    if self._is_factual_statement(para.text):
                        evidence.append(EvidenceItem(
                            evidence_id=f"E{evidence_id:03d}",
                            type=EvidenceType.FACTUAL_FINDING,
                            text=para.text[:500],
                            paragraph_ref=para.number or str(evidence_id),
                        ))
                        evidence_id += 1

        # Extract legal findings
        for section in sections:
            if section.name in ["conclusion", "breach", "causation"]:
                for para in section.paragraphs:
                    if self._is_legal_finding(para.text):
                        evidence.append(EvidenceItem(
                            evidence_id=f"E{evidence_id:03d}",
                            type=EvidenceType.LEGAL_FINDING,
                            text=para.text[:500],
                            paragraph_ref=para.number or str(evidence_id),
                        ))
                        evidence_id += 1

        return evidence

    def _parse_date(self, date_str: str) -> date | None:
        """Parse date from various formats."""
        try:
            import dateutil.parser
            return dateutil.parser.parse(date_str).date()
        except Exception:
            return None

    def _detect_source(self, url: str) -> tuple[Source | None, Jurisdiction | None]:
        """Detect source and jurisdiction from URL."""
        url_lower = url.lower()

        if "bailii.org" in url_lower:
            return Source.BAILII, Jurisdiction.UK
        elif "canlii.org" in url_lower:
            return Source.CANLII, Jurisdiction.CA
        elif "austlii.edu.au" in url_lower:
            return Source.AUSTLII, Jurisdiction.AU
        elif "courtlistener.com" in url_lower:
            return Source.COURTLISTENER, Jurisdiction.US

        return None, None

    def _extract_citation(
        self,
        text: str,
        jurisdiction: Jurisdiction | None,
    ) -> str | None:
        """Extract citation using jurisdiction-specific patterns."""
        if jurisdiction:
            patterns = self.CITATION_PATTERNS.get(jurisdiction.value, [])
        else:
            # Try all patterns
            patterns = []
            for p_list in self.CITATION_PATTERNS.values():
                patterns.extend(p_list)

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None

    def _match_section(self, heading: str) -> str | None:
        """Match heading to section type."""
        heading_lower = heading.lower().strip()

        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, heading_lower, re.IGNORECASE):
                    return section_type

        return None

    def _detect_domain(self, text: str) -> ClinicalDomain:
        """Detect clinical domain from text."""
        text_lower = text.lower()
        domain_scores: dict[ClinicalDomain, int] = {}

        for domain, indicators in self.DOMAIN_INDICATORS.items():
            score = sum(1 for ind in indicators if ind in text_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return ClinicalDomain.OTHER

    def _detect_severity(self, text: str) -> OutcomeSeverity:
        """Detect outcome severity from text."""
        text_lower = text.lower()

        for severity, indicators in self.SEVERITY_INDICATORS.items():
            if any(ind in text_lower for ind in indicators):
                return severity

        return OutcomeSeverity.TEMPORARY_HARM

    def _detect_categories(self, text: str) -> list[MalpracticeCategory]:
        """Detect malpractice categories from text."""
        text_lower = text.lower()
        categories: list[MalpracticeCategory] = []

        for category, indicators in self.CATEGORY_INDICATORS.items():
            if any(ind in text_lower for ind in indicators):
                categories.append(category)

        return categories or [MalpracticeCategory.DIAGNOSIS_ERROR]

    def _is_expert_statement(self, text: str) -> bool:
        """Check if text is an expert statement."""
        indicators = [
            "professor", "dr ", "dr.", "expert", "opinion",
            "in my view", "standard of care", "would have",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in indicators)

    def _is_factual_statement(self, text: str) -> bool:
        """Check if text is a factual statement."""
        indicators = [
            "on ", "the patient", "the claimant", "presented",
            "underwent", "was diagnosed", "attended", "complained",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in indicators)

    def _is_legal_finding(self, text: str) -> bool:
        """Check if text is a legal finding."""
        indicators = [
            "i find", "i conclude", "liability", "breach",
            "negligent", "causation", "damages", "judgment",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in indicators)
