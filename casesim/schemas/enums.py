"""Enumeration types for the case simulation schema."""

from enum import Enum


class Source(str, Enum):
    """Legal database sources."""

    BAILII = "BAILII"
    CANLII = "CanLII"
    AUSTLII = "AustLII"
    COURTLISTENER = "CourtListener"
    NZLII = "NZLII"
    HKLII = "HKLII"
    SINGAPORE = "Singapore"
    JADE = "JADE"


class Jurisdiction(str, Enum):
    """Legal jurisdictions."""

    UK = "UK"
    CA = "CA"
    AU = "AU"
    US = "US"
    NZ = "NZ"
    HK = "HK"
    SG = "SG"


class ClinicalDomain(str, Enum):
    """Primary clinical specialty involved."""

    SURGERY_GENERAL = "SURGERY_GENERAL"
    SURGERY_ORTHOPAEDIC = "SURGERY_ORTHOPAEDIC"
    SURGERY_CARDIAC = "SURGERY_CARDIAC"
    SURGERY_NEURO = "SURGERY_NEURO"
    SURGERY_VASCULAR = "SURGERY_VASCULAR"
    SURGERY_PLASTIC = "SURGERY_PLASTIC"
    SURGERY_OPHTHALMIC = "SURGERY_OPHTHALMIC"
    OBSTETRICS_GYNAECOLOGY = "OBSTETRICS_GYNAECOLOGY"
    ONCOLOGY = "ONCOLOGY"
    CARDIOLOGY = "CARDIOLOGY"
    NEUROLOGY = "NEUROLOGY"
    EMERGENCY_MEDICINE = "EMERGENCY_MEDICINE"
    INTERNAL_MEDICINE = "INTERNAL_MEDICINE"
    PAEDIATRICS = "PAEDIATRICS"
    PSYCHIATRY = "PSYCHIATRY"
    RADIOLOGY = "RADIOLOGY"
    ANAESTHESIA = "ANAESTHESIA"
    PRIMARY_CARE = "PRIMARY_CARE"
    OTHER = "OTHER"


class OutcomeSeverity(str, Enum):
    """Severity of patient outcome."""

    DEATH = "DEATH"
    PERMANENT_SEVERE_DISABILITY = "PERMANENT_SEVERE_DISABILITY"
    PERMANENT_MODERATE_DISABILITY = "PERMANENT_MODERATE_DISABILITY"
    TEMPORARY_HARM = "TEMPORARY_HARM"
    MINOR_HARM = "MINOR_HARM"
    NO_PHYSICAL_HARM = "NO_PHYSICAL_HARM"


class EvidenceType(str, Enum):
    """Types of evidence from judgments."""

    JUDGMENT_TEXT = "JUDGMENT_TEXT"
    EXPERT_TESTIMONY = "EXPERT_TESTIMONY"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    WITNESS_STATEMENT = "WITNESS_STATEMENT"
    FACTUAL_FINDING = "FACTUAL_FINDING"
    LEGAL_FINDING = "LEGAL_FINDING"
    COURT_REASONING = "COURT_REASONING"


class RequestableType(str, Enum):
    """Types of requestable information."""

    LAB = "LAB"
    IMAGING = "IMAGING"
    PATHOLOGY = "PATHOLOGY"
    CONSULT_NOTE = "CONSULT_NOTE"
    MDT_NOTE = "MDT_NOTE"
    OP_NOTE = "OP_NOTE"
    CONSENT_DISCUSSION = "CONSENT_DISCUSSION"
    FOLLOWUP = "FOLLOWUP"
    VITAL_SIGNS = "VITAL_SIGNS"
    NURSING_NOTES = "NURSING_NOTES"
    PROCEDURE_NOTE = "PROCEDURE_NOTE"


class PhaseId(str, Enum):
    """Timeline phase identifiers."""

    PRESENTATION = "presentation"
    WORKUP = "workup"
    DECISION = "decision"
    PROCEDURE = "procedure"
    POSTOP = "postop"
    FOLLOWUP = "followup"


class ActionType(str, Enum):
    """Types of clinical decision actions."""

    ORDER_TEST = "ORDER_TEST"
    CHOOSE_MANAGEMENT = "CHOOSE_MANAGEMENT"
    DISCLOSE_ALTERNATIVES = "DISCLOSE_ALTERNATIVES"
    COUNSEL_RISKS = "COUNSEL_RISKS"
    SELECT_TECHNIQUE = "SELECT_TECHNIQUE"
    ESCALATE_CARE = "ESCALATE_CARE"
    DOCUMENT_CONSENT = "DOCUMENT_CONSENT"
    REFER = "REFER"
    PRESCRIBE = "PRESCRIBE"
    DISCHARGE_DECISION = "DISCHARGE_DECISION"


class Verdict(str, Enum):
    """Legal verdict outcomes."""

    LIABILITY_FOUND = "LIABILITY_FOUND"
    NO_LIABILITY = "NO_LIABILITY"
    PARTIAL_LIABILITY = "PARTIAL_LIABILITY"
    SETTLED = "SETTLED"
    UNKNOWN = "UNKNOWN"


class MalpracticeCategory(str, Enum):
    """Major malpractice categories."""

    DIAGNOSIS_ERROR = "DIAGNOSIS_ERROR"
    TREATMENT_PROCEDURE_ERROR = "TREATMENT_PROCEDURE_ERROR"
    INFORMED_CONSENT = "INFORMED_CONSENT"
    MONITORING_FOLLOWUP = "MONITORING_FOLLOWUP"
    MEDICATION_ERROR = "MEDICATION_ERROR"
    SYSTEM_COMMUNICATION = "SYSTEM_COMMUNICATION"
    DOCUMENTATION = "DOCUMENTATION"


class ConsentSubtype(str, Enum):
    """Consent-related malpractice subtypes."""

    ALTERNATIVES_NOT_DISCLOSED = "ALTERNATIVES_NOT_DISCLOSED"
    MATERIAL_RISK_NOT_DISCLOSED = "MATERIAL_RISK_NOT_DISCLOSED"
    NOVEL_TECHNIQUE_NOT_DISCLOSED = "NOVEL_TECHNIQUE_NOT_DISCLOSED"
    PURPOSE_OR_BENEFIT_MISSTATED = "PURPOSE_OR_BENEFIT_MISSTATED"
    INADEQUATE_TIME_TO_DECIDE = "INADEQUATE_TIME_TO_DECIDE"
    CAPACITY_NOT_ASSESSED = "CAPACITY_NOT_ASSESSED"


class DiagnosisSubtype(str, Enum):
    """Diagnosis-related malpractice subtypes."""

    MISSED_DIAGNOSIS = "MISSED_DIAGNOSIS"
    DELAYED_DIAGNOSIS = "DELAYED_DIAGNOSIS"
    WRONG_DIAGNOSIS = "WRONG_DIAGNOSIS"
    FAILURE_TO_INVESTIGATE = "FAILURE_TO_INVESTIGATE"


class ProcedureSubtype(str, Enum):
    """Procedure-related malpractice subtypes."""

    WRONG_TECHNIQUE = "WRONG_TECHNIQUE"
    WRONG_SITE = "WRONG_SITE"
    RETAINED_FOREIGN_BODY = "RETAINED_FOREIGN_BODY"
    SURGICAL_COMPLICATION = "SURGICAL_COMPLICATION"
    INADEQUATE_SUPERVISION = "INADEQUATE_SUPERVISION"


class DiscoveryStatus(str, Enum):
    """Status of case discovery."""

    QUEUED = "queued"
    FETCHING = "fetching"
    FETCHED = "fetched"
    PARSING = "parsing"
    PARSED = "parsed"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"


class RejectionReason(str, Enum):
    """Reasons for rejecting a discovered case."""

    NOT_MEDICAL = "NOT_MEDICAL"
    TOO_SHORT = "TOO_SHORT"
    PROCEDURAL_ONLY = "PROCEDURAL_ONLY"
    NO_CLINICAL_FACTS = "NO_CLINICAL_FACTS"
    PAYWALLED = "PAYWALLED"
    DUPLICATE = "DUPLICATE"
    FETCH_ERROR = "FETCH_ERROR"
    PARSE_ERROR = "PARSE_ERROR"


class Sex(str, Enum):
    """Patient sex."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"
