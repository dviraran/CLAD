"""
Utility functions for the CaseSim Training GUI.

Provides sanitization, intent classification, and prompt generation.
"""

from __future__ import annotations

import re
from typing import Any

# Terms that should be filtered from prompts and patient responses
# to avoid leading the clinician toward the answer
FORBIDDEN_TERMS = [
    # Legal terms
    "montgomery",
    "consent failure",
    "informed consent",
    "breach of duty",
    "negligence",
    "negligent",
    "malpractice",
    "liability",
    "standard of care",
    "reasonable alternative",
    "material risk",
    "bolam",
    "bolitho",
    "duty to warn",
    "duty of care",
    # Judgment language
    "defendant",
    "claimant",
    "plaintiff",
    "court",
    "judge",
    "verdict",
    "damages",
    "finding",
    "ruling",
    "allegation",
    "judgment",
    "judgement",
    # Outcome hints
    "failure to",
    "should have",
    "failed to",
    "ought to",
    "wrongly",
    "incorrectly",
]

# Question starter words for intent classification
QUESTION_STARTERS = [
    "what",
    "where",
    "when",
    "who",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "tell me",
    "describe",
    "explain",
]


def sanitize_text(text: str) -> str:
    """
    Remove forbidden terms from text to avoid leading prompts.

    Args:
        text: The text to sanitize

    Returns:
        Text with forbidden terms replaced with neutral placeholders
    """
    if not text:
        return text

    sanitized = text
    for term in FORBIDDEN_TERMS:
        # Use word boundaries to avoid partial matches
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        sanitized = pattern.sub("", sanitized)

    # Clean up multiple spaces and leading/trailing whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    return sanitized


def is_clean(text: str) -> bool:
    """
    Check if text contains no forbidden terms.

    Args:
        text: The text to check

    Returns:
        True if text is clean, False if it contains forbidden terms
    """
    if not text:
        return True

    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in text_lower:
            return False
    return True


def classify_intent(message: str) -> str:
    """
    Classify a message as 'question', 'order_test', 'recommendation', or 'response'.

    Uses rule-based classification:
    - Contains test ordering language -> order_test
    - Contains diagnostic/treatment recommendations -> recommendation
    - Ends with '?' -> question
    - Starts with question words -> question
    - Short messages (<5 words) -> question
    - Otherwise -> response

    Args:
        message: The user's input message

    Returns:
        'question', 'order_test', 'recommendation', or 'response'
    """
    if not message:
        return "response"

    message_clean = message.strip()
    message_lower = message_clean.lower()

    # Test ordering keywords - detect when doctor tries to order tests
    test_order_patterns = [
        "order", "request", "arrange", "get a", "get an", "send for",
        "need a", "need an", "like a", "like an", "perform a", "do a",
        "schedule a", "schedule an", "run a", "run some",
    ]
    test_types = [
        "ct", "mri", "x-ray", "xray", "ultrasound", "scan", "blood test",
        "blood work", "labs", "ecg", "ekg", "echo", "angiogram", "biopsy",
        "culture", "urinalysis", "cbc", "chemistry", "imaging",
    ]

    # Check for test ordering intent
    has_order_verb = any(pattern in message_lower for pattern in test_order_patterns)
    has_test_type = any(test in message_lower for test in test_types)
    if has_order_verb and has_test_type:
        return "order_test"

    # Also catch direct test mentions that imply ordering
    direct_order_phrases = [
        "let's get", "let's do", "i'll order", "i would order", "we should order",
        "i recommend ordering", "please arrange", "i want to order",
    ]
    if any(phrase in message_lower for phrase in direct_order_phrases):
        return "order_test"

    # Recommendation keywords - detect when doctor gives diagnosis/treatment advice
    recommendation_patterns = [
        "i think this is", "i believe this is", "my diagnosis is",
        "the diagnosis is", "i would diagnose", "this appears to be",
        "this looks like", "i suspect", "i recommend", "i advise",
        "you should", "we should", "i would suggest", "my recommendation is",
        "the treatment should be", "i would prescribe", "i would start",
        "based on my assessment", "in my opinion", "my impression is",
        "i'm concerned about", "this is likely", "probably", "most likely",
    ]
    if any(pattern in message_lower for pattern in recommendation_patterns):
        return "recommendation"

    # Check for question mark
    if message_clean.endswith("?"):
        return "question"

    # Check for question starters
    for starter in QUESTION_STARTERS:
        if message_lower.startswith(starter + " ") or message_lower == starter:
            return "question"

    # Short messages are more likely questions
    word_count = len(message_clean.split())
    if word_count < 5:
        return "question"

    # Default to response for longer statements
    return "response"


def safe_get(obj: dict | Any, path: str, default: Any = None) -> Any:
    """
    Safely navigate nested dict with dot notation path.

    Args:
        obj: The dictionary to navigate
        path: Dot-separated path (e.g., "simulation.initial_state.chief_complaint")
        default: Default value if path not found

    Returns:
        The value at the path, or default if not found
    """
    if not isinstance(obj, dict):
        return default

    keys = path.split(".")
    current = obj

    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default

    return current if current is not None else default


def is_placeholder(value: str | None) -> bool:
    """
    Check if a value is a placeholder indicating missing data.

    Args:
        value: The value to check

    Returns:
        True if the value is a placeholder or missing
    """
    if not value:
        return True

    value_lower = str(value).lower()
    placeholders = [
        "not documented",
        "not available",
        "not specified",
        "not recorded",
        "not mentioned",
        "not stated",
        "unknown",
        "n/a",
        "none",
        "null",
        "findings not specified",
    ]

    return any(p in value_lower for p in placeholders)


def generate_clinical_prompt(initial_state: dict) -> str:
    """
    Generate a neutral clinical prompt from the initial state.

    The prompt should:
    - Present clinical context neutrally
    - Be 5-8 lines maximum
    - End with an open question

    Args:
        initial_state: The simulation.initial_state from the case

    Returns:
        A sanitized clinical prompt string
    """
    if not initial_state:
        return "A patient presents for clinical evaluation.\n\nAs the clinician, what is your approach to this case?"

    parts = []

    # Demographics
    demo = initial_state.get("patient_demographics", {})
    age = demo.get("age_at_presentation", "")
    sex = demo.get("sex", "")

    # Build demographic string
    if not is_placeholder(age) and not is_placeholder(sex):
        parts.append(f"A {age} {sex} patient")
    elif not is_placeholder(sex) and sex != "unknown":
        parts.append(f"A {sex} patient")
    else:
        parts.append("A patient")

    # Chief complaint
    chief_complaint = initial_state.get("chief_complaint", "")
    if not is_placeholder(chief_complaint):
        cc_clean = sanitize_text(chief_complaint)
        if cc_clean:
            parts.append(f"presents with {cc_clean.lower()}")
        else:
            parts.append("presents for evaluation")
    else:
        parts.append("presents for evaluation")

    # Join first sentence
    prompt = " ".join(parts) + "."

    # History of present illness
    hpi = initial_state.get("history_of_present_illness", "")
    if not is_placeholder(hpi):
        hpi_clean = sanitize_text(hpi)
        if hpi_clean:
            prompt += f" {hpi_clean}"

    # Past medical history
    pmh = initial_state.get("past_medical_history", [])
    if pmh and isinstance(pmh, list):
        pmh_items = [sanitize_text(p) for p in pmh if p and not is_placeholder(p)]
        if pmh_items:
            prompt += f"\n\nRelevant medical history: {', '.join(pmh_items)}."

    # Medications
    meds = initial_state.get("medications", [])
    if meds and isinstance(meds, list):
        med_items = [sanitize_text(m) for m in meds if m and not is_placeholder(m)]
        if med_items:
            prompt += f"\n\nCurrent medications: {', '.join(med_items)}."

    # Add neutral closing question
    prompt += "\n\nAs the treating clinician, how would you approach this consultation?"

    # Final safety check
    if not is_clean(prompt):
        # Fallback to minimal prompt if contaminated
        return "A patient presents for clinical evaluation.\n\nAs the treating clinician, how would you approach this consultation?"

    return prompt


def format_verdict(verdict: str | None) -> str:
    """
    Format a legal verdict for display.

    Args:
        verdict: The verdict string (e.g., "LIABILITY_FOUND")

    Returns:
        Human-readable verdict string
    """
    if not verdict:
        return "Unknown"

    verdict_map = {
        "LIABILITY_FOUND": "Liability Found",
        "NO_LIABILITY": "No Liability",
        "PARTIAL_LIABILITY": "Partial Liability",
        "SETTLED": "Settled",
        "UNKNOWN": "Unknown",
    }

    return verdict_map.get(verdict, verdict.replace("_", " ").title())


def format_severity(severity: str | None) -> str:
    """
    Format outcome severity for display.

    Args:
        severity: The severity string (e.g., "PERMANENT_SEVERE_DISABILITY")

    Returns:
        Human-readable severity string
    """
    if not severity:
        return "Unknown"

    severity_map = {
        "DEATH": "Death",
        "PERMANENT_SEVERE_DISABILITY": "Permanent Severe Disability",
        "PERMANENT_MODERATE_DISABILITY": "Permanent Moderate Disability",
        "TEMPORARY_HARM": "Temporary Harm",
        "MINOR_HARM": "Minor Harm",
        "NO_PHYSICAL_HARM": "No Physical Harm",
    }

    return severity_map.get(severity, severity.replace("_", " ").title())


def format_clinical_domain(domain: str | None) -> str:
    """
    Format clinical domain for display.

    Args:
        domain: The domain string (e.g., "SURGERY_GENERAL")

    Returns:
        Human-readable domain string
    """
    if not domain:
        return "Unknown"

    return domain.replace("_", " ").title()
