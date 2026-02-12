"""
LLM-based procedure extraction from clinical recommendations.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

from cost.models import ExtractedProcedure, ProcedureType

# Lazy import for OpenAI
_openai_available = False
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    pass


EXTRACTION_PROMPT = '''You are a medical billing specialist. Extract all medical procedures, tests, imaging studies, lab work, and specialist referrals from this clinical recommendation.

Output a JSON array with each procedure. For each item include:
- "procedure": The procedure name normalized to standard medical terminology
- "type": One of "imaging", "lab", "procedure", "consultation"

Types:
- "imaging": X-rays, CT, MRI, ultrasound, echocardiogram, etc.
- "lab": Blood tests, urinalysis, cultures, biopsies, etc.
- "procedure": Surgeries, endoscopies, injections, etc.
- "consultation": Specialist referrals (e.g., "refer to cardiologist" → "cardiology consultation")

Include:
- Tests and imaging (CT, MRI, X-ray, blood tests, urinalysis, etc.)
- Procedures (biopsy, endoscopy, surgery, etc.)
- Specialist referrals ("refer to cardiologist" → "cardiology consultation")

Do NOT include:
- General advice ("follow up", "monitor symptoms", "return if worse")
- Medications (unless it's an infusion/injection procedure)
- Diagnoses or assessments
- Hospital admission (unless it's for a specific procedure)

If no specific procedures are mentioned, return an empty array: []

Clinical Recommendation:
{recommendation}

Output only the JSON array, no other text.'''


class ProcedureExtractor:
    """
    Extract medical procedures from clinical recommendation text using LLM.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        """
        Initialize the procedure extractor.

        Args:
            model: OpenAI model to use (gpt-4o-mini recommended for speed/cost)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        """Get or create OpenAI client."""
        if not _openai_available:
            raise RuntimeError(
                "OpenAI package not available. Install with: pip install openai"
            )

        if self._client is None:
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            self._client = OpenAI(api_key=api_key)

        return self._client

    def extract(self, recommendation: str) -> list[ExtractedProcedure]:
        """
        Extract procedures from a single recommendation.

        Args:
            recommendation: The clinical recommendation text

        Returns:
            List of extracted procedures
        """
        if not recommendation or not recommendation.strip():
            return []

        client = self._get_client()

        prompt = EXTRACTION_PROMPT.format(recommendation=recommendation)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()
            return self._parse_response(content)

        except Exception as e:
            warnings.warn(f"Extraction failed: {e}")
            return []

    def _parse_response(self, content: str) -> list[ExtractedProcedure]:
        """Parse LLM response into ExtractedProcedure objects."""
        # Handle markdown code blocks
        if content.startswith("```"):
            # Remove ```json and ``` markers
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines
                if not line.startswith("```")
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse extraction response as JSON: {e}")
            return []

        if not isinstance(data, list):
            warnings.warn(f"Expected list, got {type(data)}")
            return []

        procedures = []
        for item in data:
            if not isinstance(item, dict):
                continue

            procedure_name = item.get("procedure", "").strip()
            if not procedure_name:
                continue

            # Parse procedure type
            type_str = item.get("type", "").lower()
            try:
                proc_type = ProcedureType(type_str) if type_str else None
            except ValueError:
                proc_type = ProcedureType.OTHER

            procedures.append(ExtractedProcedure(
                procedure_name=procedure_name,
                procedure_type=proc_type,
            ))

        return procedures

    def extract_batch(
        self,
        recommendations: list[str],
        show_progress: bool = True,
    ) -> list[list[ExtractedProcedure]]:
        """
        Extract procedures from multiple recommendations.

        Note: This processes sequentially. For high throughput, consider
        using async or the OpenAI batch API.

        Args:
            recommendations: List of recommendation texts
            show_progress: Show progress indicator

        Returns:
            List of procedure lists (one per recommendation)
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(recommendations, desc="Extracting procedures")
            except ImportError:
                iterator = recommendations
        else:
            iterator = recommendations

        for rec in iterator:
            procedures = self.extract(rec)
            results.append(procedures)

        return results


class MockProcedureExtractor:
    """
    Mock extractor for testing without API calls.

    Uses simple keyword matching to extract common procedures.
    """

    PROCEDURE_KEYWORDS = {
        # Imaging
        "x-ray": ("chest X-ray", ProcedureType.IMAGING),
        "xray": ("chest X-ray", ProcedureType.IMAGING),
        "ct scan": ("CT scan", ProcedureType.IMAGING),
        "ct ": ("CT scan", ProcedureType.IMAGING),
        "mri": ("MRI", ProcedureType.IMAGING),
        "ultrasound": ("ultrasound", ProcedureType.IMAGING),
        "echocardiogram": ("echocardiogram", ProcedureType.IMAGING),
        "echo": ("echocardiogram", ProcedureType.IMAGING),

        # Lab
        "blood test": ("complete blood count", ProcedureType.LAB),
        "cbc": ("complete blood count", ProcedureType.LAB),
        "complete blood count": ("complete blood count", ProcedureType.LAB),
        "urinalysis": ("urinalysis", ProcedureType.LAB),
        "urine test": ("urinalysis", ProcedureType.LAB),
        "bmp": ("basic metabolic panel", ProcedureType.LAB),
        "cmp": ("comprehensive metabolic panel", ProcedureType.LAB),
        "liver function": ("liver function test", ProcedureType.LAB),
        "lipid panel": ("lipid panel", ProcedureType.LAB),

        # Procedures
        "ecg": ("electrocardiogram", ProcedureType.PROCEDURE),
        "ekg": ("electrocardiogram", ProcedureType.PROCEDURE),
        "electrocardiogram": ("electrocardiogram", ProcedureType.PROCEDURE),
        "biopsy": ("biopsy", ProcedureType.PROCEDURE),
        "endoscopy": ("endoscopy", ProcedureType.PROCEDURE),
        "colonoscopy": ("colonoscopy", ProcedureType.PROCEDURE),
        "stress test": ("cardiac stress test", ProcedureType.PROCEDURE),

        # Consultations
        "refer to cardio": ("cardiology consultation", ProcedureType.CONSULTATION),
        "cardiologist": ("cardiology consultation", ProcedureType.CONSULTATION),
        "cardiology": ("cardiology consultation", ProcedureType.CONSULTATION),
        "refer to neuro": ("neurology consultation", ProcedureType.CONSULTATION),
        "neurologist": ("neurology consultation", ProcedureType.CONSULTATION),
        "refer to gastro": ("gastroenterology consultation", ProcedureType.CONSULTATION),
        "genetic testing": ("genetic testing", ProcedureType.LAB),
        "genetic counselor": ("genetic counseling consultation", ProcedureType.CONSULTATION),
    }

    def extract(self, recommendation: str) -> list[ExtractedProcedure]:
        """Extract procedures using keyword matching."""
        if not recommendation:
            return []

        text_lower = recommendation.lower()
        found = set()
        procedures = []

        for keyword, (name, proc_type) in self.PROCEDURE_KEYWORDS.items():
            if keyword in text_lower and name not in found:
                found.add(name)
                procedures.append(ExtractedProcedure(
                    procedure_name=name,
                    procedure_type=proc_type,
                ))

        return procedures

    def extract_batch(
        self,
        recommendations: list[str],
        show_progress: bool = True,
    ) -> list[list[ExtractedProcedure]]:
        """Extract from multiple recommendations."""
        return [self.extract(rec) for rec in recommendations]
