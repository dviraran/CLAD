"""Text segmentation for LLM processing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

import tiktoken

from ..config import get_settings
from ..parsing.parser import Paragraph, ParsedJudgment
from ..utils import get_logger


@dataclass
class TextSegment:
    """A segment of text for processing."""

    segment_id: int
    text: str
    token_count: int
    start_para: str | None = None
    end_para: str | None = None
    section: str | None = None
    overlap_with_previous: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class SegmentedDocument:
    """A document split into segments."""

    segments: list[TextSegment]
    total_tokens: int
    total_paragraphs: int
    document_metadata: dict = field(default_factory=dict)


class DocumentSegmenter:
    """Segments documents for LLM processing."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        model: str | None = None,
    ):
        """Initialize the segmenter."""
        self.settings = get_settings()
        self.logger = get_logger("segmentation")

        self.chunk_size = chunk_size or self.settings.extraction.chunk_size
        self.chunk_overlap = chunk_overlap or self.settings.extraction.chunk_overlap

        # Get tokenizer for the model
        model_name = model or self.settings.openai.model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def segment_parsed(self, parsed: ParsedJudgment) -> SegmentedDocument:
        """Segment a parsed judgment document."""
        # Build text blocks with paragraph references
        blocks: list[tuple[str, Paragraph]] = []

        for para in parsed.paragraphs:
            if para.text.strip():
                # Include paragraph number for reference
                if para.number:
                    text = f"[{para.number}] {para.text}"
                else:
                    text = para.text
                blocks.append((text, para))

        # Calculate total tokens
        total_tokens = sum(self.count_tokens(text) for text, _ in blocks)

        # If small enough, return as single segment
        if total_tokens <= self.chunk_size:
            full_text = "\n\n".join(text for text, _ in blocks)
            segment = TextSegment(
                segment_id=0,
                text=full_text,
                token_count=total_tokens,
                start_para=parsed.paragraphs[0].number if parsed.paragraphs else None,
                end_para=parsed.paragraphs[-1].number if parsed.paragraphs else None,
            )
            return SegmentedDocument(
                segments=[segment],
                total_tokens=total_tokens,
                total_paragraphs=len(parsed.paragraphs),
                document_metadata={
                    "title": parsed.title,
                    "citation": parsed.citation,
                    "court": parsed.court,
                },
            )

        # Split into segments
        segments = list(self._create_segments(blocks))

        return SegmentedDocument(
            segments=segments,
            total_tokens=total_tokens,
            total_paragraphs=len(parsed.paragraphs),
            document_metadata={
                "title": parsed.title,
                "citation": parsed.citation,
                "court": parsed.court,
            },
        )

    def segment_text(self, text: str) -> SegmentedDocument:
        """Segment raw text."""
        total_tokens = self.count_tokens(text)

        if total_tokens <= self.chunk_size:
            segment = TextSegment(
                segment_id=0,
                text=text,
                token_count=total_tokens,
            )
            return SegmentedDocument(
                segments=[segment],
                total_tokens=total_tokens,
                total_paragraphs=text.count("\n\n") + 1,
            )

        # Split by paragraphs
        paragraphs = text.split("\n\n")
        blocks = [(p.strip(), None) for p in paragraphs if p.strip()]

        segments = list(self._create_segments(blocks))

        return SegmentedDocument(
            segments=segments,
            total_tokens=total_tokens,
            total_paragraphs=len(paragraphs),
        )

    def _create_segments(
        self,
        blocks: list[tuple[str, Paragraph | None]],
    ) -> Iterator[TextSegment]:
        """Create segments from text blocks."""
        current_texts: list[str] = []
        current_tokens = 0
        current_start_para: str | None = None
        segment_id = 0

        # Track overlap
        overlap_blocks: list[str] = []

        for text, para in blocks:
            text_tokens = self.count_tokens(text)

            # Start new segment if this would exceed chunk size
            if current_tokens + text_tokens > self.chunk_size and current_texts:
                # Yield current segment
                yield TextSegment(
                    segment_id=segment_id,
                    text="\n\n".join(current_texts),
                    token_count=current_tokens,
                    start_para=current_start_para,
                    end_para=para.number if para else None,
                    section=para.section if para else None,
                    overlap_with_previous=segment_id > 0,
                )
                segment_id += 1

                # Calculate overlap
                overlap_blocks = []
                overlap_tokens = 0
                for t in reversed(current_texts):
                    t_tokens = self.count_tokens(t)
                    if overlap_tokens + t_tokens <= self.chunk_overlap:
                        overlap_blocks.insert(0, t)
                        overlap_tokens += t_tokens
                    else:
                        break

                # Start new segment with overlap
                current_texts = overlap_blocks.copy()
                current_tokens = overlap_tokens
                current_start_para = para.number if para else None

            # Add to current segment
            current_texts.append(text)
            current_tokens += text_tokens

            if current_start_para is None and para:
                current_start_para = para.number

        # Yield final segment
        if current_texts:
            yield TextSegment(
                segment_id=segment_id,
                text="\n\n".join(current_texts),
                token_count=current_tokens,
                start_para=current_start_para,
                overlap_with_previous=segment_id > 0,
            )

    # Keywords that indicate important legal/medical content
    PRIORITY_KEYWORDS = [
        "breach", "breached", "liable", "liability", "negligent", "negligence",
        "causation", "caused", "damages", "dismiss", "dismissed", "judgment",
        "finding", "finds", "found", "conclude", "conclusion", "verdict",
        "claimant succeeds", "claim fails", "standard of care",
        "malpractice", "duty of care", "expert evidence", "expert testimony",
    ]

    def create_evidence_context(
        self,
        parsed: ParsedJudgment,
        max_tokens: int | None = None,
    ) -> str:
        """Create a context string with SMART selection prioritizing verdict and key findings.

        Priority order:
        1. Header (case name, citation, court, date)
        2. Last 20 paragraphs (verdict, findings, conclusion)
        3. Paragraphs containing key legal terms
        4. First 15 paragraphs (background, presentation)
        5. Fill remaining with middle sections
        """
        max_tokens = max_tokens or self.chunk_size

        # Reserve tokens for each priority tier
        HEADER_BUDGET = 300
        CONCLUSION_BUDGET = int(max_tokens * 0.30)  # 30% for conclusions
        PRIORITY_BUDGET = int(max_tokens * 0.25)    # 25% for key term paragraphs
        INTRO_BUDGET = int(max_tokens * 0.25)       # 25% for intro
        MIDDLE_BUDGET = int(max_tokens * 0.15)      # 15% for middle fill

        lines: list[str] = []
        current_tokens = 0
        used_para_indices: set[int] = set()

        # Priority 1: Header
        header_parts = []
        if parsed.title:
            header_parts.append(f"Case: {parsed.title}")
        if parsed.parties and parsed.parties != parsed.title:
            header_parts.append(f"Parties: {parsed.parties}")
        if parsed.citation:
            header_parts.append(f"Citation: {parsed.citation}")
        if parsed.court:
            header_parts.append(f"Court: {parsed.court}")
        if parsed.date:
            header_parts.append(f"Date: {parsed.date}")

        if header_parts:
            header = "\n".join(header_parts)
            header_tokens = self.count_tokens(header)
            lines.append(header)
            lines.append("\n--- JUDGMENT TEXT ---\n")
            current_tokens += header_tokens + 10

        # Build paragraph list with indices
        all_paras: list[tuple[int, str, Paragraph]] = []
        for i, para in enumerate(parsed.paragraphs):
            if para.text.strip():
                para_text = f"[{para.number or i+1}] {para.text}"
                all_paras.append((i, para_text, para))

        if not all_paras:
            return "\n\n".join(lines)

        # Priority 2: Last 20 paragraphs (conclusions/verdict)
        conclusion_lines: list[str] = []
        conclusion_tokens = 0
        last_paras = all_paras[-20:] if len(all_paras) >= 20 else all_paras[-len(all_paras):]

        for idx, para_text, para in reversed(last_paras):
            para_tokens = self.count_tokens(para_text)
            if conclusion_tokens + para_tokens <= CONCLUSION_BUDGET:
                conclusion_lines.insert(0, para_text)
                conclusion_tokens += para_tokens
                used_para_indices.add(idx)

        # Priority 3: Paragraphs with key legal terms
        priority_lines: list[str] = []
        priority_tokens = 0

        for idx, para_text, para in all_paras:
            if idx in used_para_indices:
                continue
            para_lower = para_text.lower()
            has_priority_term = any(kw in para_lower for kw in self.PRIORITY_KEYWORDS)
            if has_priority_term:
                para_tokens = self.count_tokens(para_text)
                if priority_tokens + para_tokens <= PRIORITY_BUDGET:
                    priority_lines.append(para_text)
                    priority_tokens += para_tokens
                    used_para_indices.add(idx)

        # Priority 4: First 15 paragraphs (introduction/background)
        intro_lines: list[str] = []
        intro_tokens = 0
        first_paras = all_paras[:15] if len(all_paras) >= 15 else all_paras

        for idx, para_text, para in first_paras:
            if idx in used_para_indices:
                continue
            para_tokens = self.count_tokens(para_text)
            if intro_tokens + para_tokens <= INTRO_BUDGET:
                intro_lines.append(para_text)
                intro_tokens += para_tokens
                used_para_indices.add(idx)

        # Priority 5: Fill with middle paragraphs
        middle_lines: list[str] = []
        middle_tokens = 0

        for idx, para_text, para in all_paras:
            if idx in used_para_indices:
                continue
            para_tokens = self.count_tokens(para_text)
            if middle_tokens + para_tokens <= MIDDLE_BUDGET:
                middle_lines.append(para_text)
                middle_tokens += para_tokens
                used_para_indices.add(idx)

        # Assemble in logical order: intro -> priority -> middle -> conclusion
        lines.append("=== BACKGROUND/INTRODUCTION ===")
        lines.extend(intro_lines)

        if priority_lines:
            lines.append("\n=== KEY FINDINGS AND ANALYSIS ===")
            lines.extend(priority_lines)

        if middle_lines:
            lines.append("\n=== ADDITIONAL CONTEXT ===")
            lines.extend(middle_lines)

        lines.append("\n=== CONCLUSION AND VERDICT ===")
        lines.extend(conclusion_lines)

        # Log what we included
        total_paras = len(all_paras)
        included_paras = len(used_para_indices)
        self.logger.info(
            f"Smart context: included {included_paras}/{total_paras} paragraphs "
            f"(intro={len(intro_lines)}, priority={len(priority_lines)}, "
            f"middle={len(middle_lines)}, conclusion={len(conclusion_lines)})"
        )

        return "\n\n".join(lines)

    def segment_for_multi_pass(
        self,
        parsed: ParsedJudgment,
        priority_sections: list[str] | None = None,
    ) -> list[TextSegment]:
        """Create segments optimized for multi-pass extraction."""
        priority_sections = priority_sections or [
            "facts",
            "background",
            "clinical",
            "evidence",
            "consent",
            "breach",
            "causation",
            "conclusion",
        ]

        segments: list[TextSegment] = []
        segment_id = 0

        # First pass: priority sections
        priority_paras: list[tuple[str, Paragraph]] = []
        other_paras: list[tuple[str, Paragraph]] = []

        for para in parsed.paragraphs:
            if not para.text.strip():
                continue

            text = f"[{para.number or '?'}] {para.text}"

            is_priority = False
            if para.section:
                section_lower = para.section.lower()
                for priority in priority_sections:
                    if priority in section_lower:
                        is_priority = True
                        break

            if is_priority:
                priority_paras.append((text, para))
            else:
                other_paras.append((text, para))

        # Create priority segments
        if priority_paras:
            for seg in self._create_segments(priority_paras):
                seg.segment_id = segment_id
                seg.metadata["priority"] = True
                segments.append(seg)
                segment_id += 1

        # Create other segments
        if other_paras:
            for seg in self._create_segments(other_paras):
                seg.segment_id = segment_id
                seg.metadata["priority"] = False
                segments.append(seg)
                segment_id += 1

        return segments
