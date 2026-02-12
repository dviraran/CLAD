"""HTML/text parser for legal judgment documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from ..utils import get_logger


@dataclass
class Paragraph:
    """A paragraph from the judgment."""

    number: str | None
    text: str
    section: str | None = None
    is_heading: bool = False
    html_tag: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """A section of the judgment."""

    title: str
    level: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)


@dataclass
class ParsedJudgment:
    """Parsed structure of a judgment document."""

    title: str | None = None
    citation: str | None = None
    court: str | None = None
    judge: str | None = None
    date: str | None = None
    parties: str | None = None
    paragraphs: list[Paragraph] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class JudgmentParser:
    """Parser for legal judgment documents."""

    # Patterns for identifying sections
    SECTION_PATTERNS = [
        r"^(?:the\s+)?facts?\b",
        r"^(?:the\s+)?background\b",
        r"^clinical\s+(?:background|history)\b",
        r"^medical\s+(?:history|evidence)\b",
        r"^(?:the\s+)?evidence\b",
        r"^expert\s+(?:evidence|testimony|witnesses)\b",
        r"^(?:the\s+)?(?:claimant|plaintiff)['\u2019]?s?\s+case\b",
        r"^(?:the\s+)?defendant['\u2019]?s?\s+case\b",
        r"^(?:the\s+)?issues?\b",
        r"^consent\b",
        r"^informed\s+consent\b",
        r"^breach\s+of\s+duty\b",
        r"^causation\b",
        r"^damages?\b",
        r"^(?:the\s+)?law\b",
        r"^discussion\b",
        r"^analysis\b",
        r"^conclusion\b",
        r"^judgment\b",
        r"^findings?\b",
        r"^summary\b",
    ]

    # Patterns for paragraph numbers
    PARA_NUMBER_PATTERNS = [
        r"^\[?(\d+)\]?\.?\s*",  # [1] or 1. or [1].
        r"^(\d+)\)\s*",  # 1)
        r"^para(?:graph)?\s*(\d+)",  # Para 1 or Paragraph 1
    ]

    def __init__(self):
        self.logger = get_logger("parsing.parser")

    def parse_html(self, html: str) -> ParsedJudgment:
        """Parse an HTML judgment document."""
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        result = ParsedJudgment()

        # Store raw text BEFORE extracting paragraphs (which may decompose elements)
        result.raw_text = soup.get_text(separator="\n", strip=True)

        # Extract metadata
        self._extract_metadata(soup, result)

        # Extract paragraphs
        self._extract_paragraphs(soup, result)

        # Build sections
        self._build_sections(result)

        return result

    def parse_text(self, text: str) -> ParsedJudgment:
        """Parse a plain text judgment document."""
        result = ParsedJudgment()

        # Try to extract title from first lines
        lines = text.split("\n")
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if not line:
                continue

            if " v " in line or " v. " in line:
                result.title = line
                result.parties = line
                break

        # Extract paragraphs
        current_para = []
        current_number = None
        para_count = 0

        for line in lines:
            line = line.strip()

            if not line:
                if current_para:
                    text = " ".join(current_para)
                    result.paragraphs.append(
                        Paragraph(
                            number=current_number or str(para_count + 1),
                            text=text,
                            is_heading=self._is_heading_text(text),
                        )
                    )
                    para_count += 1
                    current_para = []
                    current_number = None
                continue

            # Check for paragraph number
            for pattern in self.PARA_NUMBER_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    current_number = match.group(1)
                    line = re.sub(pattern, "", line, flags=re.IGNORECASE).strip()
                    break

            if line:
                current_para.append(line)

        # Handle last paragraph
        if current_para:
            text = " ".join(current_para)
            result.paragraphs.append(
                Paragraph(
                    number=current_number or str(para_count + 1),
                    text=text,
                    is_heading=self._is_heading_text(text),
                )
            )

        # Build sections
        self._build_sections(result)

        result.raw_text = text
        return result

    def _extract_metadata(self, soup: BeautifulSoup, result: ParsedJudgment) -> None:
        """Extract metadata from HTML."""
        # Try title tag
        title_tag = soup.find("title")
        if title_tag:
            result.title = title_tag.get_text(strip=True)

        # Look for case name pattern
        for elem in soup.find_all(["h1", "h2", "p", "div"]):
            text = elem.get_text(strip=True)
            if " v " in text or " v. " in text:
                result.parties = text
                if not result.title:
                    result.title = text
                break

        # Look for neutral citation
        citation_patterns = [
            r"\[\d{4}\]\s+(?:EWHC|EWCA|UKSC|UKHL)\s+\d+",
            r"\d{4}\s+(?:ONSC|ONCA|BCSC|SCC)\s+\d+",
            r"\[\d{4}\]\s+(?:NSWSC|NSWCA|VSC|HCA)\s+\d+",
        ]

        for pattern in citation_patterns:
            match = re.search(pattern, soup.get_text())
            if match:
                result.citation = match.group(0)
                break

        # Look for judge name
        judge_patterns = [
            r"(?:before\s+)?(?:the\s+honourable\s+)?(?:mr|mrs|ms|lady|lord)\s+justice\s+\w+",
            r"(?:his|her)\s+honour\s+judge\s+\w+",
            r"deputy\s+(?:high\s+court\s+)?judge\s+\w+",
        ]

        for pattern in judge_patterns:
            match = re.search(pattern, soup.get_text()[:2000], re.IGNORECASE)
            if match:
                result.judge = match.group(0).strip()
                break

        # Look for date
        date_patterns = [
            r"(?:judgment\s+)?(?:handed\s+down\s+)?(?:on\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            r"(\w+\s+\d{1,2},?\s+\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, soup.get_text()[:3000], re.IGNORECASE)
            if match:
                result.date = match.group(1).strip()
                break

    def _extract_paragraphs(self, soup: BeautifulSoup, result: ParsedJudgment) -> None:
        """Extract paragraphs from HTML."""
        # Find the main content area
        content = soup.find("body") or soup

        # Remove BAILII-specific navigation
        for elem in content.find_all(class_=["nav", "navigation", "menu", "sidebar"]):
            elem.decompose()

        para_count = 0

        # Handle <pre> tags specially - split on double newlines
        pre_tags = content.find_all("pre")
        for pre_elem in pre_tags:
            pre_text = pre_elem.get_text()
            # Split on double newlines or patterns like paragraph markers
            chunks = re.split(r'\n\s*\n|\n(?=\{\d+\}|\[\d+\]|\d+\.)', pre_text)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk or len(chunk) < 10:
                    continue

                # Extract paragraph number
                para_number = None
                for pattern in self.PARA_NUMBER_PATTERNS:
                    match = re.match(pattern, chunk)
                    if match:
                        para_number = match.group(1)
                        chunk = re.sub(pattern, "", chunk).strip()
                        break

                # Also check for {1} style markers used in CourtListener
                curly_match = re.match(r'^\{(\d+)\}\s*', chunk)
                if curly_match:
                    para_number = curly_match.group(1)
                    chunk = re.sub(r'^\{\d+\}\s*', '', chunk).strip()

                if chunk:
                    para_count += 1
                    result.paragraphs.append(
                        Paragraph(
                            number=para_number or str(para_count),
                            text=chunk,
                            is_heading=self._is_heading_text(chunk),
                            html_tag="pre",
                        )
                    )
            # Remove the pre tag so we don't double-process
            pre_elem.decompose()

        # Process block elements
        for elem in content.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            text = elem.get_text(separator=" ", strip=True)

            if not text or len(text) < 3:
                continue

            # Skip navigation elements
            if elem.get("class") and any(
                c in str(elem.get("class")).lower()
                for c in ["nav", "menu", "footer", "header"]
            ):
                continue

            # Extract paragraph number
            para_number = None
            for pattern in self.PARA_NUMBER_PATTERNS:
                match = re.match(pattern, text)
                if match:
                    para_number = match.group(1)
                    text = re.sub(pattern, "", text).strip()
                    break

            # Check if it's a heading
            is_heading = elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]
            if not is_heading:
                is_heading = self._is_heading_text(text)

            para_count += 1
            result.paragraphs.append(
                Paragraph(
                    number=para_number or str(para_count),
                    text=text,
                    is_heading=is_heading,
                    html_tag=elem.name,
                )
            )

    def _is_heading_text(self, text: str) -> bool:
        """Determine if text is likely a heading."""
        if len(text) > 200:
            return False

        text_lower = text.lower().strip()

        # Check against section patterns
        for pattern in self.SECTION_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True

        # Short text in all caps or title case might be heading
        if len(text) < 100:
            if text.isupper() and len(text.split()) <= 10:
                return True
            if text.istitle() and len(text.split()) <= 6:
                return True

        return False

    def _build_sections(self, result: ParsedJudgment) -> None:
        """Build section structure from paragraphs."""
        current_section: Section | None = None

        for para in result.paragraphs:
            if para.is_heading:
                # Start new section
                level = 1
                if current_section is not None:
                    level = current_section.level

                # Determine level from heading type
                if para.html_tag:
                    level_map = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
                    level = level_map.get(para.html_tag, 2)

                section = Section(title=para.text, level=level)

                # Determine nesting
                if current_section is None or level <= current_section.level:
                    result.sections.append(section)
                else:
                    current_section.subsections.append(section)

                current_section = section
            else:
                # Add to current section
                if current_section is not None:
                    current_section.paragraphs.append(para)
                    para.section = current_section.title

    def get_section_text(
        self,
        result: ParsedJudgment,
        section_name: str,
        include_subsections: bool = True,
    ) -> str:
        """Get all text from a specific section."""
        section_lower = section_name.lower()

        for section in result.sections:
            if section_lower in section.title.lower():
                texts = [p.text for p in section.paragraphs]

                if include_subsections:
                    for sub in section.subsections:
                        texts.extend(p.text for p in sub.paragraphs)

                return "\n\n".join(texts)

        return ""

    def get_paragraphs_by_section(
        self,
        result: ParsedJudgment,
        section_names: list[str],
    ) -> list[Paragraph]:
        """Get paragraphs from multiple sections."""
        section_names_lower = [s.lower() for s in section_names]
        paragraphs: list[Paragraph] = []

        for para in result.paragraphs:
            if para.section:
                for section_name in section_names_lower:
                    if section_name in para.section.lower():
                        paragraphs.append(para)
                        break

        return paragraphs
