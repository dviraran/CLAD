"""Tests for parsing module."""

import pytest

from casesim.parsing import JudgmentParser, ParsedJudgment


class TestJudgmentParser:
    """Tests for JudgmentParser."""

    @pytest.fixture
    def parser(self):
        return JudgmentParser()

    @pytest.fixture
    def sample_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Smith v NHS Trust [2019] EWHC 123 (QB)</title></head>
        <body>
            <h1>Smith v NHS Trust [2019] EWHC 123 (QB)</h1>
            <p>Before: Mr Justice Williams</p>
            <p>Judgment handed down on 15 January 2019</p>

            <h2>The Facts</h2>
            <p>[1] The claimant presented to the emergency department on 1 March 2018.</p>
            <p>[2] She complained of severe abdominal pain lasting 6 hours.</p>

            <h2>Clinical Background</h2>
            <p>[3] The claimant had a history of hypertension.</p>
            <p>[4] She was taking metformin for diabetes.</p>

            <h2>Expert Evidence</h2>
            <p>[5] Professor Smith gave evidence that the standard of care required
            an urgent CT scan within 2 hours of presentation.</p>

            <h2>Breach of Duty</h2>
            <p>[6] I find that the defendant failed to meet the required standard.</p>

            <h2>Conclusion</h2>
            <p>[7] For these reasons, I find the defendant liable.</p>
        </body>
        </html>
        """

    def test_parse_html_extracts_title(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        assert "Smith v NHS Trust" in result.title

    def test_parse_html_extracts_citation(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        assert "[2019] EWHC 123" in (result.citation or "")

    def test_parse_html_extracts_paragraphs(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        assert len(result.paragraphs) > 0

    def test_parse_html_identifies_sections(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        section_names = [s.title.lower() for s in result.sections]
        assert any("facts" in name for name in section_names)

    def test_parse_html_extracts_paragraph_numbers(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        numbered = [p for p in result.paragraphs if p.number and p.number.isdigit()]
        assert len(numbered) > 0

    def test_parse_text(self, parser):
        text = """
        Smith v Jones [2020] EWHC 456 (QB)

        The Facts

        1. The patient presented on 1 January 2020.
        2. He complained of chest pain.

        Conclusion

        3. I find the defendant liable.
        """
        result = parser.parse_text(text)
        assert result.parties is not None
        assert len(result.paragraphs) > 0

    def test_get_section_text(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        facts_text = parser.get_section_text(result, "facts")
        assert "claimant" in facts_text.lower() or "presented" in facts_text.lower()

    def test_identifies_headings(self, parser, sample_html):
        result = parser.parse_html(sample_html)
        headings = [p for p in result.paragraphs if p.is_heading]
        assert len(headings) > 0


class TestParsedJudgment:
    """Tests for ParsedJudgment dataclass."""

    def test_empty_judgment(self):
        parsed = ParsedJudgment()
        assert parsed.title is None
        assert len(parsed.paragraphs) == 0

    def test_with_data(self):
        parsed = ParsedJudgment(
            title="Test Case",
            citation="[2020] EWHC 123",
            court="EWHC QB",
        )
        assert parsed.title == "Test Case"
        assert parsed.court == "EWHC QB"
