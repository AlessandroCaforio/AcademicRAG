"""
Academic Features — Specialized tools for thesis and research writing.

Provides six tools:
  1. CitationGenerator    — Extract and format citations from sources
  2. LiteratureReviewDrafter — Auto-draft literature review sections
  3. PaperComparator      — Compare papers across dimensions
  4. ClaimExtractor       — Extract claims and find contradictions
  5. ResearchGapFinder    — Identify research gaps
  6. DefensePrep          — Generate defense questions and counterarguments
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Citation:
    """Represents a citation extracted from the literature."""

    author: str
    year: str
    title: str
    page: Optional[str] = None
    quote: Optional[str] = None

    def to_latex_cite(self) -> str:
        """Generate a LaTeX-style citation key (authorYEAR)."""
        first_author = self.author.split()[0].lower() if self.author else "unknown"
        first_author = re.sub(r"[^a-z]", "", first_author)
        return f"{first_author}{self.year}"

    def to_bibtex(self, doc_type: str = "article") -> str:
        """Generate a BibTeX entry."""
        key = self.to_latex_cite()
        return (
            f"@{doc_type}{{{key},\n"
            f"    author = {{{self.author}}},\n"
            f"    title = {{{self.title}}},\n"
            f"    year = {{{self.year}}}\n"
            f"}}"
        )

    def to_apa(self) -> str:
        """Generate an APA-style reference."""
        return f"{self.author} ({self.year}). {self.title}."

    def to_inline_cite(self, style: str = "latex") -> str:
        """Generate an inline citation string."""
        if style == "latex":
            return f"\\citep{{{self.to_latex_cite()}}}"
        # APA inline
        if " " in self.author:
            parts = self.author.split()
            if "et al" in self.author.lower():
                return f"({parts[0]} et al., {self.year})"
            return f"({parts[0]}, {self.year})"
        return f"({self.author}, {self.year})"


class CitationGenerator:
    """Generate citations from retrieved sources."""

    def __init__(self):
        self.citations: List[Citation] = []

    def extract_citations(self, sources: List[Dict]) -> List[Citation]:
        self.citations = []
        for source in sources:
            meta = source.get("metadata", {})
            self.citations.append(
                Citation(
                    author=meta.get("author", "Unknown"),
                    year=meta.get("year", "n.d."),
                    title=meta.get("title", "Untitled"),
                    page=meta.get("page"),
                    quote=source.get("text", "")[:500],
                )
            )
        return self.citations

    def generate_bibliography(self, style: str = "bibtex") -> str:
        if style == "bibtex":
            return "\n\n".join(c.to_bibtex() for c in self.citations)
        return "\n".join(c.to_apa() for c in self.citations)

    def generate_inline_citations(self, style: str = "latex") -> List[str]:
        return [c.to_inline_cite(style) for c in self.citations]


class LiteratureReviewDrafter:
    """Draft literature review sections from retrieved content."""

    def __init__(self, rag_engine):
        self.engine = rag_engine

    def draft_section(
        self,
        topic: str,
        section_title: str = "Literature Review",
        style: str = "academic",
    ) -> Tuple[str, List[Dict]]:
        query = (
            f"Synthesize the academic literature on: {topic}\n\n"
            f"Structure your response as a literature review with:\n"
            f"1. An introduction to the topic\n"
            f"2. Key findings from different authors (cite specific papers)\n"
            f"3. Areas of agreement and disagreement\n"
            f"4. A synthesis connecting the findings\n\n"
            f"Use academic writing style. Reference specific authors and years."
        )
        return self.engine.query(query, top_k=6)


class PaperComparator:
    """Compare papers along specified dimensions."""

    DIMENSIONS = [
        "methodology",
        "data_sources",
        "main_findings",
        "theoretical_framework",
        "limitations",
        "policy_implications",
    ]

    def __init__(self, rag_engine):
        self.engine = rag_engine

    def compare_papers(
        self, paper_ids: List[str] = None, dimension: str = "all"
    ) -> Tuple[str, List[Dict]]:
        dims = self.DIMENSIONS if dimension == "all" else [dimension]
        dims_str = ", ".join(dims)
        query = (
            f"Create a comparison table of the papers in the literature.\n\n"
            f"Compare them on these dimensions: {dims_str}\n\n"
            f"Format as a markdown table with:\n"
            f"- Rows: Each paper (Author, Year)\n"
            f"- Columns: {dims_str}\n\n"
            f"Be specific about each paper's approach."
        )
        return self.engine.query(query, top_k=8)


class ClaimExtractor:
    """Extract and validate claims from the literature."""

    def __init__(self, rag_engine):
        self.engine = rag_engine

    def extract_claims(self, topic: str) -> Tuple[str, List[Dict]]:
        query = (
            f"Extract the key claims made in the academic literature about: {topic}\n\n"
            f"For each claim:\n"
            f"1. State the claim clearly\n"
            f"2. Identify which paper(s) make this claim\n"
            f"3. Note the evidence provided\n"
            f"4. Rate confidence (strong/moderate/weak based on evidence)\n\n"
            f"Format as a numbered list with sub-bullets for evidence."
        )
        return self.engine.query(query, top_k=6)

    def find_contradictions(self, topic: str) -> Tuple[str, List[Dict]]:
        query = (
            f"Identify contradictory or conflicting findings about: {topic}\n\n"
            f"For each contradiction:\n"
            f"1. State both positions\n"
            f"2. Identify which papers support each\n"
            f"3. Explain potential reasons for disagreement\n"
            f"4. Note which position has stronger support\n\n"
            f"Be specific about authors and findings."
        )
        return self.engine.query(query, top_k=6)


class ResearchGapFinder:
    """Identify research gaps from the literature."""

    def __init__(self, rag_engine):
        self.engine = rag_engine

    def find_gaps(self, topic: str = None) -> Tuple[str, List[Dict]]:
        if topic:
            query = (
                f"Based on the literature about {topic}, identify research gaps:\n\n"
                f"1. What questions remain unanswered?\n"
                f"2. What methodological improvements could be made?\n"
                f"3. What data or contexts haven't been studied?\n"
                f"4. What theoretical connections haven't been explored?\n\n"
                f"Be specific and suggest potential research directions."
            )
        else:
            query = (
                "Analyze the collected literature and identify research gaps:\n\n"
                "1. What are the main unresolved questions?\n"
                "2. What methodological approaches are underutilized?\n"
                "3. What contexts or populations haven't been studied?\n"
                "4. Where do the papers suggest future research is needed?\n\n"
                "Be specific about which papers identify these gaps."
            )
        return self.engine.query(query, top_k=8)


class DefensePrep:
    """Prepare for thesis defense with questions and counterarguments."""

    def __init__(self, rag_engine):
        self.engine = rag_engine

    def generate_questions(self, thesis_topic: str = None) -> Tuple[str, List[Dict]]:
        query = (
            "Based on the literature, generate challenging thesis defense questions.\n\n"
            "Consider questions about:\n"
            "1. **Methodology**: How do your methods compare to the literature?\n"
            "2. **Theory**: How does your work relate to existing frameworks?\n"
            "3. **Data**: What are potential limitations?\n"
            "4. **Alternative explanations**: What alternative interpretations exist?\n"
            "5. **Implications**: What are the policy/practical implications?\n"
            "6. **Future research**: Where does this research lead?\n\n"
            "For each question, suggest how to answer drawing on the literature."
        )
        return self.engine.query(query, top_k=6)

    def prepare_counterarguments(self, claim: str) -> Tuple[str, List[Dict]]:
        query = (
            f'For the claim: "{claim}"\n\n'
            f"1. What counterarguments might critics raise?\n"
            f"2. How does the literature support or challenge this claim?\n"
            f"3. What evidence can be used to defend this position?\n"
            f"4. What caveats or limitations should be acknowledged?\n\n"
            f"Draw on specific papers and findings."
        )
        return self.engine.query(query, top_k=6)


# ── Factory ───────────────────────────────────────────────────────────


def create_academic_features(rag_engine) -> Dict:
    """Create all academic feature instances."""
    return {
        "citation_generator": CitationGenerator(),
        "literature_drafter": LiteratureReviewDrafter(rag_engine),
        "paper_comparator": PaperComparator(rag_engine),
        "claim_extractor": ClaimExtractor(rag_engine),
        "gap_finder": ResearchGapFinder(rag_engine),
        "defense_prep": DefensePrep(rag_engine),
    }
