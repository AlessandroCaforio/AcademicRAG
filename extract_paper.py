#!/usr/bin/env python3
"""
AcademicRAG — Automated Deep Paper Extraction

Uses an LLM to extract structured information from paper chunks:
  - Constructs (theoretical concepts)
  - Variables (measured quantities)
  - Methods (empirical approaches)
  - Claims (key findings with evidence strength)

Outputs a JSON file matching the schema of manual extractions.

Usage:
    # Extract a single paper
    python extract_paper.py "Automation Rent Dissipation"

    # Extract with Claude API
    python extract_paper.py "Automation Rent Dissipation" --backend claude

    # Extract with Ollama
    python extract_paper.py "Automation Rent Dissipation" --backend ollama

    # Dry-run: show which chunks would be selected
    python extract_paper.py "Automation Rent Dissipation" --dry-run
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "papers"
PAPER_CHUNKS_PATH = PAPERS_DIR / "paper_chunks.json"
EXTRACTIONS_DIR = BASE_DIR / "extractions"

# ── LLM Settings ─────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_CODE_PATH = os.getenv("CLAUDE_CODE_PATH", "claude")

# ── Extraction Settings ──────────────────────────────────────────────────
MAX_CHUNKS_PER_CALL = 12       # Maximum chunks to include in one LLM call
MAX_CHARS_PER_CALL = 25_000    # Maximum characters of paper text per call
INTRO_PAGES = 3                # Number of initial pages to always include
CONCLUSION_PAGES = 2           # Number of final pages to always include

# Keywords for identifying methodology-relevant chunks
METHOD_SIGNALS = [
    "method", "approach", "design", "estimation", "specification",
    "instrument", "identification", "regression", "fixed effect",
    "panel", "cross-section", "difference-in-difference", "IV",
    "shift-share", "bartik", "two-way", "TWFE", "OLS", "2SLS",
    "causal", "treatment", "control group", "sample",
]

# Keywords for identifying results/findings chunks
RESULTS_SIGNALS = [
    "we find", "results show", "evidence suggests", "coefficient",
    "significant", "p-value", "standard error", "table",
    "main result", "finding", "effect of", "impact of",
    "increases", "decreases", "associated with", "consistent with",
    "estimate", "our results", "we estimate", "the effect",
    "robustness", "heterogeneous", "subgroup",
]

# ── Extraction Prompt ────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert academic researcher performing structured extraction from a scholarly paper.

Given excerpts from the paper below, extract the following categories of information. Be thorough but precise — only extract items that are clearly present in the text.

## Categories

### 1. Constructs (Theoretical Concepts)
Theoretical concepts or frameworks introduced or used by the paper.
- name: snake_case identifier (e.g., "status_threat", "task_displacement")
- definition: One-sentence definition in your own words
- quote: A verbatim quote from the paper that defines or introduces this concept (keep under 200 chars)
- page: Page number where the concept appears (from the chunk metadata)
- is_new: true if this paper INTRODUCES the concept, false if it merely uses it

### 2. Variables (Measured Quantities)
Specific variables that are measured or constructed in the paper.
- name: snake_case identifier (e.g., "robot_exposure", "vote_share")
- definition: What the variable captures
- operationalization: How it is measured/constructed (data source, formula, units)
- page: Page number
- is_new: true if this paper introduces the variable

### 3. Methods (Empirical Approaches)
Research methods, estimation strategies, or identification approaches.
- name: snake_case identifier (e.g., "shift_share_iv", "panel_fixed_effects")
- definition: What the method is
- details: Specifics of how it is applied (data, timespan, controls, first-stage F-stat, etc.)
- page: Page number
- is_new: true if this paper introduces the method (vs. using an established one)

### 4. Claims (Key Findings)
Main empirical findings or theoretical claims made by the paper.
- name: snake_case identifier (e.g., "automation_increases_inequality")
- claim: The finding stated clearly
- evidence: Specific evidence (coefficient, CI, p-value, sample size) — be precise
- page: Page number
- strength: One of: "strong" (causal, robust), "moderate" (significant but caveats), "weak" (suggestive), "suggestive" (correlational), "methodological" (about method, not substance)

## Output Format

Return ONLY valid JSON matching this exact structure (no markdown, no commentary):

{{
  "constructs": [
    {{"name": "...", "definition": "...", "quote": "...", "page": 0, "is_new": false}}
  ],
  "variables": [
    {{"name": "...", "definition": "...", "operationalization": "...", "page": 0, "is_new": false}}
  ],
  "methods": [
    {{"name": "...", "definition": "...", "details": "...", "page": 0, "is_new": false}}
  ],
  "claims": [
    {{"name": "...", "claim": "...", "evidence": "...", "page": 0, "strength": "moderate"}}
  ]
}}

## Paper Excerpts

{paper_excerpts}

## Important Rules

1. Extract ONLY information clearly stated in the provided excerpts
2. Do NOT invent or hallucinate content — if something is unclear, skip it
3. Use snake_case for all names
4. Keep quotes verbatim from the text (do not paraphrase quotes)
5. Page numbers come from the [Page N] markers in the excerpts
6. Aim for 2-6 items per category (quality over quantity)
7. For "is_new", be conservative — most concepts are NOT new to a single paper
8. Return ONLY the JSON object — no preamble, no explanation, no markdown fences"""


# ── Chunk Selection ──────────────────────────────────────────────────────


def load_chunks() -> List[Dict]:
    """Load all paper chunks from disk."""
    if not PAPER_CHUNKS_PATH.exists():
        print(f"Error: {PAPER_CHUNKS_PATH} not found.")
        print("Run add_paper.py first to ingest your PDFs.")
        sys.exit(1)

    with open(PAPER_CHUNKS_PATH) as f:
        return json.load(f)


def get_paper_titles(chunks: List[Dict]) -> List[str]:
    """Get sorted list of unique paper titles."""
    return sorted(set(c["metadata"].get("title", "unknown") for c in chunks))


def find_paper(chunks: List[Dict], query: str) -> Tuple[str, List[Dict]]:
    """Find a paper by fuzzy title match and return its chunks."""
    query_lower = query.lower().strip()

    # Group chunks by title
    papers = {}
    for c in chunks:
        title = c["metadata"].get("title", "unknown")
        papers.setdefault(title, []).append(c)

    # Try exact match first
    for title in papers:
        if title.lower() == query_lower:
            return title, papers[title]

    # Try substring match
    matches = []
    for title in papers:
        if query_lower in title.lower() or title.lower() in query_lower:
            matches.append(title)

    # Try word overlap
    if not matches:
        query_words = set(query_lower.split())
        for title in papers:
            title_words = set(title.lower().split())
            overlap = len(query_words & title_words) / max(len(query_words), 1)
            if overlap >= 0.5:
                matches.append(title)

    if len(matches) == 1:
        return matches[0], papers[matches[0]]
    elif len(matches) > 1:
        print(f"Ambiguous query '{query}'. Multiple matches:")
        for i, m in enumerate(matches, 1):
            print(f"  {i}. {m} ({len(papers[m])} chunks)")
        choice = input("Select number (or 0 to cancel): ").strip()
        if choice.isdigit() and 0 < int(choice) <= len(matches):
            title = matches[int(choice) - 1]
            return title, papers[title]
        sys.exit(0)
    else:
        print(f"No paper found matching '{query}'.")
        print("\nAvailable papers:")
        for title in sorted(papers.keys()):
            print(f"  - {title} ({len(papers[title])} chunks)")
        sys.exit(1)


def select_strategic_chunks(
    paper_chunks: List[Dict],
    max_chunks: int = MAX_CHUNKS_PER_CALL,
    max_chars: int = MAX_CHARS_PER_CALL,
) -> List[Dict]:
    """
    Select the most informative chunks from a paper.

    Strategy:
    1. Always include first N pages (abstract, introduction)
    2. Always include last N pages (conclusion)
    3. Fill remaining budget with methodology/results chunks
    4. Respect both chunk count and character limits
    """
    def _page(c):
        """Get page number as int (metadata may store it as str)."""
        p = c["metadata"].get("page", 0)
        return int(p) if str(p).isdigit() else 0

    def _chunk_num(c):
        """Get chunk number as int."""
        n = c["metadata"].get("chunk", 0)
        return int(n) if str(n).isdigit() else 0

    # Sort by page then chunk number
    sorted_chunks = sorted(
        paper_chunks,
        key=lambda c: (_page(c), _chunk_num(c)),
    )

    max_page = max(_page(c) for c in sorted_chunks)
    selected = []
    selected_ids = set()

    def add_chunk(chunk):
        """Add a chunk if within budget."""
        nonlocal selected
        if chunk["id"] in selected_ids:
            return False
        total_chars = sum(len(c["text"]) for c in selected) + len(chunk["text"])
        if len(selected) >= max_chunks or total_chars > max_chars:
            return False
        selected.append(chunk)
        selected_ids.add(chunk["id"])
        return True

    # Phase 1: Intro pages (abstract + introduction)
    for c in sorted_chunks:
        if _page(c) <= INTRO_PAGES:
            if not add_chunk(c):
                break

    # Phase 2: Conclusion pages
    conclusion_start = max(max_page - CONCLUSION_PAGES, INTRO_PAGES + 1)
    for c in sorted_chunks:
        page = _page(c)
        if conclusion_start <= page <= max_page:
            if not add_chunk(c):
                break

    # Phase 3: Method-signal chunks
    method_chunks = _score_chunks(sorted_chunks, METHOD_SIGNALS)
    for c, _score in method_chunks[:8]:
        if not add_chunk(c):
            break

    # Phase 4: Results-signal chunks
    results_chunks = _score_chunks(sorted_chunks, RESULTS_SIGNALS)
    for c, _score in results_chunks[:8]:
        if not add_chunk(c):
            break

    # Re-sort by page order for coherent reading
    selected.sort(
        key=lambda c: (_page(c), _chunk_num(c))
    )
    return selected


def _score_chunks(chunks: List[Dict], signals: List[str]) -> List[Tuple[Dict, int]]:
    """Score chunks by keyword signal density."""
    scored = []
    for c in chunks:
        text_lower = c["text"].lower()
        score = sum(1 for s in signals if s.lower() in text_lower)
        if score > 0:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def format_chunks_for_prompt(chunks: List[Dict], title: str) -> str:
    """Format selected chunks into a prompt-ready string."""
    parts = [f'Paper: "{title}"\n']

    for c in chunks:
        meta = c["metadata"]
        page = meta.get("page", "?")
        # Remove null bytes and other control characters from PDF text
        text = c["text"].replace("\x00", "").replace("\ufffd", "")
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32))
        parts.append(f"[Page {page}]\n{text}\n")

    return "\n---\n".join(parts)


# ── LLM Backends ─────────────────────────────────────────────────────────


def extract_with_claude_api(prompt: str) -> str:
    """Call Claude API for extraction."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def extract_with_ollama(prompt: str) -> str:
    """Call Ollama for extraction."""
    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


def extract_with_claude_code(prompt: str) -> str:
    """Call Claude Code CLI for extraction."""
    # Remove CLAUDECODE env var to allow nested calls (safe for --print mode)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    result = subprocess.run(
        [CLAUDE_CODE_PATH, "--print", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=180,
        cwd="/tmp",
        env=env,
    )
    if result.returncode == 0:
        # Clean ANSI codes
        text = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub(
            "", result.stdout.strip()
        )
        return text
    raise RuntimeError(f"Claude Code error: {result.stderr}")


def call_llm(prompt: str, backend: str) -> str:
    """Route to the appropriate LLM backend."""
    if backend == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to .env or use --backend ollama"
            )
        return extract_with_claude_api(prompt)
    elif backend == "claude-code":
        return extract_with_claude_code(prompt)
    elif backend == "ollama":
        return extract_with_ollama(prompt)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── JSON Parsing ─────────────────────────────────────────────────────────


def parse_extraction_json(raw: str) -> Optional[Dict]:
    """Parse the LLM's JSON output, handling common formatting issues."""
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the output
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try fixing common issues: trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def validate_extraction(data: Dict) -> Dict:
    """Validate and clean the extraction output."""
    required_keys = {"constructs", "variables", "methods", "claims"}
    for key in required_keys:
        if key not in data:
            data[key] = []

    # Validate constructs
    for item in data.get("constructs", []):
        item.setdefault("name", "unnamed")
        item.setdefault("definition", "")
        item.setdefault("quote", "")
        item.setdefault("page", 0)
        item.setdefault("is_new", False)

    # Validate variables
    for item in data.get("variables", []):
        item.setdefault("name", "unnamed")
        item.setdefault("definition", "")
        item.setdefault("operationalization", "")
        item.setdefault("page", 0)
        item.setdefault("is_new", False)

    # Validate methods
    for item in data.get("methods", []):
        item.setdefault("name", "unnamed")
        item.setdefault("definition", "")
        item.setdefault("details", "")
        item.setdefault("page", 0)
        item.setdefault("is_new", False)

    # Validate claims
    valid_strengths = {"strong", "moderate", "weak", "suggestive", "methodological"}
    for item in data.get("claims", []):
        item.setdefault("name", "unnamed")
        item.setdefault("claim", "")
        item.setdefault("evidence", "")
        item.setdefault("page", 0)
        item.setdefault("strength", "moderate")
        if item["strength"] not in valid_strengths:
            item["strength"] = "moderate"

    return data


# ── Paper ID Generation ──────────────────────────────────────────────────


def make_paper_id(title: str, author: str = "", year: str = "") -> str:
    """Generate a snake_case paper ID from metadata."""
    # Try to extract from author field
    if author:
        # "Acemoglu Restrepo" -> "acemoglu_restrepo"
        parts = author.strip().lower().split()
        author_part = "_".join(parts[:3])  # Max 3 author names
    else:
        # Fall back to first few words of title
        words = re.sub(r"[^a-z0-9\s]", "", title.lower()).split()
        author_part = "_".join(words[:3])

    if year:
        return f"{author_part}_{year}"
    return author_part


# ── Main Extraction ──────────────────────────────────────────────────────


def extract_paper(
    title: str,
    chunks: List[Dict],
    backend: str = "claude",
    verbose: bool = True,
) -> Dict:
    """
    Extract structured information from a paper's chunks.

    Returns a complete extraction dict ready for JSON serialization.
    """
    paper_chunks = [c for c in chunks if c["metadata"].get("title") == title]
    if not paper_chunks:
        raise ValueError(f"No chunks found for paper: {title}")

    # Get metadata from first chunk
    meta = paper_chunks[0]["metadata"]
    author = meta.get("author", "Unknown")
    year = meta.get("year", "")
    paper_id = make_paper_id(title, author, year)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Extracting: {title}")
        print(f"  Author: {author} | Year: {year}")
        print(f"  Total chunks: {len(paper_chunks)}")

    # Select strategic chunks
    selected = select_strategic_chunks(paper_chunks)
    total_chars = sum(len(c["text"]) for c in selected)

    if verbose:
        pages = sorted(set(c["metadata"].get("page", 0) for c in selected))
        print(f"  Selected: {len(selected)} chunks ({total_chars:,} chars)")
        print(f"  Pages: {pages}")

    # Format for prompt
    paper_text = format_chunks_for_prompt(selected, title)
    prompt = EXTRACTION_PROMPT.format(paper_excerpts=paper_text)

    if verbose:
        print(f"  Calling {backend}...")

    # Call LLM
    start = time.time()
    raw_response = call_llm(prompt, backend)
    elapsed = time.time() - start

    if verbose:
        print(f"  Response received ({elapsed:.1f}s, {len(raw_response)} chars)")

    # Parse JSON
    extracted = parse_extraction_json(raw_response)
    if extracted is None:
        print(f"  WARNING: Failed to parse JSON response. Saving raw output.")
        # Save raw for debugging
        debug_path = EXTRACTIONS_DIR / f"{paper_id}_raw.txt"
        debug_path.write_text(raw_response)
        print(f"  Raw output saved to: {debug_path}")
        extracted = {"constructs": [], "variables": [], "methods": [], "claims": []}

    # Validate
    extracted = validate_extraction(extracted)

    # Build full extraction object
    result = {
        "paper": paper_id,
        "title": title,
        "authors": author.split() if " " in author else [author],
        "year": int(year) if year.isdigit() else 0,
        "journal": "",  # Not available from chunks metadata
        "extracted": extracted,
        "meta": {
            "backend": backend,
            "extraction_time": round(elapsed, 1),
            "chunks_used": len(selected),
            "total_chunks": len(paper_chunks),
        },
    }

    if verbose:
        n_constructs = len(extracted["constructs"])
        n_variables = len(extracted["variables"])
        n_methods = len(extracted["methods"])
        n_claims = len(extracted["claims"])
        total = n_constructs + n_variables + n_methods + n_claims
        print(f"  Extracted: {total} items")
        print(f"    Constructs: {n_constructs}")
        print(f"    Variables:  {n_variables}")
        print(f"    Methods:    {n_methods}")
        print(f"    Claims:     {n_claims}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured information from academic papers"
    )
    parser.add_argument(
        "paper",
        nargs="?",
        help="Paper title (or substring) to extract. Omit to list all papers.",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "claude-code", "ollama"],
        default="claude",
        help="LLM backend (default: claude)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: extractions/<paper_id>_extraction.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show chunk selection without calling LLM",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available papers",
    )

    args = parser.parse_args()

    chunks = load_chunks()

    # List mode
    if args.list or args.paper is None:
        titles = get_paper_titles(chunks)
        print(f"\nAvailable papers ({len(titles)}):\n")
        for t in titles:
            count = sum(1 for c in chunks if c["metadata"].get("title") == t)
            print(f"  [{count:3d} chunks] {t}")
        return

    # Find the paper
    title, paper_chunks = find_paper(chunks, args.paper)

    # Dry-run: show selected chunks
    if args.dry_run:
        selected = select_strategic_chunks(paper_chunks)
        print(f"\nPaper: {title}")
        print(f"Total chunks: {len(paper_chunks)}")
        print(f"Selected: {len(selected)} chunks")
        print(f"Total chars: {sum(len(c['text']) for c in selected):,}\n")
        for i, c in enumerate(selected, 1):
            meta = c["metadata"]
            print(
                f"  {i:2d}. Page {meta.get('page', '?'):3d} | "
                f"Chunk {meta.get('chunk', '?'):3d} | "
                f"{len(c['text']):5d} chars | "
                f"{c['text'][:80].strip()}..."
            )
        return

    # Extract
    EXTRACTIONS_DIR.mkdir(exist_ok=True)
    result = extract_paper(title, chunks, backend=args.backend)

    # Save
    output_path = args.output
    if output_path is None:
        output_path = EXTRACTIONS_DIR / f"{result['paper']}_extraction.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
