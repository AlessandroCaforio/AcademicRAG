#!/usr/bin/env python3
"""
Paper Catalog Generator
=======================
Auto-generates structured metadata for ALL papers in the RAG corpus.
Pure Python heuristics — no LLM calls required.

Features:
  - Abstract extraction from PDF text
  - Key findings extraction via signal phrases
  - Methodology tagging (shift-share, TWFE, IV, RDD, etc.)
  - BibTeX key matching (fuzzy author+year+title)
  - Category assignment (labor economics, political behavior, etc.)
  - Concept graph linking

Usage:
    python3 generate_catalog.py

Inputs:
    papers/paper_chunks.json     — chunked paper text (from add_paper.py)
    references.bib               — BibTeX file (optional, for key matching)
    concept_graph.json           — concept graph (optional, for linking)

Outputs:
    paper_catalog.json           — machine-readable catalog
    paper_catalog.md             — human-readable catalog
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CHUNKS_FILE = SCRIPT_DIR / "papers" / "paper_chunks.json"
CONCEPT_GRAPH_FILE = SCRIPT_DIR / "concept_graph.json"
OUTPUT_JSON = SCRIPT_DIR / "paper_catalog.json"
OUTPUT_MD = SCRIPT_DIR / "paper_catalog.md"

# Optional: point to a BibTeX file for key matching
BIB_FILE = SCRIPT_DIR / "references.bib"

# ── Configuration ─────────────────────────────────────────────────────

# PDF stems to skip (not actual papers)
NON_PAPER_STEMS: set[str] = set()

# Known duplicates: duplicate_stem → canonical_stem
DUPLICATE_MAP: dict[str, str] = {}

# Manual metadata overrides: stem → partial metadata dict
STEM_OVERRIDES: dict[str, dict] = {}

# ── Methodology Tags ─────────────────────────────────────────────────

METHODOLOGY_KEYWORDS = {
    "shift_share": [
        "shift-share", "shift share", "bartik", "bartik instrument",
        "industry composition", "exposure measure",
    ],
    "twfe": [
        "two-way fixed effect", "twfe", "two way fixed effect",
        "difference-in-difference", "diff-in-diff",
    ],
    "iv": [
        "instrumental variable", "two-stage", "2sls", "first stage",
        "exclusion restriction",
    ],
    "rdd": [
        "regression discontinuity", "rdd", "discontinuity design",
    ],
    "survey_analysis": [
        "survey data", "survey experiment", "survey response", "likert",
    ],
    "panel_data": [
        "panel data", "longitudinal", "fixed effect", "random effect",
    ],
    "task_framework": [
        "task model", "task-based", "routine task", "task displacement",
    ],
    "structural_estimation": [
        "structural model", "structural estimation", "general equilibrium",
    ],
    "cross_national": [
        "cross-country", "cross-national", "oecd countries",
    ],
}

# ── Category Assignment ───────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "labor_economics": [
        "wage", "employment", "labor market", "automation", "robot",
        "task", "labor share", "productivity",
    ],
    "political_behavior": [
        "voting", "election", "partisan", "political", "populis",
        "polarization", "ideology", "electoral",
    ],
    "psychology": [
        "authoritarianism", "threat perception", "identity threat",
        "psychological", "attitude formation", "status threat",
    ],
    "political_economy": [
        "trade", "import competition", "globalization", "austerity",
        "inequality", "economic nationalism",
    ],
    "methodology": [
        "econometric", "identification strategy", "shift-share",
        "quasi-experiment", "causal inference",
    ],
    "technology": [
        "artificial intelligence", "machine learning", "generative ai",
        "large language model", "gpt", "deep learning",
    ],
}

# ── Signal Phrases for Key Findings ───────────────────────────────────

SIGNAL_PHRASES = [
    "we find that", "we find", "we show that", "we show",
    "i find that", "i find", "i show that", "i show",
    "results indicate", "results show", "results suggest",
    "evidence suggests", "evidence indicates",
    "the main result", "our main finding", "the key finding",
    "this paper finds", "this paper shows",
    "we document", "we demonstrate", "we estimate",
]


# ── Paper Discovery ───────────────────────────────────────────────────


def load_chunks():
    with open(CHUNKS_FILE) as f:
        return json.load(f)


def discover_papers(chunks):
    """Group chunks by PDF stem, extract metadata, deduplicate."""
    papers_by_stem = defaultdict(list)
    for chunk in chunks:
        m = re.match(r"^(.+?)_p(\d+)_c(\d+)$", chunk["id"])
        if m:
            papers_by_stem[m.group(1)].append(chunk)

    papers = {}
    for stem, stem_chunks in papers_by_stem.items():
        if stem in NON_PAPER_STEMS or stem in DUPLICATE_MAP:
            continue

        meta = stem_chunks[0].get("metadata", {})
        authors = _parse_authors(meta.get("author", "Unknown"))
        title = _clean_title(meta.get("title", stem), stem)
        year = meta.get("year", "unknown")

        if stem in STEM_OVERRIDES:
            o = STEM_OVERRIDES[stem]
            authors = o.get("authors", authors)
            title = o.get("title", title)
            year = o.get("year", year)

        paper_id = _make_paper_id(authors, year, stem, papers)
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "pdf_stem": stem,
            "chunk_count": len(stem_chunks),
            "chunks": stem_chunks,
        }

    return papers


def _parse_authors(raw_author):
    parts = raw_author.split()
    if "et" in parts and "al" in parts:
        idx = parts.index("et")
        return [" ".join(parts[:idx])] + ["et al."]
    return parts if parts else ["Unknown"]


def _clean_title(raw_title, stem):
    if raw_title and raw_title not in ("unknown", stem):
        return re.sub(r"\s+", " ", raw_title.replace("_", " ")).strip()
    parts = stem.split("_")
    year_idx = next(
        (i for i, p in enumerate(parts) if re.match(r"^\d{4}$", p)), None
    )
    title_parts = parts[year_idx + 1 :] if year_idx is not None else parts[1:]
    return " ".join(title_parts).replace("_", " ").strip()


def _make_paper_id(authors, year, stem, existing):
    first_last = re.sub(r"[^a-z]", "", authors[0].split()[-1].lower())
    if len(authors) >= 2 and authors[1] != "et al.":
        second_last = re.sub(r"[^a-z]", "", authors[1].split()[-1].lower())
        base = f"{first_last}_{second_last}_{year}"
    else:
        base = f"{first_last}_{year}"

    if base not in existing:
        return base
    # Disambiguate with a title word
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", stem)]
    for w in words:
        if w not in first_last and len(w) >= 4:
            return f"{base}_{w}"
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ── Abstract Extraction ──────────────────────────────────────────────


def extract_abstract(paper):
    chunks_sorted = sorted(
        paper["chunks"],
        key=lambda c: (c["metadata"].get("page", 0), c["metadata"].get("chunk", 0)),
    )
    early_text = " ".join(c["text"] for c in chunks_sorted[:8])

    patterns = [
        r"[Aa]bstract[:\.\s]*\n*([\s\S]{100,2000}?)(?:\n\s*(?:Keywords|JEL|Key\s*words|1[\.\s]+Introduction))",
        r"[Aa]bstract[:\.\s]*\n*([\s\S]{100,1200}?)(?:\n\n|\r\n\r\n)",
    ]
    for pat in patterns:
        m = re.search(pat, early_text)
        if m:
            abstract = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(abstract) > 600:
                cut = abstract[:650].rfind(".")
                abstract = abstract[: cut + 1] if cut > 400 else abstract[:600] + "..."
            return abstract

    for c in chunks_sorted[:5]:
        text = c["text"].strip()
        if len(text) >= 150 and not text.lower().startswith(("references", "bibliography")):
            return re.sub(r"\s+", " ", text)[:600] + "..."

    return "Abstract not available."


# ── Key Findings Extraction ───────────────────────────────────────────


def extract_key_findings(paper, max_findings=5):
    scored = []
    for chunk in paper["chunks"]:
        text = chunk["text"]
        page = chunk["metadata"].get("page", 0)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sent_lower = sent.lower()
            for phrase in SIGNAL_PHRASES:
                if phrase in sent_lower and len(sent) > 60:
                    clean = re.sub(r"\s+", " ", sent).strip()[:300]
                    scored.append((page, clean))
                    break

    # Deduplicate by Jaccard similarity
    kept = []
    for s in scored:
        words = set(s[1].lower().split())
        if not any(
            len(words & set(k[1].lower().split())) / max(len(words | set(k[1].lower().split())), 1) > 0.5
            for k in kept
        ):
            kept.append(s)

    return [f[1] for f in kept[:max_findings]]


# ── Methodology Tagging ──────────────────────────────────────────────


def tag_methodology(paper, min_hits=3):
    full_text = " ".join(c["text"].lower() for c in paper["chunks"])
    tags = []
    for tag, keywords in METHODOLOGY_KEYWORDS.items():
        hits = sum(len(re.findall(re.escape(kw), full_text)) for kw in keywords)
        if hits >= min_hits:
            tags.append(tag)
    return tags


def assign_category(paper):
    full_text = " ".join(c["text"].lower() for c in paper["chunks"][:30])
    scores = {
        cat: sum(len(re.findall(re.escape(kw), full_text)) for kw in keywords)
        for cat, keywords in CATEGORY_KEYWORDS.items()
    }
    return max(scores, key=scores.get) if scores and max(scores.values()) > 0 else "other"


# ── BibTeX Matching ───────────────────────────────────────────────────


def load_bibtex_entries():
    if not BIB_FILE.exists():
        return []
    with open(BIB_FILE) as f:
        content = f.read()
    entries = []
    for m in re.finditer(r"@\w+\{(\w+),\s*(.*?)(?=\n@|\Z)", content, re.DOTALL):
        key, body = m.group(1), m.group(2)
        author = (re.search(r"author\s*=\s*\{([^}]+)\}", body) or type("", (), {"group": lambda s, x: ""})()).group(1)
        year = (re.search(r"year\s*=\s*\{?(\d{4})\}?", body) or type("", (), {"group": lambda s, x: ""})()).group(1)
        title = (re.search(r"title\s*=\s*\{([^}]+)\}", body) or type("", (), {"group": lambda s, x: ""})()).group(1)
        first_last = ""
        if author:
            first = author.split(" and ")[0].strip()
            first_last = (first.split(",")[0] if "," in first else first.split()[-1]).strip().lower()
        entries.append({"key": key, "first_author_last": first_last, "year": year, "title": title})
    return entries


def match_bibtex(paper, bib_entries):
    if not paper["authors"] or paper["year"] == "unknown" or not bib_entries:
        return None
    first_last = re.sub(r"[^a-z]", "", paper["authors"][0].split()[-1].lower())
    title = paper["title"].lower()

    def overlap(bib_title):
        bw = set(re.findall(r"[a-z]{4,}", bib_title.lower()))
        pw = set(re.findall(r"[a-z]{4,}", title))
        return len(bw & pw) / min(len(bw), len(pw)) if bw and pw else 0

    candidates = [
        e for e in bib_entries
        if re.sub(r"[^a-z]", "", e["first_author_last"]) == first_last
        and e["year"] == paper["year"]
    ]
    if len(candidates) == 1:
        return candidates[0]["key"]
    if candidates:
        best = max(candidates, key=lambda e: overlap(e["title"]))
        if overlap(best["title"]) > 0.2:
            return best["key"]
    return None


# ── Concept Graph Linking ─────────────────────────────────────────────


def link_concepts(paper, concepts, min_mentions=3):
    if not concepts:
        return []
    full_text = " ".join(c["text"].lower() for c in paper["chunks"])
    linked = []
    for cid, concept in concepts.items():
        label = concept.get("label", "").lower()
        if not label:
            continue
        variants = [label, label.replace("-", " "), label.replace(" ", "-")]
        total = sum(len(re.findall(re.escape(v), full_text)) for v in variants)
        if total >= min_mentions:
            linked.append(cid)
    return linked


# ── Output ────────────────────────────────────────────────────────────


def write_outputs(papers):
    # JSON
    catalog = {
        "meta": {
            "total_papers": len(papers),
            "total_with_bibtex": sum(1 for p in papers.values() if p.get("bibtex_key")),
        },
        "papers": papers,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {OUTPUT_JSON}")

    # Markdown
    lines = [
        "# Paper Catalog",
        "",
        f"**Total papers**: {len(papers)}  ",
        f"**With BibTeX key**: {sum(1 for p in papers.values() if p.get('bibtex_key'))}",
        "",
        "---",
        "",
    ]
    for pid, paper in sorted(papers.items(), key=lambda x: (x[1]["year"], x[0])):
        authors_str = ", ".join(paper["authors"])
        lines.append(f"### {authors_str} ({paper['year']})")
        lines.append(f"**{paper['title']}**  ")
        lines.append(f"*{paper['chunk_count']} chunks | Category: {paper.get('category', 'unknown')}*")
        if paper.get("bibtex_key"):
            lines.append(f"  \nBibTeX: `{paper['bibtex_key']}`")
        lines.append("")
        if paper.get("abstract") and paper["abstract"] != "Abstract not available.":
            lines.append(f"> {paper['abstract'][:400]}")
            lines.append("")
        if paper.get("key_findings"):
            lines.append("**Key findings:**")
            for f in paper["key_findings"][:3]:
                lines.append(f"- {f[:200]}")
            lines.append("")
        tags = [f"`{t}`" for t in paper.get("methodology_tags", [])]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
            lines.append("")
        lines.append("---\n")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {OUTPUT_MD}")


# ── Main Pipeline ─────────────────────────────────────────────────────


def generate_catalog():
    print("=" * 60)
    print("Paper Catalog Generator")
    print("=" * 60)

    print("\n[1/6] Loading chunks...")
    chunks = load_chunks()
    print(f"  {len(chunks):,} chunks")

    print("[2/6] Discovering papers...")
    papers = discover_papers(chunks)
    print(f"  {len(papers)} unique papers")

    print("[3/6] Extracting abstracts & findings...")
    for paper in papers.values():
        paper["abstract"] = extract_abstract(paper)
        paper["key_findings"] = extract_key_findings(paper)

    print("[4/6] Tagging methodology & category...")
    for paper in papers.values():
        paper["methodology_tags"] = tag_methodology(paper)
        paper["category"] = assign_category(paper)

    print("[5/6] Matching BibTeX keys...")
    bib_entries = load_bibtex_entries()
    concepts = {}
    if CONCEPT_GRAPH_FILE.exists():
        with open(CONCEPT_GRAPH_FILE) as f:
            concepts = json.load(f).get("concepts", {})
    for paper in papers.values():
        paper["bibtex_key"] = match_bibtex(paper, bib_entries)
        paper["related_concepts"] = link_concepts(paper, concepts)

    # Remove chunk data before output
    for paper in papers.values():
        del paper["chunks"]

    print("[6/6] Writing outputs...")
    write_outputs(papers)

    # Summary
    cats = Counter(p["category"] for p in papers.values())
    print(f"\nCategories: {dict(cats.most_common())}")
    print("Done!")


if __name__ == "__main__":
    generate_catalog()
