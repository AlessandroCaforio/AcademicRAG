#!/usr/bin/env python3
"""
AcademicRAG — Concept Graph Builder

Incrementally integrates paper extractions into a concept knowledge graph.
Handles concept deduplication (fuzzy name matching), edge creation,
and graph statistics.

Usage:
    # Integrate a single extraction
    python build_graph.py extractions/paper_id_extraction.json

    # Integrate all extractions in a directory
    python build_graph.py extractions/

    # Rebuild graph from scratch (all extractions)
    python build_graph.py extractions/ --rebuild

    # Show graph statistics
    python build_graph.py --stats
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CONCEPT_GRAPH_PATH = BASE_DIR / "concept_graph.json"
EXTRACTIONS_DIR = BASE_DIR / "extractions"

# ── Matching Settings ────────────────────────────────────────────────────
NAME_SIMILARITY_THRESHOLD = 0.75   # Fuzzy match threshold for concept names
LABEL_SIMILARITY_THRESHOLD = 0.80  # Fuzzy match threshold for concept labels

# ── Type Mapping ─────────────────────────────────────────────────────────
EXTRACTION_TO_CONCEPT_TYPE = {
    "constructs": "construct",
    "variables": "variable",
    "methods": "method",
    "claims": "claim",
}


# ── Graph Operations ─────────────────────────────────────────────────────


def load_graph(path: Path = CONCEPT_GRAPH_PATH) -> Dict:
    """Load existing concept graph or create empty one."""
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path) as f:
                graph = json.load(f)
        except (json.JSONDecodeError, OSError):
            graph = {}
    else:
        graph = {}

    # Ensure required keys
    graph.setdefault("meta", {
        "last_updated": "",
        "papers_processed": [],
        "total_concepts": 0,
        "total_edges": 0,
    })
    graph.setdefault("concepts", {})
    graph.setdefault("edges", [])
    graph.setdefault("papers", {})

    return graph


def save_graph(graph: Dict, path: Path = CONCEPT_GRAPH_PATH) -> None:
    """Save concept graph to disk."""
    # Update metadata counts
    graph["meta"]["total_concepts"] = len(graph["concepts"])
    graph["meta"]["total_edges"] = len(graph["edges"])
    graph["meta"]["last_updated"] = _now()

    with open(path, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)


def _now() -> str:
    """Current date as ISO string."""
    from datetime import date
    return date.today().isoformat()


# ── Concept Matching ─────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Normalize a concept name for comparison."""
    # snake_case -> lowercase words
    return re.sub(r"[_\-]", " ", name.lower()).strip()


def find_matching_concept(
    name: str,
    existing_concepts: Dict,
    threshold: float = NAME_SIMILARITY_THRESHOLD,
) -> Optional[str]:
    """
    Find an existing concept that matches the given name.

    Returns the concept ID if a match is found, None otherwise.
    Uses normalized name comparison with SequenceMatcher.
    """
    norm_name = normalize_name(name)

    # Exact match first
    if name in existing_concepts:
        return name
    if name.lower().replace(" ", "_") in existing_concepts:
        return name.lower().replace(" ", "_")

    # Fuzzy match on normalized names
    best_match = None
    best_score = 0

    for concept_id, concept in existing_concepts.items():
        # Match against concept ID
        score_id = SequenceMatcher(
            None, norm_name, normalize_name(concept_id)
        ).ratio()

        # Match against concept label
        label = concept.get("label", "")
        score_label = SequenceMatcher(
            None, norm_name, label.lower()
        ).ratio()

        score = max(score_id, score_label)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = concept_id

    return best_match


def name_to_label(name: str) -> str:
    """Convert snake_case name to Title Case label."""
    return name.replace("_", " ").title()


# ── Integration ──────────────────────────────────────────────────────────


def integrate_extraction(graph: Dict, extraction: Dict, verbose: bool = True) -> Dict:
    """
    Integrate a paper extraction into the concept graph.

    - Adds new concepts or updates existing ones with new mentions
    - Adds the paper to the papers registry
    - Infers edges from claims and method-variable relationships
    """
    paper_id = extraction["paper"]

    # Check if already processed
    if paper_id in graph["meta"].get("papers_processed", []):
        if verbose:
            print(f"  Skipping {paper_id} (already in graph)")
        return graph

    if verbose:
        print(f"\n  Integrating: {paper_id}")

    concepts_added = 0
    concepts_updated = 0
    edges_added = 0

    # Track which concept IDs this paper introduces
    paper_concept_ids = []

    # Process each extraction category
    for category, concept_type in EXTRACTION_TO_CONCEPT_TYPE.items():
        items = extraction.get("extracted", {}).get(category, [])

        for item in items:
            name = item.get("name", "unnamed")
            if name == "unnamed" or not name:
                continue

            # Check for existing match
            existing_id = find_matching_concept(name, graph["concepts"])

            if existing_id:
                # Update existing concept
                concept = graph["concepts"][existing_id]
                if paper_id not in concept.get("mentioned_in", []):
                    concept.setdefault("mentioned_in", []).append(paper_id)
                    concepts_updated += 1

                # Add operationalization if it's a variable
                if concept_type == "variable" and item.get("operationalization"):
                    concept.setdefault("operationalizations", []).append({
                        "paper": paper_id,
                        "measure": item["operationalization"],
                        "page": item.get("page", 0),
                    })

                # Add quote if new
                if item.get("quote"):
                    existing_quotes = [
                        q.get("text", "") for q in concept.get("quotes", [])
                    ]
                    if item["quote"] not in existing_quotes:
                        concept.setdefault("quotes", []).append({
                            "paper": paper_id,
                            "text": item["quote"],
                            "page": item.get("page", 0),
                        })

                paper_concept_ids.append(existing_id)
            else:
                # Create new concept
                concept_id = name.lower().replace(" ", "_")
                new_concept = {
                    "id": concept_id,
                    "type": concept_type,
                    "label": name_to_label(name),
                    "definition": item.get("definition", ""),
                    "first_introduced_by": paper_id if item.get("is_new") else paper_id,
                    "mentioned_in": [paper_id],
                }

                # Add type-specific fields
                if concept_type == "variable" and item.get("operationalization"):
                    new_concept["operationalizations"] = [{
                        "paper": paper_id,
                        "measure": item["operationalization"],
                        "page": item.get("page", 0),
                    }]

                if item.get("quote"):
                    new_concept["quotes"] = [{
                        "paper": paper_id,
                        "text": item["quote"],
                        "page": item.get("page", 0),
                    }]

                # For claims, store the claim details
                if concept_type == "claim":
                    new_concept["claim_text"] = item.get("claim", "")
                    new_concept["evidence"] = item.get("evidence", "")
                    new_concept["strength"] = item.get("strength", "moderate")

                # For methods, store details
                if concept_type == "method":
                    new_concept["details"] = item.get("details", "")

                graph["concepts"][concept_id] = new_concept
                paper_concept_ids.append(concept_id)
                concepts_added += 1

    # Infer edges from claims
    edges_added += _infer_edges(graph, extraction, paper_concept_ids)

    # Register the paper
    graph["papers"][paper_id] = {
        "id": paper_id,
        "title": extraction.get("title", ""),
        "authors": extraction.get("authors", []),
        "year": extraction.get("year", 0),
        "processed": True,
        "concepts_extracted": paper_concept_ids,
        "main_claim": _get_main_claim(extraction),
        "method": _get_main_method(extraction),
    }

    # Update metadata
    if paper_id not in graph["meta"].get("papers_processed", []):
        graph["meta"].setdefault("papers_processed", []).append(paper_id)

    if verbose:
        print(f"    Concepts added: {concepts_added}")
        print(f"    Concepts updated: {concepts_updated}")
        print(f"    Edges added: {edges_added}")
        print(f"    Graph now: {len(graph['concepts'])} concepts, {len(graph['edges'])} edges")

    return graph


def _infer_edges(
    graph: Dict, extraction: Dict, paper_concepts: List[str]
) -> int:
    """
    Infer edges between concepts from the extraction.

    Rules:
    1. Methods → Variables they measure (type: "measures")
    2. Claims about causation between extracted variables (type: "causes")
    3. Variables that moderate/mediate others (from claim text)
    """
    paper_id = extraction["paper"]
    edges_added = 0
    existing_edges = {
        (e["source"], e["target"], e["type"]) for e in graph["edges"]
    }

    methods = extraction.get("extracted", {}).get("methods", [])
    variables = extraction.get("extracted", {}).get("variables", [])
    claims = extraction.get("extracted", {}).get("claims", [])

    # Method → Variable edges (measures)
    for method in methods:
        method_id = method["name"].lower().replace(" ", "_")
        if method_id not in graph["concepts"]:
            continue
        for var in variables:
            var_id = var["name"].lower().replace(" ", "_")
            if var_id not in graph["concepts"]:
                continue
            edge_key = (method_id, var_id, "measures")
            if edge_key not in existing_edges:
                graph["edges"].append({
                    "source": method_id,
                    "target": var_id,
                    "type": "measures",
                    "papers": [paper_id],
                })
                existing_edges.add(edge_key)
                edges_added += 1

    # Causal edges from claims
    causal_patterns = [
        r"(.+?)\s+(?:causes?|leads? to|increases?|decreases?|drives?|affects?)\s+(.+)",
        r"effect of (.+?) on (.+)",
        r"impact of (.+?) on (.+)",
        r"(.+?) → (.+)",
    ]

    for claim in claims:
        claim_text = claim.get("claim", "").lower()
        for pattern in causal_patterns:
            match = re.search(pattern, claim_text)
            if match:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()

                # Try to match to existing concepts
                source_id = find_matching_concept(
                    source_text, graph["concepts"], threshold=0.6
                )
                target_id = find_matching_concept(
                    target_text, graph["concepts"], threshold=0.6
                )

                if source_id and target_id and source_id != target_id:
                    edge_key = (source_id, target_id, "causes")
                    if edge_key not in existing_edges:
                        strength = claim.get("strength", "moderate")
                        graph["edges"].append({
                            "source": source_id,
                            "target": target_id,
                            "type": "causes",
                            "papers": [paper_id],
                            "evidence": strength,
                        })
                        existing_edges.add(edge_key)
                        edges_added += 1
                break  # Only use first matching pattern per claim

    return edges_added


def _get_main_claim(extraction: Dict) -> str:
    """Get the strongest claim from an extraction."""
    claims = extraction.get("extracted", {}).get("claims", [])
    if not claims:
        return ""

    # Prefer "strong" claims
    for claim in claims:
        if claim.get("strength") == "strong":
            return claim.get("claim", "")

    return claims[0].get("claim", "")


def _get_main_method(extraction: Dict) -> str:
    """Get the primary method from an extraction."""
    methods = extraction.get("extracted", {}).get("methods", [])
    if not methods:
        return ""
    return methods[0].get("definition", methods[0].get("name", ""))


# ── Statistics ───────────────────────────────────────────────────────────


def print_stats(graph: Dict) -> None:
    """Print graph statistics."""
    concepts = graph.get("concepts", {})
    edges = graph.get("edges", [])
    papers = graph.get("papers", {})

    print(f"\n{'='*50}")
    print(f"  Concept Graph Statistics")
    print(f"{'='*50}")
    print(f"  Papers processed:  {len(papers)}")
    print(f"  Total concepts:    {len(concepts)}")
    print(f"  Total edges:       {len(edges)}")

    # Concept type breakdown
    type_counts = defaultdict(int)
    for c in concepts.values():
        type_counts[c.get("type", "unknown")] += 1

    print(f"\n  Concepts by type:")
    for ctype, count in sorted(type_counts.items()):
        print(f"    {ctype:12s}: {count}")

    # Edge type breakdown
    edge_counts = defaultdict(int)
    for e in edges:
        edge_counts[e.get("type", "unknown")] += 1

    print(f"\n  Edges by type:")
    for etype, count in sorted(edge_counts.items()):
        print(f"    {etype:12s}: {count}")

    # Most connected concepts
    mentions = defaultdict(int)
    for c_id, c in concepts.items():
        mentions[c_id] = len(c.get("mentioned_in", []))

    print(f"\n  Most referenced concepts (top 10):")
    for c_id, count in sorted(mentions.items(), key=lambda x: -x[1])[:10]:
        label = concepts[c_id].get("label", c_id)
        ctype = concepts[c_id].get("type", "?")
        print(f"    {label:30s} ({ctype}) — {count} papers")

    # Papers by year
    year_counts = defaultdict(int)
    for p in papers.values():
        year = p.get("year", 0)
        if year:
            year_counts[year] += 1

    if year_counts:
        print(f"\n  Papers by year:")
        for year, count in sorted(year_counts.items()):
            print(f"    {year}: {count}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Build concept graph from paper extractions"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Extraction JSON file or directory of extractions",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=CONCEPT_GRAPH_PATH,
        help=f"Path to concept graph (default: {CONCEPT_GRAPH_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild graph from scratch (ignores existing graph)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics",
    )

    args = parser.parse_args()

    # Stats mode
    if args.stats:
        graph = load_graph(args.graph)
        print_stats(graph)
        return

    if args.input is None:
        parser.print_help()
        return

    # Load or create graph
    if args.rebuild:
        print("Rebuilding graph from scratch...")
        graph = load_graph(Path("/dev/null"))  # Empty graph
    else:
        graph = load_graph(args.graph)
        print(f"Loaded graph: {len(graph['concepts'])} concepts, {len(graph['edges'])} edges")

    # Find extraction files
    if args.input.is_dir():
        extraction_files = sorted(args.input.glob("*_extraction.json"))
    elif args.input.is_file():
        extraction_files = [args.input]
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)

    if not extraction_files:
        print(f"No extraction files found in {args.input}")
        return

    print(f"Found {len(extraction_files)} extraction(s) to process")

    # Integrate each extraction
    for path in extraction_files:
        try:
            with open(path) as f:
                extraction = json.load(f)
            graph = integrate_extraction(graph, extraction)
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")
            continue

    # Save
    save_graph(graph, args.graph)
    print(f"\nGraph saved to: {args.graph}")
    print_stats(graph)


if __name__ == "__main__":
    main()
