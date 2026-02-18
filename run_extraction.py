#!/usr/bin/env python3
"""
AcademicRAG — Batch Extraction Orchestrator

Processes all unextracted papers in your corpus, generating structured
extractions and incrementally building the concept graph.

Usage:
    # Extract all unprocessed papers (Claude API)
    python run_extraction.py

    # Extract with Ollama (free, local)
    python run_extraction.py --backend ollama

    # Extract only N papers (for testing)
    python run_extraction.py --limit 5

    # Resume from a specific paper
    python run_extraction.py --start-from "Paper Title"

    # Show what would be processed (no LLM calls)
    python run_extraction.py --dry-run

    # Rebuild graph after extraction
    python run_extraction.py --build-graph
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from extract_paper import (
    EXTRACTIONS_DIR,
    load_chunks,
    extract_paper,
    get_paper_titles,
)
from build_graph import (
    CONCEPT_GRAPH_PATH,
    load_graph,
    save_graph,
    integrate_extraction,
    print_stats,
)


def get_extracted_papers(extractions_dir: Path) -> Set[str]:
    """Get set of paper titles that already have extractions."""
    extracted = set()
    for path in extractions_dir.glob("*_extraction.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            title = data.get("title", "")
            if title:
                extracted.add(title)
        except (json.JSONDecodeError, KeyError):
            continue
    return extracted


def get_graph_papers(graph_path: Path) -> Set[str]:
    """Get set of paper IDs already in the concept graph."""
    if not graph_path.exists():
        return set()
    with open(graph_path) as f:
        graph = json.load(f)
    return set(graph.get("meta", {}).get("papers_processed", []))


# Papers to skip (non-paper content, appendices, etc.)
SKIP_TITLES = {
    "Appendix",
    "framework",
    "framework narrative",
}


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract all unprocessed papers"
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "claude-code", "ollama"],
        default="claude",
        help="LLM backend (default: claude)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of papers to process (0 = all)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Start processing from this paper title (skip earlier ones)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show papers to process without calling LLM",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Rebuild concept graph after extraction",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls (default: 2.0)",
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=5,
        help="Skip papers with fewer chunks than this (default: 5)",
    )

    args = parser.parse_args()

    # Load all chunks
    print("Loading paper chunks...")
    chunks = load_chunks()
    all_titles = get_paper_titles(chunks)

    # Count chunks per title
    chunk_counts = {}
    for c in chunks:
        title = c["metadata"].get("title", "unknown")
        chunk_counts[title] = chunk_counts.get(title, 0) + 1

    # Get already-extracted papers
    EXTRACTIONS_DIR.mkdir(exist_ok=True)
    already_extracted = get_extracted_papers(EXTRACTIONS_DIR)

    # Filter to unprocessed papers
    to_process = []
    start_found = args.start_from is None

    for title in all_titles:
        # Skip non-papers
        if title in SKIP_TITLES:
            continue

        # Skip papers with too few chunks
        if chunk_counts.get(title, 0) < args.min_chunks:
            continue

        # Handle --start-from
        if not start_found:
            if args.start_from.lower() in title.lower():
                start_found = True
            else:
                continue

        # Skip already extracted
        if title in already_extracted:
            continue

        to_process.append(title)

    # Apply limit
    if args.limit > 0:
        to_process = to_process[:args.limit]

    # Report
    print(f"\nCorpus: {len(all_titles)} papers, {len(chunks):,} chunks")
    print(f"Already extracted: {len(already_extracted)}")
    print(f"To process: {len(to_process)}")
    print(f"Backend: {args.backend}")

    if not to_process:
        print("\nAll papers have been extracted!")
        if args.build_graph:
            _build_graph()
        return

    # Dry-run mode
    if args.dry_run:
        print(f"\nPapers to extract:\n")
        for i, title in enumerate(to_process, 1):
            count = chunk_counts.get(title, 0)
            print(f"  {i:3d}. [{count:3d} chunks] {title}")
        print(f"\nRun without --dry-run to start extraction.")
        return

    # Process papers
    print(f"\n{'='*60}")
    print(f"Starting extraction ({len(to_process)} papers)")
    print(f"{'='*60}")

    results = {"success": 0, "failed": 0, "skipped": 0}
    start_time = time.time()

    for i, title in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] ", end="")

        try:
            result = extract_paper(
                title, chunks, backend=args.backend, verbose=True
            )

            # Save extraction
            output_path = EXTRACTIONS_DIR / f"{result['paper']}_extraction.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Saved to: {output_path}")
            results["success"] += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  FAILED: {e}")
            results["failed"] += 1

        # Rate limiting
        if i < len(to_process) and args.delay > 0:
            time.sleep(args.delay)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Extraction Complete")
    print(f"{'='*60}")
    print(f"  Successful: {results['success']}")
    print(f"  Failed:     {results['failed']}")
    print(f"  Time:       {elapsed:.0f}s ({elapsed/max(results['success'],1):.1f}s/paper)")

    # Build graph if requested
    if args.build_graph:
        _build_graph()


def _build_graph():
    """Rebuild concept graph from all extractions."""
    print(f"\n{'='*60}")
    print(f"Building Concept Graph")
    print(f"{'='*60}")

    graph = load_graph(CONCEPT_GRAPH_PATH)
    extraction_files = sorted(EXTRACTIONS_DIR.glob("*_extraction.json"))

    for path in extraction_files:
        try:
            with open(path) as f:
                extraction = json.load(f)
            graph = integrate_extraction(graph, extraction)
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    save_graph(graph, CONCEPT_GRAPH_PATH)
    print(f"\nGraph saved to: {CONCEPT_GRAPH_PATH}")
    print_stats(graph)


if __name__ == "__main__":
    main()
