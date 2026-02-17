#!/usr/bin/env python3
"""
Add a new paper to the AcademicRAG corpus.

Usage:
    python add_paper.py path/to/paper.pdf "Author Name" "Paper Title" "2024"

After adding, click "Reindex" in the Streamlit app to update the vector store.
"""
import json
import re
import sys
from pathlib import Path

from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from PDF, returns list of (page_num, text)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks with sentence-boundary awareness."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 100]


def create_paper_id(author: str, year: str, title: str) -> str:
    """Create a paper ID from metadata."""
    author_clean = (
        re.sub(r"[^a-zA-Z\s]", "", author).split()[0] if author else "Unknown"
    )
    title_words = re.sub(r"[^a-zA-Z\s]", "", title).split()[:3]
    title_clean = "_".join(title_words) if title_words else "Paper"
    return f"{author_clean}_{year}_{title_clean}"


def add_paper(pdf_path: str, author: str, title: str, year: str):
    """Add a paper to the chunks file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    papers_dir = Path(__file__).parent / "papers"
    chunks_file = papers_dir / "paper_chunks.json"

    # Load existing chunks
    if chunks_file.exists():
        with open(chunks_file) as f:
            chunks = json.load(f)
    else:
        chunks = []

    # Extract text
    print(f"Extracting text from: {pdf_path.name}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"  Found {len(pages)} pages with text")

    paper_id = create_paper_id(author, year, title)

    # Check for duplicates
    existing_ids = {c["id"].rsplit("_", 2)[0] for c in chunks}
    if paper_id in existing_ids:
        print(f"  Warning: Paper '{paper_id}' already exists. Skipping.")
        return

    # Create chunks
    new_chunks = []
    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text)
        for i, chunk_text_content in enumerate(page_chunks):
            chunk_id = f"{paper_id}_p{page_num}_c{i}"
            new_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text_content,
                    "metadata": {
                        "source": pdf_path.name,
                        "author": author,
                        "title": title,
                        "year": year,
                        "page": str(page_num),
                    },
                }
            )

    print(f"  Created {len(new_chunks)} chunks")

    chunks.extend(new_chunks)
    with open(chunks_file, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"  Saved to: {chunks_file}")
    print(f"\nPaper added! Now click 'Reindex' in the Streamlit app.")


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        print("\nExample:")
        print(
            '  python add_paper.py papers/NewPaper.pdf "Smith Jones" "My Paper Title" "2024"'
        )
        sys.exit(1)

    add_paper(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
