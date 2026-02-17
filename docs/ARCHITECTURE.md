# Architecture

## Overview

AcademicRAG is a Retrieval-Augmented Generation system designed specifically for academic literature review. It combines semantic search over paper chunks with an optional concept knowledge graph to provide contextually rich answers grounded in your paper corpus.

## Pipeline

```
PDF Papers
    │
    ▼  (add_paper.py)
paper_chunks.json ─── chunked text with metadata (author, year, title, page)
    │
    ▼  (rag_engine.py → index_papers)
ChromaDB ─── persistent vector store with all-MiniLM-L6-v2 embeddings
    │
    ▼  (rag_engine.py → retrieve)
Top-K Chunks ─── semantically similar to the user's query
    │
    ├── + Concept Graph (optional) ─── structured relationships between ideas
    │
    ▼  (rag_engine.py → _build_context)
Context String ─── formatted paper excerpts + related concepts
    │
    ▼  (rag_engine.py → generate_*)
LLM Response ─── grounded answer citing specific papers
```

## Components

### 1. Paper Ingestion (`add_paper.py`)

Converts PDFs into searchable chunks:

1. **PDF text extraction** via PyPDF2 (page by page)
2. **Chunking** at ~1500 characters with 200-char overlap
3. **Sentence-boundary awareness** — chunks break at periods, not mid-sentence
4. **Metadata tagging** — each chunk carries `author`, `year`, `title`, `page`
5. **Deduplication** — papers are identified by a stem ID to prevent double-indexing

Output: `papers/paper_chunks.json` (flat array of chunk objects)

### 2. Vector Store (ChromaDB)

- **Persistent** — stored in `data/vectors/`, survives restarts
- **Embedding model**: `all-MiniLM-L6-v2` (384-dimensional, fast, CPU-friendly)
- **Batch indexing** — 100 chunks per batch for performance
- **Reindexing** — full rebuild via UI button or `force_reindex=True`

### 3. Concept Graph (Optional)

A JSON knowledge graph linking concepts across papers:

```json
{
  "concepts": {
    "status_threat": {
      "id": "status_threat",
      "type": "construct",
      "label": "Status Threat",
      "definition": "Perceived decline in one's social group's standing",
      "first_introduced_by": "mutz_2018",
      "mentioned_in": ["mutz_2018", "anelli_2019"],
      "operationalizations": [
        {
          "paper": "mutz_2018",
          "measure": "Survey items on perceived decline",
          "page": 4
        }
      ]
    }
  },
  "edges": [
    {
      "source": "automation",
      "target": "task_displacement",
      "type": "causes",
      "papers": ["acemoglu_2022"],
      "evidence": "strong"
    }
  ]
}
```

#### Concept Types
| Type | Description | Example |
|------|------------|---------|
| `construct` | Theoretical concept | Status threat, economic anxiety |
| `variable` | Measured variable | Robot exposure, RTI, vote share |
| `method` | Empirical approach | Shift-share IV, panel FE |
| `claim` | Empirical finding | "Automation increases conservative voting" |

#### Edge Types
| Type | Meaning |
|------|---------|
| `causes` | A causes B |
| `correlates` | Empirical association |
| `moderates` | A moderates B→C |
| `mediates` | A mediates B→C |
| `defines` | Paper defines concept |
| `supports` | Paper supports claim |
| `contradicts` | Paper contradicts claim |
| `extends` | Paper extends prior work |

### 4. RAG Engine (`rag_engine.py`)

The core retrieval and generation logic:

- **Retrieve**: Semantic search via ChromaDB → top-K chunks ranked by cosine similarity
- **Enrich**: Match query against concept graph labels/definitions → add structured context
- **Build context**: Format chunks and concepts into a prompt-ready string
- **Generate**: Send context + question to one of three LLM backends

#### LLM Backend Selection
Priority order: Claude Code (MAX) → Claude API → Ollama

Each backend has automatic error handling with fallback to Ollama if the primary fails.

### 5. Academic Features (`academic_features.py`)

Six specialized tools that compose on top of the RAG engine:

| Tool | How it works |
|------|-------------|
| **CitationGenerator** | Extracts metadata from retrieved sources → formats as BibTeX/APA |
| **LiteratureReviewDrafter** | Sends a structured synthesis prompt → returns organized review |
| **PaperComparator** | Queries for comparison across N dimensions → markdown table |
| **ClaimExtractor** | Prompts for claims with evidence ratings → numbered list |
| **ResearchGapFinder** | Asks for gaps across methodology/data/theory → research directions |
| **DefensePrep** | Generates challenging questions + prepares counterarguments |

### 6. Auto-Cataloging (`generate_catalog.py`)

Builds structured metadata with **zero LLM calls** using pure Python heuristics:

1. **Paper discovery** — groups chunks by PDF stem, deduplicates
2. **Abstract extraction** — regex patterns for "Abstract" sections, fallback to first paragraph
3. **Key findings** — signal-phrase detection ("we find that", "results show", etc.) with Jaccard deduplication
4. **Methodology tagging** — keyword frequency across 9 categories
5. **BibTeX matching** — fuzzy author + year + title overlap against `.bib` file
6. **Category assignment** — keyword-based classification into 6 domains
7. **Concept linking** — label matching against the concept graph

## Configuration

All settings live in `app/config.py` and are overridable via environment variables:

| Setting | Env Var | Default |
|---------|---------|---------|
| Embedding model | — | `all-MiniLM-L6-v2` |
| Ollama model | `OLLAMA_MODEL` | `llama3.1:8b` |
| Claude model | `CLAUDE_MODEL` | `claude-sonnet-4-20250514` |
| Top-K results | `TOP_K_RESULTS` | 5 |
| Chunk size | `CHUNK_SIZE` | 1000 |
| System prompt | `SYSTEM_PROMPT` | (generic academic assistant) |

## Building Your Own Concept Graph

The concept graph is optional but significantly improves retrieval quality. You can build one manually or use an agentic loop:

### Manual Approach
1. Read each paper and identify key concepts
2. Add entries to `concept_graph.json` following the schema above
3. Add edges for relationships between concepts

### Agentic Approach
Process papers chronologically:
1. For each paper, query the RAG for main arguments, definitions, methodology, findings
2. Extract concepts (constructs, variables, methods, claims)
3. Match against existing graph — merge or create new entries
4. Extract relationships and add edges
5. Save incremental snapshots

This approach was used to build the original 75-node, 105-edge graph for the thesis corpus.
