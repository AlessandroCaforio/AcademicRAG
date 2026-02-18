# AcademicRAG

A **local-first RAG system** for academic literature review. Built with ChromaDB, Streamlit, and multiple LLM backends. Designed for researchers, graduate students, and anyone working with a corpus of academic papers.

> Originally built for a Master's thesis at Bocconi University on automation and political ideology, then generalized as a reusable tool.

## Features

### Core RAG
- **Semantic search** over your paper corpus using `all-MiniLM-L6-v2` embeddings
- **Knowledge graph integration** — retrieved context is enriched with a concept graph linking ideas across papers
- **Source citations** — every answer shows the exact paper excerpts used, with author, year, and page number

### Three LLM Backends
| Mode | Cost | Privacy | Setup |
|------|------|---------|-------|
| **Ollama** (local) | Free | Full privacy — nothing leaves your machine | `ollama serve` |
| **Claude Code** (MAX) | Included in MAX subscription | Anthropic handles data | Install [Claude Code](https://claude.ai/download) |
| **Claude API** | Pay per token | Anthropic handles data | Add `ANTHROPIC_API_KEY` to `.env` |

### Academic Tools
Six specialized tools beyond basic Q&A:

| Tool | What it does |
|------|-------------|
| **Citation Generator** | Extract citations from sources, export to LaTeX/BibTeX or APA |
| **Literature Review Drafter** | Auto-generate literature review sections with proper citations |
| **Paper Comparator** | Create comparison tables across methodology, findings, theory, etc. |
| **Claim Extractor** | Extract key claims and find contradictions between papers |
| **Research Gap Finder** | Identify unanswered questions and underexplored areas |
| **Defense Prep** | Generate potential defense questions and prepare counterarguments |

### Deep Paper Extraction (LLM-Powered)
Automated structured extraction pipeline that processes each paper and outputs:
- **Constructs** — theoretical concepts with definitions and verbatim quotes
- **Variables** — measured quantities with operationalization details
- **Methods** — empirical approaches with specifics (F-stats, controls, sample sizes)
- **Claims** — key findings with evidence and strength ratings (strong/moderate/weak)

Extractions feed into a **concept knowledge graph** that links ideas across papers with typed edges (causes, correlates, moderates, supports, contradicts).

### Auto-Cataloging (No LLM Required)
The `generate_catalog.py` script builds structured metadata for your entire corpus using pure Python heuristics:
- Abstract extraction from PDF text
- Key findings via signal-phrase detection
- Methodology tagging (shift-share, TWFE, IV, RDD, survey, panel data, ...)
- BibTeX key matching (fuzzy author + year + title overlap)
- Category assignment (labor economics, political behavior, psychology, ...)
- Concept graph linking

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/AlessandroCaforio/AcademicRAG.git
cd AcademicRAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add your papers

```bash
# Add papers one at a time
python add_paper.py papers/MyPaper.pdf "Author Name" "Paper Title" "2024"

# Or drop PDFs into papers/ and add them in bulk
for pdf in papers/*.pdf; do
    python add_paper.py "$pdf" "Unknown" "$(basename "$pdf" .pdf)" "2024"
done
```

### 3. Run

```bash
# Quick start (handles venv + Ollama check)
./run.sh

# Or manually
streamlit run app/main.py
```

Open **http://localhost:8501** in your browser.

### 4. (Optional) Extract structured knowledge

```bash
# Extract a single paper
python extract_paper.py "Paper Title" --backend claude

# Extract ALL unprocessed papers (batch mode)
python run_extraction.py --backend claude

# Build/update the concept graph from extractions
python build_graph.py extractions/

# See what would be processed (no LLM calls)
python run_extraction.py --dry-run
```

### 5. (Optional) Generate catalog

```bash
python generate_catalog.py
# Outputs: paper_catalog.json + paper_catalog.md
```

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1:8b` | Local LLM model |
| `ANTHROPIC_API_KEY` | — | For Claude API mode (optional) |
| `CLAUDE_CODE_PATH` | `claude` | Path to Claude Code CLI |
| `SYSTEM_PROMPT` | (built-in) | Custom system prompt for your research domain |
| `APP_TITLE` | `AcademicRAG` | UI title |
| `TOP_K_RESULTS` | `5` | Number of sources per query |

### Customizing the System Prompt

Set `SYSTEM_PROMPT` in `.env` to tailor responses to your research domain:

```bash
SYSTEM_PROMPT="You are a research assistant for a PhD thesis on climate economics. Base answers on retrieved paper excerpts and cite specific authors."
```

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│  Embedding   │────▶│  ChromaDB Vector  │
│ (MiniLM-L6) │     │     Store         │
└─────────────┘     └──────────────────┘
                            │
                     Top-K chunks
                            │
                            ▼
                    ┌───────────────┐
                    │ Context Builder │◀── Concept Graph (optional)
                    └───────────────┘
                            │
                    Context + Question
                            │
                            ▼
                 ┌─────────────────────┐
                 │   LLM Generation    │
                 │  (Ollama / Claude)   │
                 └─────────────────────┘
                            │
                            ▼
                   Answer + Sources
```

### Key Design Decisions

- **ChromaDB** for the vector store — persistent, local, no server needed
- **all-MiniLM-L6-v2** for embeddings — fast, good quality, runs on CPU
- **Sentence-boundary chunking** with overlap — preserves context across chunk boundaries
- **Concept graph enrichment** — optional knowledge graph adds structured relationships to raw retrieval
- **Three LLM backends** — privacy-first (Ollama) with cloud fallbacks

## Project Structure

```
AcademicRAG/
├── app/
│   ├── config.py              # All settings (env-configurable)
│   ├── main.py                # Streamlit interface
│   ├── rag_engine.py          # Core RAG: indexing, retrieval, generation
│   └── academic_features.py   # 6 academic tools
├── papers/                    # Your PDFs + paper_chunks.json
├── extractions/               # Structured paper extractions (auto-generated)
├── data/vectors/              # ChromaDB storage (auto-generated)
├── add_paper.py               # PDF → chunks ingestion
├── extract_paper.py           # LLM-powered structured extraction (single paper)
├── build_graph.py             # Concept graph builder (from extractions)
├── run_extraction.py          # Batch orchestrator (all papers)
├── generate_catalog.py        # Auto-catalog (no LLM needed)
├── run.sh                     # Quick start script
├── requirements.txt
├── .env.example
└── docs/
    └── ARCHITECTURE.md        # Detailed architecture & concept schema
```

## Example Queries

- "What are the main findings on automation and wages?"
- "Compare the methodologies used across papers"
- "What does Author (2024) argue about X?"
- "Find contradictions between Paper A and Paper B"
- "What research gaps exist in this literature?"
- "Generate defense questions for my thesis"

## Technical Details

| Component | Technology |
|-----------|-----------|
| Embeddings | `all-MiniLM-L6-v2` (384-dim, ~80MB) |
| Vector Store | ChromaDB (persistent, local) |
| Chunking | ~1500 chars, 200 char overlap, sentence-boundary aware |
| Retrieval | Top-K semantic search (cosine similarity) |
| UI | Streamlit |
| LLMs | Ollama / Claude API / Claude Code |

## License

MIT

## Acknowledgments

Built during the Master's thesis *"Automation and Political Ideology: Task Displacement and the Conservative Shift Among Exposed Workers"* at Bocconi University, supervised by Prof. Massimo Anelli.

The concept graph schema and agentic extraction pipeline were inspired by the iterative literature analysis approach described in the [architecture docs](docs/ARCHITECTURE.md).
