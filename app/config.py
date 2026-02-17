"""
AcademicRAG - Configuration

All settings can be overridden via environment variables or .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PAPERS_DIR = BASE_DIR / "papers"
DATA_DIR = BASE_DIR / "data"
VECTORS_DIR = DATA_DIR / "vectors"

# Data files
PAPER_CHUNKS_PATH = PAPERS_DIR / "paper_chunks.json"
CONCEPT_GRAPH_PATH = BASE_DIR / "concept_graph.json"  # optional

# ── Vector Store ───────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "academic_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, high-quality sentence embeddings

# ── LLM Settings ──────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Claude Code CLI (uses MAX subscription — no API costs)
CLAUDE_CODE_PATH = os.getenv("CLAUDE_CODE_PATH", "claude")

# ── RAG Settings ──────────────────────────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── UI Settings ───────────────────────────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "AcademicRAG")
APP_ICON = os.getenv("APP_ICON", "📚")

# ── System Prompts ────────────────────────────────────────────────────
# Override SYSTEM_PROMPT via environment variable for your specific thesis/project
DEFAULT_SYSTEM_PROMPT = """You are a research assistant helping with academic literature review.

You have access to a corpus of academic papers that have been indexed and chunked.

When answering questions:
1. Base your answers on the retrieved paper excerpts
2. Cite specific papers when making claims (e.g., "According to Author (Year)...")
3. Be precise about methodology and findings
4. Acknowledge when information is not in the available sources
5. Use academic tone and precise language"""

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

CONCEPT_PROMPT = """The following concepts from the literature may be relevant:
{concepts}

Use these to provide additional context when relevant."""
