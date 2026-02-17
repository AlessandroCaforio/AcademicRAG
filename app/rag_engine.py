"""
AcademicRAG Engine — Core retrieval and generation logic.

Supports three LLM backends:
  1. Ollama (local, free, private)
  2. Claude API (requires ANTHROPIC_API_KEY)
  3. Claude Code CLI (uses your MAX subscription — no API costs)
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import ollama
from chromadb.utils import embedding_functions

from app.config import (
    ANTHROPIC_API_KEY,
    CHROMA_COLLECTION_NAME,
    CLAUDE_CODE_PATH,
    CLAUDE_MODEL,
    CONCEPT_GRAPH_PATH,
    CONCEPT_PROMPT,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PAPER_CHUNKS_PATH,
    SYSTEM_PROMPT,
    TOP_K_RESULTS,
    VECTORS_DIR,
)


class RAGEngine:
    """Main RAG engine for academic literature Q&A."""

    def __init__(self, use_claude: bool = False, use_claude_code: bool = False):
        self.use_claude = use_claude and bool(ANTHROPIC_API_KEY)
        self.use_claude_code = use_claude_code

        # Initialize embedding model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Initialize ChromaDB (persistent, local)
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(VECTORS_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "Academic paper chunks"},
        )

        # Load concept graph for enhanced retrieval (optional)
        self.concepts = self._load_concepts()

        # Initialize Claude API client if needed
        if self.use_claude:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Indexing ──────────────────────────────────────────────────────

    def _load_concepts(self) -> Dict:
        """Load concept graph for context enhancement (optional)."""
        if CONCEPT_GRAPH_PATH.exists():
            with open(CONCEPT_GRAPH_PATH) as f:
                return json.load(f)
        return {}

    def index_papers(self, force_reindex: bool = False) -> int:
        """Index paper chunks into the vector store."""
        if self.collection.count() > 0 and not force_reindex:
            return self.collection.count()

        # Clear existing collection on reindex
        if force_reindex and self.collection.count() > 0:
            self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )

        if not PAPER_CHUNKS_PATH.exists():
            return 0

        with open(PAPER_CHUNKS_PATH) as f:
            chunks = json.load(f)

        # Batch insert for performance
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Retrieve relevant chunks via semantic search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                }
            )
        return retrieved

    def get_relevant_concepts(self, query: str, max_concepts: int = 5) -> List[Dict]:
        """Find concepts from the knowledge graph relevant to the query."""
        if not self.concepts or "concepts" not in self.concepts:
            return []

        query_lower = query.lower()
        relevant = []

        for concept_id, concept in self.concepts["concepts"].items():
            label = concept.get("label", "").lower()
            definition = concept.get("definition", "").lower()

            if any(term in query_lower for term in label.split()) or any(
                term in definition for term in query_lower.split() if len(term) > 3
            ):
                relevant.append(
                    {
                        "id": concept_id,
                        "label": concept.get("label"),
                        "definition": concept.get("definition"),
                        "type": concept.get("type"),
                        "papers": concept.get("mentioned_in", []),
                    }
                )

        return relevant[:max_concepts]

    # ── Context Building ──────────────────────────────────────────────

    def _build_context(self, retrieved: List[Dict], concepts: List[Dict]) -> str:
        """Build context string from retrieved chunks and concepts."""
        parts = ["## Relevant Paper Excerpts\n"]

        for i, chunk in enumerate(retrieved, 1):
            meta = chunk["metadata"]
            parts.append(
                f"### [{i}] {meta.get('author', 'Unknown')} "
                f"({meta.get('year', 'n.d.')}) - "
                f"{meta.get('title', 'Untitled')}, p.{meta.get('page', '?')}\n"
                f"{chunk['text'][:1500]}...\n"
            )

        if concepts:
            parts.append("\n## Related Concepts from Literature\n")
            for c in concepts:
                parts.append(
                    f"- **{c['label']}** ({c['type']}): {c['definition']}\n"
                )

        return "\n".join(parts)

    # ── Generation (3 backends) ───────────────────────────────────────

    def generate_ollama(self, query: str, context: str) -> str:
        """Generate response using Ollama (local, free)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0.3},
            )
            return response["message"]["content"]
        except Exception as e:
            return (
                f"Error with Ollama: {e}\n\n"
                "Make sure Ollama is running: `ollama serve`"
            )

    def generate_claude(self, query: str, context: str) -> str:
        """Generate response using Claude API."""
        try:
            response = self.anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            return f"Error with Claude API: {e}"

    def generate_claude_code(self, query: str, context: str) -> str:
        """Generate response using Claude Code CLI (uses MAX subscription)."""
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Based on the following context from academic papers, answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a comprehensive answer citing the relevant papers."
        )

        try:
            result = subprocess.run(
                [CLAUDE_CODE_PATH, "--print", "-p", full_prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd="/tmp",  # Neutral directory to avoid picking up project configs
            )

            if result.returncode == 0:
                return self._clean_claude_output(result.stdout.strip())
            else:
                error_msg = result.stderr or "Unknown error"
                return f"Claude Code error: {error_msg}"

        except subprocess.TimeoutExpired:
            return "Claude Code timed out. The query may be too complex."
        except FileNotFoundError:
            return (
                f"Claude Code not found at '{CLAUDE_CODE_PATH}'. "
                "Install it from https://claude.ai/download"
            )
        except Exception as e:
            return f"Error calling Claude Code: {e}"

    def _clean_claude_output(self, text: str) -> str:
        """Clean Claude Code output (ANSI codes, file references, status lines)."""
        # Remove ANSI escape codes
        text = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)
        # Remove file reference badges
        text = re.compile(
            r"`[A-Za-z0-9_\-\.]+\.(md|bib|txt|json|py|rtf|tex|csv)`"
        ).sub("", text)
        # Remove bare file path lines
        text = re.compile(
            r"^[\s]*[\./~]?[A-Za-z0-9_\-/]+\."
            r"(md|bib|txt|json|py|rtf|tex|csv)[\s]*$",
            re.MULTILINE,
        ).sub("", text)
        # Remove status messages
        text = re.compile(
            r"^(Reading|Loading|Analyzing|Processing).*\.{3}.*$",
            re.MULTILINE | re.IGNORECASE,
        ).sub("", text)
        # Collapse blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_valid_response(self, text: str) -> bool:
        """Check if the response contains actual content."""
        if not text or len(text) < 50:
            return False
        return len(re.findall(r"\b\w+\b", text)) > 10

    # ── Main Query ────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = TOP_K_RESULTS) -> Tuple[str, List[Dict]]:
        """
        Full RAG pipeline: retrieve → enrich with concepts → generate.

        Returns: (answer, sources)
        """
        retrieved = self.retrieve(question, top_k=top_k)
        concepts = self.get_relevant_concepts(question)
        context = self._build_context(retrieved, concepts)

        # Generate — priority: Claude Code > Claude API > Ollama
        answer = None
        if self.use_claude_code:
            answer = self.generate_claude_code(question, context)
            if not self._is_valid_response(answer):
                print("Claude Code returned invalid response, falling back to Ollama")
                answer = self.generate_ollama(question, context)
                if not answer.startswith("Error"):
                    answer = f"*[Fallback to Ollama]*\n\n{answer}"
        elif self.use_claude:
            answer = self.generate_claude(question, context)
        else:
            answer = self.generate_ollama(question, context)

        return answer, retrieved

    def get_stats(self) -> Dict:
        """Get statistics about the indexed corpus."""
        if self.use_claude_code:
            llm = "Claude Code (MAX)"
        elif self.use_claude:
            llm = f"Claude API ({CLAUDE_MODEL})"
        else:
            llm = f"Ollama ({OLLAMA_MODEL})"

        return {
            "total_chunks": self.collection.count(),
            "total_concepts": len(self.concepts.get("concepts", {})),
            "total_papers": len(self.concepts.get("papers", {})),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": llm,
        }


# ── Singleton ─────────────────────────────────────────────────────────

_engine: Optional[RAGEngine] = None


def get_engine(use_claude: bool = False, use_claude_code: bool = False) -> RAGEngine:
    """Get or create RAG engine singleton."""
    global _engine
    if (
        _engine is None
        or _engine.use_claude != use_claude
        or _engine.use_claude_code != use_claude_code
    ):
        _engine = RAGEngine(use_claude=use_claude, use_claude_code=use_claude_code)
    return _engine
