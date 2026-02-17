"""
AcademicRAG — Streamlit Interface
A local NotebookLM-style app for exploring and querying academic literature.
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_engine import get_engine, RAGEngine
from app.config import APP_TITLE, APP_ICON, TOP_K_RESULTS, ANTHROPIC_API_KEY
from app.academic_features import CitationGenerator, create_academic_features

# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stChatMessage { background-color: #f8f9fa; border-radius: 10px;
                     padding: 1rem; margin-bottom: 0.5rem; }
    .source-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; padding: 1rem; border-radius: 8px;
                   margin-bottom: 0.5rem; font-size: 0.85rem; }
    .source-card h4 { margin: 0 0 0.5rem 0; font-size: 0.95rem; }
    .stat-card { background: #f1f3f4; padding: 1rem; border-radius: 8px;
                 text-align: center; }
    .stat-number { font-size: 2rem; font-weight: bold; color: #1a73e8; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ─────────────────────────────────────────────────────


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "indexed" not in st.session_state:
        st.session_state.indexed = False


# ── Sidebar ───────────────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.markdown("## Settings")

        # Model selection
        st.markdown("### LLM Model")
        model_choice = st.radio(
            "Select model",
            options=["claude_code", "ollama", "claude_api"],
            format_func=lambda x: {
                "claude_code": "Claude Code (MAX)",
                "ollama": "Ollama (Local)",
                "claude_api": "Claude API",
            }[x],
            index=0,
            help="Claude Code uses your MAX subscription — no API costs!",
        )

        use_claude_code = model_choice == "claude_code"
        use_claude = model_choice == "claude_api"

        if model_choice == "claude_code":
            st.success("Uses your Claude MAX subscription!")
        elif model_choice == "claude_api":
            if not ANTHROPIC_API_KEY:
                st.warning("Add ANTHROPIC_API_KEY to .env")
        else:
            st.info("Make sure Ollama is running")

        top_k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=TOP_K_RESULTS,
            help="How many paper excerpts to retrieve per query",
        )

        st.divider()

        # Index management
        st.markdown("## Index")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reindex", use_container_width=True):
                with st.spinner("Indexing papers..."):
                    engine = get_engine(
                        use_claude=use_claude, use_claude_code=use_claude_code
                    )
                    count = engine.index_papers(force_reindex=True)
                    st.session_state.indexed = True
                    st.success(f"Indexed {count} chunks!")
        with col2:
            if st.button("Stats", use_container_width=True):
                engine = get_engine(
                    use_claude=use_claude, use_claude_code=use_claude_code
                )
                st.json(engine.get_stats())

        if st.session_state.engine:
            stats = st.session_state.engine.get_stats()
            st.markdown("### Current Index")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Chunks", stats["total_chunks"])
            with cols[1]:
                st.metric("Papers", stats["total_papers"])
            st.caption(f"Model: {stats['llm_model']}")

        st.divider()

        # Quick queries
        st.markdown("## Quick Queries")
        queries = [
            "What are the main findings?",
            "Compare the methodologies used",
            "What are the key debates?",
            "What research gaps exist?",
            "Summarize the theoretical frameworks",
        ]
        for q in queries:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                return use_claude, use_claude_code, top_k, q

        return use_claude, use_claude_code, top_k, None


# ── Source Rendering ──────────────────────────────────────────────────


def render_source(source: dict, idx: int):
    meta = source.get("metadata", {}) or {}
    score = source.get("score", 0) or 0
    text = source.get("text", "") or ""

    author = meta.get("author", "Unknown") or "Unknown"
    year = meta.get("year", "n.d.") or "n.d."
    page = meta.get("page", "?") or "?"
    title = meta.get("title", "Untitled") or "Untitled"

    with st.expander(f"[{idx}] {author} ({year}) - p.{page}", expanded=False):
        st.markdown(f"**{title}**")
        st.markdown(f"*Relevance: {score:.2%}*")
        st.divider()
        if text:
            st.markdown(text[:1000] + ("..." if len(text) > 1000 else ""))
        else:
            st.warning("No text content available.")


# ── Academic Tools ────────────────────────────────────────────────────


def render_academic_tools(engine):
    st.markdown("---")
    st.markdown("## Academic Tools")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Citation Generator",
            "Literature Drafter",
            "Paper Comparison",
            "Claim Extractor",
            "Research Gaps",
        ]
    )

    features = create_academic_features(engine)

    with tab1:
        st.markdown("### Generate Citations")
        st.markdown("Extract citations from your most recent query sources.")
        if st.session_state.messages and any(
            m.get("sources") for m in st.session_state.messages
        ):
            for msg in reversed(st.session_state.messages):
                if msg.get("sources"):
                    sources = msg["sources"]
                    break

            col1, col2 = st.columns(2)
            with col1:
                cite_style = st.selectbox(
                    "Citation style",
                    ["latex", "apa"],
                    format_func=lambda x: (
                        "LaTeX (\\citep{})" if x == "latex" else "APA"
                    ),
                )
            with col2:
                export_format = st.selectbox(
                    "Export format",
                    ["bibtex", "apa"],
                    format_func=lambda x: "BibTeX" if x == "bibtex" else "APA",
                )

            if st.button("Generate Citations", key="gen_cite"):
                cg = features["citation_generator"]
                citations = cg.extract_citations(sources)
                st.markdown("#### Inline Citations")
                for cite in cg.generate_inline_citations(cite_style):
                    st.code(
                        cite, language="latex" if cite_style == "latex" else None
                    )
                st.markdown("#### Bibliography")
                biblio = cg.generate_bibliography(export_format)
                st.code(
                    biblio, language="bibtex" if export_format == "bibtex" else None
                )
                st.download_button(
                    "Download Bibliography",
                    biblio,
                    file_name=(
                        "bibliography.bib"
                        if export_format == "bibtex"
                        else "bibliography.txt"
                    ),
                    mime="text/plain",
                )
        else:
            st.info("Ask a question first to get sources for citation generation.")

    with tab2:
        st.markdown("### Draft Literature Review")
        lit_topic = st.text_input(
            "Topic",
            placeholder="e.g., automation and wage inequality",
            key="lit_topic",
        )
        section_title = st.text_input(
            "Section title", value="Literature Review", key="section_title"
        )
        if st.button("Draft", key="draft_lit"):
            if lit_topic:
                with st.spinner("Drafting..."):
                    draft, sources = features["literature_drafter"].draft_section(
                        lit_topic, section_title
                    )
                st.markdown(draft)
                with st.expander("Sources"):
                    for i, src in enumerate(sources, 1):
                        render_source(src, i)
                st.download_button(
                    "Download Draft",
                    draft,
                    file_name="literature_review_draft.md",
                    mime="text/markdown",
                )

    with tab3:
        st.markdown("### Compare Papers")
        dimension = st.selectbox(
            "Dimension",
            [
                "all",
                "methodology",
                "data_sources",
                "main_findings",
                "theoretical_framework",
                "limitations",
                "policy_implications",
            ],
            format_func=lambda x: (
                "All Dimensions" if x == "all" else x.replace("_", " ").title()
            ),
        )
        if st.button("Compare", key="compare"):
            with st.spinner("Comparing..."):
                comparison, sources = features["paper_comparator"].compare_papers(
                    dimension=dimension
                )
            st.markdown(comparison)
            with st.expander("Sources"):
                for i, src in enumerate(sources, 1):
                    render_source(src, i)

    with tab4:
        st.markdown("### Extract & Analyze Claims")
        claim_topic = st.text_input(
            "Topic",
            placeholder="e.g., effect of automation on voting behavior",
            key="claim_topic",
        )
        claim_type = st.radio(
            "Analysis",
            ["extract", "contradictions"],
            format_func=lambda x: (
                "Extract Key Claims" if x == "extract" else "Find Contradictions"
            ),
            horizontal=True,
        )
        if st.button("Analyze", key="analyze_claims") and claim_topic:
            with st.spinner("Analyzing..."):
                extractor = features["claim_extractor"]
                if claim_type == "extract":
                    result, sources = extractor.extract_claims(claim_topic)
                else:
                    result, sources = extractor.find_contradictions(claim_topic)
            st.markdown(result)
            with st.expander("Sources"):
                for i, src in enumerate(sources, 1):
                    render_source(src, i)

    with tab5:
        st.markdown("### Find Research Gaps")
        gap_topic = st.text_input(
            "Topic (optional)", placeholder="Leave blank for general analysis", key="gap_topic"
        )
        if st.button("Find Gaps", key="find_gaps"):
            with st.spinner("Analyzing..."):
                gaps, sources = features["gap_finder"].find_gaps(
                    gap_topic if gap_topic else None
                )
            st.markdown(gaps)
            with st.expander("Sources"):
                for i, src in enumerate(sources, 1):
                    render_source(src, i)


# ── Defense Prep ──────────────────────────────────────────────────────


def render_defense_prep(engine):
    st.markdown("---")
    st.markdown("## Defense Preparation")
    features = create_academic_features(engine)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Generate Defense Questions")
        if st.button("Generate Questions", key="def_questions"):
            with st.spinner("Generating..."):
                questions, sources = features["defense_prep"].generate_questions()
            st.markdown(questions)
            with st.expander("References"):
                for i, src in enumerate(sources, 1):
                    render_source(src, i)

    with col2:
        st.markdown("### Prepare Counterarguments")
        claim = st.text_input(
            "Your claim to defend",
            placeholder="e.g., Automation exposure causes conservative shift",
            key="defense_claim",
        )
        if st.button("Prepare Response", key="def_counter") and claim:
            with st.spinner("Preparing..."):
                response, sources = features[
                    "defense_prep"
                ].prepare_counterarguments(claim)
            st.markdown(response)
            with st.expander("References"):
                for i, src in enumerate(sources, 1):
                    render_source(src, i)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    init_session_state()

    st.markdown(
        f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
        <span style="font-size:2.5rem;">{APP_ICON}</span>
        <div>
            <h1 style="margin:0;">{APP_TITLE}</h1>
            <p style="margin:0;color:#666;">Ask questions about your academic literature</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    use_claude, use_claude_code, top_k, quick_query = render_sidebar()

    # Initialize engine
    engine_changed = (
        st.session_state.engine is None
        or st.session_state.engine.use_claude != use_claude
        or st.session_state.engine.use_claude_code != use_claude_code
    )
    if engine_changed:
        with st.spinner("Initializing RAG engine..."):
            st.session_state.engine = get_engine(
                use_claude=use_claude, use_claude_code=use_claude_code
            )
            if st.session_state.engine.collection.count() == 0:
                count = st.session_state.engine.index_papers()
                st.toast(f"Indexed {count} paper chunks")

    engine = st.session_state.engine

    # Chat area
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("Sources", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            render_source(src, i)

    # Handle quick query
    if quick_query:
        st.session_state.messages.append({"role": "user", "content": quick_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(quick_query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = engine.query(quick_query, top_k=top_k)
                st.markdown(answer)
                with st.expander("Sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        render_source(src, i)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question about your literature..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching literature..."):
                    answer, sources = engine.query(prompt, top_k=top_k)
                if not answer or not answer.strip():
                    st.error("The LLM returned an empty response. Try a different model.")
                else:
                    st.markdown(answer)
                if sources:
                    with st.expander("Sources", expanded=True):
                        for i, src in enumerate(sources, 1):
                            render_source(src, i)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # Academic tools
    with st.expander("Academic Tools", expanded=False):
        render_academic_tools(engine)

    with st.expander("Defense Preparation", expanded=False):
        render_defense_prep(engine)


if __name__ == "__main__":
    main()
