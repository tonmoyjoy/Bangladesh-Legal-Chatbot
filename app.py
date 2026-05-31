"""
app.py — Bangladesh Legal Chatbot · Streamlit Frontend
═══════════════════════════════════════════════════════
Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

# ── Page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="বাংলাদেশ আইন সহায়তা | Bangladesh Legal Aid",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Source+Sans+3:wght@300;400;600&display=swap');

    /* Global */
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

    /* Background */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a2744 0%, #0d2137 50%, #0a1628 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "⚖️";
        position: absolute;
        right: 24px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 64px;
        opacity: 0.15;
    }
    .hero-title {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0c040;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin: 0;
        font-style: italic;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(240,192,64,0.15);
        border: 1px solid rgba(240,192,64,0.3);
        color: #f0c040;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 10px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Chat messages */
    .msg-user {
        background: #1c2b3a;
        border: 1px solid #2d4a6b;
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin: 8px 0 8px 10%;
        color: #cdd9e5;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .msg-assistant {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #f0c040;
        border-radius: 4px 12px 12px 12px;
        padding: 14px 18px;
        margin: 8px 10% 8px 0;
        color: #e6edf3;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .msg-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
        opacity: 0.6;
    }
    .label-user      { color: #58a6ff; }
    .label-assistant { color: #f0c040; }

    /* Source cards */
    .source-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #8b949e;
    }
    .source-card strong { color: #58a6ff; }

    /* Input area */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #f0c040 !important;
        box-shadow: 0 0 0 3px rgba(240,192,64,0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f0c040, #d4a017) !important;
        color: #0d1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.03em !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* Divider */
    hr { border-color: #21262d !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #f0c040 !important; }

    /* Warning / info boxes */
    .stAlert { border-radius: 8px !important; font-size: 0.88rem !important; }

    /* Scrollbar */
    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Import project modules (after path is set) ────────────────────────────────
from config import cfg
from utils.embedder import vectorstore_exists

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # {role, content, sources}
if "chain_ready" not in st.session_state:
    st.session_state.chain_ready = False


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    if cfg.GROQ_API_KEY:
        st.success("✅ Using server API key from Streamlit Secrets")
        with st.expander("Use your own key for this session", expanded=False):
            groq_key = st.text_input(
                "Custom Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Optional override for your current session only.",
            )
            if groq_key:
                cfg.GROQ_API_KEY = groq_key
    else:
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Paste your key if no server key is configured.",
        )
        if groq_key:
            cfg.GROQ_API_KEY = groq_key

    st.markdown("---")
    st.markdown("### 📂 Vector Store Status")

    db_exists = vectorstore_exists(cfg.CHROMA_DIR)
    if db_exists:
        st.success("✅ Vector store found")
    else:
        st.error("❌ No vector store found — run `python ingest.py`")

    st.markdown("---")
    st.markdown("### 🔧 RAG Settings")

    top_k = st.slider(
        "Chunks to retrieve (Top K)", min_value=1, max_value=15,
        value=cfg.TOP_K, step=1,
        help="More chunks = broader context but slower response."
    )
    cfg.TOP_K = top_k

    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
        index=0 if cfg.GROQ_MODEL not in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"] else ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"].index(cfg.GROQ_MODEL),
    )
    cfg.GROQ_MODEL = model_choice

    st.markdown("---")
    show_sources = st.toggle("Show source chunks", value=True)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "**Stack:** PyMuPDF · sentence-transformers · ChromaDB · "
        "Groq + LLaMA 3 · LangChain · Streamlit\n\n"
        "_All free. Runs locally._"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — Hero Banner
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">বাংলাদেশ আইন সহায়তা চ্যাটবট</p>
        <p class="hero-subtitle">Bangladesh Legal Aid Chatbot — Powered by RAG + LLaMA 3</p>
        <span class="hero-badge">⚡ Free · Local · Open-Source</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Guard: no vector store ────────────────────────────────────────────────────
if not db_exists:
    st.warning(
        "**Vector store not found.** "
        "Place your Bangladesh law PDFs in the `data/` folder, then run:\n"
        "```bash\npython ingest.py\n```"
    )
    st.info(
        "**Where to get Bangladesh law PDFs?**\n"
        "- [Bangladesh National Legal Aid Services](http://www.nlaso.gov.bd)\n"
        "- [Ministry of Law, Justice & Parliamentary Affairs](http://www.minlaw.gov.bd)\n"
        "- [Laws of Bangladesh (BanglaLegal)](http://bdlaws.minlaw.gov.bd)"
    )
    st.stop()

# ── Guard: no API key ─────────────────────────────────────────────────────────
if not cfg.GROQ_API_KEY:
    st.warning(
        "**Groq API key missing.**  "
        "Paste your free key in the sidebar.  "
        "Get one at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# ── Load RAG chain (once per session) ─────────────────────────────────────────
if not st.session_state.chain_ready:
    with st.spinner("⚙️ Loading RAG chain…"):
        try:
            from rag_chain import get_chain
            get_chain()
            st.session_state.chain_ready = True
        except Exception as exc:
            st.error(f"Failed to load RAG chain: {exc}")
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_QUESTIONS = [
    "What are the fundamental rights guaranteed under the Bangladesh Constitution?",
    "What is the punishment for theft under the Penal Code?",
    "Explain the laws related to land acquisition in Bangladesh.",
    "What are the rights of an arrested person under Bangladesh law?",
    "How is child custody decided under Muslim family law in Bangladesh?",
    "What are the provisions for bail under the Code of Criminal Procedure?",
]

if not st.session_state.messages:
    st.markdown("#### 💡 Try a sample question:")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.messages.append(
                    {"role": "user", "content": q, "sources": []}
                )
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'<div class="msg-label label-user">You</div>{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-assistant">'
            f'<div class="msg-label label-assistant">⚖️ Legal Assistant</div>'
            f'{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Show sources
        if show_sources and msg.get("sources"):
            with st.expander("📚 Source Chunks Used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.metadata if hasattr(src, "metadata") else {}
                    fname = meta.get("filename", "Unknown")
                    page  = meta.get("page", "?")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>[{i}] {fname} — Page {page}</strong><br>'
                        f'{src.page_content[:300]}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Ask a legal question about Bangladesh law…",
        key="user_input",
        label_visibility="collapsed",
        placeholder="e.g. What are the grounds for divorce under Muslim law in Bangladesh?",
    )

with col_btn:
    send_clicked = st.button("Ask ⚖️", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY HANDLING
# ══════════════════════════════════════════════════════════════════════════════
def run_query(question: str) -> None:
    """Execute the RAG query and update session state."""
    from rag_chain import query_with_sources

    # Add user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )

    with st.spinner("🔍 Searching legal documents and generating answer…"):
        try:
            result = query_with_sources(question)
            answer  = result["answer"]
            sources = result["sources"]
        except Exception as exc:
            answer  = f"⚠️ Error: {exc}"
            sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()


if send_clicked and user_input.strip():
    run_query(user_input.strip())
elif user_input.strip() and user_input.endswith("\n"):
    # Allow Enter key submission via text_input (single line)
    run_query(user_input.strip())


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center style='color:#30363d;font-size:0.75rem'>"
    "⚖️ For educational purposes only. Not a substitute for professional legal advice."
    "</center>",
    unsafe_allow_html=True,
)
