import streamlit as st
from extraction import extract_and_chunk
from ingestion import VectorStore
from generation import Generator
import os
import time
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "dark pyscology.pdf")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Page Config ---
st.set_page_config(
    page_title="📚 Dark Psychology RAG",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 0.95rem;
        opacity: 0.8;
        margin: 0;
    }

    .pipeline-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        transition: transform 0.2s;
    }
    .pipeline-card:hover { transform: translateY(-2px); }
    .pipeline-card .icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .pipeline-card .label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; }
    .pipeline-card .value { font-size: 1.1rem; font-weight: 500; margin-top: 0.3rem; }

    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(0, 200, 117, 0.15);
        color: #00c875;
        border: 1px solid rgba(0, 200, 117, 0.3);
    }
    .status-loading {
        background: rgba(255, 165, 0, 0.15);
        color: #ffa500;
        border: 1px solid rgba(255, 165, 0, 0.3);
    }

    .context-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #a0a0b0;
        max-height: 300px;
        overflow-y: auto;
    }

    .stChatMessage { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1> Dark Psychology — RAG Pipeline</h1>
    <p>Ask questions about the research paper. Powered by ChromaDB + Groq LLaMA 3.3</p>
</div>
""", unsafe_allow_html=True)

# --- Initialize Pipeline (cached) ---
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Extract, chunk, and ingest the PDF. Runs only once."""
    documents = extract_and_chunk(PDF_PATH)
    db = VectorStore()
    db.add_documents(documents)
    return db, len(documents)

# Load with a visual spinner
with st.spinner("📄 Processing PDF and building vector store..."):
    db, chunk_count = load_pipeline()

generator = Generator(GROQ_API_KEY)

# --- Pipeline Status Cards ---
cols = st.columns(4)
card_data = [
    ("📄", "Source", "dark pyscology.pdf"),
    ("✂️", "Chunks", str(chunk_count)),
    ("🗄️", "Vector DB", "ChromaDB"),
    ("🤖", "LLM", "LLaMA 3.3 70B"),
]
for col, (icon, label, value) in zip(cols, card_data):
    col.markdown(f"""
    <div class="pipeline-card">
        <div class="icon">{icon}</div>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- RAG Pipeline Flow ---
with st.expander("🔗 RAG Pipeline Flow", expanded=False):
    flow_cols = st.columns(7)
    steps = ["📄 PDF", "→", "✂️ Chunk", "→", "🗄️ Store", "→", "🤖 Generate"]
    for c, s in zip(flow_cols, steps):
        c.markdown(f"<div style='text-align:center; font-weight:600; font-size:0.95rem;'>{s}</div>", unsafe_allow_html=True)

# --- Chat Interface ---
st.markdown("### 💬 Ask a Question")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("📎 Retrieved Context"):
                st.markdown(f'<div class="context-box">{msg["context"]}</div>', unsafe_allow_html=True)

# Chat input
if user_query := st.chat_input("e.g. What is dark psychology?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching vector store & generating answer..."):
            # Retrieval
            retrieved_chunks = db.query(user_query, n_results=3)
            context = "\n\n".join(retrieved_chunks)

            # Generation
            answer = generator.get_response(context, user_query)

        st.markdown(answer)

        with st.expander("📎 Retrieved Context"):
            st.markdown(f'<div class="context-box">{context}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context
    })

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Info")
    st.markdown(f'<span class="status-badge status-ready">● Ready</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**PDF:** dark pyscology.pdf")
    st.markdown(f"**Chunks:** {chunk_count}")
    st.markdown(f"**Embedding:** Default ChromaDB")
    st.markdown(f"**LLM:** LLaMA 3.3 70B (Groq)")
    st.markdown(f"**Framework:** ROSE")
    st.markdown("---")
    st.markdown("### 🔄 Pipeline Steps")
    st.markdown("""
    1. **Extraction** — PyPDFLoader  
    2. **Chunking** — RecursiveCharacterTextSplitter  
    3. **Ingestion** — ChromaDB vector store  
    4. **Retrieval** — Similarity search (top 3)  
    5. **Generation** — Groq LLaMA 3.3 + ROSE
    """)
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
