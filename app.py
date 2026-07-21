import os
import streamlit as st
from dotenv import load_dotenv
from rag_utils import (
    load_embedder,
    load_local_generator,
    extract_pages,
    chunk_pages,
    build_index,
    retrieve,
    generate_answer,
    GROQ_DEFAULT_MODEL,
)

load_dotenv()  # reads GROQ_API_KEY from a local .env file, if present

st.set_page_config(page_title="AI PDF Chat (RAG)", page_icon="📄", layout="wide")

st.title("📄 AI PDF Chat")
st.caption(
    "Upload a PDF, ask questions about it, and get answers grounded in the "
    "document — with the source page(s) cited. Uses Groq's free hosted LLM "
    "if you provide a key, or falls back to a small local model with no key needed."
)


def get_configured_groq_key():
    """Looks for a Groq key in, in order: environment variables (including
    ones loaded from a local .env file), then Streamlit secrets
    (.streamlit/secrets.toml locally, or the Secrets manager on Streamlit
    Community Cloud). Returns None if not found anywhere.
    """
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Load models once per session (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return load_embedder()


@st.cache_resource(show_spinner="Loading local fallback model (first run only, ~30-60s)...")
def get_local_generator():
    return load_local_generator()


embedder = get_embedder()

# ---------------------------------------------------------------------------
# Sidebar: upload + answer engine
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.header("2. Answer engine")

    configured_key = get_configured_groq_key()
    if configured_key:
        groq_api_key = configured_key
        st.success(f"Using Groq ({GROQ_DEFAULT_MODEL}).")
    else:
        # No key found automatically — offer the manual box as a fallback
        # only in this case, so it's never shown once .env/secrets are set up.
        groq_api_key = st.text_input(
            "Groq API key (optional)",
            type="password",
            help="Free key from console.groq.com. Leave blank to use the local model instead.",
        )
        if groq_api_key:
            st.success(f"Using Groq ({GROQ_DEFAULT_MODEL}) for answers.")
        else:
            st.info("No key found — using the local Flan-T5 model (free, lower quality).")

    engine = "groq" if groq_api_key else "local"

    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Extract text per page (PyMuPDF)\n"
        "2. Chunk + embed (MiniLM)\n"
        "3. Retrieve top-k chunks (FAISS)\n"
        "4. Generate a grounded answer (Groq LLM or local Flan-T5)\n"
        "5. Cite the source page(s) as text"
    )

if uploaded_file is None:
    st.info("👈 Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()

# ---------------------------------------------------------------------------
# Build (or reuse) the index for this specific file
# ---------------------------------------------------------------------------
if "indexed_file_name" not in st.session_state or st.session_state.indexed_file_name != uploaded_file.name:
    with st.spinner("Reading and indexing the PDF..."):
        pages = extract_pages(pdf_bytes)
        chunks = chunk_pages(pages)
        if not chunks:
            st.error("Couldn't extract any text from this PDF (it may be a scanned image without OCR).")
            st.stop()
        index = build_index(chunks, embedder)

    st.session_state.indexed_file_name = uploaded_file.name
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.chat_history = []

    st.success(f"Indexed {len(pages)} pages / {len(chunks)} chunks from **{uploaded_file.name}**")

chunks = st.session_state.chunks
index = st.session_state.index

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
st.header("3. Ask a question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask something about the document...")

if question:
    with st.spinner("Retrieving relevant passages and generating an answer..."):
        retrieved = retrieve(question, chunks, index, embedder, top_k=top_k)

        local_generator = None if engine == "groq" else get_local_generator()
        result = generate_answer(
            question, retrieved,
            engine=engine,
            local_generator=local_generator,
            groq_api_key=groq_api_key,
        )

    st.session_state.chat_history.append({
        "question": question,
        "engine": engine,
        "result": result,
        "retrieved": retrieved,
    })

# Render chat history (most recent first)
for turn in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        result = turn["result"]
        st.markdown(result["answer"])

        caption_parts = [f"Engine: {turn['engine']}"]
        if result["pages"]:
            caption_parts.append("Source: page(s) " + ", ".join(str(p) for p in result["pages"]))
        st.caption(" · ".join(caption_parts))

        with st.expander("🔍 All retrieved chunks (for debugging / transparency)"):
            for r in turn["retrieved"]:
                st.markdown(f"- **p.{r['page']}** (score {r['score']:.2f}): {r['text'][:200]}...")
