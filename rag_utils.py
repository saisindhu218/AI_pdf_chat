"""
Core RAG (Retrieval-Augmented Generation) pipeline for the PDF Chat app.

Architecture:
    1. Extraction  -> PyMuPDF (fitz) pulls text per page from the uploaded PDF,
                      using block-based extraction for sensible reading order
    2. Chunking    -> text is split along paragraph/word boundaries, tagged
                      with source page number
    3. Embedding   -> sentence-transformers (all-MiniLM-L6-v2) turns each
                      chunk into a vector
    4. Indexing    -> FAISS in-memory index for fast cosine similarity search
    5. Retrieval   -> top-k most relevant chunks for a user question
    6. Generation  -> two interchangeable engines:
                      - "groq" (default if a key is set): sends retrieved
                        chunks to Groq's free-tier hosted LLM (Llama 3.1/3.3)
                        for fast, fluent, grounded answers
                      - "local" (automatic fallback, no key needed): a small
                        local model (Flan-T5-base) reads the same context and
                        writes an answer — lower quality, but works offline
                        with zero setup

Why generation instead of extractive QA: an earlier version of this used
DistilBERT-SQuAD (extractive QA) to pull an exact span out of a chunk. That
works well on Wikipedia-style trivia questions (what it was trained on), but
its confidence scores are not well-calibrated on other document types (like
resumes) — it would confidently return a wrong span at low-30s% confidence
just as often as a correct one. A generative model that actually reads the
context and writes a sentence, explicitly instructed to say when the answer
isn't present, gives far more usable answers.
"""

import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

try:
    from groq import Groq
except ImportError:
    Groq = None

# ---------------------------------------------------------------------------
# Model loading (cached by caller via st.cache_resource so this runs once)
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"


def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def load_local_generator():
    """Loads the local fallback generator (used when no Groq key is set)."""
    return pipeline("text2text-generation", model=GEN_MODEL_NAME, tokenizer=GEN_MODEL_NAME)


# ---------------------------------------------------------------------------
# 1 & 2. PDF extraction + chunking
# ---------------------------------------------------------------------------

def extract_pages(pdf_bytes: bytes):
    """Returns a list of dicts: [{"page": 1, "text": "..."}, ...] (1-indexed).

    Uses block-based extraction instead of raw "text" mode: PyMuPDF's
    get_text("blocks") returns each paragraph/text block with its position,
    which we sort top-to-bottom / left-to-right. This reads far more
    sensibly than the flat "text" mode on documents with sidebars, columns,
    or resume-style layouts, where naive extraction can interleave unrelated
    sections.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]  # text blocks only
        text_blocks.sort(key=lambda b: (round(b[1] / 8), b[0]))  # row bucket, then x
        text = "\n".join(b[4].strip() for b in text_blocks)
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(pages, chunk_size=500):
    """Splits each page's text into chunks along paragraph and word
    boundaries — never mid-word. Paragraphs are grouped together up to
    ~chunk_size chars; an overlong paragraph is split on whitespace only.
    """
    chunks = []
    for p in pages:
        paragraphs = [para.strip() for para in p["text"].split("\n") if para.strip()]

        current, current_len = [], 0

        def flush():
            if current:
                chunks.append({"page": p["page"], "text": " ".join(current)})

        for para in paragraphs:
            if len(para) > chunk_size:
                # overlong paragraph: split on word boundaries only
                flush()
                current, current_len = [], 0
                words, buf, buf_len = para.split(), [], 0
                for w in words:
                    if buf_len + len(w) + 1 > chunk_size and buf:
                        chunks.append({"page": p["page"], "text": " ".join(buf)})
                        buf, buf_len = [], 0
                    buf.append(w)
                    buf_len += len(w) + 1
                if buf:
                    chunks.append({"page": p["page"], "text": " ".join(buf)})
                continue

            if current_len + len(para) > chunk_size and current:
                flush()
                current, current_len = [], 0

            current.append(para)
            current_len += len(para)

        flush()

    return chunks


# ---------------------------------------------------------------------------
# 3 & 4. Embedding + FAISS index
# ---------------------------------------------------------------------------

def build_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# 5. Retrieval
# ---------------------------------------------------------------------------

def retrieve(question, chunks, index, embedder, top_k=4):
    q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# 6. Generation: read the retrieved chunks, write a grounded answer
# ---------------------------------------------------------------------------

_NOT_FOUND_PHRASE = "I couldn't find this in the document."


def _build_context(retrieved_chunks, max_chunks=4, max_chars_per_chunk=450):
    used = retrieved_chunks[:max_chunks]
    context_parts = [f"[Page {c['page']}] {c['text'][:max_chars_per_chunk]}" for c in used]
    return "\n\n".join(context_parts), sorted({c["page"] for c in used})


def generate_answer_local(question, retrieved_chunks, generator, max_chunks=4, max_chars_per_chunk=450):
    """Local fallback: a small generative model (Flan-T5-base) reads the
    retrieved chunks and writes a grounded answer. No API key needed, but
    noticeably less capable than a hosted LLM.
    """
    if not retrieved_chunks:
        return {"answer": _NOT_FOUND_PHRASE, "pages": []}

    context, pages = _build_context(retrieved_chunks, max_chunks, max_chars_per_chunk)

    prompt = (
        "You are answering a question using only the context below, which "
        "comes from a document. If the answer is not contained in the "
        f"context, respond exactly with: \"{_NOT_FOUND_PHRASE}\". "
        "Otherwise answer concisely in 1-3 sentences.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    try:
        output = generator(prompt, max_new_tokens=150, do_sample=False)
        answer_text = output[0]["generated_text"].strip()
    except Exception:
        answer_text = "Sorry, something went wrong generating an answer."

    not_found = _NOT_FOUND_PHRASE.lower() in answer_text.lower()
    return {"answer": answer_text, "pages": [] if not_found else pages}


def generate_answer_groq(question, retrieved_chunks, api_key, model=GROQ_DEFAULT_MODEL,
                          max_chunks=4, max_chars_per_chunk=800):
    """Cloud path: sends the retrieved chunks to Groq's free-tier hosted LLM
    (Llama 3.1/3.3) for a much stronger, more fluent grounded answer than the
    local model can produce. Requires a free Groq API key.
    """
    if Groq is None:
        return {"answer": "The `groq` package isn't installed. Run: pip install groq", "pages": []}

    if not retrieved_chunks:
        return {"answer": _NOT_FOUND_PHRASE, "pages": []}

    context, pages = _build_context(retrieved_chunks, max_chunks, max_chars_per_chunk)

    system_prompt = (
        "You answer questions using ONLY the provided context from a document. "
        f"If the answer isn't in the context, reply exactly: \"{_NOT_FOUND_PHRASE}\" "
        "Otherwise, answer clearly and concisely, and mention which page(s) "
        "your answer came from using the [Page N] markers already in the context."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=400,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        return {"answer": f"Groq API error: {e}", "pages": []}

    not_found = _NOT_FOUND_PHRASE.lower() in answer_text.lower()
    return {"answer": answer_text, "pages": [] if not_found else pages}


def generate_answer(question, retrieved_chunks, engine="local", local_generator=None,
                     groq_api_key=None, groq_model=GROQ_DEFAULT_MODEL):
    """Unified entry point. engine is 'groq' (cloud, needs api key) or
    'local' (free, no key, lower quality). Falls back to local if 'groq' is
    requested but no key is provided.
    """
    if engine == "groq" and groq_api_key:
        return generate_answer_groq(question, retrieved_chunks, groq_api_key, groq_model)
    return generate_answer_local(question, retrieved_chunks, local_generator)
